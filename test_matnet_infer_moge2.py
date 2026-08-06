import cv2
import torch
from huggingface_hub import hf_hub_download

from Material_net.dpt import MaterialNet
import mitsuba as mi

import open3d as o3d
from myutils.mesh_recon import depth_file_to_mesh, rotate_mesh_around_x
from myutils.camera_utils import (
    attach_infer_image_scaled,
    make_mitsuba_compatible_K,
    scale_intrinsics,
    write_materialist_camera_json,
    GeoCalibResult,
    _fovs_from_K,
)
from myutils.misc import linear_to_srgb
import matplotlib.pyplot as plt
import numpy as np

attach_infer_image_scaled(MaterialNet)

# Import MoGe2 helpers from inverse_img_w_mi_moge2
from inverse_img_w_mi_moge2 import (
    infer_moge2,
    estimate_camera_moge2,
)

def main() -> None:
    image_path = "/home/majortom/project/datasets/interiorverse5/005_im.exr"
    moge2_model_name = "/home/majortom/project/datasets/ckpt/moge2_vitl_normal.pt"
    test_output = "/home/majortom/project/Materialist/output_imgs/interiorverse5_testinfer_moge2"

    # ---- 读取输入图像 ----
    if image_path.lower().endswith(".exr"):
        # EXR: 线性 HDR，mi.Bitmap 直接返回 RGB float
        image_rgb = np.array(mi.Bitmap(image_path), dtype=np.float32)
        if image_rgb.ndim == 2:
            image_rgb = np.repeat(image_rgb[:, :, None], 3, axis=2)
        image_rgb = image_rgb[..., :3]
        image_rgb_original = image_rgb.copy()
        # 创建 uint8 sRGB 版本供 MoGe2 使用（MoGe2 期望 uint8 RGB）
        image_rgb_uint8 = np.clip(
            linear_to_srgb(np.clip(image_rgb, 0, None)) * 255.0, 0, 255
        ).astype(np.uint8)
        print(f"EXR loaded: shape={image_rgb.shape}, HDR max={image_rgb_original.max():.3f}")
    else:
        # 普通图像 (PNG/JPG): OpenCV BGR -> RGB
        image_bgr = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image_bgr is None:
            raise FileNotFoundError(image_path)
        if image_bgr.ndim == 2:
            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
        if image_bgr.shape[2] == 4:
            image_bgr = image_bgr[:, :, :3]
        image_rgb_uint8 = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb = image_rgb_uint8.astype(np.float32) / 255.0
        image_rgb_original = image_rgb.copy()

    # ---- MoGe2 估计相机内参 + 几何 ----
    print("Running MoGe2 inference for camera intrinsics and geometry...")
    geo_camera, moge2_output = estimate_camera_moge2(
        image_rgb_uint8,
        device="cuda",
        model_name=moge2_model_name,
    )
    print(f"MoGe2 estimated FOV: hfov={geo_camera.hfov_deg:.2f}°, vfov={geo_camera.vfov_deg:.2f}°")
    print(f"MoGe2 K:\n{geo_camera.K}")

    # ---- MatNet 材质推理 ----
    model_path = hf_hub_download(
        repo_id="Lez/MatNet",
        filename="matnet_weights.pth",
        repo_type="model",
    )

    model = MaterialNet(
        encoder="vitb",
        features=128,
        out_channels=[96, 192, 384, 768],
        use_bn=False,
        use_clstoken=False,
    )

    state_dict = torch.load(
        model_path,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state_dict)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = model.to(device).eval()

    # EXR: 传线性 float 给 MatNet（与 inverse_img_w_mi_ori 一致）
    # PNG/JPG: 传 uint8（MatNet 内部处理归一化）
    matnet_input = image_rgb if image_path.lower().endswith(".exr") else image_rgb_uint8
    pred, work_image, meta = model.infer_image_scaled(
        matnet_input,
        scale=1,
    )

    print("Resolution information:")
    print(meta)

    import os
    
    os.makedirs(test_output, exist_ok=True)

    # ---- 将 MoGe2 内参缩放到工作分辨率 ----
    work_h, work_w = work_image.shape[:2]
    original_h, original_w = image_rgb_original.shape[:2]
    K_work = scale_intrinsics(
        geo_camera.K,
        source_hw=(original_h, original_w),
        target_hw=(work_h, work_w),
        preprocess_meta=meta,
    )
    K_work = make_mitsuba_compatible_K(K_work)
    print(f"K_work (scaled to {work_w}x{work_h}):\n{K_work}")

    # ---- 保存 camera_meta.json ----
    camera_meta_path = os.path.join(test_output, "camera_meta.json")
    camera_meta = write_materialist_camera_json(
        camera_meta_path,
        K_work=K_work,
        work_hw=(work_h, work_w),
        geocalib_result=geo_camera,
        preprocess_meta=meta,
    )
    print(f"Camera meta saved: {camera_meta_path}")
    print(f"  hfov={camera_meta['x_fov'][0]:.2f}°, vfov={camera_meta['y_fov'][0]:.2f}°")
    # 保存普通显示图。
    albedo_bgr = cv2.cvtColor(
        (pred["albedo"].clip(0, 1) * 255).astype("uint8"),
        cv2.COLOR_RGB2BGR,
    )
    cv2.imwrite(os.path.join(test_output,"albedo_half.png"), albedo_bgr)

    roughness = (
        pred["roughness"].clip(0, 1) * 255
    ).astype("uint8")
    cv2.imwrite(os.path.join(test_output, "roughness_half.png"), roughness)

    metallic = (
        pred["metallic"].clip(0, 1) * 255
    ).astype("uint8")
    cv2.imwrite(os.path.join(test_output, "metallic_half.png"), metallic)

    # normal 从 [-1, 1] 映射到 [0, 255] 仅用于可视化。
    normal_vis = (
        (pred["normal"] * 0.5 + 0.5).clip(0, 1) * 255
    ).astype("uint8")
    normal_vis = cv2.cvtColor(
        normal_vis,
        cv2.COLOR_RGB2BGR,
    )
    cv2.imwrite(os.path.join(test_output, "normal_half.png"), normal_vis)

    # 相对深度归一化，仅用于查看。
    depth = pred["depth"]
    depth_vis = (
        (depth - depth.min())
        / (depth.max() - depth.min() + 1e-8)
        * 255
    ).astype("uint8")
    cv2.imwrite(os.path.join(test_output, "depth_half.png"), depth_vis)

    albedo = pred['albedo']
    normal = pred['normal']
    roughness = pred['roughness'] 
    metallic = pred['metallic'] 
    depth = pred['depth']

    mat = {}
    
    mat['gt_image'] = torch.from_numpy(image_rgb_original).cuda()
    mat['albedo'] = torch.from_numpy(albedo).cuda().clamp(0,1)
    mat['normal'] = torch.from_numpy(normal).cuda()
    mat['roughness'] = torch.from_numpy(roughness).unsqueeze(-1).cuda().clamp(0.07,1)
    mat['metallic'] = torch.from_numpy(metallic).unsqueeze(-1).cuda().clamp(0,1)
    mat['depth'] = torch.from_numpy(depth).unsqueeze(-1).cuda()

    mi.util.write_bitmap(os.path.join(test_output,'albedoPred.exr'), albedo)
    mi.util.write_bitmap(os.path.join(test_output,'normalPred.exr'), normal)
    mi.util.write_bitmap(os.path.join(test_output,'roughnessPred.exr'), roughness)
    mi.util.write_bitmap(os.path.join(test_output,'metallicPred.exr'), metallic)
    mi.util.write_bitmap(os.path.join(test_output,'depthPred.exr'), depth)
    mi.util.write_bitmap(os.path.join(test_output,'gt_image.exr'), image_rgb_original)
    # mi.util.write_bitmap 写 PNG 时自动做 linear -> sRGB gamma 校正
    mi.util.write_bitmap(os.path.join(test_output,'gt_image.png'), image_rgb_original)


    # ---- 使用 MoGe2 depth + depth_file_to_mesh 重建 mesh（与 inverse_img_w_mi_moge2 一致）----
    mesh_path = os.path.join(test_output, "mesh_moge2.ply")
    moge2_depth = moge2_output['depth']  # (H, W) metric depth
    if moge2_depth.shape[:2] != (work_h, work_w):
        moge2_depth = cv2.resize(moge2_depth, (work_w, work_h), interpolation=cv2.INTER_LINEAR)

    if not os.path.exists(mesh_path):
        # 可选: 加载用户 mesh_mask
        mesh_mask_path = os.path.join(test_output, 'mesh_mask.png')
        mesh_mask = None
        if os.path.exists(mesh_mask_path):
            mesh_mask = plt.imread(mesh_mask_path)
            mesh_mask = np.array(mesh_mask, dtype=np.bool_)
            if mesh_mask.ndim > 2:
                mesh_mask = mesh_mask[..., 0]

        # MoGe2 输出标准度量深度（越大越远），depth_file_to_mesh 直接可用
        depth_for_mesh = moge2_depth.copy()
        if mesh_mask is not None:
            depth_for_mesh[mesh_mask] = 0
            print(f"Applied mask from {mesh_mask_path} to depth map")
        mesh, b_points = depth_file_to_mesh(
            depth_for_mesh, cameraMatrix=K_work, minAngle=6, sun3d=False, depthScale=1.0
        )
        mesh = rotate_mesh_around_x(mesh, 180)
        o3d.io.write_triangle_mesh(mesh_path, mesh)
        print(f"Mesh saved: {mesh_path}")
    else:
        print(f"Using existing mesh: {mesh_path}")

    # 同时保存 MoGe2 深度和法线供参考
    mi.util.write_bitmap(os.path.join(test_output, 'moge2_depth.exr'), moge2_output['depth'])
    if moge2_output['normal'] is not None:
        mi.util.write_bitmap(os.path.join(test_output, 'moge2_normal.exr'), moge2_output['normal'])
    mi.util.write_bitmap(os.path.join(test_output, 'moge2_points.exr'), moge2_output['points'])


if __name__ == "__main__":
    main()