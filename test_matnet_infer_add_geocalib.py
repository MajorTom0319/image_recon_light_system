import cv2
import torch
from huggingface_hub import hf_hub_download

from Material_net.dpt import MaterialNet
import mitsuba as mi

import open3d as o3d
from myutils.mesh_recon import depth_file_to_mesh,rotate_mesh_around_x
from myutils.camera_utils import (
    estimate_camera_geocalib,
    make_mitsuba_compatible_K,
    scale_intrinsics,
    write_materialist_camera_json,
)
import matplotlib.pyplot as plt
import numpy as np

def main() -> None:
    image_path = "/home/majortom/project/datasets/interiorverse1/000_im.exr"

    # ---- 读取输入图像 ----
    if image_path.lower().endswith(".exr"):
        # EXR: 线性 HDR，mi.Bitmap 直接返回 RGB float
        image_rgb = np.array(mi.Bitmap(image_path), dtype=np.float32)
        if image_rgb.ndim == 2:
            image_rgb = np.repeat(image_rgb[:, :, None], 3, axis=2)
        image_rgb = image_rgb[..., :3]
        # 归一化到 [0, 1] 供 MatNet 推理（保留原始 HDR 用于 gt_image）
        image_rgb_original = image_rgb.copy()
        p99 = np.percentile(image_rgb[image_rgb > 0], 99) if (image_rgb > 0).any() else 1.0
        image_rgb = np.clip(image_rgb / max(p99, 1e-6), 0.0, 1.0)
        print(f"EXR loaded: shape={image_rgb.shape}, HDR max={image_rgb_original.max():.3f}, p99={p99:.3f}")
    else:
        # 普通图像 (PNG/JPG): OpenCV BGR -> RGB
        image_bgr = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image_bgr is None:
            raise FileNotFoundError(image_path)
        if image_bgr.ndim == 2:
            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
        if image_bgr.shape[2] == 4:
            image_bgr = image_bgr[:, :, :3]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        if np.issubdtype(image_rgb.dtype, np.integer):
            image_rgb = image_rgb.astype(np.float32) / 255.0
        image_rgb_original = image_rgb.copy()

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

    pred, work_image, meta = model.infer_image_scaled(
        image_rgb,
        scale=1,
    )

    print("Resolution information:")
    print(meta)

    import os 
    test_output = "/home/majortom/project/Materialist/output_imgs/interiorverse1_geocalib"
    os.makedirs(test_output, exist_ok=True)

    # ---- GeoCalib 估计相机内参 ----
    original_h, original_w = image_rgb_original.shape[:2]
    # GeoCalib 的 load_image 不支持 EXR，需转为 8-bit PNG 临时文件
    if image_path.lower().endswith(".exr"):
        img_for_geo = np.clip(image_rgb * 255, 0, 255).astype(np.uint8)
        geo_input_path = os.path.join(test_output, "_geo_input.png")
        cv2.imwrite(geo_input_path, cv2.cvtColor(img_for_geo, cv2.COLOR_RGB2BGR))
    else:
        geo_input_path = image_path

    geo_camera = estimate_camera_geocalib(geo_input_path, device="cuda")
    print(f"GeoCalib: hfov={geo_camera.hfov_deg:.2f}, vfov={geo_camera.vfov_deg:.2f}, "
          f"fx={geo_camera.K[0,0]:.1f}, fy={geo_camera.K[1,1]:.1f}")

    # 将原始分辨率的 K 缩放到 MatNet 工作分辨率
    work_h, work_w = pred['albedo'].shape[:2]
    K_work = scale_intrinsics(
        geo_camera.K,
        source_hw=(original_h, original_w),
        target_hw=(work_h, work_w),
        preprocess_meta=meta,
    )
    K_work = make_mitsuba_compatible_K(K_work)

    # 保存 camera_meta.json 供后续渲染/优化使用
    camera_meta_path = os.path.join(test_output, "camera_meta.json")
    write_materialist_camera_json(
        camera_meta_path,
        K_work=K_work,
        work_hw=(work_h, work_w),
        geocalib_result=geo_camera,
        preprocess_meta=meta,
    )
    print(f"Camera meta saved: {camera_meta_path} (work={work_w}x{work_h})")
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


    mesh_path = os.path.join(test_output, "mesh_depth.ply")

    mesh_mask_path = os.path.join(test_output,'mesh_mask.png')
    if os.path.exists(mesh_mask_path):
        mesh_mask = plt.imread(mesh_mask_path)
        mesh_mask = np.array(mesh_mask, dtype=np.bool_)
        if mesh_mask.ndim > 2:  # If it's an RGB image, use only the first channel
            mesh_mask = mesh_mask[..., 0]
    if not os.path.exists(mesh_path):
        depth = 2 * depth.max() - depth
        if os.path.exists(mesh_mask_path):
            depth[mesh_mask] = 0
            print(f"Applied mask from {mesh_mask_path} to depth map")
        # 使用 GeoCalib 估计的相机内参构建 mesh
        mesh, b_points = depth_file_to_mesh(depth, cameraMatrix=K_work, minAngle=6, sun3d=False, depthScale=1.0)
        mesh = rotate_mesh_around_x(mesh, 180)
        o3d.io.write_triangle_mesh(mesh_path, mesh)


if __name__ == "__main__":
    main()
