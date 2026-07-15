import cv2
import torch
from huggingface_hub import hf_hub_download

from Material_net.dpt import MaterialNet
import mitsuba as mi

import open3d as o3d
from myutils.mesh_recon import depth_file_to_mesh,rotate_mesh_around_x
import matplotlib.pyplot as plt
import numpy as np

def main() -> None:
    image_path = "/home/majortom/project/test2.png"

    # OpenCV 默认读取 BGR。
    image_bgr = cv2.imread(
        image_path,
        cv2.IMREAD_UNCHANGED,
    )

    if image_bgr is None:
        raise FileNotFoundError(image_path)

    if image_bgr.ndim == 2:
        image_bgr = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_GRAY2BGR,
        )

    if image_bgr.shape[2] == 4:
        image_bgr = image_bgr[:, :, :3]

    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    )

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
        scale=0.5,
    )

    print("Resolution information:")
    print(meta)

    import os 
    test_output = "./output_imgs/test_out_3"
    os.makedirs(test_output, exist_ok=True)
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
    
    mat['gt_image'] = torch.from_numpy(image_rgb).cuda()
    mat['albedo'] = torch.from_numpy(albedo).cuda().clamp(0,1)
    mat['normal'] = torch.from_numpy(normal).cuda()
    mat['roughness'] = torch.from_numpy(roughness).unsqueeze(-1).cuda().clamp(0.07,1)
    mat['metallic'] = torch.from_numpy(metallic).unsqueeze(-1).cuda().clamp(0,1)
    mat['depth'] = torch.from_numpy(depth).unsqueeze(-1).cuda()

    mi.util.write_bitmap(os.path.join(test_output,'albedoPred.exr'), albedo)
    mi.util.write_bitmap(os.path.join(test_output,'normalPred.exr'), normal)
    mi.util.write_bitmap(os.path.join(test_output,'roughnessPred.png'), roughness)
    mi.util.write_bitmap(os.path.join(test_output,'metallicPred.png'), metallic)
    mi.util.write_bitmap(os.path.join(test_output,'depthPred.exr'), depth)
    mi.util.write_bitmap(os.path.join(test_output,'gt_image.png'), image_rgb)


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
        # Build camera intrinsic matching the depth map size
        import math
        dh, dw = depth.shape[:2]
        fov_deg = 35.0
        focal = (dw / 2) / math.tan(math.radians(fov_deg) / 2)
        cam = o3d.camera.PinholeCameraIntrinsic(
            width=dw, height=dh,
            fx=focal, fy=focal,
            cx=(dw - 1) / 2.0, cy=(dh - 1) / 2.0,
        )
        mesh, b_points  = depth_file_to_mesh(depth,cameraMatrix=cam, minAngle=6, sun3d=False, depthScale=1.0)
        mesh = rotate_mesh_around_x(mesh, 180)
        o3d.io.write_triangle_mesh(mesh_path, mesh)


if __name__ == "__main__":
    main()