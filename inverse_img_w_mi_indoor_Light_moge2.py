import mitsuba as mi
import drjit as dr
from myutils.mi_plugin import MatDiffBSDF
from hybrid_light.io import load_ile_lights
from hybrid_light.mitsuba_builder import build_hybrid_scene_dict
from huggingface_hub import hf_hub_download
from Material_net.dpt import MaterialNet
from myutils.camera_utils import (
    make_mitsuba_compatible_K,
    scale_intrinsics,
    write_materialist_camera_json,
)
from myutils.moge2_utils import (
    camera_to_materialist_vectors,
    estimate_camera_moge2,
    prepare_moge2_depth,
    prepare_moge2_normal,
)
from myutils.mesh_recon import depth_file_to_mesh, rotate_mesh_around_x
mi.register_bsdf('MatDiffBSDF', lambda props: MatDiffBSDF(props))
import global_config
from torchvision.utils import save_image,make_grid
from tqdm import tqdm
import gc
from myutils.misc import *
from mymodels.mlps import PosMLP
from myutils.envmap_utils import lookup_envmap, importance_sample, build_envmap,sample_brdf1,sample_env1
import open3d as o3d
import argparse
import torch.nn.functional as NF
import json
import numpy as np
from torch.optim.lr_scheduler import OneCycleLR,StepLR
import matplotlib.pyplot as plt
import warnings
import sys
import time
import imageio
import os
import cv2


def _srgb_to_linear_numpy(image: np.ndarray) -> np.ndarray:
    """Convert normalized sRGB code values to linear RGB."""
    image = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    return np.where(
        image <= 0.04045,
        image / 12.92,
        np.power((image + 0.055) / 1.055, 2.4),
    ).astype(np.float32)


def _linear_to_srgb_numpy(image: np.ndarray) -> np.ndarray:
    """Convert non-negative linear RGB to normalized sRGB code values."""
    image = np.maximum(np.asarray(image, dtype=np.float32), 0.0)
    return np.where(
        image <= 0.0031308,
        image * 12.92,
        1.055 * np.power(image, 1.0 / 2.4) - 0.055,
    ).astype(np.float32)


def _linear_to_display_srgb(image: torch.Tensor) -> torch.Tensor:
    """Map linear radiance to the clipped sRGB domain used by ILE images."""
    image = image.clamp(0.0, 1.0)
    return torch.where(
        image <= 0.0031308,
        image * 12.92,
        1.055 * torch.pow(image.clamp_min(0.0031308), 1.0 / 2.4) - 0.055,
    )


def _depth_edge_mask(depth: np.ndarray, mask: np.ndarray, rtol: float = 0.04) -> np.ndarray:
    """Detect depth discontinuity edges (similar to MoGe2's depth_map_edge).
    
    Returns a boolean mask where True = edge pixel to be removed.
    """
    h, w = depth.shape
    edge_mask = np.zeros((h, w), dtype=bool)
    
    # Compute disparity
    disp = np.where(mask & (depth > 0), 1.0 / np.clip(depth, 1e-8, None), 0.0)
    
    # Check 4-neighbors for depth discontinuity
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ny = np.clip(np.arange(h) + dy, 0, h - 1)
        nx = np.clip(np.arange(w) + dx, 0, w - 1)
        neighbor_disp = disp[ny[:, None], nx[None, :]]
        # Relative depth difference
        diff = np.abs(disp - neighbor_disp)
        max_disp = np.maximum(np.abs(disp), np.abs(neighbor_disp))
        edge = mask & (diff > rtol * max_disp) & (max_disp > 0)
        edge_mask |= edge
    
    # Dilate edge mask by 1 pixel
    kernel = np.ones((3, 3), dtype=np.uint8)
    edge_mask = cv2.dilate(edge_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    return edge_mask


def build_mesh_from_moge2_points(
    points: np.ndarray,
    mask: np.ndarray,
    depth: np.ndarray,
    mesh_mask: np.ndarray = None,
    edge_threshold: float = 0.04,
) -> o3d.geometry.TriangleMesh:
    """Build a triangle mesh directly from MoGe2's point map.
    
    This replicates MoGe2's `infer --ply` mesh building logic:
    1. Remove edge pixels at depth discontinuities
    2. Triangulate the regular grid
    3. Convert from OpenCV coords (Y-down, Z-forward) to Mitsuba coords (Y-up, Z-backward)
    
    Args:
        points: (H, W, 3) point map in OpenCV camera coordinate system
        mask: (H, W) boolean valid pixel mask from MoGe2
        depth: (H, W) depth map for edge detection
        mesh_mask: optional (H, W) boolean mask from user (True = exclude)
        edge_threshold: relative threshold for depth edge removal
        
    Returns:
        open3d TriangleMesh in Mitsuba coordinate system
    """
    h, w = points.shape[:2]
    
    # Clean mask: remove edge pixels at depth discontinuities
    edge_mask = _depth_edge_mask(depth, mask, rtol=edge_threshold)
    mask_cleaned = mask & ~edge_mask
    
    # Apply user-provided mesh_mask (True = pixels to exclude)
    if mesh_mask is not None:
        if mesh_mask.shape[:2] != (h, w):
            mesh_mask = cv2.resize(mesh_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        mask_cleaned = mask_cleaned & ~mesh_mask
        print(f"Applied user mesh_mask, excluded {mesh_mask.sum()} pixels")
    
    print(f"Mesh building: {mask_cleaned.sum()} valid pixels out of {h*w} total "
          f"(removed {edge_mask.sum()} edge pixels)")
    
    # Build vertex index map: assign sequential indices to valid pixels
    valid_indices = np.full((h, w), -1, dtype=np.int32)
    valid_indices[mask_cleaned] = np.arange(mask_cleaned.sum(), dtype=np.int32)
    
    # Get vertex positions from point map
    vertices = points[mask_cleaned]  # (N, 3) in OpenCV coords
    
    # Build triangles from grid quads (vectorized)
    # For each (i, j) cell, check the 4 corners
    tl = valid_indices[:-1, :-1]  # (H-1, W-1)
    tr = valid_indices[:-1, 1:]   # (H-1, W-1)
    bl = valid_indices[1:, :-1]   # (H-1, W-1)
    br = valid_indices[1:, 1:]    # (H-1, W-1)
    
    # Triangle 1: top-left, bottom-left, top-right (where all 3 are valid)
    valid_tri1 = (tl >= 0) & (bl >= 0) & (tr >= 0)
    tri1 = np.stack([tl[valid_tri1], bl[valid_tri1], tr[valid_tri1]], axis=-1)
    
    # Triangle 2: top-right, bottom-left, bottom-right (where all 3 are valid)
    valid_tri2 = (tr >= 0) & (bl >= 0) & (br >= 0)
    tri2 = np.stack([tr[valid_tri2], bl[valid_tri2], br[valid_tri2]], axis=-1)
    
    # Combine all triangles
    tri_list = [t for t in [tri1, tri2] if len(t) > 0]
    if tri_list:
        triangles = np.concatenate(tri_list, axis=0)
    else:
        triangles = np.zeros((0, 3), dtype=np.int32)
    
    print(f"Mesh building: {len(vertices)} vertices, {len(triangles)} triangles")
    
    # Convert from OpenCV coords (X-right, Y-down, Z-forward)
    # to Mitsuba/OpenGL coords (X-right, Y-up, Z-backward)
    vertices_mitsuba = vertices * np.array([1.0, -1.0, -1.0], dtype=np.float32)
    
    # Create Open3D mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices_mitsuba.astype(np.float64))
    if len(triangles) > 0:
        mesh.triangles = o3d.utility.Vector3iVector(triangles.astype(np.int32))
    mesh.compute_vertex_normals()
    
    return mesh


def build_mesh_from_depth_k(
    depth: np.ndarray,
    K: np.ndarray,
    mask: np.ndarray = None,
    mesh_mask: np.ndarray = None,
) -> o3d.geometry.TriangleMesh:
    """Build a triangle mesh via depth map + K back-projection (vectorized).

    Uses STANDARD metric depth (larger = farther) directly -- no inversion or
    normalization needed.

    Pipeline:
      1. Back-project depth+K to 3D points (OpenCV camera coords, Z-forward)
      2. Triangulate the regular pixel grid
      3. Convert OpenCV coords (X-right, Y-down, Z-forward)
         to Mitsuba coords (X-right, Y-up, Z-backward)
    """
    h, w = depth.shape
    depth = depth.astype(np.float64)

    # Valid pixel mask
    valid = np.isfinite(depth) & (depth > 0)
    if mask is not None:
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        valid &= mask

    # Apply user-provided mesh_mask (True = pixels to exclude)
    if mesh_mask is not None:
        if mesh_mask.shape[:2] != (h, w):
            mesh_mask = cv2.resize(mesh_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        valid = valid & ~mesh_mask

    # Back-project: point = depth * (K_inv @ [u, v, 1])
    K_inv = np.linalg.inv(K.astype(np.float64))
    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    ray_x = K_inv[0, 0] * u + K_inv[0, 1] * v + K_inv[0, 2]
    ray_y = K_inv[1, 0] * u + K_inv[1, 1] * v + K_inv[1, 2]
    ray_z = K_inv[2, 0] * u + K_inv[2, 1] * v + K_inv[2, 2]
    points = np.stack([ray_x * depth, ray_y * depth, ray_z * depth], axis=-1).astype(np.float32)

    print(f"Mesh building: {valid.sum()} valid pixels out of {h * w} total")

    # Build vertex index map
    valid_indices = np.full((h, w), -1, dtype=np.int32)
    valid_indices[valid] = np.arange(valid.sum(), dtype=np.int32)
    vertices = points[valid]  # (N, 3) OpenCV coords

    # Triangulate the grid (2 triangles per 2x2 cell, vectorized)
    tl = valid_indices[:-1, :-1]
    tr = valid_indices[:-1, 1:]
    bl = valid_indices[1:, :-1]
    br = valid_indices[1:, 1:]
    tri1_valid = (tl >= 0) & (bl >= 0) & (tr >= 0)
    tri1 = np.stack([tl[tri1_valid], bl[tri1_valid], tr[tri1_valid]], axis=-1)
    tri2_valid = (tr >= 0) & (bl >= 0) & (br >= 0)
    tri2 = np.stack([tr[tri2_valid], bl[tri2_valid], br[tri2_valid]], axis=-1)
    tri_list = [t for t in [tri1, tri2] if len(t) > 0]
    triangles = np.concatenate(tri_list, axis=0) if tri_list else np.zeros((0, 3), dtype=np.int32)

    print(f"Mesh building: {len(vertices)} vertices, {len(triangles)} triangles")

    # OpenCV (X-right, Y-down, Z-forward) -> Mitsuba (X-right, Y-up, Z-backward)
    vertices_mitsuba = vertices * np.array([1.0, -1.0, -1.0], dtype=np.float32)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices_mitsuba.astype(np.float64))
    if len(triangles) > 0:
        mesh.triangles = o3d.utility.Vector3iVector(triangles.astype(np.int32))
    mesh.compute_vertex_normals()
    return mesh


def load_estimated_mesh(
    mesh_path,
    use_mesh_normal,
    cam_meta_path,
    ile_lights,
    radiance_scale=1.0,
    visible_offset=0.005,
    max_path=4,
):
    """Load Materialist geometry with fixed ILE lights and an optimizable HDRI."""
    with open(cam_meta_path, "r", encoding="utf-8") as file:
        camera_meta = json.load(file)
    scene_dict = build_hybrid_scene_dict(
        mi,
        mesh_path=mesh_path,
        camera_meta_path=cam_meta_path,
        camera_meta=camera_meta,
        lights=ile_lights,
        mode="combined",
        envmap_path="envmaps/0.hdr",
        radiance_scale=radiance_scale,
        visible_offset=visible_offset,
        use_mesh_normal=use_mesh_normal,
        max_depth=max_path,
    )

    # Preserve the parameter names used by the existing HDRI/material
    # optimization code: emitter.data and shape.bsdf.{a,r,m,n}.
    scene_dict["shape"] = scene_dict.pop("materialist_mesh")
    scene_dict["emitter"] = scene_dict.pop("far_field_env")

    for light in ile_lights:
        if not light.is_window:
            continue
        window_shape = scene_dict[f"ile_{light.name}"]
        # Omitting a BSDF does not make the aperture transparent: Mitsuba then
        # treats it as an opaque zero-reflectance surface. Keep an explicit
        # null BSDF so rays can continue through the window instead of seeing
        # a black cut-out around/backside of the directional emitter.
        window_shape["bsdf"] = {"type": "null"}
        # visible_offset is a lamp coplanarity workaround; moving a window
        # changes its physical aperture and wall alignment.
        window_shape["to_world"] = mi.ScalarTransform4f.scale(
            [light.geometry_scale] * 3
        )

    return mi.load_dict(scene_dict)


def render_mesh_preview(mesh_path, cam_meta_path, output_dir, spp=64):
    """Render the reconstructed mesh with a plain white diffuse material."""
    with open(cam_meta_path, "r", encoding="utf-8") as f:
        camera_meta = json.load(f)
    width, height = [int(v) for v in camera_meta["film.size"]]
    vfov_deg = float(camera_meta["y_fov"][0])

    scene = mi.load_dict({
        "type": "scene",
        "shape": {
            "type": "ply",
            "filename": mesh_path,
            "bsdf": {"type": "diffuse", "reflectance": 0.8},
        },
        "integrator": {"type": "path", "max_depth": 4},
        "sensor": {
            "type": "perspective",
            "fov": vfov_deg,
            "fov_axis": "y",
            "to_world": mi.ScalarTransform4f.look_at(
                origin=[0, 0, 0], target=[0, 0, -1], up=[0, 1, 0]),
            "film": {"type": "hdrfilm", "width": width, "height": height,
                     "pixel_format": "rgb"},
        },
        "emitter": {"type": "envmap", "filename": "envmaps/0.hdr"},
    })
    rendered = mi.render(scene, spp=spp, seed=42)
    rendered_np = np.array(rendered)
    mi.util.write_bitmap(os.path.join(output_dir, 'mesh_render.exr'), rendered_np)
    rendered_srgb = _linear_to_display_srgb(torch.from_numpy(rendered_np))
    save_image(rendered_srgb.permute(2, 0, 1).unsqueeze(0),
               os.path.join(output_dir, 'mesh_render.png'))
    print(f"Mesh preview saved to {os.path.join(output_dir, 'mesh_render.png')}")
    del scene, rendered
    torch.cuda.empty_cache()


@dr.wrap_ad(source='torch', target='drjit')
def render_envmap(scene,envmap,spp=64,seed=0):
    params = mi.traverse(scene)
    params['emitter.data']=envmap 
    params.update()
    rendered_img=mi.render(scene, params, spp=spp, seed=seed)

    return rendered_img

@dr.wrap_ad(source='torch', target='drjit')
def render_w_brdf(scene,albedo,roughness, metallic,normal=None,spp=64,seed=0):
    params = mi.traverse(scene)
    params['shape.bsdf.a'] = albedo
    params['shape.bsdf.r'] = roughness
    params['shape.bsdf.m'] = metallic
    if normal is not None:
        params['shape.bsdf.n'] = normal
    params.update()
    rendered_img=mi.render(scene, params, spp=spp,seed=seed)
    return rendered_img

def get_output_dir(save_name, save_path=None):
    """Determine the output directory based on save_name and save_path.
    
    Args:
        save_name: Name of the save directory
        save_path: Optional path where results should be saved
        
    Returns:
        Full path to the output directory
    """
    if save_path:
        # If save_path is provided, use it directly
        if os.path.isabs(save_path):
            return os.path.join(save_path, save_name)
        else:
            # If save_path is relative, treat it relative to OUT_DIR
            return os.path.join(global_config.OUT_DIR, save_path, save_name)
    
    # If no save_path, use the previous logic for backwards compatibility
    if os.path.isabs(save_name):
        return save_name
    else:
        return os.path.join(global_config.OUT_DIR, save_name)

def optimize_envmap_ARMN(scene,cam_cfg,mat,save_folder,use_mesh_normal,
                    output_type,optimize_order,spp=64,use_gt_scene = False,
                    model_name='pos_mlp',opt_env_from=0,
                    opt_src='arm',use_mask=False,scale_delta=0.1, save_path=None,
                    env_coordinate_type='spherical',final_spp=256):
    '''
    mat: dict, albedo:H,W,C, roughness:H,W,1, metallic:H,W,1, normal:H,W,3, depth:H,W,1, gt_image:H,W,3
    '''
    device = torch.device('cuda')
    depth = 4
    width = 256
    weight_norm = False
    env_h, env_w = 16, 32
    env_input_dims = 6 if env_coordinate_type == 'spherical' else 5
    envmap_net = PosMLP(in_dims=env_input_dims,
                            out_dims=3,
                            dims=[width] * depth,
                            skip_connection=[1,3],
                            weight_norm=weight_norm,
                            multires_view = 2,
                            output_type='envmap',
                            color_ch=3,
                            img_h=env_h,
                            img_w=env_w,
                            coordinate_type=env_coordinate_type)
    envmap_net = envmap_net.cuda()
    # opt_env = torch.optim.Adam(envmap_net.parameters(), lr=1e-4)
    
    # Get output directory
    output_dir = get_output_dir(save_folder, save_path)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        os.makedirs(os.path.join(output_dir, 'best_results'))
    
    # Create directories for intermediate results
    env_frames_dir = os.path.join(output_dir, 'env_frames')
    mat_frames_dir = os.path.join(output_dir, 'mat_frames')
    os.makedirs(env_frames_dir, exist_ok=True)
    os.makedirs(mat_frames_dir, exist_ok=True)
    
    # Lists to collect frames for videos
    env_frames = []
    mat_frames = []

    opt_normal = False
    if opt_normal:
        normal_net = PosMLP(in_dims=5,
                            out_dims=3,
                            dims=[width] * depth,
                            skip_connection=[1,3],
                            weight_norm=weight_norm,
                            output_type='normal')
        normal_net = normal_net.cuda()
        opt_normal_net = torch.optim.Adam(normal_net.parameters(), lr=1e-12)

    gt_image = mat['gt_image']
    img_h, img_w = int(gt_image.shape[0]), int(gt_image.shape[1])

    # brdf net

    if model_name == 'unet':
        raise ValueError('Do not use unet for this task')
    elif model_name == 'pos_mlp':
        if output_type == 'arm':
            multires_view = 2 # for pos embedding
            color_ch = 5
            brdf_net = PosMLP(in_dims=7,out_dims=color_ch,dims=[width] * depth,skip_connection=[1,3],weight_norm=weight_norm,multires_view=multires_view,output_type=output_type,color_ch = color_ch,img_h=img_h,img_w=img_w)
            brdf_net = brdf_net.cuda()
            # opt_brdf = torch.optim.Adam(brdf_net.parameters(), lr=1e-4)
            
        elif output_type == 'armn':
            multires_view = 0
            color_ch = 8
            brdf_net = PosMLP(in_dims=10,out_dims=color_ch,dims=[width] * depth,skip_connection=[1,3],weight_norm=weight_norm,multires_view=multires_view,output_type=output_type,color_ch = color_ch,img_h=img_h,img_w=img_w)
            brdf_net = brdf_net.cuda()
            # opt_brdf = torch.optim.Adam(brdf_net.parameters(), lr=1e-4)
    if use_gt_scene:
        gt_envmap = mat['gt_envmap']

    start_envmap = torch.ones(env_h, env_w, 3, device=device)
    start_envmap = start_envmap.reshape(-1, 3)
    
    roughness_shift = 0.7
    metallic_shift = 0.05
    optimize_parts = ''.join(optimize_order)
    if 'r' not in opt_src and 'r' not in optimize_parts:
        mat['roughness'] = mat['roughness'] * 0 + roughness_shift
    if 'm' not in opt_src and 'm' not in optimize_parts:
        mat['metallic'] = mat['metallic'] * 0 + metallic_shift

    albedo_ori = mat['albedo']
    roughness_ori = mat['roughness']
    metallic_ori = mat['metallic']
    normal_ori = mat['normal']
    normal_ori = NF.normalize(normal_ori, p=2, dim=-1)

    start_normal = normal_ori.reshape(-1,3)
    
    if 'r' not in opt_src and 'r' not in optimize_parts and opt_src != 'skip':
        roughness_ori = roughness_ori * 0 + roughness_shift
    if 'm' not in opt_src and 'm' not in optimize_parts and opt_src != 'skip':
        metallic_ori = metallic_ori * 0 + metallic_shift
    if output_type == 'armn':
        start_arm = torch.cat([albedo_ori.reshape(-1,3), roughness_ori.reshape(-1,1), metallic_ori.reshape(-1,1),normal_ori.reshape(-1,3)], dim=-1)
    elif output_type == 'arm':
        start_arm = torch.cat([albedo_ori.reshape(-1,3), roughness_ori.reshape(-1,1), metallic_ori.reshape(-1,1)], dim=-1).clamp(0,1)
    else:
        raise ValueError('output_type should be arm or armn')
    # start_arm = gt_image.reshape(1,-1,3)
    start_arm_unet = torch.cat([albedo_ori.permute(2,0,1), roughness_ori.permute(2,0,1), metallic_ori.permute(2,0,1)], dim=0).unsqueeze(0)

    num_epochs = 5000
    epoch = 0
    loop_num = 0
    optimization_history = []

    mi_params = mi.traverse(scene)
    mi_params['shape.bsdf.a'] = mat['albedo']
    mi_params['shape.bsdf.r'] = mat['roughness']
    mi_params['shape.bsdf.m'] = mat['metallic']
    # The HDRI stage renders before render_w_brdf() is called. Initialize the
    # predicted normal here as well, otherwise use_mesh_normal=False would make
    # that first stage shade with MatDiffBSDF's default normal tensor.
    mi_params['shape.bsdf.n'] = mat['normal']
    mi_params['shape.bsdf.use_mesh_normal'] = bool(use_mesh_normal)
    mi_params.update()
    
    early_stopping_all = EarlyStopping(patience=2, min_delta=0.025)
    while loop_num <= 10:
        loop_num +=1
        if loop_num == 1:
            opt_env = torch.optim.Adam(envmap_net.parameters(), lr=1e-3)
            scheduler_env = StepLR(opt_env, step_size=100, gamma=0.8)
        else:
            opt_env = torch.optim.Adam(envmap_net.parameters(), lr=1e-4)
        # optimize envmap
        if opt_src == 'skip' and opt_src == 'skip':
            patience_env = 500
        else:
            patience_env = 100
        early_stopping = EarlyStopping(patience=patience_env, min_delta=0.01)
        env_saver = SaveBest()
        with tqdm(total=num_epochs,desc='Opt envmap', unit='epochs',file=sys.stdout) as pbar:
            for epoch in range(num_epochs):
                envmap_pred = envmap_net(start_envmap)
                envmap_pred = envmap_pred.squeeze().reshape(env_h, env_w, 3)
                render_seed = loop_num * 100000 + epoch
                pred_image = render_envmap(
                    scene,
                    envmap_pred,
                    spp,
                    seed=render_seed,
                )
                pred_image_srgb = _linear_to_display_srgb(pred_image)
                gt_image_srgb = _linear_to_display_srgb(gt_image)
                loss_mse = NF.mse_loss(pred_image_srgb, gt_image_srgb)
                loss_l1 = NF.l1_loss(pred_image_srgb, gt_image_srgb)
                loss = loss_mse + loss_l1
                
                env_saver.update(loss_mse.item(), mat['albedo'], mat['roughness'], mat['metallic'],mat['normal'],envmap_pred,pred_image)
                optimization_history.append({
                    "loop": loop_num,
                    "phase": "env",
                    "epoch": epoch,
                    "loss_mse": loss_mse.item(),
                    "loss_l1": loss_l1.item(),
                })
                loss.backward()

                early_stopping(loss_mse.item())
                opt_env.step()
                opt_env.zero_grad()
                if loop_num == 1:
                    scheduler_env.step()
                pbar.set_postfix(loss=loss.item(),loss_mse=loss_mse.item())
                pbar.update(1)
                if epoch % 10 ==0 or early_stopping.early_stop:
                    all_env = torch.concat([envmap_pred], dim=1)
                    env_display = torch.zeros_like(gt_image_srgb)  # Blank canvas same size as gt_image
                    h, w = envmap_pred.shape[:2]
                    display_h = min(h * 3, env_display.shape[0] // 2)  
                    display_w = int(display_h * (w / h))
                    
                    # Position the envmap in the center of the blank image
                    start_h = (env_display.shape[0] - display_h) // 2
                    start_w = (env_display.shape[1] - display_w) // 2
                    
                    # Resize and place the envmap
                    envmap_display = NF.interpolate(
                        envmap_pred.permute(2, 0, 1).unsqueeze(0),
                        size=(display_h, display_w),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0).permute(1, 2, 0)
                    
                    env_display[start_h:start_h+display_h, start_w:start_w+display_w] = envmap_display
                    all_image = torch.concat([gt_image_srgb, pred_image_srgb,env_display], dim=1)
                    
                    save_image(all_env.permute(2, 0, 1).unsqueeze(0), os.path.join(output_dir, f'env.png'))
                    
                    # Also save to output directory for easy viewing of latest frame
                    env_frame_path = os.path.join(env_frames_dir, f'opt_env_frame_{loop_num}_{epoch:04d}.png')
                    save_image(all_image.permute(2, 0, 1).unsqueeze(0),env_frame_path)
                    env_frames.append(env_frame_path)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
                if loop_num < opt_env_from:
                    print(f'loop_num {loop_num} < opt_env_from {opt_env_from}, break')
                    break
                if 'rm' not in opt_src and loop_num == 1:
                    if opt_src != 'skip':
                        print(f'rm not in opt_src and loop_num == 1, break')
                        break

        final_envmap = env_saver.best_envmap
        mi_params['emitter.data'] = final_envmap
        mi_params.update()
        mi.util.write_bitmap(os.path.join(output_dir, 'final_envmap.hdr'), final_envmap)
        save_image(all_image.permute(2, 0, 1).unsqueeze(0),os.path.join(output_dir, f'opt_env_img.png'))
        torch.cuda.empty_cache()
        gc.collect()

        if loop_num >= opt_env_from :
            env_saver.save_results(os.path.join(output_dir, f'best_results'))
        with open(os.path.join(output_dir, "optimization_history.json"), "w", encoding="utf-8") as file:
            json.dump(optimization_history, file, indent=2)
        early_stopping_all(env_saver.best_loss)
        if early_stopping_all.early_stop:
            print("Early stopping")
            loop_num = 11
            break
        if loop_num >=3:
            break
        if opt_src == 'skip' or opt_src == 'skip':
            break
        
        ############################
        # optimize brdf
        ###########################
        if loop_num < opt_env_from and loop_num==1:
            if 'gt_envmap' in mat.keys():
                envmap4render = mat['gt_envmap']
                print('use gt envmap for brdf optimization')
            else:
                envmap4render = torch.ones(16,32,3).cuda()
                print('Use envmap = 1 for brdf optimization')
        else:
            envmap4render = final_envmap
            print('Use Optimized envmap for brdf optimization')
        
        if loop_num <=1:
            if 'r' not in opt_src and 'r' not in optimize_parts:
                mat['roughness'] = mat['roughness'] * 0 + roughness_shift
            if 'm' not in opt_src and 'm' not in optimize_parts:
                mat['metallic'] = mat['metallic'] * 0 + metallic_shift

        mi_params['emitter.data'] = envmap4render
        if use_mesh_normal:
            mi_params['shape.bsdf.use_mesh_normal'] = True
            print('Use Mesh Normal')
        else:
            mi_params['shape.bsdf.use_mesh_normal'] = False
            print('Use Predicted Normal')
        # mi_params['integrator.max_depth'] = 2
        mi_params.update()
        for optimize_part in optimize_order:
            if optimize_part == 'a' and loop_num <= 1: #skip the first loop for albedo
                continue
            material_saver = SaveBest()

            if model_name == 'none':
                print(f"Directly optimizing {optimize_part} without neural network")
                optimizable_params = {}
                if 'a' in optimize_part:
                    optimizable_params['albedo'] = torch.nn.Parameter(mat['albedo'].clone())
                if 'r' in optimize_part:
                    optimizable_params['roughness'] = torch.nn.Parameter(mat['roughness'].clone())
                if 'm' in optimize_part:
                    optimizable_params['metallic'] = torch.nn.Parameter(mat['metallic'].clone())
                if 'n' in optimize_part and not use_mesh_normal:
                    optimizable_params['normal'] = torch.nn.Parameter(mat['normal'].clone())

                opt_params = torch.optim.Adam(optimizable_params.values(), lr=3e-4)
                scheduler_params = StepLR(opt_params, step_size=100, gamma=0.8)
                
                if 'a' in optimize_part:
                    early_stopping = EarlyStopping(patience=200//loop_num, min_delta=0.005)
                else:
                    early_stopping = EarlyStopping(patience=200//loop_num, min_delta=0.001)
                    
                with tqdm(total=num_epochs, desc=f'Opt {optimize_part} directly', unit='e', file=sys.stdout) as pbar:
                    for epoch in range(num_epochs):

                        if 'a' in optimize_part:
                            mat['albedo'] = optimizable_params['albedo'].clamp(0, 1)
                        if 'r' in optimize_part:
                            mat['roughness'] = optimizable_params['roughness'].clamp(0.07, 1)
                        if 'm' in optimize_part:
                            mat['metallic'] = optimizable_params['metallic'].clamp(0, 1)
                        if 'n' in optimize_part and not use_mesh_normal:
                            mat['normal'] = NF.normalize(optimizable_params['normal'], p=2, dim=-1)
                            
                        if use_mask:
                            mat['roughness'][mat['mask']] = mat['roughness'][mat['mask']].mean()
                            mat['metallic'][mat['mask']] = mat['metallic'][mat['mask']].mean()
                            
                        if use_mesh_normal:
                            pred_image = render_w_brdf(
                                scene,
                                mat['albedo'],
                                mat['roughness'],
                                mat['metallic'],
                                None,
                                spp,
                                seed=loop_num * 100000 + epoch,
                            )
                        else:
                            pred_image = render_w_brdf(
                                scene,
                                mat['albedo'],
                                mat['roughness'],
                                mat['metallic'],
                                mat['normal'],
                                spp,
                                seed=loop_num * 100000 + epoch,
                            )

                        pred_image_srgb = _linear_to_display_srgb(pred_image)
                        gt_image_srgb = _linear_to_display_srgb(gt_image)

                        loss_mse = NF.mse_loss(pred_image_srgb, gt_image_srgb)
                        loss_l1 = NF.l1_loss(pred_image_srgb, gt_image_srgb)
                        

                        if 'a' in optimize_part:
                            loss_a = NF.l1_loss(mat['albedo'], albedo_ori)
                        else:
                            loss_a = 0
                        if 'r' in optimize_part:
                            loss_r = NF.l1_loss(mat['roughness'], roughness_ori)
                        else:
                            loss_r = 0
                        if 'm' in optimize_part:
                            loss_m = NF.l1_loss(mat['metallic'], metallic_ori)
                        else:
                            loss_m = 0
                        if 'n' in optimize_part and not use_mesh_normal:
                            loss_n = NF.l1_loss(mat['normal'], normal_ori)
                        else:
                            loss_n = 0
                            
                        scale_raito = loss_l1.detach()/loss_mse.detach()
                        aux_loss = loss_a + (loss_r + loss_m + loss_n)
                        render_loss = 3 * scale_raito * loss_mse + loss_l1
                        loss = render_loss + aux_loss * scale_delta

                        loss.backward()
                        
                        material_saver.update(
                            loss_mse.item(),
                            mat['albedo'],
                            mat['roughness'],
                            mat['metallic'],
                            mat['normal'],
                            envmap4render,
                            pred_image,
                            None,
                        )
                        optimization_history.append({
                            "loop": loop_num,
                            "phase": f"material_{optimize_part}",
                            "epoch": epoch,
                            "loss_mse": loss_mse.item(),
                            "loss_l1": loss_l1.item(),
                            "loss_aux": float(aux_loss.detach().item()) if torch.is_tensor(aux_loss) else float(aux_loss),
                        })
                        
                        for param_group in opt_params.param_groups:
                            current_lr = param_group['lr']

                        early_stopping(loss_mse.item())
                        opt_params.step()
                        opt_params.zero_grad()
                        if current_lr > 1.5e-4:
                            scheduler_params.step()

                        pbar.set_postfix(loss=loss.item(), render=render_loss.item(), aux=aux_loss.item(), 
                                         lr=f"{current_lr:.2e}")
                        pbar.update(1)
                        
                        if epoch % 10 == 0 or early_stopping.early_stop:
                            all_image = torch.stack([gt_image_srgb, pred_image_srgb, mat['albedo'], 
                                                    mat['roughness'].repeat(1, 1, 3), 
                                                    mat['metallic'].repeat(1, 1, 3), mat['normal']], dim=0)
                            all_image = make_grid(all_image.permute(0, 3, 1, 2), nrow=3)
                            
                            mat_frame_path = os.path.join(mat_frames_dir, f'mat_frame_{loop_num}_{optimize_part}_{epoch:04d}.png')
                            save_image(all_image, mat_frame_path)
                            mat_frames.append(mat_frame_path)
                            
                        if early_stopping.early_stop:
                            print("Early stopping")
                            if 'a' in optimize_part:
                                mat['albedo'] = material_saver.best_albedo
                            if 'r' in optimize_part:
                                mat['roughness'] = material_saver.best_roughness
                            if 'm' in optimize_part:
                                mat['metallic'] = material_saver.best_metallic
                            if 'n' in optimize_part:
                                mat['normal'] = material_saver.best_normal
                            break
                            
                mat['albedo'] = material_saver.best_albedo
                mat['roughness'] = material_saver.best_roughness
                mat['metallic'] = material_saver.best_metallic
                mat['normal'] = material_saver.best_normal
                
                material_saver.save_results(os.path.join(output_dir, f'best_results'))
                
                torch.cuda.empty_cache()
                gc.collect()
            else:

                opt_brdf = torch.optim.AdamW(brdf_net.parameters(), lr=3e-4)
                scheduler_brdf = StepLR(opt_brdf, step_size=100, gamma=0.8)
                # scheduler_brdf = OneCycleLR(opt_brdf, max_lr=5e-4, total_steps=num_epochs//50//loop_num)
                if 'a' in optimize_part:
                    early_stopping = EarlyStopping(patience=200//loop_num, min_delta=0.005)
                else:
                    early_stopping = EarlyStopping(patience=200//loop_num, min_delta=0.001)
                with tqdm(total=num_epochs,desc=f'Opt {optimize_part}', unit='e',file=sys.stdout) as pbar:
                    for epoch in range(num_epochs):
                        if model_name == 'unet':
                            albedo_pred, roughness_pred,metallic_pred = brdf_net(start_arm_unet)
                            if 'a' in optimize_part:
                                albedo = albedo_pred.squeeze(0).permute(1,2,0)
                                mat['albedo'] = albedo
                            if 'r' in optimize_part:
                                roughness = roughness_pred.squeeze(0).permute(1,2,0)
                                mat['roughness'] = roughness
                            if 'm' in optimize_part:
                                metallic = metallic_pred.squeeze(0).permute(1,2,0)
                                mat['metallic'] = metallic

                        elif model_name == 'mlp' or model_name == 'pos_mlp':
                            arm_pred = brdf_net(start_arm)
                            albedo = (arm_pred[...,0:3]).clamp(0,1)
                            roughness = (arm_pred[...,3:4] * 0.93 + 0.07).clamp(0,1)
                            metallic = (arm_pred[...,4:5]).clamp(0,1)
                            if output_type == 'armn':
                                normal = NF.normalize(arm_pred[...,5:8],p=2, dim=1)
                            if 'a' in optimize_part:
                                mat['albedo'] = albedo.reshape(img_h, img_w, 3)
                            if 'r' in optimize_part:
                                mat['roughness'] = roughness.reshape(img_h, img_w, 1)
                            if 'm' in optimize_part:
                                mat['metallic'] = metallic.reshape(img_h, img_w, 1)
                            if 'n' in optimize_part:
                                mat['normal'] = normal.reshape(img_h, img_w, 3)
                        else:
                            raise ValueError('model_name should be unet or mlp or pos_mlp')
                        if use_mask:
                            mat['roughness'][mat['mask']] = mat['roughness'][mat['mask']].mean()
                            mat['metallic'][mat['mask']] = mat['metallic'][mat['mask']].mean()
                        if use_mesh_normal:
                            pred_image = render_w_brdf(
                                scene,
                                mat['albedo'],
                                mat['roughness'],
                                mat['metallic'],
                                None,
                                spp,
                                seed=loop_num * 100000 + epoch,
                            )
                        else:
                            pred_image = render_w_brdf(
                                scene,
                                mat['albedo'],
                                mat['roughness'],
                                mat['metallic'],
                                mat['normal'],
                                spp,
                                seed=loop_num * 100000 + epoch,
                            )
                        pred_image_srgb = _linear_to_display_srgb(pred_image)
                        gt_image_srgb = _linear_to_display_srgb(gt_image)

                        loss_mse = NF.mse_loss(pred_image_srgb, gt_image_srgb)
                        loss_l1 = NF.l1_loss(pred_image_srgb, gt_image_srgb)
                        if 'a' in optimize_part:
                            loss_a = NF.l1_loss(albedo.reshape(img_h, img_w, 3), albedo_ori)
                        else:
                            loss_a = 0
                        if 'r' in optimize_part:
                            loss_r = NF.l1_loss(roughness.reshape(img_h, img_w, 1), roughness_ori)
                        else:
                            loss_r = 0
                        if 'm' in optimize_part:
                            loss_m = NF.l1_loss(metallic.reshape(img_h, img_w, 1), metallic_ori)
                        else:
                            loss_m = 0
                        if 'n' in optimize_part:
                            loss_n = NF.l1_loss(normal.reshape(img_h, img_w, 3), normal_ori)
                        else:
                            loss_n = 0
                        scale_raito = loss_l1.detach()/loss_mse.detach()
                        aux_loss = loss_a + (loss_r + loss_m + loss_n)
                        render_loss = 3 * scale_raito * loss_mse + loss_l1
                        loss = render_loss + aux_loss * scale_delta

                        loss.backward()
                        current_weights = brdf_net.state_dict()
                        
                        material_saver.update(
                            loss_mse.item(),
                            mat['albedo'],
                            mat['roughness'],
                            mat['metallic'],
                            mat['normal'],
                            envmap4render,
                            pred_image,
                            current_weights,
                        )
                        optimization_history.append({
                            "loop": loop_num,
                            "phase": f"material_{optimize_part}",
                            "epoch": epoch,
                            "loss_mse": loss_mse.item(),
                            "loss_l1": loss_l1.item(),
                            "loss_aux": float(aux_loss.detach().item()) if torch.is_tensor(aux_loss) else float(aux_loss),
                        })
                        for param_group in opt_brdf.param_groups:
                            current_lr = param_group['lr']

                        early_stopping(loss_mse.item())
                        opt_brdf.step()
                        opt_brdf.zero_grad()
                        if current_lr > 1.5e-4:
                            scheduler_brdf.step()

                        pbar.set_postfix(loss=loss.item(),render=render_loss.item(),aux=aux_loss.item(),lr=f"{current_lr:.2e}")
                        pbar.update(1)
                        if epoch % 10 == 0 or early_stopping.early_stop:
                            all_image = torch.stack([gt_image_srgb,pred_image_srgb,mat['albedo'], mat['roughness'].repeat(1, 1, 3), mat['metallic'].repeat(1, 1, 3),mat['normal']], dim=0)
                            all_image = make_grid(all_image.permute(0,3,1,2), nrow=3)
                            
                            # Save to frames directory with frame number in filename
                            mat_frame_path = os.path.join(mat_frames_dir, f'mat_frame_{loop_num}_{optimize_part}_{epoch:04d}.png')
                            save_image(all_image, mat_frame_path)
                            mat_frames.append(mat_frame_path)
                            

                        if early_stopping.early_stop:
                            print("Early stopping")
                            if 'a' in optimize_part:
                                mat['albedo'] = material_saver.best_albedo
                            if 'r' in optimize_part:
                                mat['roughness'] = material_saver.best_roughness
                            if 'm' in optimize_part:
                                mat['metallic'] = material_saver.best_metallic
                            if 'n' in optimize_part:
                                mat['normal'] = material_saver.best_normal

                            break
                    torch.cuda.empty_cache()
                    gc.collect()
                mat['albedo'] = material_saver.best_albedo
                mat['roughness'] = material_saver.best_roughness
                mat['metallic'] = material_saver.best_metallic
                mat['normal'] = material_saver.best_normal
                if material_saver.best_brdfnet_weight is not None:
                    brdf_net.load_state_dict(material_saver.best_brdfnet_weight)

                material_saver.save_results(os.path.join(output_dir, f'best_results'))
            with open(os.path.join(output_dir, "optimization_history.json"), "w", encoding="utf-8") as file:
                json.dump(optimization_history, file, indent=2)
        # optimize_order = ['arm']

        
    # After all optimization is done, create videos from collected frames
    if env_frames:
        create_video_from_frames(env_frames, os.path.join(output_dir, 'env_optimization.mp4'), fps=10)
    
    if mat_frames:
        create_video_from_frames(mat_frames, os.path.join(output_dir, 'mat_optimization.mp4'), fps=10)

    # Final high-quality render with best materials
    if final_spp > spp:
        print(f"Final high-quality render with spp={final_spp} ...")
        final_params = mi.traverse(scene)
        final_params['emitter.data'] = final_envmap
        final_params.update()
        if use_mesh_normal:
            final_img = render_w_brdf(
                scene,
                mat['albedo'],
                mat['roughness'],
                mat['metallic'],
                None,
                final_spp,
                seed=42,
            )
        else:
            final_img = render_w_brdf(
                scene,
                mat['albedo'],
                mat['roughness'],
                mat['metallic'],
                mat['normal'],
                final_spp,
                seed=42,
            )
        hq_path = os.path.join(output_dir, 'best_results', 'rendered_img_hq.exr')
        final_img_np = final_img.detach().cpu().numpy()
        mi.util.write_bitmap(hq_path, final_img_np)
        final_preview = np.clip(
            _linear_to_srgb_numpy(np.clip(final_img_np, 0.0, 1.0)) * 255.0 + 0.5,
            0,
            255,
        ).astype(np.uint8)
        cv2.imwrite(
            os.path.join(output_dir, 'best_results', 'rendered_img_hq.png'),
            cv2.cvtColor(final_preview, cv2.COLOR_RGB2BGR),
        )
        print(f"Saved high-quality render: {hq_path}")

# Add this helper function to create videos from frames
def create_video_from_frames(frame_paths, output_path, fps=10):
    if not frame_paths:
        print(f"No frames found to create video: {output_path}")
        return
    try:
        print(f"Creating video from {len(frame_paths)} frames: {output_path}")
        frames = [imageio.imread(path) for path in tqdm(frame_paths, desc="Loading frames")]
        imageio.mimwrite(output_path, frames, format='ffmpeg', fps=fps, quality=8)
        print(f"Video saved to: {output_path}")
    except Exception as e:
        print(f"Error creating video: {str(e)}")

def countdown(seconds):
    while seconds:
        mins, secs = divmod(seconds, 60)
        timeformat = '{:02d}:{:02d}'.format(mins, secs)
        print(timeformat, end='\r')
        time.sleep(1)
        seconds -= 1
    print('00:00')

def inverse_image(
    img_inverse_path,
    save_name,
    opt_src,
    opt_order,
    use_mask,
    opt_env_from,
    lights_json,
    save_path=None,
    env_coordinate_type="spherical",
    model_name="pos_mlp",
    work_scale=0.5,
    rebuild_mesh=False,
    moge2_model_name="Ruicheng/moge-2-vitl-normal",
    geometry_scale=None,
    radiance_scale=1.0,
    visible_offset=0.005,
):
    print(f"Inverse image: {img_inverse_path}")
    spp = 64
    use_sh = False

    output_dir = get_output_dir(save_name, save_path)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "best_results"), exist_ok=True)

    # ILE geometry and radiance stay fixed. The existing optimization below
    # continues to update only the HDRI and Materialist material parameters.
    light_set = load_ile_lights(
        lights_json,
        include_windows=True,
        geometry_scale=geometry_scale,
    )
    print(
        f"Loaded {len(light_set.lights)} fixed IndoorLightEditing lights: "
        + ", ".join(light.name for light in light_set.lights)
    )

    # Read the input image – support both EXR (linear HDR) and PNG/JPG (sRGB)
    is_exr = img_inverse_path.lower().endswith('.exr')
    if is_exr:
        # EXR: mi.Bitmap returns linear RGB float32
        raw_image = np.array(mi.Bitmap(img_inverse_path), dtype=np.float32)
        if raw_image.ndim == 2:
            raw_image = np.repeat(raw_image[:, :, None], 3, axis=2)
        raw_image = raw_image[..., :3]
        print(f"EXR loaded: shape={raw_image.shape}, range=[{raw_image.min():.4f}, {raw_image.max():.4f}]")
        # MoGe2 consumes an 8-bit display image; MatNet and the inverse-render
        # target keep the original linear HDR values in raw_image.
        raw_image_uint8 = np.clip(
            _linear_to_srgb_numpy(raw_image) * 255.0 + 0.5,
            0,
            255,
        ).astype(np.uint8)
    else:
        # PNG/JPG: cv2 returns uint8 BGR
        image_bgr = cv2.imread(
            img_inverse_path,
            cv2.IMREAD_UNCHANGED,
        )
        if image_bgr is None:
            raise FileNotFoundError(img_inverse_path)

        if image_bgr.ndim == 2:
            image_bgr = cv2.cvtColor(
                image_bgr,
                cv2.COLOR_GRAY2BGR,
            )

        if image_bgr.shape[2] == 4:
            alpha = image_bgr[:, :, 3]
            alpha_max = (
                np.iinfo(alpha.dtype).max
                if np.issubdtype(alpha.dtype, np.integer)
                else 1.0
            )
            if np.any(alpha != alpha_max):
                raise ValueError(
                    "Input has non-opaque alpha; composite it onto an "
                    "explicit background before inverse rendering."
                )
            image_bgr = image_bgr[:, :, :3]

        image_rgb = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2RGB,
        )
        if np.issubdtype(image_rgb.dtype, np.integer):
            dtype_max = float(np.iinfo(image_rgb.dtype).max)
            image_srgb = image_rgb.astype(np.float32) / dtype_max
        elif np.issubdtype(image_rgb.dtype, np.floating):
            if not np.isfinite(image_rgb).all():
                raise ValueError("PNG/JPG input contains NaN or Inf")
            image_srgb = np.asarray(image_rgb, dtype=np.float32)
        else:
            raise TypeError(f"Unsupported image dtype: {image_rgb.dtype}")
        image_srgb = np.clip(image_srgb, 0.0, 1.0)
        raw_image_uint8 = np.clip(
            image_srgb * 255.0 + 0.5,
            0,
            255,
        ).astype(np.uint8)
        # MatNet was trained on linear images. Keep the sRGB uint8 copy only
        # for MoGe2 and use linear float32 everywhere else.
        raw_image = _srgb_to_linear_numpy(image_srgb)
        warnings.warn('The input image is in PNG/JPG format, assume it is sRGB.', UserWarning)

    original_h, original_w = raw_image.shape[:2]
    
    if not np.isfinite(work_scale) or work_scale <= 0:
        raise ValueError("work_scale must be a finite positive number")

    # Use MoGe2 for camera intrinsics and geometry estimation
    print("Running MoGe2 inference for camera intrinsics and geometry...")
    geo_camera, moge2_output = estimate_camera_moge2(
        raw_image_uint8, 
        device="cuda", 
        model_name=moge2_model_name
    )
    print(f"MoGe2 estimated FOV: hfov={geo_camera.hfov_deg:.2f}°, vfov={geo_camera.vfov_deg:.2f}°")

    pred_mat = None
    preprocess_meta = None
    if opt_src != "skip" or opt_order != ["skip"]:
        model_path = hf_hub_download(
            repo_id="Lez/MatNet",
            filename="matnet_weights.pth",
            repo_type="model",
        )
        matnet = MaterialNet(
            encoder="vitb",
            features=128,
            out_channels=[96, 192, 384, 768],
            use_bn=False,
            use_clstoken=False,
        )
        matnet.load_state_dict(torch.load(model_path, weights_only=True))
        matnet = matnet.cuda().eval()
        pred_mat, img_inverse, preprocess_meta = matnet.infer_image_scaled(
            raw_image,
            scale=work_scale,
        )
    else:
        work_h = max(1, int(round(original_h * work_scale)))
        work_w = max(1, int(round(original_w * work_scale)))
        interpolation = cv2.INTER_AREA if work_scale <= 1.0 else cv2.INTER_CUBIC
        img_inverse = cv2.resize(raw_image, (work_w, work_h), interpolation=interpolation)
        preprocess_meta = {
            "original_size": [original_w, original_h],
            "work_size": [work_w, work_h],
            "crop_box": [0, 0, original_w, original_h],
            "scale_x": work_w / float(original_w),
            "scale_y": work_h / float(original_h),
            "pixel_transform": [
                [work_w / float(original_w), 0.0, 0.0],
                [0.0, work_h / float(original_h), 0.0],
                [0.0, 0.0, 1.0],
            ],
            "center_crop": False,
        }

    work_h, work_w = img_inverse.shape[:2]
    K_work = scale_intrinsics(
        geo_camera.K,
        source_hw=(original_h, original_w),
        target_hw=(work_h, work_w),
        preprocess_meta=preprocess_meta,
    )
    K_work = make_mitsuba_compatible_K(K_work)

    camera_meta_path = os.path.join(output_dir, "camera_meta.json")
    camera_meta = write_materialist_camera_json(
        camera_meta_path,
        K_work=K_work,
        work_hw=(work_h, work_w),
        geocalib_result=geo_camera,
        preprocess_meta=preprocess_meta,
    )

    c2w = torch.tensor(camera_meta["to_world"], dtype=torch.float32)[0][:3, :]
    cam_cfg = {
        "to_world": c2w,
        "fov": float(camera_meta["y_fov"][0]),
        "fov_axis": "y",
        "K": torch.from_numpy(K_work),
        "width": work_w,
        "height": work_h,
    }

    print("Camera configuration:")
    print(f"  source        = moge2 ({moge2_model_name})")
    print(f"  original size = {original_w} x {original_h}")
    print(f"  working size  = {work_w} x {work_h}")
    print(f"  hfov / vfov   = {camera_meta['x_fov'][0]:.3f} / {camera_meta['y_fov'][0]:.3f} deg")
    print(f"  fx / fy       = {K_work[0, 0]:.3f} / {K_work[1, 1]:.3f} px")
    print("  MoGe2 roll/pitch ignored; Materialist pose is fixed.")

    if pred_mat is not None:
        albedo = pred_mat["albedo"]
        moge2_normal_camera = None
        moge2_normal_valid_mask = None
        # Match the normal map to the MoGe2 depth mesh. MoGe2 normals use
        # OpenCV camera coordinates (Y down, Z forward), while the mesh is
        # rotated by Rx(180 degrees) below, so the vectors need the same
        # (x, y, z) -> (x, -y, -z) transform.
        if moge2_output.get("normal") is not None:
            moge2_normal_camera, moge2_normal_valid_mask = prepare_moge2_normal(
                moge2_output,
                target_hw=(work_h, work_w),
            )
            normal = camera_to_materialist_vectors(moge2_normal_camera)

            # Invalid MoGe normals should not turn valid geometry black. The
            # Example1 normal map is fully valid, but retain MaterialNet as a
            # per-pixel fallback for other inputs.
            if not moge2_normal_valid_mask.all():
                fallback_normal = np.asarray(pred_mat["normal"], dtype=np.float32)
                fallback_length = np.linalg.norm(
                    fallback_normal,
                    axis=-1,
                    keepdims=True,
                )
                fallback_normal = np.divide(
                    fallback_normal,
                    fallback_length,
                    out=np.zeros_like(fallback_normal),
                    where=fallback_length > 1e-8,
                )
                normal = np.where(
                    moge2_normal_valid_mask[..., None],
                    normal,
                    fallback_normal,
                ).astype(np.float32)
                normal_source = "moge2_with_materialnet_fallback"
            else:
                normal_source = "moge2"
            print(
                "Using MoGe2 normal map in Materialist coordinates "
                f"({moge2_normal_valid_mask.sum()}/"
                f"{moge2_normal_valid_mask.size} valid pixels)"
            )
        else:
            normal = np.asarray(pred_mat["normal"], dtype=np.float32)
            normal_source = "materialnet"
            print("Using MaterialNet normal map (MoGe2 normal not available)")
        
        roughness = pred_mat["roughness"]
        metallic = pred_mat["metallic"]
        
        # Use MoGe2 depth instead of MaterialNet depth. Invalid MoGe pixels are
        # kept out of interpolation and converted to the mesh sentinel value 0.
        depth, moge2_valid_mask = prepare_moge2_depth(
            moge2_output,
            target_hw=(work_h, work_w),
        )
        print(
            "Using MoGe2 depth map "
            f"({moge2_valid_mask.sum()}/{moge2_valid_mask.size} valid pixels)"
        )

        if depth.shape[:2] != (work_h, work_w):
            raise RuntimeError(f"Depth size {depth.shape[:2]} != working size {(work_h, work_w)}")

        mat = {
            "gt_image": torch.from_numpy(img_inverse).float().cuda(),
            "albedo": torch.from_numpy(albedo).cuda().clamp(0, 1),
            "normal": torch.from_numpy(normal).cuda(),
            "roughness": torch.from_numpy(roughness).unsqueeze(-1).cuda().clamp(0.07, 1),
            "metallic": torch.from_numpy(metallic).unsqueeze(-1).cuda().clamp(0, 1),
            "depth": torch.from_numpy(depth).unsqueeze(-1).cuda(),
        }

        mi.util.write_bitmap(os.path.join(output_dir, "albedoPred.exr"), albedo)
        mi.util.write_bitmap(os.path.join(output_dir, "normalPred.exr"), normal)
        if moge2_normal_camera is not None:
            mi.util.write_bitmap(
                os.path.join(output_dir, "moge2_normal_camera.exr"),
                moge2_normal_camera,
            )
            mi.util.write_bitmap(
                os.path.join(output_dir, "moge2_normal.exr"),
                camera_to_materialist_vectors(moge2_normal_camera),
            )
            np.save(
                os.path.join(output_dir, "moge2_normal_valid_mask.npy"),
                moge2_normal_valid_mask,
                allow_pickle=False,
            )
        mi.util.write_bitmap(os.path.join(output_dir, "roughnessPred.png"), roughness)
        mi.util.write_bitmap(os.path.join(output_dir, "metallicPred.png"), metallic)
        mi.util.write_bitmap(os.path.join(output_dir, "depthPred.exr"), depth)
        # img_inverse is linear float32 for both EXR and PNG/JPG inputs.
        mi.util.write_bitmap(os.path.join(output_dir, "gt_image.exr"), img_inverse)
        gt_image_srgb_u8 = np.clip(
            _linear_to_srgb_numpy(img_inverse) * 255.0 + 0.5,
            0,
            255,
        ).astype(np.uint8)
        if not cv2.imwrite(
            os.path.join(output_dir, "gt_image.png"),
            cv2.cvtColor(gt_image_srgb_u8, cv2.COLOR_RGB2BGR),
        ):
            raise RuntimeError("Failed to write gt_image.png")
        
        # Save MoGe2 point cloud if available
        if moge2_output['points'] is not None:
            moge2_points = moge2_output['points']
            if moge2_points.shape[:2] != (work_h, work_w):
                # Resize point map
                moge2_points_resized = cv2.resize(moge2_points, (work_w, work_h), interpolation=cv2.INTER_LINEAR)
            else:
                moge2_points_resized = moge2_points
            mi.util.write_bitmap(os.path.join(output_dir, "moge2_points.exr"), moge2_points_resized)

        config = {
            "img_path": img_inverse_path,
            "save_name": save_name,
            "opt_src": opt_src,
            "opt_order": opt_order,
            "use_mask": use_mask,
            "opt_env_from": opt_env_from,
            "env_coordinate_type": env_coordinate_type,
            "model_name": model_name,
            "work_scale": work_scale,
            "camera_source": "moge2",
            "moge2_model": moge2_model_name,
            "lights_json": str(light_set.source_path),
            "fixed_ile_lights": [light.name for light in light_set.lights],
            "fixed_ile_window_count": light_set.metadata["windows_included"],
            "ile_geometry_scale": light_set.metadata["geometry_scale"],
            "ile_radiance_scale": radiance_scale,
            "ile_visible_lamp_offset": visible_offset,
            "camera_meta_path": camera_meta_path,
            "image_size": [work_h, work_w],
            "spp": spp,
            "optimization_color_space": "clipped_linear_to_standard_srgb",
            "saved_render_color_space": "linear",
            "render_seed_strategy": "fixed_loop_epoch_schedule",
            "window_sample_weight_clamp": float(
                os.environ.get("MATERIALIST_WINDOW_SAMPLE_WEIGHT_CLAMP", "100.0")
            ),
            "output_type": "armn" if "n" in str(opt_order) else "arm",
            "normal_source": normal_source,
            "use_mesh_normal": False,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as file:
            json.dump(config, file, indent=4, ensure_ascii=False)

        if use_mask:
            mask_root_dir = os.path.join(output_dir, "best_results")
            mask_path = os.path.join(mask_root_dir, "mask.png")
            if os.path.exists(mask_path):
                mask = plt.imread(mask_path)
                mask = torch.tensor(np.asarray(mask)).bool().cuda()[..., 0]
                mat["mask"] = mask
            else:
                warnings.warn("No mask found; continuing without mask.", UserWarning)
                countdown(20)
                use_mask = False

        mesh_path = os.path.join(output_dir,f'{save_name}.ply')
        mesh_mask_path = os.path.join(output_dir,'mesh_mask.png')
        mesh_mask = None
        if os.path.exists(mesh_mask_path):
            mesh_mask = plt.imread(mesh_mask_path)
            mesh_mask = np.array(mesh_mask, dtype=np.bool_)
            if mesh_mask.ndim > 2:  # If it's an RGB image, use only the first channel
                mesh_mask = mesh_mask[..., 0]
            if mesh_mask.shape != (work_h, work_w):
                mesh_mask = cv2.resize(
                    mesh_mask.astype(np.uint8),
                    (work_w, work_h),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
        if rebuild_mesh or not os.path.exists(mesh_path):
            # MoGe2 outputs standard metric depth (larger = farther), which is
            # exactly what depth_file_to_mesh expects for back-projection.
            # No inversion needed (unlike MaterialNet's inverse depth).
            depth_for_mesh = depth.copy()
            if mesh_mask is not None:
                depth_for_mesh[mesh_mask] = 0
                print(f"Applied mask from {mesh_mask_path} to depth map")
            mesh, b_points  = depth_file_to_mesh(depth_for_mesh,cameraMatrix=K_work, minAngle=6, sun3d=False, depthScale=1.0)
            mesh = rotate_mesh_around_x(mesh, 180)
            o3d.io.write_triangle_mesh(mesh_path, mesh)
            print(f"Rebuilt mesh with per-image K: {mesh_path}")
        else:
            print(f"Using existing mesh: {mesh_path}")
            print("Pass --rebuild_mesh after changing image scale or camera intrinsics.")

        # Render mesh preview with initial predictions
        render_mesh_preview(mesh_path, camera_meta_path, output_dir, spp=64)

        if opt_env_from > 1:
            opt_envmap_path = os.path.join(output_dir, "best_results", "envmap.hdr")
            if os.path.exists(opt_envmap_path):
                print(f"Load envmap from {opt_envmap_path}")
                mat["gt_envmap"] = torch.from_numpy(
                    np.array(mi.Bitmap(opt_envmap_path))
                ).cuda()
            else:
                print(f"No envmap found in {opt_envmap_path}; using envmap=1 instead")
    else:
        print("Load pre-optimized BRDF")
        mesh_path = os.path.join(output_dir, f"{save_name}.ply")
        opted_albedo = np.array(mi.Bitmap(os.path.join(output_dir, "best_results", "albedo.exr")), dtype=np.float32)
        opted_roughness = np.array(mi.Bitmap(os.path.join(output_dir, "best_results", "roughness.exr")), dtype=np.float32)
        opted_metallic = np.array(mi.Bitmap(os.path.join(output_dir, "best_results", "metallic.exr")), dtype=np.float32)
        opted_normal = np.array(mi.Bitmap(os.path.join(output_dir, "best_results", "normal.exr")), dtype=np.float32)

        # Old checkpoints may contain an untransformed normal map that was
        # harmless only because use_mesh_normal used to ignore it. Recompute
        # the fixed shading normal from the current MoGe2 result when resuming.
        if moge2_output.get("normal") is not None:
            moge2_normal_camera, moge2_normal_valid_mask = prepare_moge2_normal(
                moge2_output,
                target_hw=(work_h, work_w),
            )
            moge2_normal = camera_to_materialist_vectors(moge2_normal_camera)
            if opted_normal.shape != moge2_normal.shape:
                raise ValueError(
                    "Pre-optimized normal resolution does not match the "
                    f"current MoGe2 result: {opted_normal.shape} vs "
                    f"{moge2_normal.shape}"
                )
            opted_normal = np.where(
                moge2_normal_valid_mask[..., None],
                moge2_normal,
                opted_normal,
            ).astype(np.float32)
            normal_source = "moge2_with_preoptimized_fallback"
        else:
            normal_source = "preoptimized"
        mat = {
            "albedo": torch.from_numpy(opted_albedo).cuda().clamp(0, 1),
            "roughness": torch.from_numpy(opted_roughness).unsqueeze(-1).cuda().clamp(0.07, 1),
            "metallic": torch.from_numpy(opted_metallic).unsqueeze(-1).cuda().clamp(0, 1),
            "normal": torch.from_numpy(opted_normal).cuda(),
            "gt_image": torch.from_numpy(img_inverse).float().cuda(),
        }

    # The geometry and shading normals now share the same MoGe2 source. Keep
    # use_mesh_normal disabled even when normal is fixed; "n" only controls
    # whether that MoGe2 initialization is optimized with the material.
    use_mesh_normal = False
    if "n" in str(opt_order):
        output_type = "armn"
        print(f"Optimize normal map initialized from {normal_source}")
    else:
        output_type = "arm"
        print(f"Use fixed normal map from {normal_source}")

    # Keep resumed runs from leaving the old use_mesh_normal=true metadata in
    # config.json after the actual scene has switched to the MoGe2 normal map.
    config_path = os.path.join(output_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as file:
            config = json.load(file)
        config.update({
            "normal_source": normal_source,
            "use_mesh_normal": use_mesh_normal,
            "output_type": output_type,
        })
        with open(config_path, "w", encoding="utf-8") as file:
            json.dump(config, file, indent=4, ensure_ascii=False)

    scene = load_estimated_mesh(
        mesh_path,
        use_mesh_normal,
        camera_meta_path,
        light_set.lights,
        radiance_scale=radiance_scale,
        visible_offset=visible_offset,
    )
    optimize_envmap_ARMN(
        cam_cfg=cam_cfg,
        scene=scene,
        save_folder=save_name,
        mat=mat,
        use_mesh_normal=use_mesh_normal,
        output_type=output_type,
        optimize_order=opt_order,
        use_gt_scene=False,
        model_name=model_name,
        spp=spp,
        opt_env_from=opt_env_from,
        opt_src=opt_src,
        use_mask=use_mask,
        save_path=save_path,
        env_coordinate_type=env_coordinate_type,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Inverse-render a given image using MoGe2 for geometry estimation",
    )
    parser.add_argument("--img_inverse_path", required=True, type=str)
    parser.add_argument("--save_name", required=True, type=str)
    parser.add_argument("--opt_src", required=True, type=str, default="arm")
    parser.add_argument("--opt_order", nargs="+", default=["arm"])
    parser.add_argument("--use_mask", action="store_true")
    parser.add_argument("--opt_env_from", default=0, type=int)
    parser.add_argument(
        "--lights_json",
        "--lights-json",
        required=True,
        type=str,
        help=(
            "IndoorLightEditing light_predictions.json; visible/invisible "
            "lamps and windows are loaded as fixed local lights"
        ),
    )
    parser.add_argument(
        "--geometry_scale",
        "--geometry-scale",
        default=None,
        type=float,
        help="Optional ILE geometry scale override",
    )
    parser.add_argument(
        "--radiance_scale",
        "--radiance-scale",
        default=1.0,
        type=float,
        help="Fixed global multiplier for all ILE lamp/window radiance",
    )
    parser.add_argument(
        "--visible_offset",
        "--visible-offset",
        default=0.005,
        type=float,
        help="Move visible lamps toward the camera; windows are not moved",
    )
    parser.add_argument("--save_path", default=None, type=str)
    parser.add_argument(
        "--env_coordinate_type",
        default="spherical",
        choices=["spherical", "uv"],
    )
    parser.add_argument(
        "--model_name",
        default="pos_mlp",
        choices=["pos_mlp", "none"],
    )
    parser.add_argument(
        "--work_scale",
        default=0.5,
        type=float,
        help="Aspect-ratio-preserving scale applied to the complete input image",
    )
    parser.add_argument(
        "--rebuild_mesh",
        action="store_true",
        help="Regenerate the mesh with the current working-resolution K",
    )
    parser.add_argument(
        "--moge2_model",
        default="/home/majortom/project/datasets/ckpt/moge2_vitl_normal.pt",
        type=str,
        help="MoGe2 model name from HuggingFace Hub",
    )
    args = parser.parse_args()
    if args.geometry_scale is not None and args.geometry_scale <= 0:
        parser.error("--geometry-scale must be positive")
    if args.radiance_scale <= 0:
        parser.error("--radiance-scale must be positive")
    if args.visible_offset < 0:
        parser.error("--visible-offset must be non-negative")
    return args


def inverse_real(args):
    inverse_image(
        args.img_inverse_path,
        args.save_name,
        args.opt_src,
        args.opt_order,
        use_mask=args.use_mask,
        opt_env_from=args.opt_env_from,
        lights_json=args.lights_json,
        save_path=args.save_path,
        env_coordinate_type=args.env_coordinate_type,
        model_name=args.model_name,
        work_scale=args.work_scale,
        rebuild_mesh=args.rebuild_mesh,
        moge2_model_name=args.moge2_model,
        geometry_scale=args.geometry_scale,
        radiance_scale=args.radiance_scale,
        visible_offset=args.visible_offset,
    )


if __name__ == '__main__':
    args = parse_args()
    try:
        inverse_real(args)
    finally:
        dr.sync_thread()
        gc.collect()
        torch.cuda.empty_cache()
        dr.flush_malloc_cache()
