import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose

from .dinov2 import DINOv2
from .util.blocks import FeatureFusionBlock, _make_scratch
from .util.transform import Resize, NormalizeImage, PrepareForNet
import warnings
import numpy as np

def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class ConvBlock(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_feature, out_feature, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_feature),
            nn.ReLU(True)
        )
    
    def forward(self, x):
        return self.conv_block(x)


class DPTHead(nn.Module):
    def __init__(
        self, 
        in_channels, 
        features=256, 
        use_bn=False, 
        out_channels=[256, 512, 1024, 1024], 
        use_clstoken=False,
        output_type='depth',
    ):
        super(DPTHead, self).__init__()
        
        self.use_clstoken = use_clstoken
        self.output_type = output_type
        
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])
        
        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))
        
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )
        
        self.scratch.stem_transpose = None
        
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)
        
        head_features_1 = features
        head_features_2 = 32
        
        self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
        
        if output_type == 'depth':
            self.scratch.output_conv2 = nn.Sequential(
                nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
                nn.ReLU(True),
                nn.Identity(),
            )
        elif output_type == 'material': # Albedo, Roughness, Metallic, Normal
            self.scratch.output_conv2 = nn.Sequential(
                nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(head_features_2, 8, kernel_size=1, stride=1, padding=0),
            )
        else:
            raise NotImplementedError

    
    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]
            
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            
            out.append(x)

        layer_1, layer_2, layer_3, layer_4 = out
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])        
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        out = self.scratch.output_conv1(path_1)
        out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        if self.output_type == 'depth':
            out = self.scratch.output_conv2(out)
        elif self.output_type == 'material':
            out = self.scratch.output_conv2(out)
            arm = out[:, :5]
            arm = nn.ReLU()(arm)
            normal = out[:, 5:8]
            normal = nn.Tanh()(normal)
            normal = F.normalize(normal,p=2, dim=1, eps=1e-6)
            out = torch.cat((arm, normal), 1)
            
        return out


class MaterialNet(nn.Module):
    def __init__(
        self, 
        encoder='vitb', 
        features=128, 
        out_channels=[96, 192, 384, 768], 
        use_bn=False, 
        use_clstoken=False
    ):
        super().__init__()
        
        self.intermediate_layer_idx = {
            'vitb': [2, 5, 8, 11], 
        }
        
        self.encoder = encoder
        self.pretrained = DINOv2(model_name=encoder)
        
        self.depth_head = DPTHead(self.pretrained.embed_dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken,output_type='depth')
        self.material_head = DPTHead(self.pretrained.embed_dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken, output_type='material')
        # breakpoint()

    def forward(self, x):
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
        
        features = self.pretrained.get_intermediate_layers(x, self.intermediate_layer_idx[self.encoder], return_class_token=True)
        depth = self.depth_head(features, patch_h, patch_w)
        depth = F.relu(depth)

        armn = self.material_head(features, patch_h, patch_w)
        albedo = armn[:, :3]
        roughness = armn[:, 3:4]
        metallic = armn[:, 4:5]
        normal = armn[:, 5:8]

        out = {
            'depth': depth,
            'albedo': albedo,
            'roughness': roughness,
            'metallic': metallic,
            'normal': normal
        }
        return out
    
    @torch.no_grad()
    def infer_image_scaled(
        self,
        raw_image: np.ndarray,
        scale: float = 0.5,
        patch_size: int = 14,
    ):
        """
        保持输入图像宽高比，将分辨率缩小 scale 倍，
        然后 padding 到 patch_size 的整数倍。

        Args:
            raw_image:
                H x W x 3 的 numpy 图像。
                整数输入会按 dtype 最大值归一化；float 输入保持原值，
                因而可以表示线性 HDR，调用方需明确其颜色空间。
            scale:
                网络工作分辨率相对于输入图像的比例；该值会直接改变
                MatNet 的实际推理分辨率。
                scale=0.5 表示长宽分别缩小两倍。
            patch_size:
                DINOv2 ViT-B/14 的 patch size，固定为 14。

        Returns:
            pred:
                MatNet 预测结果，所有属性尺寸均为 work_h x work_w。
            work_image:
                缩小后的输入图像，不包含 padding，
                可直接作为后续 inverse rendering 的目标图像。
            meta:
                原始尺寸、工作尺寸、网络实际输入尺寸及内参缩放矩阵。
                所有尺寸字段均使用 [width, height] 顺序。
        """
        if not isinstance(raw_image, np.ndarray):
            raise TypeError(
                f"raw_image must be np.ndarray, got {type(raw_image)}"
            )

        if raw_image.ndim != 3 or raw_image.shape[2] < 3:
            raise ValueError(
                f"Expected HxWx3 image, got shape {raw_image.shape}"
            )

        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"scale must be a finite positive number, got {scale}")

        if not isinstance(patch_size, int) or patch_size <= 0:
            raise ValueError(f"patch_size must be a positive integer, got {patch_size}")

        # 只保留 RGB，避免某些 PNG 带 alpha 通道。
        raw_image = raw_image[..., :3]

        orig_h, orig_w = raw_image.shape[:2]
        if orig_h <= 0 or orig_w <= 0:
            raise ValueError(f"Input image dimensions must be positive, got {raw_image.shape}")

        # 严格按照同一个 scale 缩小，保持宽高比。
        work_h = max(1, int(round(orig_h * scale)))
        work_w = max(1, int(round(orig_w * scale)))

        interpolation = (
            cv2.INTER_AREA
            if work_h < orig_h or work_w < orig_w
            else cv2.INTER_CUBIC
        )

        if (work_h, work_w) == (orig_h, orig_w):
            work_image = raw_image.copy()
        else:
            work_image = cv2.resize(
                raw_image,
                (work_w, work_h),
                interpolation=interpolation,
            )

        # 向上 padding 到 14 的整数倍。
        # 不直接拉伸到 14n，避免改变图像宽高比。
        net_h = int(np.ceil(work_h / patch_size) * patch_size)
        net_w = int(np.ceil(work_w / patch_size) * patch_size)

        pad_bottom = net_h - work_h
        pad_right = net_w - work_w

        # reflect padding 比填 0 更不容易在边界制造黑色伪影。
        padded_image = cv2.copyMakeBorder(
            work_image,
            top=0,
            bottom=pad_bottom,
            left=0,
            right=pad_right,
            borderType=cv2.BORDER_REFLECT_101,
        )

        # Integer images are code values; float images may be genuine linear
        # HDR and must never be silently divided by 255 based on their range.
        if np.issubdtype(padded_image.dtype, np.integer):
            dtype_max = float(np.iinfo(padded_image.dtype).max)
            image = padded_image.astype(np.float32) / dtype_max
        else:
            image = padded_image.astype(np.float32)

        if not np.isfinite(image).all():
            raise ValueError("MatNet input contains NaN or Inf")
        if image.max() > 10:
            warnings.warn(
                "Float MatNet input contains HDR values above 10; values are "
                "preserved because float inputs are interpreted as linear radiance.",
                UserWarning,
            )

        # PrepareForNet 会执行 HWC -> CHW 等转换。
        transform = Compose([
            PrepareForNet(),
        ])

        image = transform({"image": image})["image"]
        image = torch.from_numpy(image).unsqueeze(0)

        # 跟随模型所在设备，而不是写死 CUDA。
        device = next(self.parameters()).device
        image = image.to(device)

        mat_pred = self.forward(image)

        # 网络输出包含 padding 区域，需要裁回实际工作分辨率。
        depth = mat_pred["depth"][:, :, :work_h, :work_w]
        albedo = mat_pred["albedo"][:, :, :work_h, :work_w]
        roughness = mat_pred["roughness"][:, :, :work_h, :work_w]
        metallic = mat_pred["metallic"][:, :, :work_h, :work_w]
        normal = mat_pred["normal"][:, :, :work_h, :work_w]

        # 裁剪或插值后重新归一化 normal，保证 ||N|| = 1。
        normal = F.normalize(
            normal,
            p=2,
            dim=1,
            eps=1e-6,
        )

        pred = {
            "depth": depth[0, 0].cpu().numpy(),
            "albedo": albedo[0].permute(1, 2, 0).cpu().numpy(),
            "roughness": roughness[0, 0].cpu().numpy(),
            "metallic": metallic[0, 0].cpu().numpy(),
            "normal": normal[0].permute(1, 2, 0).cpu().numpy(),
        }

        scale_x = work_w / float(orig_w)
        scale_y = work_h / float(orig_h)
        pixel_transform = np.array(
            [
                [scale_x, 0.0, 0.0],
                [0.0, scale_y, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        meta = {
            # All image-size fields use [width, height]. This matches OpenCV's
            # resize convention and the camera metadata consumers.
            "original_size": [int(orig_w), int(orig_h)],
            "work_size": [int(work_w), int(work_h)],
            "network_size": [int(net_w), int(net_h)],
            "crop_box": [0, 0, int(orig_w), int(orig_h)],
            "scale_x": float(scale_x),
            "scale_y": float(scale_y),
            "pixel_transform": pixel_transform.tolist(),
            "padding": {
                "top": 0,
                "bottom": pad_bottom,
                "left": 0,
                "right": pad_right,
            },
            "scale": float(scale),
            "requested_scale": float(scale),
            "center_crop": False,
            "patch_size": patch_size,
        }

        return pred, work_image, meta
    
    @torch.no_grad()
    def infer_image(self, raw_image, input_size=518):
        '''
        input: raw_image (np.array), shape: (H, W, 3)
        output: dict, keys: ['depth', 'albedo', 'roughness', 'metallic', 'normal'], values: np.array
        '''
        image, (h, w) = self.image2tensor(raw_image, input_size)
        if image.mean() >= 10:
            warnings.warn('Pixel intensity is too high, input dtype may be wrong. Dividing by 255 to avoid Error.', UserWarning)
            image = image / 255.0
        mat_pred = self.forward(image)
        depth = mat_pred['depth']
        albedo = mat_pred['albedo']
        roughness = mat_pred['roughness']
        metallic = mat_pred['metallic']
        normal = mat_pred['normal']
        
        depth = F.interpolate(depth, (h, w), mode="bilinear", align_corners=True)[0, 0]
        albedo = F.interpolate(albedo, (h, w), mode="bilinear", align_corners=True)[0].permute(1, 2, 0)
        roughness = F.interpolate(roughness, (h, w), mode="bilinear", align_corners=True)[0,0]
        metallic = F.interpolate(metallic, (h, w), mode="bilinear", align_corners=True)[0,0]
        normal = F.interpolate(normal, (h, w), mode="bilinear", align_corners=True)[0].permute(1, 2, 0)
        return {'depth': depth.cpu().numpy(), 'albedo': albedo.cpu().numpy(), 'roughness': roughness.cpu().numpy(), 'metallic': metallic.cpu().numpy(), 'normal': normal.cpu().numpy()}
    
    def image2tensor(self, raw_image, input_size=518):        
        transform = Compose([
            Resize(
                width=input_size,
                height=input_size,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            PrepareForNet(),
        ])
        
        h, w = raw_image.shape[:2]

        if raw_image.dtype == 'uint8':
            image = raw_image / 255.0
        else:
            image = raw_image

        image = transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0)
        
        DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        image = image.to(DEVICE)
        return image, (h, w)
