# Materialist 项目简介

Materialist 是一个面向单张室内图像的逆渲染、重新打光与材质编辑项目。当前实现结合 MatNet、MoGe2、IndoorLightEditing 和 Mitsuba，从输入图像中恢复相机、几何、材质及显式灯光，并在统一的物理渲染场景中生成局部灯光、环境光或两者融合的结果。在此基础上，项目增加了一个可选的 FLUX.2 Klein Base 4B 照片真实化阶段：将已经基本正确的 PBR 渲染和对应 normal 作为双参考，只恢复材质与成像细节，同时尽量保持原有构图、光照、阴影和色感。

## 当前系统结构

```text
                              输入 RGB
                                 │
               ┌─────────────────┴─────────────────┐
               │                                   │
               ▼                                   ▼
        Materialist 分支                    IndoorLightEditing 分支
        ├── MoGe2 相机/K                    ├── visible lamp
        ├── MoGe2 metric depth              ├── invisible lamp
        ├── mesh 重建                       ├── visible window
        └── MatNet 材质                     ├── invisible window
                                            └── mesh/center + radiance 参数
               │                                   │
               └─────────────────┬─────────────────┘
                                 ▼
                         hybrid_light 适配层
                     坐标 / 尺度 / 投影 / emitter
                                 │
                                 ▼
                          Mitsuba PBR 渲染
                    local / env / local + env
                                 │
                   ┌─────────────┴─────────────┐
                   │                           │
                   ▼                           ▼
             线性 PBR EXR                 对齐 normal EXR
          曝光 / ACES / sRGB          解码 / 单位化 / RGB 编码
                   │                           │
                   └─────────────┬─────────────┘
                                 ▼
                    FLUX.2 Klein 双参考编辑
                                 │
                                 ▼
                         照片真实化 PNG
```

Materialist 负责场景几何和 BRDF，IndoorLightEditing 作为显式灯光 proposal 模型，Mitsuba 负责阴影、多次反射和材质—光照交互。FLUX.2 只处在物理渲染之后，负责照片质感可视化，不参与相机、几何、材质或光照求解。当前 prototype 不接入 IndoorLightEditing 的 neural indirect renderer。

## 核心能力

- 使用 MatNet 预测 albedo、roughness、metallic、normal 和相对 depth。
- 使用 MoGe2 恢复 metric depth、normal、points、有效 mask 和相机内参。
- 使用统一工作分辨率和相机 K 将深度反投影为 Materialist mesh。
- 为 IndoorLightEditing 输出原图尺寸、稠密、有限的 `float32 depth.npy`。
- 读取 IndoorLightEditing 导出的 `light_predictions.json`、lamp OBJ 和 window OBJ。
- 将 visible/invisible lamp 转换为 Mitsuba area emitter。
- 在固定灯光优化入口中，将 visible/invisible window 转换为保留 sun/sky/ground 三组方向球面高斯的可微 Mitsuba emitter。
- 使用 Materialist mesh 和逐像素 BRDF 进行 local-only、env-only 或 combined 渲染。
- 输出灯光投影诊断、raw EXR、OptiX 去噪结果和渲染 manifest。
- 保留原有环境光优化、HDRI 重渲染、材质编辑和离散灯光实验能力。
- 固定灯光优化默认采用 MoGe2 normal、PRB、独立多种子验证、灯具/网格边缘损失 mask，并将 256 SPP 最终 raw 渲染与去噪预览分开保存。
- 使用 PBR color + aligned normal 双参考，将最终渲染细化为照片级 PNG，并完整保存预处理与推理参数。

## 新增文件与功能

### MoGe2 与 IndoorLightEditing 输入

| 文件 | 功能 |
| --- | --- |
| `myutils/moge2_utils.py` | MoGe2 模型加载、mask-aware depth/points/normal resize，以及 camera → Materialist 坐标变换 |
| `test_matnet_infer_moge2.py` | 联合生成 MatNet 材质、MoGe2 几何、正确颜色空间 GT、强制重建 mesh、有效 mask、manifest 和 ILE `depth.npy` |
| `Material_net/dpt.py` | 唯一的 `infer_image_scaled()` 实现，统一 scale、宽高比和 patch padding；float HDR 不再被按 255 静默缩放 |

mesh 使用的 MoGe2 depth 保留“无效值为 0”的约定，mesh builder 会保证这些点不被三角形引用；IndoorLightEditing 的 `depth.npy` 则会用最近有效深度补齐，避免零深度参与反投影。`moge2_normal.exr` / `moge2_points.exr` 已转换到 Materialist 坐标，原始 MoGe camera-space 数据另存为 `*_camera.exr`。

### Hybrid 显式灯光桥接

| 文件 | 功能 |
| --- | --- |
| `hybrid_light/light_types.py` | 定义统一的 lamp/window mesh light、三组 window SG 参数和灯光集合数据结构 |
| `hybrid_light/io.py` | 解析并校验 ILE JSON、lamp RGB、window sun/sky/ground、OBJ 路径和 depth scale |
| `hybrid_light/coordinate.py` | 封装 ILE → Materialist 坐标、3D→2D 投影和反投影 |
| `hybrid_light/ile_window_emitter.py` | ILE 有限窗口 emitter：按出射方向计算 sun/sky/ground 三球面高斯辐射度 |
| `hybrid_light/mitsuba_builder.py` | 构造 Materialist mesh、相机、lamp area emitter、directional window emitter 和可选 envmap 场景 |
| `hybrid_light/visualization.py` | 将灯光中心、OBJ 投影轮廓和 lamp/window mask 绘制到输入图像 |
| `hybrid_light/README.md` | Hybrid prototype 的运行方式和当前约束 |
| `scripts/render_ile_lights.py` | 初版融合渲染入口，支持验证、消融和三种光照模式 |
| `scripts/optimize_ile_farfield.py` | Stage B 逐灯 scale 优化，以及 Stage C 32×16 far-field HDRI 优化入口 |
| `scripts/optimize_ile_farfield_materials.py` | 关闭 Stage B、固定全部 ILE lamp/window，支持逐像素或 PosMLP 参数化，依次优化 Stage C far-field HDRI、Stage D roughness/metallic，最后优化 albedo；包含 masked LDR loss、周期验证和独立最终渲染 |
| `tests/test_hybrid_light.py` | 坐标方向、投影 round-trip、尺度同步、lamp/window JSON、SG 参数、window null BSDF 与 offset 测试 |
| `tests/test_optimize_ile_farfield_materials.py` | 显示域裁剪、损失 mask、深度/有效区边缘和 validation gate 回归测试 |
| `materialist_indoorlightediting_hybrid_method.md` | 完整方法设计、阶段规划和实验建议 |

### FLUX.2 照片真实化

| 文件 | 功能 |
| --- | --- |
| `flux2_opt.py` | 读取线性 PBR EXR 与对齐 normal EXR，完成双参考 FLUX.2 Klein Base 4B 推理、显存调度和结果记录 |

该功能的目标不是让生成模型重新设计场景，而是为已经完成的 PBR 渲染补足真实照片中常见的细节：材质微纹理、细微表面变化、自然 roughness、边缘与接触细节，以及轻微相机质感。PBR color 始终是主参考，决定构图、相机、物体、光照、阴影、亮度、颜色和整体色感；normal 只是较弱的辅助几何参考，用于稳定平面朝向、轮廓和边界。

默认运行逻辑如下：

```text
rendered_img_hq.exr（线性 HDR）
  → 有限值/负值清理
  → 自动曝光
  → ACES tone mapping
  → linear RGB → sRGB
  → 1024×768 PBR 主参考 ─────────────┐
                                      ├→ FLUX.2 Klein Base 4B → 照片级 PNG
normal.exr（signed normal）           │
  → 自动识别 [-1,1] / [0,1]          │
  → 单位向量归一化                    │
  → (n+1)/2 RGB 编码                  │
  → 512×384 normal 辅助参考 ─────────┘
```

normal 使用较低默认分辨率，是因为 Klein 的多参考接口没有单独的参考权重参数；减少 normal token 数可以在提供几何信息的同时，避免伪彩色法线改变 PBR 的外观和场景语义。两张源 EXR 必须具有相同宽高并逐像素对齐。默认 prompt 保持简洁，只要求从 CGI/PBR 外观恢复为真实照片质感，并明确不改变光照和色感。

模型位于：

```text
/home/majortom/project/datasets/ckpt/FLUX.2-klein-base-4B
```

默认离线运行命令：

```bash
/home/majortom/miniconda3/envs/materialist5090/bin/python \
  /home/majortom/project/Materialist/flux2_opt.py --offline
```

默认推理采用 BF16、50 steps、guidance scale 4.0 和 seed 2026。可通过 `--input`、`--normal`、`--output`、`--normal-encoding`、`--long-edge`、`--normal-long-edge`、`--steps`、`--guidance-scale`、`--seed` 与 `--memory-mode` 覆盖。`--memory-mode auto` 会在显存充足时整模型放入 CUDA，否则使用 Accelerate CPU offload。

## Hybrid 渲染模式

通用 `scripts/render_ile_lights.py` 目前保持原有 lamp-only 默认语义，支持：

- `--mode local`：只使用 ILE visible/invisible lamps。
- `--mode env`：只使用给定的 environment map。
- `--mode combined`：同时使用显式局部灯和 environment map。
- `--visible-only` / `--invisible-only`：逐类检查灯光贡献和面朝向。
- `--dry-run`：不加载 CUDA/Mitsuba，只验证文件、相机、坐标和投影。
- `--geometry-scale`：显式覆盖 ILE 到 Materialist 的几何尺度。
- `--radiance-scale`：校准 ILE 相对强度与 Mitsuba radiance。
- `--use-pred-normal`：使用预测 normal；默认使用 mesh normal。

当前 Example1 的运行方式见 `hybrid_light/README.md`。

逐灯与 far-field 优化使用独立入口 `scripts/optimize_ile_farfield.py`。Stage B 保持 ILE RGB 色度不变，只优化每盏灯的非负 scalar scale；Stage C 固定局部灯，优化内部形状为 `(16, 32, 3)` 的 HDR envmap，并加入能量与球面周期 TV 正则。

固定灯光实验使用 `scripts/optimize_ile_farfield_materials.py`：完全跳过 Stage B，默认从同一份 JSON 加载 visible/invisible lamp 和 visible/invisible window。Lamp 保持原始 mesh area radiance；window 保持有限 OBJ 孔径和显式 `null` BSDF，并按 `src/srcSky/srcGrd` 的 `[RGB, direction, concentration]` 参数计算方向辐射度。只有 visible lamp 应用共面 offset，window 保持预测位置。全部 ILE 局部灯只统一乘固定 `--radiance-scale`，Stage C 只优化 HDRI；随后 Stage D 冻结 lamp、window 和 HDRI，默认先联合优化 roughness/metallic 并冻结最优结果，最后单独优化 albedo。该入口默认使用 MoGe2 normal，`--use-mesh-normal` 仅用于 PLY normal 消融；默认可微积分器为 PRB、`max_depth=4`。`--model_name none` 保留逐像素优化，`--model_name pos_mlp` 使用 Materialist PosMLP 同时参数化球面 HDRI 和 UV 空间 ARM；直接 Adam 启用 AMSGrad 与 masked update，PosMLP 显式关闭 weight decay。两者都在裁剪后的标准 sRGB 显示域计算 masked loss，排除可见 emitter、深度断层、mesh 有效区边缘，并以固定多种子周期 validation 选择 checkpoint 和 early stopping。新版完整 inference manifest 优先使用同工作分辨率的线性 `gt_image.exr`，旧结果回退 ILE JSON 原始图；曝光对齐仍为可选。Stage C 只输出一份权威的 `farfield_optimized_32x16.exr/.hdr`，Stage D 从该 EXR 重建场景并校验 env tensor 不变。验证默认每个 seed 64 SPP，最终 raw 渲染独立使用 256 SPP；OptiX 结果仅作为单独预览，不能覆盖 raw EXR 或参与指标。

## 其他主要入口

| 文件或目录 | 用途 |
| --- | --- |
| `inverse_img_w_mi_ori.py` | 原始 Materialist 逆渲染流程 |
| `inverse_img_w_mi_geocalib.py` | 使用 GeoCalib/FOV 内参的逆渲染流程 |
| `inverse_img_w_mi_moge2.py` | 使用 MoGe2 相机与几何的完整逆渲染流程 |
| `render_matnet_pre_with_hdri.py` | 使用指定 HDRI 渲染 MatNet 预测结果 |
| `render_final.py` | 渲染优化后的最终材质和环境光结果 |
| `flux2_opt.py` | 使用 PBR color 和 aligned normal 进行最终照片真实化后处理 |
| `mat_edit.py` / `trans_edit.py` | 材质与近似透明效果编辑 |
| `optimal_lights/` | 半球离散灯光与材质联合优化实验 |
| `scripts/` | Hybrid 渲染、消融实验、环境光推理和可视化工具 |

## 主要输入与输出

Materialist run 目录通常为 `output_imgs/<save_name>/`，包括：

- `camera_meta.json`：工作尺寸、K、FOV、主点和相机姿态。
- `mesh_moge2.ply` 或 `<save_name>.ply`：单视图重建 mesh。
- `albedoPred.exr`、`roughnessPred.exr`、`metallicPred.exr`、`normalPred.exr`：初始材质。
- `best_results/`：经过逆渲染优化的材质、envmap 和重建图像。

FLUX.2 后处理当前默认读取 `output_imgs/indoorlight_example1_ile2/best_results/rendered_img_hq.exr` 和同目录下的 `normal.exr`，并生成：

- `flux2_photoreal_with_normal.png`：最终 1024×768 照片真实化结果。
- `flux2_photoreal_with_normal_render_reference.png`：经过曝光、ACES 和 sRGB 转换的 PBR 主参考。
- `flux2_photoreal_with_normal_normal_reference.png`：解码、单位化后的 RGB normal 辅助参考。
- `flux2_photoreal_with_normal.json`：模型、绝对路径、两张参考图职责和尺寸、prompt、seed、采样参数、曝光、normal 统计、版本及耗时。

ILE bridge 输入包括：

- `light_predictions.json`。
- `visLampPred_*.obj`、`invLampPred_*.obj`、`visWinPred_*.obj`、`invWinPred_*.obj`。
- visible/invisible lamp 的中心和 RGB intensity。
- visible/invisible window 的 center、OBJ，以及 `src/srcSky/srcGrd` 三组 `[RGB, direction xyz, concentration]`。
- 可选的 lamp/window mask、相机和 depth scale 元数据。

Hybrid 输出默认位于 `<materialist-dir>/hybrid_ile_render/`：

- `lights_projected.png`：灯光投影和 mask 对齐诊断。
- `render_<mode>_raw.exr`：未经去噪的线性渲染结果。
- `render_<mode>.exr` / `.png`：OptiX 去噪结果。
- `render_manifest.json`：mesh、材质、灯光、尺度和渲染参数记录。

## 环境与当前边界

项目主要面向 Python 3.10、PyTorch、CUDA、Mitsuba 3、Dr.Jit、Open3D、OpenCV 和 NumPy。FLUX.2 可选阶段额外使用 Diffusers、Transformers、Accelerate 和 Safetensors。MatNet 权重通过 Hugging Face 获取；FLUX.2、MoGe2、GeoCalib 和 IndoorLightEditing 使用本地源码或权重。

当前固定灯光优化入口已经接入全部 lamp/window，并保留 window 的 sun/sky/ground 方向性表示；Stage B 逐灯强度入口和通用渲染入口仍保持 lamp-only，以避免把方向窗口错误压成单一 RGB scale。ILE intensity 仍是 Mitsuba radiance 的相对初值，window emitter 目前也采用均匀孔径位置采样而非按高集中度 sun lobe 做方向 importance sampling，因此尖锐阳光分量可能需要较高 SPP。单视图 mesh 仍无法完整表达画面外房间几何。首次使用新的 Python emitter 时，Dr.Jit/CUDA 需要进行一次 JIT 编译，后续运行可复用缓存。

FLUX.2 输出属于生成式、display-referred 的最终视觉结果，不保证逐像素保持 radiance、BRDF 或几何，也不应作为 HDR、材质分解和 relighting 的权威输出。研究评估中应保留原始 PBR EXR，并把 FLUX 结果单独用于照片真实度、结构保持和主观评价。当前 Base 4B Diffusers 权重目录约为 15 GB；24 GB RTX 5090 可完成 BF16 整模型推理，较小显存设备需要 CPU offload。
