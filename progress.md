# Materialist 当前实施进展

更新时间：2026-08-14

当前阶段：相机、MoGe2 metric geometry、MatNet 材质和 IndoorLightEditing 显式灯光已经接入同一 Mitsuba 场景。固定灯光入口现会加载 visible/invisible lamp 与 visible/invisible window：lamp 使用普通 mesh area emitter，window 使用保留 ILE sun/sky/ground 三球面高斯的方向性有限面 emitter。Stage B 逐灯 scale 仍保持 lamp-only；Stage C 32×16 far-field HDRI 和“关闭 Stage B，固定全部 ILE 灯光后优化 ARM”的主流程均已实现。固定灯光优化现已补齐 MoGe2 normal、window null BSDF/无 offset、PRB、masked LDR loss、周期多种子 validation、独立 256 SPP raw final render，以及更稳健的 Adam 配置。逐像素与 PosMLP 两种参数化均已在 Example1 跑通。在物理渲染之后，项目还提供 FLUX.2 Klein Base 4B 双参考照片真实化入口。下一步应进行新的完整 256 SPP 正式优化、跨场景消融和 FLUX 后处理保真度评估。

## 当前数据链路

```text
输入 RGB
  ├── MoGe2 → K / metric depth / normal / mask / mesh
  ├── MatNet → albedo / roughness / metallic / normal
  └── IndoorLightEditing
        ← 同一份 MoGe depth.npy
        → visible/invisible lamp mesh + center + RGB
        → visible/invisible window mesh + center
        → window sun/sky/ground [RGB, direction, concentration]

Materialist mesh + Materialist BRDF + ILE local lights
  → lamp: Mitsuba area emitter
  → window: ILE directional SG mesh emitter
  → local / env / combined PBR rendering

PBR rendered EXR + aligned signed normal EXR
  → color: auto exposure + ACES + linear-to-sRGB
  → normal: [-1,1] decode + unit normalization + RGB encoding
  → FLUX.2 Klein Base 4B multi-reference editing
  → photograph-like PNG + processed references + inference JSON
```

## 模块状态

| 模块 | 状态 | 当前情况 |
| --- | --- | --- |
| MatNet 材质预测 | 已完成重构 | 仓库内统一使用一套 `infer_image_scaled()`；scale、宽高比、patch padding 和输出尺寸语义明确 |
| MoGe2 相机与几何 | 已集成 | 输出 metric depth、normal、points、有效 mask 和相机 K |
| MoGe 无效值处理 | 已完成 | mask 外以及 `NaN/Inf/非正` 深度不会进入 mesh；depth/points/normal resize 均使用 mask-aware 插值 |
| ILE depth 输入 | 已完成 | 输出原图尺寸、稠密、有限、正值的 `float32 depth.npy`，与 mesh 使用相同的 MoGe depth source |
| Mesh 重建 | 已集成 | 使用工作分辨率 K 和 metric depth，每次推理强制重建；0-depth/mask sentinel 已修复 |
| 相机元数据 | 已集成 | `camera_meta.json` 由 mesh、Mitsuba sensor 和 Materialist BSDF 共用，并正确记录 MoGe2/GeoCalib 来源 |
| ILE 灯光 JSON/OBJ | 已接入 | 解析 visible/invisible lamp 与 window 的 center、geometry、mask、radiance 参数和可选 depth scale |
| ILE → Mitsuba 灯光 | 已跑通 | lamp OBJ 使用普通 area emitter；window OBJ 使用 sun/sky/ground 三 SG 方向 emitter |
| Hybrid 渲染入口 | 已实现 | 通用入口保持 lamp-only，支持 `local`、`env`、`combined`、dry-run 和 visible/invisible 消融 |
| Hybrid 调试输出 | 已实现 | 输出灯光投影、mask overlay、raw EXR、OptiX 去噪图和 manifest |
| HDRI / 原始 envmap 流程 | 已有 | 保留 Materialist 原环境光优化、HDRI 重渲染和旋转能力 |
| 灯光强度 refinement | 主流程已实现 | 固定几何/材质/灯几何/色度，每盏灯只优化 scalar radiance scale |
| 32×16 far-field HDRI | 主流程已实现 | 固定 Stage B 局部灯，优化非负低频 envmap，并加入能量和球面 TV 正则 |
| 固定灯光下的 ARM refinement | 主流程已实现 | 跳过 Stage B，固定全部 lamp/window，先优化 HDRI，再固定全部光照优化 albedo/roughness/metallic |
| 固定灯光优化稳定性 | 已实现并 smoke 验证 | MoGe2 normal、window null BSDF、PRB depth 4、masked clipped-sRGB loss、固定多种子 validation、独立 raw final render |
| FLUX.2 照片真实化后处理 | 已实现并端到端验证 | PBR color 为 1024×768 主参考，normal 为 512×384 辅助参考；保持光照、阴影、亮度和色感，只恢复照片细节 |
| FLUX.2 模型与复现记录 | 已完成 | Base 4B Diffusers 权重位于固定 checkpoint 目录；输出 prompt、seed、曝光、normal 解码、依赖版本和耗时 JSON |
| Local + far-field 联合 refinement | 待实现 | 当前采用分阶段冻结策略，尚未进行最后的小步联合优化 |
| 自动化测试 | 初步建立 | 已有 26 个颜色、HDR、坐标、mask、mesh、尺度、JSON、window SG、显示域裁剪与 validation gate 回归测试 |

## 已完成实现

### 1. MoGe2 安全几何链路

- `myutils/moge2_utils.py` 集中处理模型加载、推理和深度准备。
- MoGe2 采用 `apply_mask=False`，有效性由独立 mask 管理，避免默认的 `inf` 深度污染下游。
- mesh depth 的无效区域保持为 0；缩放时无效值不会参与有效深度插值。
- `depth_file_to_mesh()` 的 `-1` sentinel 初始化错误已修复，0-depth/mask 像素不会再从 `(1,1)` 错误继承深度。
- MoGe2 depth、points 和 normal 在导出前均进行有限值检查；points/normal 同时输出原始 camera-space 与经过 `Rx(180°)` 的 Materialist-space 版本。

### 2. MatNet 工作分辨率统一

- 移除了动态挂载的第二套缩放推理实现。
- `scale` 会真实控制 MatNet 工作尺寸，而不只是后处理输出大小。
- 网络输入自动 padding 到 ViT patch size，输出裁回工作尺寸。
- normal 输出重新单位化，相机 K 使用相同的 pixel transform 映射。
- MatNet 明确使用线性 RGB；8/16-bit 普通图像先按 sRGB 解码，float HDR 保留真实辐射值，不再根据最大值静默除以 255。
- GT、材质、MoGe maps、K 与 mesh 全部使用同一 work resolution；albedo PNG 使用正确显示 gamma，材质 EXR 写出前执行物理范围约束。

### 3. IndoorLightEditing 深度适配

- `test_matnet_infer_moge2.py` 在完整推理和 mesh 重建成功后输出 ILE 使用的 `depth.npy`。
- 输出严格为原图 `(H, W)`、`float32`、有限正值 metric depth。
- MoGe 无效像素使用最近有效深度补齐，避免 ILE 反投影出现零深度几何。
- ILE 和 Materialist mesh 当前使用同一份 MoGe metric depth，减少尺度及结构漂移。
- 输出 `moge2_valid_mask.*`、`mesh_valid_mask.png`、`mesh_depth.exr` 和 `inference_manifest.json`，防止不同运行版本的 GT、K、depth 与 mesh 静默混用。

### 4. Hybrid light bridge

- 新增 `hybrid_light/`，分离数据类型、JSON 读取、坐标转换、Mitsuba scene 构建和可视化。
- 直接复用 ILE 导出的 visible/invisible lamp OBJ，比重新估计 rectangle 更忠实于当前预测结果。
- 固定灯光入口同时复用 visible/invisible window OBJ，并完整读取 `src`、`srcSky`、`srcGrd` 三组 `[RGB, direction xyz, concentration]`。
- 新增 `hybrid_light/ile_window_emitter.py`。窗口辐射度按 `Σ RGB_k exp(λ_k(min(dot(d_k,ω)-1,0)))` 求和，保留 sun/sky/ground 的方向性，不简化为 uniform rectangle。
- 当前坐标统一为 `+X right、+Y up、-Z forward`；转换函数即使当前为 identity 也保留独立接口。
- 如果 JSON 标记 depth normalization，center 和整个 light mesh 会同步乘 `depth_scale`。
- visible lamp 默认朝相机平移 0.005 m，降低与单视图 scene mesh 共面的风险。
- ILE lamp RGB 与 window 三组 RGB 都作为相对 radiance 初值，并由同一个全局 `--radiance-scale` 校准；window direction 和 concentration 不随该 scale 改变。

### 5. Hybrid renderer

- `scripts/render_ile_lights.py` 自动查找 Materialist mesh、camera 和初始/优化材质。
- 支持 local-only、env-only 和 combined scene。
- 支持 visible-only / invisible-only 消融和不加载 CUDA 的 dry-run。
- raw linear EXR 始终保留；默认额外输出 OptiX 去噪 EXR/PNG。
- `render_manifest.json` 记录所有输入路径、灯光、scale、offset、spp 和输出路径。

### 6. Stage B + Stage C 优化入口

- 新增 `scripts/optimize_ile_farfield.py`，不修改原始渲染入口。
- Stage B 的每盏灯满足 `optimized_rgb = ILE_RGB × scalar_scale`，不会自由漂移色度。
- Stage B loss 使用显示空间 Charbonnier、少量 MSE 和 `log(scale)` 先验。
- Stage C 固定优化后的 local lamps，只优化 `(16, 32, 3)` far-field HDR tensor。
- Stage C 使用非负/最大值约束、mean-square energy 正则和横向周期 TV 正则。
- 适配 Mitsuba envmap 自动增加周期接缝列的行为：内部为 16×33，导出仍严格为 16×32。
- 保存逐灯 scale、两阶段 history、初始/优化渲染、EXR/HDR envmap 和总 manifest。

### 7. 关闭 Stage B 的 HDRI + ARM 优化入口

- 新增 `scripts/optimize_ile_farfield_materials.py`，与已有优化入口相互独立。
- 该入口调用 `load_ile_lights(..., include_windows=True)`，自动加载 visible/invisible lamp 和 visible/invisible window，无需增加命令行参数。
- Lamp RGB 与 window sun/sky/ground RGB 只乘固定的全局 `--radiance-scale`，整个实验不创建逐灯或逐窗口优化器。
- Stage C 固定全部 ILE lamp/window 和输入材质，只优化 32×16 far-field HDRI。
- Stage D 冻结 lamp、window 和优化后的 HDRI，只优化当前阶段选定的 ARM 通道；参数可以是逐像素张量或 PosMLP 权重。
- 新增 `--model_name {none,pos_mlp}`：默认 `none` 保留现有逐像素优化；`pos_mlp` 同时用于 HDRI 和材质优化。
- PosMLP HDRI 使用球面坐标，ARM 使用图像 UV 坐标；loss、约束、阶段顺序、checkpoint 和最终输出格式与逐像素分支一致。
- PosMLP HDRI 输出层会匹配 `--farfield-init`，并通过 PyTorch ↔ Mitsuba/Dr.Jit 可微桥回传渲染梯度；该模式需要 CUDA。
- 显式 `--target` 优先；否则新版完整 inference manifest 使用同工作分辨率的线性 `gt_image.exr`，旧结果回退 ILE JSON 原始图；`target.png` 只用于预览，不参与 loss。
- 材质优化参考 `inverse_img_w_mi_ori.py`：显示空间自适应 MSE + L1、相对输入材质的 L1 prior、StepLR 等价调度和相对改进式 early stopping；逐像素使用 Adam，PosMLP 使用 AdamW。
- detached mean-exposure matching 改为显式 opt-in，默认关闭，以保证训练目标和最终物理渲染亮度一致。
- Stage C 训练候选与最终选中 HDRI 分开保存；只有通过高 SPP gate 的候选才成为权威 HDRI，否则恢复初始 HDRI。最终选中 HDRI 会被重新加载并固定用于 ARM 阶段。
- Stage C 与 ARM 都使用固定多 seed、高 SPP validation gate。若候选 HDRI 的 display MSE 劣于初始 HDRI，则恢复初始 HDRI作为权威 EXR 和 Stage D 输入，同时保留 `farfield_candidate_combined.*` 供诊断；manifest 还记录 HDRI 重载误差和材质优化期间的冻结误差，两者必须为 0。
- 默认材质顺序改为 `rm -> a`：先联合优化 roughness/metallic 并恢复该阶段最优 checkpoint，再冻结 RM、单独优化 albedo；每阶段使用独立 optimizer、学习率调度、early stopping 和输入材质 prior。
- `material_phase_summaries.json` 记录阶段顺序、最优 MSE 与冻结通道误差；实测 RM 阶段的 albedo 和 A 阶段的 roughness/metallic 最大变化均为 0。仍可通过 `--material-order` 做顺序消融，旧 `--material-channels` 仅作为单阶段兼容选项。Normal 固定且默认来自 MoGe2，不优化 normal map。
- `optimization_manifest.json` 新增 `fixed_local_lights`、`fixed_window_count` 和 `fixed_local_radiance_scale`，可以确认正式运行是否确实包含全部 ILE 灯光。
- 默认固定 MoGe2 normal；只有 `--use-mesh-normal` 才切换到 PLY 插值 normal。Normal 始终冻结，不纳入优化变量。
- Window 显式使用 Mitsuba `null` BSDF，且不再应用 visible offset；只有 visible lamp 保留 0.005 m 共面规避偏移，投影诊断与实际 scene 使用同一规则。
- 默认使用 `prb`、`max_depth=4`；直接远场和材质优化的 Mitsuba Adam 启用 AMSGrad 与 masked updates，PosMLP 的 Adam/AdamW 显式设置 `weight_decay=0`。
- LDR 目标使用先裁剪线性值到 `[0,1]`、再做标准 sRGB transfer 的 loss；可见 lamp/window mask、深度断层、有效 mesh 边界及其膨胀邻域不参与 Stage C/D loss。
- 默认每 25 次以两个固定 seed、每 seed 64 SPP 做 validation；checkpoint 和 early stopping 使用 masked validation display MSE，而不是单次训练噪声。
- 验证与最终渲染预算分离。最终默认使用独立 seed、256 SPP raw render；`rendered_img.exr` 和 `rendered_img_final_raw.exr` 保持未经去噪，OptiX 仅输出单独的 denoised preview。
- 输出优化 HDRI、ARM EXR/PNG、两阶段 history、loss masks、raw/denoised 分离渲染和 schema v6 manifest。

### 8. FLUX.2 双参考照片真实化后处理

- 新增 `flux2_opt.py`，作为 Mitsuba PBR 渲染后的可选可视化阶段；该入口不回写或重新估计 mesh、BRDF、灯光和 HDRI。
- 使用 `black-forest-labs/FLUX.2-klein-base-4B` 的 Diffusers 多参考编辑接口。模型按组件下载到 `/home/majortom/project/datasets/ckpt/FLUX.2-klein-base-4B`，支持断点续传、分片完整性检查和 `--offline` 本地推理。
- 第一参考图默认为 `output_imgs/indoorlight_example1_ile2/best_results/rendered_img_hq.exr`。它是线性 HDR PBR 渲染，先清理 `NaN/Inf/负辐射值`，以亮度中位数进行自动曝光，再经过 ACES tone mapping 和 linear RGB → sRGB 转换。
- 第二参考图默认为同目录下的 `normal.exr`。该输入使用独立的数据图解码，不能走颜色 EXR 的负值截断和 tone mapping；支持自动判断 `[-1,1]` signed 或 `[0,1]` unsigned 编码，之后逐像素重新单位化并映射到 RGB normal-map 可视化。
- 两张源 EXR 必须像素对齐且具有相同原始宽高，否则立即报错，避免把错误法线约束施加到彩色渲染。
- PBR color 默认放大为 1024×768，并决定输出宽高；normal 默认使用 512×384。Klein 当前没有显式的 per-reference weight，减少 normal token 数可以保留几何边界，同时降低伪彩色法线对材质、光照和场景语义的干扰。
- 调用顺序固定为 `image=[render_reference, normal_reference]`。简化后的默认 prompt 仅要求补足真实材质纹理、细微表面瑕疵、自然 roughness、边缘、接触细节和轻微相机质感，并要求构图、相机、物体、光照、阴影、亮度、色彩与整体色感保持不变。
- 默认采用 BF16、50 steps、guidance scale 4.0、seed 2026。`--memory-mode auto` 会根据空闲显存选择整模型 CUDA 或 Accelerate CPU offload；24 GB RTX 5090 当前走完整 CUDA 路径。
- 每次推理保存最终 PNG、PBR 显示参考、normal RGB 参考和 JSON。JSON 记录绝对输入/模型/输出路径、两张参考图职责与尺寸、prompt、采样参数、EXR 曝光、normal 范围与单位长度统计、软件版本和推理时间。
- FLUX 输出是 display-referred 的生成式照片可视化，不是物理辐射结果。正式 relighting、材质评估、HDR 指标和 PBR 消融仍必须使用原始线性 EXR。

默认离线运行命令为：

```bash
/home/majortom/miniconda3/envs/materialist5090/bin/python \
  /home/majortom/project/Materialist/flux2_opt.py --offline
```

批量实验时可分别覆盖 `--input`、`--normal` 和 `--output`；使用 `--normal-encoding` 控制法线编码，使用 `--long-edge` 控制主参考/输出分辨率，使用 `--normal-long-edge` 控制 normal 辅助强度，使用 `--steps`、`--guidance-scale` 和 `--seed` 控制采样。

## 已完成验证

### 数值与接口验证

- 本次新增优化入口已通过 `py_compile`；本次维护的入口与文档已通过定向 `git diff --check`。
- 颜色、MoGe、mesh、Hybrid、window SG、validation gate 与 PosMLP 数值稳定性的 26 个单元测试全部通过，其中包括：
  - 相机中心投影；
  - `+X` 投向图像右侧；
  - `+Y` 投向图像上方；
  - 3D→2D→3D round-trip；
  - depth scale、lamp/window JSON 读取和三组 SG 参数校验。
- 新增 window 不偏移且使用 null BSDF、HDR 值在 LDR loss 中裁剪、masked metric、深度断层两侧与 valid/invalid mesh 边界测试。
- 8-bit/16-bit PNG sRGB→linear→PNG 往返、float HDR 不除以 255、非透明 alpha 拒绝策略均已验证。
- MoGe normal/points 的 `Rx(180°)` 坐标变换、mask-aware resize 和 normal 单位化均已验证。
- 0-depth synthetic case 已验证不会被 mesh 三角形引用。
- MoGe depth 已验证 `NaN/Inf/非正值/mask` 清理及不同尺寸下的传播。
- MatNet 已验证 `scale=1`、`scale=0.5`、输出尺寸、normal 长度和 K 缩放。

### Example1 端到端验证

- Materialist 相机水平 FOV：约 58.32°。
- ILE 当前假设水平 FOV：57.95°。
- visible lamp center 投影与 lamp mask 中心误差：约 0.63 px。
- 当前 ILE 未对 MoGe depth 进行 normalization，因此 geometry scale 为 1。
- Example1 已于 2026-08-10 全量重建：`gt_image.exr` 与原图标准 linear RGB 完全一致，`gt_image.png` 与原 PNG 逐像素一致。
- 重建后的 albedo/roughness/metallic 全部有限并处于合法范围；MoGe camera/materialist vector 变换误差为 0，mesh 重投影最大误差小于 `1e-12 px`。
- 修正后的 MoGe predicted normal 已通过独立 2 spp Mitsuba smoke render：输出全有限、非黑屏，约 53.5% 像素在该低采样下获得非零 local-light contribution。
- Lamp-only 入口成功加载 1 个 visible lamp 和 1 个 invisible lamp；固定灯光优化入口成功加载这两个 lamp，以及 1 个 visible window 和 1 个 invisible window，共 4 个 local lights。
- local-only、visible-only、invisible-only 均得到有效非空照明。
- 真实 Example1 window 参数已验证为三组有限 `(3, 7)` SG；方向向量在 loader 中归一化，负 radiance、零方向或负 concentration 会被拒绝。
- 16×16、16 spp 的独立消融中，visible window、invisible window 和两者组合均得到有限非黑结果；非零像素比例分别约为 48.8%、97.7% 和 94.9%。
- 使用真实四灯场景的 LLVM AD 与 `cuda_ad_rgb` 轻量 smoke test 均得到全有限渲染，并对材质参数产生有限、非零梯度，说明 window emitter 没有阻断 Stage C/D 可微链路。
- 64 spp local-only raw EXR 全部为有限值，约 98.6% 像素获得非零贡献。
- 已生成 OptiX 去噪后的可视化结果。
- 2-step Stage B smoke test 成功，visible/invisible scale 均收到梯度并更新。
- 2-step Stage C smoke test 成功，loss 由约 0.3861 降至 0.3199。
- smoke test 导出的 far-field EXR/HDR 均为 `(16, 32, 3)`、全有限值。
- 固定灯光的顺序优化 smoke test 成功：2-step Stage C 后按 2-step RM、2-step A 执行，两个材质阶段的冻结通道误差、HDRI 重载误差及材质阶段 HDRI 变化量均为 0。
- PosMLP 2-step smoke test 成功：far-field display MSE 从约 `0.166886` 降至 `0.164928`，RM 网络产生有效更新；导出的 HDRI/ARM/render 全有限，材质阶段 HDRI 变化量为 0。
- Torch sRGB 在零亮度处的梯度发散已修复，并增加有限梯度回归测试。
- 已确认旧 `hybrid_ile_farfield_material_opt/` 使用了错误的双重 gamma target，不能作为有效结果基线。
- 修正版 `target.png` 与 ILE 原始 `im.png` 像素级一致；正式 v2 实验关闭曝光对齐并通过 validation gate。
- Stage C 直接 32×16 tensor 在高 SPP validation 上仍可能退化；v6 保留该逐像素基线并新增 PosMLP 平滑参数化，两者都会把训练选出的唯一 best HDRI 交给顺序材质阶段并保留 metrics，后续需通过正式实验比较并继续降低方差。
- 重建优先 ARM 设置将 validation PSNR 从约 13.797 dB 提升到 13.954 dB，三种材质平均变化约 0.010–0.016。
- 使用指定 `BRDFLight_size0.200_.../light_predictions.json` 的 320×240 Example1，direct 与 PosMLP 分支都已完成 `prb`、`max_depth=4` 的端到端最小 smoke test；两种模式均完成 Stage C、Stage D、checkpoint 恢复、最终 raw 输出和 schema v6 manifest。
- Example1 的 emitter/depth-edge mask 默认保留约 87.8% 像素参与 loss；该数字会随场景 mask、depth 和膨胀半径变化。

### FLUX.2 Example1 ILE2 双参考验证

- `rendered_img_hq.exr` 与 `normal.exr` 均为 320×240 且逐像素对齐；normal 三通道范围约为 `[-0.99984, 0.999999]`，没有非有限值或零向量，解码前向量长度的 1%、50%、99% 分位数均为 1。
- PBR color 自动曝光结果为约 `-0.696 EV`，经 ACES 和 sRGB 转换后作为 1024×768 第一参考；normal 映射后作为 512×384 第二参考。
- 本机已安装 Torch 2.11.0+cu128、Diffusers 0.39.0、Transformers 5.15.0、Accelerate 1.14.0 和 Safetensors 0.8.0；模型组件完整性和离线加载均已验证。
- RTX 5090 D v2 上以 BF16、50 steps、guidance 4.0、seed 2026 完成简化 prompt 的双参考推理，最终生成阶段耗时约 41.60 秒，输出为 1024×768 RGB PNG。
- 功能开发阶段已在先前的 classroom 原型上对比 normal 使用 1024×768 和 512×384 两种辅助分辨率。等分辨率 normal 容易过度影响场景语义；512×384 对几何轮廓仍有帮助且更接近 PBR 主参考，因此设为当前默认值。
- 最终结果没有出现 normal 伪彩色、拼图或多图输出；材质微观纹理、边缘和接触关系获得照片化细化。默认 prompt 已进一步缩短，避免模型重新设计已经基本正确的 PBR 场景。
- `flux2_opt.py` 已通过语法、参数、EXR/normal 预处理、对齐检查和 `git diff --check` 验证。

Hybrid 当前结果目录：

```text
output_imgs/indoorlightediting_test/hybrid_ile_render/
├── lights_projected.png
├── render_local_raw.exr
├── render_local.exr
├── render_local.png
├── render_manifest.json
├── visible_only/
└── invisible_only/
```

FLUX.2 Example1 ILE2 当前结果目录：

```text
output_imgs/indoorlight_example1_ile2/best_results/
├── rendered_img_hq.exr
├── normal.exr
├── flux2_photoreal_with_normal.png
├── flux2_photoreal_with_normal_render_reference.png
├── flux2_photoreal_with_normal_normal_reference.png
└── flux2_photoreal_with_normal.json
```

## 后续目标

### Stage B：逐灯强度优化（主流程完成，待正式实验）

目标：固定 mesh、材质、灯位置和灯几何，只优化每盏灯的标量强度。

```text
radiance_i = exp(alpha_i) * ile_rgb_i
```

后续工作：

1. 使用 200 次以上正式迭代检查 loss 收敛和 scale 稳定性。
2. 调节 Charbonnier/MSE/scale prior 权重与学习率。
3. 增加灯光区域、饱和区域和有效 mesh 区域的 loss mask 消融。
4. 绘制每盏灯 scale 曲线，并比较优化前后 local-only 重建误差。

完成标准：优化后 reconstruction loss 稳定下降，且 visible/invisible lamp 不出现明显退化。

### Stage C：32×16 far-field HDRI（主流程完成，待正式实验）

目标：固定 Stage B local lamps，用受约束低频 HDRI 补足窗外、画外和远场 residual illumination。

计划实现：

- 完成 300 次以上正式优化并调节初始化、学习率、TV 和 energy 权重。
- 在相同 seed、迭代数和 loss 下对比 `--model_name none` 与 `pos_mlp` 的收敛、平滑性及高 SPP validation。
- 分别输出 local-only、farfield-only 和 combined contribution。
- 检查 HDRI 是否出现高频热点、边界断裂或吞掉 local lamps。
- 比较 constant ambient 与 32×16 HDRI，判断是否需要保留更简单的中间阶段。

完成标准：暗部得到补偿，同时灯罩高光和局部阴影仍主要由显式灯产生。

### Stage D：固定光照后的 RM → A 材质优化（主流程完成，待正式实验）

目标：关闭逐灯强度优化，在 Stage C HDRI 收敛后冻结全部光照；先联合优化 roughness/metallic，冻结其 best checkpoint，再单独优化 albedo。

后续工作：

- 使用 500 次以上正式迭代，检查 best checkpoint 和各材质通道的变化。
- 对比逐像素与 PosMLP 的材质细节保持、空间平滑先验和最终重建误差。
- 调节 `material-prior-weight`，并比较开启/关闭曝光对齐。
- 分别执行 `r`、`m`、`rm`、`rm -> a` 和单阶段 `arm` 消融，观察顺序及光照—材质歧义。
- 增加材质边缘保持或低频正则前，先评估原始 Materialist loss 的基线。

完成标准：重建误差稳定下降，材质仍保持可编辑性，不通过极端 albedo/metallic 吞掉照明误差。

### Stage E：Local + far-field 小步联合 refinement

目标：在分阶段结果稳定后，同时小步调整 local scale 与 far-field HDRI。

计划实现：

- 使用比单阶段更低的学习率。
- 保留 local scale 的 Stage B prior，限制 far-field 总能量。
- 监控每盏 local lamp 的贡献，防止联合优化将其压到 0。
- 与当前“Stage B 冻结后只优化 HDRI”的结果进行对照。

完成标准：envmap 改善整体颜色和低频照明，但不会替代显式近场灯。

### Stage F：数据接口和工程化

1. 在 ILE exporter 中正式写入图像尺寸、K/FOV、坐标系、`depth_normalized` 和 `depth_scale`。
2. 修正 Materialist `camera_meta.json` 中仍偏向 GeoCalib 的 schema/source 命名。
3. 为 mesh 增加图像、depth、K、scale 和生成版本记录，防止复用过期 mesh。
4. 将 MoGe → ILE → light export → Materialist render 串成一个可复现入口。
5. 将可选的 FLUX.2 双参考后处理接入统一入口，但保持它与权威线性 PBR 输出分离。
6. 增加多场景测试、性能统计和自动结果对比。

### 后续研究扩展

- 对高 concentration sun lobe 增加方向 importance sampling，降低当前均匀窗口位置采样的方差。
- 增加 lamp-only、window-only、lamp+window、lamp+window+far-field 的正式高 SPP 消融。
- 研究是否需要在 Stage B 中为 window 三组 SG 增加共享或分组 radiance scale；当前固定灯光入口不优化 window 参数。
- light position、orientation 和 size 的有限范围优化。
- 更可靠的 visible lamp 双面/定向发光表示。
- 多视角或补全几何，以改善画面外灯光的遮挡和多次反射。
- 材质—灯光交替 refinement 和物体插入评估。
- 在多场景上比较 PBR-only 与 PBR+normal FLUX 结果，统计 LPIPS/结构边缘保持和人工照片真实度评分。
- 消融 `normal-long-edge={256,512,768,1024}`、seed、guidance 和 prompt 长度，测量 normal 几何约束与主图外观保真之间的平衡。

## 已知限制与风险

- ILE RGB intensity 与 Mitsuba radiance 尚未建立物理单位映射。
- 固定灯光入口已支持 window，但 Stage B 和通用 renderer 仍默认 lamp-only；不能把 window 三组方向参数直接压成 uniform RGB rectangle。
- Lamp area emitter 与 window emitter 都按 OBJ winding 单面发光；当前样例通过消融验证，但新场景仍需检查法线朝向。
- Window emitter 保留 SG 辐射度，但目前仍均匀采样窗口孔径，没有专门 importance sample 很尖锐的 sun lobe，因此高 concentration 窗口可能需要更多 SPP。
- 可见灯 mesh 可能与 Materialist 单视图 mesh 接近共面，目前只通过小幅 offset 缓解。
- 单视图 mesh 不是封闭房间，画面外 invisible lamp 的遮挡和间接光只能近似表达。
- 即使加入显式 window，local-only 仍不包含完整画外远场和 neural indirect illumination，因此整体亮度低于输入并不一定是坐标错误。
- 主渲染和自定义 Materialist BSDF 仍依赖 CUDA；部分路径与模型权重依赖本机环境。
- 单图逆渲染存在材质、几何、曝光和光照间的固有歧义。
- FLUX.2 是生成式后处理，无法保证逐像素保持辐射度、BRDF 或几何，因此其 PNG 不能替代线性 EXR 做物理指标和编辑一致性评估。
- Klein 多参考接口没有显式 per-reference weight；当前用较低 normal 分辨率间接控制影响。若 PBR 与 normal 不对齐或 normal 本身缺失结构，模型仍可能修改场景语义。
- Base 4B 权重目录约占 15 GB，完整 BF16 CUDA 推理需要较多显存；显存不足时会切换 CPU offload，但运行时间会明显增加。
