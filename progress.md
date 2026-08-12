# Materialist 当前实施进展

更新时间：2026-08-11

当前阶段：相机、MoGe2 metric geometry、MatNet 材质和 IndoorLightEditing 显式灯光已经接入同一 Mitsuba 场景。Example1 已完成 local-only、visible-only 和 invisible-only 渲染；Stage B 逐灯 scale、Stage C 32×16 far-field HDRI，以及“关闭 Stage B，HDRI 后优化 ARM 材质”的分阶段主流程均已实现。固定灯光入口现同时支持逐像素和 PosMLP 两种参数化，并已通过最小 GPU smoke test。下一步进行正式迭代、超参数调节和两种参数化的结果评估。

## 当前数据链路

```text
输入 RGB
  ├── MoGe2 → K / metric depth / normal / mask / mesh
  ├── MatNet → albedo / roughness / metallic / normal
  └── IndoorLightEditing
        ← 同一份 MoGe depth.npy
        → visible/invisible lamp mesh + center + RGB

Materialist mesh + Materialist BRDF + ILE lamps
  → Mitsuba area emitters
  → local / env / combined PBR rendering
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
| ILE 灯光 JSON/OBJ | 已接入 | 解析 visible/invisible lamp center、RGB、geometry 和可选 depth scale |
| ILE → Mitsuba 灯光 | Prototype 已跑通 | lamp OBJ 作为 Mitsuba area emitter 加载；window 暂不接入 |
| Hybrid 渲染入口 | 已实现 | 支持 `local`、`env`、`combined`、dry-run 和 visible/invisible 消融 |
| Hybrid 调试输出 | 已实现 | 输出灯光投影、mask overlay、raw EXR、OptiX 去噪图和 manifest |
| HDRI / 原始 envmap 流程 | 已有 | 保留 Materialist 原环境光优化、HDRI 重渲染和旋转能力 |
| 灯光强度 refinement | 主流程已实现 | 固定几何/材质/灯几何/色度，每盏灯只优化 scalar radiance scale |
| 32×16 far-field HDRI | 主流程已实现 | 固定 Stage B 局部灯，优化非负低频 envmap，并加入能量和球面 TV 正则 |
| 固定灯光下的 ARM refinement | 主流程已实现 | 可跳过 Stage B，先优化 HDRI，再固定全部光照优化输入 albedo/roughness/metallic |
| Local + far-field 联合 refinement | 待实现 | 当前采用分阶段冻结策略，尚未进行最后的小步联合优化 |
| 自动化测试 | 初步建立 | 已有 14 个颜色、HDR、坐标、mask、mesh、尺度和 JSON 回归测试 |

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
- 当前坐标统一为 `+X right、+Y up、-Z forward`；转换函数即使当前为 identity 也保留独立接口。
- 如果 JSON 标记 depth normalization，center 和整个 light mesh 会同步乘 `depth_scale`。
- visible lamp 默认朝相机平移 0.005 m，降低与单视图 scene mesh 共面的风险。
- ILE RGB 作为相对 radiance 初值，并提供全局 `--radiance-scale` 校准入口。

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
- ILE lamp RGB 只乘固定的全局 `--radiance-scale`，整个实验不创建逐灯优化器。
- Stage C 先固定灯光和输入材质，只优化 32×16 far-field HDRI。
- Stage D 冻结局部灯和优化后的 HDRI，只优化当前阶段选定的 ARM 通道；参数可以是逐像素张量或 PosMLP 权重。
- 新增 `--model_name {none,pos_mlp}`：默认 `none` 保留现有逐像素优化；`pos_mlp` 同时用于 HDRI 和材质优化。
- PosMLP HDRI 使用球面坐标，ARM 使用图像 UV 坐标；loss、约束、阶段顺序、checkpoint 和最终输出格式与逐像素分支一致。
- PosMLP HDRI 输出层会匹配 `--farfield-init`，并通过 PyTorch ↔ Mitsuba/Dr.Jit 可微桥回传渲染梯度；该模式需要 CUDA。
- 显式 `--target` 优先；否则新版完整 inference manifest 使用同工作分辨率的线性 `gt_image.exr`，旧结果回退 ILE JSON 原始图；`target.png` 只用于预览，不参与 loss。
- 材质优化参考 `inverse_img_w_mi_ori.py`：显示空间自适应 MSE + L1、相对输入材质的 L1 prior、StepLR 等价调度和相对改进式 early stopping；逐像素使用 Adam，PosMLP 使用 AdamW。
- detached mean-exposure matching 改为显式 opt-in，默认关闭，以保证训练目标和最终物理渲染亮度一致。
- Stage C 的回退与 candidate/selected 多份 HDRI 已移除；唯一的优化 HDRI 会被重新加载并固定用于 ARM 阶段。
- manifest 记录 HDRI 重载误差和材质优化期间的冻结误差，两者必须为 0；ARM 仍保留同 seed 高 SPP validation gate。
- 默认材质顺序改为 `rm -> a`：先联合优化 roughness/metallic 并恢复该阶段最优 checkpoint，再冻结 RM、单独优化 albedo；每阶段使用独立 optimizer、学习率调度、early stopping 和输入材质 prior。
- `material_phase_summaries.json` 记录阶段顺序、最优 MSE 与冻结通道误差；实测 RM 阶段的 albedo 和 A 阶段的 roughness/metallic 最大变化均为 0。仍可通过 `--material-order` 做顺序消融，旧 `--material-channels` 仅作为单阶段兼容选项。首轮固定 mesh normal，不优化 normal map。
- 输出优化 HDRI、ARM EXR/PNG、两阶段 history、阶段渲染和统一 manifest。

## 已完成验证

### 数值与接口验证

- 本次新增优化入口已通过 `py_compile`；本次维护的入口与文档已通过定向 `git diff --check`。
- 颜色、MoGe、mesh、Hybrid 与 PosMLP 数值稳定性的 16 个单元测试全部通过，其中包括：
  - 相机中心投影；
  - `+X` 投向图像右侧；
  - `+Y` 投向图像上方；
  - 3D→2D→3D round-trip；
  - depth scale 和 JSON 读取。
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
- 成功加载 1 个 visible lamp 和 1 个 invisible lamp。
- local-only、visible-only、invisible-only 均得到有效非空照明。
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

当前结果目录：

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
5. 增加多场景测试、性能统计和自动结果对比。

### 后续研究扩展

- ILE window 的 sun/sky/ground 方向性表示适配。
- light position、orientation 和 size 的有限范围优化。
- 更可靠的 visible lamp 双面/定向发光表示。
- 多视角或补全几何，以改善画面外灯光的遮挡和多次反射。
- 材质—灯光交替 refinement 和物体插入评估。

## 已知限制与风险

- ILE RGB intensity 与 Mitsuba radiance 尚未建立物理单位映射。
- 当前只支持 lamp；window 不应直接简化为 uniform rectangle。
- Mitsuba area emitter 为单面发光；当前样例通过消融验证，但新场景仍需检查 OBJ winding。
- 可见灯 mesh 可能与 Materialist 单视图 mesh 接近共面，目前只通过小幅 offset 缓解。
- 单视图 mesh 不是封闭房间，画面外 invisible lamp 的遮挡和间接光只能近似表达。
- local-only 不包含窗外和远场照明，因此整体亮度低于输入并不一定是坐标错误。
- 主渲染和自定义 Materialist BSDF 仍依赖 CUDA；部分路径与模型权重依赖本机环境。
- 单图逆渲染存在材质、几何、曝光和光照间的固有歧义。
