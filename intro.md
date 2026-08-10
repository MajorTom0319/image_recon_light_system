# Materialist 项目简介

Materialist 是一个面向单张室内图像的逆渲染、重新打光与材质编辑项目。当前实现结合 MatNet、MoGe2、IndoorLightEditing 和 Mitsuba，从输入图像中恢复相机、几何、材质及显式灯光，并在统一的物理渲染场景中生成局部灯光、环境光或两者融合的结果。

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
        ├── mesh 重建                       ├── 灯光 mesh/center
        └── MatNet 材质                     └── RGB intensity
               │                                   │
               └─────────────────┬─────────────────┘
                                 ▼
                         hybrid_light 适配层
                     坐标 / 尺度 / 投影 / emitter
                                 │
                                 ▼
                          Mitsuba PBR 渲染
                    local / env / local + env
```

Materialist 负责场景几何和 BRDF，IndoorLightEditing 作为显式灯光 proposal 模型，Mitsuba 负责阴影、多次反射和材质—光照交互。当前 prototype 不接入 IndoorLightEditing 的 neural indirect renderer。

## 核心能力

- 使用 MatNet 预测 albedo、roughness、metallic、normal 和相对 depth。
- 使用 MoGe2 恢复 metric depth、normal、points、有效 mask 和相机内参。
- 使用统一工作分辨率和相机 K 将深度反投影为 Materialist mesh。
- 为 IndoorLightEditing 输出原图尺寸、稠密、有限的 `float32 depth.npy`。
- 读取 IndoorLightEditing 导出的 `light_predictions.json` 和 lamp OBJ。
- 将 visible/invisible lamp 转换为 Mitsuba area emitter。
- 使用 Materialist mesh 和逐像素 BRDF 进行 local-only、env-only 或 combined 渲染。
- 输出灯光投影诊断、raw EXR、OptiX 去噪结果和渲染 manifest。
- 保留原有环境光优化、HDRI 重渲染、材质编辑和离散灯光实验能力。

## 新增文件与功能

### MoGe2 与 IndoorLightEditing 输入

| 文件 | 功能 |
| --- | --- |
| `myutils/moge2_utils.py` | MoGe2 模型加载、有限值/mask 清理、mesh depth 和稠密 depth 准备 |
| `test_matnet_infer_moge2.py` | 联合检查 MatNet、MoGe2、相机、材质和 mesh，并输出 ILE 所需的 `depth.npy` |
| `Material_net/dpt.py` | 唯一的 `infer_image_scaled()` 实现，统一 scale、宽高比和 patch padding 语义 |

mesh 使用的 MoGe2 depth 保留“无效值为 0”的约定；IndoorLightEditing 的 `depth.npy` 则会用最近有效深度补齐，避免零深度参与反投影。

### Hybrid 显式灯光桥接

| 文件 | 功能 |
| --- | --- |
| `hybrid_light/light_types.py` | 定义统一的 mesh area light 和灯光集合数据结构 |
| `hybrid_light/io.py` | 解析并校验 ILE JSON、灯光中心、RGB、OBJ 路径和 depth scale |
| `hybrid_light/coordinate.py` | 封装 ILE → Materialist 坐标、3D→2D 投影和反投影 |
| `hybrid_light/mitsuba_builder.py` | 构造 Materialist mesh、相机、ILE area emitter 和可选 envmap 场景 |
| `hybrid_light/visualization.py` | 将灯光中心、OBJ 投影轮廓和 lamp mask 绘制到输入图像 |
| `hybrid_light/README.md` | Hybrid prototype 的运行方式和当前约束 |
| `scripts/render_ile_lights.py` | 初版融合渲染入口，支持验证、消融和三种光照模式 |
| `scripts/optimize_ile_farfield.py` | Stage B 逐灯 scale 优化，以及 Stage C 32×16 far-field HDRI 优化入口 |
| `scripts/optimize_ile_farfield_materials.py` | 关闭 Stage B、固定 ILE 灯强度，依次优化 Stage C far-field HDRI、Stage D roughness/metallic，最后优化 albedo |
| `tests/test_hybrid_light.py` | 坐标方向、投影 round-trip、尺度同步和 JSON 读取测试 |
| `materialist_indoorlightediting_hybrid_method.md` | 完整方法设计、阶段规划和实验建议 |

## Hybrid 渲染模式

`scripts/render_ile_lights.py` 支持：

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

固定灯光实验使用 `scripts/optimize_ile_farfield_materials.py`：完全跳过 Stage B，先固定 ILE radiance 优化 HDRI，再冻结全部光照，默认先联合优化 roughness/metallic 并冻结最优结果，最后单独优化 albedo。每个材质阶段使用独立 Adam/StepLR 与 early stopping，并检查未参与该阶段的材质通道严格不变。默认直接使用 ILE JSON 记录的原始输入图作为 target，材质 loss 参考 `inverse_img_w_mi_ori.py` 的显示空间 MSE/L1 和输入材质先验；曝光对齐改为可选。Stage C 只输出一份权威的 `farfield_optimized_32x16.exr/.hdr`，Stage D 会从该 EXR 重新建场景，并校验材质优化前后 env tensor 完全不变。

## 其他主要入口

| 文件或目录 | 用途 |
| --- | --- |
| `inverse_img_w_mi_ori.py` | 原始 Materialist 逆渲染流程 |
| `inverse_img_w_mi_geocalib.py` | 使用 GeoCalib/FOV 内参的逆渲染流程 |
| `inverse_img_w_mi_moge2.py` | 使用 MoGe2 相机与几何的完整逆渲染流程 |
| `render_matnet_pre_with_hdri.py` | 使用指定 HDRI 渲染 MatNet 预测结果 |
| `render_final.py` | 渲染优化后的最终材质和环境光结果 |
| `mat_edit.py` / `trans_edit.py` | 材质与近似透明效果编辑 |
| `optimal_lights/` | 半球离散灯光与材质联合优化实验 |
| `scripts/` | Hybrid 渲染、消融实验、环境光推理和可视化工具 |

## 主要输入与输出

Materialist run 目录通常为 `output_imgs/<save_name>/`，包括：

- `camera_meta.json`：工作尺寸、K、FOV、主点和相机姿态。
- `mesh_moge2.ply` 或 `<save_name>.ply`：单视图重建 mesh。
- `albedoPred.exr`、`roughnessPred.exr`、`metallicPred.exr`、`normalPred.exr`：初始材质。
- `best_results/`：经过逆渲染优化的材质、envmap 和重建图像。

ILE bridge 输入包括：

- `light_predictions.json`。
- `visLampPred_*.obj`、`invLampPred_*.obj`。
- visible/invisible lamp 的中心和 RGB intensity。
- 可选的 lamp mask、相机和 depth scale 元数据。

Hybrid 输出默认位于 `<materialist-dir>/hybrid_ile_render/`：

- `lights_projected.png`：灯光投影和 mask 对齐诊断。
- `render_<mode>_raw.exr`：未经去噪的线性渲染结果。
- `render_<mode>.exr` / `.png`：OptiX 去噪结果。
- `render_manifest.json`：mesh、材质、灯光、尺度和渲染参数记录。

## 环境与当前边界

项目主要面向 Python 3.10、PyTorch、CUDA、Mitsuba 3、Dr.Jit、Open3D、OpenCV 和 NumPy。MatNet 权重通过 Hugging Face 获取；MoGe2、GeoCalib 和 IndoorLightEditing 使用本地源码或权重。

当前 hybrid prototype 只接 lamp，不接 window 的 sun/sky/ground 方向性表示；ILE intensity 仍是 Mitsuba radiance 的相对初值；单视图 mesh 也无法完整表达画面外房间几何。因此当前实现用于验证坐标、尺度、显式灯位置和物理渲染链路，下一阶段将继续完成逐灯强度优化与受约束的 local + far-field lighting。
