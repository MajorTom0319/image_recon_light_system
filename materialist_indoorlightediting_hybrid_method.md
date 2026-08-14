# Materialist + IndoorLightEditing：单图显式灯光重建与物理渲染联合框架

> 目标：利用 **IndoorLightEditing** 从单张室内图像中恢复显式灯光参数，并将这些灯光参数转换为 **Materialist / Mitsuba** 可以直接使用的物理灯源，随后结合 Materialist 的材质、几何与可微渲染能力进行进一步渲染与灯光优化。
>
> 这份文档定位为：**方法设计文档 + 工程实现说明 + Codex 开发参考**。
>
> 第一阶段目标不是直接完成最终论文，而是先做出一个可运行、可调试、可定量分析的 prototype，为后续的 Hybrid Near-field + Far-field Lighting Reconstruction 打基础。

> **当前实现状态（2026-08-14）**：本文档前半部分保留了最初“先完成 lamp、暂缓 window”的研究路线，作为设计演进记录；它不再代表当前代码边界。`scripts/optimize_ile_farfield_materials.py` 已经接入 visible/invisible lamp 和 visible/invisible window。Lamp 使用普通 Mitsuba mesh area emitter；window 使用有限 OBJ 孔径、显式 `null` BSDF，并通过 `hybrid_light/ile_window_emitter.py` 保留 ILE `src/srcSky/srcGrd` 三组 `[RGB, direction, concentration]` 球面高斯。Window 保持预测位置，不应用 visible offset；offset 只用于 visible lamp。固定灯光优化默认使用 MoGe2 normal、PRB `max_depth=4`、灯具/深度/mesh-edge masked clipped-sRGB loss、周期固定多 seed validation，以及独立的 256 SPP raw final render。Direct Adam 启用 AMSGrad/masked updates，PosMLP 显式关闭 weight decay。Stage B 逐灯 scale 和通用 renderer 仍默认 lamp-only。

---

# 1. 方法概述

## 1.1 核心思想

当前 Materialist 的灯光恢复主要依赖环境贴图（environment map）：

\[
L = L_{\mathrm{env}}
\]

这种表示适合表达：

- 全局环境光；
- 远场光照；
- 不可见的大尺度光照；
- 低频 illumination。

但是对于室内场景中的局部灯源，例如：

- 顶灯；
- 吊灯；
- 台灯；
- 壁灯；
- 局部隐藏灯；

单纯使用 envmap 很难准确表达：

- 灯源真实 3D 位置；
- 灯源尺寸；
- 距离衰减；
- 局部高光；
- 软阴影；
- 真实遮挡关系。

因此，本项目首先利用 **IndoorLightEditing** 提供的显式灯光恢复能力，得到局部灯源参数：

\[
L_{\mathrm{ILE}}
=
\{E_1,E_2,\dots,E_N\}
\]

然后将这些灯光参数转换到 Materialist / Mitsuba 所使用的场景坐标系中，构建真正的物理 area emitter。

最终：

\[
\boxed{
I
\rightarrow
\text{IndoorLightEditing}
\rightarrow
\text{Explicit Lights}
\rightarrow
\text{Materialist/Mitsuba}
\rightarrow
\text{Physical Rendering}
}
\]

在 prototype 成功之后，再进一步扩展为：

\[
\boxed{
L
=
L_{\mathrm{local}}
+
L_{\mathrm{far}}
}
\]

其中：

- \(L_{\mathrm{local}}\)：由 IndoorLightEditing 初始化的显式近场灯；
- \(L_{\mathrm{far}}\)：由 Materialist envmap 表示的全局 / 远场 / residual illumination。

---

# 2. 系统总体流程

整体系统建议拆成两个并行分支。

```text
                           Input RGB
                              │
              ┌───────────────┴────────────────┐
              │                                │
              ▼                                ▼
      IndoorLightEditing                   Materialist
              │                                │
              │                                ├── Material / MatNet
              │                                │
              │                                ├── Geometry / Mesh
              │                                │
              ▼                                ▼
      Explicit Light Params             Material + Geometry
              │                                │
              └───────────────┬────────────────┘
                              │
                              ▼
                     Light Converter
                              │
              coordinate / scale / FOV
                              │
                              ▼
                     Mitsuba Scene
                              │
             ┌────────────────┴───────────────┐
             │                                │
             ▼                                ▼
       Explicit Local Lights             Global Env Light
             │                                │
             └────────────────┬───────────────┘
                              ▼
                  Differentiable PBR Rendering
                              │
                              ▼
                   Rendered Image / Loss
                              │
                              ▼
                      Light Refinement
```

第一版不要求一次完成所有模块。

建议按照以下阶段逐步实现：

```text
Stage 0: Materialist baseline
Stage 1: IndoorLightEditing 输出灯光 JSON
Stage 2: 坐标 / FOV / scale 转换
Stage 3: 将灯光导入 Mitsuba
Stage 4: 仅使用 ILE 灯光渲染
Stage 5: 优化灯光强度
Stage 6: 加入 ambient / low-frequency envmap
Stage 7: Hybrid local + global lighting
Stage 8: 后续位置 / 尺寸 / 材质联合优化
```

---

# 3. 第一阶段研究目标

第一版 prototype 不追求一次完成最终方法。

只需要证明以下五件事情：

1. IndoorLightEditing 能够对输入图像输出具有空间意义的显式灯源；
2. 这些灯源能够正确转换到 Materialist / Mitsuba 的坐标系；
3. 转换之后的 Mitsuba 灯源在图像中的投影位置与原始灯源位置基本一致；
4. 使用这些显式灯光，可以在 Materialist 重建的场景上产生合理的局部照明和阴影；
5. 进一步优化每盏灯的强度以后，渲染结果可以更接近输入图像。

如果这五件事情可以成立，那么后续 Hybrid Lighting 方法的基础就已经跑通。

---

# 4. 两个项目在系统中的职责

## 4.1 IndoorLightEditing 的职责

IndoorLightEditing 不再作为最终 renderer 使用。

它的主要职责是：

\[
\boxed{
\text{Lighting Proposal / Lighting Initialization}
}
\]

即：

```text
RGB / Depth / Mask
        ↓
IndoorLightEditing
        ↓
Explicit 3D Light Parameters
```

当前固定灯光优化入口提取：

- visible lamp；
- invisible lamp；
- visible window；
- invisible window。

历史第一版先完成 lamp；当前实现已在不改动主优化循环的前提下补齐方向性 window adapter。

---

## 4.2 Materialist 的职责

Materialist 提供：

- 材质预测；
- 场景 mesh；
- Mitsuba scene；
- physically based rendering；
- differentiable optimization。

Materialist 最终负责：

\[
\boxed{
R(G,M,L)
}
\]

其中：

- \(G\)：geometry；
- \(M\)：material；
- \(L\)：从 IndoorLightEditing 转换来的 lighting。

后续再扩展成：

\[
R(G,M,L_{\mathrm{local}},L_{\mathrm{env}})
\]

---

# 5. IndoorLightEditing 需要提取的灯光参数

建议不要直接把 IndoorLightEditing 的渲染结果交给 Materialist。

正确做法是直接提取其 **中间灯源参数**。

---

# 6. Visible Lamp

对于 visible lamp，第一版主要提取：

\[
E_i^{vis}
=
(p_i,c_i)
\]

其中：

- \(p_i\)：预测的 3D 灯源中心；
- \(c_i\)：预测的 RGB 光源强度 / radiance-like 参数。

建议同时保存：

- 2D lamp mask；
- mask bounding box；
- predicted depth；
- local normal；
- depth normalization scale。

第一版中，visible lamp 的几何形状可以有两种实现方式。

---

## 6.1 Visible Lamp V0：Rectangle Approximation

将 visible lamp 近似为一个矩形面积光：

\[
E_i =
(p_i,R_i,w_i,h_i,c_i)
\]

需要估计：

- center；
- orientation；
- width；
- height。

最简单可以：

```text
center      ← ILE predicted center
orientation ← local surface normal
width       ← 由 lamp mask 粗略估计
height      ← 由 lamp mask 粗略估计
radiance    ← ILE RGB × scale
```

优点：

- 实现简单；
- Mitsuba 原生支持 rectangle + area emitter；
- 很适合第一阶段验证。

---

## 6.2 Visible Lamp V1：Lamp Surface Mesh

第二种方式是复用 IndoorLightEditing 中 visible lamp renderer 的思路。

根据：

```text
lamp mask
+
depth
+
normal
+
predicted center
```

生成 lamp surface mesh：

```text
lamp_0.ply
```

然后在 Mitsuba 中：

```text
PLY Mesh
+
Area Emitter
```

这种方式比单纯 rectangle 更贴近 ILE 本身的灯源表示。

推荐开发顺序：

```text
V0 Rectangle
    ↓
验证 pipeline
    ↓
V1 Surface Mesh
```

不要反过来。

---

# 7. Invisible Lamp

Invisible lamp 对 Materialist 更容易接入。

建议统一表示为：

\[
E_i^{inv}
=
(p_i,A_i,c_i)
\]

其中：

- \(p_i\)：3D center；
- \(A_i\)：三个 3D box axes；
- \(c_i\)：RGB light intensity。

假设：

\[
A_i =
[a_x,a_y,a_z]
\]

即可构建一个 3D box。

对应的八个顶点：

\[
p_i
\pm \frac{a_x}{2}
\pm \frac{a_y}{2}
\pm \frac{a_z}{2}
\]

然后生成 emissive box mesh：

```text
Invisible Lamp
      ↓
center + axes
      ↓
box vertices
      ↓
triangle mesh
      ↓
Mitsuba area emitter
```

第一阶段可先将整个 box 表面设为 emitter。

后续可以根据 ILE 原始实现决定只让部分表面发光。

---

# 8. Window Adapter（当前已实现）

最初版本暂缓 window，因为它的 illumination 比 lamp 更复杂；当前固定灯光入口已经完成适配。

原因是 window 的 illumination 比 lamp 更复杂。

window 不只是：

```text
rectangle + uniform RGB
```

IndoorLightEditing 对 window 实际包含：

- center；
- normal；
- x axis；
- y axis；
- sun lighting；
- sky lighting；
- ground lighting；
- directional parameters。

如果直接把它压成 uniform area light，会损失原模型中的重要方向信息。

当前实现采用：

```text
window OBJ finite aperture
+
sun / sky / ground directional SG radiance
```

对场景点指向窗口的方向 \(\omega\)，辐射度为：

\[
L(\omega)=\sum_{k\in\{sun,sky,ground\}}
c_k\exp\left(\lambda_k\min(d_k\cdot\omega-1,0)\right).
\]

普通 area emitter 的 radiance texture 在直接光采样时拿不到接收方向，不能正确实现该函数。因此 `hybrid_light/ile_window_emitter.py` 覆盖 `sample_direction`、`eval_direction` 和直接命中 `eval`，窗口 OBJ 继续负责孔径、距离、遮挡和软阴影。当前 `optimize_ile_farfield_materials.py` 默认启用 window；Stage B 入口仍保持 lamp-only。

---

# 9. 中间灯光数据格式

强烈建议在两个项目之间定义一个统一的中间格式。

不要让 Materialist 直接 import IndoorLightEditing 的模型对象。

推荐：

```text
IndoorLightEditing
        ↓
lights.json
        ↓
Materialist
```

---

# 10. 推荐 JSON Schema

示例：

```json
{
  "version": "0.1",
  "image": {
    "original_width": 640,
    "original_height": 480,
    "canonical_width": 640,
    "canonical_height": 480
  },

  "camera": {
    "coordinate_system": "camera",
    "fov_x_deg": 57.95,
    "fx": null,
    "fy": null,
    "cx": null,
    "cy": null,
    "depth_scale": 1.0
  },

  "lights": [
    {
      "id": 0,
      "type": "visible_lamp",
      "center": [0.15, 1.20, -3.50],
      "rgb": [5.2, 4.8, 4.1],

      "mask_path": "lampMask_0.png",

      "geometry": {
        "representation": "rectangle",
        "normal": [0.0, -1.0, 0.0],
        "width": 0.50,
        "height": 0.20
      }
    },

    {
      "id": 1,
      "type": "invisible_lamp",
      "center": [-1.1, 1.8, -4.5],

      "axes": [
        [0.6, 0.0, 0.0],
        [0.0, 0.25, 0.0],
        [0.0, 0.0, 0.15]
      ],

      "rgb": [3.4, 3.0, 2.7]
    }
  ]
}
```

---

# 11. 建议增加灯光置信度字段

后续为了筛选不可靠灯源，可以扩展：

```json
{
  "confidence": 0.91,
  "source": "IndoorLightEditing",
  "is_visible": true
}
```

甚至：

```json
{
  "position_confidence": 0.8,
  "geometry_confidence": 0.6,
  "radiance_confidence": 0.7
}
```

第一版可以不使用，但最好预留接口。

---

# 12. 图像预处理必须统一

这是整个 prototype 中最重要的工程问题之一。

IndoorLightEditing 和 Materialist 不能分别对输入图像独立 crop。

错误方式：

```text
Original Image
   ├── ILE resize to 320×240
   └── Materialist center crop + resize 512×512
```

这样会造成：

- 灯的 2D mask 不一致；
- FOV 不一致；
- camera ray 不一致；
- 3D light 投影错位。

正确方式：

```text
Original Image
      ↓
Canonical Preprocessing
      ↓
Canonical Image
   ┌───────────────┐
   ↓               ↓
ILE input      Materialist input
resize only       resize only
```

---

# 13. Canonical Image

建议增加一个统一 preprocessing 脚本：

```text
prepare_input.py
```

输出：

```text
workdir/sample_x/
    original.png
    canonical.png
    metadata.json
```

`metadata.json` 中记录：

```json
{
  "original_size": [1920, 1080],
  "crop_box": [x0, y0, x1, y1],
  "canonical_size": [640, 480],
  "scale_x": 0.333,
  "scale_y": 0.444
}
```

两个项目都只读取 canonical image。

---

# 14. Camera Coordinate 统一

建议整个联合系统内部统一采用：

```text
Camera Coordinate
```

推荐定义：

```text
camera center = (0, 0, 0)

+x → image right
+y → image up
-z → camera forward
```

即 Mitsuba / graphics 风格：

\[
x \rightarrow right
\]

\[
y \rightarrow up
\]

\[
z \rightarrow backward
\]

camera looking direction：

\[
(0,0,-1)
\]

---

# 15. IndoorLightEditing 到内部坐标转换

如果 IndoorLightEditing 输出本身已经采用：

```text
+x right
+y up
-z forward
```

那么方向与内部约定基本一致。

但不要直接假设一定正确。

必须写 unit test。

推荐函数：

```python
def ile_to_internal_point(p):
    ...
```

```python
def ile_to_internal_vector(v):
    ...
```

```python
def internal_to_mitsuba_point(p):
    ...
```

即使第一版只是 identity，也必须通过函数封装。

不要在各个文件里手写：

```python
p[1] *= -1
p[2] *= -1
```

---

# 16. Camera FOV 问题

IndoorLightEditing 与 Materialist 默认 FOV 可能不同。

这是不能忽略的问题。

原则：

\[
\boxed{
K_{\mathrm{ILE}}
=
K_{\mathrm{Materialist}}
}
\]

最好最终统一 camera intrinsics：

\[
K=
\begin{bmatrix}
f_x&0&c_x\\
0&f_y&c_y\\
0&0&1
\end{bmatrix}
\]

---

# 17. 第一版 Camera 方案

可以采用两个方案。

## 方案 A：统一固定 FOV

例如统一：

```text
FOV = 57.95°
```

然后：

- IndoorLightEditing 使用 57.95°；
- Materialist Mitsuba sensor 也改成同样 FOV。

这是第一版最简单的方案。

---

## 方案 B：后续使用 MoGe Camera

后续更加合理：

```text
Input RGB
    ↓
MoGe-2
    ↓
camera intrinsics
```

然后：

```text
ILE   ← same K
Mitsuba ← same K
```

这是最终方法更推荐的方案。

第一版不必马上实现。

---

# 18. FOV / Projection Debug

必须实现：

```python
project_point_to_image()
```

输入：

```text
3D point
camera intrinsics
```

输出：

```text
2D pixel
```

对于每个 visible lamp：

\[
p_i^{ILE}
\rightarrow
(u_i,v_i)
\]

应该基本落在：

```text
lamp mask
```

内部。

推荐调试条件：

\[
\pi(p_i)
\in M_i
\]

或者至少：

\[
\|\pi(p_i)-center(M_i)\|
< \tau
\]

其中第一阶段：

```text
τ = 20~30 pixels
```

可接受。

---

# 19. Depth Scale 问题

IndoorLightEditing 可能会对 depth 做 normalization。

因此它预测的 3D light center 可能处于 normalized scene scale。

假设：

\[
D_{\mathrm{ILE}}
=
D_{\mathrm{real}}/s
\]

那么：

\[
p_{\mathrm{ILE}}
=
p_{\mathrm{real}}/s
\]

转换回 Materialist mesh：

\[
\boxed{
p_{\mathrm{Materialist}}
=
s\cdot p_{\mathrm{ILE}}
}
\]

同时：

\[
A_{\mathrm{Materialist}}
=
s\cdot A_{\mathrm{ILE}}
\]

即：

- center 要乘 scale；
- axes 要乘 scale；
- width / height 要乘 scale。

---

# 20. 深度尺度必须写入 lights.json

例如：

```json
{
  "camera": {
    "depth_normalized": true,
    "depth_scale": 1.482
  }
}
```

Materialist adapter 中：

```python
if depth_normalized:
    center *= depth_scale
    axes *= depth_scale
```

---

# 21. 一个非常重要的原则：不要同时混用两个独立深度

例如：

```text
IndoorLightEditing 用自己的 depth
Materialist mesh 用 MoGe depth
```

如果两者尺度 / 几何差异很大：

```text
ILE 灯位置
    ↓
Materialist mesh
```

会发生漂移。

第一版最好：

```text
IndoorLightEditing
和
Materialist geometry

尽量使用同一 depth source
```

推荐优先级：

```text
方案 1：
全部使用 IndoorLightEditing / 原 pipeline 的一致 depth

方案 2：
全部统一使用 MoGe-2 depth

方案 3：
不得已才各自使用不同 depth
```

---

# 22. 光照强度单位不能直接完全相信

IndoorLightEditing 输出的 RGB intensity 是为它自身的 neural / physical renderer 学习的参数。

Materialist 中 Mitsuba area emitter 的 `radiance` 有自己的物理意义。

因此不要直接认为：

```text
ILE rgb = [10,10,10]
```

等价于：

```text
Mitsuba radiance = [10,10,10]
```

正确做法是把 ILE 结果看作：

```text
relative RGB / radiance initialization
```

---

# 23. 灯光强度参数化

推荐：

\[
c_i^{0}
=
c_i^{ILE}
\]

为灯的颜色 / 相对 RGB。

再增加：

\[
\alpha_i
\]

真正 Mitsuba radiance：

\[
\boxed{
L_i
=
\exp(\alpha_i)c_i^{0}
}
\]

初始化：

\[
\alpha_i = 0
\]

优化：

\[
\alpha_i
\]

这样可以保证：

\[
L_i > 0
\]

而且不会出现负灯光强度。

---

# 24. 可选：颜色也参与小范围优化

后续可以：

\[
L_i
=
\exp(\alpha_i)
\cdot
\operatorname{softplus}(c_i^{ILE}+\Delta c_i)
\]

但第一版：

```text
固定 RGB 色度
只优化 scalar intensity
```

最稳定。

---

# 25. Materialist Scene 修改

当前 Materialist 可抽象为：

```text
Scene
├── Mesh
├── Sensor
├── Integrator
└── Environment Map
```

需要改为：

```text
Scene
├── Mesh
├── Sensor
├── Integrator
│
├── Global Env Emitter
│
├── Local Lamp 0
├── Local Lamp 1
├── Local Lamp 2
│
└── ...
```

---

# 26. 建议新增 Light Adapter

新增目录：

```text
hybrid_light/
```

推荐：

```text
hybrid_light/
├── __init__.py
├── io.py
├── coordinate.py
├── light_types.py
├── ile_adapter.py
├── mitsuba_builder.py
├── intensity.py
└── visualization.py
```

---

# 27. `light_types.py`

建议定义统一 dataclass：

```python
@dataclass
class AreaLight:
    id: int
    light_type: str

    center: np.ndarray
    normal: np.ndarray

    width: float
    height: float

    rgb: np.ndarray

    confidence: float = 1.0
```

Invisible lamp：

```python
@dataclass
class BoxLight:
    id: int

    center: np.ndarray
    axes: np.ndarray

    rgb: np.ndarray

    confidence: float = 1.0
```

---

# 28. `io.py`

负责：

```python
load_lights_json()
save_lights_json()
```

不要把 JSON parsing 写在 renderer 内部。

---

# 29. `coordinate.py`

统一负责：

```python
ile_to_internal()
internal_to_mitsuba()
scale_point()
scale_axes()
project_point()
unproject_pixel()
```

以及 camera 数据：

```python
CameraModel
```

---

# 30. `ile_adapter.py`

负责：

```text
IndoorLightEditing output
        ↓
Unified Light Objects
```

例如：

```python
def convert_visible_lamp(ile_result, metadata):
    ...
```

```python
def convert_invisible_lamp(ile_result, metadata):
    ...
```

---

# 31. `mitsuba_builder.py`

负责：

```python
build_rectangle_emitter(light)
build_box_emitter(light)
build_mesh_emitter(light)
```

返回 Mitsuba scene dict。

不要把 scene 构造逻辑继续塞进一个大文件。

---

# 32. Rectangle Area Light

Mitsuba 中逻辑上应类似：

```python
{
    "type": "rectangle",

    "to_world": ...,

    "emitter": {
        "type": "area",

        "radiance": {
            "type": "rgb",
            "value": [r, g, b]
        }
    }
}
```

其中 `to_world` 负责：

- translation；
- rotation；
- scale。

---

# 33. Rectangle `to_world` 构造

假设：

```text
center = p
normal = n
width = w
height = h
```

需要建立正交基：

\[
t_1,t_2,n
\]

满足：

\[
t_1\perp t_2
\]

\[
t_1\perp n
\]

\[
t_2\perp n
\]

然后构建：

```text
local rectangle
    ↓
scale(width, height)
    ↓
rotate to normal
    ↓
translate to center
```

建议写：

```python
def build_frame_from_normal(normal):
    ...
```

---

# 34. Invisible Box Light

可以先：

```text
axes
 ↓
8 vertices
 ↓
12 triangles
 ↓
temporary ply
 ↓
Mitsuba mesh + area emitter
```

比尝试在 Mitsuba dictionary 内构建复杂 box 更容易 debug。

---

# 35. Temporary Geometry Cache

建议：

```text
workdir/sample_x/lights/
    lamp_000.ply
    lamp_001.ply
```

以及：

```text
converted_lights.json
```

方便直接用 MeshLab / Open3D 查看。

---

# 36. 不要搬 IndoorLightEditing 的 indirect renderer

IndoorLightEditing 中：

```text
direct lighting
+
shadow prediction
+
indirect lighting network
```

这些后处理模块不需要接入 Materialist。

Materialist 已经使用 Mitsuba path tracing。

所以新的职责划分是：

```text
IndoorLightEditing：
只预测显式灯参数

Mitsuba：
处理真实 light transport
```

包括：

- direct illumination；
- shadow；
- multiple bounce；
- indirect illumination；
- BRDF interaction。

这样系统会更加统一。

---

# 37. 第一版 Rendering Mode

建议实现三个 renderer mode。

```python
render_local_only()
```

```python
render_env_only()
```

```python
render_combined()
```

输出：

```text
render_local.exr
render_env.exr
render_combined.exr
```

这样可以非常容易看出：

```text
Local Light 到底解释了什么？
Envmap 到底解释了什么？
```

---

# 38. Stage A：只使用 IndoorLightEditing 灯光

第一版最重要的实验：

\[
\boxed{
L = L_{\mathrm{ILE}}
}
\]

关闭 Materialist 原 envmap。

只使用：

```text
visible lamp
+
invisible lamp
```

进行 Mitsuba path tracing。

目的：

> 验证 ILE → Mitsuba 的灯光转换本身是否成立。

---

# 39. Stage A 不做 optimization

初始版本：

```text
geometry fixed
material fixed
light position fixed
light geometry fixed
light intensity fixed
```

只 render。

输出：

```text
Input
ILE lamp masks
3D lamps
Mitsuba projection
Local-only render
```

不要先优化。

---

# 40. Stage B：只优化灯光强度

在 Stage A 正确后：

固定：

\[
p_i
\]

\[
R_i
\]

\[
s_i
\]

\[
c_i
\]

只优化：

\[
\alpha_i
\]

即：

\[
L_i
=
e^{\alpha_i}c_i
\]

损失：

\[
L_{\mathrm{rgb}}
=
\|I_r-I_t\|_1
\]

第一版先用最简单 loss。

---

# 41. Stage B Loss

建议：

\[
L
=
L_{\mathrm{rgb}}
+
\lambda_\alpha L_{\mathrm{intensity}}
\]

其中：

\[
L_{\mathrm{intensity}}
=
\sum_i \alpha_i^2
\]

避免 ILE 初值被完全抛弃。

---

# 42. Stage C：Local + Uniform Ambient

只使用 local lamp，很多区域可能偏暗。

因为：

- indirect illumination 不完整；
- invisible source 不一定全部恢复；
- ambient illumination 未建模。

因此增加一个统一环境光：

\[
L_{\mathrm{amb}}
=
c_{\mathrm{amb}}
\]

可以先用：

```text
constant env
```

只优化：

\[
RGB_{\mathrm{amb}}
\]

或者一个 exposure：

\[
\beta
\]

---

# 43. Stage C 模型

\[
\boxed{
L
=
L_{\mathrm{ILE}}
+
L_{\mathrm{ambient}}
}
\]

优化：

\[
\{\alpha_i\}
+
c_{\mathrm{amb}}
\]

这样 local light 负责局部结构；

ambient 负责背景整体亮度。

---

# 44. Stage D：Local + Low-Frequency Envmap

再进一步：

\[
\boxed{
L
=
L_{\mathrm{local}}
+
L_{\mathrm{far}}
}
\]

其中：

\[
L_{\mathrm{far}}
\]

不使用 unrestricted 高分辨率 envmap。

先使用低分辨率：

```text
8×16
```

或者：

```text
16×32
```

再上采样给 Mitsuba。

目的是避免：

```text
envmap
```

把所有局部灯光作用重新吸收掉。

---

# 45. Local / Global 职责划分

最终希望：

## Local

解释：

```text
direct light
shadow
local highlight
near-field attenuation
```

## Global

解释：

```text
ambient
distant source
missing source
global residual
```

即：

\[
L=L_{\mathrm{local}}+L_{\mathrm{far}}
\]

---

# 46. 为什么不直接使用 Materialist 原始 full envmap

如果同时给：

```text
ILE local lamps
+
high-dimensional envmap
```

然后从头一起 optimize，很容易：

```text
local lamps → 被优化得很弱
envmap      → 吸收大部分照明
```

这样显式灯就失去了意义。

因此必须控制 envmap 自由度。

---

# 47. Stage E：Global → Local 分阶段优化

后续可以：

```text
Step 1:
固定 local
优化 ambient / low-freq env

Step 2:
固定 env
优化 local intensity

Step 3:
local + env joint refine
```

即：

\[
L_{\mathrm{far}}
\rightarrow
L_{\mathrm{local}}
\rightarrow
L_{\mathrm{joint}}
\]

---

# 48. Stage F：Material Refinement

最后才考虑 Materialist 材质。

顺序：

```text
Geometry
Material Init
ILE Light Init

      ↓

Global Light

      ↓

Local Light

      ↓

Joint Lighting

      ↓

Material Refinement
```

即：

\[
\boxed{
Global
\rightarrow
Local
\rightarrow
Joint Light
\rightarrow
Material
}
\]

这是后续论文版本的 staged optimization。

---

# 49. 第一版不优化灯位置

强烈建议：

```text
position fixed
orientation fixed
size fixed
```

只优化 intensity。

因为灯位置改变会导致：

```text
shadow boundary moving
```

涉及 visibility discontinuity。

这比普通颜色 / 强度参数难优化很多。

第一版先证明：

> learned light initialization + physical refinement 可行。

---

# 50. 第二版灯几何优化

等第一版稳定后才加入：

\[
p=p_0+\Delta p
\]

限制：

\[
\|\Delta p\|<\delta
\]

例如：

```text
δ = 0.1 ~ 0.3 m
```

orientation：

\[
R=R_0R(\Delta\theta)
\]

size：

\[
s=s_0\exp(\Delta s)
\]

不要允许完全自由移动。

---

# 51. 几何优化建议

初步更推荐：

```text
coarse grid search
+
gradient intensity optimization
```

例如灯位置：

```text
x ∈ {-0.1, 0, +0.1}
y ∈ {-0.1, 0, +0.1}
z ∈ {-0.1, 0, +0.1}
```

评估后选择最佳位置。

这比一开始直接对 visibility 做 gradient 更稳定。

---

# 52. Debug 优先级

整个系统最重要的调试顺序：

```text
1. Image crop
2. Camera FOV
3. Coordinate convention
4. Depth scale
5. Light projection
6. Light geometry
7. Light intensity
8. Material
9. Optimization
```

不要跳步骤。

---

# 53. 必须生成的 Debug 输出

对每个 sample 至少输出：

```text
debug/
├── input.png
├── lamp_masks.png
├── depth.png
├── material_albedo.png
├── material_roughness.png
├── mesh_camera.png
├── lights_3d.png
├── lights_projected.png
├── render_local.exr
├── render_local.png
├── render_env.exr
├── render_combined.exr
└── comparison.png
```

---

# 54. `lights_projected.png`

这是第一阶段最重要的 debug 图。

在 input 上画：

```text
lamp mask
+
ILE center projection
+
Mitsuba center projection
+
rectangle corners
```

例如：

```text
red dot    = ILE center
green dot  = converted Mitsuba center
blue box   = projected area light
yellow mask = GT / detected lamp mask
```

如果这些对不上：

> 不要继续优化。

---

# 55. 必须写的 Unit Test

## Test 1：中心投影

给定：

\[
p=[0,0,-3]
\]

应该投到 image center。

---

## Test 2：X 方向

如果：

\[
x>0
\]

应该投到图像右侧。

---

## Test 3：Y 方向

如果：

\[
y>0
\]

应该投到图像上方。

---

## Test 4：round-trip

\[
P
\rightarrow
(u,v,D)
\rightarrow
P'
\]

满足：

\[
\|P-P'\|<\epsilon
\]

---

## Test 5：Scale

深度 scale 改变以后：

\[
center
\]

与：

\[
axes
\]

必须同步改变。

---

# 56. 推荐第一批测试图

第一轮不要使用复杂图。

优先选择：

```text
单个 visible ceiling lamp
无大面积 window
灯完全在画面中
阴影明显
场景结构简单
```

然后：

```text
单个 invisible lamp
```

再：

```text
多个 lamps
```

最后才：

```text
window + lamp
```

---

# 57. 初步实验矩阵

建议先做：

| ID | Lighting | Optimization | Env | 目的 |
|---|---|---|---|---|
| A | Materialist Env | Env | Full | 原始 baseline |
| B | ILE Lamp | None | None | 检查迁移 |
| C | ILE Lamp | Intensity | None | 检查物理 refinement |
| D | ILE Lamp | Intensity | Constant | 增加 ambient |
| E | ILE Lamp | Intensity | Low-Freq | Hybrid prototype |

---

# 58. 推荐评价指标

第一阶段可以简单：

## Reconstruction

\[
PSNR
\]

\[
SSIM
\]

\[
LPIPS
\]

但不要只看 RGB。

---

# 59. 灯位置评价

OpenRooms 上可以：

\[
E_p
=
\|p_{\mathrm{pred}}-p_{\mathrm{GT}}\|_2
\]

比较：

```text
IndoorLightEditing raw
Converted light
Refined light
```

---

# 60. 阴影评价

如果有 GT shadow：

```text
Shadow IoU
Boundary F-score
PSNR
```

尤其建议比较：

```text
Materialist Env
ILE Lamp
ILE Lamp + refinement
```

---

# 61. Direct Shading

只开某一盏灯：

\[
I_i
=
R(G,M,E_i)
\]

比较 GT 单灯 direct shading。

这个指标比 full RGB reconstruction 更能反映灯参数是否正确。

---

# 62. Probe Evaluation

后续可以在桌面 / 地面上插入：

```text
diffuse sphere
chrome sphere
rough sphere
metal sphere
```

分别使用：

```text
GT Light
Materialist Env
ILE Light
Refined ILE Light
```

渲染。

如果显式灯正确：

- shadow direction；
- shadow softness；
- highlight；
- illumination direction；

应该更接近 GT。

---

# 63. Object Insertion

最终应用：

```text
Input scene
   ↓
recover lighting
   ↓
insert sphere / bunny / teapot
```

比较：

```text
Materialist envmap
ILE explicit lamp
Hybrid lighting
```

这是非常直观的 qualitative result。

---

# 64. Real Image Application

真实图不需要 GT。

展示：

```text
Input
Material
Geometry
Detected lamps
3D explicit lamps
Reconstruction
Light removal
Light intensity editing
Light color editing
Object insertion
```

---

# 65. 第一版代码目录建议

假设你在现有 Materialist 修改项目中开发：

```text
image_recon_light_system/
│
├── indoorlight_bridge/
│   │
│   ├── README.md
│   │
│   ├── io.py
│   ├── coordinate.py
│   ├── camera.py
│   ├── light_types.py
│   ├── ile_adapter.py
│   ├── visible_lamp.py
│   ├── invisible_lamp.py
│   ├── mitsuba_builder.py
│   ├── projection.py
│   └── visualization.py
│
├── scripts/
│   ├── export_ile_lights.py
│   ├── convert_ile_lights.py
│   ├── debug_light_projection.py
│   ├── render_ile_lights.py
│   └── optimize_light_intensity.py
│
├── configs/
│   ├── ile_local_only.yaml
│   ├── ile_ambient.yaml
│   └── ile_hybrid.yaml
│
├── evaluation/
│   ├── evaluate_reconstruction.py
│   ├── evaluate_light_position.py
│   ├── evaluate_shadow.py
│   └── evaluate_probe.py
│
└── workdir/
```

---

# 66. IndoorLightEditing 侧建议修改

不要大改 ILE 网络。

只需要：

```text
原 inference
    ↓
获得灯参数
    ↓
export JSON
```

建议新建：

```text
exportLights.py
```

或者在 inference 末尾调用：

```python
export_lights_json(...)
```

---

# 67. `export_lights_json()` 输入

建议：

```python
def export_lights_json(
    output_path,

    vis_lamp_centers,
    vis_lamp_srcs,

    inv_lamp_centers,
    inv_lamp_axes,
    inv_lamp_srcs,

    camera_metadata,
    depth_scale,
    masks=None,
):
    ...
```

---

# 68. Materialist 侧第一版入口

建议不要继续改原始 main script 太多。

新建：

```text
run_ile_materialist.py
```

逻辑：

```python
image = load_image()

material = load_or_predict_material()

mesh = load_or_reconstruct_mesh()

lights = load_lights_json()

lights = convert_lights_to_materialist(
    lights,
    camera,
    depth_scale
)

scene = build_mitsuba_scene(
    mesh=mesh,
    material=material,
    lights=lights,
    env=None
)

image_render = render(scene)

save(image_render)
```

---

# 69. 第一版 Optimization 入口

新建：

```text
optimize_ile_lights.py
```

伪代码：

```python
lights = load_lights()

for light in lights:
    light.log_intensity = mi.Float(0.0)

for step in range(num_steps):

    for light in lights:
        radiance = exp(light.log_intensity) * light.rgb_init
        update_mitsuba_light(light, radiance)

    pred = mi.render(scene, params, spp=spp)

    loss_rgb = l1(pred, target)

    loss_reg = sum(
        light.log_intensity ** 2
        for light in lights
    )

    loss = loss_rgb + lambda_reg * loss_reg

    optimizer.step()
```

实际实现时根据 Materialist 当前的 Dr.Jit / Mitsuba optimizer 结构调整。

---

# 70. 推荐配置文件

例如：

```yaml
camera:
  use_canonical_fov: true
  fov_deg: 57.95

lighting:
  use_visible_lamps: true
  use_invisible_lamps: true
  use_windows: true  # fixed-light material entry; Stage B remains lamp-only

  env_mode: none

optimization:
  optimize_intensity: true
  optimize_position: false
  optimize_size: false
  optimize_color: false

  num_steps: 300
  spp: 32

loss:
  rgb_l1: 1.0
  intensity_prior: 0.01
```

Hybrid：

```yaml
lighting:
  env_mode: low_frequency
  env_height: 8
  env_width: 16
```

---

# 71. 开发 Milestone 1

## 目标

从 IndoorLightEditing 输出：

```text
lights.json
```

成功标准：

```text
可以读取所有 visible / invisible lamp 参数
```

暂时不涉及 Materialist。

---

# 72. Milestone 2

## 目标

在 Open3D / trimesh / matplotlib 3D 中：

```text
camera
+
scene point cloud
+
predicted lamps
```

正确显示。

成功标准：

```text
灯大致位于对应的天花板 / 墙体区域
```

---

# 73. Milestone 3

## 目标

把灯中心投影回输入图。

成功标准：

```text
visible lamp center
位于 lamp mask 内或附近
```

如果失败：

```text
先检查 FOV / crop / coordinate / depth scale
```

---

# 74. Milestone 4

## 目标

Mitsuba 中只建立：

```text
camera
+
empty simple plane
+
ILE area light
```

不要先上完整 Materialist scene。

验证灯是否能正确：

```text
发光
投阴影
产生合理 attenuation
```

---

# 75. Milestone 5

## 目标

将灯导入 Materialist scene。

只 render：

```text
local light only
```

成功标准：

```text
场景中对应区域确实被照亮
阴影方向基本合理
```

---

# 76. Milestone 6

## 目标

优化灯强度。

比较：

```text
ILE raw
vs
ILE + intensity refinement
```

成功标准：

```text
RGB reconstruction 变好
灯仍保持原来的空间位置
```

---

# 77. Milestone 7

加入 constant ambient。

比较：

```text
Local only
Local + ambient
```

---

# 78. Milestone 8

加入 low-frequency envmap。

形成：

\[
\boxed{
L=L_{\mathrm{local}}+L_{\mathrm{far}}
}
\]

这就是 Hybrid Lighting prototype。

---

# 79. 第一阶段禁止做的事情

为了防止 scope 失控：

```text
不要重新训练 IndoorLightEditing
不要重新训练 Materialist
不要重新训练 MatNet
不要加入 Diffusion
不要加入 3DGS
不要做 Gaussian emitter
不要先做 window
不要先优化 lamp geometry
不要先优化 material
不要同时使用高自由度 envmap + local lights
```

---

# 80. 主要风险 1：Camera 不一致

症状：

```text
lamp 3D center 看似合理
但是投影回图像完全错位
```

优先检查：

```text
crop
resize
FOV
principal point
coordinate
```

不是 intensity。

---

# 81. 主要风险 2：Depth Scale 不一致

症状：

```text
灯投影位置差不多正确
但在 3D 中离 scene 太近 / 太远
```

优先检查：

```text
depth normalization
depth scale
mesh scale
```

---

# 82. 主要风险 3：灯尺寸过大 / 过小

症状：

```text
阴影过软
或
阴影过硬
```

先固定：

```text
position
intensity
```

调：

```text
light width / height
```

---

# 83. 主要风险 4：ILE 强度不能直接映射

症状：

```text
灯位置和阴影正确
但整体亮度严重不对
```

不要改 geometry。

直接：

\[
L_i=e^{\alpha_i}c_i^{ILE}
\]

优化 \(\alpha_i\)。

---

# 84. 主要风险 5：Envmap 吞掉 local light

症状：

```text
hybrid joint optimization 后
local lamp intensity → 接近 0

envmap → 出现强峰
```

解决：

```text
降低 envmap resolution
限制 env energy
分阶段 optimize
增加 local prior
```

---

# 85. 主要风险 6：材质吸收灯光

症状：

```text
albedo 出现 shadow
roughness 出现 illumination pattern
```

解决：

```text
前期完全冻结 material
最后才 refine material
增加 material consistency prior
```

---

# 86. 后续可扩展的正式论文版本

prototype 成功后，可以逐渐变成：

\[
I
\rightarrow
G_0,M_0,L_0
\]

其中：

```text
G0 ← MoGe-2
M0 ← MatNet
L0 ← IndoorLightEditing
```

然后：

\[
L=L_{\mathrm{local}}+L_{\mathrm{far}}
\]

通过：

\[
\text{Global}
\rightarrow
\text{Local}
\rightarrow
\text{Joint}
\rightarrow
\text{Material}
\]

进行可微优化。

---

# 87. 后续可替换 IndoorLightEditing

IndoorLightEditing 在最终论文中可以作为：

```text
baseline initializer
```

之后我们可以实现自己的：

```text
2D Light Detector
+
MoGe Geometry
+
3D Emitter Initialization
```

形成：

```text
ILE initialization
vs
Ours geometry-aware initialization
```

这样当前 prototype 的工作不会浪费。

---

# 88. Window Adapter 当前实现与后续优化

当前已经从 ILE 读取：

\[
W_i=
(p_i,n_i,x_i,y_i)
\]

并保留：

```text
sun directional lobe
sky lobe
ground lobe
```

当前 Mitsuba 表示为：

```text
finite window OBJ
+ explicit null BSDF
+ sun SG
+ sky SG
+ ground SG
```

这里必须使用显式 `null` BSDF：删除 `bsdf` 字段会让 Mitsuba 补上默认不透明材质，导致相机看到黑色孔洞。Window 也不能复用 visible lamp 的朝相机 offset，否则会改变物理孔径并破坏与墙面的对齐；实际 scene 和投影诊断均只移动 visible lamp。

该 emitter 已通过 LLVM AD 与 CUDA AD 的有限值、非零梯度 smoke test；新的 direct/PosMLP 固定灯光优化还在 Example1 上通过了 PRB `max_depth=4` 端到端 smoke test。后续不再是“是否接入 window”，而是研究高 concentration sun lobe 的 importance sampling、window-only 消融，以及是否为三组 SG 增加受约束的 radiance refinement。

---

# 89. 后续 Area Light Representation Ablation

未来可以比较：

```text
Point Light
Rectangle Area Light
Mesh Area Light
Gaussian Area Light
```

第一版只需要：

```text
Rectangle
```

或者：

```text
ILE surface mesh
```

---

# 90. 后续正式实验

OpenRooms 上：

```text
Materialist Env
IndoorLightEditing
ILE → Mitsuba
ILE + refinement
Hybrid Ours
```

主要评价：

```text
RGB reconstruction
Light position
Light orientation
Light size
Shadow
Direct shading
Probe relighting
Object insertion
```

---

# 91. 方法最终一句话描述

可以把整个方法概括为：

> We use IndoorLightEditing as a learned explicit-light proposal network to initialize spatially meaningful indoor light sources from a single image, convert these lights into physically based Mitsuba emitters in the Materialist scene, and refine their radiometric parameters through differentiable rendering. The explicit local illumination is further complemented by a constrained global environment component, resulting in a hybrid near-field and far-field illumination representation for single-image inverse rendering.

中文：

> 本方法首先利用 IndoorLightEditing 从单张室内图像中预测显式三维灯源，并将其作为局部近场照明初值；随后将这些灯源经过相机、坐标与尺度统一后转换为 Materialist/Mitsuba 中的物理面积光源，结合 Materialist 恢复的几何与材质进行 PBR 渲染，并通过可微优化校正灯光强度。进一步加入受约束的全局环境光，用于补偿不可见灯源和远场照明，从而形成“显式局部灯光 + 全局环境光”的混合灯光重建框架。

---

# 92. 推荐 Codex 第一阶段任务拆分

不要一次告诉 Codex：

```text
“把 IndoorLightEditing 和 Materialist 合起来。”
```

建议分任务。

---

## Task 1

```text
阅读 IndoorLightEditing inference 代码，
找到 visible lamp 和 invisible lamp 的所有预测输出，
增加一个 exporter，
将灯光中心、axes、RGB、mask、camera FOV、depth normalization scale
导出为 lights.json。
不要修改模型结构。
```

---

## Task 2

```text
在 Materialist 项目中新建 indoorlight_bridge 模块，
实现 lights.json parser 和统一 light dataclass。
暂时不要改渲染逻辑。
```

---

## Task 3

```text
实现 camera coordinate / projection 模块。
要求可以将 JSON 中的 3D lamp center 投影回输入图片，
并生成 debug overlay。
```

---

## Task 4

```text
实现 rectangle area emitter builder。
输入 center、normal、width、height、RGB，
输出 Mitsuba-compatible scene dictionary。
先在简单 plane scene 中测试。
```

---

## Task 5

```text
将 IndoorLightEditing visible lamp 转换为 Mitsuba rectangle area emitter，
加入 Materialist scene。
关闭 envmap，只使用 local lamps 渲染。
```

---

## Task 6

```text
增加每盏灯的 log-intensity 参数 alpha，
令 radiance = exp(alpha) * ILE_RGB，
冻结 position/orientation/size，
使用 Materialist differentiable rendering 优化 alpha。
```

---

## Task 7

```text
加入 constant ambient emitter，
比较 local-only 与 local+ambient。
```

---

## Task 8

```text
加入低分辨率 environment map，
形成 local + global hybrid lighting。
限制 envmap resolution，避免 envmap 吸收全部局部灯光。
```

---

# 93. 推荐第一阶段完成标准

当下面内容全部成功时，可以认为 prototype 第一版完成：

- [ ] IndoorLightEditing 可以导出 `lights.json`
- [ ] visible lamp center 可以正确投影回 mask
- [ ] invisible lamp box 可以在 3D 中正确显示
- [ ] Materialist / Mitsuba 能加载 ILE 灯
- [ ] local-only rendering 可以正常产生局部照明
- [ ] local-only rendering 可以产生 shadow
- [ ] 可以优化每盏灯 intensity
- [ ] 优化后 RGB reconstruction 有改善
- [ ] constant ambient 可以补足背景亮度
- [ ] low-frequency envmap 可以和 local lights 共存
- [ ] local light 不会在 joint optimization 中完全退化到 0
- [ ] 能输出 debug visualization

---

# 94. 最优先实现顺序

最终推荐严格按照：

```text
1. ILE export
2. JSON interface
3. canonical image / camera
4. 3D → 2D projection test
5. depth scale test
6. Mitsuba simple area light test
7. ILE → Materialist local-only render
8. intensity refinement
9. ambient light
10. low-frequency envmap
11. hybrid optimization
12. evaluation
```

其中：

\[
\boxed{
1\sim7
}
\]

是最核心的第一阶段。

---

# 95. 最重要的判断标准

在继续做复杂优化之前，首先检查：

\[
\boxed{
\pi(p_{\mathrm{ILE}})
\approx
\text{lamp mask center}
}
\]

以及：

\[
\boxed{
R(G,M,L_{\mathrm{ILE}})
}
\]

是否可以在正确的位置产生局部 illumination 和 shadow。

如果这两个条件不成立：

```text
不要加新的 loss
不要调学习率
不要加 envmap
不要优化 material
```

先解决：

```text
coordinate
camera
scale
geometry
```

---

# 96. 最终 prototype 形式

第一阶段最终系统应该达到：

```text
Input RGB
    │
    ├── IndoorLightEditing
    │       ↓
    │   Explicit Lamps
    │       ↓
    │   lights.json
    │
    └── Materialist
            ↓
        Geometry + Material
            │
            ├──────── lights.json
            │
            ▼
        Mitsuba Scene
            │
            ▼
        Local Lighting
            │
            ▼
     Intensity Refinement
            │
            ▼
        Ambient / Env
            │
            ▼
      Hybrid Lighting
            │
            ▼
        Final Rendering
```

最终对应数学形式：

\[
\boxed{
L_i^{local}
=
e^{\alpha_i}L_i^{ILE}
}
\]

\[
\boxed{
L
=
\sum_i L_i^{local}
+
L_{\mathrm{far}}
}
\]

并通过：

\[
\boxed{
L^*
=
\arg\min_L
D(R(G,M,L),I)
+
\lambda L_{\mathrm{prior}}
}
\]

完成 refinement。

---

# 97. 总结

这个初步系统的核心不是简单地“把 IndoorLightEditing 和 Materialist 拼接”。

它应该被实现成：

\[
\boxed{
\text{Learned Explicit Lighting Proposal}
+
\text{Physically Based Differentiable Refinement}
}
\]

其中：

- IndoorLightEditing 解决 **灯在哪里、是什么形状、是什么大致颜色/强度**；
- Materialist 解决 **场景材质与几何**；
- Mitsuba 负责 **真实 PBR 光传输**；
- differentiable optimization 负责 **校正灯光参数**；
- constrained envmap 负责 **补偿全局和不可见 illumination**。

历史第一版从 lamp 开始；当前固定灯光路径已经扩展为：

```text
Visible/Invisible Lamp
+
Visible/Invisible Directional Window
+
Coordinate Conversion
+
Mitsuba Differentiable Emitters
+
Far-field HDRI + Material Refinement
```

这是最小、最稳、最容易调试，同时与后续论文完整方向高度一致的实现路线。
