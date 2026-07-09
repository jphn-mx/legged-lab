# IK Config 参数标定方法

IK config JSON 文件位于 `general_motion_retargeting/ik_configs/`，每个文件对应一个机器人。

---

## 参数结构

```json
{
    "robot_root_name": "base",
    "human_root_name": "pelvis",
    "human_height_assumption": 1.8,
    "human_scale_table": { ... },
    "ik_match_table1": {
        "robot_link_name": ["human_joint_name", pos_weight, rot_weight, [pos_offset], [rot_offset]]
    },
    "ik_match_table2": { ... }
}
```

---

## 一、human_scale_table

### 原理

`scale_human_data()` 的缩放公式（见 `motion_retarget.py:259`）：

```
target_Z = scale_joint × (joint_Z - pelvis_Z) + scale_pelvis × pelvis_Z
```

- **pelvis** 直接等比缩放世界坐标，控制机器人站立高度
- **其他关节** 缩放的是相对 pelvis 的距离，不是绝对坐标

### 推导步骤

**Step 1：从运动数据读取人体关节高度**

选站立帧（第0帧），运行：

```python
import torch, smplx, numpy as np
from smplx.joint_names import JOINT_NAMES

body = smplx.create('assets/body_models', 'smplx', gender='neutral', use_pca=False)
data = np.load('motion_data/ACCAD/.../B3_-_walk1_stageii.npz', allow_pickle=True)
nb = body.num_betas

out = body(
    betas=torch.tensor(data['betas']).float().view(1,-1)[:, :nb],
    global_orient=torch.tensor(data['root_orient'][[0]]).float(),
    body_pose=torch.tensor(data['pose_body'][[0]]).float(),
    transl=torch.tensor(data['trans'][[0]]).float(),
)
j = out.joints[0].detach().numpy()

for name in ['pelvis', 'left_hip', 'left_knee', 'left_ankle']:
    idx = JOINT_NAMES.index(name)
    print(f'{name}: Z={j[idx,2]:.4f}m  Y={j[idx,1]:.4f}m')
```

归一化到 `human_height_assumption`（默认1.8m）：

```python
h_actual = 1.66 + 0.1 * data['betas'][0]  # 估算实际身高
ratio = h_actual / 1.8
pelvis_Z_ref = pelvis_Z / ratio
knee_Z_ref   = knee_Z   / ratio
ankle_Z_ref  = ankle_Z  / ratio
```

**Step 2：从 XML 累加机器人关节高度**

沿运动链累加 `pos` 的 Z 分量，得到各 link 距地面高度。
地面高度 = 足端 link 到 base 的 Z 累加取反。

示例（A1 V2）：

```
base        = 0.455m  (由足端反推)
Link_R4     = 0.277m  (膝关节)
Link_R6     = 0.000m  (足端，定义为地面)
```

**Step 3：反解 scale**

```python
scale_pelvis = robot_base_Z / pelvis_Z_ref

scale_knee = (robot_knee_Z - scale_pelvis * pelvis_Z_ref) / (knee_Z_ref - pelvis_Z_ref)

scale_foot = (0 - scale_pelvis * pelvis_Z_ref) / (ankle_Z_ref - pelvis_Z_ref)
```

**human_height_assumption 的作用**：运行时代码会计算 `ratio = actual_height / assumption`，并将所有 scale 乘以该 ratio，自动适配不同身高的数据文件。因此 config 里的 scale 只需对参考身高（1.8m）标定一次。

---

## 二、pos_offset

### 原理

scale 只修正 Z 方向（高度）。Y 方向（左右宽度）上，机器人腿的间距与人体不同，需要 pos_offset 补偿。

代码 `motion_retarget.py:278-282`：pos_offset 在旋转后的局部坐标系下表达，再转到全局坐标。

### 推导

**Step 1：从 XML 累加机器人 link 的 Y 偏移（相对 base）**

```
Link_R4 的 Y 偏移 = -0.021 - 0.06025 - 0.0987 = -0.180m
```

**Step 2：从运动数据读取人体对应关节的 Y 偏移（相对 pelvis），乘以 scale**

```python
human_knee_Y_relative = knee_Y - pelvis_Y  # 约 -0.107m (右腿)
scaled_human_knee_Y   = human_knee_Y_relative * scale_knee  # 约 -0.042m
```

**Step 3：差值即为 pos_offset 的 Y 分量**

```
pos_offset_Y = robot_link_Y - scaled_human_joint_Y
             = -0.180 - (-0.042) = -0.138m
```

实践中取近似整数（如 -0.08），IK solver 会容忍剩余误差。

---

## 三、rot_offset

### 原理

编码 SMPL-X 关节坐标系到 MuJoCo body 坐标系的旋转差。格式为四元数 `[w, x, y, z]`。

代码 `motion_retarget.py:275`：
```python
updated_quat = (R.from_quat(human_quat) * rot_offset).as_quat()
```

### 如何确定

- **不能设为 identity `[1,0,0,0]`**：SMPL-X 和 MuJoCo 的关节轴向不同，identity 会导致姿态完全错误。
- **复用同关节结构的机器人**：若新机器人的关节轴向（XML 中 `<joint axis=...>`）与已有机器人相同，直接复用其 rot_offset。
- **严格推导**：在静止站立姿态下，令 `rot_smplx * rot_offset = rot_robot_body`，反解 `rot_offset = rot_smplx⁻¹ * rot_robot_body`。

---

## 四、pos_weight / rot_weight

IK 优化目标的权重，数值越大该约束越强。

| 典型设置 | 含义 |
|---------|------|
| `pos=100, rot=10` | 位置强约束，方向弱约束（base、foot） |
| `pos=0, rot=10` | 只约束方向，不约束位置（hip，允许 IK 自由决定位置） |
| `pos=10, rot=5` | table2 精解阶段，松一些 |
| `rot=50` | foot 在 table2 中强制脚底朝下 |

---

## 五、两阶段 IK（table1 / table2）

- **table1**：粗解。重点约束 base 高度和 foot 位置，hip/knee 只约束方向。
- **table2**：精解。在 table1 结果基础上，加入 foot 朝向约束（强制脚底朝下），放宽 hip/knee 权重。

两阶段分离的原因：同时约束所有项时，脚底朝向和膝盖位置容易冲突，分阶段求解更稳定。
