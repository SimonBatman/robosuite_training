# 环境安装
- Python: **3.9.x**（conda 环境）
- Robosuite: **1.4.0**
- MuJoCo: 安装自 **conda-forge**（建议 `mujoco 2.3.x`）
- NumPy: **< 2.0**（建议 `1.26.x`）
- PyTorch: **可用 GPU**：推荐 `pytorch + pytorch-cuda=11.8`（或 pip cu118 wheel）  
  或 **CPU**：pip/conda CPU 版本
- OpenCV: **opencv-python-headless**
- 其他：`termcolor, imageio, imageio-ffmpeg, stable-baselines3, gym>=0.21, tensorboard`

## 2. 推荐的 conda 环境（一键方案）
> 推荐用于服务器 / GPU 实验（conda 管理依赖，兼容性强）

**environment.yml**

```yaml
name: robosuite
channels:
  - conda-forge
  - pytorch
  - nvidia
  - defaults
dependencies:
  - python=3.9
  - numpy<2.0
  - mujoco              # from conda-forge
  - robosuite=1.4.0     # installed via pip later or pip install robosuite==1.4.0
  - pip
  - pip:
      - robosuite==1.4.0
      - opencv-python-headless
      - termcolor
      - imageio
      - imageio-ffmpeg
      - stable-baselines3[extra]
      - gym>=0.21
      - tensorboard
```
>（可选）安装 intel-openmp（若出现 iJIT_NotifyEvent 或 OpenMP 符号问题）
conda install -y -c intel intel-openmp

# robosuite_training

目的：在 Robosuite（Panda / Lift 等单臂操作任务）上做论文级的机器人 RL 研究。  
本仓库遵循“实验可复现、MDP 明确”的原则，统一把环境原生返回的 observation 处理成 **`obs_v1`**（实验冻结版），并把 `obs_v1` 作为 baseline 与后续方法的一致接口。

---

## obs_v1（论文级规范）

### 设计原则（简要）
1. **明确定义 MDP 的观测空间**：论文/实验中所有 baseline 和方法使用同一版 `obs_v1`（除非作为消融实验）。  
2. **避免 shortcut / privileged 信息**：除非明确说明，禁止使用只在仿真/数据集里可得但现实不可得的“object-state”或内部量。  
3. **便于复现**：输出 flat 的 1D numpy 向量，包含字段、维度与归一化方式的明确说明。  
4. **学术友好**：角度用 sin/cos 表示、坐标尽量用相对坐标（提高泛化）、对尺度做归一化或以 workspace 尺度缩放。

---

### obs_v1 字段（建议的标准版 `obs_v1`）
（下面字段在 Robosuite `Lift` 的原生 obs 中通常可得；字段名为 flat 向量按顺序排列）

1. **Joint angles**：`robot0_joint_pos_sin` (D_j), `robot0_joint_pos_cos` (D_j)  
   - 使用 sin/cos 避免角度跳变与周向不连续。  
2. **Joint velocities**：`robot0_joint_vel` (D_j)  
3. **End-effector position**：`robot0_eef_pos` (3) — *world frame* 或者 **相对物体**（推荐转成相对坐标，见下）  
4. **End-effector orientation**：`robot0_eef_quat` (4) — 若不需要姿态可省略或改用 `eef_rpy`  
5. **Gripper state**：`robot0_gripper_qpos` (1) ，`robot0_gripper_qvel` (1)  
6. **Object relative position**：`gripper_to_cube_pos` (3)  *（推荐）* —— 把 object 的位置编码为相对于 EEF 的向量，以提升平移不变性  
7. **Object orientation (option)**：`cube_quat` (4) — 可选（若任务对朝向敏感）  
8. **Proprioceptive summary**：`robot0_proprio-state` （如果 robosuite 提供）或手工构造的关节 summary（可选）  
9. **不包含 / 禁用**：`object-state`（若包含仿真内部信息如 velocity/contacts，不建议作为默认输入，除非你的论文明确讨论 privileged info）

> **维度示例（Panda 7-DoF）**：  
> joint pos sin/cos: 7+7 = 14  
> joint vel: 7  
> eef pos: 3  
> eef quat: 4  
> gripper qpos/qvel: 1+1 = 2  
> gripper_to_cube_pos: 3  
> total ≈ 34（不含 object quat）

---

### 字段变换 / 归一化（必须明确）
- **角度**：已用 `sin`/`cos` 表示，无需额外归一化。  
- **位移（位置）**：按 workspace 半径或已知边界 `([min,max])` 做线性缩放至 `[-1,1]` 或按训练集均值/标准差做标准化。论文中需写明 normalization 参数与训练/测试时的统一方式。  
- **速度 / 角速度 / qvel**：按经验做 clip（如 `[-10,10]`）再按最大绝对值缩放到 `[-1,1]`，或者按训练集标准差归一化。  
- **四元数**：保证归一化（`q / ||q||`），或转换为 Euler / 旋转矩阵（注意分片不连续性）。  
- **缺失值**：若 observation 在某些时刻缺失（通常不会），请用 0 填充并记录 mask（但标准 `Lift` 不应出现）。

---

### 是否允许 privileged 信息（论文写法）
- 默认 **不允许** 使用 `object-state`（如果它包含仿真私有量）。  
- 若你的方法需要 privileged info（例如 Oracle guider），必须明确写入论文：**在哪些实验中**使用了 privileged 信息，并单独给出 *non-privileged* 对照实验。

---

## flatten_obs 示例代码（Python）

```python
import numpy as np

def flatten_obs(obs):
    # 假设 obs 是 robosuite 返回的 dict（如 obs from Lift）
    # 1) joint sin/cos
    jp_sin = obs["robot0_joint_pos_sin"]    # shape (7,)
    jp_cos = obs["robot0_joint_pos_cos"]    # shape (7,)
    jvel   = obs["robot0_joint_vel"]        # shape (7,)
    eef_pos = obs["robot0_eef_pos"]         # shape (3,)
    eef_quat = obs["robot0_eef_quat"]       # shape (4,)
    gripper_qpos = np.array([obs["robot0_gripper_qpos"]])  # shape (1,)
    gripper_qvel = np.array([obs["robot0_gripper_qvel"]])  # shape (1,)

    # 推荐使用相对坐标（相对于 eef），若已有 gripper_to_cube_pos，可直接用
    gripper_to_cube = obs.get("gripper_to_cube_pos", None)
    if gripper_to_cube is None:
        # 退化方案： cube_pos - eef_pos
        cube_pos = obs.get("cube_pos", np.zeros(3))
        gripper_to_cube = cube_pos - eef_pos

    # optional object orientation
    cube_quat = obs.get("cube_quat", np.zeros(4))

    # concat in defined order
    flat = np.concatenate([
        jp_sin, jp_cos, jvel,
        eef_pos, eef_quat,
        gripper_qpos, gripper_qvel,
        gripper_to_cube,
        cube_quat
    ], axis=0).astype(np.float32)

    # Example normalization (user should replace with dataset-level normalization if needed)
    # workspace_scale = 1.0
    # flat = np.clip(flat/ workspace_scale, -1.0, 1.0)
    return flat
