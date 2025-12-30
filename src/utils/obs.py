import numpy as np

class RunningMeanStd:
    def __init__(self, shape=()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    def update(self, x):
        x = np.array(x, dtype=np.float64)
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0] if x.ndim > 1 else 1.0
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * (self.count * batch_count / tot_count)
        new_var = M2 / tot_count
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x, clip=10.0):
        return np.clip((x - self.mean) / (np.sqrt(self.var) + 1e-8), -clip, clip)

class ObsV1:
    def __init__(self, include_cube_quat=True, normalize=False):
        self.include_cube_quat = include_cube_quat
        self.normalize = normalize
        self.running = None

    def _build_flat(self, obs):
        def g(k, default):
            return np.asarray(obs.get(k, default), dtype=np.float32)
        jp_sin = g("robot0_joint_pos_sin", np.zeros(7))
        jp_cos = g("robot0_joint_pos_cos", np.zeros(7))
        jvel   = g("robot0_joint_vel", np.zeros(7))
        eef_pos = g("robot0_eef_pos", np.zeros(3))
        eef_quat = g("robot0_eef_quat", np.zeros(4))
        gripper_qpos = np.asarray([obs.get("robot0_gripper_qpos", 0.0)], dtype=np.float32)
        gripper_qvel = np.asarray([obs.get("robot0_gripper_qvel", 0.0)], dtype=np.float32)
        gripper_to_cube = obs.get("gripper_to_cube_pos", None)
        if gripper_to_cube is None:
            cube_pos = g("cube_pos", np.zeros(3))
            gripper_to_cube = cube_pos - eef_pos
        gripper_to_cube = np.asarray(gripper_to_cube, dtype=np.float32)
        cube_quat = g("cube_quat", np.zeros(4))
        parts = [jp_sin, jp_cos, jvel, eef_pos, eef_quat, gripper_qpos, gripper_qvel, gripper_to_cube]
        if self.include_cube_quat:
            parts.append(cube_quat)
        return np.concatenate(parts).astype(np.float32)

    def flatten(self, obs):
        flat = self._build_flat(obs)
        if self.normalize:
            if self.running is None:
                self.running = RunningMeanStd(shape=flat.shape)
            self.running.update(flat[None, :])
            flat = self.running.normalize(flat)
        return flat

    def dim(self):
        fake = {
            "robot0_joint_pos_sin": np.zeros(7),
            "robot0_joint_pos_cos": np.zeros(7),
            "robot0_joint_vel": np.zeros(7),
            "robot0_eef_pos": np.zeros(3),
            "robot0_eef_quat": np.zeros(4),
            "robot0_gripper_qpos": 0.0,
            "robot0_gripper_qvel": 0.0,
            "gripper_to_cube_pos": np.zeros(3),
            "cube_quat": np.zeros(4),
        }
        return self._build_flat(fake).shape[0]
