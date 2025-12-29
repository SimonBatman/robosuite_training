import robosuite as suite

env = suite.make(
    env_name="Lift",
    robots="Panda",
    use_camera_obs=False,        # ❗ 关键：关相机
    has_renderer=False,          # ❗ 不开窗口
    has_offscreen_renderer=False # ❗ 不离屏渲染
)

obs = env.reset()
print("Lift env reset OK")
print("Observation keys:", obs.keys())
