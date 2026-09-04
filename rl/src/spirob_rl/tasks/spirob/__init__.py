"""Spirob task family: three goal formulations on one tendon-driven tentacle.

* **Reach** -- hold the TCP at a static random target.
* **Shape** -- hit a TCP target *and* a mid-chain target, i.e. command the whole
  posture. Both come from one forward-kinematics pose, so the pair is always
  achievable.
* **Trajectory** -- follow a target that sweeps continuously along the reachable
  arc, with preview points.
* **Wrap** -- coil around a randomly placed, randomly sized cylinder that spawns
  on one side of the spirob. The reset is biased to lean toward that same side
  (see ``wrap_env_cfg.py``).

Each is crossed with the sensor-ablation ladder (see ``SensorLevel`` in
``base_env_cfg.py``) and gets a ``-DrPlay`` twin that replays a trained policy
under full-width domain randomization. Shared scaffolding lives in
``base_env_cfg.py`` and ``mdp/``; each variant file holds only its command and
reward terms.
"""

from mjlab.tasks.registry import register_mjlab_task

from .reach_env_cfg import reach_env_cfg, reach_ppo_runner_cfg
from .shape_env_cfg import shape_env_cfg, shape_ppo_runner_cfg
from .trajectory_env_cfg import trajectory_env_cfg, trajectory_ppo_runner_cfg
from .wrap_env_cfg import wrap_env_cfg, wrap_ppo_runner_cfg

# Sensor ablation: one task per level of actor observability. The critic always
# sees the full state, so differences between these runs isolate how much the
# actor's sensor suite matters.
_SENSOR_LEVELS = {
  "Force": "force",
  "": "tendon",
  "Imu": "imu",
  "Joints": "joints",
  "Oracle": "oracle",
}

# Task-id prefix and the config factories per variant. The reach prefix keeps
# its original "Tcp-Reach" spelling so existing checkpoints and commands in the
# README keep working after the task-family split.
_VARIANTS = {
  "RlExplor-Spirob-Tcp-Reach": (reach_env_cfg, reach_ppo_runner_cfg),
  "RlExplor-Spirob-Shape": (shape_env_cfg, shape_ppo_runner_cfg),
  "RlExplor-Spirob-Trajectory": (trajectory_env_cfg, trajectory_ppo_runner_cfg),
  "RlExplor-Spirob-Wrap": (wrap_env_cfg, wrap_ppo_runner_cfg),
}

for _prefix, (_env_cfg_fn, _rl_cfg_fn) in _VARIANTS.items():
  for _suffix, _level in _SENSOR_LEVELS.items():
    _task_id = f"{_prefix}-{_suffix}" if _suffix else _prefix
    register_mjlab_task(
      task_id=_task_id,
      env_cfg=_env_cfg_fn(sensor_level=_level),
      play_env_cfg=_env_cfg_fn(play=True, sensor_level=_level),
      rl_cfg=_rl_cfg_fn(sensor_level=_level),
    )
    # DR-play twin: same policy and checkpoints, but play runs under full-width
    # domain randomization so the behavior it produces can be watched.
    register_mjlab_task(
      task_id=f"{_task_id}-DrPlay",
      env_cfg=_env_cfg_fn(sensor_level=_level),
      play_env_cfg=_env_cfg_fn(play=True, sensor_level=_level, dr_in_play=True),
      rl_cfg=_rl_cfg_fn(sensor_level=_level),
    )
