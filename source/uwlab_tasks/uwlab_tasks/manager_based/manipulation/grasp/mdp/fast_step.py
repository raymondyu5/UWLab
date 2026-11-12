"""Opt-in speedups for ``ManagerBasedRLEnv.step`` that remove redundant per-substep Python work.

Profiling (scripts/profile/profile_env_step.py, cube task, 512 envs) showed that ``scene.update`` and
``scene.write_data_to_sim`` cost as much as PhysX itself, because they run on every one of the ``decimation``
substeps:

* ContactSensor.update (x7 sensors, ~2.8 ms each over the env step): every consumer in the grasp tasks reads the
  sensor buffers (``sensor._data``) only after the substep loop, so only the last substep's update matters.
* Articulation.write_data_to_sim: the action is constant over the substeps and all actuators are implicit (PD runs
  in PhysX), so rewriting identical joint targets on substeps 2..N is redundant.

``enable_fast_substeps(env)`` patches both on one env instance. Skipped silently where the assumption does not hold
(non-implicit actuators, active external wrenches). History buffers of the patched contact sensors are not used by the
grasp tasks; do not enable this if a task reads ``net_forces_w_history`` / air-time.
"""

from __future__ import annotations

import torch


def enable_fast_substeps(env, contact_sensors: bool = True, robot_write: bool = True) -> dict:
    """Patch ``env.unwrapped`` in place. Returns a dict describing what was patched."""
    from isaaclab.actuators import ImplicitActuator
    from isaaclab.sensors import ContactSensor

    u = env.unwrapped
    decimation = u.cfg.decimation
    patched = {"contact_sensors": [], "robot_write": False}

    if contact_sensors:
        for name, sensor in u.scene.sensors.items():
            if not isinstance(sensor, ContactSensor):
                continue
            orig_update = sensor.update

            def update(dt, force_recompute=False, _orig=orig_update):
                # _sim_step_counter is incremented at the start of every substep, so %decimation==0 is the last one.
                if u._sim_step_counter % decimation == 0:
                    _orig(dt, force_recompute)

            sensor.update = update
            patched["contact_sensors"].append(name)

    if robot_write:
        robot = u.scene.articulations.get("robot") if hasattr(u.scene, "articulations") else None
        if robot is not None and all(isinstance(a, ImplicitActuator) for a in robot.actuators.values()):
            in_step = [False]
            orig_step = u.step
            orig_write = robot.write_data_to_sim

            def step(*a, **k):
                in_step[0] = True
                try:
                    return orig_step(*a, **k)
                finally:
                    in_step[0] = False

            def write_data_to_sim():
                if in_step[0] and decimation > 1 and u._sim_step_counter % decimation != 1:
                    wrench_active = robot._instantaneous_wrench_composer.active or robot._permanent_wrench_composer.active
                    if not wrench_active:
                        return
                orig_write()

            u.step = step
            robot.write_data_to_sim = write_data_to_sim
            patched["robot_write"] = True

    return patched


def enable_skippable_obs(env) -> None:
    """Let a caller skip the observation computation of intermediate env steps.

    ``RFSWrapper`` runs ``residual_step`` env steps per PPO step but only consumes the observation after the last one
    (policy obs, point cloud, resets). Set ``env.unwrapped._skip_obs_compute = True`` before an env.step to have
    ``observation_manager.compute`` return the previously computed observation instead of recomputing it (the point
    cloud alone costs ~2.4 ms/step). Only valid when no observation term keeps per-call state (history buffers,
    delays) -- true for the grasp tasks. Changes the torch RNG stream (point-cloud sampling), so runs are
    statistically, not bitwise, equivalent.
    """
    u = env.unwrapped
    u._skip_obs_compute = False
    om = u.observation_manager
    orig_compute = om.compute
    cache = {}

    def compute(*a, **k):
        if u._skip_obs_compute and "obs" in cache:
            return cache["obs"]
        out = orig_compute(*a, **k)
        cache["obs"] = out
        return out

    om.compute = compute
