from __future__ import annotations

import torch
import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors.contact_sensor import ContactSensorCfg

# Bottle cap offset in local -X frame: 13.22cm (cap is at X=-0.132 in mesh frame)
BOTTLE_CAP_OFFSET = (-0.132179, 0.0, 0.0)

# Success: cap tip XY within 4cm of cup center, z in [cup_z+0.20, cup_z+0.30]
# i.e. 5cm to 15cm above the cup rim (~0.15m tall).
NEAR_MISS_XY_RADIUS = 0.15
SUCCESS_XY_RADIUS = 0.04
GRASPED_Z_THRESHOLD = 0.05
HEALTHY_Z_RANGE = (0.17, 0.3)

MAX_XY_DIST = 0.60

def compute_tip_pos(env):
    bottle = env.scene["grasp_object"]
    cup = env.scene["pink_cup"]
    bottle_pos = bottle.data.root_pos_w - env.scene.env_origins
    bottle_quat = bottle.data.root_quat_w
    cup_pos = cup.data.root_pos_w - env.scene.env_origins
    cap_offset = torch.tensor(list(BOTTLE_CAP_OFFSET), device=env.device)
    w = bottle_quat[:, 0:1]
    q = bottle_quat[:, 1:]
    t = 2.0 * torch.linalg.cross(q, cap_offset.unsqueeze(0).expand_as(q))
    tip_pos = bottle_pos + cap_offset.unsqueeze(0) + w * t + torch.linalg.cross(q, t)
    return bottle_pos, tip_pos, cup_pos

def is_grasped(env) -> torch.Tensor:
    bottle = env.scene["grasp_object"]
    bottle_pos = bottle.data.root_pos_w - env.scene.env_origins
    return bottle_pos[:, 2] > GRASPED_Z_THRESHOLD

def is_healthy_z(env) -> torch.Tensor:
    bottle_pos, tip_pos, cup_pos = compute_tip_pos(env)
    return (tip_pos[:, 2] >= cup_pos[:, 2] + HEALTHY_Z_RANGE[0]) & (tip_pos[:, 2] <= cup_pos[:, 2] + HEALTHY_Z_RANGE[1])

def is_near_miss(env) -> torch.Tensor:
    bottle_pos, tip_pos, cup_pos = compute_tip_pos(env)
    xy_dist = torch.norm(tip_pos[:, :2] - cup_pos[:, :2], dim=1)
    is_near_cup = (xy_dist < NEAR_MISS_XY_RADIUS)
    healthy_z = is_healthy_z(env)   
    return healthy_z & is_near_cup

def is_success(env) -> torch.Tensor:
    bottle_pos, tip_pos, cup_pos = compute_tip_pos(env)
    xy_dist = torch.norm(tip_pos[:, :2] - cup_pos[:, :2], dim=1)
    is_near_cup = (xy_dist < SUCCESS_XY_RADIUS)
    healthy_z = is_healthy_z(env)
    return healthy_z & is_near_cup

def _cup_toppled(env, cup_name: str = "pink_cup", angle_thresh_rad: float = 0.524) -> torch.Tensor:
    cup = env.scene[cup_name]
    current_quat = cup.data.root_quat_w
    spawn_quat = torch.tensor(
        [[0.707, 0.707, 0.0, 0.0]], device=env.device, dtype=current_quat.dtype
    ).expand(env.num_envs, -1)
    dot = (spawn_quat * current_quat).sum(dim=1)
    current_quat = torch.where(dot.unsqueeze(1) < 0, -current_quat, current_quat)
    q_rel = math_utils.quat_mul(math_utils.quat_conjugate(spawn_quat), current_quat)
    rotation_angle = 2.0 * torch.atan2(
        torch.norm(q_rel[:, 1:4], dim=1),
        torch.abs(q_rel[:, 0]),
    )
    return rotation_angle > angle_thresh_rad

def _joint_vel_l2(env, asset_name: str) -> torch.Tensor:
    robot = env.scene[asset_name]
    return torch.sum(robot.data.joint_vel ** 2, dim=1)

def _joint_pos_limits(env, asset_name: str, soft_ratio: float = 0.9) -> torch.Tensor:
    robot = env.scene[asset_name]
    joint_pos = robot.data.joint_pos
    lower = robot.data.soft_joint_pos_limits[:, :, 0]
    upper = robot.data.soft_joint_pos_limits[:, :, 1]
    lower_violation = torch.clamp(lower * soft_ratio - joint_pos, min=0.0)
    upper_violation = torch.clamp(joint_pos - upper * soft_ratio, min=0.0)
    return torch.sum(lower_violation + upper_violation, dim=1)

def _action_rate_l2(env) -> torch.Tensor:
    return torch.sum((env.action_manager.action - env.action_manager.prev_action) ** 2, dim=1)

class SimplePourReward:
    """
    Simple reward class for bourbon pouring task. 

    It should be
    
    - sparse reward for grasping the bottle (is_grasped)
    - sparse reward for lifting the bottle to a healthy z position (is_healthy_z)
    - if is_healthy_z: then add dense reward for xy euclidean distance of tip to cup center (raw distance tends to be between .4 and 0)
    - is_success: sparse reward for is_success

    - joint_vel_penalty: dense penalty for joint velocity
    - joint_limit_penalty: dense penalty for joint position limits
    - action_rate_penalty: dense penalty for action rate
    - cup_topple_penalty: sparse penalty for cup_toppled
    """

    def __init__(self, asset_name: str = "robot", cup_name: str = "pink_cup"):
        self.asset_name = asset_name
        self.cup_name = cup_name

    def pour_rewards(self, env):
        _, tip_position, cup_pos = compute_tip_pos(env)
        xy_dist = torch.norm(tip_position[:, :2] - cup_pos[:, :2], dim=1)

        grasped = is_grasped(env).float()
        healthy_z = is_healthy_z(env).float()
        success = is_success(env).float()
        near_miss = is_near_miss(env).float()

        xy_reward = torch.clamp(1.0 - xy_dist / MAX_XY_DIST, min=0.0, max=1.0) 
        cup_topple_penalty = _cup_toppled(env, cup_name=self.cup_name).float()

        joint_vel_penalty = _joint_vel_l2(env, asset_name=self.asset_name)
        joint_limit_penalty = _joint_pos_limits(env, asset_name=self.asset_name)
        action_rate_penalty = _action_rate_l2(env)

        # reward bounded above by 1 + 2 + 5 + 10 = 18
        final = (
            1.0 * grasped
            + 2.0 * xy_reward * healthy_z # Dense XY shaping only once the tip is in a valid pouring-height band.
            + 5.0 * xy_reward * near_miss 
            + 10.0 * success
            - 10.0 * cup_topple_penalty
            - joint_vel_penalty * 1.0e-3
            - joint_limit_penalty * 6.0e-1
            - action_rate_penalty * 5e-3
        )

        return torch.nan_to_num(final, nan=0.0, posinf=0.0, neginf=0.0)

class PourReward:
    """Reward class for bourbon pouring task.

    Standalone port of SingleHandPourRew from IsaacLab's bimanual_franka_pour_rew.py.

    Changes from IsaacLab version:
    - No bimanual wrapper — single hand only, no hand_side prefix
    - No init_robot_pose offset (always zeros in UWLab)
    - Reset references stored as instance attrs (not env.__dict__)
    - reset_init_height computed once at reset, not every step
    - Single get_target_object_pose with cup_top_z + 0.03 (reward-side offset)
    - No _shared_tensors class-var hack
    - Contact/finger sensor prim paths use UWLab conventions (Robot/right_hand/*)
    """

    _FINGER_SENSOR_LINK = {
        "palm_lower": "palm_lower",
        "fingertip": "fingertip_sensor",
        "thumb_fingertip": "thumb_sensor",
        "fingertip_2": "fingertip_2_sensor",
        "fingertip_3": "fingertip_3_sensor",
    }

    def __init__(
        self,
        asset_name: str,
        object_name: str,
        cup_name: str,
        fingers_name_list: list,
        init_height: float,
        bottle_cap_offset: tuple,
    ):
        self.asset_name = asset_name
        self.object_name = object_name
        self.cup_name = cup_name
        self.fingers_name_list = fingers_name_list
        self.init_height = init_height
        self.bottle_cap_offset = torch.tensor(bottle_cap_offset, dtype=torch.float32)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Reward scale per finger: [palm_lower, thumb_fingertip, fingertip, fingertip_2, fingertip_3]
        self.finger_reward_scale = torch.as_tensor([0.5, 1.0, 2, 1.5, 1.0]).to(
            self.device).unsqueeze(0)

        self._component_sums = {}
        self._component_count = 0

        # Reset references — populated by capture_reset_references event, None until first reset
        self.cup_reset_pos_ref = None    # (num_envs, 3)
        self.cup_reset_quat_ref = None   # (num_envs, 4)
        self.default_bottle_quat = None  # (num_envs, 4)
        self.reset_init_height = None    # (num_envs,) — actual env-local bottle Z at reset

        # Shared state updated by pour_rewards each step
        self.object_pose = None
        self.cup_pose = None
        self.cup_center_xy = None
        self.cup_top_z = None
        self.contact_or_not = None
        self.finger_object_dev = None
        self.finger_pose = None
        self.tip_pos = None

    # ------------------------------------------------------------------
    # Scene setup — called in task __post_init__
    # ------------------------------------------------------------------

    def setup_wrist_sensor(self, scene_cfg):
        from isaaclab.sensors.contact_sensor import ContactSensorCfg
        wrist_sensor = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link6",
            update_period=0.0,
            history_length=3,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Table"],
            debug_vis=False,
        )
        setattr(scene_cfg, "panda_link6_contact", wrist_sensor)

    def setup_finger_entities(self, scene_cfg):
        from isaaclab.assets import RigidObjectCfg
        for link_name in self.fingers_name_list:
            rigid_cfg = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Robot/right_hand/" + link_name,
                spawn=None,
            )
            setattr(scene_cfg, link_name, rigid_cfg)

    def setup_finger_sensors(self, scene_cfg, object_prim_name: str = "GraspObject"):
        from isaaclab.sensors.contact_sensor import ContactSensorCfg
        filter_expr = ["{ENV_REGEX_NS}/" + object_prim_name]
        for link_name in self.fingers_name_list:
            sensor_link = self._FINGER_SENSOR_LINK.get(link_name, link_name)
            sensor = ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/Robot/right_hand/" + sensor_link,
                update_period=0.0,
                history_length=3,
                filter_prim_paths_expr=filter_expr,
                debug_vis=False,
            )
            setattr(scene_cfg, f"{link_name}_contact", sensor)

    # ------------------------------------------------------------------
    # Reset reference capture — register as EventTerm(mode="reset")
    # Must run AFTER reset_object and reset_cup_object events.
    # ------------------------------------------------------------------

    def capture_reset_references(self, env, env_ids):
        """Capture cup and bottle orientations immediately after reset.

        Stored as instance attributes so termination functions can read them
        without touching env.__dict__.
        """
        num_envs = env.num_envs
        device = env.device

        if self.cup_reset_pos_ref is None:
            self.cup_reset_pos_ref = torch.zeros(num_envs, 3, device=device)
            self.cup_reset_quat_ref = torch.zeros(num_envs, 4, device=device)
            self.default_bottle_quat = torch.zeros(num_envs, 4, device=device)
            self.reset_init_height = torch.zeros(num_envs, device=device)

        cup_state = env.scene[self.cup_name]._data.root_state_w[env_ids, :7].clone()
        cup_state[:, :3] -= env.scene.env_origins[env_ids]
        self.cup_reset_pos_ref[env_ids] = cup_state[:, :3]
        self.cup_reset_quat_ref[env_ids] = cup_state[:, 3:7]

        bottle_state = env.scene[self.object_name]._data.root_state_w[env_ids, :7].clone()
        self.default_bottle_quat[env_ids] = bottle_state[:, 3:7]

        # Actual env-local z of bottle at reset (world z minus env origin z)
        bottle_z_local = bottle_state[:, 2] - env.scene.env_origins[env_ids, 2]
        self.reset_init_height[env_ids] = bottle_z_local

    # ------------------------------------------------------------------
    # Main reward
    # ------------------------------------------------------------------

    def pour_rewards(self, env):
        self.get_object_info(env)
        self.get_cup_info(env)
        self.get_finger_info(env)
        self.get_contact_info(env)

        r_approach = self.finger2object_rewards(env)
        r_contact = self.object2fingercontact_rewards(env)
        r_lift = self.lift_to_intermediate_rewards(env)
        r_cap = self.cap_to_target_rewards(env)
        r_orientation = self.pour_orientation_rewards(env)
        r_link6 = self.penalty_contact(env)
        r_cup_topple = self.cup_topple_penalty(env)
        r_cap_proximity = self.cap_proximity_penalty(env)

        joint_vel_penalty = self._joint_vel_l2(env)
        joint_limit_penalty = self._joint_pos_limits(env)
        action_rate_penalty = self._action_rate_l2(env)

        final = (r_approach * 0.30
                 + r_contact * 0.3
                 + r_lift * 0.3
                 + r_cap * 1.0
                 + r_orientation * 2.0
                 + r_link6
                 + r_cup_topple
                 - r_cap_proximity * 0.2
                 - joint_vel_penalty * 1.0e-3
                 - joint_limit_penalty * 6.0e-1
                 - action_rate_penalty * 5e-3)

        components = {
            "approach": r_approach.mean().item(),
            "contact": r_contact.mean().item(),
            "lift": r_lift.mean().item(),
            "cap": r_cap.mean().item(),
            "orientation": r_orientation.mean().item(),
            "link6_penalty": r_link6.mean().item(),
            "cup_topple_penalty": r_cup_topple.mean().item(),
            "cap_proximity_penalty": r_cap_proximity.mean().item(),
            "joint_vel_penalty": joint_vel_penalty.mean().item(),
            "joint_limit_penalty": joint_limit_penalty.mean().item(),
            "action_rate_penalty": action_rate_penalty.mean().item(),
        }
        for k, v in components.items():
            self._component_sums[k] = self._component_sums.get(k, 0.0) + v
        self._component_count += 1

        return torch.nan_to_num(final, nan=0.0, posinf=0.0, neginf=0.0)

    # ------------------------------------------------------------------
    # Reward components
    # ------------------------------------------------------------------

    def finger2object_rewards(self, env):
        finger_dist = torch.clip(
            torch.linalg.norm(self.finger_object_dev, dim=2), 0.02, 0.8)
        reward = torch.clip((1.0 / (0.1 + finger_dist)) - 2.0, 0.0, 4.5)
        reward[:, :-2] *= 2.4 / 3
        reward[:, -2:] *= 1.3
        reward = torch.sum(reward, dim=1) / reward.shape[1]
        return reward

    def object2fingercontact_rewards(self, env):
        finger_rewards_scale = (torch.sum(self.contact_or_not, dim=1) >= 2).int()
        return torch.sum(self.contact_or_not.to(torch.float32), dim=1) + finger_rewards_scale * 1.0

    def lift_to_intermediate_rewards(self, env):
        """Reward lifting cap toward intermediate height (5cm above cup rim), gated by >=3 finger contacts."""
        gate = (torch.sum(self.contact_or_not, dim=1) >= 3).float()
        self._compute_tip_pos()
        target_z = self.cup_top_z + 0.05
        dist_z = torch.abs(self.tip_pos[:, 2] - target_z)
        reward = torch.clip(1 - dist_z / 0.5, 0.0, 1.0) * 10
        return reward * gate

    def cap_to_target_rewards(self, env):
        """Reward cap tip in XY circle (4cm) and Z band [cup_z+20cm, cup_z+30cm]. Matches is_success."""
        self._compute_tip_pos()

        z_band_low = self.cup_pose[:, 2] + 0.20
        z_band_high = self.cup_pose[:, 2] + 0.30

        xy_dist = torch.linalg.norm(self.tip_pos[:, :2] - self.cup_center_xy, dim=1)
        xy_dist = torch.nan_to_num(xy_dist, nan=10.0, posinf=10.0, neginf=10.0)
        xy_reward = torch.clip(1.0 - xy_dist / 0.04, 0.0, 1.0)

        tip_z = self.tip_pos[:, 2]
        z_below = tip_z < z_band_low
        z_above = tip_z > z_band_high
        z_reward = torch.ones(env.num_envs, device=env.device, dtype=torch.float32)
        z_reward[z_below] = torch.clip(1.0 - (z_band_low[z_below] - tip_z[z_below]) / 0.1, 0.0, 1.0)
        z_reward[z_above] = torch.clip(1.0 - (tip_z[z_above] - z_band_high[z_above]) / 0.1, 0.0, 1.0)

        return (xy_reward * z_reward) * 20

    def pour_orientation_rewards(self, env):
        """Reward keeping bottle at spawn orientation. Gated by 5cm lift."""
        ref_quat = env.scene[self.object_name].data.default_root_state[:, 3:7]
        delta_quat = math_utils.quat_mul(
            self.object_pose[:, 3:7],
            math_utils.quat_inv(ref_quat))
        axis_angle = math_utils.axis_angle_from_quat(delta_quat)
        rotation_magnitude = torch.linalg.norm(axis_angle, dim=1)
        rotation_magnitude = torch.nan_to_num(rotation_magnitude, nan=1.0, posinf=1.0, neginf=0.0)
        return torch.clip(1 - rotation_magnitude / 0.5, 0.0, 1.0) * 5

    def cap_proximity_penalty(self, env):
        self._compute_tip_pos()
        cap_pos = self.tip_pos.unsqueeze(1)
        dist = torch.linalg.norm(self.finger_pose[..., :3] - cap_pos, dim=2)
        proximity = torch.clip(1.0 - dist / 0.04, 0.0, 1.0)
        return proximity.sum(dim=1)

    def cup_topple_penalty(self, env):
        """Penalize knocking over the cup. Uses fixed spawn quat (matches cup_toppled termination)."""
        current_quat = self.cup_pose[:, 3:7]
        spawn_quat = torch.tensor(
            [[0.707, 0.707, 0.0, 0.0]], device=env.device, dtype=current_quat.dtype
        ).expand(env.num_envs, -1)
        dot = (spawn_quat * current_quat).sum(dim=1)
        current_quat_corrected = torch.where(dot.unsqueeze(1) < 0, -current_quat, current_quat)
        q_rel = math_utils.quat_mul(math_utils.quat_conjugate(spawn_quat), current_quat_corrected)
        rotation_angle = 2.0 * torch.atan2(
            torch.norm(q_rel[:, 1:4], dim=1),
            torch.abs(q_rel[:, 0]),
        )
        return (rotation_angle > 0.524).float() * -1.0

    def penalty_contact(self, env):
        """Penalty for wrist (link6) contact with table."""
        sensor = env.scene["panda_link6_contact"]
        force_data = torch.linalg.norm(
            sensor._data.net_forces_w.reshape(env.num_envs, 3), dim=1).unsqueeze(1)
        return torch.clip((force_data > 4).int() * 2 - 1, 0.0, 1.0).reshape(-1) * -0.5

    # ------------------------------------------------------------------
    # State reading
    # ------------------------------------------------------------------

    def get_object_info(self, env):
        self.object_pose = env.scene[self.object_name]._data.root_state_w[:, :7].clone()
        self.object_pose[:, :3] -= env.scene.env_origins

    def get_cup_info(self, env):
        self.cup_pose = env.scene[self.cup_name]._data.root_state_w[:, :7].clone()
        self.cup_pose[:, :3] -= env.scene.env_origins
        self.cup_center_xy = self.cup_pose[:, :2]
        self.cup_top_z = self.cup_pose[:, 2] + 0.15  # cup is ~15cm tall, origin at bottom

    def get_finger_info(self, env):
        self.finger_pose = []
        for name in self.fingers_name_list:
            finger = env.scene[name]
            finger_pose = finger._data.root_state_w[:, :7].clone()
            finger_pose[:, :3] -= env.scene.env_origins
            self.finger_pose.append(finger_pose.unsqueeze(1))
        self.finger_pose = torch.cat(self.finger_pose, dim=1)

        body_offset = torch.tensor([0.06, 0.0, 0.0], device=self.object_pose.device).unsqueeze(0).expand(self.object_pose.shape[0], -1)
        body_offset_world = math_utils.quat_apply(self.object_pose[:, 3:7], body_offset)
        grasp_target_pos = self.object_pose[:, :3] + body_offset_world

        grasp_target_expanded = grasp_target_pos.unsqueeze(1).repeat_interleave(
            len(self.fingers_name_list), dim=1)
        self.finger_object_dev = grasp_target_expanded[..., :3] - self.finger_pose[..., :3]

    def get_contact_info(self, env):
        sensor_data = []
        for name in self.fingers_name_list:
            sensor = env.scene[f"{name}_contact"]
            force_data = torch.linalg.norm(
                sensor._data.force_matrix_w.reshape(env.num_envs, 3), dim=1).unsqueeze(1)
            sensor_data.append(force_data)
        self.contact_or_not = (torch.cat(sensor_data, dim=1) > 2.0).int()

    def get_target_object_pose(self, env):
        """Compute target bottle pose: cap 3cm above cup rim, centered over cup."""
        cup_top_z = self.cup_pose[:, 2] + 0.15
        cup_center_xy = self.cup_pose[:, :2]

        target_quat = self.default_bottle_quat.clone().to(env.device) if self.default_bottle_quat is not None \
            else torch.zeros(env.num_envs, 4, device=env.device)

        cap_offset = self.bottle_cap_offset.to(env.device).unsqueeze(0).expand(env.num_envs, -1)
        cap_offset_world = math_utils.quat_apply(target_quat, cap_offset)

        desired_cap_pos = torch.zeros(env.num_envs, 3, device=env.device)
        desired_cap_pos[:, :2] = cup_center_xy
        desired_cap_pos[:, 2] = cup_top_z + 0.10

        target_pos = desired_cap_pos - cap_offset_world

        target_pose = torch.zeros(env.num_envs, 7, device=env.device)
        target_pose[:, :3] = target_pos
        target_pose[:, 3:7] = target_quat
        return target_pose

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_tip_pos(self):
        local_offset = self.bottle_cap_offset.to(self.object_pose.device).unsqueeze(0).expand(
            self.object_pose.shape[0], -1)
        world_offset = math_utils.quat_apply(self.object_pose[:, 3:7], local_offset)
        self.tip_pos = self.object_pose[:, :3] + world_offset

    def _get_reset_init_height(self, env):
        if self.reset_init_height is not None:
            return self.reset_init_height
        return torch.full((env.num_envs,), self.init_height, device=env.device)

    def _joint_vel_l2(self, env):
        robot = env.scene[self.asset_name]
        return torch.sum(robot.data.joint_vel ** 2, dim=1)

    def _joint_pos_limits(self, env, soft_ratio: float = 0.9):
        robot = env.scene[self.asset_name]
        joint_pos = robot.data.joint_pos
        lower = robot.data.soft_joint_pos_limits[:, :, 0]
        upper = robot.data.soft_joint_pos_limits[:, :, 1]
        lower_violation = torch.clamp(lower * soft_ratio - joint_pos, min=0.0)
        upper_violation = torch.clamp(joint_pos - upper * soft_ratio, min=0.0)
        return torch.sum(lower_violation + upper_violation, dim=1)

    def _action_rate_l2(self, env):
        return torch.sum(
            (env.action_manager.action - env.action_manager.prev_action) ** 2, dim=1)

    # ------------------------------------------------------------------
    # Obs term callables — wire as ObsTerm(func=...) in task __post_init__
    # ------------------------------------------------------------------

    def obs_cup_pose(self, env):
        """Cup position (3D) in env-relative frame."""
        if self.cup_pose is None:
            self.get_cup_info(env)
        return self.cup_pose[:, :3]

    def obs_manipulated_object_pose(self, env):
        """Bottle pose (7D) in env-relative frame."""
        if self.object_pose is None:
            self.get_object_info(env)
        return self.object_pose

    def obs_target_object_pose(self, env):
        """Computed target bottle pose (7D): cap 3cm above cup."""
        if self.cup_pose is None:
            self.get_cup_info(env)
        return self.get_target_object_pose(env)

    def obs_contact(self, env):
        """(N, num_fingers) binary contact per finger."""
        if self.contact_or_not is None:
            self.get_contact_info(env)
        return self.contact_or_not

    def obs_object_in_tip(self, env):
        """(N, num_fingers*3) bottle-center-to-finger displacement vectors, flattened.

        Uses raw bottle center (no grasp_target_offset) to match IsaacLab's
        SingleHandPourObs.object_in_tip exactly. get_finger_info() uses the offset
        for reward shaping but that must not bleed into the PPO obs.
        """
        if self.object_pose is None:
            self.get_object_info(env)
        if self.finger_pose is None:
            self.get_finger_info(env)
        object_pos = self.object_pose[:, :3].unsqueeze(1)          # (B, 1, 3)
        dev = object_pos - self.finger_pose[..., :3]               # (B, num_fingers, 3)
        return dev.reshape(env.num_envs, -1)
