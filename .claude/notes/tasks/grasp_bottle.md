# Task: GraspBottle (UW-FrankaLeap-GraspBottle-*)

**Source**: `source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/grasp/config/franka_leap/tasks/bottle.py`

## Registered Environments

| ID | Config Class |
|----|-------------|
| `UW-FrankaLeap-GraspBottle-JointAbs-v0` | `GraspBottleFrankaLeapJointAbsCfg` |
| `UW-FrankaLeap-GraspBottle-IkRel-v0` | `GraspBottleFrankaLeapIkRelCfg` |
| `UW-FrankaLeap-GraspBottle-IkAbs-v0` | `GraspBottleFrankaLeapIkAbsCfg` |
| `UW-FrankaLeap-GraspBottle-JointAbs-SysidApplied-v0` | `GraspBottleFrankaLeapJointAbsSysidAppliedCfg` |

## Task Goal

Grasp a bourbon bottle and lift it to a target position above the table.  
**Success**: bottle z ≥ 0.25 m above table (accounting for table height randomization offset).

## Timing

- Horizon: 180 steps
- Control: 10 Hz (60 Hz physics / 6 decimation)
- Episode length: (180 + 15 warmup) × 6 × (1/60) ≈ 19.5 seconds

## Scene

- **Bottle**: `assets/bourbon/rigid_object.usd`, scale (1.1, 1.1, 1.1)
- **Table block**: `assets/table/table_block.usd`, scale (1.2, 1.0, 0.10), kinematic (doesn't fall), centered at `(0.55, -0.10, 0.0)`
  - Used for table height randomization — actual block height is randomly sampled at reset

## Spawn / Reset Randomizations

### Bottle
- Default pos: `(0.55, -0.10, 0.11)` (env-local)
- Default rot: `(0.707, 0.0, 0.0, -0.707)` — 90° around world Z, so local -X (cap) points toward world +Y (toward cup)
- Pose range (uniform):
  - x: ±0.10 m → x ∈ [0.45, 0.65]
  - y: ±0.10 m → y ∈ [-0.20, 0.00]
  - z/roll/pitch/yaw: 0
- `reset_height` respects actual table block z (which varies due to table height randomization)

### Table Block Height
- `table_z_range = (0.0, 0.05)` → table top z ∈ [0.0, 0.05] m above default
- Randomized via `reset_table_block` event at every reset (compared to pour task which has 0.0 range)

### Material Randomization (`randomize_object_material`)
- `min_step_count_between_reset=800` → not every reset, only every 800+ env steps
- Static friction: U[0.3, 1.5]
- Dynamic friction: U[0.3, 1.2]
- Restitution: 0.0 (no bounce)
- 64 material buckets

### Mass Randomization (`randomize_object_mass`)
- `min_step_count_between_reset=800`
- Uniform scale: U[0.8, 1.5] × default mass

### Robot Reset
- Arm: `ARM_RESET = [0.311, 0.004, -0.311, -2.051, 0.001, 2.055, 0.781]`
- Hand: `HAND_RESET` (16D, specific LEAP grasp pose)

### Reset Height Capture
- After reset, `capture_reset_height` records the actual bottle z (accounts for table offset)
- Used to gate lift rewards relative to actual start height

## Observations

| Key | Shape | Description |
|-----|-------|-------------|
| `joint_pos` | (23,) | All joint positions |
| `arm_joint_pos` | (7,) | Arm joints only |
| `hand_joint_pos` | (16,) | LEAP hand joints |
| `ee_pose` | (7,) | EE pos+quat (world frame) |
| `target_object_pose` | (7,) | Fixed target pos `(0.60, 0.10, 0.40)` + current object rotation |
| `manipulated_object_pose` | (7,) | Bottle pos+quat (env-local) |
| `contact_obs` | (5,) | Binary contact per finger (>4N threshold) |
| `object_in_tip` | (15,) | Finger-to-object displacement vectors, flattened (5 fingers × 3D) |
| `seg_pc` | (3, 2048) | Segmented point cloud: arm (256 pts) + hand (64 pts) + bottle (128 pts), noise=0.02 |

**Note**: Unlike pour task, seg_pc has `pcd_noise=0.02` (Gaussian noise on PCD during RL training too, not just BC).

### Finger Names (`FINGERS_NAME_LIST`)
`["palm_lower", "fingertip", "thumb_fingertip", "fingertip_2", "fingertip_3"]`

## Reward Function (`GraspReward.grasp_rewards`)

Single `RewTerm` with weight 4.0 calling `grasp_rew.grasp_rewards`. All components combined inside:

### Components

**`finger2object_rewards`**: Dense reward pulling fingers toward bottle
- Per-finger: `clip(1/(0.1 + dist) - 2.0, 0, 4.5)` where dist = clip(finger-to-object dist, 0.02, 0.8)
- Summed over 5 fingers, divided by 5, scaled by 3.0

**`object2fingercontact_rewards`**: Reward for ≥2 fingers in contact
- Per-finger contact if force > 4N
- `sum(contact) * 3 + (sum >= 2) * 2.0`

**`liftobject_rewards`**: Dense reward for lifting toward target
- `lift_reward = clip((obj_z - init_z) / 0.2, -0.1, 1.0) * 80`, capped [-2, 60]
- `lift_reward_scale = (obj_z - init_z >= 0.02)` — gates target distance reward
- `target_dist_reward = clip(1 - dist/0.3, -0.1, 1.0) * 30` (only if lifted ≥2cm)
- `x_penalty = clip(|x_dev|/0.20, 0, 1.5) * -5` (penalize X offset from target, only when lifted)
- `topple_reward`: penalizes object rotation (xy Euler angle sum > 0.1 rad), scaled by lift_reward_scale
- **Target position**: `(0.60, 0.10, 0.40)` (fixed, env-local)

**`penalty_contact`**: Wrist-table contact penalty
- `-15` if panda_link6 contact force > 4N

**Penalties**:
- Joint velocity L2: -1e-3
- Action rate L2: -5e-3

### Success Criterion (`is_success`)
```python
obj_z >= BOTTLE_SUCCESS_HEIGHT + table_z_offset
# where BOTTLE_SUCCESS_HEIGHT = 0.25m
# table_z_offset = current table block z - default table block z
```

## Terminations

| Condition | Type |
|-----------|------|
| `time_out` | Episode length (180 steps) |

No early termination for dropped object (only timeout).

## Action Variants

| Suffix | Action Space | Warmup |
|--------|-------------|--------|
| `JointAbs` | 23D absolute joint positions | Hold at ARM_RESET + HAND_RESET |
| `IkRel` | 6D delta EE + 16D abs hand | Zero action |
| `IkAbs` | 7D abs EE pose + 16D abs hand | Hold current EE + current hand |

## SysidApplied Variant

`GraspBottleFrankaLeapJointAbsSysidAppliedCfg` uses:
- `FRANKA_LEAP_REAL_GAINS_ARM_ACTUATOR_DELAYED_CFG` (delayed actuator model matching real hardware)
- `apply_sysid_params_on_reset` event applies identified system parameters
- Used for sim-to-real transfer overlay scripts

## Differences from PinkCup Task

| Feature | GraspBottle | GraspPinkCup |
|---------|-------------|--------------|
| Object | Bourbon bottle (scale 1.1x) | Pink cup (scale 1.0x) |
| Spawn pos | (0.55, -0.10, 0.11) | (0.55, 0.00, 0.11) |
| Spawn rot | -90° yaw (cap toward +Y) | 90° pitch (upright cup) |
| XY range | ±0.10m | ±0.05m |
| Success height | 0.25m | 0.20m |
| Horizon | 180 steps | 200 steps |
| PCD noise | 0.02 | 0.02 |
