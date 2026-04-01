# Task: GraspPinkCup (UW-FrankaLeap-GraspPinkCup-*)

**Source**: `source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/grasp/config/franka_leap/tasks/pink_cup.py`

## Registered Environments

| ID | Config Class |
|----|-------------|
| `UW-FrankaLeap-GraspPinkCup-JointAbs-v0` | `GraspPinkCupFrankaLeapJointAbsCfg` |
| `UW-FrankaLeap-GraspPinkCup-IkRel-v0` | `GraspPinkCupFrankaLeapIkRelCfg` |
| `UW-FrankaLeap-GraspPinkCup-IkAbs-v0` | `GraspPinkCupFrankaLeapIkAbsCfg` |
| `UW-FrankaLeap-GraspPinkCup-JointAbs-SysidApplied-v0` | `GraspPinkCupFrankaLeapJointAbsSysidAppliedCfg` |

## Task Goal

Grasp a pink YCB cup and lift it above the table.  
**Success**: cup z ≥ 0.20 m above table (accounting for table height randomization offset).

## Timing

- Horizon: 200 steps
- Control: 10 Hz (60 Hz physics / 6 decimation)
- Episode length: (200 + 15 warmup) × 6 × (1/60) ≈ 21.5 seconds

## Scene

- **Pink cup**: `assets/pink_cup/rigid_object.usd`, scale (1.0, 1.0, 1.0)
  - Note: the 0.35 scale from the original YCB recenter_ycb.yaml is baked into the USD already
- **Table block**: `assets/table/table_block.usd`, scale (1.2, 1.0, 0.10), kinematic, centered at `(0.55, 0.00, 0.0)`

## Spawn / Reset Randomizations

### Pink Cup
- Default pos: `(0.55, 0.0, 0.11)` (env-local)
- Default rot: `(0.707, 0.707, 0.0, 0.0)` — 90° around X axis, placing cup upright
- Pose range (uniform):
  - x: ±0.05 m → x ∈ [0.50, 0.60]
  - y: ±0.05 m → y ∈ [-0.05, 0.05]
  - z/roll/pitch/yaw: 0
- Smaller XY range than the bottle task (±5cm vs ±10cm)

### Table Block Height
- `table_z_range = (0.0, 0.05)` → table top z ∈ [0.0, 0.05] m above default (same as bottle task)

### Material Randomization (`randomize_object_material`)
- `min_step_count_between_reset=800`
- Static friction: U[0.3, 1.5]
- Dynamic friction: U[0.3, 1.2]
- Restitution: 0.0
- 64 buckets

### Mass Randomization (`randomize_object_mass`)
- `min_step_count_between_reset=800`
- Uniform scale: U[0.8, 1.5] × default mass

### Robot Reset
- Arm + hand reset to `ARM_RESET + HAND_RESET` (same as all Franka-LEAP tasks)

### Reset Height Capture
- `capture_reset_height` event runs after `reset_object`
- Records actual cup z at reset to gate lift rewards

## Observations

| Key | Shape | Description |
|-----|-------|-------------|
| `joint_pos` | (23,) | All joint positions |
| `arm_joint_pos` | (7,) | Arm joints only |
| `hand_joint_pos` | (16,) | LEAP hand joints |
| `ee_pose` | (7,) | EE pos+quat (world frame) |
| `target_object_pose` | (7,) | Fixed target pos `(0.60, 0.10, 0.40)` + current cup rotation |
| `manipulated_object_pose` | (7,) | Cup pos+quat (env-local) |
| `contact_obs` | (5,) | Binary contact per finger (>4N threshold) |
| `object_in_tip` | (15,) | Finger-to-object displacement vectors, flattened (5 × 3D) |
| `seg_pc` | (3, 2048) | Segmented PCD: arm (256) + hand (64) + cup (128), noise=0.02 |

## Reward Function (`GraspReward.grasp_rewards`, weight=4.0)

Identical reward structure to GraspBottle — same `GraspReward` class, same weights. Parameterized by:
- `object_name = "grasp_object"` (the pink cup)
- `init_height = 0.11`
- `target_pos = (0.60, 0.10, 0.40)`

### Components (same as GraspBottle)

**`finger2object_rewards`**:
- Per-finger: `clip(1/(0.1+dist) - 2.0, 0, 4.5)`, summed/5 × 3.0

**`object2fingercontact_rewards`**:
- Contact if force > 4N; `sum(contact)*3 + (sum≥2)*2.0`

**`liftobject_rewards`**:
- Lift reward: `clip((z - init_z)/0.2, -0.1, 1.0) * 80`, capped [-2, 60]
- Target distance reward (once lifted ≥2cm): `clip(1 - dist/0.3, -0.1, 1.0) * 30`
- X penalty: `clip(|x_dev|/0.20, 0, 1.5) * -5` when lifted
- Topple penalty: penalizes xy rotation sum > 0.1 rad

**`penalty_contact`**: -15 if wrist (panda_link6) contacts table (>4N force)

**Penalties**: joint_vel -1e-3, action_rate -5e-3

### Success Criterion
```python
cup_z >= SUCCESS_HEIGHT + table_z_offset
# SUCCESS_HEIGHT = 0.20m (slightly lower than bottle's 0.25m)
```

## Terminations

| Condition | Type |
|-----------|------|
| `time_out` | Episode length (200 steps) |

No early termination.

## Action Variants

| Suffix | Action Space | Warmup |
|--------|-------------|--------|
| `JointAbs` | 23D absolute joint positions | Hold at ARM_RESET + HAND_RESET |
| `IkRel` | 6D delta EE + 16D abs hand | Zero action |
| `IkAbs` | 7D abs EE pose + 16D abs hand | Hold current EE + current hand |

## SysidApplied Variant

Same as GraspBottle: uses `FRANKA_LEAP_REAL_GAINS_ARM_ACTUATOR_DELAYED_CFG` + sysid parameter reset event.

## Key Differences vs GraspBottle

| Feature | GraspPinkCup | GraspBottle |
|---------|-------------|-------------|
| Object | Pink YCB cup (upright) | Bourbon bottle (sideways, cap toward +Y) |
| USD scale | 1.0x | 1.1x |
| Spawn pos | (0.55, 0.00, 0.11) | (0.55, -0.10, 0.11) |
| Spawn rot | 90° X (upright) | -90° Z (horizontal) |
| XY range | ±0.05m | ±0.10m |
| Success height | 0.20m | 0.25m |
| Horizon | 200 steps | 180 steps |
| Table block center | (0.55, 0.00) | (0.55, -0.10) |

The pink cup task was the original "canonical" grasp task in this codebase (ported from rl_env_ycb_cam_custom_init_pink_cup.yaml). The bottle task was added later for the pour pipeline.
