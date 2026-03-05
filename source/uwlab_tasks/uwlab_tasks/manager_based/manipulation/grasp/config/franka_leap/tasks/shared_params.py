
############################################
################ task info #################
############################################

# From rl_env_ycb_synthetic_pc_custom_init_pink_cup.yaml:
#   right_reset_joint_pose (arm, 7D)
ARM_RESET = [
    3.1088299e-01, 4.0700440e-03, -3.1125304e-01, -2.0509737e+00,
    1.4107295e-03, 2.0548446e+00, 7.8060406e-01,
]
#   right_reset_hand_joint_pose (hand, 16D)
HAND_RESET = [
    0.35281801223754883, 0.6442744731903076, 0.29912877082824707, 0.34514832496643066,
    -0.03681302070617676, -0.06749272346496582, -0.09357023239135742, -0.14725971221923828,
    0.0659637451171875, 0.43411898612976074, 0.05982780456542969, 0.013808250427246094,
    0.03221607208251953, -0.009201288223266602, 0.029148101806640625, 0.0046045780181884766,
]

############################################
################ piping info ###############
############################################
ARM_MESH_DIR = "/workspace/uwlab/assets/robot/franka_leap/raw_mesh" # directory containing the arm mesh files
HAND_MESH_DIR = "/workspace/uwlab/assets/robot/franka_leap/raw_mesh" # directory containing the hand mesh files
FINGERS_NAME_LIST = ["palm_lower", "fingertip", "thumb_fingertip", "fingertip_2", "fingertip_3"] # list of finger names
