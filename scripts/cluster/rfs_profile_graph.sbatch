#!/bin/bash
# RFS step profile: CFM CUDA graph (fp32) then CUDA graph + bf16. Submit: scripts/cluster/submit.sh scripts/cluster/rfs_profile_graph.sbatch
#SBATCH --job-name=rfs_profile_graph
#SBATCH --account=weirdlab
#SBATCH --partition=gpu-l40
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:l40:1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/mmfs1/gscratch/weirdlab/will/UWLab/logs/slurm/%x-%j.out

cd /mmfs1/gscratch/weirdlab/will/UWLab
for CFG in arm_rfs_joint_cfg_graph arm_rfs_joint_cfg_graph_bf16; do
  echo "=== RUN $CFG ==="
  bash scripts/cluster/run_in_container.sh "timeout 400 ./uwlab.sh -p scripts/reinforcement_learning/sb3/rfs/train.py --task UW-FrankaLeap-GraspCube-JointAbs-v0 --num_envs 512 --diffusion_path logs/base/bc_cfm_pcd_cube_base70_retry40_0508_absjoint_h16_hist4_extnoise_fast --cfg configs/rl/$CFG.yaml --asymmetric_ac --profile_interval 20 --no_eval_video --eval_interval 100000000 --headless"
done
