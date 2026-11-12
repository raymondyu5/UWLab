#!/bin/bash
# CPU-only diagnostic: which python packages exist in the container, with and without the persistent overlay.
#SBATCH --job-name=uwlab_diag
#SBATCH --account=weirdlab
#SBATCH --partition=ckpt
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/mmfs1/gscratch/weirdlab/will/UWLab/logs/slurm/%x-%j.out
cd /mmfs1/gscratch/weirdlab/will/UWLab
CMD='./uwlab.sh -p -c "import importlib.util as u; print({m: bool(u.find_spec(m)) for m in [\"torchcfm\",\"dill\",\"imageio\",\"zarr\",\"hydra\"]})"'
echo "== no overlay"; bash scripts/cluster/run_in_container.sh "$CMD"
echo "== overlay (ro)"; USE_OVERLAY=1 bash scripts/cluster/run_in_container.sh "$CMD"
