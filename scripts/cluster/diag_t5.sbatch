#!/bin/bash
# CPU-only diagnostic: transformers + flan-t5-small available (offline, from /gscratch/weirdlab/will/hf_cache) inside the container.
#SBATCH --job-name=uwlab_diag_t5
#SBATCH --account=weirdlab
#SBATCH --partition=ckpt
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/mmfs1/gscratch/weirdlab/will/UWLab/logs/slurm/%x-%j.out
cd /mmfs1/gscratch/weirdlab/will/UWLab
bash scripts/cluster/run_in_container.sh './uwlab.sh -p -c "
import transformers, torch
from transformers import AutoTokenizer, T5EncoderModel
print(\"transformers\", transformers.__version__)
tok = AutoTokenizer.from_pretrained(\"google/flan-t5-small\"); enc = T5EncoderModel.from_pretrained(\"google/flan-t5-small\").eval()
s = [\"move the hand 12.3 cm forward and 45.6 cm left and 0.7 cm down\", \"shift the hand 100.0 cm backward and 4.5 cm right and 21.0 cm up\"]
b = tok(s, return_tensors=\"pt\", padding=\"max_length\", max_length=64)
print(\"n tokens\", b.attention_mask.sum(1).tolist(), tok.convert_ids_to_tokens(b.input_ids[0][: int(b.attention_mask[0].sum())]))
print(enc(input_ids=b.input_ids, attention_mask=b.attention_mask).last_hidden_state.shape)
"'
