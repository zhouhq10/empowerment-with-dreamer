#!/bin/bash
#SBATCH -J dreamer                 # Job name
#SBATCH --ntasks=1                 # Number of tasks
#SBATCH --cpus-per-task=4          # Number of CPU cores per task
#SBATCH --nodes=1                  # Ensure that all cores are on the same machine with nodes=1
#SBATCH --time=1-00:00
#SBATCH --partition=a100-fat-galvani
#SBATCH --gres=gpu:1               # Changed from 2 to 1 to match the number of GPUs per node
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G                  # Total memory pool for all cores (see also --mem-per-cpu); exceeding this number will cause your job to fail.
#SBATCH --output=/mnt/lustre/work/wu/wkn758/dreamer_log/%j.out       # File to which STDOUT will be written - make sure this is not on $HOME
#SBATCH --error=/mnt/lustre/work/wu/wkn758/dreamer_log/%j.err        # File to which STDERR will be written - make sure

# module load Python/3.11.10-GCCcore-11.2.0q
# module load CUDA/12.1.1q
# module load FFmpeg/6.0-GCCcore-12.3.0
# source $HOME/venvs/dreamerv3/bin/activate
# cd ..
# a100-fat-galvani

# export PATH="/mnt/lustre/work/wu/wkn758/ffmpeg-git-20240629-amd64-static:$PATH"

# source $HOME/.bashrc # Does not work anymore for some reason
# Instead of sourcing .bashrc, initialize conda directly.
eval "$(conda shell.bash hook)"
conda activate /mnt/lustre/work/wu/wkn758/conda_envs/empower

# Set a writable cache directory for matplotlib to avoid permission errors.
export MPLCONFIGDIR="$WORK/matplotlib_cache"

# set dir to world model directory for command afterwards to work
cd /mnt/lustre/work/wu/wkn758/empowerment-and-human-behavior/world_model

# python -m dreamerv3 --configs crafter-pretrain-infogain \
# --logdir "/mnt/lustre/work/wu/wkn758/dreamer_pretrain/crafter/infogain" \

# python -m dreamerv3 --configs crafter-transfer \
# --logdir "/mnt/lustre/work/wu/wkn758/dreamer_pretrain/crafter/infogain_empowerment_mul_basedonpreviousstate_transfer" \

# python -m dreamerv3 --configs minigrid-pretrain-mixedenv-empowerment-replaybuffer20000 \
# --logdir "/mnt/lustre/work/wu/wkn758/dreamer_pretrain/mixedenv/MixedEnv/latentsempowerment_replaybuffer20000" \

# python -m dreamerv3 --configs minigrid-pretrain-obs-lava \
# --logdir "/mnt/lustre/work/wu/wkn758/dreamer_pretrain/lava/DistShift1/latentsnovelty_32" \

# python -m dreamerv3 --configs minigrid-pretrain-obs-dynamics \
# --logdir "/mnt/lustre/work/wu/wkn758/dreamer_pretrain/dynamics/Dynamic-Obstacles-16x16/pseudonovelty_64" \

# python -m dreamerv3 --configs minigrid-pretrain-obs-walllava \
# --logdir "/mnt/lustre/work/wu/wkn758/dreamer_pretrain/lava/LavaCrossingS9N1/extrinsic" \



# # # Minigrid Transfer
# replacements=("extrinsic" "latentsnovelty_32" "latentsempowerment" "latentsinfogain" "obscbet" "obsnovelty" "pseudonovelty_128")
# replacement=${replacements[$SLURM_ARRAY_TASK_ID]}

# logdir_base="/mnt/lustre/work/wu/wkn758/dreamer_transfer/lava/LavaCrossingS9N1_LavaCrossingS9N2"
# checkpoint_base="/mnt/lustre/work/wu/wkn758/dreamer_pretrain/lava/LavaCrossingS9N1"
# logdir="${logdir_base}/${replacement}"
# checkpoint="${checkpoint_base}/${replacement}/checkpoint.ckpt"
# python -m dreamerv3 --configs minigrid-transfer-wall \
# --logdir "$logdir" \
# --run.from_checkpoint "$checkpoint"


# logdir_base="/mnt/lustre/work/wu/wkn758/dreamer_transfer/wall/SimpleCrossingS9N1_SimpleCrossingS9N2"
# checkpoint_base="/mnt/lustre/work/wu/wkn758/dreamer_pretrain/wall/SimpleCrossingS9N1"
# logdir="${logdir_base}/${replacement}"
# checkpoint="${checkpoint_base}/${replacement}/checkpoint.ckpt"
# python -m dreamerv3 --configs minigrid-transfer-wall \
# --logdir "$logdir" \
# --run.from_checkpoint "$checkpoint"

# # logdir_base="/mnt/lustre/work/wu/wkn758/dreamer_transfer/lava/DistShift1_DistShift2"
# # checkpoint_base="/mnt/lustre/work/wu/wkn758/dreamer_pretrain/lava/DistShift1"
# # logdir="${logdir_base}/${replacement}"
# # checkpoint="${checkpoint_base}/${replacement}/checkpoint.ckpt"
# # python -m dreamerv3 --configs minigrid-transfer-lava \
# # --logdir "$logdir" \
# # --run.from_checkpoint "$checkpoint"

# # logdir_base="/mnt/lustre/work/wu/wkn758/dreamer_transfer/dynamics/Dynamic-Obstacles-16x16_Dynamic-Obstacles-8x8"
# # checkpoint_base="/mnt/lustre/work/wu/wkn758/dreamer_pretrain/dynamics/Dynamic-Obstacles-16x16"
# # logdir="${logdir_base}/${replacement}"
# # checkpoint="${checkpoint_base}/${replacement}/checkpoint.ckpt"

# # python -m dreamerv3 --configs minigrid-transfer-dynamics \
# # --logdir "$logdir" \
# # --run.from_checkpoint "$checkpoint"


# Fine-tuning with extrinsic reward (transfer via frozen policy)
replacement="extrinsic"  # This defines which pretrained agent to fine-tune (with extrinsic reward only, via crafter-transfer-extrinsic config)
logdir_base="/mnt/lustre/work/wu/wkn758/dreamer_transfer/crafter"
checkpoint_base="/mnt/lustre/work/wu/wkn758/dreamer_pretrain/crafter"
logdir="${logdir_base}/${replacement}"
checkpoint="${checkpoint_base}/${replacement}/checkpoint.ckpt"
python -m dreamerv3 --configs crafter-transfer-extrinsic \
--logdir "$logdir" \
--run.from_checkpoint "$checkpoint"

# ## Fine-tuning with infogain reward (transfer via frozen policy)
# replacement="extrinsic" # This defines which pretrained agent to fine-tune (with infogain reward only, via crafter-transfer-infogain config)
# logdir_base="/mnt/lustre/work/wu/wkn758/dreamer_transfer/crafter"
# checkpoint_base="/mnt/lustre/work/wu/wkn758/dreamer_pretrain/crafter"
# logdir="${logdir_base}/${replacement}"
# checkpoint="${checkpoint_base}/${replacement}/checkpoint.ckpt"
# python -m dreamerv3 --configs crafter-transfer-infogain \
# --logdir "$logdir" \
# --run.from_checkpoint "$checkpoint"


# # Actual fine-tuning (with extrinsic reward) instead of only doing transfer via frozen policy
# replacement="infogain"  # This defines which pretrained agent to fine-tune (with extrinsic reward only, via crafter-finetune-extrinsic config)
# logdir_base="/mnt/lustre/work/wu/wkn758/dreamer_finetune/crafter"
# checkpoint_base="/mnt/lustre/work/wu/wkn758/dreamer_pretrain/crafter"
# logdir="${logdir_base}/${replacement}"
# checkpoint="${checkpoint_base}/${replacement}/checkpoint.ckpt"
# python -m dreamerv3 --configs crafter-finetune-extrinsic \
# --logdir "$logdir" \
# --run.from_checkpoint "$checkpoint"

# # Actual fine-tuning (with infogain reward) instead of only doing transfer via frozen policy
# replacement="none_replaybuffer20000"  # This defines which pretrained agent to fine-tune (with infogain reward only, via crafter-finetune-infogain config)
# logdir_base="/mnt/lustre/work/wu/wkn758/dreamer_finetune_infogain_replaybuffer20000/crafter"
# checkpoint_base="/mnt/lustre/work/wu/wkn758/dreamer_pretrain/crafter"
# logdir="${logdir_base}/${replacement}"
# checkpoint="${checkpoint_base}/${replacement}/checkpoint.ckpt"
# python -m dreamerv3 --configs crafter-finetune-infogain-replaybuffer20000 \
# --logdir "$logdir" \
# --run.from_checkpoint "$checkpoint"
