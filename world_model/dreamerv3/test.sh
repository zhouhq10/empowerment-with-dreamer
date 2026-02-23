#!/bin/bash
#SBATCH --ntasks=1                 # Number of tasks
#SBATCH --cpus-per-task=8          # Number of CPU cores per task
#SBATCH --nodes=1                  # Ensure that all cores are on the same machine with nodes=1
#SBATCH --time=0-2:00
#SBATCH --partition=a100-fat-galvani
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --mem=50G                  # Total memory pool for all cores (see also --mem-per-cpu); exceeding this number will cause your job to fail.
#SBATCH --output=/mnt/qb/work/wu/wkn601/dreamer_log/%j.out       # File to which STDOUT will be written - make sure this is not on $HOME
#SBATCH --error=/mnt/qb/work/wu/wkn601/dreamer_log/%j.err        # File to which STDERR will be written - make sure
##SBATCH --array=0

# module load Python/3.11.10-GCCcore-11.2.0
# module load CUDA/12.1.1q
# module load FFmpeg/6.0-GCCcore-12.3.0
# source $HOME/venvs/dreamerv3/bin/activate
# cd ..

export PATH="/mnt/qb/work/wu/wkn601/ffmpeg-git-20240629-amd64-static:$PATH"


# Minigrid debug
python -m dreamerv3 --configs minigrid-test --logdir "/mnt/qb/work/wu/wkn601/dreamer_test/Unlock_test"

# python -m dreamerv3 --configs minigrid-unlock-extr-intr --logdir "/mnt/qb/work/wu/wkn601/dreamer_full/unlock-extr-intr"

# Minigrid Pre-train
# python -m dreamerv3 --configs minigrid-pre-train --logdir "/mnt/qb/work/wu/wkn601/dreamer_fully_observe/logs/dreamerv3/minigrid-pre-train-1M-Doorkey8x8-2"

# Minigrid Transfer
# python -m dreamerv3 --configs minigrid-transfer \
# --logdir /mnt/qb/work/wu/wkn601/dreamer_full/minigrid-transfer-1M-Unlock \
# --run.from_checkpoint /mnt/qb/work/wu/wkn601/dreamer_full/logs/dreamerv3/minigrid-pre-train-1M-Doorkey8x8-2/checkpoint.ckpt

# Crafter Tabula Rasa / Base (change intrinsic and path)
# python -m dreamerv3 --configs crafter-cbet --logdir ./logs/dreamerv3/crafter-cbet-0-coeff --run.intrinsic True --run.intr_reward_coeff 0.0 --seed 0

# Crafter Pre-train
# python -m dreamerv3 --configs crafter-pre-train --logdir ./logs/dreamerv3/crafter-pre-train-1M-2

# Crafter Transfer
# python -m dreamerv3 --configs crafter-transfer --logdir ./logs/dreamerv3/crafter-transfer-1M --run.from_checkpoint ./logs/dreamerv3/crafter-pre-train-1M/checkpoint.ckpt

# Convergence Test
# python -m dreamerv3 --configs crafter-cbet --logdir ./logs/dreamerv3/crafter-base-2M --run.intrinsic False