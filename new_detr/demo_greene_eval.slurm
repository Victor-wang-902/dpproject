#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH --job-name=baseline_detr
#SBATCH --mail-type=END
#SBATCH --mail-user=zw2374@nyu.edu
#SBATCH --time=48:00:00
#SBATCH --output=detr_%j.out
#SBATCH --error=detr_%j.err

singularity exec --nv \
--overlay /scratch/zw2374/overlay-50G-10M.ext3:ro \
--overlay /scratch/xl3136/dl-sp22-final-project/dataset/unlabeled_224.sqsh \
--overlay /scratch/xl3136/dl-sp22-final-project/dataset/labeled.sqsh \
/scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif \
/bin/bash -c "
source /ext3/env.sh
conda activate dp_proj
python main.py --batch_size 2 --no_aux_loss --eval --resume barlow2_new/checkpoint.pth
"
