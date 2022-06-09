#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=10
#SBATCH --time=48:00:00
#SBATCH --mem=32GB
#SBATCH --job-name=demo
#SBATCH --mail-type=END
#SBATCH --mail-user=hrr288@nyu.edu
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --gres=gpu:2

singularity exec --nv \
--overlay /scratch/hrr288/hrr_env/pytorch1.7.0-cuda11.0.ext3:ro \
--overlay /scratch/xl3136/dl-sp22-final-project/dataset/unlabeled_224.sqsh \
--overlay /scratch/xl3136/dl-sp22-final-project/dataset/labeled.sqsh \
/scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif \
/bin/bash -c "
source /ext3/env.sh; python3 main.py "
