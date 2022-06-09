#!/bin/bash

## change the last two digits to your team id
#SBATCH --account=csci_ga_2572_2022sp_08

## change the partition number to use different number of GPUs
#SBATCH --partition=n1s8-v100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

##SBATCH --partition=n1s16-v100-2
##SBATCH --gres=gpu:2
##SBATCH --cpus-per-task=16

##SBATCH --partition=n1c24m128-v100-4
##SBATCH --gres=gpu:4
##SBATCH --cpus-per-task=24

#SBATCH --time=24:00:00
#SBATCH --output=demo_%j.out
#SBATCH --error=demo_%j.err
#SBATCH --exclusive
#SBATCH --requeue
##python main.py --batch_size 32 --lr 1e-6 --lr_backbone 1e-7 --world_size 4 --num_workers 8 --epochs 300 --output_dir barlow2_distributed
##python main.py --batch_size 2 --lr 1e-5 --lr_backbone 0 --epochs 150 --world_size 1 --num_workers 8 --bbox_loss_coef 8 --giou_loss_coef 4 --eos_coef 1 --position_embedding learned --output_dir updetr_low_lr_new_coef_fix_bb_learned_pos --resume barlow/checkpoint/updetr.pth
##python main.py --batch_size 2 --lr 1e-5 --lr_backbone 0 --epochs 150 --world_size 1 --num_workers 8 --bbox_loss_coef 8 --giou_loss_coef 4 --eos_coef 1  --output_dir updetr_low_lr_new_coef_fix_bb --resume barlow/checkpoint/updetr.pth
##python main.py --batch_size 2 --lr 1e-5 --lr_backbone 1e-6 --epochs 150 --world_size 1 --num_workers 8 --bbox_loss_coef 8 --giou_loss_coef 4 --eos_coef 1  --output_dir updetr_low_lr_new_coef --resume barlow/checkpoint/updetr.pth
##python main.py --batch_size 2 --lr 1e-4 --lr_backbone 1e-5 --epochs 150 --world_size 1 --position_embedding learned --num_workers 8 --bbox_loss_coef 6 --giou_loss_coef 4 --eos_coef 0.31  --resume barlow/checkpoint/updetr.pth --output_dir updetr_new_coef_learned_pos
export S
INGULARITY_CACHEDIR=/scratch/$USER

singularity exec --nv \
--bind /scratch \
--overlay /scratch/DL22SP/conda.ext3:ro \
--overlay /scratch/zw2374/dataset/unlabeled_224.sqsh \
--overlay /scratch/zw2374/dataset/labeled.sqsh \
/share/apps/images/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif \
/bin/bash -c "
source /ext3/env.sh
conda activate dp_proj
python main.py --batch_size 2 --lr 1e-5 --lr_backbone 1e-6 --lr_drop=100 --epochs 150 --world_size 1 --num_workers 8 --bbox_loss_coef 5 --giou_loss_coef 2 --eos_coef 0.08 --position_embedding sine --output_dir updetr_new_coef_fix_bb --resume barlow/checkpoint/updetr.pth --load_bb barlow/checkpoint/checkpoint2.pth
"
