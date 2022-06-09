#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH --job-name=baseline_detr
#SBATCH --mail-type=END
#SBATCH --mail-user=zw2374@nyu.edu
#SBATCH --time=48:00:00
#SBATCH --output=detr_%j.out
#SBATCH --error=detr_%j.err
##python main.py --batch_size 4 --lr 1e-4 --lr_backbone 1e-5 --epochs 150 --world_size 1 --num_workers 8 --output_dir barlow2_new
##torchrun main.py --batch_size 32 --lr 1e-4 --lr_backbone 1e-5 --num_queries 100 --epochs 150 --num_workers 8 --world_size 4 --output_dir barlow2_new_dist --dist --dist_url tcp://localhost:58471
##python main.py --batch_size 2 --lr 1e-5 --lr_backbone 0 --epochs 150 --world_size 1 --num_workers 8 --bbox_loss_coef 8 --giou_loss_coef 4 --eos_coef 1  --output_dir barlow2_low_lr_new_coef_fix_bb
##python main.py --batch_size 2 --lr 1e-5 --lr_backbone 1e-6 --epochs 150 --world_size 1 --num_workers 8 --bbox_loss_coef 8 --giou_loss_coef 4 --eos_coef 1  --output_dir barlow2_low_lr_new_coef


singularity exec --nv \
--overlay /scratch/zw2374/overlay-50G-10M.ext3:ro \
--overlay /scratch/xl3136/dl-sp22-final-project/dataset/unlabeled_224.sqsh \
--overlay /scratch/xl3136/dl-sp22-final-project/dataset/labeled.sqsh \
/scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif \
/bin/bash -c "
source /ext3/env.sh
conda activate dp_proj
python main.py --batch_size 2 --lr 1e-4 --lr_backbone 1e-5 --epochs 150 --world_size 1 --position_embedding learned --num_workers 8 --bbox_loss_coef 8 --giou_loss_coef 4 --eos_coef 1  --output_dir barlow2_new_coef_learned_pos
"
