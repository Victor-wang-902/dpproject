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
##python -m distributed.launch --proc_per_node=4 --master_port=50000 driver.py
##python driver.py configs/faster_rcnn/faster_rcnn_swin_fpn_1x_coco.py --work-dir=swin_new_test
##MASTER_ADDR=localhost MASTER_PORT=64214 python -m torch.distributed.launch --nproc_per_node=4 --master_port=64214 driver.py configs/faster_rcnn/faster_rcnn_swin_fpn_1x_coco.py --launcher slurm
##python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=64217 driver.py configs/faster_rcnn/faster_rcnn_swin_fpn_1x_coco.py --launcher slurm
##python driver_detr.py configs/detr/detr_r50_8x2_150e_coco.py
##python driver.py configs/faster_rcnn/faster_rcnn_swin_fpn_1x_coco_stretch.py --work-dir=swin_new_test_2
##python demo.py
##python test.py configs/faster_rcnn/faster_rcnn_swin_fpn_1x_coco_stretch.py swin_new_test_2/epoch_5.pth --eval mAP --out swin_new_test_2/result_5.pkl
##python driver.py configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco_custom.py --work-dir=r50_test
##python driver.py configs/faster_rcnn/faster_rcnn_swin_fpn_1x_coco_stretch.py --work-dir=swin_new_test_3
mkdir /tmp/$USER
export SINGULARITY_CACHEDIR=/tmp/$USER


singularity exec --nv \
--bind /scratch \
--overlay /scratch/DL22SP/conda.ext3:ro \
--overlay /scratch/zw2374/dataset/unlabeled_224.sqsh \
--overlay /scratch/zw2374/dataset/labeled.sqsh \
/share/apps/images/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif \
/bin/bash -c "
source /ext3/env.sh
conda activate openmmlab
python driver.py configs/faster_rcnn/faster_rcnn_swin_fpn_1x_coco_stretch.py --work-dir=swin_new_test_3
"
