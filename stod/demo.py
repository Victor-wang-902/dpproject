# Feel free to modifiy this file. 
# It will only be used to verify the settings are correct 
# modified from https://pytorch.org/docs/stable/data.html

import os
from mmdet.apis import set_random_seed
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import utils
from engine import train_one_epoch, evaluate
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
from mmdet.models import build_detector
from dataset import UnlabeledDataset, LabeledDataset
import transforms as T
from mmcv import Config, DictAction



def get_transform(train):
    transforms = []
    
    if train:
        transforms.append(T.ToTensor())
        transforms.append(T.RandomHorizontalFlip(0.5))
        return T.Compose(transforms)
    

    else:
        transforms.append(T.RandomResize([(256,256)], max_size=256))
        transforms.append(T.Pad(256))
        transforms.append(T.ToTensor())
        transforms.append(T.Normalize([0.49, 0.468, 0.414], [0.286, 0.278, 0.297]))
        return T.Compose(transforms)
    

def build_cofig(cfg):
    cfg.dataset_type = 'MyDataset'
    cfg.data_root = '/labeled/'

    cfg.data.test.type = 'MyDataset'
    cfg.data.test.data_root = '/labeled/'
    cfg.data.test.ann_file = 'training/'
    cfg.data.test.img_prefix = 'training/images'

    cfg.data.train.type = 'MyDataset'
    cfg.data.train.data_root = '/labeled/'
    cfg.data.train.ann_file = 'training/'
    cfg.data.train.img_prefix = 'training/images'

    cfg.data.val.type = 'MyDataset'
    cfg.data.val.data_root = '/labeled/'
    cfg.data.val.ann_file = 'validation/'
    cfg.data.val.img_prefix = 'validation/images'

    # modify num classes of the model in box head
    cfg.model.roi_head.bbox_head.num_classes = 100
    cfg.model.backbone.num_classes=0
    cfg.model.backbone.drop_path_rate=0.0
    #cfg.model.backbone.pretrained = ""
    cfg.train_pipeline[3].policies[0][0].img_scale=[(224,1333),(256,1333),(288,1333),(320,1333)]
    cfg.train_pipeline[3].policies[1][2].img_scale=[(224,1333),(256,1333),(288,1333),(320,1333)]
    cfg.data.train.pipeline = cfg.train_pipeline
    cfg.data.val.pipeline = cfg.test_pipeline
    cfg.data.test.pipeline = cfg.test_pipeline

    # If we need to finetune a model based on a pre-trained detector, we need to
    # use load_from to set the path of checkpoints.
    #cfg.load_from = 'checkpoints/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco_20210526_095054-1f77628b.pth'

    # Set up working dir to save files and logs.
    #cfg.work_dir = './swin_new_test_2'

    # The original learning rate (LR) is set for 8-GPU training.
    # We divide it by 8 since we only use one GPU.
    cfg.lr_config.warmup = "linear"
    cfg.log_config.interval = 10
    cfg.optimizer_config.grad_clip=dict(max_norm=2, norm_type=2)

    # Change the evaluation metric since we use customized dataset.
    cfg.evaluation.metric = 'mAP'
    # We can set the evaluation interval to reduce the evaluation times
    cfg.evaluation.interval = 3
    # We can set the checkpoint saving interval to reduce the storage cost
    cfg.checkpoint_config.interval = 3

    # Set seed thus the results are more reproducible
    cfg.seed = 0
    set_random_seed(0, deterministic=False)

    # We can also use tensorboard to log the training process
    cfg.log_config.hooks = [
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')]


    # We can initialize the logger for training and have a look
    # at the final config used for training
    print(f'Config:\n{cfg.pretty_text}')    
    return cfg



def get_model(cfg,device):

    model = init_detector(
            cfg, checkpoint="swin_new_test_2/epoch_30.pth", device=device)

    return model

def main():
    cfg = Config.fromfile("swin_new_test_2/faster_rcnn_swin_fpn_1x_coco_stretch.py")
    cfg = build_cofig(cfg)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    num_classes = 100
    valid_dataset = LabeledDataset(root='/labeled', split="validation", transforms=get_transform(train=False))
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=2, shuffle=False, num_workers=2, collate_fn=utils.collate_fn)

    model = get_model(cfg,device)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)


    evaluate(model, valid_loader, device=device)

    print("That's it!")

if __name__ == "__main__":
    main()
