import copy
import os.path as osp

import yaml

import mmcv
import numpy as np

from mmcv import Config
from stod.mmdet.datasets.builder import DATASETS
from stod.mmdet.mmdet.datasets.custom import CustomDataset
from stod.mmdet.mmdet.apis import set_random_seed
from stod.mmdet.datasets import build_dataset
from stod.mmdet.models import build_detector
from stod.mmdet.apis import train_detector


@DATASETS.register_module()
class MyDataset(CustomDataset):

    CLASSES = (
            'cup or mug',
            'bird',
            'hat with a wide brim',
            'person',
            'dog',
            'lizard',
            'sheep',
            'wine bottle',
            'bowl',
            'airplane',
            'domestic cat',
            'car',
            'porcupine',
            'bear',
            'tape player',
            'ray',
            'laptop',
            'zebra',
            'computer keyboard',
            'pitcher',
            'artichoke',
            'tv or monitor',
            'table',
            'chair',
            'helmet',
            'traffic light',
            'red panda',
            'sunglasses',
            'lamp',
            'bicycle',
            'backpack',
            'mushroom',
            'fox',
            'otter',
            'guitar',
            'microphone',
            'strawberry',
            'stove',
            'violin',
            'bookshelf',
            'sofa',
            'bell pepper',
            'bagel',
            'lemon',
            'orange',
            'bench',
            'piano',
            'flower pot',
            'butterfly',
            'purse',
            'pomegranate',
            'train',
            'drum',
            'hippopotamus',
            'ski',
            'ladybug',
            'banana',
            'monkey',
            'bus',
            'miniskirt',
            'camel',
            'cream',
            'lobster',
            'seal',
            'horse',
            'cart',
            'elephant',
            'snake',
            'fig',
            'watercraft',
            'apple',
            'antelope',
            'cattle',
            'whale',
            'coffee maker',
            'baby bed',
            'frog',
            'bathing cap',
            'crutch',
            'koala bear',
            'tie',
            'dumbbell',
            'tiger',
            'dragonfly',
            'goldfish',
            'cucumber',
            'turtle',
            'harp',
            'jellyfish',
            'swine',
            'pretzel',
            'motorcycle',
            'beaker',
            'rabbit',
            'nail',
            'axe',
            'salt or pepper shaker',
            'croquet ball',
            'skunk',
            'starfish'
            )


    def load_annotations(self, ann_file):
        cat2label = {
                    'cup or mug': 0,
                    'bird': 1,
                    'hat with a wide brim': 2,
                    'person': 3,
                    'dog': 4,
                    'lizard': 5,
                    'sheep': 6,
                    'wine bottle': 7,
                    'bowl': 8,
                    'airplane': 9,
                    'domestic cat': 10,
                    'car': 11,
                    'porcupine': 12,
                    'bear': 13,
                    'tape player': 14,
                    'ray': 15,
                    'laptop': 16,
                    'zebra': 17,
                    'computer keyboard': 18,
                    'pitcher': 19,
                    'artichoke': 20,
                    'tv or monitor': 21,
                    'table': 22,
                    'chair': 23,
                    'helmet': 24,
                    'traffic light': 25,
                    'red panda': 26,
                    'sunglasses': 27,
                    'lamp': 28,
                    'bicycle': 29,
                    'backpack': 30,
                    'mushroom': 31,
                    'fox': 32,
                    'otter': 33,
                    'guitar': 34,
                    'microphone': 35,
                    'strawberry': 36,
                    'stove': 37,
                    'violin': 38,
                    'bookshelf': 39,
                    'sofa': 40,
                    'bell pepper': 41,
                    'bagel': 42,
                    'lemon': 43,
                    'orange': 44,
                    'bench': 45,
                    'piano': 46,
                    'flower pot': 47,
                    'butterfly': 48,
                    'purse': 49,
                    'pomegranate': 50,
                    'train': 51,
                    'drum': 52,
                    'hippopotamus': 53,
                    'ski': 54,
                    'ladybug': 55,
                    'banana': 56,
                    'monkey': 57,
                    'bus': 58,
                    'miniskirt': 59,
                    'camel': 60,
                    'cream': 61,
                    'lobster': 62,
                    'seal': 63,
                    'horse': 64,
                    'cart': 65,
                    'elephant': 66,
                    'snake': 67,
                    'fig': 68,
                    'watercraft': 69,
                    'apple': 70,
                    'antelope': 71,
                    'cattle': 72,
                    'whale': 73,
                    'coffee maker': 74,
                    'baby bed': 75,
                    'frog': 76,
                    'bathing cap': 77,
                    'crutch': 78,
                    'koala bear': 79,
                    'tie': 80,
                    'dumbbell': 81,
                    'tiger': 82,
                    'dragonfly': 83,
                    'goldfish': 84,
                    'cucumber': 85,
                    'turtle': 86,
                    'harp': 87,
                    'jellyfish': 88,
                    'swine': 89,
                    'pretzel': 90,
                    'motorcycle': 91,
                    'beaker': 92,
                    'rabbit': 93,
                    'nail': 94,
                    'axe': 95,
                    'salt or pepper shaker': 96,
                    'croquet ball': 97,
                    'skunk': 98,
                    'starfish': 99
                    }

        # load image list from file
        image_list = [i for i in range(1,30001)] if "train" in ann_file  else [i for i in range(30001, 50001)]
        #image_list = mmcv.list_from_file(self.ann_file)

    
        data_infos = []
        # convert annotations to middle format
        for image_id in image_list:
            filename = f'{self.img_prefix}/{image_id}.JPEG'
            image = mmcv.imread(filename)
            height, width = image.shape[:2]
    
            data_info = dict(filename=f'{image_id}.JPEG', width=width, height=height)
    
            # load annotations
            label_prefix = self.img_prefix.replace('images', 'labels')
            lines = yaml.load(osp.join(label_prefix, f'{image_id}.yml'), Loader=yaml.FullLoader)
    
            bbox_names = [lines["labels"]]
            bboxes = [[float(b) for b in box] for box in lines["bboxes"]]
            gt_bboxes = []
            gt_labels = []
            gt_bboxes_ignore = []
            gt_labels_ignore = []
    
            # filter 'DontCare'
            for bbox_name, bbox in zip(bbox_names, bboxes):
                if bbox_name in cat2label:
                    gt_labels.append(cat2label[bbox_name])
                    gt_bboxes.append(bbox)
                else:
                    gt_labels_ignore.append(-1)
                    gt_bboxes_ignore.append(bbox)

            data_anno = dict(
                bboxes=np.array(gt_bboxes, dtype=np.float32).reshape(-1, 4),
                labels=np.array(gt_labels, dtype=np.long),
                bboxes_ignore=np.array(gt_bboxes_ignore,
                                       dtype=np.float32).reshape(-1, 4),
                labels_ignore=np.array(gt_labels_ignore, dtype=np.long))

            data_info.update(ann=data_anno)
            data_infos.append(data_info)

        return data_infos

def build_cofig():
    cfg = Config.fromfile("Swin-Transformer-Object-Detection/configs/swin/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_1x_coco_custom.py")
    cfg.dataset_type = 'MyDataset'
    cfg.data_root = '/labeled/'

    cfg.data.test.type = 'MyDataset'
    cfg.data.test.data_root = '/labeled/'
    cfg.data.test.ann_file = 'train'
    cfg.data.test.img_prefix = 'training/images'

    cfg.data.train.type = 'MyDataset'
    cfg.data.train.data_root = '/labeled/'
    cfg.data.train.ann_file = 'train'
    cfg.data.train.img_prefix = 'training/images'

    cfg.data.val.type = 'MyDataset'
    cfg.data.val.data_root = '/labeled/'
    cfg.data.val.ann_file = 'val'
    cfg.data.val.img_prefix = 'validation/images'

    # modify num classes of the model in box head
    cfg.model.roi_head.bbox_head.num_classes = 100
    # If we need to finetune a model based on a pre-trained detector, we need to
    # use load_from to set the path of checkpoints.
    #cfg.load_from = 'checkpoints/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco_20210526_095054-1f77628b.pth'

    # Set up working dir to save files and logs.
    cfg.work_dir = './swin_test'

    # The original learning rate (LR) is set for 8-GPU training.
    # We divide it by 8 since we only use one GPU.
    cfg.lr_config.warmup = 10000
    cfg.log_config.interval = 10

    # Change the evaluation metric since we use customized dataset.
    cfg.evaluation.metric = 'mAP'
    # We can set the evaluation interval to reduce the evaluation times
    cfg.evaluation.interval = 12
    # We can set the checkpoint saving interval to reduce the storage cost
    cfg.checkpoint_config.interval = 12

    # Set seed thus the results are more reproducible
    cfg.seed = 0
    set_random_seed(0, deterministic=False)
    cfg.gpu_ids = range(1)

    # We can also use tensorboard to log the training process
    cfg.log_config.hooks = [
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')]


    # We can initialize the logger for training and have a look
    # at the final config used for training
    print(f'Config:\n{cfg.pretty_text}')    
    return cfg

def main():
    # Build dataset
    cfg = build_cofig()
    datasets = [build_dataset(cfg.data.train)]

    # Build the detector
    model = build_detector(cfg.model)
    # Add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES

    # Create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    train_detector(model, datasets, cfg, distributed=False, validate=True)

if __name__ == "__main__":
    main()

