import copy
import os.path as osp
import yaml

import mmcv
import numpy as np

from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset
from mmdet.apis import set_random_seed
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector

import argparse
import os
import time
import warnings

import torch
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import get_git_hash

from mmdet import __version__
from mmdet.apis import set_random_seed, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import collect_env, get_root_logger

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
        image_list = [i for i in range(1,30001)] if "training" in ann_file else [i for i in range(30001, 50001)]
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
            with open(osp.join(label_prefix, f'{image_id}.yml'), "rb") as f:
                lines = yaml.load(f, Loader=yaml.FullLoader)
            bbox_names = [l for l in lines["labels"]]
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
    cfg.bbox_head.num_classes = 100
    cfg.model.backbone.pretrained = "../../Obj_SSL_barlow/checkpoint/checkpoint2.pth"

    cfg.img_norm_cfg.mean = [124.95,119.34,105.57]
    cfg.img_norm_cfg.std = [72.93,70.89,75.735]
    cfg.train_pipeline[3].policies[0][0].img_scale=[(224,1333),(256,1333),(288,1333),(320,1333)]
    cfg.train_pipeline[3].policies[1][2].img_scale=[(224,1333),(256,1333),(288,1333),(320,1333)]
    cfg.test_pipeline[1].img_scale=(1333,224)
    cfg.data.train.pipeline = cfg.train_pipeline
    cfg.data.val.pipeline = cfg.test_pipeline
    cfg.data.test.pipeline = cfg.test_pipeline

    # If we need to finetune a model based on a pre-trained detector, we need to
    # use load_from to set the path of checkpoints.
    #cfg.load_from = 'checkpoints/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco_20210526_095054-1f77628b.pth'

    # Set up working dir to save files and logs.
    cfg.work_dir = './detr_res_test'

    # The original learning rate (LR) is set for 8-GPU training.
    # We divide it by 8 since we only use one GPU.
    cfg.log_config.interval = 10

    # Change the evaluation metric since we use customized dataset.
    cfg.evaluation.metric = 'mAP'
    # We can set the evaluation interval to reduce the evaluation times
    cfg.evaluation.interval = 3
    # We can set the checkpoint saving interval to reduce the storage cost
    cfg.checkpoint_config.interval = 3

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


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both '
            'specified, --options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    cfg = build_cofig(cfg)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text
    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.config)

    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__ + get_git_hash()[:7],
            CLASSES=datasets[0].CLASSES)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    train_detector(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()
