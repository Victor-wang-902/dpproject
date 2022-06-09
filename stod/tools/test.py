import copy
import os.path as osp
import yaml
import numpy as np
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset
from mmdet.apis import set_random_seed
import argparse
import os
import warnings
import time
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import get_git_hash

from mmdet import __version__
from mmdet.utils import collect_env, get_root_logger

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
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
    cfg.model.roi_head.bbox_head.num_classes = 100
    cfg.model.backbone.pretrained="../../Obj_SSL_barlow/checkpoint/swin.pth"
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

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
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
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both '
            'specified, --options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    cfg = build_cfg(cfg)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None

    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                  args.show_score_thr)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            print(dataset.evaluate(outputs, **eval_kwargs))


if __name__ == '__main__':
    main()
