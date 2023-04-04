import argparse
import copy
import os
import os.path as osp
import time
import warnings
import mmcv
import torch
import torch.distributed as dist
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import get_git_hash




import nsml 
from nsml import DATASET_PATH, IS_ON_NSML
import pandas as pd
import shutil
from tqdm import tqdm
import json
from PIL import Image
import git
def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config',type=str, default='cascadeRCNN_swinL.py', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--auto-resume',
        action='store_true',
        help='resume from the latest checkpoint automatically')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='(Deprecated, please use --gpu-id) number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument(
        '--diff-seed',
        action='store_true',
        help='Whether or not set different seeds for different ranks')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        default=True,
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
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    #NSML
    parser.add_argument("--pause", type=int, default=0)
    parser.add_argument("--num_classes", type=int, default=144)
    parser.add_argument("--mode", type=str, default="train")
    
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
    ## data processing##
    if args.mode=='train':
        print("data processing start")
        train_df=pd.read_csv('train_label.csv',sep=",")
        group=pd.DataFrame(train_df.groupby(['class_idx'])['image_file_name'].apply(list)).reset_index()
        ratio=0.7
        dataset_t=dict()
        dataset_v=dict()
        dataset_t['image_file_name']=[]
        dataset_v['image_file_name']=[]
        for files in group['image_file_name']:
            dataset_t['image_file_name'].extend(files[:int(len(files)*ratio)])
            dataset_v['image_file_name'].extend(files[int(len(files)*ratio):])
        os.makedirs("data/train2017", exist_ok=True)
        os.makedirs("data/val2017", exist_ok=True)
        os.makedirs("data/annotations", exist_ok=True)

        for name in tqdm(dataset_t['image_file_name']):
            shutil.copy2(os.path.join(DATASET_PATH, "train", "train_data/")+name,'data/train2017/'+name)
        for name in tqdm(dataset_v['image_file_name']):
            shutil.copy2(os.path.join(DATASET_PATH, "train", "train_data/")+name,'data/val2017/'+name)

        classes =   ('place1', 'place2', 'place3', 'place4', 'place5', 'place6', 'place7', 'place8', 'place9', 'place10', 
                    'place11', 'place12', 'place13', 'place14', 'place15', 'place16', 'place17', 'place18', 'place19', 'place20', 
                    'place21', 'place22', 'place23', 'place24', 'place25', 'place26', 'place27', 'place28', 'place29', 'place30', 
                    'place31', 'place32', 'place33', 'place34', 'place35', 'place36', 'place37', 'place38', 'place39', 'place40', 
                    'place41', 'place42', 'place43', 'place44', 'place45', 'place46', 'place47', 'place48', 'place49', 'place50', 
                    'place51', 'place52', 'place53', 'place54', 'place55', 'place56', 'place57', 'place58', 'place59', 'place60', 
                    'place61', 'place62', 'place63', 'place64', 'place65', 'place66', 'place67', 'place68', 'place69', 'place70', 
                    'place71', 'place72', 'place73', 'place74', 'place75', 'place76', 'place77', 'place78', 'place79', 'place80', 
                    'place81', 'place82', 'place83', 'place84', 'place85', 'place86', 'place87', 'place88', 'place89', 'place90', 
                    'place91', 'place92', 'place93', 'place94', 'place95', 'place96', 'place97', 'place98', 'place99', 'place100', 
                    'place101', 'place102', 'place103', 'place104', 'place105', 'place106', 'place107', 'place108', 'place109', 
                    'place110', 'place111', 'place112', 'place113', 'place114', 'place115', 'place116', 'place117', 'place118', 
                    'place119', 'place120', 'place121', 'place122', 'place123', 'place124', 'place125', 'place126', 'place127', 
                    'place128', 'place129', 'place130', 'place131', 'place132', 'place133', 'place134', 'place135', 'place136', 
                    'place137', 'place138', 'place139', 'place140', 'place141', 'place142', 'place143')

        categories=[{'id':idx+1,'name':name} for idx,name in enumerate(classes)]
        images=[]
        annotations=[]
        file_path='data/train2017'
        for idx,file in enumerate(dataset_t['image_file_name']):
            info=train_df[train_df['image_file_name']==file]
            im = Image.open(file_path+'/'+file)
            w, h = im.size
            image = {"id": int(info.image_idx),
                    "width": int(w),
                    "height": int(h),
                    "file_name": str(file)
                    }
            images.append(image)
            annotation = {
                    "id":idx,
                    "image_id": int(info.image_idx),
                    "category_id": int(info.class_idx),
                    "area": (float(info.roi_x1)-float(info.roi_x0))*(float(info.roi_y1)-float(info.roi_y0)),
                    "segmentation": [[float(info.roi_x0),float(info.roi_y0),float(info.roi_x1),float(info.roi_y1)]],
                    "bbox": [float(info.roi_x0),float(info.roi_y0),float(info.roi_x1)-float(info.roi_x0),float(info.roi_y1)-float(info.roi_y0)],
                    "iscrowd": 0 
                    }
            annotations.append(annotation)
        results={}
        results['images']=images
        results['annotations']=annotations
        results['categories']=categories
        file_path='data/annotations/instances_train2017.json'
        with open(file_path, 'w') as f:
            json.dump(results, f,indent="\t")

        images=[]
        annotations=[]
        file_path='data/val2017'
        for idx,file in enumerate(dataset_v['image_file_name']):
            info=train_df[train_df['image_file_name']==file]
            im = Image.open(file_path+'/'+file)
            w, h = im.size
            image = {"id": int(info.image_idx),
                    "width": int(w),
                    "height": int(h),
                    "file_name": str(file)
                    }
            images.append(image)
            annotation = {
                    "id":idx,
                    "image_id": int(info.image_idx),
                    "category_id": int(info.class_idx),
                    "area": (float(info.roi_x1)-float(info.roi_x0))*(float(info.roi_y1)-float(info.roi_y0)),
                    "segmentation": [[float(info.roi_x0),float(info.roi_y0),float(info.roi_x1),float(info.roi_y1)]],
                    "bbox": [float(info.roi_x0),float(info.roi_y0),float(info.roi_x1)-float(info.roi_x0),float(info.roi_y1)-float(info.roi_y0)],
                    "iscrowd": 0 
                    }
            annotations.append(annotation)
        results={}
        results['images']=images
        results['annotations']=annotations
        results['categories']=categories
        file_path='data/annotations/instances_val2017.json'
        with open(file_path, 'w') as f:
            json.dump(results, f,indent="\t")
    print("data processing done")
    cfg = Config.fromfile(args.config)

    # replace the ${key} with the value of cfg.key
    cfg = replace_cfg_vals(cfg)

    # update data root according to MMDET_DATASETS
    update_data_root(cfg)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            warnings.warn('Can not find "auto_scale_lr" or '
                          '"auto_scale_lr.enable" or '
                          '"auto_scale_lr.base_batch_size" in your'
                          ' configuration file. Please update all the '
                          'configuration files to mmdet >= 2.24.1.')

    # set multi-process settings
    setup_multi_processes(cfg)

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
    cfg.auto_resume = args.auto_resume
    if args.gpus is not None:
        cfg.gpu_ids = range(1)
        warnings.warn('`--gpus` is deprecated because we only support '
                      'single GPU mode in non-distributed training. '
                      'Use `gpus=1` now.')
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed training. Use the first GPU '
                      'in `gpu_ids` now.')
    if args.gpus is None and args.gpu_ids is None:
        cfg.gpu_ids = [args.gpu_id]

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

    cfg.device = get_device()
    # set random seeds
    seed = init_random_seed(args.seed, device=cfg.device)
    seed = seed + dist.get_rank() if args.diff_seed else seed
    logger.info(f'Set random seed to {seed}, '
                f'deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    meta['seed'] = seed
    meta['exp_name'] = osp.basename(args.config)

    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()

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
    
    git.Git().clone('https://github.com/open-mmlab/mmdetection.git')
    git.Git().clone('https://github.com/open-mmlab/mmclassification.git')
    shutil.move('coco.py','mmdetection/mmdet/datasets/coco.py')
    shutil.move('mmdetection/mmdet','.')
    shutil.move('mmclassification/mmcls','.')
    from mmdet import __version__
    from mmdet.apis import init_random_seed, set_random_seed, train_detector
    from mmdet.datasets import build_dataset
    from mmdet.models import build_detector
    from mmdet.utils import (collect_env, get_device, get_root_logger,
                         replace_cfg_vals, setup_multi_processes,
                         update_data_root)
    import mmcls
    main()
    folder_path='mmdetection'
    nsml.save_folder('mm','work_dirs')

    
