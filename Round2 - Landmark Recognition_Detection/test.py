
import argparse
from logging import root
import os
import os.path as osp
from pickle import TRUE
import time
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)


import nsml 
from nsml import DATASET_PATH, IS_ON_NSML
import pandas as pd
import shutil
from tqdm import tqdm
import json
from PIL import Image
import git
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('--config',type=str,default='cascadeRCNN_swinL.py', help='train config file path')
    parser.add_argument('--checkpoint',type=str,default='epoch.pth', help='checkpoint file')
    parser.add_argument(
        '--work-dir',type=str,default='results',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--format-only',default=TRUE,
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
        # type=str,
        # default='jsonfile_prefix=./results',
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

    #NSML
    parser.add_argument("--pause", type=int, default=0)
    parser.add_argument("--num_classes", type=int, default=144)
    parser.add_argument("--mode", type=str, default="test")

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

def bind_nsml():
    def load(model_dir, **kwargs):
        print('Model loaded')

    def save(model_dir, **kwargs):
        print('Saved')

    def infer(root_path, top_k=1):

        f=open('results.bbox.json')
        result_dict=json.loads(f.read())
        ids=[]
        preds=[]
        for idx,info in enumerate(result_dict):

            id=[np.array(info['image_id'])]

            temp=info['bbox']
            temp[2]+=temp[0]
            temp[3]+=temp[1]
            boxes=np.array([temp],dtype='f')
            labels=np.array([info['category_id']]).astype(int)
            scores=np.array([info['score']],dtype='f')

            pred=[{'boxes': boxes, 'labels': labels, 'scores': scores}]

            if idx ==0:
                ids=id
                preds=pred
            else:
                ids=np.concatenate((ids,id),axis=0)
                preds=np.concatenate((preds,pred),axis=0)


        return {"ids": ids, "preds": preds}

    nsml.bind(save=save, load=load, infer=infer) 

def load_model(model_dir):
    folder_dir=model_dir+'/'+os.listdir(model_dir)[0]
    shutil.move(folder_dir+'/'+'best_bbox_mAP_75_epoch_5.pth','epoch.pth')

    
def main():
    git.Git().clone('https://github.com/open-mmlab/mmdetection.git')
    shutil.move('coco.py','mmdetection/mmdet/datasets/coco.py')
    shutil.move('mmdetection/mmdet','.')
    from mmdet import __version__
    from mmdet.apis import multi_gpu_test, single_gpu_test
    from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
    from mmdet.models import build_detector
    from mmdet.utils import (build_ddp, build_dp, compat_cfg, get_device,
                         replace_cfg_vals, setup_multi_processes,
                         update_data_root)

    args = parse_args()
    nsml.load(session='KR96387/airush2022-2-2/1646',checkpoint='mm',load_fn=load_model)

    # data processing##
    if args.mode=='test':
        root_path = os.path.join(DATASET_PATH, 'test')
        images_dir=os.listdir(root_path+'/'+'test_data')
    


        os.makedirs("data/val2017", exist_ok=True)
        os.makedirs("data/annotations", exist_ok=True)

        for image in tqdm(images_dir):
            shutil.copy2(root_path+'/'+'test_data/'+image,'data/val2017/'+image)



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

        file_path='data/val2017/'
        for idx,file in enumerate(images_dir):

            im = Image.open(file_path+file)
            w, h = im.size
            image = {"id": str(int(file.split("_")[0])),
                    "width": int(w),
                    "height": int(h),
                    "file_name": str(file)
                    }
            images.append(image)
        results={}
        results['images']=images
        results['categories']=categories

        file_path='data/annotations/instances_val2017.json'
        with open(file_path, 'w') as f:
            json.dump(results, f,indent="\t")
        print("data processing done")


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
    
        # replace the ${key} with the value of cfg.key
        cfg = replace_cfg_vals(cfg)
    
        # update data root according to MMDET_DATASETS
        update_data_root(cfg)
    
        if args.cfg_options is not None:
            cfg.merge_from_dict(args.cfg_options)
    
        cfg = compat_cfg(cfg)
    
        # set multi-process settings
        setup_multi_processes(cfg)
    
        # set cudnn_benchmark
        if cfg.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True
    
        if 'pretrained' in cfg.model:
            cfg.model.pretrained = None
        elif 'init_cfg' in cfg.model.backbone:
            cfg.model.backbone.init_cfg = None
    
        if cfg.model.get('neck'):
            if isinstance(cfg.model.neck, list):
                for neck_cfg in cfg.model.neck:
                    if neck_cfg.get('rfp_backbone'):
                        if neck_cfg.rfp_backbone.get('pretrained'):
                            neck_cfg.rfp_backbone.pretrained = None
            elif cfg.model.neck.get('rfp_backbone'):
                if cfg.model.neck.rfp_backbone.get('pretrained'):
                    cfg.model.neck.rfp_backbone.pretrained = None
    
        if args.gpu_ids is not None:
            cfg.gpu_ids = args.gpu_ids[0:1]
            warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                          'Because we only support single GPU mode in '
                          'non-distributed testing. Use the first GPU '
                          'in `gpu_ids` now.')
        else:
            cfg.gpu_ids = [args.gpu_id]
        cfg.device = get_device()
        # init distributed env first, since logger depends on the dist info.
        if args.launcher == 'none':
            distributed = False
        else:
            distributed = True
            init_dist(args.launcher, **cfg.dist_params)
    
        test_dataloader_default_args = dict(
            samples_per_gpu=1, workers_per_gpu=2, dist=distributed, shuffle=False)
    
        # in case the test dataset is concatenated
        if isinstance(cfg.data.test, dict):
            cfg.data.test.test_mode = True
            if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
                # Replace 'ImageToTensor' to 'DefaultFormatBundle'
                cfg.data.test.pipeline = replace_ImageToTensor(
                    cfg.data.test.pipeline)
        elif isinstance(cfg.data.test, list):
            for ds_cfg in cfg.data.test:
                ds_cfg.test_mode = True
            if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
                for ds_cfg in cfg.data.test:
                    ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)
    
        test_loader_cfg = {
            **test_dataloader_default_args,
            **cfg.data.get('test_dataloader', {})
        }
    
        rank, _ = get_dist_info()
        # allows not to create
        if args.work_dir is not None and rank == 0:
    
            print(osp.abspath(args.work_dir))
            mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
            timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
            
            json_file = osp.join(args.work_dir, f'eval_{timestamp}.json')
    
    
        # build the dataloader
        dataset = build_dataset(cfg.data.test)
        data_loader = build_dataloader(dataset, **test_loader_cfg)
    
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
            model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)
            outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                      args.show_score_thr)
        else:
            model = build_ddp(
                model,
                cfg.device,
                device_ids=[int(os.environ['LOCAL_RANK'])],
                broadcast_buffers=False)
            outputs = multi_gpu_test(
                model, data_loader, args.tmpdir, args.gpu_collect
                or cfg.evaluation.get('gpu_collect', False))
        
        result_files, tmp_dir=dataset.format_results(outputs, jsonfile_prefix='results')
        print("result json made")

    
    


if __name__ == '__main__':
        args = argparse.ArgumentParser()
        args.add_argument("--mode", type=str, default="train")
        args.add_argument("--iteration", type=str, default='0')
        args.add_argument("--pause", type=int, default=0)
        args.add_argument("--seed",type=int,default=42)
        config = args.parse_args()


        main()

        if IS_ON_NSML:
            bind_nsml()
            if config.pause:
                nsml.paused(scope=locals())
