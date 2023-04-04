from tkinter import W
from ensemble_boxes import *

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
def bind_nsml():
    def load(model_dir, **kwargs):
        print('Model loaded')

    def save(model_dir, **kwargs):
        print('Saved')

    def infer(root_path, top_k=1):
        # ids=open('ids.json')
        # ids=json.loads(ids.read())
        # preds=open('preds.json')
        # preds=json.loads(preds.read())
        # for idx,info in enumerate(preds):
        #     print(info)        
        # for idx,info in enumerate(ids):
        #     print(info)

        ids = np.load('ids.npy' ,allow_pickle=True)
        preds = np.load('preds.npy',allow_pickle=True)

        return {"ids": ids, "preds": preds}

    nsml.bind(save=save, load=load, infer=infer) 
def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('--config',type=str,default='cascade.py', help='train config file path')
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

def main(model_name):

    args = parse_args()
    args.config=model_name+'.py'
    args.checkpoint=model_name+'.pth'
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
        
        result_files, tmp_dir=dataset.format_results(outputs, jsonfile_prefix=model_name)
        print("result json made")

def load_model(model_dir):
    print(model_dir)
    folder_dir=model_dir+'/'+os.listdir(model_dir)[0]
    print(folder_dir)
    idx=0
    file_name=os.listdir(model_dir)[0]
    print(file_name)

    if not os.path.isfile(file_name+'.pth'):
        shutil.move(folder_dir+'/'+'latest.pth',file_name+'.pth')
    else:
        while os.path.isfile(file_name+'.pth'):
            file_name+=str(idx)
            idx+=1
    
        shutil.move(folder_dir+'/'+'latest.pth',file_name+'.pth')
        file_name=file_name[:-1]
    print(os.listdir('.'))




weights = [1,1,1,1,1]

iou_thr = 0.65
skip_box_thr = 0.0001


if __name__ == '__main__':
        args = argparse.ArgumentParser()
        args.add_argument("--mode", type=str, default="train")
        args.add_argument("--iteration", type=str, default='0')
        args.add_argument("--pause", type=int, default=0)
        args.add_argument("--seed",type=int,default=42)
        config = args.parse_args()

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
        nsml.load(session='KR96387/airush2022-2-2/1236',checkpoint='mm',load_fn=load_model)
        nsml.load(session='KR96387/airush2022-2-2/1048',checkpoint='mm',load_fn=load_model)
        nsml.load(session='KR96387/airush2022-2-2/1381',checkpoint='mm',load_fn=load_model)
        nsml.load(session='KR96387/airush2022-2-2/1609',checkpoint='mm',load_fn=load_model)
        nsml.load(session='KR96387/airush2022-2-2/1367',checkpoint='mm',load_fn=load_model)
        main('cascade')
        main('drcnn')
        main('sparse')
        main('sparse0')
        main('cascadeRCNN_swinL')


        file_path='data/annotations/instances_val2017.json'
        size_info=open(file_path)
        size_info=json.loads(size_info.read())
        size_info=pd.DataFrame(size_info['images'])
        size_info.rename(columns={'id':'image_id'},inplace=True)
        file_list = os.listdir('.')
        json_list = [file for file in file_list if file.endswith(".json")]
        print(json_list)
    
        import pandas as pd
        boxes=[]
        labels=[]
        scores=[]
        # 1st
        df1=pd.read_json(json_list[0])
        df1['image_id']=[str(x) for x in df1['image_id']]

        for i in df1['bbox']:
            i[2]+=i[0]
            i[3]+=i[1]
        df1=pd.merge(df1,size_info,how='outer')

        df1['bbox']=df1['bbox'].fillna({i: [0,0,0,0] for i in df1.index})
        df1['score']=df1['score'].fillna(0)
        df1['category_id']=df1['category_id'].fillna(0)

        df1=df1.sort_values('image_id')

        #2nd
        df2=pd.read_json(json_list[1])
        df2['image_id']=[str(x) for x in df2['image_id']]
        for i in df2['bbox']:
            i[2]+=i[0]
            i[3]+=i[1]
        df2=pd.merge(df2,size_info,how='outer')

        df2['bbox']=df2['bbox'].fillna({i: [0,0,0,0] for i in df2.index})
        df2['score']=df2['score'].fillna(0)
        df2['category_id']=df2['category_id'].fillna(0)
        df2=df2.sort_values('image_id')

        #3rd
        df3=pd.read_json(json_list[2])
        df3['image_id']=[str(x) for x in df3['image_id']]
        for i in df3['bbox']:
            i[2]+=i[0]
            i[3]+=i[1]
        df3=pd.merge(df3,size_info,how='outer')

        df3['bbox']=df3['bbox'].fillna({i: [0,0,0,0] for i in df3.index})
        df3['score']=df3['score'].fillna(0)
        df3['category_id']=df3['category_id'].fillna(0)
        df3=df3.sort_values('image_id')

        #4th
        df4=pd.read_json(json_list[3])
        df4['image_id']=[str(x) for x in df4['image_id']]
        for i in df4['bbox']:
            i[2]+=i[0]
            i[3]+=i[1]
        df4=pd.merge(df4,size_info,how='outer')

        df4['bbox']=df4['bbox'].fillna({i: [0,0,0,0] for i in df4.index})
        df4['score']=df4['score'].fillna(0)
        df4['category_id']=df4['category_id'].fillna(0)
        df4=df4.sort_values('image_id')        
        
        #5th
        df5=pd.read_json(json_list[3])
        df5['image_id']=[str(x) for x in df5['image_id']]
        for i in df5['bbox']:
            i[2]+=i[0]
            i[3]+=i[1]
        df5=pd.merge(df5,size_info,how='outer')

        df5['bbox']=df5['bbox'].fillna({i: [0,0,0,0] for i in df5.index})
        df5['score']=df5['score'].fillna(0)
        df5['category_id']=df5['category_id'].fillna(0)
        df5=df5.sort_values('image_id')


        ids=[]
        preds=[]
        for row1,row2,row3,row4,row5 in zip(df1.iterrows(),df2.iterrows(),df3.iterrows(),df4.iterrows(),df5.iterrows()):
            row1=row1[1]
            row2=row2[1]
            row3=row3[1]
            row4=row4[1]
            row5=row5[1]

            id=[np.array(row1['image_id'])]
            w,h=row1['width'],row1['height']
            row1['bbox'][0]/=w
            row1['bbox'][1]/=h
            row1['bbox'][2]/=w
            row1['bbox'][3]/=h

            row2['bbox'][0]/=w
            row2['bbox'][1]/=h
            row2['bbox'][2]/=w
            row2['bbox'][3]/=h
            
            row3['bbox'][0]/=w
            row3['bbox'][1]/=h
            row3['bbox'][2]/=w
            row3['bbox'][3]/=h   

            row4['bbox'][0]/=w
            row4['bbox'][1]/=h
            row4['bbox'][2]/=w
            row4['bbox'][3]/=h

            row5['bbox'][0]/=w
            row5['bbox'][1]/=h
            row5['bbox'][2]/=w
            row5['bbox'][3]/=h

            boxes=[[row1['bbox']],[row2['bbox']],[row3['bbox']],[row4['bbox']],[row5['bbox']]]
            print(boxes)
            scores=[[row1['score']],[row2['score']],[row3['score']],[row4['score']],[row5['score']]]
            labels=[[row1['category_id']],[row2['category_id']],[row3['category_id']],[row4['category_id']],[row5['category_id']]]
            boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
            for box in boxes:
                box[0]*=w
                box[1]*=h
                box[2]*=w
                box[3]*=h
            boxes=np.array(boxes,dtype='f')
            labels=np.array(labels).astype(int)
            scores=np.array(scores,dtype='f')
            pred=[{'boxes': boxes, 'labels': labels, 'scores': scores}]
            if len(ids)==0:
                ids=id
                preds=pred
            else:
                ids=np.concatenate((ids,id),axis=0)
                preds=np.concatenate((preds,pred),axis=0)
        np.save('ids', ids) 
        np.save('preds', preds) 




        if IS_ON_NSML:
            bind_nsml()
            if config.pause:
                nsml.paused(scope=locals())
        