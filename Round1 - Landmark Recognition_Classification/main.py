import nsml
from nsml import DATASET_PATH, IS_ON_NSML
import random
import unicodedata
import os
import sys
import time
import numpy as np
import pandas as pd
import cv2
import PIL.Image
 
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
 
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler, RandomSampler, SequentialSampler
 
import albumentations as A
 
from sklearn.model_selection import StratifiedKFold




from datetime import datetime

print(datetime.now())

import random
import timm
import albumentations as A
from model import Net
from loss import ArcFaceLoss,loss_fn
from scheduler import GradualWarmupScheduler
from trainer import train_epoch,val_end,val_epoch
from args import args,tr_aug,val_aug
from evaluation import set_seed
from landmark_dataset import LandmarkRecognitionDataset2,LandmarkRecognitionDataset3

import wandb
os.environ["WANDB_API_KEY"]='d616471daf61d4547785b10eaa219b3b17c72e7e'
wandb.init(project="Landmark", entity="ili0820")
wandb.run.name='kaggle_arcface'+str(args.batch_size)+str(args.img_size)
wandb.run.save()
wandb.log({"embedding_size": args.embedding_size})
wandb.log({"arcface_s": args.arcface_s})
wandb.log({"arcface_m": args.arcface_m})
wandb.log({"batch_size": args.batch_size})
wandb.log({"n_splits": args.n_splits})
wandb.log({"out_dim": args.out_dim})
wandb.log({"img_size": args.img_size})
import argparse
print(args)
def main():
    set_seed(args.seed)
    arg = argparse.ArgumentParser()

    arg.add_argument("--mode", type=str, default='train')
    arg.add_argument("--pause", type=int, default=0)
    config = arg.parse_args()

    if config.mode=='train':
        csvfile = open(os.path.join(DATASET_PATH, "train", "train_label"), newline='', encoding='utf-8')
        import csv
        csvread = csv.reader(csvfile, delimiter=',')


        # 'image_idx','landmark_id','class_id','id'
        train = pd.DataFrame(csvread)

        train.columns = ['image_id','class_id','_','_','file_name']
        train['class_id']=train['class_id'].astype(int)
        train['image_id']=train['image_id'].astype(int)
        skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
        train['fold'] = 0

        for idx, [trn, val] in enumerate(skf.split(train, train['class_id'])):
            train.loc[val, 'fold'] = idx
        train['filepath'] =  [os.path.join(os.path.join(DATASET_PATH, "train", "train_data"),id) for lm_id, id in zip(train['class_id'], train['file_name'])]
        # train['filepath'] = train['filepath'].apply(lambda x: unicodedata.normalize('NFD', x))
        print(train)

        if args.class_weights == "log":
                val_counts = train.class_id.value_counts().sort_index().values
                class_weights = 1/np.log1p(val_counts)
                class_weights = (class_weights / class_weights.sum()) * args.n_classes
                class_weights = torch.tensor(class_weights, dtype=torch.float32)
        else:
            class_weights = None
    
        trn = train.loc[train['fold']!=args.fold].reset_index(drop=True)
        val = train.loc[train['fold']==args.fold].reset_index(drop=True)

    
        print(f'trn size : {trn.class_id.nunique()}, last batch size : {trn.shape[0]%args.batch_size}') #: 1049
        # # print(len(trn)) #: 70481
        # # image size : (540, 960, 3)

        if args.DEBUG:
            trn = trn.iloc[:2500]
            val = val.iloc[:2500]

        train_dataset = LandmarkRecognitionDataset2(trn, aug=tr_aug, normalization=args.normalization)
        valid_dataset = LandmarkRecognitionDataset2(val, aug=val_aug, normalization=args.normalization)
    
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=0, shuffle=True, pin_memory=True)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size, num_workers=0, shuffle=False, pin_memory=True)
        from ptflops import get_model_complexity_info
        with torch.cuda.device(0):
            model  = Net(args)
            if torch.cuda.is_available():
                model.cuda()
            macs, params = get_model_complexity_info(model, (3, args.img_size, args.img_size), as_strings=True,
                                               print_per_layer_stat=True, verbose=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))
        cuda = torch.cuda.is_available()
        wandb.watch(model)
          # optimizer definition
        metric_crit = ArcFaceLoss(args.arcface_s, args.arcface_m, crit=args.crit, weight=class_weights)
        metric_crit_val = ArcFaceLoss(args.arcface_s, args.arcface_m, crit=args.crit, weight=None, reduction="sum")
        if args.optim=='sgd':
            optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': metric_crit.parameters()}], lr=args.lr, momentum=0.9, nesterov=True, weight_decay=args.weight_decay)
        elif args.optim=='adamw':
            optimizer = torch.optim.AdamW([{'params': model.parameters()}, {'params': metric_crit.parameters()}], lr=args.lr, weight_decay=args.weight_decay, amsgrad=False)

        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.cosine_epo)
        scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=args.warmup_epo, after_scheduler=scheduler_cosine)
        if IS_ON_NSML:
            bind_nsml(model, optimizer,scheduler_cosine,scheduler_warmup, cuda)
            # bind_nsml(model, scheduler_cosine,scheduler_warmup, cuda)
            if config.pause:
                nsml.paused(scope=locals())
        
        
        nsml.load(session='KR96387/airush2022-1-4/600',checkpoint=30)
        val_pp = 0.
        best_metric=0
        for epoch in range(1, args.cosine_epo+args.warmup_epo+1):
        
            scheduler_warmup.step(epoch-1)
            print(time.ctime(), 'Epoch:', epoch)
    
            train_loss = train_epoch(metric_crit, epoch, model, train_loader, optimizer)
            val_outputs = val_epoch(metric_crit_val, model, valid_loader)
            results = val_end(val_outputs)
            
            val_loss = results['val_loss']
            val_gap = results['val_gap']
            val_gap_landmarks = results['val_gap_landmarks']
            val_gap_pp = results['val_gap_pp']
            val_gap_landmarks_pp = results['val_gap_landmarks_pp']
            wandb.log({"val_loss": val_loss})
            wandb.log({"val_gap": val_gap})
            wandb.log({"val_gap_landmarks": val_gap_landmarks})
            wandb.log({"val_gap_pp": val_gap_pp})
            wandb.log({"val_gap_landmarks_pp": val_gap_landmarks_pp})
            content = time.ctime() + ' ' + f'Fold {args.fold}, Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, train loss: {train_loss:.5f}, valid loss: {val_loss:.5f}'
            print(content)
            print(f'val_gap: {val_gap}, val_gap_landmarks: {val_gap_landmarks}')
            print(f'val_gap_pp: {val_gap_pp}, val_gap_landmarks_pp: {val_gap_landmarks_pp}')
            

            if IS_ON_NSML:
                                nsml.save(str(epoch + 1))
                                if best_metric<val_gap_landmarks_pp:
                                    print('new best!')
                                    nsml.save('best')
                                    best_metric=val_gap_landmarks_pp


            if val_gap_pp > val_pp:
                print('val_gap_pp_max ({:.6f} --> {:.6f}). Saving model ...'.format(val_pp, val_gap_pp))
                val_pp = val_gap_pp
    else:
        
        model  = Net(args)
        cuda = torch.cuda.is_available()
        if cuda:
                model.cuda()
        
        metric_crit_val = ArcFaceLoss(args.arcface_s, args.arcface_m, crit=args.crit, weight=None, reduction="sum")
        if args.optim=='sgd':
            optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': metric_crit_val.parameters()}], lr=args.lr, momentum=0.9, nesterov=True, weight_decay=args.weight_decay)
        elif args.optim=='adamw':
            optimizer = torch.optim.AdamW([{'params': model.parameters()}, {'params': metric_crit_val.parameters()}], lr=args.lr, weight_decay=args.weight_decay, amsgrad=False)

        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.cosine_epo)
        scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=args.warmup_epo, after_scheduler=scheduler_cosine)
        if IS_ON_NSML:
            print('bind start')
            bind_nsml(model, optimizer,scheduler_cosine,scheduler_warmup, cuda)
            
            if config.pause:
                nsml.paused(scope=locals())


def bind_nsml(model, optimizer,scheduler_cosine,scheduler_warmup, cuda):
    print('binding...')
    def load(model_dir, **kwargs):
        print('load start')
        device = torch.device('cpu')
        if cuda:
            device = torch.device('cuda')
        state = torch.load(os.path.join(model_dir, 'model.pth'), map_location=device)
        model.load_state_dict(state['model'])
        # if 'optimizer' in state:
        #     optimizer.load_state_dict(state['optimizer'])
        if 'scheduler_cosine' in state:
            scheduler_cosine.load_state_dict(state['scheduler_cosine'])
        if 'scheduler_warmup' in state:
            scheduler_warmup.load_state_dict(state['scheduler_warmup'])
        print('Model loaded')


    def save(model_dir, **kwargs):
        os.makedirs(model_dir, exist_ok=True)
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler_cosine': scheduler_cosine.state_dict(),
            'scheduler_warmup': scheduler_warmup.state_dict()
        }
        torch.save(state, os.path.join(model_dir, 'model.pth'))
        print('Saved')

    def infer(input,apply_sigmoid=True):

        print('infer start')
        dataset = LandmarkRecognitionDataset3(os.path.join(input, "test_data"),aug=val_aug, normalization=args.normalization)
        loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size)
        model.eval()

        sig = torch.nn.Sigmoid()
        res_preds = None
        res_ids = None

        for iter, batch in enumerate(tqdm(loader)):
            image=batch['input']
            image_id=batch['image_id']
            if cuda:
                image = image.cuda()
            preds = model(image)['logits']
            if apply_sigmoid:
                sig.to(preds.device)
                preds = sig(preds)
            preds = preds.detach().cpu().numpy()
            if iter == 0:
                res_preds = preds
                res_ids = image_id
            else:
                res_preds = np.concatenate((res_preds, preds), axis=0)
                res_ids = np.concatenate((res_ids, image_id), axis=0)
        print(res_ids)
        print(res_preds)
        return {"ids": res_ids, "preds": res_preds}


    nsml.bind(save=save, load=load, infer=infer) # 'nsml.bind' function must be called at the end.
    print('done...!')


if __name__ == '__main__':
    main()


    


