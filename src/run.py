# !pip -q install git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git
# !pip -q install geffnet
# !pip install -U git+https://github.com/albu/albumentations --no-cache-dir
from conf import *
from loader import *
from models import *
from trainer import *
from loss import *
from scheduler import *

import random

import os
import sys
import time
import numpy as np
import pandas as pd
import cv2
import PIL.Image

from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset

import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler, RandomSampler, SequentialSampler

import albumentations as A
import geffnet

from sklearn.model_selection import StratifiedKFold

def set_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def main():
    set_seed(args.seed)

    train = pd.read_csv('../data/train.csv')
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    train['fold'] = 0
    for idx, [trn, val] in enumerate(skf.split(train, train['label'])):
        train.loc[val, 'fold'] = idx

    if args.class_weights == "log":
        val_counts = train.label.value_counts().sort_index().values
        class_weights = 1/np.log1p(val_counts)
        class_weights = (class_weights / class_weights.sum()) * args.n_classes
        class_weights = torch.tensor(class_weights, dtype=torch.float32)
    else:
        class_weights = None
  
    trn = train.loc[train['fold']!=args.fold].reset_index(drop=True)
    val = train.loc[train['fold']==args.fold].reset_index(drop=True)

    print(f'trn size : {trn.label.nunique()}, last batch size : {trn.shape[0]%args.batch_size}') #: 1049
    # print(len(trn)) #: 70481
    # image size : (540, 960, 3)
    
    if args.DEBUG:
        trn = trn.iloc[:250]
        val = val.iloc[:250]
    
    train_dataset = LMDataset(trn, aug=args.tr_aug, normalization=args.normalization)
    valid_dataset = LMDataset(val, aug=args.val_aug, normalization=args.normalization)

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=False)

    model = Net(args)
    model = model.to(args.device)
    # model.load_state_dict(torch.load('/home/hhl/바탕화면/dacon/dacon21/model/new_train/best_tf_efficientnet_b1_ns_best_fold_0.pth'))

    # optimizer definition
    metric_crit = ArcFaceLoss(args.arcface_s, args.arcface_m, crit=args.crit, weight=class_weights)
    metric_crit_val = ArcFaceLoss(args.arcface_s, args.arcface_m, crit=args.crit, weight=None, reduction="sum")
    if args.optim=='sgd':
        optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': metric_crit.parameters()}], lr=args.lr, momentum=0.9, nesterov=True, weight_decay=args.weight_decay)
    elif args.optim=='adamw':
        optimizer = torch.optim.AdamW([{'params': model.parameters()}, {'params': metric_crit.parameters()}], lr=args.lr, weight_decay=args.weight_decay, amsgrad=False)
    
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.cosine_epo)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=args.warmup_epo, after_scheduler=scheduler_cosine)
    
    optimizer.zero_grad()
    optimizer.step()
    
    val_pp = 0.
    model_file = f'../model/{args.backbone}_best_fold_{args.fold}.pth'
    for epoch in range(1, args.cosine_epo+args.warmup_epo):
        # print(optimizer.param_groups[0]['lr'])

        # scheduler_cosine.step(epoch-1)
        scheduler_warmup.step(epoch-1)
        print(time.ctime(), 'Epoch:', epoch)

        train_loss = train_epoch(metric_crit, epoch, model, train_loader, optimizer)
        if epoch>1:
            val_outputs = val_epoch(metric_crit_val, model, valid_loader)
            np.save('../submit/val_outputs_best.npy', val_outputs)
            results = val_end(val_outputs)
            print(results)

            val_loss = results['val_loss']
            val_gap = results['val_gap']
            # val_gap_landmarks = results['val_gap_landmarks']
            # val_gap_pp = results['val_gap_pp']
            # val_gap_landmarks_pp = results['val_gap_landmarks_pp']
            # np.save('../submit/val_outputs.npy', val_outputs)
            # content = time.ctime() + ' ' + f'Fold {args.fold}, Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, train loss: {train_loss:.5f}, valid loss: {val_loss:.5f}, val_gap: {val_gap:.4f}, val_gap_landmarks: {val_gap_landmarks:.4f}, val_gap_pp: {val_gap_pp:.4f}, val_gap_landmarks_pp: {val_gap_landmarks_pp:.4f}'
            content = time.ctime() + ' ' + f'Fold {args.fold}, Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, train loss: {train_loss:.5f}, valid loss: {val_loss:.5f}, val_gap: {val_gap:.4f}'
            print(content)
            with open(f'../model/log_fold_{args.backbone}_{args.fold}.txt', 'a') as appender:
                appender.write(content + '\n')
            
            val_gap_pp = val_gap
            if val_gap_pp > val_pp:
                print('val_gap_pp_max ({:.6f} --> {:.6f}). Saving model ...'.format(val_pp, val_gap_pp))
                torch.save(model.state_dict(), model_file)
                val_pp = val_gap_pp

        
    
if __name__ == '__main__':
    main()
