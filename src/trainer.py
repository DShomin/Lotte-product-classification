from conf import *

from sklearn.utils.class_weight import compute_class_weight

import os
import sys
import time
import numpy as np
import pandas as pd
import cv2
import PIL.Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer

import torch
from torch.utils.data import TensorDataset, DataLoader,Dataset
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import albumentations as A
import geffnet

import timm
from loss import *


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def global_average_precision_score(y_true, y_pred, ignore_non_landmarks=False):
    indexes = np.argsort(y_pred[1])[::-1]
    queries_with_target = (y_true < args.n_classes).sum()
    correct_predictions = 0
    total_score = 0.
    i = 1
    for k in indexes:
        if ignore_non_landmarks and y_true[k] == args.n_classes:
            continue
        if y_pred[0][k] == args.n_classes:
            continue
        relevance_of_prediction_i = 0
        if y_true[k] == y_pred[0][k]:
            correct_predictions += 1
            relevance_of_prediction_i = 1
        precision_at_rank_i = correct_predictions / i
        total_score += precision_at_rank_i * relevance_of_prediction_i
        i += 1
    return 1 / queries_with_target * total_score

def comp_metric(y_true, logits, ignore_non_landmarks=False):
    
    score = global_average_precision_score(y_true, logits, ignore_non_landmarks=ignore_non_landmarks)
    return score

def optimizer_zero_grad(epoch, batch_idx, optimizer, optimizer_idx):
    # optimizer.zero_grad()
    for param in self.model.parameters():
        param.grad = None

def train_epoch(metric_crit, epoch, model, loader, optimizer):
    criterion = nn.CrossEntropyLoss()
    model.train()
    train_loss = []
    arcface = []
    bar = tqdm(loader)
    for batch in bar:
        batch['input'] = batch['input'].to(args.device)
        batch['target'] = batch['target'].to(args.device)
        
        optimizer.zero_grad()

        logits = model(batch)
        loss = loss_fn(metric_crit, batch, logits)
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ')

            print(loss, batch, logits, batch['input'].shape, batch['target'].shape)
            exit(1)

        if args.arcface_s is None:
            s = metric_crit.s.detach().cpu().numpy()
        elif args.arcface_s == -1:
            s = 0
        else:
            s = metric_crit.s
        
        if args.distributed_backend == "ddp":
            step = epoch*args.batch_size*len(args.gpus.split(','))*args.gradient_accumulation_steps
        else:
            step = epoch*args.batch_size*args.gradient_accumulation_steps

        loss.backward()
        # clipping point
        # if clipping:
        #     timm.utils.adaptive_clip_grad(model.parameters())
        optimizer.step()
        loss_np = loss.detach().cpu().numpy()
        train_loss.append(loss_np)
        arcface.append(s)
        
        bar.set_description('loss: %.5f, arcface_s: %.5f' % (loss_np, s))
    
    train_loss = np.mean(train_loss)
    arcface = np.mean(arcface)

    return train_loss

def get_trans(img, I):
    if I >= 4:
        img = img.transpose(2,3)
    if I % 4 == 0:
        return img
    elif I % 4 == 1:
        return img.flip(2)
    elif I % 4 == 2:
        return img.flip(3)
    elif I % 4 == 3:
        return img.flip(2).flip(3)


def val_epoch(metric_crit_val, model, loader, n_test=1, get_output=False):
    model.eval()
    val_outputs = []
    with torch.no_grad():
        for batch in tqdm(loader):
            batch['input'] = batch['input'].to(args.device)
            batch['target'] = batch['target'].to(args.device)

            output_dict = model(batch, get_embeddings=True)
            loss = loss_fn(metric_crit_val, batch, output_dict, val=True)

            # temp_batch = batch.copy()
            # for I in range(n_test):
            #     if I == 0:
            #         output_dict = model(temp_batch, get_embeddings=True)
            #         logits = output_dict['logits']
            #         embeddings = output_dict['embeddings']
            #     else:
            #         temp_batch['input'] = get_trans(temp_batch['input'], I)
            #         output_dict2 = model(temp_batch, get_embeddings=False)
            #         logits += output_dict2['logits']
            # else:
            #     logits /= n_test
            #     output_dict['logits'] = logits[0]

            # (values, indices) = torch.topk(logits, 3, dim=1)
            # preds = indices[:, 0]
            # preds_conf = values[:, 0]

            logits = output_dict['logits']
            embeddings = output_dict['embeddings']

            preds_conf, preds = torch.max(logits.softmax(1),1)

            # allowed_classes = torch.Tensor(list(range(args.n_classes))).long().to(logits.device)

            # preds_conf_pp, preds_pp = torch.max(logits.gather(1,allowed_classes.repeat(logits.size(0),1)).softmax(1),1)
            # preds_pp = allowed_classes[preds_pp]

            targets = batch['target']

            output = dict({
                'idx':batch['idx'],
                'embeddings': embeddings,
                'val_loss': loss.view(1),
                # 'val_loss': torch.tensor([0], device='cuda:0'),
                'preds': preds,
                'preds_conf':preds_conf,
                # 'preds_pp': preds_pp,
                # 'preds_conf_pp':preds_conf_pp,
                'targets': targets,
                
            })
            val_outputs += [output] 

    return val_outputs

def val_end(val_outputs):
    out_val = {}
    for key in val_outputs[0].keys():
        out_val[key] = torch.cat([o[key] for o in val_outputs])

    device = out_val["targets"].device

    for key in out_val.keys():
            out_val[key] = out_val[key].detach().cpu().numpy().astype(np.float32)

    val_score = comp_metric(out_val["targets"], [out_val["preds"], out_val["preds_conf"]])
    val_score_landmarks = comp_metric(out_val["targets"], [out_val["preds"], out_val["preds_conf"]])

    # val_score_pp = comp_metric(out_val["targets"], [out_val["preds_pp"], out_val["preds_conf_pp"]])
    # val_score_landmarks_pp = comp_metric(out_val["targets"], [out_val["preds_pp"], out_val["preds_conf_pp"]])

    val_loss_mean = np.sum(out_val["val_loss"])
    
    results = {'val_loss': val_loss_mean,
                     'val_gap':val_score,
                     'val_gap_landmarks':val_score_landmarks,
                    #  'val_gap_pp':val_score_pp,
                    #  'val_gap_landmarks_pp':val_score_landmarks_pp,
                    }

    return results