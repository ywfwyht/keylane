'''
Date: 2023-02-13 05:29:52
Author: yang_haitao
LastEditors: yanghaitao yang_haitao@leapmotor.com
LastEditTime: 2023-02-22 01:28:18
FilePath: /K-Lane/home/work_dir/work/keylane/train.py
'''
import argparse
import os, sys
import torch
import torch.backends.cudnn as cudnn
import time
import shutil
import logging
import datetime
import random
import numpy as np
from tqdm import tqdm 
from pathlib import Path

from models.criterion import loss_criterion
from dataset.build_dataloader import build_dataset, build_dataloader
from models.build_network import build_network
from utils.metric_utils import calc_measures
from utils.config import Config
import torch.distributed as dist


def main(cfg=None):
    def log_string(str):
        logger.info(str)
        print(str)

    # log_dir
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    log_dir = Path('./logs')
    log_dir.mkdir(exist_ok=True)
    if cfg.log_dir is None:
        log_dir = log_dir.joinpath(timestr)
    else:
        log_dir = log_dir.joinpath(cfg.log_dir)
    log_dir.mkdir(exist_ok=True)
    checkpoints_dir = log_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)

    # Logger
    logger = logging.getLogger('Model')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_hander = logging.FileHandler('%s/log.txt' % (log_dir))
    file_hander.setLevel(logging.INFO)
    file_hander.setFormatter(formatter)
    logger.addHandler(file_hander)

    cudnn.benchmark = True
    # torch.manual_seed(cfg.seed)
    # torch.cuda.manual_seed_all(cfg.seed)
    # random.seed(cfg.seed)
    # np.random.seed(cfg.seed)

    # init_process_group
    if cfg.distributed:
        dist.init_process_group(backend='nccl', init_method='env://')

    # dataloader
    train_dataset = build_dataset(cfg=cfg, split='train')
    train_loader = build_dataloader(cfg=cfg, split='train')
    test_dataset = build_dataset(cfg=cfg, split='test')
    test_loader = build_dataloader(cfg=cfg, split='test')
    log_string("The number of training data is: %d" % len(train_dataset))
    log_string("The number of test data is: %d" % len(test_dataset))

    # init network
    model = build_network(cfg=cfg)
    
    try:
        checkpoint = torch.load(str(checkpoints_dir) + 'best_model.pth')
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model.')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0

    # optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=cfg.lr, 
        betas=(0.9, 0.999), 
        eps=1e-8, 
        weight_decay=cfg.decay_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=10000)
    
    # train network
    best_acc = 0
    for epoch in range(start_epoch, cfg.epochs):
        train_loss, train_acc, train_f1 = train(epoch, model, optimizer, train_loader, cfg)
        
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        log_string(f'Train epoch:{epoch}, lr:{lr}, loss:{train_loss}, acc:{train_acc}, f1:{train_f1}')
        if epoch % 5 == 0:
            logger.info('Save Model...')
            savepath = str(checkpoints_dir) + '/model.pth'
            log_string('Saving at {}'.format(savepath))
            state = {
                'epoch':epoch,
                'model_state_dict':model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string('Saving model...')
        # validate
        val_loss, val_acc, val_f1 = evaluate(epoch, model, optimizer, test_loader, cfg)
        log_string(f'Validate epoch:{epoch}, lr:{lr}, loss:{val_loss}, acc:{val_acc}, f1:{val_f1}')
        if val_acc > best_acc:
            best_acc = val_acc
            state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            savepath = str(checkpoints_dir) + '/best_model.pth'
            log_string('Saving at {}'.format(savepath))
            torch.save(state, savepath)
            log_string('Saving model..., best acc {}.' % best_acc)
    
    dist.barrier()

def train(epoch, model, optimizer, data_loader, cfg=None):
    model.train()

    data_loader = tqdm(data_loader, total=len(data_loader), smoothing=0.9)
    for step, data in enumerate(data_loader):
        img = data['proj']
        label = data['label']
        img = img.to(cfg.device)
        label = label.to(cfg.device)

        # forward
        pred = model(img)
        loss_dict = loss_criterion(pred, label)
        loss = loss_dict['loss']
        conf_pred_out = pred[:, 7, :, :]
        cls_pred_out = pred[:, :7, :, :]
        mean_acc = []
        mean_f1 = []
        for batch_idx in range(cfg.batch_size):
            cls_label = label[batch_idx].cpu().detach().numpy()
            conf_label = np.where(cls_label == 255, 0, 1)

            cls_pred = torch.nn.functional.softmax(cls_pred_out[batch_idx], dim=0)
            pred_idx = torch.argmax(cls_pred, dim=0)
            pred_classes = torch.max(cls_pred, dim=0)[1]
            conf_pred_raw = conf_pred_out[batch_idx].cpu().detach().numpy()
            is_flip = False
            if is_flip:
                conf_pred_raw = np.flip(np.flip(conf_pred_raw, 0), 1)
            conf_pred = np.where(conf_pred_raw > cfg.conf_thr, 1, 0)
            acc, precision, recall, f1 = calc_measures(conf_label, conf_pred, 'conf')
            mean_acc.append(acc)
            mean_f1.append(f1)
        mean_acc = np.mean(mean_acc)
        mean_f1 = np.mean(mean_f1)       

        # backward
        optimizer.zero_grad()
        loss.requires_grad_(True)
        loss.backward()
        data_loader.set_description(f'Epoch[{epoch}/{cfg.epochs}]')
        lr = optimizer.param_groups[0]['lr']
        print(f' loss: {loss}\n lr: {lr}\n mean_acc: {mean_acc}\n mean_f1: {mean_f1}\n precision: {precision}\n recall: {recall}')
   
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)
        
        optimizer.step()
        # optimizer.zero_grad()

    return loss, mean_acc, mean_f1 # per-batch

    

@torch.no_grad()
def evaluate(epoch, model, optimizer, data_loader, cfg=None):
    model.eval()

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        img = data['proj']
        label = data['label']
        img = img.to(cfg.device)
        label = label.to(cfg.device)

        # forward
        pred = model(img)
        loss_dict = loss_criterion(pred, label)
        loss = loss_dict['loss']

        conf_pred_out = pred[:, 7, :, :]
        cls_pred_out = pred[:, :7, :, :]
        mean_acc = []
        mean_f1 = []
        for batch_idx in range(cfg.batch_size):
            cls_label = label[batch_idx].cpu().detach().numpy()
            conf_label = np.where(cls_label == 255, 0, 1)

            cls_pred = torch.nn.functional.softmax(cls_pred_out[batch_idx], dim=0)
            pred_idx = torch.argmax(cls_pred, dim=0)
            pred_classes = torch.max(cls_pred, dim=0)[1]
            conf_pred_raw = conf_pred_out[batch_idx].cpu().detach().numpy()
            is_flip = False
            if is_flip:
                conf_pred_raw = np.flip(np.flip(conf_pred_raw, 0), 1)
            conf_pred = np.where(conf_pred_raw > cfg.conf_thr, 1, 0)
            acc, _, _, f1 = calc_measures(conf_label, conf_pred, 'conf')
            mean_acc.append(acc)
            mean_f1.append(f1)
        mean_acc = np.mean(mean_acc)
        mean_f1 = np.mean(mean_f1)

        # backward
        optimizer.zero_grad()
        loss.requires_grad_(True)
        loss.backward()
        data_loader.set_description(f'Epoch[{epoch}/{cfg.epochs}]')
        lr = optimizer.param_groups[0]['lr']
        print(f' loss: {loss}\n lr: {lr}\n mean_acc: {mean_acc}\n mean_f1: {mean_f1}')

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)
        
        optimizer.step()
        # optimizer.zero_grad()

    return loss, mean_acc, mean_f1



if __name__=="__main__":

    parse = argparse.ArgumentParser(description="args for train")
    parse.add_argument('--gpus', default='0,1', type=str)
    parse.add_argument('--local_rank', default=1, type=int)
    parse.add_argument('--distributed', default=False, type=bool)
    args = parse.parse_args()

    config_file = 'config/vit_config.py'
    cfg = Config.fromfile(config_file)
    cfg.gpus = len(args.gpus.split(','))
    cfg.local_rank = args.local_rank
    cfg.distributed = args.distributed
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    print(cfg)
    main(cfg)