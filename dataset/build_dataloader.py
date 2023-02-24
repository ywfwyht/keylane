'''
Date: 2023-02-15 02:06:53
Author: yang_haitao
LastEditors: yanghaitao yang_haitao@leapmotor.com
LastEditTime: 2023-02-20 12:37:00
FilePath: /K-Lane/home/work_dir/work/keylane/dataset/build_dataloader.py
'''
import torch
import numpy as np
from torch.utils.data import DataLoader, DistributedSampler
from dataset.klane import KLane


def build_dataset(cfg=None, split='train'):
    dataset = KLane(cfg.dataset_path, split, cfg=cfg)

    return dataset


def build_dataloader(cfg=None, split='train'):
    dataset = build_dataset(cfg=cfg, split=split)
    # for DDP
    if cfg.distributed:
        data_sampler = DistributedSampler(dataset=dataset)
        data_loader = DataLoader(dataset=dataset,
                             batch_size=cfg.batch_size,
                             sampler=data_sampler,
                             num_workers=cfg.workers,
                             pin_memory=True,
                             drop_last=True,
                             worker_init_fn=lambda x: np.random.seed(x + cfg.seed))

    else:
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.workers,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=lambda x: np.random.seed(x + cfg.seed)
        )

    return data_loader