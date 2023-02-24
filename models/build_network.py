'''
'''

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel, DataParallel
from models.network.vit_lanetnet import VisionTransformer


def build_network(cfg=None):
    # init network
    model = VisionTransformer(img_size=1152, depth=1)
    
    
    torch.cuda.manual_seed_all(cfg.seed)

    if cfg.distributed:
        local_rank = dist.get_rank()
        torch.cuda.set_device(local_rank)
        cfg.device = torch.device("cuda", local_rank)
        model = DistributedDataParallel(module=model.to(device=cfg.device), 
                                        device_ids=[local_rank],
                                        broadcast_buffers=False,
                                        find_unused_parameters=True)
    else:
        # cfg.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cfg.device = torch.device('cuda')
        model = DataParallel(
            module=model,
            # device_ids=range(cfg.gpus)
            device_ids=[0,1]
        ).cuda()


    return model
    