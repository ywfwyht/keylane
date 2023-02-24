# /bin/bash
###
 # @Date: 2023-02-14 06:41:47
 # @Author: yang_haitao
 # @LastEditors: yanghaitao yang_haitao@leapmotor.com
 # @LastEditTime: 2023-02-15 06:01:56
 # @FilePath: /K-Lane/home/work_dir/work/keylane/distributed_train.sh
### 
######## 单机多GPU #######
# torch.distributed.launch以命令行方式将args.local_rank变量注入到每个进程
# --nproc_per_node创建的进程数，几个gpu就创建几个
# --nnodes表示使用几个节点（几台机器），由于是单机，即设为1
# python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 train_gpu_0.py
# sh distributed_train.sh 2
NUM_GPUS_PER_NODE=$1
NUM_NODES=${NUM_NODES:-1}
NODE_RANK=${NODE_RANK:-0}

torchrun \
    --nproc_per_node=$NUM_GPUS_PER_NODE \
    --nnodes=$NUM_NODES \
    --node_rank=$NODE_RANK \
    train.py