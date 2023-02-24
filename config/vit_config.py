'''
Date: 2023-02-13 06:17:47
Author: yang_haitao
LastEditors: yanghaitao yang_haitao@leapmotor.com
LastEditTime: 2023-02-21 01:40:58
FilePath: /K-Lane/home/work_dir/work/keylane/config/vit_config.py
'''



seed = 2023

load_from = None
finetune_from = None

log_dir = None
view = False
workers=12

eval_ep = 1
save_ep = 1

epochs = 50
batch_size = 2
decay_rate = 1e-4
# lr = 0.0002
lr = 0.0004
# total_iter = (7687 // batch_size) * epochs
total_iter = (1597 // batch_size) * epochs


featuremap_out_channel = 64

filter_mode = 'xyz'
list_filter_roi = [0.02, 46.08, -11.52, 11.52, -2.0, 1.5]  # get rid of 0, 0 points
list_roi_xy = [0.0, 46.08, -11.52, 11.52]
list_grid_xy = [0.04, 0.02]
list_img_size_xy = [1152, 1152]

conf_thr = 0.5
view = True

# BGR Format to OpenCV
cls_lane_color = [
    (0, 0, 255),
    (0, 50, 255),
    (0, 255, 255),
    (0, 255, 0),
    (255, 0, 0),
    (255, 0, 100)
]


### Setting Here ###
dataset_path = '/home/data/klane' 
# dataset_type = 'KLane'
dataset_type = 'LeapLane'

dataset = dict(
    train=dict(
        type=dataset_type,
        data_root=dataset_path,
        split='train',
    ),
    test=dict(
        type=dataset_type,
        data_root=dataset_path,
        split='test',
    )
)
