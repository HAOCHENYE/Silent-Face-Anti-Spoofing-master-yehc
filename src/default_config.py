# -*- coding: utf-8 -*-
# @Time : 20-6-4 上午9:12
# @Author : zhuying
# @Company : Minivision
# @File : default_config.py
# @Software : PyCharm
# --*-- coding: utf-8 --*--
"""
default config for training
"""
import time
import torch
from datetime import datetime
from easydict import EasyDict
from src.utility import make_if_not_exist, get_width_height, get_kernel
import os

def get_default_config():
    conf = EasyDict()

    # ----------------------training---------------
    conf.lr = 0.1
    # [9, 13, 15]
    conf.milestones = [60, 90, 110]  # down learing rate
    conf.gamma = 0.1
    conf.epochs = 120
    conf.momentum = 0.9
    conf.batch_size = 96
    conf.ft_height = 10
    conf.ft_width = 10
    conf.input_size = [80, 80]
    # model
    conf.num_classes = 3
    conf.input_channel = 3
    conf.embedding_size = 128

    # dataset
    # conf.train_root_path = '/media/traindata_ro/users/yl3334/wangch/celeba_type/train/images'
    conf.train_root_path = '/media/traindata_ro/users/yl3334/wangch/celeba_type/train/images'
    conf.eval_root_path = '/media/traindata_ro/users/yl3334/wangch/celeba_type/val/images'

    conf.prefix = "/media/traindata_ro/users/yl3334/wangch/celeba_spoof"

    conf.train_annotation_path = os.path.join(conf.prefix, 'metas/intra_test/train_label_bbox.json')
    conf.val_annotation_path = os.path.join(conf.prefix, 'metas/intra_test/test_label_bbox.json')
    conf.model_path='work_dirs/RepVGGNet'
    # save file path
    # conf.eval_snapshot_dir_path = '/home/ubuntu/Silent-Face-Anti-Spoofing-master/train_Adam_weight_0.1-Add/eval_snapshot'
    # log path
    conf.time = time.strftime('%Y%m%d', time.localtime(time.time()))
    conf.log_path = 'work_dirs/RepVGGNet/logs'
    # tensorboard
    conf.board_loss_every = 10
    # save model/iter
    conf.save_every = 30

    return conf


def update_config(args, conf):
    conf.devices = args.devices
    conf.patch_info = args.patch_info
    w_input, h_input = get_width_height(args.patch_info)

    conf.kernel_size = get_kernel(h_input, w_input)
    conf.device = "cuda:{}".format(conf.devices[0]) if torch.cuda.is_available() else "cpu"

    # resize fourier image size
    # conf.ft_height = 2*conf.kernel_size[0]
    # conf.ft_width = 2*conf.kernel_size[1]



    # eval_snapshot_dir = '{}/{}'.format(conf.eval_snapshot_dir_path, eval_job_name)
    make_if_not_exist(conf.model_path)
    make_if_not_exist(conf.log_path)


    return conf
