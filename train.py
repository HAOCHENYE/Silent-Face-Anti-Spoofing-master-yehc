# -*- coding: utf-8 -*-
# @Time : 20-6-3 下午5:39
# @Author : zhuying
# @Company : Minivision
# @File : train.py
# @Software : PyCharm
from torchvision.transforms import RandomCrop
import argparse
import os
from src.train_main import TrainMain
# from src.train_eval_main import TrainMain
from src.default_config import get_default_config, update_config


def parse_args():
    """parsing and configuration"""
    desc = "Silence-FAS"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--device_ids", type=str, default="2", help="which gpu id, 0123")
    parser.add_argument("--patch_info", type=str, default="org_1_80-60",
                        help="[org_1_80-60 / 1_80-80 / 2.7_80-80 / 4_80-80]")
    args = parser.parse_args()
    cuda_devices = [int(elem) for elem in args.device_ids]
    # cuda_devices = [args.device_ids]
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, cuda_devices))
    args.devices = [x for x in cuda_devices]
    # args.devices= [args.device_ids]
    return args


if __name__ == "__main__":
    args = parse_args()
    conf = get_default_config()
    conf = update_config(args, conf)
    trainer = TrainMain(conf)
    trainer.train_model()

