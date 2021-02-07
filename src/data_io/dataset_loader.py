# -*- coding: utf-8 -*-
# @Time : 20-6-4 下午3:40
# @Author : zhuying
# @Company : Minivision
# @File : dataset_loader.py
# @Software : PyCharm

from torch.utils.data import DataLoader
from src.data_io.dataset_folder import DatasetFolderFT
from src.data_io import transform as trans
from torchvision.transforms import transforms
from src.data_io.dataset_folder import CelebASpoofDataset




def get_train_loader(conf):
    train_transform = trans.Compose([
        trans.ToPILImage(),
        trans.RandomResizedCrop(size=tuple(conf.input_size),
                                scale=(0.9, 1.1)),
        trans.ColorJitter(brightness=0.4,
                          contrast=0.4, saturation=0.4, hue=0.1),
        trans.RandomRotation(10),
        trans.RandomHorizontalFlip(),
        trans.ToTensor(),
        trans.Normalize(mean=127.5, std=128),
    ])
    root_path = '{}/{}'.format(conf.train_root_path, conf.patch_info)

    trainset = DatasetFolderFT(root_path, train_transform,
                               None, conf.ft_width, conf.ft_height)
    # trainset = CelebASpoofDataset(conf.train_annotation_path,
    #                               root_prefix=conf.prefix,
    #                               transform=train_transform,
    #                               ft_width=conf.ft_width,
    #                               ft_height=conf.ft_height)
    train_loader = DataLoader(
        trainset,
        batch_size=conf.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=16)
    return train_loader

def get_eval_loader(conf):
    eval_transform = trans.Compose([
        trans.ToPILImage(),
        trans.RandomResizedCrop(size=tuple(conf.input_size),
                                scale=(0.9, 1.1)),
        trans.RandomHorizontalFlip(),
        trans.ToTensor(),
        trans.Normalize(mean=127.5, std=128)
    ])
    root_path = '{}/{}'.format(conf.eval_root_path, conf.patch_info)
    evalset = DatasetFolderFT(root_path, eval_transform,
                               None, conf.ft_width, conf.ft_height)
    # evalset = CelebASpoofDataset(conf.val_annotation_path,
    #                               root_prefix=conf.prefix,
    #                               transform=eval_transform,
    #                               ft_width=conf.ft_width,
    #                               ft_height=conf.ft_height)

    # evalset = DatasetFolderFT(root_path, eval_transform,
    #                            None, conf.ft_width, conf.ft_height)
    eval_loader = DataLoader(
        evalset,
        batch_size=conf.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=16)
    return eval_loader
