# -*- coding: utf-8 -*-
# @Time : 20-6-4 下午4:04
# @Author : zhuying
# @Company : Minivision
# @File : dataset_folder.py
# @Software : PyCharm
from tqdm import tqdm
import cv2
import torch
from torchvision import datasets
import numpy as np
import json
import os
def opencv_loader(path):
    img = cv2.imread(path)
    return img


class CelebASpoofDataset(object):
    SPOOF_CLASSES={"Spoof Type":
                          {
                              0: "Live", 1: "Photo", 2: "Poster", 3: "A4", 4: "Face Mask", 5: "Upper Body Mask",
                              6: "Region Mask", 7: "PC", 8: "Pad", 9: "Phone", 10: "3D Mask"},
             "Illumination Condition":
                          {
                              0: "Live", 1: "Normal", 2: "Poster", 3: "Strong", 4: "Back", 5: "Dark"
                          },
             "Environment":
                          {
                              0: "Live", 1: "Indoor", 2: "OutDoor"
                          }
            }
    CLASSES = {"Spoof Type": 40, "Illumination Condition":41,  "Environment":42, "Live": 43}
    def __init__(self, annotations,
                       root_prefix="",
                       classes="Spoof Type",
                       transform=None,
                       target_transform=None,
                       ft_width=10,
                       ft_height=10):
        self.transform = transform
        self.prefix = root_prefix
        self.CatId = self.CLASSES[classes]
        self.ft_width = ft_width
        self.ft_height = ft_height
        self.samples = []

        self.read_json(annotations)

    def read_json(self, path):
        anns = json.load(open(path, 'r'))
        for image_path, label in tqdm(anns.items(), desc="Loading training jsonfile"):
            image_path = os.path.join(self.prefix, image_path)
            self.samples.append({image_path: label})
        print("Dataset : {} images".format(len(anns)))

    def __getitem__(self, idx):
        sample = self.samples[idx]
        for image_path, meta in sample.items():
            if sample is None:
                print('image is None --> ', image_path)

            image = cv2.imread(image_path)
            # image = opencv_loader("/media/traindata_ro/users/yl3334/wangch/celeba_spoof/Data/train/3994/live/155732.jpg")
            # image = np.ones((3, 100, 100))
            # image = np.ones((80, 80, 3)).astype(np.uint8)

            target = meta[self.CatId]
            box = meta[-4:]

            width = image.shape[1]
            height = image.shape[0]

            x1, y1, w, h = box
            x1, y1, w, h = int(x1/224*width), int(y1/224*height), int( w/224*width), int(h/224*height)

            # cv2.rectangle(image, (x1, y1), (x1+w, y1+h), color=(0, 0, 255))
            # cv2.putText(image, self.SPOOF_CLASSES["Spoof Type"][target], (20, 20), 0, 1, color=(0, 0, 255))
            # cv2.imwrite(os.path.join("test_image", image_path[-9:]), image)

            left_noise = int(np.random.random() * w)
            right_noise = int(np.random.random() * w)
            top_noise = int(np.random.random() * h)
            bottom_noise = int(np.random.random() * h)

            x1 = max(0, x1 - left_noise)
            y1 = max(0, y1 - top_noise)
            x2 = min(width, x1+w + right_noise)
            y2 = min(height, y1+h + bottom_noise)

            image = image[y1:y2, x1:x2, :]


        ft_image = generate_FT(image)
        ft_image = cv2.resize(ft_image, (self.ft_width, self.ft_height))
        ft_image = torch.from_numpy(ft_image).float()
        ft_image = torch.unsqueeze(ft_image, 0)

        image = self.transform(image)

        # image = torch.ones((3, 80, 80))
        # ft_image = torch.ones((1, 10, 10))
        # target = 1
        return image, ft_image, target

    def __len__(self):
        return len(self.samples)

class DatasetFolderFT(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None,
                 ft_width=10, ft_height=10, loader=opencv_loader):
        super(DatasetFolderFT, self).__init__(root, transform, target_transform, loader)
        self.root = root
        self.ft_width = ft_width
        self.ft_height = ft_height

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        # generate the FT picture of the sample
        ft_sample = generate_FT(sample)
        if sample is None:
            print('image is None --> ', path)
        if ft_sample is None:
            print('FT image is None -->', path)
        assert sample is not None

        ft_sample = cv2.resize(ft_sample, (self.ft_width, self.ft_height))
        ft_sample = torch.from_numpy(ft_sample).float()
        ft_sample = torch.unsqueeze(ft_sample, 0)

        if self.transform is not None:
            try:
                sample = self.transform(sample)
            except Exception as err:
                print('Error Occured: %s' % err, path)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, ft_sample, target


def generate_FT(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    fimg = np.log(np.abs(fshift)+1)
    maxx = -1
    minn = 100000
    for i in range(len(fimg)):
        if maxx < max(fimg[i]):
            maxx = max(fimg[i])
        if minn > min(fimg[i]):
            minn = min(fimg[i])
    fimg = (fimg - minn+1) / (maxx - minn+1)
    return fimg