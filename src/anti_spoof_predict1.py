# -*- coding: utf-8 -*-
# @Time : 20-6-9 上午10:20
# @Author : zhuying
# @Company : Minivision
# @File : anti_spoof_predict.py
# @Software : PyCharm

import os
import cv2
import math
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
import torchvision

from src.model_lib.MiniFASNet import MiniFASNetV1, MiniFASNetV2,MiniFASNetV1SE,MiniFASNetV2SE
from src.data_io import transform as trans
from src.utility import get_kernel, parse_model_name

MODEL_MAPPING = {
    'MiniFASNetV1': MiniFASNetV1,
    'MiniFASNetV2': MiniFASNetV2,
    'MiniFASNetV1SE':MiniFASNetV1SE,
    'MiniFASNetV2SE':MiniFASNetV2SE
}


def load_model(self, model_path):
    # define model

    # load model weight
    state_dict = torch.load(model_path, map_location=self.device)
    keys = iter(state_dict)
    first_layer_name = keys.__next__()
    if first_layer_name.find('module.') >= 0:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for key, value in state_dict.items():
            if str(key)[0:12] == "module.model":
                name_key = key[13:]
                new_state_dict[name_key] = value
        self.model.load_state_dict(new_state_dict)
    else:
        self.model.load_state_dict(state_dict)
    return None

def pretrain(model, state_dict):
    own_state = model.state_dict()
    keys = iter(state_dict)
    first_layer_name = keys.__next__()
    if first_layer_name.find('module.') >= 0:
        for name, param in state_dict.items():
            if str(name)[0:12] == "module.model":
                realname = name[13:]
                if isinstance(param, torch.nn.Parameter):
                    param = param.data
                try:
                    own_state[realname].copy_(param)
                except:
                    print('While copying the parameter named {}, '
                          'whose dimensions in the model are {} and '
                          'whose dimensions in the checkpoint are {}.'
                          .format(realname, own_state[name].size(), param.size()))
                    print("But don't worry about it. Continue pretraining.")

class Detection:
    def __init__(self):
        """
        # caffemodel = "/home/ubuntu/Silent-Face-Anti-Spoofing-master/saved_logs/models/2.7_80x80_MiniFASNetV2/2.7_80x80_MiniFASNetV2.caffemodel"
        # deploy = "/home/ubuntu/Silent-Face-Anti-Spoofing-master/saved_logs/models/2.7_80x80_MiniFASNetV2/deploy.prototxt"
        # self.detector = cv2.dnn.readNetFromCaffe(deploy, caffemodel)
        # self.detector_confidence = 0.6
        """
    def get_bbox(self, img):
        height, width = img.shape[0], img.shape[1]
        aspect_ratio = width / height
        if img.shape[1] * img.shape[0] >= 192 * 192:
            img = cv2.resize(img,
                             (int(192 * math.sqrt(aspect_ratio)),
                              int(192 / math.sqrt(aspect_ratio))), interpolation=cv2.INTER_LINEAR)

        blob = cv2.dnn.blobFromImage(img, 1, mean=(104, 117, 123))
        self.detector.setInput(blob, 'data')
        out = self.detector.forward('detection_out').squeeze()
        max_conf_index = np.argmax(out[:, 2])
        left, top, right, bottom = out[max_conf_index, 3]*width, out[max_conf_index, 4]*height, \
                                   out[max_conf_index, 5]*width, out[max_conf_index, 6]*height
        bbox = [int(left), int(top), int(right-left+1), int(bottom-top+1)]
        return bbox


class AntiSpoofPredict(Detection):
    def __init__(self):
        super(AntiSpoofPredict, self).__init__()
        self.num_class = 2
        self.model_path = "/home/ubuntu/Silent-Face-Anti-Spoofing-master/saved_logs/snapshot/Anti_Spoofing_1_80-80/2021-01-11-13-19_Anti_Spoofing_1_80-80_model_iter-39500.pth"
        model_name = os.path.basename(self.model_path)
        h_input, w_input, model_type, _ = parse_model_name(model_name)
        self.kernel_size = get_kernel(h_input, w_input, )
        self.model = MODEL_MAPPING["MiniFASNetV2SE"](conv6_kernel=self.kernel_size,num_classes=self.num_class)
        checkpoint = torch.load(self.model_path,map_location="cuda:3")
        pretrain(self.model, checkpoint)

        self.new_width = self.new_height = 224
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((self.new_width, self.new_height)),
            torchvision.transforms.ToTensor(),
        ])
        device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()
        # self.model_name = os.path.basename(self.model_path)
        # h_input, w_input, model_type, _ = parse_model_name(self.model_name)
        # self.kernel_size = get_kernel(h_input, w_input, )
        # self.net = MODEL_MAPPING["MiniFASNetV2SE"](conv6_kernel=self.kernel_size).to(self.device)


    def preprocess_data(self, image,device):
        processed_data = Image.fromarray(image)
        processed_data = self.transform(processed_data)
        # test_transform = trans.Compose([
        #     trans.ToTensor(),
        # ])
        # img = test_transform(image)
        # processed_data = img.unsqueeze(0).to(device)
        processed_data = processed_data.to(device)
        return processed_data

    def eval_image(self, image):
        data = torch.stack(image, dim=0)
        channel = 3
        device= torch.device("cuda:3")
        # self._load_model(self.model_path)
        input_var = data.view(-1, channel, data.size(2), data.size(3)).to(device=device)
        with torch.no_grad():
            rst = self.model(input_var).detach()
            # result = self.model.forward(image)
        return rst.reshape(-1, self.num_class)

    def predict(self, img):
        real_data = []
        device = torch.device("cuda:3")
        for image in img:
            data = self.preprocess_data(image,device)
            real_data.append(data)
            # result = self.eval_image(data)

            # result = self.model.forward(img)
        rst = self.eval_image(real_data)
        rst = torch.nn.functional.softmax(rst,dim=1).cpu.numpy()
            # real_data.append(result)
        probability = np.array(rst)
        # img = img.unsqueeze(0).to(self.device)
        # self._load_model(model_path)
        # self.model.eval()
        # with torch.no_grad():
        #     result = self.model.forward(img)
        #     result = F.softmax(result).cpu().numpy()
        return probability











