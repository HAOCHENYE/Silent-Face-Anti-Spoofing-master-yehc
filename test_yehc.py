# import common
# import eval_tool
# from train.small import eval_tool as eval_tool
import torch
import torch.nn as nn
import cv2
import torch.nn.functional as F
import numpy as np
from torchvision.transforms import transforms
from torchvision.transforms.transforms import ToTensor
from collections import  OrderedDict
# from dbface import DBFace
import tensorwatch as tw
# from model.DBFaceSmallH import DBFace
from src.model_lib.MultiFTNet import MultiFTNet
from src.model_lib.MiniFASNet import MiniFASNetV1, MiniFASNetV2,MiniFASNetV1SE,MiniFASNetV2SE
# import torch.onnx.symbolic_opset11
# import torch.onnx.symbolic_helper
from src.utility import get_kernel
from src.model_lib.RepVGGNet import RepVggFTNet

def load(module, prefix=''):
    for name, child in module._modules.items():
        if not hasattr(child, 'fuse_conv'):
            load(child, prefix + name + '.')
        else:
            child.fuse_conv()

class OnnxModel(nn.Module):
    def __init__(self, **kwargs):
        super(OnnxModel, self).__init__()
        # self.model = MiniFASNetV2(conv6_kernel = kernel_size)
        self.model = RepVggFTNet(32, [48, 64, 128], [4, 10, 8], num_out=2, num_classes=3)
        try:
            # self.model.load("work_dirs/RepVGGNet/epoch_77.pth")
            clean_state_dict = OrderedDict()
            state_dict = torch.load("work_dirs/RepVGGNet/epoch_77.pth")
            for name, value in state_dict.items():
                if name.startswith("module."):
                    clean_state_dict[name.lstrip("module.")] = value
            self.model.load_state_dict(clean_state_dict)
            load(self.model)
            # load_state_dict(self.model, clean_state_dict)
        except:
            print("There is no model loaded!! Initial parameters will be saved")

    def forward(self, x):
        cls = self.model(x)
        cls = F.softmax(cls)
        return cls

model = OnnxModel()
model = model.to("cuda:2")
model.eval()


image = cv2.imread("images/sample/image_F3.jpg")
image = cv2.resize(image, (80, 80))
image = np.transpose(image, (2, 0, 1))
image = torch.tensor(image)
# image = image.unsqueeze(0)

image = image.unsqueeze(0)
image = image.to("cuda:2")
image = image.float()
with torch.no_grad():
    result = model(image)

print(result)