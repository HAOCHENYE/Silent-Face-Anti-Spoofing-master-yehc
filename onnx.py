import torch
import torch.nn as nn
from src.utility import get_kernel
from src.model_lib.RepVGGNet import RepVggFTNet

from collections import OrderedDict

def load(module, prefix=''):
    for name, child in module._modules.items():
        if not hasattr(child, 'fuse_conv'):
            load(child, prefix + name + '.')
        else:
            child.fuse_conv()


class OnnxModel(nn.Module):
    def __init__(self, **kwargs):
        super(OnnxModel, self).__init__()
        kernel_size = get_kernel(80, 60)
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
        except:
            print("There is no model loaded!! Initial parameters will be saved")

    def forward(self, x):
        cls = self.model(x)
        # cls = F.softmax(cls,dim=1)
        return cls


model = OnnxModel()
# print(tw.model_stats(model, [1,3, 80, 60]))
model.eval()
# model.cuda("cuda:2")

# common.mkdirs_from_file_path(f"{jobdir}/last.onnx")

# dummy = torch.zeros((1, 3, 1152, 2048)).cuda()
# dummy = torch.zeros((1, 3, 320, 320)).cuda("cuda:2")
dummy = torch.zeros((1, 3, 80, 80))
# torch.onnx.export(model, dummy, f"{jobdir}/model.onnx", output_names=["hm", "pool_hm", "tlrb", "landmark"], opset_version=9, verbose=True,)
# torch.onnx.export(model, dummy, f"{jobdir}/model.onnx", output_names=["hm", "pool_hm", "tlrb", "landmark"], opset_version=11, verbose=True)
# torch.onnx.export(model, dummy, f"{jobdir}/model.onnx", opset_version=11, verbose=True, output_names=["hm", "pool_hm", "tlrb", "landmark"], )
# torch.onnx._export(model, dummy, f"{jobdir}/model.onnx", export_params=True, verbose=True, output_names=["pool_hm", "tlrb", "landmark", "hm"],opset_version=11)
torch.onnx.export(model, dummy, "work_dirs/RepVGGNet/SilentRepVGG.onnx", export_params=True, verbose=True, input_names=["data"], output_names=["Spoof"],opset_version=11)




