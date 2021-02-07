from lr_finder import LRFinder
from src.model_lib.MultiFTNet import MultiFTNet
from src.model_lib.MiniFASNet import MiniFASNetV1, MiniFASNetV2,MiniFASNetV1SE,MiniFASNetV2SE
from src.utility import get_kernel
from torch.nn import CrossEntropyLoss, MSELoss
from torch import optim
from src.data_io.dataset_loader import get_train_loader,get_eval_loader
from src.default_config import get_default_config, update_config
from train import parse_args
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
kernel_size = get_kernel(80, 60)
model = MultiFTNet(conv6_kernel = kernel_size)
cls_criterion = CrossEntropyLoss()
FT_criterion = MSELoss()
from torch import optim
# optimizer = optim.SGD(model.parameters(),
#                                    lr=0.1,
#                                    weight_decay=5e-4,
#                                    momentum=0.9)
optimizer = optim.AdamW(model.parameters())
lr_finder = LRFinder(model, optimizer, cls_criterion,FT_criterion)
conf = get_default_config()
args = parse_args()
conf = update_config(args, conf)
trainloader = get_train_loader(conf)
val_loader = get_eval_loader(conf)
lr_finder.range_test(trainloader, end_lr=1, num_iter=100, step_mode="linear")
lr_finder.plot(log_lr=False)
lr_finder.reset()
