from util.RepVGG import RepVGGBlock
import torch.nn as nn
from collections import OrderedDict
from mmcv.cnn import ConvModule
from mmcv.cnn import constant_init, kaiming_init
import os


class FTGenerator(nn.Module):
    def __init__(self, in_channels=48, out_channels=1):
        super(FTGenerator, self).__init__()

        self.ft = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, out_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.ft(x)


class RepVGGStage(nn.Module):
    def __init__(self, in_ch, stage_ch, num_block, kernel_size=3, group=1):
        super(RepVGGStage, self).__init__()
        LayerDict = OrderedDict()

        for num in range(num_block):
            if num == 0:
                LayerDict["Block{}".format(num)] = RepVGGBlock(in_ch, stage_ch, group=group, kernel_size=kernel_size, stride=2)
                LayerDict["Block{}_1x1".format(num)] = ConvModule(stage_ch, stage_ch, groups=group, kernel_size=1, stride=1)
                continue
            LayerDict["Block{}".format(num)] = RepVGGBlock(stage_ch, stage_ch, group=group, kernel_size=kernel_size, stride=1)
            LayerDict["Block{}_1x1".format(num)] = ConvModule(stage_ch, stage_ch, groups=group, kernel_size=1, stride=1)
        self.Block = nn.Sequential(LayerDict)

    def forward(self, x):
        return self.Block(x)



class RepVGGNet(nn.Module):
    def __init__(self,
                 stem_channels,
                 stage_channels,
                 block_per_stage,
                 kernel_size=3,
                 num_out=5,
                 norm_cfg=dict(type='BN', requires_grad=True)
                 ):
        super(RepVGGNet, self).__init__()
        if isinstance(kernel_size, int):
            kernel_sizes = [kernel_size for _ in range(len(stage_channels))]
        if isinstance(kernel_size, list):
            assert len(kernel_size) == len(stage_channels), \
            "if kernel_size is list, len(kernel_size) should == len(stage_channels)"
            kernel_sizes = kernel_size

        assert num_out <= len(stage_channels), 'num output should be less than stage channels!'

        self.stage_nums = len(stage_channels)
        self.stem = ConvModule(3, stem_channels, kernel_size=3, stride=2, padding=1,
                               norm_cfg=norm_cfg)
        '''defult end_stage is the last stage'''
        self.start_stage = len(stage_channels)-num_out+1

        self.stages = nn.ModuleList()
        self.last_stage = len(stage_channels)
        in_channel = stem_channels
        for num_stages in range(self.stage_nums):
            stage = RepVGGStage(in_channel, stage_channels[num_stages],
                                            block_per_stage[num_stages],
                                            kernel_size=kernel_sizes[num_stages],
                                            group=1)
            in_channel = stage_channels[num_stages]
            self.stages.append(stage)

        self.last_conv = nn.Sequential(
            ConvModule(in_channels=in_channel, out_channels=in_channel, kernel_size=1, padding=0),
            ConvModule(in_channels=in_channel, out_channels=512, kernel_size=5, padding=0)
        )

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            import torch
            assert os.path.isfile(pretrained), "file {} not found.".format(pretrained)
            self.load_state_dict(torch.load(pretrained), strict=False)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')


    def forward(self, x):
        x = self.stem(x)
        for i in range(self.start_stage):
            x = self.stages[i](x)
        out = []
        for i in range(self.start_stage, len(self.stages)):
            out.append(x)
            x = self.stages[i](x)
        out.append(x)
        out.append(self.last_conv(x))
        return out

class RepVggFTNet(nn.Module):
    def __init__(self, stem_channels,
                       stage_channels,
                       block_per_stage,
                       kernel_size=3,
                       num_out=5,
                       norm_cfg=dict(type='BN', requires_grad=True),
                       num_classes=11):
        super(RepVggFTNet, self).__init__()
        self.backbone = RepVGGNet(stem_channels,
                                  stage_channels,
                                  block_per_stage,
                                  kernel_size,
                                  num_out,
                                  norm_cfg=dict(type='BN', requires_grad=True))
        self.backbone.init_weights()
        self.num_classes = num_classes
        self.FTGenerator = FTGenerator(in_channels=64)
        self.fc1 = nn.Sequential(ConvModule(in_channels=512, out_channels=128, kernel_size=1, padding=0, norm_cfg=dict(type='BN', requires_grad=True), act_cfg=None),
                                 nn.Dropout(p=0.2),)

        self.fc2 = ConvModule(in_channels=128, out_channels=num_classes, kernel_size=1, padding=0, act_cfg=None)

    def forward(self, x):
        x_s4 = self.backbone(x)[-1]
        x_s3 = self.backbone(x)[-3]
        feature = self.fc1(x_s4)
        cls = self.fc2(feature)
        ft = self.FTGenerator(x_s3)
        cls = cls.view(-1, 3)


        return cls, ft, feature, self.fc2

        # return cls