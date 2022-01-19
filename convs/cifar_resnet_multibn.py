'''
Reference:
https://github.com/khurramjaved96/incremental-learning/blob/autoencoders/model/resnet32.py
'''
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
IMAGE_SCALE = 2.0/255

class MixBatchNorm2d(nn.BatchNorm2d):
    '''
    if the dimensions of the tensors from dataloader is [N, 3, 224, 224]
    that of the inputs of the MixBatchNorm2d should be [2*N, 3, 224, 224].

    If you set batch_type as 'mix', this network will using one batchnorm (main bn) to calculate the features corresponding to[:N, 3, 224, 224],
    while using another batch normalization (auxiliary bn) for the features of [N:, 3, 224, 224].
    During training, the batch_type should be set as 'mix'.

    During validation, we only need the results of the features using some specific batchnormalization.
    if you set batch_type as 'clean', the features are calculated using main bn; if you set it as 'adv', the features are calculated using auxiliary bn.

    Usually, we use to_clean_status, to_adv_status, and to_mix_status to set the batch_type recursively. It should be noticed that the batch_type should be set as 'adv' while attacking.
    '''

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(MixBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.aux_bn = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum, affine=affine,
                                     track_running_stats=track_running_stats)
        self.task_id = 0

    def forward(self, input):

        if self.task_id == 0:
            out1 = super(MixBatchNorm2d, self).forward(input)
            out2 = None
        elif self.task_id == 1:
            out1 = None
            out2 = self.aux_bn(input)
        elif self.task_id == 2:
            out1 = super(MixBatchNorm2d, self).forward(input)
            out2 = self.aux_bn(input)
        else:
            assert 1==0

        return {
            "out1": out1,
            "out2": out2
        }


def to_status(m, status):
    if hasattr(m, 'task_id'):
        m.task_id = status


to_task_0 = partial(to_status, status=0)
to_task_1 = partial(to_status, status=1)
to_task_2 = partial(to_status, status=2)


class DownsampleA(nn.Module):
    def __init__(self, nIn, nOut, stride):
        super(DownsampleA, self).__init__()
        assert stride == 2
        self.avg = nn.AvgPool2d(kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.avg(x)
        return torch.cat((x, x.mul(0)), 1)


class ResNetBasicblock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None ,norm_layer=None):
        super(ResNetBasicblock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv_a = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn_a = norm_layer(planes)

        self.conv_b = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_b = norm_layer(planes)

        self.downsample = downsample

    def forward(self, x):
        residual = x

        basicblock = self.conv_a(x)
        basicblock = self.bn_a(basicblock)
        basicblock = F.relu(basicblock, inplace=True)

        basicblock = self.conv_b(basicblock)
        basicblock = self.bn_b(basicblock)

        if self.downsample is not None:
            residual = self.downsample(x)

        return F.relu(residual + basicblock, inplace=True)


class CifarResNet(nn.Module):
    """
    ResNet optimized for the Cifar Dataset, as specified in
    https://arxiv.org/abs/1512.03385.pdf
    """

    def __init__(self, block, depth, channels=3 , num_classes=10 , norm_layer = None ):
        super(CifarResNet, self).__init__()

        self.num_classes = num_classes
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.cur_task_id = 0

        # Model type specifies number of layers for CIFAR-10 and CIFAR-100 model
        assert (depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110'
        layer_blocks = (depth - 2) // 6

        self.conv_1_3x3 = nn.Conv2d(channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_1 = norm_layer(16)

        self.inplanes = 16
        self.stage_1 = self._make_layer(block, 16, layer_blocks, 1)
        self.stage_2 = self._make_layer(block, 32, layer_blocks, 2)
        self.stage_3 = self._make_layer(block, 64, layer_blocks, 2)
        self.avgpool = nn.AvgPool2d(8)
        self.out_dim = 64 * block.expansion
        # self.fc = nn.Linear(64*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = DownsampleA(self.inplanes, planes * block.expansion, stride)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample , norm_layer = self._norm_layer))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes , norm_layer = self._norm_layer ))

        return nn.Sequential(*layers)

    def forward(self, x):

        if self.cur_task_id == 0:
            self.apply(to_task_0)
        elif self.cur_task_id == 1:
            self.apply(to_task_1)
        else:
            print('gnn')
            assert 1 == 0

        x = self.conv_1_3x3(x)  # [bs, 16, 32, 32]
        x = F.relu(self.bn_1(x), inplace=True)

        x_1 = self.stage_1(x)  # [bs, 16, 32, 32]
        x_2 = self.stage_2(x_1)  # [bs, 32, 16, 16]
        x_3 = self.stage_3(x_2)  # [bs, 64, 8, 8]

        pooled = self.avgpool(x_3)  # [bs, 64, 1, 1]
        features = pooled.view(pooled.size(0), -1)  # [bs, 64]


        # logits = self.fc(features)
        # return logits
        return {
            'fmaps': [x_1, x_2, x_3],
            'features': features
        }

    @property
    def last_conv(self):
        return self.stage_3[-1].conv_b


def resnet20mnist():
    """Constructs a ResNet-20 model for MNIST."""
    model = CifarResNet(ResNetBasicblock, 20, 1)
    return model


def resnet32mnist():
    """Constructs a ResNet-32 model for MNIST."""
    model = CifarResNet(ResNetBasicblock, 32, 1)
    return model


def resnet20():
    """Constructs a ResNet-20 model for CIFAR-10."""
    model = CifarResNet(ResNetBasicblock, 20)
    return model


# def resnet32():
#     """Constructs a ResNet-32 model for CIFAR-10."""
#     model = CifarResNet(ResNetBasicblock, 32)
#     return model

def resnet32( num_classes = 100):
    """Constructs a ResNet-32 model for CIFAR-10."""
    # if mix:
    #     model = CifarResNet(ResNetBasicblock, 32 , norm_layer=MixBatchNorm2d)
    # else:
    #     model = CifarResNet(ResNetBasicblock, 32)
    # return model
    model = CifarResNet(ResNetBasicblock, 32 , norm_layer=MixBatchNorm2d)

    return model



def resnet44():
    """Constructs a ResNet-44 model for CIFAR-10."""
    model = CifarResNet(ResNetBasicblock, 44)
    return model


def resnet56():
    """Constructs a ResNet-56 model for CIFAR-10."""
    model = CifarResNet(ResNetBasicblock, 56)
    return model


def resnet110():
    """Constructs a ResNet-110 model for CIFAR-10."""
    model = CifarResNet(ResNetBasicblock, 110)
    return model
