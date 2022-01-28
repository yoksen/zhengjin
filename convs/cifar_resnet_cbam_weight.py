import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from collections import OrderedDict

__all__ = ['ResNetKW', 'resnet18_cbam_kw']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class KernelWeight(nn.Module):
    def __init__(self, n_channel):
        super(KernelWeight, self).__init__()
        self.n_channel = n_channel
        self.weights = nn.Parameter(torch.randn(n_channel))

    def forward(self, x):
        return x * self.weights.view(1, self.n_channel, 1, 1)


class Normalization(nn.Module):
    def __init__(self, mean, std, n_channels=3):
        super(Normalization, self).__init__()
        self.n_channels=n_channels
        if mean is None:
            mean = [.0] * n_channels
        if std is None:
            std = [.1] * n_channels
        self.mean = torch.tensor(list(mean)).reshape((1, self.n_channels, 1, 1))
        self.std = torch.tensor(list(std)).reshape((1, self.n_channels, 1, 1))
        self.mean = nn.Parameter(self.mean)
        self.std = nn.Parameter(self.std)
    
    def forward(self, x):
        y = (x - self.mean / self.std)
        return y

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        #my add
        self.kw1 = KernelWeight(planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        #my add
        self.kw2 = KernelWeight(planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)

        out = self.kw1(out)
        
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)

        out = self.kw2(out)

        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNetKW(nn.Module):

    def __init__(self, block, layers, num_classes=100, normed=False):
        self.inplanes = 64
        super(ResNetKW, self).__init__()

        self.norm = Normalization([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
        self.normed = normed

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        
        self.kw = KernelWeight(64)

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.feature = nn.AvgPool2d(4, stride=1)
        self.out_dim = 512 * block.expansion
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                KernelWeight(planes * block.expansion),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.normed:
            x = self.norm(x)
        x = self.conv1(x)
        
        x = self.kw(x)
        
        x = self.bn1(x)
        x = self.relu(x)
        x_1 = self.layer1(x)
        x_2 = self.layer2(x_1)
        x_3 = self.layer3(x_2)
        x_4 = self.layer4(x_3)
        dim = x_4.size()[-1]
        pool = nn.AvgPool2d(dim, stride=1)
        pooled = pool(x_4)
        features = pooled.view(x.size(0), -1)
        return {
            'fmaps': [x_1, x_2, x_3, x_4],
            'features': features
        }


def resnet18_cbam_kw(pretrained=False, normed=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetKW(BasicBlock, [2, 2, 2, 2], normed=normed, **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet18'])
        now_state_dict        = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict)
    return model


def is_fc(name):
    if "fc" in name:
        return True
    else:
        return False

def is_bn(name):
    if "bn" in name or "downsample.2" in name:
        return True
    else:
        return False
    
def is_kw(name):
    if "kw" in name or "downsample.1" in name:
        return True
    else:
        return False

if __name__ == "__main__":
    #change parameter
    # pretrained_dict = torch.load("/data/junjie/code/zhengjin/saved_parameters/imagenet200_simsiam_pretrained_model.pth")
    # state_dict = OrderedDict()
    # for k, v in pretrained_dict.items():
    #     if "downsample.1" in k:
    #         temp = k.split(".")
    #         temp[-2] = "2"
    #         state_dict[".".join(temp)] = v
    #         print(".".join(temp))
    #     else:
    #         state_dict[k] = v
    
    # torch.save(state_dict, "/data/junjie/code/zhengjin/saved_parameters/imagenet200_simsiam_pretrained_model_kw.pth")
    
    model = resnet18_cbam_kw(normed=True)
    for name, param in model.named_parameters():
        if is_bn(name) or is_fc(name) or is_kw(name):
            print(name)
    # pretrained_dict = torch.load("/data/junjie/code/zhengjin/saved_parameters/imagenet200_simsiam_pretrained_model_kw.pth")
    # state_dict = model.state_dict()
    # # print(state_dict["layer4.1.bn2.running_mean"])
    
    # state_dict.update(pretrained_dict)
    # model.load_state_dict(state_dict)
    # for k, v in model.state_dict().items():
    #     if is_bn(k) or is_fc(k) or is_kw(k):
    #         print(k)
    # print(model.state_dict()["layer4.1.bn2.running_mean"])

    # x = torch.randn(4, 3, 32, 32)
    # output = model(x)

    # for k, v in pretrained_dict.items():
    #     if "featureExactor" in k and "fc" not in k:
    #         changed_dict[".".join(k.split(".")[1:])] = v
    
    # torch.save(changed_dict, "/data/junjie/code/zhengjin/saved_parameters/imagenet200_simsiam_pretrained_model.pth")