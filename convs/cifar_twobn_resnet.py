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



# advprop
def get_kernel(size, nsig, mode='gaussian', device='cuda:0'):
    if mode == 'gaussian':
        # since we have to normlize all the numbers
        # there is no need to calculate the const number like \pi and \sigma.
        vec = torch.linspace(-nsig, nsig, steps=size).to(device)
        vec = torch.exp(-vec * vec / 2)
        res = vec.view(-1, 1) @ vec.view(1, -1)
        res = res / torch.sum(res)
    elif mode == 'linear':
        # originally, res[i][j] = (1-|i|/(k+1)) * (1-|j|/(k+1))
        # since we have to normalize it
        # calculate res[i][j] = (k+1-|i|)*(k+1-|j|)
        vec = (size + 1) / 2 - torch.abs(torch.arange(-(size + 1) / 2, (size + 1) / 2 + 1, step=1)).to(device)
        res = vec.view(-1, 1) @ vec.view(1, -1)
        res = res / torch.sum(res)
    else:
        raise ValueError("no such mode in get_kernel.")
    return res


class NoOpAttacker():

    def attack(self, image, label, model):
        return image, -torch.ones_like(label)


class PGDAttacker():
    def __init__(self, num_iter, epsilon, step_size, kernel_size=15, prob_start_from_clean=0.0, translation=False,
                 device='cuda:0'):
        step_size = max(step_size, epsilon / num_iter)
        self.num_iter = num_iter
        self.epsilon = epsilon * IMAGE_SCALE
        self.step_size = step_size * IMAGE_SCALE
        self.prob_start_from_clean = prob_start_from_clean
        self.device = device
        self.translation = translation
        if translation:
            # this is equivalent to deepth wise convolution
            # details can be found in the docs of Conv2d.
            # "When groups == in_channels and out_channels == K * in_channels, where K is a positive integer, this operation is also termed in literature as depthwise convolution."
            self.conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=kernel_size, stride=(kernel_size - 1) // 2,
                                  bias=False, groups=3).to(self.device)
            self.gkernel = get_kernel(kernel_size, nsig=3, device=self.device).to(self.device)
            self.conv.weight = self.gkernel

    def _create_random_target(self, label, num_classes=1000):
        label_offset = torch.randint_like(label, low=0, high=num_classes)
        return (label + label_offset) % num_classes

    def attack(self, image_clean, label, model, fc_layer , original=False, num_classes=1000):
        if original:
            target_label = label
        else:
            target_label = self._create_random_target(label, num_classes)
        lower_bound = torch.clamp(image_clean - self.epsilon, min=-1., max=1.)
        upper_bound = torch.clamp(image_clean + self.epsilon, min=-1., max=1.)

        ori_images = image_clean.clone().detach()

        init_start = torch.empty_like(image_clean).uniform_(-self.epsilon, self.epsilon)

        start_from_noise_index = (torch.randn([]) > self.prob_start_from_clean).float()
        start_adv = image_clean + start_from_noise_index * init_start

        adv = start_adv
        for i in range(self.num_iter):
            adv.requires_grad = True
            # logits = model(adv)
            features = model(adv)['features']

            # print('attack fc_layer',fc_layer)
            logits = fc_layer(features)['logits']

            losses = F.cross_entropy(logits, target_label)
            g = torch.autograd.grad(losses, adv,
                                    retain_graph=False, create_graph=False)[0]
            if self.translation:
                g = self.conv(g)
            if original:
                adv = adv + torch.sign(g) * self.step_size
            else:
                adv = adv - torch.sign(g) * self.step_size
            adv = torch.where(adv > lower_bound, adv, lower_bound)
            adv = torch.where(adv < upper_bound, adv, upper_bound).detach()

        return adv, target_label


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
        self.batch_type = 'clean'

    def forward(self, input):
        if self.batch_type == 'adv':
            input = self.aux_bn(input)
        elif self.batch_type == 'clean':
            input = super(MixBatchNorm2d, self).forward(input)
        else:
            assert self.batch_type == 'mix'
            batch_size = input.shape[0]
            # input0 = self.aux_bn(input[: batch_size // 2])
            # input1 = super(MixBatchNorm2d, self).forward(input[batch_size // 2:])
            input0 = super(MixBatchNorm2d, self).forward(input[:batch_size // 2])
            input1 = self.aux_bn(input[batch_size // 2:])
            input = torch.cat((input0, input1), 0)
        return input


def to_status(m, status):
    if hasattr(m, 'batch_type'):
        m.batch_type = status


to_clean_status = partial(to_status, status='clean')
to_adv_status = partial(to_status, status='adv')
to_mix_status = partial(to_status, status='mix')


class DeepInversionFeatureHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''

    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)

        # forcing mean and variance to match between two distributions
        # other ways might work better, i.g. KL divergence

        if module.batch_type == 'clean':
            r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(
                module.running_mean.data - mean, 2)

        if module.batch_type == 'adv':
            r_feature = torch.norm(module.aux_bn.running_var.data - var, 2) + torch.norm(
                module.aux_bn.running_mean.data - mean, 2)

        self.r_feature = r_feature
        # must have no output

    def close(self):
        self.hook.remove()





class DownsampleA(nn.Module):
    def __init__(self, nIn, nOut, stride):
        super(DownsampleA, self).__init__()
        assert stride == 2
        self.avg = nn.AvgPool2d(kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.avg(x)
        return torch.cat((x, x.mul(0)), 1)


class DownsampleB(nn.Module):
    def __init__(self, nIn, nOut, stride):
        super(DownsampleB, self).__init__()
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=1, stride=stride, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(nOut)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class DownsampleC(nn.Module):
    def __init__(self, nIn, nOut, stride):
        super(DownsampleC, self).__init__()
        assert stride != 1 or nIn != nOut
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=1, stride=stride, padding=0, bias=False)

    def forward(self, x):
        x = self.conv(x)
        return x


class DownsampleD(nn.Module):
    def __init__(self, nIn, nOut, stride):
        super(DownsampleD, self).__init__()
        assert stride == 2
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=2, stride=stride, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(nOut)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class ResNetBasicblock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None ,norm_layer=None):
        super(ResNetBasicblock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv_a = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        # self.bn_a = nn.BatchNorm2d(planes)
        self.bn_a = norm_layer(planes)

        self.conv_b = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn_b = nn.BatchNorm2d(planes)
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

        # Model type specifies number of layers for CIFAR-10 and CIFAR-100 model
        assert (depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110'
        layer_blocks = (depth - 2) // 6

        self.conv_1_3x3 = nn.Conv2d(channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_1 = norm_layer(16)

        self.inplanes = 16
        self.stage_1 = self._make_layer(block, 16, layer_blocks, 1)
        self.stage_2 = self._make_layer(block, 32, layer_blocks, 2)
        self.stage_3 = self._make_layer(block, 64, layer_blocks, 2)
        #change it
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.avgpool = nn.AvgPool2d(8)
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

    def _forward_impl(self, x):
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



class CifarAdvResNet(CifarResNet):
    '''
    The modified model using ResNet in torchvision.models.resnet.
    Usually we using DataParallel to wrap this model,
    so you'd better set the attacker and mixbn before using DataParallel.
    '''

    def __init__(self, block, depth, channels = 3 , num_classes=100,
                 norm_layer=None, attacker=NoOpAttacker()):

        super().__init__(block, depth, channels=channels  ,num_classes=num_classes ,norm_layer=norm_layer)

        self.attacker = attacker
        self.mixbn = False
        self.loss_r_feature_layers = []

    def set_attacker(self, attacker):
        self.attacker = attacker

    def set_mixbn(self, mixbn):
        self.mixbn = mixbn

    def forward(self, x, labels , cur_fc_layer):
        training = self.training
        input_len = len(x)
        # only during training do we need to attack, and cat the clean and auxiliary pics
        if training:
            self.eval()
            # 是否产生对抗样本
            self.apply(to_adv_status)
            if isinstance(self.attacker, NoOpAttacker):
                images = x
                targets = labels
            else:
                aux_images, _ = self.attacker.attack(x, labels, self._forward_impl,cur_fc_layer ,num_classes=cur_fc_layer.out_features)
                images = torch.cat([x, aux_images], dim=0)
                targets = torch.cat([labels, labels], dim=0)
            self.train()
            # 是否使用双bn
            if self.mixbn:
                # the DataParallel usually cat the outputs along the first dimension simply,
                # so if we don't change the dimensions, the outputs will be something like
                # [clean_batches_gpu1, adv_batches_gpu1, clean_batches_gpu2, adv_batches_gpu2...]
                # Then it will be hard to distinguish clean batches and adversarial batches.
                self.apply(to_mix_status)
                # return self._forward_impl(images).view(2, input_len, -1).transpose(1, 0), targets.view(2,input_len).transpose(1, 0)
                return self._forward_impl(images), targets
            else:
                self.apply(to_clean_status)
                return self._forward_impl(images), targets

        else:
            self.apply(to_clean_status)
            images = x
            targets = labels

            # self.apply(to_adv_status)
            # self.eval()

            # aux_images,_ = self.attacker.attack(x, labels, self._forward_impl, num_classes=self.num_classes)
            # images = aux_images
            # images = torch.cat([x, aux_images], dim = 0)
            # targets = torch.cat([labels, labels], dim = 0)
            return self._forward_impl(images), targets

    def fv(self, x):

        x = self.conv_1_3x3(x)  # [bs, 16, 32, 32]
        x = F.relu(self.bn_1(x), inplace=True)

        x_1 = self.stage_1(x)  # [bs, 16, 32, 32]
        x_2 = self.stage_2(x_1)  # [bs, 32, 16, 16]
        x_3 = self.stage_3(x_2)  # [bs, 64, 8, 8]

        pooled = self.avgpool(x_3)  # [bs, 64, 1, 1]
        features = pooled.view(pooled.size(0), -1)  # [bs, 64]
        # logits = self.fc(features)

        return features


    def fm(self, x):

        x = self.conv_1_3x3(x)  # [bs, 16, 32, 32]
        x = F.relu(self.bn_1(x), inplace=True)

        x_1 = self.stage_1(x)  # [bs, 16, 32, 32]
        x_2 = self.stage_2(x_1)  # [bs, 32, 16, 16]
        x_3 = self.stage_3(x_2)  # [bs, 64, 8, 8]

        # pooled = self.avgpool(x_3)  # [bs, 64, 1, 1]
        # features = pooled.view(pooled.size(0), -1)  # [bs, 64]
        # logits = self.fc(features)

        return x_3


    def set_hook(self):
        print("register_hook")
        for block in list(self.children())[:-2]:
            # print((type(block)))
            if isinstance(block, MixBatchNorm2d):
                # print("register")
                self.loss_r_feature_layers.append(DeepInversionFeatureHook(block))
            elif isinstance(block, nn.Sequential):
                for module in block.modules():
                    # print(type(module))
                    if isinstance(module, MixBatchNorm2d):
                        # print("register")
                        self.loss_r_feature_layers.append(DeepInversionFeatureHook(module))

        # for block in list(self.children())[:-2]:
        #     print((type(block)))
        #     if isinstance(block, nn.BatchNorm2d):
        #         # print("register")
        #         self.loss_r_feature_layers.append(DeepInversionFeatureHook(block))
        #     elif isinstance(block, nn.Sequential):
        #         for module in block.modules():
        #             print(type(module))
        #             if isinstance(module, nn.BatchNorm2d):
        #                 # print("register")
        #                 self.loss_r_feature_layers.append(DeepInversionFeatureHook(module))
    def remove_hook(self):

        print('remove hook!')
        # 关闭所有注册的hook
        for featurehook in self.loss_r_feature_layers:
            featurehook.close()

        # 清空 DeepInversionFeatureHook 实例
        self.loss_r_feature_layers.clear()


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

def resnet32( mix = False , num_classes = 100):
    """Constructs a ResNet-32 model for CIFAR-10."""
    # if mix:
    #     model = CifarResNet(ResNetBasicblock, 32 , norm_layer=MixBatchNorm2d)
    # else:
    #     model = CifarResNet(ResNetBasicblock, 32)
    # return model

    if mix:
        attacker = PGDAttacker(num_iter=1, epsilon=1, step_size=1)
        model = CifarAdvResNet(ResNetBasicblock, 32 , num_classes=num_classes  , norm_layer=MixBatchNorm2d , attacker=attacker)
        model.set_mixbn(True)
    else:
        model = CifarAdvResNet(ResNetBasicblock, 32 , num_classes=num_classes)
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
