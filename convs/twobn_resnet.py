import torch
import torch.nn as nn
# from torchvision.models.utils import load_state_dict_from_url
from functools import partial
import torch.nn.functional as F
IMAGE_SCALE = 2.0/255


# __all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
#            'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
#            'wide_resnet50_2', 'wide_resnet101_2']

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



model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


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


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.num_classes = num_classes
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_dim = 512 * block.expansion
        # self.fc = nn.Linear(512 * block.expansion, self.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x_1 = self.layer1(x)
        x_2 = self.layer2(x_1)
        x_3 = self.layer3(x_2)
        x_4 = self.layer4(x_3)

        pooled = self.avgpool(x_4)
        features = torch.flatten(pooled, 1)
        # x = self.fc(x)

        return {
            'fmaps': [x_1, x_2, x_3, x_4],
            'features': features
        }

    def forward(self, x):
        return self._forward_impl(x)


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


class AdvResNet(ResNet):
    '''
    The modified model using ResNet in torchvision.models.resnet.
    Usually we using DataParallel to wrap this model,
    so you'd better set the attacker and mixbn before using DataParallel.
    '''

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, attacker=NoOpAttacker()):
        super().__init__(block, layers, num_classes=num_classes, zero_init_residual=zero_init_residual,
                         groups=groups, width_per_group=width_per_group,
                         replace_stride_with_dilation=replace_stride_with_dilation,
                         norm_layer=norm_layer)
        self.attacker = attacker
        self.mixbn = False
        self.loss_r_feature_layers = []

    def set_attacker(self, attacker):
        self.attacker = attacker

    def set_mixbn(self, mixbn):
        self.mixbn = mixbn

    def forward(self, x, labels , cur_fc_layer):
        # print( 'AdvResNet(ResNet):  input ' , type(x) )

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

        self.apply(to_clean_status)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        return x


    def fm(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.avgpool(x)

        return x


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
    def remove_hook(self):

        print('remove hook!')
        # 关闭所有注册的hook
        for featurehook in self.loss_r_feature_layers:
            featurehook.close()

        # 清空 DeepInversionFeatureHook 实例
        self.loss_r_feature_layers.clear()


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = AdvResNet(block, layers, **kwargs)
    if pretrained:
        raise ValueError('do not set pretrained as True, since we aim at training from scratch')
        # state_dict = load_state_dict_from_url(model_urls[arch],
        #                                       progress=progress)
        # model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)
