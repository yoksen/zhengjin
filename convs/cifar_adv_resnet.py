'''
Reference:
https://github.com/khurramjaved96/incremental-learning/blob/autoencoders/model/resnet32.py
'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        # print(self.mean.device)
        y = (x - self.mean / self.std)
        return y

class LinfPGD(nn.Module):
    """Projected Gradient Decent(PGD) attack.
    Can be used to adversarial training.
    """
    def __init__(self, model, epsilon=8/255, step=2/255, iterations=20, criterion=None, random_start=True, targeted=False):
        super(LinfPGD, self).__init__()
        # Arguments of PGD

        self.model = model
        self.epsilon = epsilon
        self.step = step
        self.iterations = iterations
        self.random_start = random_start
        self.targeted = targeted

        self.criterion = criterion
        if self.criterion is None:
            self.criterion = lambda model, input, target: nn.functional.cross_entropy(model(input)["logits"], target)

        # Model status
        self.training = self.model.training

    def project(self, perturbation):
        # Clamp the perturbation to epsilon Lp ball.
        return torch.clamp(perturbation, -self.epsilon, self.epsilon)

    def compute_perturbation(self, adv_x, x):
        # Project the perturbation to Lp ball
        perturbation = self.project(adv_x - x)
        # Clamp the adversarial image to a legal 'image'
        perturbation = torch.clamp(x+perturbation, 0., 1.) - x

        return perturbation

    def onestep(self, x, perturbation, target):
        # Running one step for 
        adv_x = x + perturbation
        adv_x.requires_grad = True
        
        atk_loss = self.criterion(self.model, adv_x, target)

        self.model.zero_grad()
        atk_loss.backward()
        grad = adv_x.grad
        # Essential: delete the computation graph to save GPU ram
        adv_x.requires_grad = False

        if self.targeted:
            adv_x = adv_x.detach() - self.step * torch.sign(grad)
        else:
            adv_x = adv_x.detach() + self.step * torch.sign(grad)
        perturbation = self.compute_perturbation(adv_x, x)

        return perturbation

    def _model_freeze(self):
        for param in self.model.parameters():
            param.requires_grad=False

    def _model_unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad=True

    def random_perturbation(self, x):
        perturbation = torch.rand_like(x).to(device=self.device)
        perturbation = self.compute_perturbation(x+perturbation, x)

        return perturbation

    def attack(self, x, target):
        # x = x.to(self.device)
        # target = target.to(self.device)
        self.device = x.device

        self.training = self.model.training

        self.model.eval()
        self._model_freeze()

        perturbation = torch.zeros_like(x).to(self.device)
        if self.random_start:
            perturbation = self.random_perturbation(x)

        with torch.enable_grad():
            for i in range(self.iterations):
                perturbation = self.onestep(x, perturbation, target)

        self._model_unfreeze()
        if self.training:
            self.model.train()

        return x + perturbation

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

        r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(
            module.running_mean.data - mean, 2)

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

class ResNetBasicblock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super(ResNetBasicblock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv_a = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn_a = nn.BatchNorm2d(planes)

        self.conv_b = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_b = nn.BatchNorm2d(planes)

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
    def __init__(self, block, depth, channels=3, num_classes=10, norm_layer=None):
        super(CifarResNet, self).__init__()

        self.num_classes = num_classes
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        # Model type specifies number of layers for CIFAR-10 and CIFAR-100 model
        assert (depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110'
        layer_blocks = (depth - 2) // 6

        self.norm = Normalization([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
        self.conv_1_3x3 = nn.Conv2d(channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_1 = norm_layer(16)

        self.inplanes = 16
        self.stage_1 = self._make_layer(block, 16, layer_blocks, 1)
        self.stage_2 = self._make_layer(block, 32, layer_blocks, 2)
        self.stage_3 = self._make_layer(block, 64, layer_blocks, 2)
        #change it
        self.avgpool = nn.AvgPool2d(8)
        self.out_dim = 64 * block.expansion
        self.fc = nn.Linear(64*block.expansion, num_classes)

        self.loss_r_feature_layers = []

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

    def forward(self, x, norm=True):
        if norm:
            x = self.norm(x)
        x_0 = self.conv_1_3x3(x)  # [bs, 16, 32, 32]
        x_0 = F.relu(self.bn_1(x_0), inplace=True)

        x_1 = self.stage_1(x_0)  # [bs, 16, 32, 32]
        x_2 = self.stage_2(x_1)  # [bs, 32, 16, 16]
        x_3 = self.stage_3(x_2)  # [bs, 64, 8, 8]

        pooled = self.avgpool(x_3)  # [bs, 64, 1, 1]
        features = pooled.view(pooled.size(0), -1)  # [bs, 64]
        logits = self.fc(features)

        return {
            'fmaps': [x_1, x_2, x_3],
            'features': features,
            'logits': logits
        }
    
    def fv(self, x, norm=True):
        if norm:
            x = self.norm(x)
        x_0 = self.conv_1_3x3(x)  # [bs, 16, 32, 32]
        x_0 = F.relu(self.bn_1(x_0), inplace=True)

        x_1 = self.stage_1(x_0)  # [bs, 16, 32, 32]
        x_2 = self.stage_2(x_1)  # [bs, 32, 16, 16]
        x_3 = self.stage_3(x_2)  # [bs, 64, 8, 8]

        pooled = self.avgpool(x_3)  # [bs, 64, 1, 1]
        features = pooled.view(pooled.size(0), -1)  # [bs, 64]

        return features
    
    def set_hook(self):
        print("register_hook")
        for block in list(self.children())[:-2]:
            if isinstance(block, nn.BatchNorm2d):
                self.loss_r_feature_layers.append(DeepInversionFeatureHook(block))
            elif isinstance(block, nn.Sequential):
                for module in block.modules():
                    if isinstance(module, nn.BatchNorm2d):
                        self.loss_r_feature_layers.append(DeepInversionFeatureHook(module))


    def remove_hook(self):
        print('remove hook!')
        # 关闭所有注册的hook
        for featurehook in self.loss_r_feature_layers:
            featurehook.close()

        # 清空 DeepInversionFeatureHook 实例
        self.loss_r_feature_layers.clear()
    
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=100):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.norm = Normalization([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.feature = nn.AvgPool2d(4, stride=1)
        self.out_dim = 512 * block.expansion
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.loss_r_feature_layers = []

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
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, norm=True):
        if norm:
            x = self.norm(x)
        x = self.conv1(x)
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
        logits = self.fc(features)

        return {
            'fmaps': [x_1, x_2, x_3],
            'features': features,
            'logits': logits
        }
    

    def fv(self, x, norm=True):
        if norm:
            x = self.norm(x)
        x = self.conv1(x)
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

        return features
    
    def set_hook(self):
        print("register_hook")
        for block in list(self.children())[:-2]:
            if isinstance(block, nn.BatchNorm2d):
                self.loss_r_feature_layers.append(DeepInversionFeatureHook(block))
            elif isinstance(block, nn.Sequential):
                for module in block.modules():
                    if isinstance(module, nn.BatchNorm2d):
                        self.loss_r_feature_layers.append(DeepInversionFeatureHook(module))

    def remove_hook(self):
        print('remove hook!')
        # 关闭所有注册的hook
        for featurehook in self.loss_r_feature_layers:
            featurehook.close()

        # 清空 DeepInversionFeatureHook 实例
        self.loss_r_feature_layers.clear()

def resnet32(num_classes=100):
    """Constructs a ResNet-32 model for CIFAR-100."""
    conv_net = CifarResNet(ResNetBasicblock, 32, num_classes)
    return conv_net

def resnet18(num_classes=100):
    """Constructs a ResNet-18 model for CIFAR-100."""
    conv_net = ResNet(ResNetBasicblock, [2, 2, 2, 2], num_classes)
    return conv_net