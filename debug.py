

import torch
from utils.inc_net import IncrementalNet , Twobn_IncrementalNet
from torchvision import transforms
from torchvision import datasets
import numpy as np
from torch.nn import functional as F
import convs.cifar_multibn_resnet as cifar_multibn_resnet

# icarl
#
# {
#     "prefix": "reproduce",
#     "dataset": "imagenet100",
#     "memory_size": 2000,
#     "memory_per_class": 300,
#     "fixed_memory": true,
#     "shuffle": true,
#
#     "init_cls": 10,
#     "increment": 10,
#     "model_name": "icarl",
#     "convnet_type": "resnet18",
#     "device": ["1"],
#     "seed": [100, 60, 50],
#
#
#     "_about_ucir_": true,
#     "lr":2.0,
#     "lamda_base":10,
#     "pretrained":false,
#     "input_size":128,
#     "k":2,
#     "eval_nme":true
# }




def check_net():
    device = torch.device('cpu')

    net = Twobn_IncrementalNet( 'twobn_resnet' , False ).to(device)
    net.update_fc(10)

    net.freeze()
    print(net.convnet)


    images = torch.randn(4,3,224,224).to(device)
    labels = torch.tensor([1,2,3,4]).to(device)

    out , labels = net(images , labels)

    print('out keys' , out.keys())
    print('out logits shape' , out['logits'].shape)
    print('labels  ' , labels)




def split_images_labels(imgs):
    # split trainset.imgs in ImageFolder
    images = []
    labels = []
    for item in imgs:
        images.append(item[0])
        labels.append(item[1])

    return np.array(images), np.array(labels)


class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None


class iImageNet100(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255)
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(100).tolist()

    def download_data(self):
        train_dir = '/GPUFS/sysu_rxwang_1/data/miniImageNet/train/'
        test_dir = '/GPUFS/sysu_rxwang_1/data/miniImageNet/test/'

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

def ignore():
    logits = torch.tensor([[-1.0339, -0.0994,  1.7978],
                           [ 1.0141, -0.1629, -0.0178],
                           [-0.0725,  1.0893, -1.4079]])

    # logits = torch.tensor([[-1.0339, -0.0994,  1.7978],
    #                        [ 1.0141, -0.1629, -0.0178],
    #                        [10,  10, 10]])

    print(logits)
    lables = torch.tensor([0,1,2])

    loss = F.cross_entropy(logits,lables , ignore_index=2)
    print(loss)


if __name__ == '__main__':

    x = torch.randn(2, 3)
    print(x)

    result =  torch.cat(x, 0)
    print(result)