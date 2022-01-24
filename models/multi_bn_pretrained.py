import logging
from statistics import mode
from matplotlib.pyplot import cla
import numpy as np
import os
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net import IncrementalNet
from utils.toolkit import target2onehot, tensor2numpy
from convs.linears import SimpleLinear

EPSILON = 1e-8

# CIFAR100, resnet18_cbam
# epochs_init = 101
epochs_init = 5
lrate_init = 1e-4
milestones_init = [45, 90]
lrate_decay_init = 0.1
weight_decay_init = 2e-4
class_aug = False
fix_parameter = True

# epochs = 101
epochs = 5
lrate = 1e-3
milestones = [45, 90]
lrate_decay = 0.1
weight_decay = 2e-4  # illness
optim_type = "adam"
batch_size = 64
reset_bn = False


# CIFAR100, ResNet32
# epochs_init = 70
# lrate_init = 1e-2
# milestones_init = [49, 63]
# lrate_decay_init = 0.1
# weight_decay_init = 1e-5


# epochs = 70
# lrate = 1e-2
# milestones = [49, 63]
# lrate_decay = 0.1
# weight_decay = 1e-5  # illness
# optim_type = "adam"
# batch_size = 128

num_workers = 4
hyperparameters = ["epochs_init", "lrate_init", "milestones_init", "lrate_decay_init",
                   "weight_decay_init", "epochs","lrate", "milestones", "lrate_decay", 
                   "weight_decay","batch_size", "num_workers", "optim_type", "reset_bn", "class_aug", "fix_parameter"]


def is_fc(name):
    if "fc" in name:
        return True
    else:
        return False

def is_bn(name):
    if "running_mean" in name or "running_var" in name or "num_batches_tracked" in name:
        return True
    else:
        return False

class multi_bn_pretrained(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._networks = []
        self._convnet_type = args['convnet_type']
        assert args['convnet_type'] == "resnet18_cbam", "wrong convnet_type"
        self._seed = args['seed']

        # log hyperparameter
        logging.info(50*"-")
        logging.info("log_hyperparameters")
        logging.info(50*"-")
        for item in hyperparameters:
            logging.info('{}: {}'.format(item, eval(item)))

    def after_task(self):
        self._known_classes = self._total_classes
        if self._cur_task == 0:
            if not os.path.exists("./saved_model/multi_bn_pretrained_{}.pth".format(self._seed)):
                torch.save(self._networks[self._cur_task].state_dict(), "./saved_model/multi_bn_pretrained_{}.pth".format(self._seed))

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._cur_class = data_manager.get_task_size(self._cur_task)
        self._total_classes = self._known_classes + self._cur_class

        self._networks.append(IncrementalNet(self._convnet_type, False))
        if self._cur_task == 0:
            #load pretrained model
            state_dict = self._networks[self._cur_task].convnet.state_dict()
            pretrained_dict = torch.load("./saved_parameters/imagenet200_simsiam_pretrained_model.pth")
            state_dict.update(pretrained_dict)
            self._networks[self._cur_task].convnet.load_state_dict(state_dict)

            #compare the difference between using and unusing class augmentation in first session
            if class_aug:
                self.augnumclass = self._total_classes + int(self._cur_class*(self._cur_class-1)/2)
                self._networks[self._cur_task].update_fc(self.augnumclass)
            else:
                self._networks[self._cur_task].update_fc(self._cur_class)
            # self._network.update_fc(self.augnumclass)
        else:
            self._networks[self._cur_task].update_fc(data_manager.get_task_size(self._cur_task))
            state_dict = self._networks[self._cur_task].convnet.state_dict()
            state_dict.update(self._networks[0].convnet.state_dict())
            # print(self._networks[self._cur_task].convnet.state_dict()["layer4.1.bn2.running_mean"])
            
            self._networks[self._cur_task].convnet.load_state_dict(state_dict)
            # print(self._networks[self._cur_task].convnet.state_dict()["layer4.1.bn2.running_mean"])
            # self._networks[self._cur_task].state_dict().update(self._networks[0].state_dict())
            if reset_bn:
                self.reset_bn(self._networks[self._cur_task].convnet)
            # print(self._networks[self._cur_task].convnet.state_dict()["layer4.1.bn2.running_mean"])
        
        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        # Loader
        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train',
                                                mode='train')
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='test', 
                                                mode='test')
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        # Procedure
        if len(self._multiple_gpus) > 1:
            self._networks[self._cur_task] = nn.DataParallel(self._networks[self._cur_task], self._multiple_gpus)
        
        self._train(self._networks[self._cur_task], self.train_loader, self.test_loader)

    def _train(self, model, train_loader, test_loader):
        model.to(self._device)
        
        if self._cur_task == 0:
            if fix_parameter:
                for name, param in model.named_parameters():
                    if is_fc(name) or is_bn(name):
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                if optim_type == "adam":
                    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lrate_init, weight_decay=weight_decay_init)
                else:
                    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lrate_init, momentum=0.9, weight_decay=weight_decay_init)  # 1e-3
            
            else:
                if optim_type == "adam":
                    optimizer = optim.Adam(model.parameters(), lr=lrate_init, weight_decay=weight_decay_init)
                else:
                    optimizer = optim.SGD(model.parameters(), lr=lrate_init, momentum=0.9, weight_decay=weight_decay_init)  # 1e-3
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones_init, gamma=lrate_decay_init)
                # for name, param in model.named_parameters():
                #     if param.requires_grad:
                #         print(name)
                        # param.requires_grad = True
        else:
            for name, param in model.named_parameters():
                if is_fc(name) or is_bn(name):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            
            # for name, param in model.named_parameters():
            #     if param.requires_grad:
            #         print(name)
                    # param.requires_grad = True
            
            if optim_type == "adam":
                optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lrate, weight_decay=weight_decay)
            else:
                optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lrate, weight_decay=weight_decay)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=lrate_decay)
        self._update_representation(model, train_loader, test_loader, optimizer, scheduler)

    def reset_bn(self, model):
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                # print("reset_bn")
                m.reset_running_stats()
                m.reset_parameters()

    def _update_representation(self, model, train_loader, test_loader, optimizer, scheduler):
        if self._cur_task == 0:
            epochs_num = epochs_init
        else:
            epochs_num = epochs

        prog_bar = tqdm(range(epochs_num))

        #if temp < 1, it will make the output of softmax sharper
        temp = 0.1
        for _, epoch in enumerate(prog_bar):
            model.train()
            losses = 0.
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                if self._cur_task == 0:
                    if class_aug:
                        inputs, targets = self.classAug(inputs, targets)
                    logits = model(inputs)['logits']
                else:
                    logits = model(inputs)['logits']

                loss = nn.CrossEntropyLoss()(logits/temp, targets - self._known_classes)
                # loss = F.binary_cross_entropy_with_logits(logits, onehots)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                # acc
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq((targets - self._known_classes).expand_as(preds)).cpu().sum()
                total += len(targets)
            
            if self._cur_task == 0 and epoch == epochs_num - 1 and class_aug:
                weight = model.fc.weight.data
                bias = model.fc.bias.data
                in_feature = model.fc.in_features
                model.fc = SimpleLinear(in_feature, self._total_classes)
                model.fc.weight.data = weight[:self._total_classes]
                model.fc.bias.data = bias[:self._total_classes]
                print("The num of total classes is {}".format(self._total_classes))

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)
            test_acc = self._compute_accuracy(model, test_loader)
            info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                self._cur_task, epoch+1, epochs_num, losses/len(train_loader), train_acc, test_acc)
            prog_bar.set_description(info)

            logging.info(info)

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)['logits']
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == (targets - self._known_classes)).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct)*100 / total, decimals=2)

    #at most, the num of samples will be 5 times of origin
    def classAug(self, x, y, alpha=20.0, mix_times=4):  # mixup based
        batch_size = x.size()[0]
        mix_data = []
        mix_target = []
        for _ in range(mix_times):
            #Returns a random permutation of integers 
            index = torch.randperm(batch_size).to(self._device)
            for i in range(batch_size):
                if y[i] != y[index][i]:
                    new_label = self.generate_label(y[i].item(), y[index][i].item())
                    lam = np.random.beta(alpha, alpha)
                    if lam < 0.4 or lam > 0.6:
                        lam = 0.5
                    mix_data.append(lam * x[i] + (1 - lam) * x[index, :][i])
                    mix_target.append(new_label)

        new_target = torch.Tensor(mix_target)
        y = torch.cat((y, new_target.to(self._device).long()), 0)
        for item in mix_data:
            x = torch.cat((x, item.unsqueeze(0)), 0)
        return x, y
    
    def generate_label(self, y_a, y_b):
        if self._old_network == None:
            y_a, y_b = y_a, y_b
            #make sure y_a < y_b
            assert y_a != y_b
            if y_a > y_b:
                tmp = y_a
                y_a = y_b
                y_b = tmp
            #calculate the sum of arithmetic sequence and then sum the bias
            label_index = ((2 * self._total_classes - y_a - 1) * y_a) / 2 + (y_b - y_a) - 1
        else:
            y_a = y_a - self._known_classes
            y_b = y_b - self._known_classes
            assert y_a != y_b
            if y_a > y_b:
                tmp = y_a
                y_a = y_b
                y_b = tmp
            label_index = int(((2 * self._cur_class - y_a - 1) * y_a) / 2 + (y_b - y_a) - 1)
        return label_index + self._total_classes