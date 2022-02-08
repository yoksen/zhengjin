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
# # epochs_init = 5
# lrate_init = 1e-3
# milestones_init = [45, 90]
# lrate_decay_init = 0.1
# weight_decay_init = 2e-4

# epochs = 101
# # epochs = 5
# lrate = 1e-3
# milestones = [45, 90]
# lrate_decay = 0.1
# weight_decay = 2e-4  # illness
# optim_type = "adam"
# batch_size = 64

#temp is used for softmax default 0.1
temp = 0.1

#whether first session fix convs layers
fix_parameter = False

#update bn type
#["default", "last", "first", "pretrained"]
# bn_type = "default"
bn_type = "last"
# bn_type = "first"


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

# sd198, ResNet18
epochs_init = 200
# epochs_init = 5
lrate_init = 1e-2
milestones_init = [100, 150]
lrate_decay_init = 0.1
weight_decay_init = 1e-5

epochs = 200
# epochs = 5
lrate = 1e-2
milestones = [100, 150]
lrate_decay = 0.1
weight_decay = 1e-5  # illness
optim_type = "sgd"
batch_size = 32
#temp is used for softmax default 0.1
temp = 1

num_workers = 4
hyperparameters = ["epochs_init", "lrate_init", "milestones_init", "lrate_decay_init",
                   "weight_decay_init", "epochs","lrate", "milestones", "lrate_decay", 
                   "weight_decay", "batch_size", "num_workers", "optim_type", "fix_parameter", 
                   "bn_type", "temp"]

class multi_bn(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._networks = []
        self._convnet_type = args['convnet_type']
        # assert args['convnet_type'] == "resnet18_cbam", "wrong convnet_type"
        self._seed = args['seed']
        self._task_acc = []
        self._init_cls = args['init_cls']
        self._increment = args['increment']

        # log hyperparameter
        logging.info(50*"-")
        logging.info("log_hyperparameters")
        logging.info(50*"-")
        for item in hyperparameters:
            logging.info('{}: {}'.format(item, eval(item)))

    def after_task(self):
        self._known_classes = self._total_classes

        weighted_accs = self.caculate_weighted_average(self._init_cls, self._increment, self._task_acc)
        logging.info(50*"-")
        logging.info("log_accs")
        logging.info(50*"-")
        
        logging.info("task acc is {}".format(self._task_acc))
        logging.info("weighted acc is {}".format(weighted_accs))

        if self._cur_task == 0:
            if not os.path.exists("./saved_model/multi_bn_{}.pth".format(self._seed)):
                torch.save(self._networks[self._cur_task].state_dict(), "./saved_model/multi_bn_{}.pth".format(self._seed))

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._cur_class = data_manager.get_task_size(self._cur_task)
        self._total_classes = self._known_classes + self._cur_class

        self._networks.append(IncrementalNet(self._convnet_type, False))

        if self._convnet_type == "resnet32":
            dst_key = "stage_3.4.bn_b."
        elif self._convnet_type in ["resnet18_cbam", "resnet18", "resnet101"]:
            dst_key = "layer4.1.bn2."
        # elif self._convnet_type == "resnet18":
        #     dst_key = "layer4.1.bn2."

        if self._cur_task == 0:
            self._networks[self._cur_task].update_fc(self._cur_class)

        else:
            self._networks[self._cur_task].update_fc(data_manager.get_task_size(self._cur_task))
            state_dict = self._networks[self._cur_task].convnet.state_dict()
            logging.info("{}running_mean before update: {}".format(dst_key, self._networks[self._cur_task].convnet.state_dict()[dst_key + "running_mean"][:5]))
            logging.info("{}weight before update: {}".format(dst_key, self._networks[self._cur_task].convnet.state_dict()[dst_key + "weight"][:5]))
            logging.info("{}bias before update: {}".format(dst_key, self._networks[self._cur_task].convnet.state_dict()[dst_key + "bias"][:5]))
            
            #["default", "last", "first", "pretrained"]
            if bn_type == "default":
                logging.info("update_bn_with_default_setting")
                state_dict.update(self._networks[self._cur_task - 1].convnet.state_dict())
                self._networks[self._cur_task].convnet.load_state_dict(state_dict)
                self.reset_bn(self._networks[self._cur_task].convnet)
            elif bn_type == "last":
                logging.info("update_bn_with_last_model")
                state_dict.update(self._networks[self._cur_task - 1].convnet.state_dict())
                self._networks[self._cur_task].convnet.load_state_dict(state_dict)
            elif bn_type == "first":
                logging.info("update_bn_with_first_model")
                state_dict.update(self._networks[0].convnet.state_dict())
                self._networks[self._cur_task].convnet.load_state_dict(state_dict)
    
            logging.info("{}running_mean after update: {}".format(dst_key, self._networks[self._cur_task].convnet.state_dict()[dst_key + "running_mean"][:5]))
            logging.info("{}weight after update: {}".format(dst_key, self._networks[self._cur_task].convnet.state_dict()[dst_key + "weight"][:5]))
            logging.info("{}bias after update: {}".format(dst_key, self._networks[self._cur_task].convnet.state_dict()[dst_key + "bias"][:5]))

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

        logging.info("{}running_mean after training: {}".format(dst_key, self._networks[self._cur_task].convnet.state_dict()[dst_key + "running_mean"][:5]))
        logging.info("{}weight after training: {}".format(dst_key, self._networks[self._cur_task].convnet.state_dict()[dst_key + "weight"][:5]))
        logging.info("{}bias after training: {}".format(dst_key, self._networks[self._cur_task].convnet.state_dict()[dst_key + "bias"][:5]))

    def _train(self, model, train_loader, test_loader):
        model.to(self._device)
        
        if self._cur_task == 0:
            if fix_parameter:
                logging.info("parameters need grad")
                for name, param in model.named_parameters():
                    if model.convnet.is_fc(name) or model.convnet.is_bn(name):
                        logging.info(name)
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
        else:
            logging.info("parameters need grad")
            for name, param in model.named_parameters():
                if model.convnet.is_fc(name) or model.convnet.is_bn(name):
                    logging.info(name)
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            if optim_type == "adam":
                optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lrate, weight_decay=weight_decay)
            else:
                optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lrate, weight_decay=weight_decay)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=lrate_decay)
        self._update_representation(model, train_loader, test_loader, optimizer, scheduler)

    def reset_bn(self, model):
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.reset_running_stats()
                m.reset_parameters()

    def _update_representation(self, model, train_loader, test_loader, optimizer, scheduler):
        if self._cur_task == 0:
            epochs_num = epochs_init
        else:
            epochs_num = epochs

        prog_bar = tqdm(range(epochs_num))
        #if temp < 1, it will make the output of softmax sharper
        # temp = 0.1
        for _, epoch in enumerate(prog_bar):
            model.train()
            losses = 0.
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                
                logits = model(inputs)['logits']

                loss = nn.CrossEntropyLoss()(logits/temp, targets - self._known_classes)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                # acc
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq((targets - self._known_classes).expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)
            test_acc = self._compute_accuracy(model, test_loader)
            if epoch == epochs_num - 1:
                self._task_acc.append(round(test_acc, 2))
            
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
    

    def caculate_weighted_average(self, init_class, increment, task_acc):
        weighted_accs = []
        class_each_step = []
        for i in range(len(task_acc)):
            if i == 0:
                class_each_step.append(init_class)
            else:
                class_each_step.append(increment)
        class_each_step = np.array(class_each_step)
        task_acc = np.array(task_acc)

        for i in range(len(task_acc)):
            temp_acc = class_each_step[:i+1] / sum(class_each_step[:i+1])
            weighted_accs.append(round(sum(task_acc[:i+1] * temp_acc), 2))
        
        return weighted_accs