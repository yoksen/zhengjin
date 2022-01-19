import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net import IncrementalNet
from utils.toolkit import target2onehot, tensor2numpy

EPSILON = 1e-8

# CIFAR100, resnet18_cbam
# epochs_init = 70
# lrate_init = 1e-3
# milestones_init = [49, 63]
# lrate_decay_init = 0.1
# weight_decay_init = 1e-5


# epochs = 70
# lrate = 1e-3
# milestones = [49, 63]
# lrate_decay = 0.1
# weight_decay = 1e-5  # illness
# optim_type = "adam"
# batch_size = 64


# CIFAR100, ResNet32
epochs_init = 70
lrate_init = 1e-2
milestones_init = [49, 63]
lrate_decay_init = 0.1
weight_decay_init = 1e-5


epochs = 70
lrate = 1e-2
milestones = [49, 63]
lrate_decay = 0.1
weight_decay = 1e-5  # illness
optim_type = "adam"
batch_size = 128

num_workers = 4
hyperparameters = ["epochs_init", "lrate_init", "milestones_init", "lrate_decay_init","weight_decay_init", "epochs","lrate", "milestones", "lrate_decay", "weight_decay","batch_size", "num_workers", "optim_type"]



class multi_bn(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args['convnet_type'], False)
        self._network2 = IncrementalNet(args['convnet_type'], False)

        # log hyperparameter
        logging.info(50*"-")
        logging.info("log_hyperparameters")
        logging.info(50*"-")
        for item in hyperparameters:
            logging.info('{}: {}'.format(item, eval(item)))

    def after_task(self):
        self._known_classes = self._total_classes
        if self._cur_task == 0:
            self._network2.state_dict().update(self._network.state_dict())

    def incremental_train(self, data_manager):
        self._cur_task += 1
        if self._cur_task <= 1:
            self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)

            if self._cur_task == 0:
                self._network.update_fc(data_manager.get_task_size(self._cur_task))
            else:
                self._network2.update_fc(data_manager.get_task_size(self._cur_task))
            logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

            # Loader
            train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train',
                                                    mode='train', appendent=self._get_memory())
            self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
            test_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='test', mode='test')
            self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

            # Procedure
            if len(self._multiple_gpus) > 1:
                self._network = nn.DataParallel(self._network, self._multiple_gpus)
            
            if self._cur_task == 0:
                self._train(self._network, self.train_loader, self.test_loader)
            else:
                self._train(self._network2, self.train_loader, self.test_loader)
        else:
            pass

    def _train(self, model, train_loader, test_loader):
        model.to(self._device)
        
        if self._cur_task == 0:
            if optim_type == "adam":
                optimizer = optim.Adam(model.parameters(), lr=lrate_init, weight_decay=weight_decay_init)
            else:
                optimizer = optim.SGD(model.parameters(), lr=lrate_init, momentum=0.9, weight_decay=weight_decay_init)  # 1e-3
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones_init, gamma=lrate_decay_init)
        else:
            for name, param in model.named_parameters():
                if "fc" in name or "bn" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            
            if optim_type == "adam":
                optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lrate, weight_decay=weight_decay)
            else:
                optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lrate, weight_decay=weight_decay)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=lrate_decay)
        self._update_representation(model, train_loader, test_loader, optimizer, scheduler)

    def _update_representation(self, model, train_loader, test_loader, optimizer, scheduler):
        if self._cur_task == 0:
            epochs_num = epochs_init
        else:
            epochs_num = epochs
        prog_bar = tqdm(range(epochs_num))
        for _, epoch in enumerate(prog_bar):
            model.train()
            losses = 0.
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = model(inputs)['logits']
                onehots = target2onehot(targets - self._known_classes, self._total_classes - self._known_classes)

                loss = F.binary_cross_entropy_with_logits(logits, onehots)

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