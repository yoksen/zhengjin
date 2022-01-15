import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net import IncrementalNet,Twobn_IncrementalNet
from utils.toolkit import target2onehot, tensor2numpy
from scipy.spatial.distance import cdist
from utils.pgd_attack import create_attack

EPSILON = 1e-8

# ImageNet1000, ResNet18

# epochs = 60
# lrate = 0.1
# milestones = [40]
# lrate_decay = 0.1
# batch_size = 128
# weight_decay = 1e-5
# num_workers = 16


# CIFAR100, resnet18_2bn_cbam
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

# CIFAR100, resnet32_2bn
epochs_init = 70
# lrate_init = 1e-2
lrate_init = 1e-3
milestones_init = [49, 63]
lrate_decay_init = 0.1
weight_decay_init = 1e-5

epochs = 70
# lrate = 1e-2
lrate = 1e-3
milestones = [49, 63]
lrate_decay = 0.1
weight_decay = 1e-5  # illness
optim_type = "adam"
batch_size = 64

num_workers = 4

iterations = 2000

hyperparameters = ["epochs_init", "lrate_init", "milestones_init", "lrate_decay_init","weight_decay_init",\
                   "epochs","lrate", "milestones", "lrate_decay", "weight_decay", "optim_type", "batch_size", "iterations"]


class twobn_cl(BaseLearner):

    def __init__(self, args):
        print('create twobn cl!!')
        super().__init__(args)
        self._network = Twobn_IncrementalNet(args['convnet_type'], False)

        # log hyperparameter
        logging.info(50*"-")
        logging.info("log_hyperparameters")
        logging.info(50*"-")
        for item in hyperparameters:
            logging.info('{}: {}'.format(item, eval(item)))

    def after_task(self):
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
        logging.info('Exemplar size: {}'.format(self.exemplar_size))

        if self._cur_task <= 1:
            self.save_model()

    def save_model(self):
        if len(self._multiple_gpus) > 1:
            torch.save(self._network.module.state_dict(), "./saved_model/icarl_{}.pt".format(self._cur_task))
        else:
            torch.save(self._network.state_dict(), "./saved_model/icarl_{}.pt".format(self._cur_task))

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        # Loader
        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train',
                                                 mode='train', appendent=self._get_memory())
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        # Procedure
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train_adv(self.train_loader, self.test_loader)
        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train_adv(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._old_network is not None:
            self._old_network.to(self._device)

        if self._cur_task == 0:
            if optim_type == "adam":
                optimizer = optim.Adam(self._network.parameters(), lr=lrate_init, weight_decay=weight_decay_init)
            else:
                optimizer = optim.SGD(self._network.parameters(), lr=lrate_init, momentum=0.9, weight_decay=weight_decay_init)  # 1e-3
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones_init, gamma=lrate_decay_init)
        else:
            if optim_type == "adam":
                optimizer = optim.Adam(self._network.parameters(), lr=lrate, weight_decay=weight_decay)
            else:
                optimizer = optim.SGD(self._network.parameters(), lr=lrate, weight_decay=weight_decay)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=lrate_decay)
        self._update_representation_adv(train_loader, test_loader, optimizer, scheduler)

    def _update_representation_adv(self, train_loader, test_loader, optimizer, scheduler):
        if self._cur_task == 0:
            epochs_num = epochs_init
        else:
            epochs_num = epochs
        prog_bar = tqdm(range(epochs_num))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                # [N,C,H,W]
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                bs = inputs.shape[0]
                ori_targets = targets

                # [2N,class_num]
                ret_dict , targets = self._network(inputs, targets)  # here!

                # [2N,class_num]
                logits = ret_dict['logits']
                onehots = target2onehot(targets, self._total_classes)

                if self._old_network is None:
                    loss = F.binary_cross_entropy_with_logits(logits, onehots)
                else:
                    old_ret_dict ,ori_targets = self._old_network(inputs , ori_targets)
                    old_logits  = old_ret_dict['logits'].detach()
                    old_onehots = torch.sigmoid(old_logits)
                    new_onehots = onehots.clone()
                    new_onehots[:bs, :self._known_classes] = old_onehots
                    new_onehots[bs:, :self._known_classes] = old_onehots
                    loss = F.binary_cross_entropy_with_logits(logits, new_onehots)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                # acc
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            test_acc = self._compute_accuracy(self._network, test_loader)
            info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                self._cur_task, epoch + 1, epochs_num, losses / len(train_loader), train_acc, test_acc)
            prog_bar.set_description(info)

        logging.info(info)

    # Polymorphism
    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            targets = targets.to(self._device)

            with torch.no_grad():
                ret_dict , targets = model(inputs , targets)
                outputs = ret_dict['logits']
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets.cpu()).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct)*100 / total, decimals=2)

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            targets = targets.to(self._device)

            with torch.no_grad():
                ret_dict, targets = self._network(inputs, targets)
                outputs = ret_dict['logits']
            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

    def _eval_nme(self, loader, class_means):
        self._network.eval()
        vectors, y_true = self._extract_vectors(loader)
        vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T

        dists = cdist(class_means, vectors, 'sqeuclidean')  # [nb_classes, N]
        scores = dists.T  # [N, nb_classes], choose the one with the smallest distance

        return np.argsort(scores, axis=1)[:, :self.topk], y_true  # [N, topk]

    def _extract_vectors(self, loader):
        self._network.eval()
        vectors, targets = [], []
        for _, _inputs, _targets in loader:
            tmp = _targets.clone()
            _targets = _targets.numpy()

            if isinstance(self._network, nn.DataParallel):
                _vectors = tensor2numpy(self._network.module.extract_vector(_inputs.to(self._device)))
            else:
                # _vectors = tensor2numpy(self._network.extract_vector(_inputs.to(self._device)))
                ret_dict , _ = self._network(_inputs.to(self._device), tmp.to(self._device))
                _vectors = ret_dict['features']
                _vectors = tensor2numpy(_vectors)


            vectors.append(_vectors)
            targets.append(_targets)

        return np.concatenate(vectors), np.concatenate(targets)