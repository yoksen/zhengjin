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


# CIFAR100, ResNet32
epochs = 100
lrate = 0.1
milestones = [40,60,90]
lrate_decay = 0.1
batch_size = 128
# weight_decay = 1e-5
weight_decay = 5e-4
num_workers = 4




# CIFAR100, ResNet32
# epochs = 70
# lrate = 0.1
# milestones = [30,50,60]
# lrate_decay = 0.1
# batch_size = 128
# # weight_decay = 1e-5
# weight_decay = 5e-4
# num_workers = 4


class twobn_cl(BaseLearner):

    def __init__(self, args):
        print('create twobn cl!!')
        super().__init__(args)
        self._network = Twobn_IncrementalNet(args['convnet_type'], False)

    def after_task(self):
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
        logging.info('Exemplar size: {}'.format(self.exemplar_size))

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
        # self._train(self.train_loader, self.test_loader)
        self._train_adv(self.train_loader, self.test_loader)
        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._old_network is not None:
            self._old_network.to(self._device)
        optimizer = optim.SGD(self._network.parameters(), lr=lrate, momentum=0.9, weight_decay=weight_decay)  # 1e-5
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=lrate_decay)
        self._update_representation(train_loader, test_loader, optimizer, scheduler)

    def _train_adv(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._old_network is not None:
            self._old_network.to(self._device)
        optimizer = optim.SGD(self._network.parameters(), lr=lrate, momentum=0.9, weight_decay=weight_decay)  # 1e-5
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=lrate_decay)

        self._update_representation_adv(train_loader, test_loader, optimizer, scheduler)

    def _update_representation_adv(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(epochs))
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
                ret_dict , targets = self._network( inputs , targets )  # here!

                # [2N,class_num]
                logits = ret_dict['logits']
                onehots = target2onehot(targets, self._total_classes)

                if self._old_network is None:
                    loss = F.binary_cross_entropy_with_logits(logits, onehots)
                else:
                    # old_onehots = torch.sigmoid(self._old_network(inputs)['logits'].detach())

                    old_ret_dict ,ori_targets = self._old_network( inputs , ori_targets )
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
            # train_acc = self._compute_accuracy(self._network, train_loader)
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            test_acc = self._compute_accuracy(self._network, test_loader)
            info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                self._cur_task, epoch + 1, epochs, losses / len(train_loader), train_acc, test_acc)
            prog_bar.set_description(info)

        logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(epochs))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)['logits']
                onehots = target2onehot(targets, self._total_classes)

                if self._old_network is None:
                    loss = F.binary_cross_entropy_with_logits(logits, onehots)
                else:
                    old_onehots = torch.sigmoid(self._old_network(inputs)['logits'].detach())
                    new_onehots = onehots.clone()
                    new_onehots[:, :self._known_classes] = old_onehots
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
            # train_acc = self._compute_accuracy(self._network, train_loader)
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            test_acc = self._compute_accuracy(self._network, test_loader)
            info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                self._cur_task, epoch + 1, epochs, losses / len(train_loader), train_acc, test_acc)
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


    # # to be finish!
    # def _construct_exemplar_unified(self, data_manager, m):
    #     logging.info('Constructing exemplars for new classes...({} per classes)'.format(m))
    #     _class_means = np.zeros((self._total_classes, self.feature_dim))
    #
    #     # Calculate the means of old classes with newly trained network
    #     for class_idx in range(self._known_classes):
    #         mask = np.where(self._targets_memory == class_idx)[0]
    #         class_data, class_targets = self._data_memory[mask], self._targets_memory[mask]
    #
    #         class_dset = data_manager.get_dataset([], source='train', mode='test',
    #                                               appendent=(class_data, class_targets))
    #         class_loader = DataLoader(class_dset, batch_size=batch_size, shuffle=False, num_workers=4)
    #         vectors, _ = self._extract_vectors(class_loader)
    #         vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
    #         mean = np.mean(vectors, axis=0)
    #         mean = mean / np.linalg.norm(mean)
    #
    #         _class_means[class_idx, :] = mean
    #
    #     # Construct exemplars for new classes and calculate the means
    #     for class_idx in range(self._known_classes, self._total_classes):
    #         data, targets, class_dset = data_manager.get_dataset(np.arange(class_idx, class_idx+1), source='train',
    #                                                              mode='test', ret_data=True)
    #         class_loader = DataLoader(class_dset, batch_size=batch_size, shuffle=False, num_workers=4)
    #
    #         vectors, _ = self._extract_vectors(class_loader)
    #         vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
    #         class_mean = np.mean(vectors, axis=0)
    #
    #         # Select
    #         selected_exemplars = []
    #         exemplar_vectors = []
    #         for k in range(1, m+1):
    #             S = np.sum(exemplar_vectors, axis=0)  # [feature_dim] sum of selected exemplars vectors
    #             mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
    #             i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))
    #
    #             selected_exemplars.append(np.array(data[i]))  # New object to avoid passing by inference
    #             exemplar_vectors.append(np.array(vectors[i]))  # New object to avoid passing by inference
    #
    #             vectors = np.delete(vectors, i, axis=0)  # Remove it to avoid duplicative selection
    #             data = np.delete(data, i, axis=0)  # Remove it to avoid duplicative selection
    #
    #         selected_exemplars = np.array(selected_exemplars)
    #         exemplar_targets = np.full(m, class_idx)
    #         self._data_memory = np.concatenate((self._data_memory, selected_exemplars)) if len(self._data_memory) != 0 \
    #             else selected_exemplars
    #         self._targets_memory = np.concatenate((self._targets_memory, exemplar_targets)) if \
    #             len(self._targets_memory) != 0 else exemplar_targets
    #
    #         # Exemplar mean
    #         exemplar_dset = data_manager.get_dataset([], source='train', mode='test',
    #                                                  appendent=(selected_exemplars, exemplar_targets))
    #         exemplar_loader = DataLoader(exemplar_dset, batch_size=batch_size, shuffle=False, num_workers=4)
    #         vectors, _ = self._extract_vectors(exemplar_loader)
    #         vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
    #         mean = np.mean(vectors, axis=0)
    #         mean = mean / np.linalg.norm(mean)
    #
    #         _class_means[class_idx, :] = mean
    #
    #     self._class_means = _class_means

