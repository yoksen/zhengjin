import logging
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net import IncrementalNetWithBias,Twobn_IncrementalNetWithBias
from utils.toolkit import target2onehot, tensor2numpy
from scipy.spatial.distance import cdist
EPSILON = 1e-8

# ImageNet1000, ResNet18
'''
epochs = 100
lrate = 0.1
milestones = [30, 60, 80, 90]
lrate_decay = 0.1
batch_size = 256
split_ratio = 0.1
T = 2
weight_decay = 1e-4
num_workers = 16
'''

# CIFAR100, ResNet32, 10 base

# train_dataset has changed
epochs = 250
# epochs = 1
lrate = 0.1
milestones = [100, 150, 200]
lrate_decay = 0.1
batch_size = 128
# split_ratio = 0.1
split_ratio = 0.2
T = 2
weight_decay = 2e-4
num_workers = 4


class twobn_bic(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = Twobn_IncrementalNetWithBias(args['convnet_type'], False, bias_correction=True)
        self._class_means = None

    def after_task(self):
        # self.save_checkpoint(logfilename)
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
        logging.info('Exemplar size: {}'.format(self.exemplar_size))

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        # Loader
        if self._cur_task >= 1:

            print(' split ', int(split_ratio * self._memory_size / self._known_classes))
            # assert 1==0

            # train_dset, val_dset = data_manager.get_dataset_with_split(np.arange(self._known_classes,
            #                                                                      self._total_classes),
            #                                                            source='train', mode='train',
            #                                                            appendent=self._get_memory(),
            #                                                            val_samples_per_class=int(
            #                                                             split_ratio *
            #                                                                self._memory_size/self._known_classes))

            train_dset, val_dset = data_manager.get_dataset_with_split(np.arange(self._known_classes,
                                                                                 self._total_classes),
                                                                       source='train', mode='train',
                                                                       appendent=self._get_memory(),
                                                                       val_samples_per_class=int(
                                                                           split_ratio *
                                                                           self._memory_per_class))
            self.val_loader = DataLoader(val_dset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
            logging.info('Stage1 dset: {}, Stage2 dset: {}'.format(len(train_dset), len(val_dset)))
            self.lamda = self._known_classes / self._total_classes
            logging.info('Lambda: {:.3f}'.format(self.lamda))
        else:
            train_dset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),
                                                  source='train', mode='train',
                                                  appendent=self._get_memory())
        test_dset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')

        self.train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.test_loader = DataLoader(test_dset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        # Procedure
        self._log_bias_params()
        self._stage1_training(self.train_loader, self.test_loader)
        if self._cur_task >= 1:
            self._stage2_bias_correction(self.val_loader, self.test_loader)

        # Exemplars
        self.build_rehearsal_memory(data_manager, self.samples_per_class)

        # Extract from DataParallel
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
        self._log_bias_params()

    def _run(self, train_loader, test_loader, optimizer, scheduler, stage):
        for epoch in range(1, epochs + 1):
            self._network.train()
            losses = 0.
            for i, (_, inputs, targets) in enumerate(train_loader):

                # [N,C,H,W]
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                bs = inputs.shape[0]
                ori_targets = targets

                # [2N,class_num]
                ret_dict , targets = self._network( inputs , targets )  # here!

                # [2N,class_num]
                logits = ret_dict['logits']


                if stage == 'training':
                    clf_loss = F.cross_entropy(logits, targets)
                    if self._old_network is not None:

                        old_ret_dict, ori_targets = self._old_network(inputs, ori_targets)
                        old_logits = old_ret_dict['logits'].detach()

                        hat_pai_k = F.softmax(old_logits / T, dim=1)
                        log_pai_k = F.log_softmax(logits[:bs, :self._known_classes] / T, dim=1)
                        distill_loss = -torch.mean(torch.sum(hat_pai_k * log_pai_k, dim=1))
                        loss = distill_loss * self.lamda + clf_loss * (1 - self.lamda)
                    else:
                        loss = clf_loss
                elif stage == 'bias_correction':
                    loss = F.cross_entropy(torch.softmax(logits, dim=1), targets)
                else:
                    raise NotImplementedError()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

            scheduler.step()
            train_acc = self._compute_accuracy(self._network, train_loader)
            test_acc = self._compute_accuracy(self._network, test_loader)
            info = '{} => Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.3f}, Test_accy {:.3f}'.format(
                stage, self._cur_task, epoch, epochs, losses / len(train_loader), train_acc, test_acc)
            logging.info(info)

    def _stage1_training(self, train_loader, test_loader):
        '''
        if self._cur_task == 0:
            loaded_dict = torch.load('./dict_0.pkl')
            self._network.load_state_dict(loaded_dict['model_state_dict'])
            self._network.to(self._device)
            return
        '''

        # Freeze bias layer and train stage1 layer
        ignored_params = list(map(id, self._network.bias_layers.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params, self._network.parameters())
        network_params = [{'params': base_params, 'lr': lrate, 'weight_decay': weight_decay},
                          {'params': self._network.bias_layers.parameters(), 'lr': 0, 'weight_decay': 0}]
        optimizer = optim.SGD(network_params, lr=lrate, momentum=0.9, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=lrate_decay)

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._network.to(self._device)
        if self._old_network is not None:
            self._old_network.to(self._device)

        self._run(train_loader, test_loader, optimizer, scheduler, stage='training')

    def _stage2_bias_correction(self, val_loader, test_loader):
        if isinstance(self._network, nn.DataParallel):
            self._network = self._network.module
        # Freeze stage1 layer and train bias layer
        network_params = [{'params': self._network.bias_layers[-1].parameters(), 'lr': lrate,
                           'weight_decay': weight_decay}]
        optimizer = optim.SGD(network_params, lr=lrate, momentum=0.9, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=lrate_decay)

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._network.to(self._device)

        self._run(val_loader, test_loader, optimizer, scheduler, stage='bias_correction')

    def _log_bias_params(self):
        logging.info('Parameters of bias layer:')
        params = self._network.get_bias_params()
        for i, param in enumerate(params):
            logging.info('{} => {:.3f}, {:.3f}'.format(i, param[0], param[1]))

    # Polymorphism blow

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