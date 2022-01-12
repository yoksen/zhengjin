import logging
import numpy as np
from tqdm import tqdm
import torch
import random
import os
import time
import errno
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import  transforms as T
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
epochs_init = 160
lrate_init = 1.0
milestones_init = [100, 150, 200]
lrate_decay_init = 0.1
weight_decay_init = 1e-4

epochs = 160
lrate = 1.0
milestones = [100 , 150 , 200]
lrate_decay = 0.1
weight_decay = 1e-4
batch_size = 128
num_workers = 4
duplex = True

iterations = 2000

hyperparameters = ["epochs_init", "lrate_init", "milestones_init", "lrate_decay_init","weight_decay_init",\
                   "epochs","lrate", "milestones", "lrate_decay", "weight_decay","batch_size", "num_workers",\
                   "duplex", "iterations"]


def get_image_prior_losses(inputs_jit):
    # COMPUTE total variation regularization loss
    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]

    loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
    loss_var_l1 = (diff1.abs() / 255.0).mean() + (diff2.abs() / 255.0).mean() + (
            diff3.abs() / 255.0).mean() + (diff4.abs() / 255.0).mean()
    loss_var_l1 = loss_var_l1 * 255.0
    return loss_var_l1, loss_var_l2

def save_imgs(batch_img, task_id ,class_id):

    toPIL = T.ToPILImage()
    bs = batch_img.shape[0]
    save_path_list =[]

    for i in range(bs):
        img = toPIL(batch_img[i].detach().cpu())
        img_dir = f'./data/task_{task_id}/{class_id}/'

        if not os.path.exists(img_dir):
            try:
                os.makedirs(img_dir)
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise
                pass
        file_name = f'{i}.png'
        save_path = os.path.join(img_dir, file_name)
        img.save(save_path)
        save_path_list.append(save_path)

    return save_path_list

class icarl_regularization_v2(BaseLearner):
    def __init__(self, args):
        print('create icarl_regularization!!')
        super().__init__(args)
        self._network = Twobn_IncrementalNet(args['convnet_type'], False)
        self._data_train_inverse, self._targets_train_inverse = np.array([]), np.array([])

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
            logging.info('Save model: {}'.format(self._cur_task))
            self.save_model()

    def save_model(self):
        if len(self._multiple_gpus) > 1:
            torch.save(self._network.module.state_dict(), "./saved_model/icarl_regularization_v2_duplex_{}_{}.pt".format(duplex, self._cur_task))
        else:
            torch.save(self._network.state_dict(), "./saved_model/icarl_regularization_v2_duplex_{}_{}.pt".format(duplex, self._cur_task))

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)

        self._network.update_fc(self._total_classes)
        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        # Loader
        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train',
                                                 mode='train')
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        # Procedure
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train_adv(self.train_loader, self.test_loader)
        
        if duplex:
            self._get_train_inverse_data(data_manager=data_manager)
            train_inverse_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train',
                                                    mode='train', appendent=self._get_train_inverse_memory())
            self.train_inverse_loader = DataLoader(train_inverse_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
            self._train_adv(self.train_inverse_loader, self.test_loader)

        #update memory
        if self._total_classes != sum(data_manager._increments):
            self.build_rehearsal_memory(data_manager, self.samples_per_class)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train_adv(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._old_network is not None:
            self._old_network.to(self._device)

        if self._cur_task == 0:
            optimizer = optim.SGD(self._network.parameters(), lr=lrate_init, momentum=0.9, weight_decay=weight_decay_init)  # 1e-4
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones_init, gamma=lrate_decay_init)
        else:
            optimizer = optim.SGD(self._network.parameters(), lr=lrate, momentum=0.9, weight_decay=weight_decay)  # 1e-5
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
        #my add
        # self.topk = 1

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

    def _construct_exemplar_unified(self, data_manager, m):
        logging.info('Constructing exemplars for new classes...({} per classes)'.format(m))
        _class_means = np.zeros((self._total_classes, self.feature_dim))

        # Calculate the means of old classes with newly trained network
        for class_idx in range(self._known_classes):

            # update old classes images
            mask = np.where(self._targets_memory == class_idx)[0]
            class_data, class_targets = self._data_memory[mask], self._targets_memory[mask]

            class_dset = data_manager.get_dataset([], source='train', mode='test',
                                                  appendent=(class_data, class_targets))
            class_loader = DataLoader(class_dset, batch_size=batch_size, shuffle=False, num_workers=4)

            # inverse_images_path
            inverse_old_class_images_ = []
            for _ , images , _ in class_loader:
                #to be updated
                inverse_images_batch = self.rebuild_image_fv_bn(images.to(self._device), self._network.convnet, randstart=True)
                inverse_images_batch = inverse_images_batch.detach().cpu().numpy().transpose(0,2,3,1)
                inverse_images_batch = (inverse_images_batch*255).astype(np.uint8)
                inverse_old_class_images_.extend(inverse_images_batch)

            inverse_old_class_images_ = np.array(inverse_old_class_images_)
            self._data_memory[mask] = inverse_old_class_images_


            # update old classes mean
            mask = np.where(self._targets_memory == class_idx)[0]
            class_data, class_targets = self._data_memory[mask], self._targets_memory[mask]

            class_dset = data_manager.get_dataset([], source='train', mode='test',
                                                  appendent=(class_data, class_targets))
            class_loader = DataLoader(class_dset, batch_size=batch_size, shuffle=False, num_workers=4)
            vectors, _ = self._extract_vectors(class_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            _class_means[class_idx, :] = mean

        # Construct exemplars for new classes and calculate the means
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, class_dset = data_manager.get_dataset(np.arange(class_idx, class_idx+1), source='train',
                                                                 mode='test', ret_data=True)

            class_loader = DataLoader(class_dset, batch_size=batch_size, shuffle=False, num_workers=4)

            # get inverse image
            inverse_images_ = []
            for _ , images , _ in class_loader:
                inverse_images_batch = self.rebuild_image_fv_bn(images.to(self._device), self._network.convnet, randstart=True)
                inverse_images_batch = inverse_images_batch.detach().cpu().numpy().transpose(0,2,3,1)
                inverse_images_batch = (inverse_images_batch*255).astype(np.uint8)
                inverse_images_.extend(inverse_images_batch)

            inverse_images_ = np.array(inverse_images_)
            inverse_targets = np.full(inverse_images_.shape[0], class_idx)

            # Exemplar mean
            exemplar_dset = data_manager.get_dataset([], source='train', mode='test',
                                                     appendent=(inverse_images_, inverse_targets))
            exemplar_loader = DataLoader(exemplar_dset, batch_size=batch_size, shuffle=False, num_workers=4)
            vectors, _ = self._extract_vectors(exemplar_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)
            _class_means[class_idx, :] = mean

            # add to memory
            self._data_memory = np.concatenate((self._data_memory, inverse_images_)) if len(self._data_memory) != 0 \
                else inverse_images_
            self._targets_memory = np.concatenate((self._targets_memory, inverse_targets)) if \
                len(self._targets_memory) != 0 else inverse_targets
        
        self._class_means = _class_means
        logging.info('Finishing constructing inverse exemplars for known classes. The number of exemplars is {}'.format(self._data_memory.shape[0]))

    def _get_train_inverse_data(self, data_manager):
        logging.info('Constructing inverse exemplars for new classes.')
        # get inverse image
        exemplar_dset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train', mode='test',
                                                    appendent=[])
        exemplar_loader = DataLoader(exemplar_dset, batch_size=batch_size, shuffle=False, num_workers=4)

        inverse_images_ = []
        inverse_targets_ = []
        for _ , images, targets in exemplar_loader:
            inverse_images_batch = self.rebuild_image_fv_bn(images.to(self._device), self._network.convnet, randstart=True)
            inverse_images_batch = inverse_images_batch.detach().cpu().numpy().transpose(0,2,3,1)
            inverse_images_batch = (inverse_images_batch*255).astype(np.uint8)
            inverse_images_.extend(inverse_images_batch)
            inverse_targets_.extend(targets)

        inverse_images_ = np.array(inverse_images_)
        inverse_targets_ = np.array(inverse_targets_)

        self._data_train_inverse = inverse_images_
        self._targets_train_inverse = inverse_targets_
        logging.info('Finishing constructing inverse exemplars for new classes. The number of exemplars is {}'.format(self._data_train_inverse.shape[0]))

    def _get_train_inverse_memory(self):
        if len(self._data_train_inverse) == 0:
            return None
        else:
            if len(self._data_memory) == 0:
                logging.info('Return data for consistency regularization. The number of exemplars is {}'.format(self._data_train_inverse.shape[0]))
                return (self._data_train_inverse, self._targets_train_inverse)
            else:
                data_train = np.concatenate((self._data_train_inverse, self._data_memory), axis=0)
                targets_train = np.concatenate((self._targets_train_inverse, self._targets_memory), axis=0)
                logging.info('Return data for consistency regularization. The number of exemplars is {}'.format(data_train.shape[0]))
                return (data_train, targets_train)

    def rebuild_image_fv_bn(self,image, model, randstart=True):
        model.eval()
        model.set_hook()
        normalize = T.Normalize(mean=(.485, .456, .406),
                                std=(.229, .224, .225))

        with torch.no_grad():
            # ori_fv = model.fv( image.to(device) )
            ori_fv = model.fv(image)

        def criterion(x, y):
            rnd_fv = model.fv(normalize(x))
            return torch.div(torch.norm(rnd_fv - ori_fv, dim=1), torch.norm(ori_fv, dim=1)).mean()



        if randstart == True:
            if len(image.shape) == 3:
                rand_x = torch.randn_like(image.unsqueeze(0), requires_grad=True, device=self._device)
            else:
                rand_x = torch.randn_like(image, requires_grad=True, device=self._device)


        start_time = time.time()
        lr = 0.01
        r_feature = 1e-3

        lim_0 = 10
        lim_1 = 10
        var_scale_l2 = 1e-4
        var_scale_l1 = 0.0
        l2_scale = 1e-5
        first_bn_multiplier = 1

        loss_max = 1e4
        best_img = None
        optimizer = optim.Adam([rand_x], lr=lr, betas=[0.5, 0.9], eps=1e-8)
        for i in range(iterations):
            # roll
            off1 = random.randint(-lim_0, lim_0)
            off2 = random.randint(-lim_1, lim_1)
            inputs_jit = torch.roll(rand_x, shifts=(off1, off2), dims=(2, 3))

            # R_prior losses
            loss_var_l1, loss_var_l2 = get_image_prior_losses(inputs_jit)

            # l2 loss on images
            loss_l2 = torch.norm(inputs_jit.view(inputs_jit.shape[0], -1), dim=1).mean()

            # main loss
            main_loss = criterion(inputs_jit, torch.tensor([0]))

            # bn loss
            if iterations == 600:
                if i <= 200:
                    r_feature = 1e-3
                elif i <= 400:
                    r_feature = 1e-2
                elif i <= 600:
                    r_feature = 5e-2
            elif iterations == 2000:
                if i <= 500:
                    r_feature = 1e-3
                elif i <= 1200:
                    r_feature = 5e-3
                elif i <= 2000:
                    r_feature = 1e-2

            rescale = [first_bn_multiplier] + [1. for _ in range(len(model.loss_r_feature_layers) - 1)]
            loss_r_feature = sum(
                [rescale[idx] * item.r_feature for idx, item in enumerate(model.loss_r_feature_layers)])
            loss = main_loss + r_feature * loss_r_feature + var_scale_l2 * loss_var_l2 + var_scale_l1 * loss_var_l1 + l2_scale * loss_l2

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            rand_x.data = torch.clamp(rand_x.data, 0, 1)
            # if (i+1) % 100 == 0 :
            #     print(i, 'rand_strat ', randstart)
            #     print(
            #         f'loss {loss:.3f} , fv_loss {main_loss:.3f} , loss_r_fea {loss_r_feature:.3f} , loss_l2 {loss_l2:.3f} , loss_var_l2 {loss_var_l2:.3f}')

            best_img = rand_x.clone().detach()

        print("inverse --- %s seconds ---" % (time.time() - start_time))
        model.remove_hook()
        return best_img