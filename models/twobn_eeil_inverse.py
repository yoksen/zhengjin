import logging
import numpy as np
import torch
import errno
import os
import time
import random
from torchvision import transforms
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import IncrementalNet,Twobn_IncrementalNet
from models.base import BaseLearner
from scipy.spatial.distance import cdist
from utils.toolkit import tensor2numpy

EPSILON = 1e-8

# ImageNet1000, ResNet18


# CIFAR100, ResNet32

vector_num_per_class = 480

epochs_init = 90
# epochs_init = 1
lrate_init = 0.1
milestones_init = [50, 60]


# epochs = 60
# # epochs = 1
# lrate = 0.1
# milestones = [50, 60]


epochs = 90
# epochs = 1
lrate = 0.1
milestones = [50, 60]


epochs_finetune = 40
# epochs_finetune = 1
lrate_finetune = 0.01
milestones_finetune = [10, 20]

lrate_decay = 0.1
batch_size = 128
T = 2
weight_decay = 1e-3
num_workers = 4




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




class twobn_eeil_inverse(BaseLearner):

    def __init__(self, args):
        print('create twobn cl!!')
        super().__init__(args)
        self._inverse_data_memory, self._inverse_targets_memory = np.array([]), np.array([])

        self._network = Twobn_IncrementalNet(args['convnet_type'], False)
        self._seen_classes = []

    def after_task(self):
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
        logging.info('Exemplar size: {}'.format(self.exemplar_size))

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self.task_size = data_manager.get_task_size(self._cur_task)
        self._total_classes = self._known_classes + self.task_size
        self._network.update_fc(self._total_classes)
        self._seen_classes.append(self.task_size)
        logging.info('vector_num_per_class  {}'.format(vector_num_per_class))
        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        # Loader
        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train',
                                                 mode='train', appendent=self._get_both_memory())
        print('train_dataset  size ', len(train_dataset))

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        # Procedure
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(data_manager, self.train_loader, self.test_loader)
        if self._total_classes != sum(data_manager._increments):
            self.build_rehearsal_memory(data_manager, self.samples_per_class)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, data_manager, train_loader, test_loader):
        self._network.to(self._device)
        if self._old_network is not None:
            self._old_network.to(self._device)
        if self._cur_task == 0:
            optimizer = optim.SGD(self._network.parameters(), lr=lrate_init, momentum=0.9, weight_decay=weight_decay)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones_init,
                                                       gamma=lrate_decay)
            self._is_finetuning = False
            self._run(self.train_loader, self.test_loader, epochs_init, optimizer, scheduler, 'Training')
            return

        # New + exemplars
        optimizer = optim.SGD(self._network.parameters(), lr=lrate, momentum=0.9, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=lrate_decay)
        self._is_finetuning = False
        self._run(self.train_loader, self.test_loader, epochs, optimizer, scheduler, 'Training')

        # Finetune
        if self._fixed_memory:
            finetune_samples_per_class = self._memory_per_class
            self._construct_exemplar_for_finetune(data_manager, finetune_samples_per_class)
        else:
            finetune_samples_per_class = self._memory_size//self._known_classes
            self._reduce_exemplar(data_manager, finetune_samples_per_class)
            self._construct_exemplar(data_manager, finetune_samples_per_class)

        if len(self._multiple_gpus) > 1:
            self._old_network = self._network.module.copy().freeze()
        else:
            self._old_network = self._network.copy().freeze()
        finetune_train_dataset = data_manager.get_dataset([], source='train', mode='train',
                                                          appendent=self._get_memory())
        finetune_train_loader = DataLoader(finetune_train_dataset, batch_size=batch_size,
                                           shuffle=True, num_workers=num_workers)
        optimizer = optim.SGD(self._network.parameters(), lr=lrate_finetune, momentum=0.9, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones_finetune,
                                                   gamma=lrate_decay)
        self._is_finetuning = True
        self._run(finetune_train_loader, self.test_loader, epochs_finetune, optimizer, scheduler, 'Finetuning')

        # Remove the temporary exemplars of new classes
        if self._fixed_memory:
            self._data_memory = self._data_memory[:-self._memory_per_class*self.task_size]
            self._targets_memory = self._targets_memory[:-self._memory_per_class*self.task_size]
            # Check
            assert len(np.setdiff1d(self._targets_memory, np.arange(0, self._known_classes))) == 0, 'Exemplar error!'

    def _run(self, train_loader, test_loader, epochs_, optimizer, scheduler, process):
        prog_bar = tqdm(range(epochs_))
        for _, epoch in enumerate(prog_bar, start=1):
            self._network.train()
            losses = 0.
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                # inputs, targets = inputs.to(self._device), targets.to(self._device)
                # logits = self._network(inputs)['logits']

                # [N,C,H,W]
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                bs = inputs.shape[0]
                ori_targets = targets

                # [2N,class_num]
                ret_dict , targets = self._network( inputs , targets )  # here!

                # [2N,class_num]
                logits = ret_dict['logits']

                # CELoss
                clf_loss = F.cross_entropy(logits, targets, ignore_index=-1)

                if self._cur_task == 0:
                    distill_loss = torch.zeros(1, device=self._device)
                else:
                    finetuning_task = (self._cur_task + 1) if self._is_finetuning else self._cur_task
                    distill_loss = 0.
                    # old_logits = self._old_network(inputs)['logits']

                    old_ret_dict ,ori_targets = self._old_network( inputs , ori_targets )
                    old_logits  = old_ret_dict['logits'].detach()

                    # print('logits shape',logits.shape)
                    # print('old logits shape', old_logits.shape)
                    # assert 1==0

                    for i in range(1, finetuning_task+1):
                        lo = sum(self._seen_classes[:i-1])
                        hi = sum(self._seen_classes[:i])
                        distill_loss += F.binary_cross_entropy(
                            F.softmax(logits[:int(bs), lo:hi] / T, dim=1),
                            F.softmax(old_logits[:, lo:hi] / T, dim=1)
                        )

                loss = clf_loss + distill_loss
                losses += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # acc
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            # train_acc = self._compute_accuracy(self._network, train_loader)
            train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)
            test_acc = self._compute_accuracy(self._network, test_loader)
            info1 = '{} => '.format(process)
            info2 = 'Task {}, Epoch {}/{}, Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                self._cur_task, epoch+1, epochs_, losses/len(train_loader), train_acc, test_acc)
            prog_bar.set_description(info1 + info2)

        logging.info(info1 + info2)

    # Polymorphism

    def _get_both_memory(self):
        if len(self._data_memory) == 0:
            return None
        else:
            _data_ = np.concatenate( (self._data_memory, self._inverse_data_memory) )
            _targets_ = np.concatenate( (self._targets_memory , self._inverse_targets_memory) )

            return (_data_ , _targets_)

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

    def _construct_exemplar_unified(self, data_manager, m):
        logging.info('Constructing exemplars for new classes...({} per classes)'.format(m))
        _class_means = np.zeros((self._total_classes, self.feature_dim))

        # Calculate the means of old classes with newly trained network
        for class_idx in range(self._known_classes):
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


        # update inverse data memory
        for class_idx in range(self._known_classes):

            # update old classes images
            # mask = np.where(self._targets_memory == class_idx)[0]
            mask = np.arange(class_idx*vector_num_per_class , (class_idx+1)*vector_num_per_class)
            class_data, class_targets = self._inverse_data_memory[mask], self._inverse_targets_memory[mask]

            class_dset = data_manager.get_dataset([], source='train', mode='test',
                                                  appendent=(class_data, class_targets))
            class_loader = DataLoader(class_dset, batch_size=class_data.shape[0], shuffle=False, num_workers=4)

            # inverse_images_path
            inverse_old_class_images_ = []
            for _ , images , _ in class_loader:
                inverse_images_batch = self.rebuild_image_fv_bn(images.to(self._device), self._network.convnet, randstart=True)

                if class_dset.use_path:
                    save_path_list = save_imgs(batch_img=inverse_images_batch, task_id=self._cur_task, class_id=class_idx)
                    inverse_old_class_images_.extend(save_path_list)
                else:
                    inverse_images_batch = inverse_images_batch.detach().cpu().numpy().transpose(0,2,3,1)
                    inverse_images_batch = (inverse_images_batch*255).astype(np.uint8)
                    inverse_old_class_images_.extend(inverse_images_batch)

            inverse_old_class_images_ = np.array( inverse_old_class_images_ )
            self._inverse_data_memory[mask] = inverse_old_class_images_


        # Construct exemplars for new classes and calculate the means
        for class_idx in range(self._known_classes, self._total_classes):

            data, targets, class_dset = data_manager.get_dataset(np.arange(class_idx, class_idx+1), source='train',
                                                                 mode='test', ret_data=True)

            class_loader = DataLoader(class_dset, batch_size=batch_size, shuffle=False, num_workers=4)

            vectors, _ = self._extract_vectors(class_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            class_mean = np.mean(vectors, axis=0)
            data_num_per_class = data.shape[0]

            # print('data  type' , type(data) )
            # print('data  shape', data.shape)
            #
            # print('data 0 type ' , type(data[0]))
            # print('data 0 shape ' , data[0].shape)
            # print('data 0 ' , data[0])
            #
            # assert  False

            # Select
            selected_exemplars = []
            selected_exemplars_to_be_inv = []
            exemplar_vectors = []

            for k in range(1, m + vector_num_per_class + 1):
                S = np.sum(exemplar_vectors, axis=0)  # [feature_dim] sum of selected exemplars vectors
                mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))

                if k <= m:
                    selected_exemplars.append(np.array(data[i]))  # New object to avoid passing by inference
                    exemplar_vectors.append(np.array(vectors[i]))  # New object to avoid passing by inference
                else:
                    selected_exemplars_to_be_inv.append(np.array(data[i]))  # New object to avoid passing by inference
                    exemplar_vectors.append(np.array(vectors[i]))  # New object to avoid passing by inference

                vectors = np.delete(vectors, i, axis=0)  # Remove it to avoid duplicative selection
                data = np.delete(data, i, axis=0)  # Remove it to avoid duplicative selection

            selected_exemplars = np.array(selected_exemplars)
            exemplar_targets = np.full(m, class_idx)

            # convert rest data label to -1
            selected_exemplars_to_be_inv = np.array(selected_exemplars_to_be_inv)
            exemplars_to_be_inv_targets = np.full(vector_num_per_class ,-1)

            # get inverse image from the selected_exemplars_to_be_inv
            exemplar_dset = data_manager.get_dataset([], source='train', mode='test',
                                                     appendent=(selected_exemplars_to_be_inv,exemplars_to_be_inv_targets))
            exemplar_loader = DataLoader(exemplar_dset, batch_size=selected_exemplars_to_be_inv.shape[0] , shuffle=False, num_workers=4)

            inverse_images_ = []
            for _ , images , _ in exemplar_loader:
                inverse_images_batch = self.rebuild_image_fv_bn(images.to(self._device), self._network.convnet, randstart=True)

                if class_dset.use_path:
                    save_path_list = save_imgs(batch_img=inverse_images_batch, task_id=self._cur_task,class_id=class_idx)
                    inverse_images_.extend(save_path_list)
                else:
                    inverse_images_batch = inverse_images_batch.detach().cpu().numpy().transpose(0,2,3,1)
                    inverse_images_batch = (inverse_images_batch*255).astype(np.uint8)

                    inverse_images_.extend( inverse_images_batch)

            inverse_images_ = np.array( inverse_images_ )



            # Exemplar mean
            exemplar_dset = data_manager.get_dataset([], source='train', mode='test',
                                                     appendent=(selected_exemplars, exemplar_targets))
            exemplar_loader = DataLoader(exemplar_dset, batch_size=batch_size, shuffle=False, num_workers=4)
            vectors, _ = self._extract_vectors(exemplar_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            _class_means[class_idx, :] = mean


            # add to memory
            self._data_memory = np.concatenate((self._data_memory, selected_exemplars)) if len(self._data_memory) != 0 \
                else selected_exemplars
            self._targets_memory = np.concatenate((self._targets_memory, exemplar_targets)) if \
                len(self._targets_memory) != 0 else exemplar_targets

            self._inverse_data_memory = np.concatenate((self._inverse_data_memory, inverse_images_)) if len(self._inverse_data_memory) != 0 \
                else inverse_images_
            self._inverse_targets_memory = np.concatenate((self._inverse_targets_memory, exemplars_to_be_inv_targets)) if \
                len(self._inverse_targets_memory) != 0 else exemplars_to_be_inv_targets

        logging.info(f'Constructing exemplars finish! data_memory size {self._data_memory.shape[0]},inverse_data_memory size {self._inverse_data_memory.shape[0]}')
        self._class_means = _class_means

    def rebuild_image_fv_bn(self,image, model, randstart=True):

        model.eval()
        model.set_hook()
        normalize = transforms.Normalize(mean=(.485, .456, .406),
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
        iterations_per_layer = 600

        # lr = 0.01
        lr = 0.01
        # lr_scheduler = lr_cosine_policy(lr, 100, iterations_per_layer)
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
        for i in range(iterations_per_layer):
            # learning rate scheduling
            # lr_scheduler(optimizer, i, i)

            # roll
            off1 = random.randint(-lim_0, lim_0)
            off2 = random.randint(-lim_1, lim_1)
            inputs_jit = torch.roll(rand_x, shifts=(off1, off2), dims=(2, 3))

            # do not roll
            # inputs_jit = rand_x

            # R_prior losses
            loss_var_l1, loss_var_l2 = get_image_prior_losses(inputs_jit)

            # l2 loss on images
            loss_l2 = torch.norm(inputs_jit.view(inputs_jit.shape[0], -1), dim=1).mean()

            # main loss
            main_loss = criterion(inputs_jit, torch.tensor([0]))

            # bn loss
            if i <= 200:
                r_feature = 1e-3
            elif i <= 400:
                r_feature = 1e-2
            elif i <= 600:
                r_feature = 5e-2

            rescale = [first_bn_multiplier] + [1. for _ in range(len(model.loss_r_feature_layers) - 1)]
            loss_r_feature = sum(
                [rescale[idx] * item.r_feature for idx, item in enumerate(model.loss_r_feature_layers)])
            loss = main_loss + r_feature * loss_r_feature + var_scale_l2 * loss_var_l2 + var_scale_l1 * loss_var_l1 + l2_scale * loss_l2

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            rand_x.data = torch.clamp(rand_x.data, 0, 1)

            if (i+1) % 100 == 0 :
                print(i, 'rand_strat ', randstart)
                print(
                    f'loss {loss:.3f} , fv_loss {main_loss:.3f} , loss_r_fea {loss_r_feature:.3f} , loss_l2 {loss_l2:.3f} , loss_var_l2 {loss_var_l2:.3f}')

            best_img = rand_x.clone().detach()
            # if loss_max > loss.item():
            #     best_img = rand_x.clone().detach()
            #     loss_max = loss.item()

        print("inverse --- %s seconds ---" % (time.time() - start_time))
        model.remove_hook()
        return best_img

    def _construct_exemplar_for_finetune(self, data_manager, m):
        logging.info('Constructing exemplars for new classes...({} per classes)'.format(m))
        _class_means = np.zeros((self._total_classes, self.feature_dim))

        # Calculate the means of old classes with newly trained network
        for class_idx in range(self._known_classes):
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

            vectors, _ = self._extract_vectors(class_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            class_mean = np.mean(vectors, axis=0)

            # Select
            selected_exemplars = []
            exemplar_vectors = []
            for k in range(1, m+1):
                S = np.sum(exemplar_vectors, axis=0)  # [feature_dim] sum of selected exemplars vectors
                mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))

                selected_exemplars.append(np.array(data[i]))  # New object to avoid passing by inference
                exemplar_vectors.append(np.array(vectors[i]))  # New object to avoid passing by inference

                vectors = np.delete(vectors, i, axis=0)  # Remove it to avoid duplicative selection
                data = np.delete(data, i, axis=0)  # Remove it to avoid duplicative selection

            selected_exemplars = np.array(selected_exemplars)
            exemplar_targets = np.full(m, class_idx)
            self._data_memory = np.concatenate((self._data_memory, selected_exemplars)) if len(self._data_memory) != 0 \
                else selected_exemplars
            self._targets_memory = np.concatenate((self._targets_memory, exemplar_targets)) if \
                len(self._targets_memory) != 0 else exemplar_targets

            # Exemplar mean
            exemplar_dset = data_manager.get_dataset([], source='train', mode='test',
                                                     appendent=(selected_exemplars, exemplar_targets))
            exemplar_loader = DataLoader(exemplar_dset, batch_size=batch_size, shuffle=False, num_workers=4)
            vectors, _ = self._extract_vectors(exemplar_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            _class_means[class_idx, :] = mean

        self._class_means = _class_means