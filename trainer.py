import sys
import logging
import copy
import torch
import datetime
import os
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters


def train(args):
    seed_list = copy.deepcopy(args['seed'])
    device = copy.deepcopy(args['device'])

    for seed in seed_list:
        args['seed'] = seed
        args['device'] = device
        _train(args)

def _train(args):
    _set_random()
    _set_device(args)
    print_args(args)

    data_manager = DataManager(args['dataset'], args['shuffle'], args['seed'], args['init_cls'], args['increment'])
    model = factory.get_model(args['model_name'], args)
    inverse_nme_accy = None

    cnn_curve, nme_curve, inverse_nme_curve = {'top1': [], 'top5': []}, {'top1': [], 'top5': []}, {'top1': [], 'top5': []}
    for task in range(data_manager.nb_tasks):
        if args['model_name'] != "multi_bn":
            logging.info('All params: {}'.format(count_parameters(model._network)))
            logging.info('Trainable params: {}'.format(count_parameters(model._network, True)))
        
        model.incremental_train(data_manager)

        if args["model_name"] in ["icarl_regularization_v4", "icarl_regularization_v10", "icarl_generator_fixed"] :
            cnn_accy, nme_accy, inverse_nme_accy = model.eval_task()
        elif args["model_name"] == "multi_bn":
            pass
        else:
            cnn_accy, nme_accy = model.eval_task()
        model.after_task()

        if args["model_name"] != "multi_bn":
            if nme_accy is not None and inverse_nme_accy is not None:
                logging.info('CNN: {}'.format(cnn_accy['grouped']))
                logging.info('NME: {}'.format(nme_accy['grouped']))
                logging.info('Inverse_NME: {}'.format(inverse_nme_accy['grouped']))

                cnn_curve['top1'].append(cnn_accy['top1'])
                cnn_curve['top5'].append(cnn_accy['top5'])

                nme_curve['top1'].append(nme_accy['top1'])
                nme_curve['top5'].append(nme_accy['top5'])

                inverse_nme_curve['top1'].append(inverse_nme_accy['top1'])
                inverse_nme_curve['top5'].append(inverse_nme_accy['top5'])

                logging.info('CNN top1 curve: {}'.format(cnn_curve['top1']))
                logging.info('CNN top5 curve: {}'.format(cnn_curve['top5']))
                logging.info('NME top1 curve: {}'.format(nme_curve['top1']))
                logging.info('NME top5 curve: {}\n'.format(nme_curve['top5']))
                logging.info('Inverse NME top1 curve: {}'.format(inverse_nme_curve['top1']))
                logging.info('Inverse NME top5 curve: {}\n'.format(inverse_nme_curve['top5']))
            elif nme_accy is not None:
                logging.info('CNN: {}'.format(cnn_accy['grouped']))
                logging.info('NME: {}'.format(nme_accy['grouped']))

                cnn_curve['top1'].append(cnn_accy['top1'])
                cnn_curve['top5'].append(cnn_accy['top5'])

                nme_curve['top1'].append(nme_accy['top1'])
                nme_curve['top5'].append(nme_accy['top5'])

                logging.info('CNN top1 curve: {}'.format(cnn_curve['top1']))
                logging.info('CNN top5 curve: {}'.format(cnn_curve['top5']))
                logging.info('NME top1 curve: {}'.format(nme_curve['top1']))
                logging.info('NME top5 curve: {}\n'.format(nme_curve['top5']))
            else:
                logging.info('No NME accuracy.')
                logging.info('CNN: {}'.format(cnn_accy['grouped']))

                cnn_curve['top1'].append(cnn_accy['top1'])
                cnn_curve['top5'].append(cnn_accy['top5'])

                logging.info('CNN top1 curve: {}'.format(cnn_curve['top1']))
                logging.info('CNN top5 curve: {}\n'.format(cnn_curve['top5']))


def _set_device(args):
    device_type = args['device']
    gpus = []

    for device in device_type:
        if device_type == -1:
            device = torch.device('cpu')
        else:
            device = torch.device('cuda:{}'.format(device))

        gpus.append(device)

    args['device'] = gpus


def print_args(args):
    for key, value in args.items():
        logging.info('{}: {}'.format(key, value))
    
def _set_random():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
