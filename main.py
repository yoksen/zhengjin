import json
import argparse
import os
import datetime
import sys
import warnings
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
warnings.filterwarnings("ignore")
import logging
from trainer import train


def main():
    args = setup_parser().parse_args()
    param = load_json(args.config)
    args = vars(args)  # Converting argparse Namespace to a dict.
    args.update(param)  # Add parameters from json

    setup_logging(args)
    train(args)


def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)

    return param


def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorthms.')
    parser.add_argument('--config', type=str, default='./config.json',
                        help='Json file of settings.')

    return parser


def setup_logging(args):
    logpath = "./results/{}/base_{}/incre_{}/{}".format(args['dataset'], args['init_cls'], args['increment'], args['model_name'])
    if not os.path.exists(logpath):
        os.makedirs(logpath)
    
    nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    logfile = '{}_pretrained-{}_fixed_memory-{}_{}.log'.format(args['convnet_type'], args['pretrained'], args['fixed_memory'], nowTime)
    

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(filename)s] => %(message)s',
        handlers=[
            logging.FileHandler(filename=os.path.join(logpath, logfile)),
            logging.StreamHandler(sys.stdout)
        ]
    )


if __name__ == '__main__':
    main()