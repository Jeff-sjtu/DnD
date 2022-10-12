import argparse
import logging
import os

import torch

from .utils.config import update_config

parser = argparse.ArgumentParser(description='Residual Log-Likehood')

"----------------------------- Experiment options -----------------------------"
parser.add_argument('--cfg',
                    help='experiment configure file name',
                    required=True,
                    type=str)
parser.add_argument('--exp-id', default='default', type=str,
                    help='Experiment ID')

"----------------------------- General options -----------------------------"
parser.add_argument('--snapshot', default=2, type=int,
                    help='How often to take a snapshot of the model (0 = never)')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id')

parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--world-size', default=-1, type=int,
                    help='world size for distributed training')
parser.add_argument('--dist-url', default='tcp://192.168.1.219:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none',
                    help='job launcher')

"----------------------------- Training options -----------------------------"
parser.add_argument('--seed', default=123123, type=int,
                    help='random seed')

"----------------------------- Log options -----------------------------"
parser.add_argument('--ckpt',
                    help='checkpoint file name',
                    type=str)


args = parser.parse_args()
cfg_file_name = os.path.basename(args.cfg)
cfg = update_config(args.cfg)

cfg['FILE_NAME'] = cfg_file_name

num_gpu = torch.cuda.device_count()
args.world_size = num_gpu

args.work_dir = './exp/{}-{}/'.format(args.exp_id, cfg_file_name)
args.device = torch.device("cuda:" + str(args.gpu) if args.gpu >= 0 else "cpu")

if not os.path.exists("./exp/{}-{}".format(args.exp_id, cfg_file_name)):
    os.makedirs("./exp/{}-{}".format(args.exp_id, cfg_file_name), exist_ok=True)

logger = logging.getLogger('')

cfg['LOSS']['TRAIN_TYPE'] = cfg['MODEL']['TRAIN_TYPE']
cfg['MODEL']['EVAL_TYPE'] = cfg['TEST']['EVAL_TYPE']
