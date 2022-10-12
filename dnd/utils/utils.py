# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import os
import yaml
import torch
import logging
import operator
from tqdm import tqdm
from os import path as osp
from functools import reduce
from typing import List, Union
from collections import OrderedDict


def save_obj(v, f, file_name='output.obj'):
    obj_file = open(file_name, 'w')
    for i in range(len(v)):
        obj_file.write('v ' + str(v[i][0]) + ' ' + str(v[i][1]) + ' ' + str(v[i][2]) + '\n')
    for i in range(len(f)):
        obj_file.write('f ' + str(f[i][0] + 1) + '/' + str(f[i][0] + 1) + ' ' + str(f[i][1] + 1) + '/' + str(f[i][1] + 1) + ' ' + str(f[i][2] + 1) + '/' + str(f[i][2] + 1) + '\n')
    obj_file.close()


def check_data_pararell(train_weight):
    new_state_dict = OrderedDict()
    for k, v in train_weight.items():
        name = k[7:] if k.startswith('module') else k  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


def move_dict_to_device(dict, device, tensor2float=False):
    for k, v in dict.items():
        if isinstance(v, torch.Tensor):
            if tensor2float:
                dict[k] = v.float().to(device)
            else:
                dict[k] = v.to(device)


def get_from_dict(dict, keys):
    return reduce(operator.getitem, keys, dict)


def tqdm_enumerate(iter):
    i = 0
    for y in tqdm(iter):
        yield i, y
        i += 1


def iterdict(d):
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = dict(v)
            iterdict(v)
    return d


def accuracy(output, target):
    _, pred = output.topk(1)
    pred = pred.view(-1)

    correct = pred.eq(target).sum()

    return correct.item(), target.size(0) - correct.item()


def lr_decay(optimizer, step, lr, decay_step, gamma):
    lr = lr * gamma ** (step / decay_step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def step_decay(optimizer, step, lr, decay_step, gamma):
    lr = lr * gamma ** (step / decay_step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def read_yaml(filename):
    return yaml.load(open(filename, 'r'))


def write_yaml(filename, object):
    with open(filename, 'w') as f:
        yaml.dump(object, f)


def save_dict_to_yaml(obj, filename, mode='w'):
    with open(filename, mode) as f:
        yaml.dump(obj, f, default_flow_style=False)


def save_to_file(obj, filename, mode='w'):
    with open(filename, mode) as f:
        f.write(obj)


def concatenate_dicts(dict_list, dim=0):
    rdict = dict.fromkeys(dict_list[0].keys())
    for k in rdict.keys():
        rdict[k] = torch.cat([d[k] for d in dict_list], dim=dim)
    return rdict


def bool_to_string(x: Union[List[bool], bool]) -> Union[List[str], str]:
    """
    boolean to string conversion
    :param x: list or bool to be converted
    :return: string converted thing
    """
    if isinstance(x, bool):
        return [str(x)]
    for i, j in enumerate(x):
        x[i] = str(j)
    return x


def checkpoint2model(checkpoint, key='gen_state_dict'):
    state_dict = checkpoint[key]
    print(f'Performance of loaded model on 3DPW is {checkpoint["performance"]:.2f}mm')
    # del state_dict['regressor.mean_theta']
    return state_dict


def get_optimizer(model, optim_type, lr, weight_decay, momentum):
    if optim_type in ['sgd', 'SGD']:
        opt = torch.optim.SGD(lr=lr, params=model.parameters(), momentum=momentum)
    elif optim_type in ['Adam', 'adam', 'ADAM']:
        opt = torch.optim.Adam(lr=lr, params=model.parameters(), weight_decay=weight_decay)
    else:
        raise ModuleNotFoundError
    return opt


def create_logger(logdir, phase='train'):
    os.makedirs(logdir, exist_ok=True)

    log_file = osp.join(logdir, f'{phase}_log.txt')

    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=log_file,
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger


class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
