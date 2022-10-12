import logging
import os
import random

import numpy as np
import torch
import torch.utils.data
from dnd.datasets.loaders import get_data_loaders
from dnd.models import builder
from dnd.args import cfg, logger, args
from dnd.trainer import Trainer
from itertools import chain

num_gpu = torch.cuda.device_count()


def setup_seed(seed):

    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():

    torch.cuda.set_device(args.gpu)

    # Log
    cfg_file_name = os.path.basename(args.cfg)
    filehandler = logging.FileHandler(
        './exp/{}-{}/training.log'.format(args.exp_id, cfg_file_name))
    streamhandler = logging.StreamHandler()

    logger.setLevel(logging.INFO)
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)

    logger.info('******************************')
    logger.info(args)
    logger.info('******************************')
    logger.info(cfg)
    logger.info('******************************')

    setup_seed(args.seed)

    # Model Initialize
    m = preset_model(cfg)
    m.cuda(args.gpu)

    motion_disc = None

    criterion = builder.build_loss(cfg.LOSS).cuda()

    if cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = torch.optim.Adam(
            chain(m.parameters(), criterion.parameters()),
            lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WD)

    if cfg.TRAIN.LR_SCHD == 'multi-step':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=cfg.TRAIN.LR_STEP,
            gamma=cfg.TRAIN.LR_FACTOR)

    data_loaders = get_data_loaders(args, cfg)

    Trainer(
        args=args,
        cfg=cfg,
        logger=logger,
        data_loaders=data_loaders,
        model=m,
        optimizer=optimizer,
        criterion=criterion,
        lr_scheduler=lr_scheduler,
        performance_type='min',
        lr_schd_type=cfg.TRAIN.LR_SCHD,
        motion_disc=motion_disc
    ).fit()


def preset_model(cfg):
    model = builder.build_model(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)

    return model


def get_optimizer(model, optim_type, lr, weight_decay, momentum):
    if optim_type in ['sgd', 'SGD']:
        opt = torch.optim.SGD(lr=lr, params=model.parameters(), momentum=momentum)
    elif optim_type in ['Adam', 'adam', 'ADAM']:
        opt = torch.optim.Adam(lr=lr, params=model.parameters(), weight_decay=weight_decay)
    else:
        raise ModuleNotFoundError
    return opt


if __name__ == "__main__":
    main()
