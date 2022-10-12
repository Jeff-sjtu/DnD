"""Dataset Loader, developed based on VIBE."""
import random

import numpy as np
from easydict import EasyDict
from torch.utils.data import ConcatDataset, DataLoader
from .mix_dataset import MixDataset

from . import *  # noqa: F401,F403


def get_test_loaders(args, cfg):

    preset_cfg = cfg.DATA_PRESET
    if preset_cfg.TEST_OVERLAP:
        overlap = (preset_cfg.SEQLEN - 1) / float(preset_cfg.SEQLEN)
    else:
        overlap = 0

    # ===== Evaluation dataset =====
    if isinstance(cfg.DATASET.TESTSET, list):
        test_loder_list = []
        for dataset in cfg.DATASET.TESTSET:
            dataset.PRESET = preset_cfg
            test_db = eval(dataset.NAME)(
                dataset, dataset.ANN_FILE, overlap=overlap, train=False)

            test_loader = DataLoader(
                dataset=test_db,
                batch_size=cfg.TRAIN.BATCH_SIZE,
                shuffle=False,
                num_workers=cfg.TRAIN.NUM_WORKERS,
            )
            test_loder_list.append(test_loader)
        return test_loder_list
    else:
        cfg.DATASET.TESTSET.PRESET = preset_cfg
        test_db = eval(cfg.DATASET.TESTSET.NAME)(
            cfg.DATASET.TESTSET, cfg.DATASET.TESTSET.ANN_FILE, overlap=overlap, train=False)

        test_loader = DataLoader(
            dataset=test_db,
            batch_size=cfg.TRAIN.BATCH_SIZE,
            shuffle=False,
            num_workers=cfg.TRAIN.NUM_WORKERS,
        )

        return test_loader


def get_data_loaders(args, cfg):

    def _init_fn(worker_id):
        np.random.seed(args.seed + worker_id)
        random.seed(args.seed + worker_id)

    def get_3d_datasets(dataset_cfgs, preset_cfg, overlap):
        datasets = []
        partitions = []
        sum_w = 0
        for dataset_cfg in dataset_cfgs:
            dataset_cfg.PRESET = preset_cfg
            db = eval(dataset_cfg.NAME)(
                dataset_cfg, dataset_cfg.ANN_FILE, overlap=overlap, train=True)
            datasets.append(db)
            partitions.append(dataset_cfg.W)
            sum_w += dataset_cfg.W
        if len(datasets) == 1 or sum_w == 0:
            return ConcatDataset(datasets)
        else:
            # return ConcatDataset(datasets)
            return MixDataset(datasets, partitions)

    cfg = EasyDict(cfg)

    preset_cfg = cfg.DATA_PRESET
    overlap = (preset_cfg.SEQLEN - 1) / float(preset_cfg.SEQLEN)

    data_3d_batch_size = cfg.TRAIN.BATCH_SIZE

    # ===== 3D keypoint datasets =====
    train_3d_dataset_names = cfg.DATASET.TRAINSET_LIST_3D
    train_3d_db = get_3d_datasets(train_3d_dataset_names, preset_cfg, overlap)

    train_3d_loader = DataLoader(
        dataset=train_3d_db,
        batch_size=data_3d_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.TRAIN.NUM_WORKERS,
    )

    # ===== Evaluation dataset =====
    cfg.DATASET.VALSET.PRESET = preset_cfg

    test_overlap = 0

    valid_db = eval(cfg.DATASET.VALSET.NAME)(
        cfg.DATASET.VALSET, cfg.DATASET.VALSET.ANN_FILE, overlap=test_overlap, train=False)

    valid_loader = DataLoader(
        dataset=valid_db,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.TRAIN.NUM_WORKERS,
    )

    return train_3d_loader, valid_loader
