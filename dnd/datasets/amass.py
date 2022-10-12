"""AMASS dataset, developed based on VIBE."""
import os
import os.path as osp

import joblib
import numpy as np
import torch
import torch.utils.data as data
from dnd.models.builder import DATASET
from .dataset_3d import Dataset3D


@DATASET.register_module
class AMASS(Dataset3D):
    def __init__(self, cfg, ann_file, overlap=0.75, train=True):
        db_name = 'amass'

        # overlap = overlap if is_train else 0
        super(AMASS, self).__init__(
            cfg=cfg,
            ann_file=ann_file,
            overlap=overlap,
            train=train,
            dataset_name=db_name,
        )
        print(f'{db_name} - number of dataset objects {self.__len__()}')
