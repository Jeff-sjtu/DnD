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
from dnd.models.smpl.lbs import batch_rodrigues
import torch
import torch.nn as nn
import torch.nn.functional as F

from .builder import MODEL
from .subnet import DynamicsSubNet


@MODEL.register_module
class DND(nn.Module):
    def __init__(
            self,
            pretrained='data/base_data/spin_model_checkpoint.pth.tar',
            **cfg
    ):

        super(DND, self).__init__()

        self._preset_cfg = cfg['PRESET']
        self.seq_len = self._preset_cfg['SEQLEN']
        self.n_layers = cfg['TGRU']['NUM_LAYERS']
        hidden_size = cfg['TGRU']['HIDDEN_SIZE']
        bidirectional = cfg['TGRU']['BIDIRECTIONAL']
        add_linear = cfg['TGRU']['ADD_LINEAR']
        use_residual = cfg['TGRU']['RESIDUAL']
        self.train_type = cfg['TRAIN_TYPE']
        self.eval_type = cfg['EVAL_TYPE']

        pretrained_enc = cfg['PRETRAIN_ENC']

        # self.encoder = TemporalEncoder(
        #     n_layers=self.n_layers,
        #     hidden_size=hidden_size,
        #     bidirectional=bidirectional,
        #     add_linear=add_linear,
        #     use_residual=use_residual,
        # )

        self.use_smplx = True
        self.regressor = DynamicsSubNet(use_smplx=self.use_smplx, dtype=torch.float32)
        self.epoch = 0

        if pretrained and os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained, map_location='cpu')['model']

            self.regressor.load_state_dict(pretrained_dict, strict=False)
            print(f'=> loaded pretrained model from \'{pretrained}\'')

        if pretrained_enc and os.path.isfile(pretrained_enc):
            pretrained_dict = torch.load(pretrained_enc, map_location='cpu')['gen_state_dict']

            self.load_state_dict(pretrained_dict, strict=False)
            print(f'=> loaded pretrained encoder from \'{pretrained_enc}\'')

    def forward(self, input, J_regressor=None, test_nkpts=14, inp_euler=None, **kwargs):
        # input size NTF
        batch_size, seqlen = input.shape[:2]
        self.regressor.epoch = self.epoch

        ones = torch.ones_like(input)
        ones.requires_grad_()
        input = input * ones

        if self.training:
            smpl_output = self.regressor(input, J_regressor=J_regressor, seq_len=seqlen, inp_euler=inp_euler, **kwargs)
        else:
            smpl_output = self.regressor(input, J_regressor=J_regressor, avg_beta=True, seq_len=seqlen, test_nkpts=test_nkpts, **kwargs)

        return smpl_output
