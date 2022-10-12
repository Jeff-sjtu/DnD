# This script is borrowed and extended from https://github.com/nkolot/SPIN/blob/master/models/hmr.py
# Adhere to their licence to use this script

import math
import os.path as osp
from random import betavariate

import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.activation import ReLU
from dnd.utils.geometry import (quat2mat, rot6d_to_rotmat,
                                           rotation_matrix_to_angle_axis,
                                           mat2quat)

from .smpl.lbs import batch_rodrigues, rot_mat_to_euler, vertices2joints, euler_to_rot_mat2, rot_mat_to_euler_T
from .smpl.SMPL import H36M_TO_J14, SMPL_MEAN_PARAMS, SMPL_layer_dynamics
from .smpl.SMPL_quat import SMPL_quat_layer_dynamics
from .smpl.model_smplx import SMPL as SMPL_x
import torch.nn.functional as F
from .layers.positional_encoding import PositionalEncoding
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

train_seqlen = 32


class ConvHead(nn.Module):

    def __init__(self, seqlen):
        super().__init__()

        self.conv1 = nn.Conv1d(24 * 3, 256,
                               kernel_size=3, stride=1, padding=1,
                               padding_mode='replicate')
        self.bn1 = nn.BatchNorm1d(256)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(256, 256,
                               kernel_size=5, stride=1, padding=2,
                               padding_mode='replicate')
        self.bn2 = nn.BatchNorm1d(256)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv1d(256, 256,
                               kernel_size=5, stride=1, padding=2,
                               padding_mode='replicate')
        self.bn3 = nn.BatchNorm1d(256)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv_out = nn.Conv1d(256, seqlen, kernel_size=1, stride=1, padding=0)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, feat):
        # feat: [B, T, C] -> [B, C, T]
        feat = feat.transpose(1, 2)
        feat = self.relu1(self.bn1(self.conv1(feat)))
        feat = self.relu2(self.bn2(self.conv2(feat)))
        feat = self.relu3(self.bn3(self.conv3(feat)))
        # feat = self.relu4(self.bn4(self.conv4(feat)))

        # prob_map: [B, C2, T]
        prob_map = self.conv_out(feat)
        # prob_map = self.softmax(prob_map).transpose(1, 2)
        prob_map = self.softmax(prob_map)

        return prob_map


class TemporalRNN(nn.Module):
    def __init__(self, n_layers, input_size, hidden_size, output_size):
        super(TemporalRNN, self).__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            bidirectional=True,
        )
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        # x: [B, C, T]
        b, c, t = x.shape
        x = x.permute(2, 0, 1)  # BCT -> TBC
        y, _ = self.gru(x)
        y = F.relu(y)
        y = self.linear(y.view(-1, y.size(-1)))
        y = y.view(t, b, -1)

        y = y.permute(1, 2, 0)  # TBC -> BCT
        return y


class AttenivePDC(nn.Module):
    def __init__(self, npose, diff_input=False):
        super(AttenivePDC, self).__init__()

        self.hidden_dim = 256
        inp_dim = npose * 2 if diff_input else npose
        self.inp_layer = nn.Sequential(
            nn.Conv1d(inp_dim, self.hidden_dim, kernel_size=3, stride=1, padding=1),
            # nn.Conv1d(npose, self.hidden_dim, kernel_size=1, stride=1, padding=0),
            # nn.LayerNorm(train_seqlen),
            nn.ReLU()
        )
        self.inp_layer_t0 = nn.Sequential(
            nn.Conv1d(inp_dim, self.hidden_dim, kernel_size=3, stride=1, padding=1),
            # nn.Conv1d(npose, self.hidden_dim, kernel_size=1, stride=1, padding=0),
            # nn.LayerNorm(train_seqlen),
            nn.ReLU()
        )
        self.npose = npose

        self.enc_layers = self.build_encoder(2)

        self.enc_time0_layers = self.build_encoder(2)

        self.q0_layer = self.build_decoder(n_blocks=2, out_dim=npose * 2)
        self.lt0_layer = self.build_decoder(n_blocks=2, out_dim=3 * 2)
        self.dgt0_layer = TemporalRNN(2, self.hidden_dim, 1024, 3)

        self.kp_layer = TemporalRNN(2, self.hidden_dim, 1024, npose + 3)
        self.kd_layer = TemporalRNN(2, self.hidden_dim, 1024, npose + 3)
        self.alpha_layer = TemporalRNN(2, self.hidden_dim, 1024, npose + 3)

        self.residual_gt_layer = TemporalRNN(2, self.hidden_dim, 1024, 3 + 3 + 3)

    def build_decoder(self, n_blocks, out_dim):
        layers = []
        for n in range(n_blocks):
            layers.append(nn.Conv1d(
                self.hidden_dim, self.hidden_dim,
                padding=1, padding_mode='replicate',
                kernel_size=3, stride=1))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv1d(
            self.hidden_dim, out_dim,
            padding=1, padding_mode='replicate',
            kernel_size=3, stride=1)
        )
        return nn.Sequential(*layers)

    def build_encoder(self, n_blocks):
        layers = []
        for n in range(n_blocks):
            block1 = nn.Sequential(
                nn.Conv1d(
                    self.hidden_dim, self.hidden_dim,
                    kernel_size=3, stride=1, padding=1,
                    padding_mode='replicate'),
                # nn.Dropout(),
                # nn.BatchNorm1d(self.hidden_dim),
                nn.LayerNorm(train_seqlen),  # layer norm across time
                nn.ReLU())

            block2 = nn.Sequential(
                nn.Conv1d(
                    self.hidden_dim, self.hidden_dim,
                    # kernel_size=1, stride=1, padding=0,
                    kernel_size=3, stride=1, padding=1,
                    padding_mode='replicate'),
                # nn.Dropout(),
                # nn.BatchNorm1d(self.hidden_dim),
                nn.LayerNorm(train_seqlen),
                nn.ReLU(),
            )
            layers.append(block1)
            layers.append(block2)
        layers = nn.ModuleList(layers)
        return layers

    def forward(self, x):
        # x: [B, T, C] -> [B, C, T]
        x = x.transpose(1, 2)
        x0 = x

        x0 = self.inp_layer_t0(x0)
        x = self.inp_layer(x)
        for n in range(2):
            x = (x + self.enc_layers[2 * n](x))
            x = self.enc_layers[2 * n + 1](x)

            x0 = (x0 + self.enc_time0_layers[2 * n](x0))
            x0 = self.enc_time0_layers[2 * n + 1](x0)

        # for n in range(2):
        #     x = self.dec_layers[n](x)

        # out = self.out_layer(x)
        # # x: [B, C, T] -> [B, T, C]
        # out = out.transpose(1, 2)

        kp = self.kp_layer(x)
        kd = self.kd_layer(x)
        alpha = self.alpha_layer(x)
        # kp = kp.transpose(1, 2)
        q0 = self.q0_layer(x)
        # dq0 = self.dq0_layer(x)
        lt0 = self.lt0_layer(x)
        kp = kp.transpose(1, 2)
        kd = kd.transpose(1, 2)
        alpha = alpha.transpose(1, 2)

        # dq0 = self.dq0_layer(x0)
        dgt0 = self.dgt0_layer(x0)
        pid_gt = self.residual_gt_layer(x0)
        # [B, C, T] -> [B, T, C]
        q0 = q0.transpose(1, 2)
        # dq0 = dq0.transpose(1, 2)
        q0, dq0 = q0[:, :, :self.npose], q0[:, :, self.npose:]

        # dq0 = dq0.transpose(1, 2)
        dgt0 = dgt0.transpose(1, 2)
        pid_gt = pid_gt.transpose(1, 2)
        lt0 = lt0.transpose(1, 2)
        # lt0, q0 = q0[:, :, :3], q0[:, :, 3:]
        # dlt0, dq0 = dq0[:, :, :3], dq0[:, :, 3:]
        lt0, dlt0 = lt0[:, :, :3], lt0[:, :, 3:]

        kp_gt, kd_gt, alpha_gt = pid_gt[:, :, :3], pid_gt[:, :, 3:6], pid_gt[:, :, 6:]
        # return kp, kd, alpha, q0, dq0, dgt0
        # return kd, alpha, q0, dq0, dgt0, lt0, dlt0
        pid_output = {
            'kp': kp,
            'kd': kd,
            'alpha': alpha,
            'q0': q0,
            'dq0': dq0,
            'dgt0': dgt0,
            'lt0': lt0,
            'dlt0': dlt0,
            'kp_gt': kp_gt,
            'kd_gt': kd_gt,
            'alpha_gt': alpha_gt
        }
        return pid_output


class DynaNet(nn.Module):
    def __init__(self, npose, num_contact):
        super(DynaNet, self).__init__()
        self.num_contact = num_contact

        self.hidden_dim = 256
        self.inp_layer = nn.Sequential(
            nn.Conv1d(npose * 2, self.hidden_dim, kernel_size=3, stride=1, padding=1),
            # nn.Conv1d(npose, self.hidden_dim, kernel_size=1, stride=1, padding=0),
            # nn.LayerNorm(train_seqlen),
            nn.ReLU()
        )
        self.enc_layers = self.build_encoder(2)

        self.contact_layer = TemporalRNN(2, self.hidden_dim, 1024, num_contact)
        self.inertia_force_layer = TemporalRNN(2, self.hidden_dim, 1024, 3)
        self.inertia_angular_layer = TemporalRNN(2, self.hidden_dim, 1024, 3)
        self.gravity_layer = TemporalRNN(2, self.hidden_dim, 1024, 3)

        self.w_y = 200
        self.force_layer = TemporalRNN(2, self.hidden_dim, 1024, num_contact * 3)

        self.act = nn.Softplus()

        self.g_const = 9.81
        self.inertia_const = 5

    def build_encoder(self, n_blocks):
        layers = []
        for n in range(n_blocks):
            block1 = nn.Sequential(
                nn.Conv1d(
                    self.hidden_dim, self.hidden_dim,
                    kernel_size=3, stride=1, padding=1,
                    padding_mode='replicate'),
                # nn.Dropout(),
                # nn.BatchNorm1d(self.hidden_dim),
                nn.LayerNorm(train_seqlen),  # layer norm across time
                nn.ReLU())

            block2 = nn.Sequential(
                nn.Conv1d(
                    self.hidden_dim, self.hidden_dim,
                    # kernel_size=1, stride=1, padding=0,
                    kernel_size=3, stride=1, padding=1,
                    padding_mode='replicate'),
                # nn.Dropout(),
                # nn.BatchNorm1d(self.hidden_dim),
                nn.LayerNorm(train_seqlen),
                nn.ReLU(),
            )
            layers.append(block1)
            layers.append(block2)
        layers = nn.ModuleList(layers)
        return layers

    def forward(self, x):
        # x: [B, T, C]
        bs, seq_len = x.shape[:2]
        x = x.transpose(1, 2)

        x = self.inp_layer(x)
        for n in range(2):
            x = (x + self.enc_layers[2 * n](x))
            x = self.enc_layers[2 * n + 1](x)

        contact = self.contact_layer(x).sigmoid()
        assert contact.shape[1] == self.num_contact
        contact = contact.transpose(1, 2)

        residual_gravity = self.gravity_layer(x).tanh()
        assert residual_gravity.shape[1] == 3
        residual_gravity = residual_gravity.transpose(1, 2)

        gravity = torch.zeros_like(residual_gravity)
        gravity[:, :, 1] = self.g_const
        gravity = gravity + residual_gravity
        gravity = self.g_const * gravity / torch.norm(gravity, p=2, dim=-1, keepdim=True)

        inertia_force = self.inertia_force_layer(x)
        assert inertia_force.shape[1] == 3
        inertia_angular = self.inertia_angular_layer(x).tanh() * math.pi * 0.5
        assert inertia_angular.shape[1] == 3

        inertia_force = inertia_force.transpose(1, 2)
        inertia_force = self.inertia_const * inertia_force.reshape(bs, seq_len, 3, 1).tanh()
        inertia_angular = inertia_angular.transpose(1, 2)

        force = self.force_layer(x)
        force = force.transpose(1, 2).contiguous()
        assert force.shape[2] == self.num_contact * 3
        force = force.reshape(bs, seq_len, self.num_contact, 3).clone()
        force[:, :, :, 1] = -self.act(force[:, :, :, 1].clone()) * self.w_y

        # inertia_force[:, :, 2] = inertia_force[:, :, 2] * 5

        # force = self.force_layer(x)
        # force = force.transpose(1, 2).contiguous()
        # assert force.shape[2] == self.num_contact * 3
        # force = force.reshape(bs, seq_len, self.num_contact, 3).clone()
        # force[:, :, :, 1] = -self.act(force[:, :, :, 1].clone()) * self.w_y

        return contact, gravity[:, :, :, None], inertia_force, inertia_angular, force


class DynamicsSubNet(nn.Module):
    def __init__(self, smpl_mean_params=SMPL_MEAN_PARAMS, dtype=torch.float32, use_smplx=False):
        super(DynamicsSubNet, self).__init__()
        print('SMPL dtype', dtype)
        npose = 24 * 6

        self.pd_controller = AttenivePDC(24 * 3, diff_input=True)
        self.dyna_net = DynaNet(24 * 3, 6)

        self.use_attentive = True

        if self.use_attentive:
            self.attentive = ConvHead(train_seqlen)

        self.quat = False
        self.smplx = use_smplx

        if self.smplx:
            self.smpl = SMPL_x('model_files', create_transl=False, batch_size=64)
        else:
            self.smpl = SMPL_layer_dynamics(
                model_path='model_files/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl', dtype=dtype)

        self.smpl_dyna = SMPL_layer_dynamics(
            model_path='model_files/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl', dtype=dtype)

        # optimization - Train
        BATCH = 14
        res_vel_q = cp.Variable((75))
        pred_vel_q = cp.Parameter((75))
        J1 = cp.Parameter((3, 75))
        J2 = cp.Parameter((3, 75))
        J3 = cp.Parameter((3, 75))
        J4 = cp.Parameter((3, 75))
        # J = cp.Parameter((12, 75))

        delta_vel_T = cp.Variable(3)
        W1 = cp.Parameter()
        W2 = cp.Parameter()
        W3 = cp.Parameter()
        W4 = cp.Parameter()
        orig_delta_vel_T = cp.Parameter(3)
        # delta_vel_T1 = cp.Parameter(3)
        # delta_vel_T2 = cp.Parameter(3)
        # delta_vel_T3 = cp.Parameter(3)
        # delta_vel_T4 = cp.Parameter(3)

        constraints = [
            cp.pnorm(J1 @ res_vel_q + W1 * delta_vel_T, p=1) <= 1e-2,
            cp.pnorm(J2 @ res_vel_q + W2 * delta_vel_T, p=1) <= 1e-2,
            cp.pnorm(J3 @ res_vel_q + W3 * delta_vel_T, p=1) <= 1e-2,
            cp.pnorm(J4 @ res_vel_q + W4 * delta_vel_T, p=1) <= 1e-2,
        ]
        objective = cp.Minimize(cp.pnorm(res_vel_q - pred_vel_q, p=1) + 0.01 * cp.pnorm(orig_delta_vel_T - delta_vel_T, p=1))

        problem = cp.Problem(objective, constraints)
        assert problem.is_dpp(), f'=== Problem is not DDP. {problem.is_dpp()} ==='

        parameters = [pred_vel_q, J1, J2, J3, J4,
                      W1, W2, W3, W4, orig_delta_vel_T]
        # parameters = [pred_vel_q, J, delta_vel_T]
        self.cvxpylayer = CvxpyLayer(problem, parameters=parameters, variables=[res_vel_q, delta_vel_T])

    def forward(self, x, init_pose=None, init_shape=None, init_cam=None, n_iter=3, test_nkpts=14,
                is_train=False, J_regressor=None, run_smpl=True, avg_beta=False, seq_len=None, inp_euler=None, **kwargs):
        batch_size = x.shape[0] * x.shape[1]

        # pred_transl, pred_aa, pred_shape = x[:, :, :3], x[:, :, 3:-10], x[:, :, -10:]
        pred_transl, pred_rotmat, pred_shape = x[:, :, :3], x[:, :, 3:-10], x[:, :, -10:]
        # pred_rotmat = batch_rodrigues(pred_aa.reshape(-1, 3)).reshape(-1, seq_len, 24, 3, 3)
        pred_rotmat = pred_rotmat.reshape(-1, seq_len, 24, 3, 3)

        pred_rotmat = pred_rotmat.reshape(-1, seq_len, 24, 3, 3)
        pred_rotmat = pred_rotmat.transpose(0, 1).reshape(seq_len, -1, 3, 3)
        pred_euler_angle = rot_mat_to_euler_T(pred_rotmat)
        pred_euler_angle = pred_euler_angle.reshape(seq_len, -1, 24, 3)
        pred_euler_angle = pred_euler_angle.transpose(0, 1)

        if self.use_attentive:
            pred_euler_angle = pred_euler_angle.reshape(-1, seq_len, 24 * 3)
            pred_transl = pred_transl.reshape(-1, seq_len, 3)
            # [B, T, C]
            input_x = pred_euler_angle
            # [B, T2, T, 1]
            prob_map = self.conv_map(input_x).clone()[:, :, :, None]

            pred_euler_angle_new = torch.sum(prob_map * pred_euler_angle[:, None, :, :], dim=2).clone()
            pred_transl_new = torch.sum(prob_map * pred_transl[:, None, :, :], dim=2).clone()

            pred_euler_angle_new = pred_euler_angle_new.reshape(-1, seq_len, 24 * 3)
        else:
            pred_euler_angle_new = pred_euler_angle
            pred_transl_new = pred_transl

        pred_rotmat = pred_rotmat.transpose(0, 1)
        pred_pose = pred_euler_angle

        transl = pred_transl.reshape(-1, 3)
        pred_cam = torch.stack([
            1000. / (256. * transl[:, 2] + 1e-9),
            transl[:, 0],
            transl[:, 1]], dim=-1)

        pred_cam_old = pred_cam.clone()
        transl_old = transl.clone()

        # print(transl[0, :], transl[-1, :], 'transl orig')

        pred_transl = transl.reshape(-1, seq_len, 3)
        pred_transl[:, :, 2] = pred_transl[:, :, 2]

        pred_rotmat = pred_rotmat.reshape(-1, seq_len, 24 * 9)

        pred_shape = pred_shape.reshape(-1, seq_len, 10)

        # inp_q = pred_euler_angle
        pred_euler_angle = pred_euler_angle.reshape(-1, seq_len, 24 * 3)
        pred_euler_angle_new = pred_euler_angle_new.reshape(-1, seq_len, 24 * 3)
        if inp_euler is not None:
            inp_euler, inp_beta = inp_euler

            inp_q = torch.cat((pred_euler_angle, inp_euler), dim=0)
            inp_t = torch.cat((pred_transl, torch.zeros_like(inp_euler[:, :, :3])), dim=0)
            inp_q_new = torch.cat((pred_euler_angle_new, inp_euler), dim=0)
            inp_t_new = torch.cat((pred_transl_new, torch.zeros_like(inp_euler[:, :, :3])), dim=0)

            pred_shape = torch.cat((pred_shape, inp_beta), dim=0)
            batch_size = batch_size + inp_euler.shape[0] * seq_len
        else:
            inp_q = pred_euler_angle
            inp_t = pred_transl
            inp_q_new = pred_euler_angle_new
            inp_t_new = pred_transl_new

        inp_t[:, :, 2] = inp_t[:, :, 2] / 2
        inp_t_new[:, :, 2] = inp_t_new[:, :, 2] / 2

        inp_q_with_t = torch.cat((inp_t, inp_q), dim=2)

        inp_dq = torch.zeros_like(inp_q)
        inp_dq[:, :-1] = inp_q[:, 1:] - inp_q[:, :-1]
        inp_dq[:, -1] = inp_dq[:, -2]

        inp = torch.cat((inp_q, inp_dq), dim=2)

        kine_out_orig = self.run_smpl(inp_q_with_t.reshape(-1, 75), pred_shape.reshape(-1, 10))
        M_orig = kine_out_orig.M
        M_inv_orig = torch.inverse(M_orig).reshape(-1, seq_len, 75, 75)

        pid_output = self.pd_controller(inp)
        # ext_force: [B, T, 6, 3]
        # contact: [B, T, 6]
        # gravity: [B, T, 3, 1]
        # i_force: [B, T, 3, 1]
        temporal_contact, temporal_gravity, temporal_i_linear, temporal_i_angular_v, ext_force = self.dyna_net(inp)

        dq_dt2_list, e_tau_list, i_tau, expected_tau = self.smpl_dyna.forward_dynamics(
            kine_out_orig, 0,
            ext_force.reshape(batch_size, 6, 3, 1),
            temporal_contact.reshape(batch_size, 6),
            inertia_angular=temporal_i_angular_v.reshape(batch_size, 3, 1),
            g=temporal_gravity.reshape(batch_size, 3, 1),
        )

        temporal_i_angular = torch.zeros_like(temporal_i_angular_v)
        time_interval = 1 / 25
        for t in range(1, seq_len):
            temporal_i_angular[:, t, :] = torch.sum(temporal_i_angular_v[:, :t, :], dim=1) * time_interval
        # [B, T, 3, 3]
        temporal_i_angular_rotmat = euler_to_rot_mat2(temporal_i_angular.reshape(-1, 3)).reshape(-1, seq_len, 3, 3)

        expected_tau = expected_tau.reshape(-1, seq_len, 75)
        i_tau = i_tau.reshape(-1, seq_len, 75)

        kp = pid_output['kp'].sigmoid() * 0.3
        kd = pid_output['kd'].sigmoid() * 2
        alpha = pid_output['alpha'].tanh().clone()

        kp_q = kp[:, :, 3:]
        kp_t = kp[:, :, :3]

        kd_q = kd[:, :, 3:]
        kd_t = kd[:, :, :3]

        # ## dalpha ##
        alpha_q = alpha[:, :, 3:] * math.pi * 2
        alpha_t = alpha[:, :, :3]
        alpha_t[:, :, 2] = alpha_t[:, :, 2] * 3

        alpha_q = alpha_q[:, 1:] - alpha_q[:, :-1]
        alpha_q = alpha_q[:, 1:] - alpha_q[:, :-1]
        alpha_t = alpha_t[:, 1:] - alpha_t[:, :-1]
        alpha_t = alpha_t[:, 1:] - alpha_t[:, :-1]

        # alpha_q = alpha[:, :, 3:] * math.pi
        # alpha_t = alpha[:, :, :3]

        q0 = pid_output['q0'][:, [0], :].tanh().clone() * math.pi * 0.5 + inp_q[:, [0], :]
        dq0 = pid_output['dq0'][:, [0], :].tanh().clone() * math.pi * 0.5 + inp_dq[:, [0], :]
        lt0 = pid_output['lt0'][:, [0], :].tanh().clone() * 0.5 + inp_t[:, [0], :]
        dlt0 = pid_output['dlt0'][:, [0], :].tanh().clone() * 0.5 + (inp_t[:, [1], :] - inp_t[:, [0], :])

        if torch.sum(torch.isnan(q0)) > 0:
            print('wrong q0')
            raise Exception
        if torch.sum(torch.isnan(lt0)) > 0:
            print('wrong lt0')
            raise Exception
        if torch.sum(torch.isnan(dlt0)) > 0:
            print('wrong dlt0')
            raise Exception
        kp_gt = pid_output['kp_gt'].sigmoid()
        kd_gt = pid_output['kd_gt'].sigmoid() * 2

        alpha_gt = pid_output['alpha_gt'].tanh()
        dd_alpha_gt = alpha_gt[:, 1:] - alpha_gt[:, :-1]
        dd_alpha_gt = dd_alpha_gt[:, 1:] - dd_alpha_gt[:, :-1]

        dgt0 = alpha_gt[:, [1], :] - alpha_gt[:, [0], :]
        gt0 = alpha_gt[:, [0], :]

        q_list = [q0]
        dq_list = [dq0]

        local_t_list = [lt0]
        local_dt_list = [dlt0]

        global_t_list = [gt0]
        global_dt_list = [dgt0]

        inertia_list = []
        internal_tau_list = []
        residual_global_ddt_list = []

        time_interval = 1 / 25
        for t in range(1, seq_len):
            prev_q = q_list[t - 1].squeeze(dim=1)
            M_inv = M_inv_orig[:, t - 1]
            # kine_out_prev = self.run_smpl(prev_q, pred_shape[:, t - 1, :])
            # M = kine_out_prev.M.detach()
            # with torch.no_grad():
            #     M_inv = torch.inverse(M)
            if t < seq_len - 1:
                R_cam_T = temporal_i_angular_rotmat[:, t - 1]

                ddq = M_inv.matmul(kp_q[:, [t - 1], :] * (inp_q_new[:, [t], :] - q_list[t - 1]) - kd_q[:, [t - 1], :] * dq_list[t - 1] + alpha_q[:, [t - 1], :])
                local_ddt = kp_t[:, [t - 1], :] * (inp_t[:, [t], :] - local_t_list[t - 1]) - kd_t[:, [t - 1], :] * local_dt_list[t - 1] + alpha_t[:, [t - 1], :]

                # ddq = alpha_q[:, [t - 1], :]
                # local_ddt = alpha_t[:, [t - 1], :]

                # [B, 1, 75]
                ddq_ext = M_inv.matmul(expected_tau[:, t - 1, :, None]).transpose(1, 2)
                ddq_ext = ddq_ext.float() * time_interval * time_interval

                ddq_I = M_inv.matmul(i_tau[:, t - 1, :, None]).transpose(1, 2)
                ddq_I = ddq_I.float() * time_interval * time_interval

                residual_global_ddt = dd_alpha_gt[:, [t - 1], :]
                global_ddt = residual_global_ddt + ddq_ext[:, :, :3]

                global_ddt = R_cam_T.matmul(global_ddt.transpose(1, 2)).transpose(1, 2)

                ddq = ddq + ddq_ext[:, :, 3:] + ddq_I[:, :, 3:]
                local_ddt = local_ddt + ddq_ext[:, :, :3] + ddq_I[:, :, :3]

                ddt_inertia = local_ddt[:, :, :3] - global_ddt

                inertia_list.append(ddt_inertia)

                ddq_internal_tau = ddq - ddq_ext[:, :, 3:]
                internal_tau_list.append(ddq_internal_tau)

                dq = dq_list[t - 1] + ddq

                local_dt = local_dt_list[t - 1] + local_ddt
                global_dt = global_dt_list[t - 1] + global_ddt

                q = q_list[t - 1] + dq_list[t - 1]
                local_t = local_t_list[t - 1] + local_dt_list[t - 1]
                global_t = global_t_list[t - 1] + global_dt_list[t - 1]

                q_list.append(q)
                dq_list.append(dq)

                local_t_list.append(local_t)
                local_dt_list.append(local_dt)

                global_t_list.append(global_t)
                global_dt_list.append(global_dt)
                residual_global_ddt_list.append(residual_global_ddt)
            else:
                q = q_list[t - 1] + dq_list[t - 1]
                local_t = local_t_list[t - 1] + local_dt_list[t - 1]
                global_t = global_t_list[t - 1] + global_dt_list[t - 1]

                q_list.append(q)
                local_t_list.append(local_t)
                global_t_list.append(global_t)

            # diff_accel_transl.append(inp_accel[:, t - 1, :6])

        pred_inertia = torch.cat(inertia_list, dim=1)
        pred_internal_tau = torch.cat(internal_tau_list, dim=1)

        pred_q = torch.cat(q_list, dim=1)
        pred_local_t = torch.cat(local_t_list, dim=1)
        if not direct_gt:
            pred_global_t = torch.cat(global_t_list, dim=1)

        pred_pose = pred_q.reshape(batch_size, 24 * 3)
        transl = pred_local_t.reshape(batch_size, 3)
        global_transl = pred_global_t.reshape(-1, seq_len, 3)

        global_transl = global_transl - global_transl[:, [0], :]
        pred_global_motion = torch.cat((global_transl, pred_q), dim=2)

        residual_global_ddt = torch.cat(residual_global_ddt_list, dim=1)

        # Optimization
        cvx_opt = self.epoch > 20
        pred_pose = pred_pose.reshape(-1, seq_len, 24 * 3)
        transl = transl.reshape(-1, seq_len, 3)
        pred_global_transl = global_transl.reshape(-1, seq_len, 3)

        q_with_t = torch.cat((transl, pred_pose), dim=2)
        vel_q_with_t = q_with_t[:, 1:] - q_with_t[:, :-1]

        l2g_t = pred_global_transl - transl
        vel_l2g_t = l2g_t[:, 1:] - l2g_t[:, :-1]

        if cvx_opt:
            orig_vel_l2g_t = l2g_t[:, 1:] - l2g_t[:, :-1]
            orig_delta_vel_T = None

            vel_l2g_t_list = []
            q_list = [q_with_t[:, [0]]]
            for t in range(1, seq_len):
                contact_t = temporal_contact[:, t]
                prev_q = q_list[t - 1]

                pred_vel_q = vel_q_with_t[:, [t - 1]]

                if orig_delta_vel_T is None:
                    orig_delta_vel_T = orig_vel_l2g_t[:, t - 1]

                kine_out_prev = self.run_smpl2(prev_q, pred_shape[:, t - 1, :])
                Jv_joints = kine_out_prev.Jv_joints

                J1 = Jv_joints[self.smpl_dyna.CONTACT_JOINTS[2]]
                J2 = Jv_joints[self.smpl_dyna.CONTACT_JOINTS[3]]
                J3 = Jv_joints[self.smpl_dyna.CONTACT_JOINTS[4]]
                J4 = Jv_joints[self.smpl_dyna.CONTACT_JOINTS[5]]
                W1 = (contact_t[:, [2]] > 0.5).float()
                W2 = (contact_t[:, [3]] > 0.5).float()
                W3 = (contact_t[:, [4]] > 0.5).float()
                W4 = (contact_t[:, [5]] > 0.5).float()

                J1 = J1 * W1[:, :, None]
                J2 = J2 * W2[:, :, None]
                J3 = J3 * W3[:, :, None]
                J4 = J4 * W4[:, :, None]

                solution_vel_q, solution_delta_vel_T = self.cvxpylayer(
                    pred_vel_q.squeeze(1),
                    J1, J2, J3, J4,
                    W1.squeeze(1), W2.squeeze(1), W3.squeeze(1), W4.squeeze(1),
                    orig_delta_vel_T
                    # delta_vel_T1.squeeze(1), delta_vel_T2.squeeze(1), delta_vel_T3.squeeze(1), delta_vel_T4.squeeze(1)
                )
                # print(solution_vel_q.shape, solution_delta_vel_T.shape)
                solution_vel_q = solution_vel_q.unsqueeze(1)
                orig_delta_vel_T = solution_delta_vel_T
                solution_delta_vel_T = solution_delta_vel_T.unsqueeze(1)

                vel_l2g_t_list.append(solution_delta_vel_T)

                q = q_list[t - 1] + solution_vel_q
                q_list.append(q)

            pred_q = torch.cat(q_list, dim=1)
            vel_l2g_t = torch.cat(vel_l2g_t_list, dim=1)

            pred_pose = pred_q[:, :, 3:]
            transl = pred_q[:, :, :3]
        # End Optimization

        pred_pose = pred_pose.reshape(batch_size, 24 * 3)
        transl = transl.reshape(batch_size, 3)

        transl[:, 2] = transl[:, 2] * 2
        # print(transl[0, :], transl[-1, :], 'transl')
        pred_cam = torch.stack([
            1000. / (256. * transl[:, 2] + 1e-8),
            transl[:, 0],
            transl[:, 1]], dim=-1)

        # transl = transl_old
        # pred_cam = pred_cam_old

        pred_vel = pred_q[:, 1:] - pred_q[:, :-1]
        pred_accel = pred_vel[:, 1:] - pred_vel[:, :-1]

        transl_accel = pred_accel[:, :, :3]
        # print(transl_accel[-1, -1, :3], 'transl accel ==========')

        pred_rotmat = euler_to_rot_mat2(pred_pose.reshape(-1, 3)).reshape(-1, 24, 3, 3)

        pred_euler_angle = pred_pose.reshape(batch_size, 24, 3)

        q = torch.cat((transl[:, None, :], pred_euler_angle), dim=1)
        pred_shape = pred_shape.reshape(batch_size, 10)

        if avg_beta:
            pred_shape = pred_shape.reshape(-1, seq_len, 10)
            mean_pred_shape = torch.mean(pred_shape, dim=1, keepdim=True)
            mean_pred_shape = mean_pred_shape.expand_as(pred_shape)
            pred_shape = mean_pred_shape.reshape(-1, 10)

        pred_output = self.smpl(
            betas=pred_shape,
            body_pose=pred_rotmat[:, 1:],
            global_orient=pred_rotmat[:, [0]],
            pose2rot=False
        )

        pred_vertices = pred_output['vertices']
        pred_joints = pred_output['joints']

        pred_14joints = pred_joints[:, 25:39]

        if not is_train and J_regressor is not None:
            J_regressor_batch = J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1).to(pred_vertices.device)
            pred_joints = torch.matmul(J_regressor_batch, pred_vertices)
            if test_nkpts == 14:
                pred_joints = pred_joints[:, H36M_TO_J14, :]
            else:
                # pred_joints = pred_joints[:, H36M_TO_J17, :]
                pred_joints = pred_joints

        if self.smplx:
            pred_keypoints_2d = projection(pred_joints, transl, transl=True)

        pose = rotation_matrix_to_angle_axis(pred_rotmat.reshape(-1, 3, 3)).reshape(-1, 72)

        if torch.sum(torch.isnan(transl)) > 0:
            print('wrong transl')
            raise Exception
        if torch.sum(torch.isnan(pred_keypoints_2d)) > 0:
            print('wrong pred_keypoints_2d')
            raise Exception
        if torch.sum(torch.isnan(pred_joints)) > 0:
            print('wrong pred_joints')
            raise Exception

        output = [{
            'theta': torch.cat([pred_cam, pose, pred_shape], dim=1).reshape(batch_size // seq_len, seq_len, -1),
            'pred_cam': pred_cam,
            'pred_pose': pred_pose,
            'pred_shape': pred_shape,
            'verts': pred_vertices.reshape(batch_size // seq_len, seq_len, -1, 3),
            'kp_2d': pred_keypoints_2d.reshape(batch_size // seq_len, seq_len, -1, 2),
            'kp_3d': pred_joints.reshape(batch_size // seq_len, seq_len, -1, 3),
            'pred_14joints': pred_14joints,
            'rotmat': pred_rotmat.reshape(batch_size // seq_len, seq_len, -1, 3, 3),
            'transl': transl.reshape(batch_size // seq_len, seq_len, 3),
            'global_transl': global_transl,
            'vel_l2g_t': vel_l2g_t,
            # 'q': q,
            'pred_euler_angle': pred_euler_angle,
            'kine_out': pred_output,
            'contact': temporal_contact,
            'expected_tau': expected_tau,
            'transl_accel': transl_accel,
            'pred_inertia': pred_inertia,
            'ext_force': ext_force,
            'internal_tau': pred_internal_tau,
            'residual_global_ddt': residual_global_ddt,
            'global_motion': pred_global_motion
        }]
        return output

    def run_smpl(self, inp_q, betas):
        b, _ = inp_q.shape
        inp_q = inp_q.reshape(-1, 75)
        betas = betas.reshape(-1, 10)

        kine_out_orig = self.smpl_dyna(
            pose_angle=inp_q[:, 3:].reshape(-1, 24, 3).type(self.smpl_dyna.dtype),
            betas=betas.type(self.smpl_dyna.dtype),
            transl=inp_q[:, :3].type(self.smpl_dyna.dtype),
            with_d=False
        )

        kine_out_orig = self.smpl_dyna.update_mass(kine_out_orig)
        kine_out_orig = self.smpl_dyna.get_jacobian(kine_out_orig)

        # e, _ = torch.symeig(kine_out_orig.M)
        # max_e, _ = torch.max(e, dim=1)
        # min_e, _ = torch.min(e, dim=1)
        # # print(kine_out_orig.M[0][:10, :10])
        # print(max_e / min_e)

        return kine_out_orig

    def run_smpl2(self, inp_q, betas):
        inp_q = inp_q.reshape(-1, 75)
        betas = betas.reshape(-1, 10)

        kine_out_orig = self.smpl_dyna(
            pose_angle=inp_q[:, 3:].reshape(-1, 24, 3).type(self.smpl_dyna.dtype),
            betas=betas.type(self.smpl_dyna.dtype),
            transl=inp_q[:, :3].type(self.smpl_dyna.dtype),
            with_d=False
        )

        kine_out_orig = self.smpl_dyna.get_jacobian2(kine_out_orig)

        return kine_out_orig


def projection(pred_joints, pred_transl, transl=False):

    batch_size = pred_joints.shape[0]
    camera_center = torch.zeros(batch_size, 2, device=pred_joints.device, dtype=pred_joints.dtype)
    pred_keypoints_2d = perspective_projection(pred_joints,
                                               rotation=torch.eye(3, device=pred_joints.device, dtype=pred_joints.dtype).unsqueeze(0).expand(batch_size, -1, -1),
                                               translation=pred_transl,
                                               focal_length=1000.,
                                               camera_center=camera_center)
    # Normalize keypoints to [-1,1]
    pred_keypoints_2d = pred_keypoints_2d / (256. / 2.)
    return pred_keypoints_2d


def perspective_projection(points, rotation, translation,
                           focal_length, camera_center):
    """
    This function computes the perspective projection of a set of points.
    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
        focal_length (bs,) or scalar: Focal length
        camera_center (bs, 2): Camera center
    """
    batch_size = points.shape[0]
    K = torch.zeros([batch_size, 3, 3], device=points.device, dtype=points.dtype)
    K[:, 0, 0] = focal_length
    K[:, 1, 1] = focal_length
    K[:, 2, 2] = 1.
    K[:, :-1, -1] = camera_center

    # Transform points
    points = torch.einsum('bij,bkj->bki', rotation, points)
    if translation is not None:
        points = points + translation.unsqueeze(1)

    # Apply perspective distortion
    projected_points = points / (points[:, :, -1].unsqueeze(-1) + 1e-6)

    # Apply camera intrinsics
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points)

    return projected_points[:, :, :-1]
