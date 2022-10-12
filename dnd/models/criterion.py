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

import torch
import torch.nn as nn
import numpy as np
from dnd.utils.geometry import batch_rodrigues, rotation_matrix_to_angle_axis
from .builder import LOSS


@LOSS.register_module
class LossTransl(nn.Module):
    def __init__(
            self,
            TRAIN_TYPE,
            WEIGHT,
    ):
        super(LossTransl, self).__init__()
        self.e_loss_weight = WEIGHT['KP_2D_W']
        self.e_3d_loss_weight = WEIGHT['KP_3D_W']
        self.e_3d_accel_loss_weight = WEIGHT['KP_3D_ACCEL_W']
        self.e_pose_loss_weight = WEIGHT['POSE_W']
        self.e_shape_loss_weight = WEIGHT['SHAPE_W']

        self.vel_reg_weight = WEIGHT['VEL_REG']
        self.acc_reg_weight = WEIGHT['ACC_REG']
        self.transl_reg_weight = WEIGHT['TRANSL_REG']

        self.contact_weight = WEIGHT['CONTACT_W']
        self.inertia_weight = WEIGHT['INERTIA_W']
        self.global_transl_weight = WEIGHT['G_TRANSL_W']

        self.d_motion_loss_weight = WEIGHT.get('D_MOTION_LOSS_W', 0)

        self.criterion_shape = nn.L1Loss()
        # self.criterion_keypoints = nn.MSELoss(reduction='none')
        self.criterion_keypoints = nn.L1Loss(reduction='none')
        # self.criterion_accel = nn.L1Loss(reduction='none')
        self.criterion_accel = nn.MSELoss(reduction='none')
        self.criterion_regr = nn.MSELoss()
        # self.criterion_accel = nn.MSELoss().to(self.device)
        # self.criterion_accel = nn.L1Loss().to(self.device)
        self.criterion_attention = nn.CrossEntropyLoss()

        self.enc_loss = batch_encoder_disc_l2_loss

        self.train_type = TRAIN_TYPE

        self.debug = True

    def forward(
            self,
            generator_outputs,
            data_3d,
            data_amass=None,
            mot_D=None
    ):
        # to reduce time dimension
        reduce = lambda x: x.contiguous().view((x.shape[0] * x.shape[1],) + x.shape[2:])  # noqa
        # flatten for weight vectors
        flatten = lambda x: x.reshape(-1)  # noqa
        # accumulate all predicted thetas from IEF
        # accumulate_thetas = lambda x: torch.cat([output['theta'] for output in x], 0)

        real_2d = data_3d['kp_2d']

        if data_amass is not None:
            sample_amass_count = data_amass['kp_3d'].shape[0]
            real_amass_3d = data_amass['kp_3d']
            real_3d = torch.cat((data_3d['kp_3d'], data_amass['kp_3d']), dim=0)

            w_3d = torch.cat(
                (
                    data_3d['w_3d'].type(torch.bool),
                    data_amass['w_3d'].type(torch.bool)
                ), dim=0)
            w_smpl = torch.cat(
                (
                    data_3d['w_smpl'].type(torch.bool),
                    data_amass['w_smpl'].type(torch.bool)
                ), dim=0)
            data_3d_theta = torch.cat(
                (
                    data_3d['theta'],
                    data_amass['theta']
                ), dim=0)
        else:
            sample_amass_count = 0
            real_3d = data_3d['kp_3d']

            w_3d = data_3d['w_3d'].type(torch.bool)
            w_smpl = data_3d['w_smpl'].type(torch.bool)
            data_3d_theta = data_3d['theta']

            real_global_transl = data_3d['transl']
            transl_mask = data_3d['w_transl']
            real_contact = data_3d['contact']

        seq_len = real_2d.shape[1]

        # real_contact = torch.cat(
        #     (
        #         data_3d['contact'],
        #         data_amass['contact']
        #     ), dim=0)
        # real_global_transl = data_amass['transl']

        # real_transl = data_3d['transl'][:, select_frame, :]

        real_2d = reduce(real_2d)
        real_3d = reduce(real_3d)
        data_3d_theta = reduce(data_3d_theta)
        real_contact = reduce(real_contact)

        # total_predict_thetas = accumulate_thetas(generator_outputs)

        if isinstance(generator_outputs, list):
            preds = generator_outputs[-1]
        else:
            preds = generator_outputs
        pred_j3d = preds['kp_3d']
        pred_theta = preds['theta']
        pred_rotmat = preds['rotmat']

        # transl_loss = 'global_transl' in preds.keys() or True
        transl_loss = True and not self.debug
        if transl_loss:
            pred_global_transl = preds['global_transl']

        contact_loss = 'contact' in preds.keys()
        if contact_loss:
            pred_contact = preds['contact']
            pred_contact = reduce(pred_contact)
        # pred_transl = preds['transl'][sample_2d_count:]

        # theta_size = pred_theta.shape[:2]

        pred_theta = reduce(pred_theta)
        pred_rotmat = reduce(pred_rotmat)
        # pred_j2d = reduce(preds['kp_2d'][:-sample_amass_count])
        pred_j2d = reduce(preds['kp_2d'])
        pred_j3d = reduce(pred_j3d)

        # w_time = torch.ones_like(w_smpl) * 0.1
        # w_time[:, last_frame] = 1

        w_3d = flatten(w_3d)
        w_smpl = flatten(w_smpl)

        pred_theta = pred_theta[w_smpl]
        pred_j3d = pred_j3d[w_3d]
        data_3d_theta = data_3d_theta[w_smpl]
        real_3d = real_3d[w_3d]

        # Generator Loss
        loss_kp_2d = self.keypoint_loss(pred_j2d, real_2d, openpose_weight=1., gt_weight=1.) * self.e_loss_weight
        loss_kp_3d = self.keypoint_3d_loss(pred_j3d, real_3d)

        loss_accel = self.accel_3d_loss(pred_j3d, real_3d)
        loss_kp_3d = loss_kp_3d * self.e_3d_loss_weight
        # loss_accel_3d = self.accel_3d_loss(pred_accel, real_accel) * 100 #self.e_3d_loss_weight
        # loss_attention = self.attetion_loss(pred_scores)

        real_shape, pred_shape = data_3d_theta[:, 75:], pred_theta[:, 75:]
        real_pose, pred_pose = data_3d_theta[:, 3:75], pred_theta[:, 3:75]

        loss_dict = {
            'loss_kp_2d': loss_kp_2d,
            'loss_kp_3d': loss_kp_3d,
        }
        info_dict = {}

        if 'residual_global_ddt' in preds.keys():
            residual_global_ddt = preds['residual_global_ddt']
            loss_dict['residual_ddt'] = 10 * torch.mean(residual_global_ddt ** 2)

        if transl_loss:
            # align first frame
            pred_global_transl = pred_global_transl - pred_global_transl[:, [0], :]
            real_global_transl = real_global_transl - real_global_transl[:, [0], :]

            pred_global_transl = pred_global_transl * transl_mask
            real_global_transl = real_global_transl * transl_mask

            v_g_transl = pred_global_transl[:, 1:, :] - pred_global_transl[:, :-1, :]
            a_g_transl = v_g_transl[:, 1:, :] - v_g_transl[:, :-1, :]

            v_real_transl = real_global_transl[:, 1:, :] - real_global_transl[:, :-1, :]
            a_real_transl = v_real_transl[:, 1:, :] - v_real_transl[:, :-1, :]

            loss_g_transl = torch.mean(torch.abs(pred_global_transl - real_global_transl))
            loss_a_g_transl = torch.mean(torch.abs(a_g_transl - a_real_transl))
            loss_dict['loss_global_transl'] = self.global_transl_weight * loss_g_transl * 1
            loss_dict['loss_acc_global_transl'] = self.global_transl_weight * loss_a_g_transl

        loss_dict['loss_acc'] = self.e_3d_accel_loss_weight * loss_accel

        if contact_loss and not self.debug:
            loss_contact = self.contact_loss(pred_contact, real_contact) * self.contact_weight  # * self.tau_weight
            loss_contact_entropy = self.contact_entropy(pred_contact) * self.contact_weight  # * self.tau_weight
            # loss_contact = self.contact_focal_loss(pred_contact, real_contact) * self.contact_weight  # * self.tau_weight
            loss_dict['loss_contact'] = loss_contact
            loss_dict['loss_contact_entropy'] = loss_contact_entropy * 1

            contact_acc = self.calc_contact_acc(pred_contact, real_contact)
            info_dict['contact_acc'] = contact_acc

        if pred_theta.shape[0] > 0:
            loss_pose, loss_shape = self.smpl_losses(pred_rotmat, pred_shape, real_pose, real_shape)
            loss_shape = loss_shape * self.e_shape_loss_weight
            loss_pose = loss_pose * self.e_pose_loss_weight
            loss_dict['loss_shape'] = loss_shape
            loss_dict['loss_pose'] = loss_pose

            # loss_smooth_vel, loss_smooth_acc = self.smooth_losses(pred_pose, real_pose)

            # loss_dict['loss_smooth_vel'] = self.vel_reg_weight * loss_smooth_vel
            # loss_dict['loss_smooth_acc'] = self.acc_reg_weight * loss_smooth_acc

        if mot_D is not None and not self.debug:
            pred_global_motion = preds['global_motion']
            loss_mot_disc = self.enc_loss(mot_D(pred_global_motion))
            loss_dict['loss_e_disc'] = loss_mot_disc * self.d_motion_loss_weight

        gen_loss = torch.stack(list(loss_dict.values())).sum()
        if torch.sum(torch.isnan(gen_loss)) > 0:
            print(loss_dict)
            raise Exception
        return gen_loss, loss_dict, info_dict

    def calc_contact_acc(self, pred_contact, gt_contact):
        gt_mask = gt_contact[:, :, 1]
        gt_label = gt_contact[:, :, 0].float()
        pred_label = (pred_contact > 0.5).float()

        correct = (pred_label[gt_mask > 0.5] == gt_label[gt_mask > 0.5]).float().sum()
        total_num = (gt_mask > 0.5).float().sum()
        acc = correct / total_num
        return acc

    def contact_loss(self, pred_contact, gt_contact):
        gt_mask = gt_contact[:, :, 1]
        gt_label = gt_contact[:, :, 0]
        # use sample
        loss = - gt_label * torch.log(pred_contact + 1e-6) - (1 - gt_label) * torch.log(1 - pred_contact + 1e-6)
        # loss = - gt_label * torch.log(pred_contact) - (1 - gt_label) * torch.log(1 - pred_contact)
        loss = loss * gt_mask
        return loss.mean()

    def contact_entropy(self, pred_contact):
        loss = - pred_contact * torch.log(pred_contact + 1e-6) - (1 - pred_contact) * torch.log(1 - pred_contact + 1e-6)
        return loss.mean()

    def transl_loss(self, gt_transl, pred_transl):
        weight = gt_transl[:, :, [3]]
        gt_transl = gt_transl[:, :, :3]

        print(gt_transl.shape, pred_transl.shape)

        return torch.mean(weight * (gt_transl - pred_transl) ** 2)

    def accel_3d_loss(self, pred_keypoints_3d, gt_keypoints_3d):
        pred_keypoints_3d = pred_keypoints_3d[:, 25:39, :]
        gt_keypoints_3d = gt_keypoints_3d[:, 25:39, :]

        if len(gt_keypoints_3d) > 0:
            gt_pelvis = (gt_keypoints_3d[:, 2, :] + gt_keypoints_3d[:, 3, :]) / 2
            gt_keypoints_3d = gt_keypoints_3d - gt_pelvis[:, None, :]
            pred_pelvis = (pred_keypoints_3d[:, 2, :] + pred_keypoints_3d[:, 3, :]) / 2
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis[:, None, :]

            seqlen = 16
            pred_keypoints_3d = pred_keypoints_3d.reshape(-1, seqlen, 14, 3)
            gt_keypoints_3d = gt_keypoints_3d.reshape(-1, seqlen, 14, 3)

            vel_pred = pred_keypoints_3d[:, 1:, :, :] - pred_keypoints_3d[:, :-1, :, :]
            accel_pred = vel_pred[:, 1:, :, :] - vel_pred[:, :-1, :, :]

            vel_gt = gt_keypoints_3d[:, 1:, :, :] - gt_keypoints_3d[:, :-1, :, :]
            accel_gt = vel_gt[:, 1:, :, :] - vel_gt[:, :-1, :, :]

            vel_loss = self.criterion_accel(vel_pred, vel_gt).mean()
            accel_loss = self.criterion_accel(accel_pred, accel_gt).mean()
            return vel_loss + accel_loss
        else:
            return torch.FloatTensor(1).fill_(0.).to(self.device)

    def keypoint_loss(self, pred_keypoints_2d, gt_keypoints_2d, openpose_weight, gt_weight):
        """
        Compute 2D reprojection loss on the keypoints.
        The loss is weighted by the confidence.
        The available keypoints are different for each dataset.
        """
        conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
        conf[:, :25] *= openpose_weight
        conf[:, 25:] *= gt_weight
        loss = (conf * self.criterion_keypoints(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])).mean()
        return loss

    def keypoint_3d_loss(self, pred_keypoints_3d, gt_keypoints_3d):
        """
        Compute 3D keypoint loss for the examples that 3D keypoint annotations are available.
        The loss is weighted by the confidence.
        """
        pred_keypoints_3d = pred_keypoints_3d[:, 25:39, :]
        gt_keypoints_3d = gt_keypoints_3d[:, 25:39, :]
        # conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
        # gt_keypoints_3d = gt_keypoints_3d[:, :, :-1].clone()
        # gt_keypoints_3d = gt_keypoints_3d
        # conf = conf
        pred_keypoints_3d = pred_keypoints_3d
        if len(gt_keypoints_3d) > 0:
            gt_pelvis = (gt_keypoints_3d[:, 2, :] + gt_keypoints_3d[:, 3, :]) / 2
            gt_keypoints_3d = gt_keypoints_3d - gt_pelvis[:, None, :]
            pred_pelvis = (pred_keypoints_3d[:, 2, :] + pred_keypoints_3d[:, 3, :]) / 2
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis[:, None, :]
            # print(conf.shape, pred_keypoints_3d.shape, gt_keypoints_3d.shape)
            # return (conf * self.criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d)).mean()
            return self.criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d).mean()
        else:
            return torch.FloatTensor(1).fill_(0.).to(self.device)

    def smpl_losses(self, pred_rotmat, pred_betas, gt_pose, gt_betas):
        # pred_rotmat_valid = batch_rodrigues(pred_rotmat.reshape(-1, 3)).reshape(-1, 24, 3, 3)
        pred_rotmat_valid = pred_rotmat
        gt_rotmat_valid = batch_rodrigues(gt_pose.reshape(-1, 3)).reshape(-1, 24, 3, 3)
        pred_betas_valid = pred_betas
        gt_betas_valid = gt_betas
        if len(pred_rotmat_valid) > 0:
            loss_regr_pose = self.criterion_regr(pred_rotmat_valid, gt_rotmat_valid)
            # loss_regr_pose = self.rotation_loss(pred_rotmat_valid, gt_rotmat_valid)
            loss_regr_betas = self.criterion_regr(pred_betas_valid, gt_betas_valid)
        else:
            loss_regr_pose = torch.FloatTensor(1).fill_(0.).to(self.device)
            loss_regr_betas = torch.FloatTensor(1).fill_(0.).to(self.device)
        return loss_regr_pose, loss_regr_betas

    def rotation_loss(self, pred_R, gt_R):
        pred_R = pred_R.reshape(-1, 3, 3)
        gt_R = gt_R.reshape(-1, 3, 3)
        R = pred_R.matmul(gt_R.transpose(1, 2))
        residual_aa = rotation_matrix_to_angle_axis(R)

        loss = torch.mean(residual_aa ** 2)
        return loss


def batch_encoder_disc_l2_loss(disc_value):
    '''
        Inputs:
            disc_value: N x 1
    '''
    k = disc_value.shape[0]
    return torch.sum((disc_value - 1.0) ** 2) * 1.0 / k
