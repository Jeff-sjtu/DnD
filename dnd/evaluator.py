"""Evaluator, developed based on VIBE."""

from collections import defaultdict

import joblib
import numpy as np
import torch
from tqdm import tqdm

from .utils.eval_utils import (batch_compute_similarity_transform_torch,
                               compute_accel, compute_accel_T,
                               compute_error_accel, compute_error_accel_T,
                               compute_error_vel_T, compute_error_verts)
from .utils.kp_utils import convert_kps
from .utils.utils import move_dict_to_device

J49_TO_MPII3D = list(range(25, 39)) + [39, 41, 43]


class Evaluator():
    def __init__(
            self,
            args,
            cfg,
            logger,
            test_loader,
            model,
    ):
        self.test_db = test_loader.dataset.db
        if test_loader.dataset.dataset_name == 'h36m_17':
            self.eval_dataset = 'h36m_17'
        elif 'h36m' in str(test_loader.dataset):
            self.eval_dataset = 'h36m'
        else:
            self.eval_dataset = '3dpw'
        self.test_loader = test_loader
        self.model = model

        self.evaluation_accumulators = dict.fromkeys(['pred_j3d', 'target_j3d', 'target_theta', 'pred_verts'])
        self.evaluate_type = cfg.TEST.EVAL_TYPE

        self.log_dir = './exp/{}-{}/'.format(args.exp_id, cfg.FILE_NAME)
        self.logger = logger
        self.device = args.device

    def validate(self):
        self.model.eval()

        J_regressor = torch.from_numpy(np.load('./model_files/J_regressor_h36m.npy')).float()

        tqdm_loader = tqdm(self.test_loader, dynamic_ncols=True)

        for i, target in enumerate(tqdm_loader):
            move_dict_to_device(target, self.device)

            inp = target['hybrik_pred']
            batch = len(inp)

            with torch.no_grad():
                if self.eval_dataset == 'mpii3d':
                    J_regressor = None

                if self.eval_dataset == 'h36m_17':
                    test_nkpts = 17
                else:
                    test_nkpts = 14

                preds = self.model(inp, J_regressor=J_regressor, test_nkpts=test_nkpts)

                if isinstance(preds, list):
                    preds = preds[-1]
                target_file_name = target['image_name']

                if self.eval_dataset == 'mpii3d':
                    n_kp = 17
                    pred_j3d = preds['kp_3d'].view(-1, 49, 3).cpu().numpy()
                else:
                    n_kp = preds['kp_3d'].shape[-2]
                    pred_j3d = preds['kp_3d'].view(-1, n_kp, 3).cpu().numpy()

                if self.eval_dataset == 'mpii3d':
                    pred_j3d = convert_kps(pred_j3d.reshape(-1, 49, 3), src='spin', dst='mpii3d_test')
                    pred_j3d = pred_j3d.reshape(-1, 17, 3)

                pred_verts = preds['verts'].view(-1, 6890, 3).cpu().numpy()
                transl = preds['transl'].view(-1, 1, 3).cpu().numpy()
                if not self.model.use_smplx:
                    pred_verts = pred_verts - transl

                if self.evaluate_type == 'all':
                    seqlen = target['kp_3d'].shape[1]
                    pred_verts = pred_verts.reshape(-1, seqlen, 6890, 3)
                    pred_j3d = pred_j3d.reshape(-1, seqlen, *pred_j3d.shape[1:])

                    target_j3d = target['kp_3d'].view(-1, seqlen, n_kp, 3).cpu().numpy()
                    target_theta = target['theta'].view(-1, seqlen, 85).cpu().numpy()
                    target_valid = target['valid'].view(-1, seqlen, 1).cpu().numpy()
                    target_test = target['test_tensor'].view(-1, seqlen, 1, 1).cpu().numpy()
                elif self.evaluate_type == 'mid':
                    seq_len = target['kp_3d'].shape[1]
                    mid_frame = int(seq_len / 2)
                    target_j3d = target['kp_3d'][:, [mid_frame], :, :].view(-1, n_kp, 3).cpu().numpy()
                    target_theta = target['theta'][:, [mid_frame], :].view(-1, 85).cpu().numpy()
                    target_valid = target['valid'][:, [mid_frame], :].view(-1, 1).cpu().numpy()
                    target_test = target['test_tensor'][:, [mid_frame], :].view(-1, 1, 1).cpu().numpy()
                elif self.evaluate_type == 'last':
                    seq_len = target['kp_3d'].shape[1]
                    last_frame = int(seq_len - 1)
                    target_j3d = target['kp_3d'][:, [last_frame], :, :].view(-1, n_kp, 3).cpu().numpy()
                    target_theta = target['theta'][:, [last_frame], :].view(-1, 85).cpu().numpy()
                # target_theta = target['theta'].view(-1, 85).cpu().numpy()

                target_bbox = target['bbox'].cpu().numpy()
                pred_transl = preds['transl'].view(-1, seqlen, 3).cpu().numpy()
                pred_theta = preds['theta'].reshape(-1, seqlen, 85).cpu().numpy()
                pred_transl_vel = pred_transl[:, 1:, :] - pred_transl[:, :-1, :]

                # print(pred_verts.shape, target_theta.shape, pred_j3d.shape, target_j3d.shape)
                for b in range(batch):
                    vid = target['vid_name'][b]
                    if vid not in self.evaluation_accumulators.keys():
                        self.evaluation_accumulators[vid] = defaultdict(list)

                    self.evaluation_accumulators[vid]['pred_verts'].append(pred_verts[[b]])
                    self.evaluation_accumulators[vid]['target_theta'].append(target_theta[[b]])

                    self.evaluation_accumulators[vid]['pred_j3d'].append(pred_j3d[[b]])
                    self.evaluation_accumulators[vid]['target_j3d'].append(target_j3d[[b]])

                    self.evaluation_accumulators[vid]['target_valid'].append(target_valid[[b]])
                    self.evaluation_accumulators[vid]['target_test'].append(target_test[[b]])
                    image_name_list = []
                    for t in range(seqlen):
                        image_name_list.append(target_file_name[t][b])
                    self.evaluation_accumulators[vid]['image_name'].append(image_name_list)
                    self.evaluation_accumulators[vid]['transl'].append(pred_transl[[b]])

                    self.evaluation_accumulators[vid]['bbox'].append(target_bbox[[b]])

    def evaluate_all(self):
        full_res = defaultdict(list)
        dyna_res = defaultdict(list)

        for vid, eval_res in self.evaluation_accumulators.items():
            for k, v in eval_res.items():
                # [B, T, K, ...]
                eval_res[k] = np.vstack(v)

            pred_j3ds = eval_res['pred_j3d']
            target_j3ds = eval_res['target_j3d']
            target_valids = eval_res['target_valid']

            target_test = eval_res['target_test']

            # if self.eval_dataset == 'mpii3d':
            #     bs, seqlen = pred_j3ds.shape[:2]
            #     pred_j3ds = convert_kps(pred_j3ds.reshape(-1, 17, 3), src='mpii3d_test', dst='common')
            #     target_j3ds = convert_kps(target_j3ds.reshape(-1, 17, 3), src='mpii3d_test', dst='common')
            #     pred_j3ds = pred_j3ds.reshape(bs, seqlen, 14, 3)
            #     target_j3ds = target_j3ds.reshape(bs, seqlen, 14, 3)

            pred_j3ds = torch.from_numpy(pred_j3ds).float()
            target_j3ds = torch.from_numpy(target_j3ds).float()
            target_valids = torch.from_numpy(target_valids).float()

            if target_valids.sum() < 1:
                continue

            if self.eval_dataset == 'mpii3d':
                # pred_pelvis = pred_j3ds[:, :, [-3], :]
                # target_pelvis = target_j3ds[:, :, [-3], :]
                pred_pelvis = (pred_j3ds[:, :, [8], :] + pred_j3ds[:, :, [11], :]) / 2.0
                target_pelvis = (target_j3ds[:, :, [8], :] + target_j3ds[:, :, [11], :]) / 2.0

                # pred_j3ds = pred_j3ds[:, :-3, :]
                # target_j3ds = target_j3ds[:, :-3, :]

                # pred_pelvis = (pred_j3ds[:, [2], :] + pred_j3ds[:, [3], :]) / 2.0
                # target_pelvis = (target_j3ds[:, [2], :] + target_j3ds[:, [3], :]) / 2.0
            elif self.eval_dataset == 'h36m_17':
                pred_pelvis = pred_j3ds[:, :, [0], :]
                target_pelvis = target_j3ds[:, :, [0], :]
            else:
                pred_pelvis = (pred_j3ds[:, :, [2], :] + pred_j3ds[:, :, [3], :]) / 2.0
                target_pelvis = (target_j3ds[:, :, [2], :] + target_j3ds[:, :, [3], :]) / 2.0

            pred_j3ds -= pred_pelvis
            target_j3ds -= target_pelvis

            pred_j3ds_flat = pred_j3ds.reshape(-1, *pred_j3ds.shape[2:])
            target_j3ds_flat = target_j3ds.reshape(-1, *target_j3ds.shape[2:])
            target_valids_flat = target_valids.reshape(-1, *target_valids.shape[2:])
            # errors = torch.sqrt(((pred_j3ds - target_j3ds) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
            errors = torch.sqrt(((pred_j3ds_flat - target_j3ds_flat) ** 2).sum(dim=-1))
            errors = ((errors * target_valids_flat).sum() / (target_valids_flat.sum() * pred_j3ds_flat.shape[1])).cpu().numpy()

            S1_hat = batch_compute_similarity_transform_torch(pred_j3ds_flat, target_j3ds_flat)
            # errors_pa = torch.sqrt(((S1_hat - target_j3ds) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
            errors_pa = torch.sqrt(((S1_hat - target_j3ds_flat) ** 2).sum(dim=-1))
            errors_pa = ((errors_pa * target_valids_flat).sum() / (target_valids_flat.sum() * pred_j3ds_flat.shape[1])).cpu().numpy()

            pred_verts = eval_res['pred_verts']
            target_theta = eval_res['target_theta']

            pred_verts_flat = pred_verts.reshape(-1, *pred_verts.shape[2:])
            target_theta_flat = target_theta.reshape(-1, *target_theta.shape[2:])
            m2mm = 1000

            pve = np.mean(compute_error_verts(target_theta=target_theta_flat, pred_verts=pred_verts_flat)) * m2mm
            accel = np.mean(compute_accel_T(pred_j3ds)) * m2mm
            gt_accel = np.mean(compute_accel_T(target_j3ds)) * m2mm
            accel_err = np.mean(compute_error_accel_T(joints_pred=pred_j3ds, joints_gt=target_j3ds)) * m2mm
            vel_err = np.mean(compute_error_vel_T(joints_pred=pred_j3ds, joints_gt=target_j3ds)) * m2mm
            mpjpe = np.mean(errors) * m2mm
            pa_mpjpe = np.mean(errors_pa) * m2mm

            test_accel = np.mean(compute_accel_T(target_test))

            full_res['mpjpe'].append(mpjpe)
            full_res['pa_mpjpe'].append(pa_mpjpe)
            full_res['accel'].append(accel)
            full_res['gt_accel'].append(gt_accel)
            full_res['accel_err'].append(accel_err)
            full_res['vel_err'].append(vel_err)
            full_res['pve'].append(pve)

            full_res['test_accel'].append(test_accel)

        mpjpe = np.mean(full_res['mpjpe'])
        pa_mpjpe = np.mean(full_res['pa_mpjpe'])
        accel = np.mean(full_res['accel'])
        gt_accel = np.mean(full_res['gt_accel'])
        accel_err = np.mean(full_res['accel_err'])
        vel_err = np.mean(full_res['vel_err'])
        pve = np.mean(full_res['pve'])
        test_accel = np.mean(full_res['test_accel'])
        eval_dict = {
            'mpjpe': mpjpe,
            'pa-mpjpe': pa_mpjpe,
            'accel': accel,
            'gt_accel': gt_accel,
            'pve': pve,
            'accel_err': accel_err,
            'vel_err': vel_err,
            # 'test_accel': test_accel
        }

        log_str = ' '.join([f'{k.upper()}: {v:.4f},'for k, v in eval_dict.items()])
        self.logger.info(log_str)

    def run(self):
        self.evaluation_accumulators = dict()
        self.validate()
        self.evaluate_all()

