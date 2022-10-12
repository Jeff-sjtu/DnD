import os.path as osp
import shutil
from collections import defaultdict

import numpy as np
import torch
from torch.nn.utils import clip_grad
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .utils.eval_utils import (batch_compute_similarity_transform_torch,
                               compute_accel_T,
                               compute_error_accel_T,
                               compute_error_vel_T, compute_error_verts)
from .utils.kp_utils import convert_kps
from .utils.utils import AverageMeter, move_dict_to_device


class Trainer(object):
    def __init__(
        self,
        args,
        cfg,
        logger,
        data_loaders,
        model,
        optimizer,
        criterion,
        lr_schd_type,
        lr_scheduler=None,
        performance_type='min',
        motion_disc=None
    ):

        self.train_3d_loader, self.valid_loader = data_loaders

        self.train_3d_iter = iter(self.train_3d_loader)

        self.model = model
        self.optimizer = optimizer

        if motion_disc is not None:
            self.use_disc = True
            self.mot_D, self.dis_motion_optimizer, self.motion_lr_scheduler = motion_disc
            self.dis_motion_update_steps = cfg.MOT_DISCR.UPDATE_STEPS
        else:
            self.use_disc = False
            self.mot_D = None

        self.start_epoch = cfg.TRAIN.BEGIN_EPOCH
        self.end_epoch = cfg.TRAIN.END_EPOCH
        self.num_iters_per_epoch = cfg.TRAIN.NUM_ITERS_PER_EPOCH

        self.disc_weight = cfg.LOSS.WEIGHT.D_MOTION_LOSS_W

        self.train_with_gt_theta = getattr(cfg.DATA_PRESET, 'GT_THETA_INPUT', False)

        self.criterion = criterion
        self.lr_scheduler = lr_scheduler
        self.lr_schd_type = lr_schd_type
        self.device = args.device

        self.log_dir = './exp/{}-{}/'.format(args.exp_id, cfg.FILE_NAME)

        self.performance_type = performance_type
        self.evaluate_type = cfg.TEST.EVAL_TYPE
        self.train_global_step = 0
        self.valid_global_step = 0
        self.epoch = 0
        self.best_performance = float('inf') if performance_type == 'min' else -float('inf')

        self.evaluation_accumulators = dict.fromkeys(['pred_j3d', 'target_j3d', 'target_theta', 'pred_verts'])

        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.logger = logger

        if self.valid_loader.dataset.dataset_name == 'h36m_17':
            self.eval_dataset = 'h36m_17'
        elif 'h36m' in str(self.valid_loader.dataset):
            self.eval_dataset = 'h36m'
        else:
            self.eval_dataset = '3dpw'

        self.debug = True

    def train(self):

        losses = AverageMeter()
        kp_2d_loss = AverageMeter()
        kp_3d_loss = AverageMeter()
        contact_acc = AverageMeter()
        g_transl_loss = AverageMeter()

        # timer = {
        #     'data': 0,
        #     'forward': 0,
        #     'loss': 0,
        #     'backward': 0,
        #     'batch': 0,
        # }

        self.model.train()

        # start = time.time()

        current_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        self.logger.info(f'############# Starting Epoch {self.epoch} | LR: {current_lr} #############')

        tqdm_loader = tqdm(range(self.num_iters_per_epoch), dynamic_ncols=True)
        self.model.epoch = self.epoch

        for i in tqdm_loader:
            target_3d = None

            try:
                target_3d = next(self.train_3d_iter)
            except StopIteration:
                self.train_3d_iter = iter(self.train_3d_loader)
                target_3d = next(self.train_3d_iter)

            move_dict_to_device(target_3d, self.device)

            self.optimizer.zero_grad()
            inp = target_3d['hybrik_pred']

            preds = self.model(inp)

            # timer['forward'] = time.time() - start
            # start = time.time()

            loss, loss_dict, info_dict = self.criterion(
                generator_outputs=preds,
                data_3d=target_3d,
                mot_D=self.mot_D
            )

            # timer['loss'] = time.time() - start
            # start = time.time()

            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()

            if isinstance(inp, tuple):
                inp = inp[0]
            losses.update(loss.item(), inp.shape[0])
            kp_2d_loss.update(loss_dict['loss_kp_2d'].item(), inp.shape[0])
            kp_3d_loss.update(loss_dict['loss_kp_3d'].item(), inp.shape[0])

            if not self.debug:
                g_transl_loss.update(loss_dict['loss_global_transl'].item(), inp.shape[0])
                contact_acc.update(info_dict['contact_acc'].item(), inp.shape[0])

            # timer['backward'] = time.time() - start
            # timer['batch'] = timer['data'] + timer['forward'] + timer['loss'] + timer['backward']
            # start = time.time()

            tqdm_loader.set_description(
                'loss: {loss:.2f} | L2d: {twod:.2f} | L3d: {threed:.2f} | Cacc: {cacc:.2f}'.format(
                    loss=losses.avg,
                    twod=kp_2d_loss.avg,
                    threed=kp_3d_loss.avg,
                    cacc=contact_acc.avg * 100)
            )
            for k, v in loss_dict.items():
                self.writer.add_scalar('train_loss/' + k, v, global_step=self.train_global_step)

            self.writer.add_scalar('train_loss/loss', loss.item(), global_step=self.train_global_step)

            self.train_global_step += 1

        tqdm_loader.close()

    def fit(self):

        for epoch in range(self.start_epoch, self.end_epoch):
            self.epoch = epoch

            self.train()
            # if (epoch + 1) * self.args.snapshot == 0:
            # performance = self.evaluate()
            self.validate()
            if self.evaluate_type == 'all':
                performance = self.evaluate_all()
            else:
                performance = self.evaluate_new()

            if self.lr_schd_type == 'dynamic':
                self.lr_scheduler.step(performance)
            elif self.lr_schd_type == 'multi-step':
                self.lr_scheduler.step()
            else:
                self.lr_scheduler.step()

            # log the learning rate
            for param_group in self.optimizer.param_groups:
                self.writer.add_scalar('lr/gen_lr', param_group['lr'], global_step=self.epoch)

            self.logger.info(f'Epoch {epoch+1} performance: {performance:.4f}')

            self.save_model(performance, epoch)

        self.writer.close()

    def save_model(self, performance, epoch):
        save_dict = {
            'epoch': epoch,
            'gen_state_dict': self.model.state_dict(),
            'performance': performance,
            'gen_optimizer': self.optimizer.state_dict(),
        }

        filename = osp.join(self.log_dir, 'checkpoint.pth.tar')
        torch.save(save_dict, filename)

        if self.performance_type == 'min':
            is_best = performance < self.best_performance
        else:
            is_best = performance > self.best_performance

        if is_best:
            self.logger.info('Best performance achived, saving it!')
            self.best_performance = performance
            shutil.copyfile(filename, osp.join(self.log_dir, 'model_best.pth.tar'))

            with open(osp.join(self.log_dir, 'best.txt'), 'w') as f:
                f.write(str(float(performance)))

    def validate(self):
        self.model.eval()
        self.evaluation_accumulators = dict()

        J_regressor = torch.from_numpy(np.load('./model_files/J_regressor_h36m.npy')).float()

        tqdm_loader = tqdm(self.valid_loader, dynamic_ncols=True)

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

                # convert to 14 keypoint format for evaluation

                if self.eval_dataset == 'mpii3d':
                    n_kp = 17
                    pred_j3d = preds['kp_3d'].view(-1, 49, 3).cpu().numpy()
                else:
                    n_kp = preds['kp_3d'].shape[-2]
                    pred_j3d = preds['kp_3d'].view(-1, n_kp, 3).cpu().numpy()

                if self.eval_dataset == 'mpii3d':
                    pred_j3d = convert_kps(pred_j3d.reshape(-1, 49, 3), src='spin', dst='mpii3d_test')
                    pred_j3d = pred_j3d.reshape(-1, 17, 3)

                # n_kp = preds['kp_3d'].shape[-2]
                # pred_j3d = preds['kp_3d'].view(-1, n_kp, 3).cpu().numpy()
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
                elif self.evaluate_type == 'mid':
                    seq_len = target['kp_3d'].shape[1]
                    mid_frame = int(seq_len / 2)
                    target_j3d = target['kp_3d'][:, [mid_frame], :, :].view(-1, n_kp, 3).cpu().numpy()
                    target_theta = target['theta'][:, [mid_frame], :].view(-1, 85).cpu().numpy()
                    target_valid = target['valid'][:, [mid_frame], :].view(-1, 1).cpu().numpy()
                elif self.evaluate_type == 'last':
                    seq_len = target['kp_3d'].shape[1]
                    last_frame = int(seq_len - 1)
                    target_j3d = target['kp_3d'][:, [last_frame], :, :].view(-1, n_kp, 3).cpu().numpy()
                    target_theta = target['theta'][:, [last_frame], :].view(-1, 85).cpu().numpy()
                # target_theta = target['theta'].view(-1, 85).cpu().numpy()

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

    def evaluate_all(self):
        full_res = defaultdict(list)

        for vid, eval_res in self.evaluation_accumulators.items():
            for k, v in eval_res.items():
                eval_res[k] = np.vstack(v)

            pred_j3ds = eval_res['pred_j3d']
            target_j3ds = eval_res['target_j3d']
            target_valids = eval_res['target_valid']

            pred_j3ds = torch.from_numpy(pred_j3ds).float()
            target_j3ds = torch.from_numpy(target_j3ds).float()
            target_valids = torch.from_numpy(target_valids).float()

            if target_valids.sum() < 1:
                continue

            if self.eval_dataset == 'mpii3d':
                pred_pelvis = pred_j3ds[:, :, [-3], :]
                target_pelvis = target_j3ds[:, :, [-3], :]
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

            # pred_pelvis = (pred_j3ds[:, :, [2], :] + pred_j3ds[:, :, [3], :]) / 2.0
            # target_pelvis = (target_j3ds[:, :, [2], :] + target_j3ds[:, :, [3], :]) / 2.0

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

            # pred_j3ds_flat = pred_j3ds.reshape(-1, *pred_j3ds.shape[2:])
            # target_j3ds_flat = target_j3ds.reshape(-1, *target_j3ds.shape[2:])

            # errors = torch.sqrt(((pred_j3ds_flat - target_j3ds_flat) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
            # S1_hat = batch_compute_similarity_transform_torch(pred_j3ds_flat, target_j3ds_flat)
            # errors_pa = torch.sqrt(((S1_hat - target_j3ds_flat) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
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

            full_res['mpjpe'].append(mpjpe)
            full_res['pa_mpjpe'].append(pa_mpjpe)
            full_res['accel'].append(accel)
            full_res['gt_accel'].append(gt_accel)
            full_res['accel_err'].append(accel_err)
            full_res['vel_err'].append(vel_err)
            full_res['pve'].append(pve)

        mpjpe = np.mean(full_res['mpjpe'])
        pa_mpjpe = np.mean(full_res['pa_mpjpe'])
        accel = np.mean(full_res['accel'])
        gt_accel = np.mean(full_res['gt_accel'])
        accel_err = np.mean(full_res['accel_err'])
        vel_err = np.mean(full_res['vel_err'])
        pve = np.mean(full_res['pve'])
        eval_dict = {
            'mpjpe': mpjpe,
            'pa-mpjpe': pa_mpjpe,
            'accel': accel,
            'gt_accel': gt_accel,
            'pve': pve,
            'accel_err': accel_err,
            'vel_err': vel_err
        }

        for k, v in eval_dict.items():
            self.writer.add_scalar(f'error/{k}', v, global_step=self.epoch)

        log_str = ' '.join([f'{k.upper()}: {v:.4f},'for k, v in eval_dict.items()])
        self.logger.info(log_str)

        return pa_mpjpe


def batch_adv_disc_l2_loss(real_disc_value, fake_disc_value):
    '''
        Inputs:
            disc_value: N x 25
    '''
    ka = real_disc_value.shape[0]
    kb = fake_disc_value.shape[0]
    lb, la = torch.sum(fake_disc_value ** 2) / kb, torch.sum((real_disc_value - 1) ** 2) / ka
    return la, lb, la + lb
