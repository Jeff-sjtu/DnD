"""3D dataset, developed based on VIBE."""
import math
import os
import os.path as osp

import joblib
import numpy as np
import torch
import torch.utils.data as data
from dnd.models.smpl.lbs import rot_mat_to_euler, rot_mat_to_euler_T, euler_to_rot_mat2, quat2mat
from dnd.utils.geometry import batch_rodrigues, rotation_matrix_to_angle_axis
from dnd.utils.img_utils import (get_video_crop, get_single_image_crop,
                                            normalize_2d_kp, split_into_chunks,
                                            transfrom_keypoints, transfrom_keypoints24, trans_point2d)
from dnd.utils.kp_utils import convert_kps
import cv2
import random


def quaternion_to_aa(quaternion):
    '''
    input quaternion: [B, 96]
    return axis-angle: [B, 72]
    '''
    batch_size = quaternion.shape[0]
    quaternion = quaternion.reshape(batch_size, 24, 4)
    angle = np.arccos(quaternion[:, :, 0:1]) * 2
    axis = quaternion[:, :, 1:] / \
        np.linalg.norm(quaternion[:, :, 1:], axis=2, keepdims=True)

    aa = angle * axis
    aa = aa.reshape(batch_size, -1)
    return aa


class Dataset3D(data.Dataset):
    def __init__(self, cfg, ann_file, overlap=0, train=True, dataset_name=None):
        super().__init__()
        self.root = cfg['ROOT']
        self._cfg = cfg
        self._preset_cfg = cfg['PRESET']
        self.use_video = getattr(cfg['PRESET'], 'VIDEO', False)
        self.is_train = train

        self._ann_file = os.path.join(
            self.root, f'{ann_file}'
        )

        if 'nosmpl' in ann_file:
            self.no_smpl = True
        else:
            self.no_smpl = False
        self.dataset_name = dataset_name
        self.seqlen = self._preset_cfg.SEQLEN
        self.overlap = overlap
        self.stride = int(self.seqlen * (1 - self.overlap) + 0.5)
        self.db = self.load_db()

        self.debug = False

        filtered = (self.dataset_name == '3dpw')
        self.vid_indices = split_into_chunks(self.db, self.seqlen, self.stride, filtered)

    def __len__(self):
        return len(self.vid_indices)

    def __getitem__(self, index):
        return self.get_single_item(index)

    def load_db(self):
        if osp.isfile(self._ann_file):
            db = joblib.load(self._ann_file, 'r')
        else:
            raise ValueError(f'{self._ann_file} do not exists.')

        print(f'Loaded {self._ann_file}.')
        return db

    def get_sequence(self, start_index, end_index, data):
        if start_index != end_index:
            return data[start_index:end_index + 1].copy()
        else:
            return data[start_index:start_index + 1].copy().repeat(self.seqlen, axis=0)

    def get_single_item(self, index):
        start_index, end_index = self.vid_indices[index]

        if self.dataset_name == '3dpw':
            if self.is_train and 'joints2D_14' in self.db.keys():
                kp_2d = convert_kps(self.get_sequence(start_index, end_index, self.db['joints2D_14']), src='common', dst='spin')
            else:
                kp_2d = convert_kps(self.get_sequence(start_index, end_index, self.db['joints2D']), src='common', dst='spin')
            kp_3d = self.get_sequence(start_index, end_index, self.db['joints3D'])
        elif self.dataset_name == 'h36m':
            kp_2d = self.get_sequence(start_index, end_index, self.db['joints2D'])
            if self.is_train:
                # kp_3d = self.get_sequence(start_index, end_index, self.db['joints3D_49'])
                kp_xyz17 = self.get_sequence(start_index, end_index, self.db['xyz_17'])
                kp_xyz17 = convert_kps(kp_xyz17, src='h36m', dst='spin')
                kp_3d = kp_xyz17
            else:
                kp_3d = convert_kps(self.get_sequence(start_index, end_index, self.db['joints3D']), src='spin', dst='common')
                kp_xyz17 = self.get_sequence(start_index, end_index, self.db['xyz_17'])
                kp_xyz17 = convert_kps(kp_xyz17, src='h36m', dst='common')
                kp_3d = kp_xyz17
        elif self.dataset_name == 'h36m_17':
            kp_2d = self.get_sequence(start_index, end_index, self.db['joints2D'])
            kp_3d = self.get_sequence(start_index, end_index, self.db['xyz_17'])

        kp_2d_tensor = np.zeros((self.seqlen, 49, 3), dtype=np.float32)
        if self.is_train:
            nj = 49
        else:
            if self.dataset_name == 'mpii3d' or self.dataset_name == 'h36m_17':
                nj = 17
            else:
                nj = 14
        kp_3d_tensor = np.zeros((self.seqlen, nj, 3), dtype=np.float32)

        if self.dataset_name == '3dpw':
            # f = np.array([1961, 1969])
            pose = self.get_sequence(start_index, end_index, self.db['pose'])
            shape = self.get_sequence(start_index, end_index, self.db['shape'])

            theta = pose.reshape(self.seqlen, 24, 3)
            angle = np.linalg.norm(theta, axis=2, keepdims=True)
            axis = theta / angle
            mask = angle > math.pi
            angle[mask] = angle[mask] - 2 * math.pi
            new_theta = axis * angle

            # rotmat_1 = batch_rodrigues(torch.from_numpy(pose).reshape(-1, 3)).reshape(16, 24, 3, 3)
            # rotmat_2 = batch_rodrigues(torch.from_numpy(new_theta).reshape(-1, 3)).reshape(16, 24, 3, 3)
            # assert torch.max(torch.abs(rotmat_1 - rotmat_2)) < 1e-6
            pose = new_theta.reshape(self.seqlen, 72)

            # if torch.max(torch.abs(dq_dt2)) > 500:
            #     return self.get_single_item(index + 8)

            w_smpl = torch.ones(self.seqlen).float()
            w_3d = torch.ones(self.seqlen).float()
            contact = torch.zeros(self.seqlen, 6, 2).float()

            hybrik_quat = self.get_sequence(start_index, end_index, self.db['thetas'])
            hybrik_betas = self.get_sequence(start_index, end_index, self.db['betas'])
            # hybrik_aa = quaternion_to_aa(hybrik_quat.reshape(self.seqlen, 24 * 4)).reshape(self.seqlen, 24 * 3)

            hybrik_quat = torch.from_numpy(hybrik_quat)
            # hybrik_rotmat = quat2mat(hybrik_quat.reshape(-1, 4)).reshape(self.seqlen, 24 * 9)
            hybrik_rotmat = hybrik_quat.reshape(self.seqlen, 24 * 9)

            hybrik_rotmat = hybrik_rotmat.numpy()
            hybrik_transl = self.get_sequence(start_index, end_index, self.db['transl'])

            # hybrik_pred = np.concatenate((hybrik_transl, hybrik_aa, hybrik_betas), axis=1)
            hybrik_pred = np.concatenate((hybrik_transl, hybrik_rotmat, hybrik_betas), axis=1)
        elif self.dataset_name == 'h36m' or self.dataset_name == 'h36m_17':
            pose = self.get_sequence(start_index, end_index, self.db['pose'])
            shape = self.get_sequence(start_index, end_index, self.db['shape'])

            w_smpl = torch.ones(self.seqlen).float()
            w_3d = torch.ones(self.seqlen).float()

            if self.is_train:
                contact = self.get_sequence(start_index, end_index, self.db['contact'])
                contact = torch.from_numpy(contact.copy()).float()
            else:
                contact = torch.zeros(self.seqlen, 6, 2).float()

            hybrik_quat = self.get_sequence(start_index, end_index, self.db['thetas'])
            hybrik_betas = self.get_sequence(start_index, end_index, self.db['betas'])

            hybrik_rotmat = hybrik_quat.reshape(self.seqlen, 24 * 9)

            hybrik_transl = self.get_sequence(start_index, end_index, self.db['transl'])

            hybrik_pred = np.concatenate((hybrik_transl, hybrik_rotmat, hybrik_betas), axis=1)
        elif self.dataset_name == 'infer':
            kp_2d = np.zeros((self.seqlen, 49, 3), dtype=np.float32)
            kp_3d = np.zeros((self.seqlen, 14, 3), dtype=np.float32)

            theta = np.zeros((self.seqlen, 24, 3))
            contact = torch.zeros(self.seqlen, 6, 2).float()
            pose = np.zeros((self.seqlen, 72))
            shape = np.zeros((self.seqlen, 10))

            w_smpl = torch.zeros(self.seqlen).float()
            w_3d = torch.zeros(self.seqlen).float()

            hybrik_quat = self.get_sequence(start_index, end_index, self.db['thetas'])
            hybrik_betas = self.get_sequence(start_index, end_index, self.db['betas'])
            hybrik_rotmat = hybrik_quat.reshape(self.seqlen, 24 * 9)

            hybrik_transl = self.get_sequence(start_index, end_index, self.db['transl'])
            hybrik_pred = np.concatenate((hybrik_transl, hybrik_rotmat, hybrik_betas), axis=1)

        elif self.dataset_name == 'amass':
            thetas = self.get_sequence(start_index, end_index, self.db['theta'])
            kp_3d = self.get_sequence(start_index, end_index, self.db['joints3D_49'])
            root, pose, shape, _, raw_contact, _ = (
                thetas[:, :3],
                thetas[:, 3:72 + 3],
                thetas[:, 75:85],
                thetas[:, 85:85 + 72],
                thetas[:, 157:165],
                thetas[:, 165:],
            )
            contact = torch.zeros(self.seqlen, 6, 2).float()

            kp_2d_tensor = np.zeros((self.seqlen, 49, 3), dtype=np.float32)

            kp_3d_tensor = np.zeros((self.seqlen, 49, 3), dtype=np.float32)

            w_smpl = torch.ones(self.seqlen).float()
            w_3d = torch.ones(self.seqlen).float()
            # w_3d = torch.zeros(self.seqlen).float()

            # print(torch.max(test_pose), torch.max(dq_dt), torch.max(dq_dt2), '===')
            pose_theta = torch.from_numpy(pose).reshape(self.seqlen, 24, 3).float()
            # pose_rotmat = batch_rodrigues(pose_theta.reshape(-1, 3)).reshape(self.seqlen, -1, 3, 3)
            # pose_euler = rot_mat_to_euler_T(pose_rotmat.reshape(self.seqlen, 24, 3, 3)).contiguous()

            raw_contact = torch.from_numpy(raw_contact[:, [0, 1, 4, 5]]).float()
            contact[:, 2, 0] = raw_contact[:, 1]
            contact[:, 2, 1] = 1
            contact[:, 3, 0] = raw_contact[:, 3]
            contact[:, 3, 1] = 1
            contact[:, 4, 0] = raw_contact[:, 0]
            contact[:, 4, 1] = 1
            contact[:, 5, 0] = raw_contact[:, 2]
            contact[:, 5, 1] = 1

            # hybrik_pred = np.concatenate((root, pose, shape), axis=1)

            # [T, 72]
            aa_tensor = torch.from_numpy(pose).float().reshape(self.seqlen, 24, 3)
            rotmat_tensor = batch_rodrigues(aa_tensor.reshape(-1, 3)).reshape(self.seqlen, 24, 3, 3)
            euler_angle_tensor = rot_mat_to_euler_T(rotmat_tensor)
            euler_angle_tensor = euler_angle_tensor.reshape(self.seqlen, 24, 3).contiguous()
            euler_angle_tensor = euler_angle_tensor.reshape(self.seqlen, 72)

            # noise = torch.randn_like(euler_angle_tensor) * math.pi / 9
            # noise[:, :3] = noise[:, :3] / 3

            # part-based noise
            noise = torch.randn_like(euler_angle_tensor).reshape(self.seqlen, 24, 3) * math.pi / 90
            limb1 = [1, 2, 16, 17]
            limb2 = [4, 5, 18, 19]
            noise[:, limb1, :] = torch.randn_like(noise[:, limb1, :]) * math.pi / 36
            noise[:, limb2, :] = torch.randn_like(noise[:, limb2, :]) * math.pi / 9
            noise = noise.reshape(self.seqlen, 72)

            noisy_euler = euler_angle_tensor + noise
            noisy_rotmat = euler_to_rot_mat2(noisy_euler.reshape(-1, 3)).reshape(self.seqlen, 24 * 9)
            # noisy_aa = rotation_matrix_to_angle_axis(noisy_rotmat).reshape(self.seqlen, 72)

            # hybrik_pred = np.concatenate((root, noisy_aa, shape), axis=1)
            hybrik_pred = np.concatenate((root, noisy_rotmat, shape), axis=1)

            global_transl = torch.from_numpy(root).float()
            global_transl = global_transl - global_transl[[0], :]
            global_motion = torch.cat((global_transl, euler_angle_tensor), dim=1)
            w_global_gotion = np.ones((self.seqlen))
        else:
            raise NotImplementedError

        if self.dataset_name == 'amass':
            bbox = np.zeros((self.seqlen, 4))
            kp_2d = np.zeros((self.seqlen, 49, 3))
        else:
            bbox = self.get_sequence(start_index, end_index, self.db['bbox']).copy()

        if self.dataset_name == '3dpw':
            BSCALE = 1.25
        elif self.dataset_name == 'h36m' or self.dataset_name == 'h36m_17':
            BSCALE = 1
        elif self.dataset_name == 'amass':
            BSCALE = 1.25
        elif self.dataset_name == 'infer':
            BSCALE = 1
        else:
            raise NotImplementedError

        scale = BSCALE

        theta_tensor = np.zeros((self.seqlen, 85), dtype=np.float32)

        for idx in range(self.seqlen):
            # crop image and transform 2d keypoints
            new_kp_2d_i = kp_2d[idx].copy()

            new_kp_2d_i[:, :2], trans, scale = transfrom_keypoints(
                kp_2d=new_kp_2d_i[:, :2].copy(),
                center_x=bbox[idx, 0],
                center_y=bbox[idx, 1],
                width=bbox[idx, 2],
                height=bbox[idx, 3],
                patch_width=256,
                patch_height=256,
                do_augment=False,
                scale=scale
            )

            new_kp_2d_i[:, :2] = normalize_2d_kp(new_kp_2d_i[:, :2], 256)

            # theta shape (85,)
            theta = np.concatenate((np.array([1., 0., 0.]), pose[idx], shape[idx]), axis=0)

            # kp_2d_tensor[idx] = kp_2d[idx]
            kp_2d_tensor[idx] = new_kp_2d_i
            theta_tensor[idx] = theta
            kp_3d_tensor[idx] = kp_3d[idx]

        if self.dataset_name == 'mpii3d':
            kp_2d_tensor[:, :, :] = 0
        elif self.dataset_name == 'amass':
            kp_2d_tensor[:, :, :] = 0

        if self.dataset_name == 'amass':
            image_name = ['' for i in range(self.seqlen)]
            w_transl = np.ones((self.seqlen, 3))
        else:
            image_name = self.db['img_name'][start_index:end_index + 1].tolist()
            root = np.zeros((self.seqlen, 3))
            w_transl = np.zeros((self.seqlen, 3))

            global_motion = torch.zeros((self.seqlen, 75))
            w_global_gotion = np.zeros((self.seqlen))

        target = {
            'theta': torch.from_numpy(theta_tensor).float(),  # camera, pose and shape
            'kp_2d': torch.from_numpy(kp_2d_tensor).float(),  # 2D keypoints transformed according to bbox cropping
            'kp_3d': torch.from_numpy(kp_3d_tensor).float(),  # 3D keypoints
            'hybrik_pred': torch.from_numpy(hybrik_pred).float(),
            'w_smpl': w_smpl,
            'w_3d': w_3d,
            'ind': index,
            'contact': contact,
            'transl': torch.from_numpy(root).float(),
            'w_transl': torch.from_numpy(w_transl).float(),
            'vid_name': self.db['vid_name'][start_index],
            'image_name': image_name,
            'bbox': torch.from_numpy(bbox.copy()).float(),    # bbox: [c_x, c_y, w, h]
            'test_tensor': torch.arange(start_index, end_index + 1)[:, None].float(),
            'global_motion': global_motion,
            'w_global_gotion': torch.from_numpy(w_global_gotion).float()
        }

        if self.dataset_name == 'mpii3d' and not self.is_train:
            target['valid'] = torch.from_numpy(self.db['valid_i'][start_index:end_index + 1].copy())

        if self.dataset_name == 'h36m' and not self.is_train:
            target['valid'] = np.ones((self.seqlen, 1), dtype=np.float32)

        if self.dataset_name == 'h36m_smpl' and not self.is_train:
            target['valid'] = np.ones((self.seqlen, 1), dtype=np.float32)

        if self.dataset_name == 'h36m_17' and not self.is_train:
            target['valid'] = np.ones((self.seqlen, 1), dtype=np.float32)

        if self.dataset_name == '3dpw' and not self.is_train:
            target['valid'] = np.ones((self.seqlen, 1), dtype=np.float32)

            vn = self.get_sequence(start_index, end_index, self.db['vid_name']).copy()
            fi = self.get_sequence(start_index, end_index, self.db['frame_id']).copy()

            target['instance_id'] = [f'{v}/{f}'for v, f in zip(vn, fi)]

        return target
