import os.path as osp

import numpy as np
import torch
import torch.nn as nn
from easydict import EasyDict as edict

from .lbs import (euler_to_rot_mat, lbs, lbs_with_d, matadd_d,
                  matmul_d, quat_to_rotmat, euler_to_rot_mat3, vec2skewmat,
                  vec2skewmat_d)
from .vertex_joint_selector import VertexJointSelector
from .vertex_ids import vertex_ids as VERTEX_IDS

try:
    import cPickle as pk
except ImportError:
    import pickle as pk


# Map joints to SMPL joints
JOINT_MAP = {
    'OP Nose': 24, 'OP Neck': 12, 'OP RShoulder': 17,
    'OP RElbow': 19, 'OP RWrist': 21, 'OP LShoulder': 16,
    'OP LElbow': 18, 'OP LWrist': 20, 'OP MidHip': 0,
    'OP RHip': 2, 'OP RKnee': 5, 'OP RAnkle': 8,
    'OP LHip': 1, 'OP LKnee': 4, 'OP LAnkle': 7,
    'OP REye': 25, 'OP LEye': 26, 'OP REar': 27,
    'OP LEar': 28, 'OP LBigToe': 29, 'OP LSmallToe': 30,
    'OP LHeel': 31, 'OP RBigToe': 32, 'OP RSmallToe': 33, 'OP RHeel': 34,
    'Right Ankle': 8, 'Right Knee': 5, 'Right Hip': 45,
    'Left Hip': 46, 'Left Knee': 4, 'Left Ankle': 7,
    'Right Wrist': 21, 'Right Elbow': 19, 'Right Shoulder': 17,
    'Left Shoulder': 16, 'Left Elbow': 18, 'Left Wrist': 20,
    'Neck (LSP)': 47, 'Top of Head (LSP)': 48,
    'Pelvis (MPII)': 49, 'Thorax (MPII)': 50,
    'Spine (H36M)': 51, 'Jaw (H36M)': 52,
    'Head (H36M)': 53, 'Nose': 24, 'Left Eye': 26,
    'Right Eye': 25, 'Left Ear': 28, 'Right Ear': 27
}
JOINT_NAMES = [
    'OP Nose', 'OP Neck', 'OP RShoulder',
    'OP RElbow', 'OP RWrist', 'OP LShoulder',
    'OP LElbow', 'OP LWrist', 'OP MidHip',
    'OP RHip', 'OP RKnee', 'OP RAnkle',
    'OP LHip', 'OP LKnee', 'OP LAnkle',
    'OP REye', 'OP LEye', 'OP REar',
    'OP LEar', 'OP LBigToe', 'OP LSmallToe',
    'OP LHeel', 'OP RBigToe', 'OP RSmallToe', 'OP RHeel',
    'Right Ankle', 'Right Knee', 'Right Hip',
    'Left Hip', 'Left Knee', 'Left Ankle',
    'Right Wrist', 'Right Elbow', 'Right Shoulder',
    'Left Shoulder', 'Left Elbow', 'Left Wrist',
    'Neck (LSP)', 'Top of Head (LSP)',
    'Pelvis (MPII)', 'Thorax (MPII)',
    'Spine (H36M)', 'Jaw (H36M)',
    'Head (H36M)', 'Nose', 'Left Eye',
    'Right Eye', 'Left Ear', 'Right Ear'
]

JOINT_IDS = {JOINT_NAMES[i]: i for i in range(len(JOINT_NAMES))}
JOINT_REGRESSOR_TRAIN_EXTRA = osp.join('model_files', 'J_regressor_extra.npy')
SMPL_MEAN_PARAMS = osp.join('model_files', 'smpl_mean_params.npz')
SMPL_MODEL_DIR = 'model_files'
H36M_TO_J17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]
H36M_TO_J14 = H36M_TO_J17[:14]


def to_tensor(array, dtype=torch.float32):
    if 'torch.tensor' not in str(type(array)):
        return torch.tensor(array, dtype=dtype)


class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)


class SMPL_layer_dynamics(nn.Module):
    NUM_JOINTS = 23
    NUM_BODY_JOINTS = 23
    NUM_BETAS = 10
    JOINT_NAMES = [
        'pelvis', 'left_hip', 'right_hip',      # 2
        'spine1', 'left_knee', 'right_knee',    # 5
        'spine2', 'left_ankle', 'right_ankle',  # 8
        'spine3', 'left_foot', 'right_foot',    # 11
        'neck', 'left_collar', 'right_collar',  # 14
        'jaw',                                  # 15
        'left_shoulder', 'right_shoulder',      # 17
        'left_elbow', 'right_elbow',            # 19
        'left_wrist', 'right_wrist',            # 21
        'left_thumb', 'right_thumb',            # 23
        # 'head', 'left_middle', 'right_middle',  # 26
        # 'left_bigtoe', 'right_bigtoe'           # 28
    ]
    LEAF_NAMES = [
        'head', 'left_middle', 'right_middle', 'left_bigtoe', 'right_bigtoe'
    ]
    CONTACT_JOINTS_NAME = [
        'left_hip', 'right_hip', 'left_ankle', 'right_ankle', 'left_foot', 'right_foot'
    ]
    root_idx_17 = 0
    root_idx_smpl = 0

    def __init__(self,
                 model_path,
                 gender='neutral',
                 dtype=torch.float32,
                 vertex_ids=None,
                 use_euler_angle=True):
        ''' SMPL model layers

        Parameters:
        ----------
        model_path: str
            The path to the folder or to the file where the model
            parameters are stored
        gender: str, optional
            Which gender to load
        '''
        super(SMPL_layer_dynamics, self).__init__()

        self.ROOT_IDX = self.JOINT_NAMES.index('pelvis')
        # self.LEAF_IDX = [self.JOINT_NAMES.index(name) for name in self.LEAF_NAMES]
        self.CONTACT_JOINTS = [self.JOINT_NAMES.index(name) for name in self.CONTACT_JOINTS_NAME]
        self.SPINE3_IDX = 9
        self.use_euler_angle = use_euler_angle

        with open(model_path, 'rb') as smpl_file:
            self.smpl_data = Struct(**pk.load(smpl_file, encoding='latin1'))

        self.gender = gender

        # dtype = torch.float64
        self.dtype = dtype

        self.faces = self.smpl_data.f

        ''' Register Buffer '''
        # Faces
        self.register_buffer('faces_tensor',
                             to_tensor(to_np(self.smpl_data.f, dtype=np.int64), dtype=torch.long))

        # The vertices of the template model, (6890, 3)
        self.register_buffer('v_template',
                             to_tensor(to_np(self.smpl_data.v_template), dtype=dtype))

        # The shape components
        # Shape blend shapes basis, (6890, 3, 10)
        self.register_buffer(
            'shapedirs',
            to_tensor(to_np(self.smpl_data.shapedirs), dtype=dtype))

        # Pose blend shape basis: 6890 x 3 x 23*9, reshaped to 6890*3 x 23*9
        num_pose_basis = self.smpl_data.posedirs.shape[-1]
        # 23*9 x 6890*3
        posedirs = np.reshape(self.smpl_data.posedirs, [-1, num_pose_basis]).T
        self.register_buffer('posedirs',
                             to_tensor(to_np(posedirs), dtype=dtype))

        # Vertices to Joints location (23 + 1, 6890)
        self.register_buffer(
            'J_regressor',
            to_tensor(to_np(self.smpl_data.J_regressor), dtype=dtype))
        # Vertices to Human3.6M Joints location (17, 6890)
        h36m_jregressor = np.load('./model_files/J_regressor_h36m.npy')
        self.register_buffer(
            'J_regressor_h36m',
            to_tensor(to_np(h36m_jregressor), dtype=dtype))
        # Vertices to extra joints
        J_regressor_extra = np.load(JOINT_REGRESSOR_TRAIN_EXTRA)
        self.register_buffer('J_regressor_extra', torch.tensor(
            J_regressor_extra, dtype=dtype))
        joints = [JOINT_MAP[i] for i in JOINT_NAMES]
        self.register_buffer('joint_map', torch.tensor(joints, dtype=torch.long))

        # indices of parents for each joints
        parents = torch.zeros(len(self.JOINT_NAMES), dtype=torch.long)
        parents[:(self.NUM_JOINTS + 1)] = to_tensor(to_np(self.smpl_data.kintree_table[0])).long()
        parents[0] = -1

        self.register_buffer(
            'children_map',
            self._parents_to_children(parents))
        # (24,)
        self.register_buffer('parents', parents)

        # (6890, 23 + 1)
        self.register_buffer('lbs_weights',
                             to_tensor(to_np(self.smpl_data.weights), dtype=dtype))

        # print(self.parents)
        # print(self.children_map)
        self.parents_with_leaf = self.parents.tolist() + [15, 22, 23, 10, 11]
        self.children_with_leaf = self._parents_to_children_with_leaf(self.parents_with_leaf)

        if vertex_ids is None:
            # SMPL and SMPL-H share the same topology, so any extra joints can
            # be drawn from the same place
            vertex_ids = VERTEX_IDS['smplh']

        leaf_vertex_ids = {
            'head': 411,
            'left_middle': 2445,
            'right_middle': 5905,
            'left_bigtoe': 3216,
            'right_bigtoe': 6617,
        }
        self.register_buffer(
            'leaf_vertex_ids',
            to_tensor([411, 2445, 5905, 3216, 6617], dtype=torch.long)
        )
        self.vertex_joint_selector = VertexJointSelector(
            vertex_ids=vertex_ids)

        self.mass = 75
        self.len_to_mass = 15

    def _parents_to_children(self, parents):
        children = torch.ones_like(parents) * -1
        for i in range(self.NUM_JOINTS):
            if children[parents[i]] < 0:
                children[parents[i]] = i
        # for i in self.LEAF_IDX:
        #     if i < children.shape[0]:
        #         children[i] = -1

        children[self.SPINE3_IDX] = -3
        children[0] = 3
        children[self.SPINE3_IDX] = self.JOINT_NAMES.index('neck')

        return children

    def _parents_to_children_with_leaf(self, parents):
        children = [[] for i in range(len(parents))]
        for i in range(1, len(parents)):
            children[parents[i]].append(i)
        # for i in self.LEAF_IDX:
        #     if i < children.shape[0]:
        #         children[i] = -1

        return children

    def forward(self,
                pose_angle,
                betas,
                transl=None,
                with_d=True,
                return_verts=True):
        ''' Forward pass for the SMPL model

            Parameters
            ----------
            pose_angle: torch.tensor, optional, shape Bx(J*3)
                It should be a tensor that contains joint rotations in
                axis-angle format. (default=None)
            betas: torch.tensor, optional, shape Bx10
                It can used if shape parameters
                `betas` are predicted from some external model.
                (default=None)
            global_orient: torch.tensor, optional, shape Bx3
                Global Orientations.
            transl: torch.tensor, optional, shape Bx3
                Global Translations.
            return_verts: bool, optional
                Return the vertices. (default=True)

            Returns
            -------
        '''
        batch_size = pose_angle.shape[0]

        # concate root orientation with thetas
        # rot_mats: [Bx3x3] * K
        # axes:     [Bx3x3] * K
        rot_mats = []
        axes = []
        if with_d:
            pose_angle = pose_angle.reshape(batch_size, self.NUM_JOINTS + 1, 3)
            cnt_variable = 3
            self.total_variable = 75
            for k in range(24):
                name_list = [f'q{k}_x', f'q{k}_y', f'q{k}_z']
                Rk, ak = euler_to_rot_mat(pose_angle[:, k, :], name_list, cnt_variable, self.total_variable)
                cnt_variable += 3
                rot_mats.append(Rk)
                axes.append(ak)
        else:
            pose_rotmats, axes_tensor = euler_to_rot_mat3(pose_angle.reshape(-1, 3))
            pose_rotmats = pose_rotmats.reshape(-1, 24, 3, 3)
            axes_tensor = axes_tensor.reshape(-1, 24, 3, 3)
            self.total_variable = 1
            for k in range(24):
                Rk = pose_rotmats[:, k]
                ak = axes_tensor[:, k]
                rot_mats.append(Rk)
                axes.append(ak)

        # vertices: (B, N, 3), joints: (B, K, 3)
        vertices, joints, masses, axes, rot_mats, joints_from_verts_extra, template_J = lbs_with_d(
            betas, rot_mats, axes, self.v_template,
            self.shapedirs, self.posedirs,
            self.J_regressor, self.J_regressor_extra, self.parents_with_leaf,
            self.lbs_weights, dtype=self.dtype, leaf_vertex_ids=self.leaf_vertex_ids)

        # 29, 29, 24, 24
        # print(len(joints), len(masses), len(axes), len(rot_mats))

        if transl is not None:
            transl_with_d = transl[:, :, None]
            if with_d:
                transl_with_d.dq = torch.zeros(transl.shape[0], self.total_variable, 3, 1, device=pose_angle.device)
                transl_with_d.dq[:, 0, 0, :] = 1
                transl_with_d.dq[:, 1, 1, :] = 1
                transl_with_d.dq[:, 2, 2, :] = 1

            joints = [matadd_d(transl_with_d, item) for item in joints]
            masses = [matadd_d(transl_with_d, item) for item in masses]

            vertices += transl.unsqueeze(dim=1)
            joints_from_verts_extra += transl.unsqueeze(dim=1)

        output = edict(
            vertices=vertices,
            joints=joints,
            masses=masses,
            axes=axes,
            rot_mats=rot_mats,
            joints_from_verts_extra=joints_from_verts_extra,
            template_J=template_J)
        return output

    def forward_without_d(self, pose_euler, betas, transl=None, return_verts=True):

        pose_rotmats, axes = euler_to_rot_mat3(pose_euler.reshape(-1, 3))
        pose_rotmats = pose_rotmats.reshape(-1, 24, 3, 3)
        axes = axes.reshape(-1, 24, 3, 3)
        # vertices: (B, N, 3), joints: (B, K, 3)
        vertices, joints, rot_mats, joints_from_verts_extra, template_J = lbs(
            betas, pose_rotmats, self.v_template,
            self.shapedirs, self.posedirs,
            self.J_regressor, self.J_regressor_extra, self.parents_with_leaf,
            self.lbs_weights, dtype=self.dtype, pose2rot=False,
            leaf_vertex_ids=self.leaf_vertex_ids)

        if transl is not None:
            joints += transl.unsqueeze(dim=1)
            vertices += transl.unsqueeze(dim=1)
            joints_from_verts_extra += transl.unsqueeze(dim=1)

        axes = torch.split(axes, 1, dim=1)
        axes = [item.squeeze(1) for item in axes]

        output = edict(
            vertices=vertices,
            joints=joints,
            axes=axes,
            rot_mats=rot_mats,
            joints_from_verts_extra=joints_from_verts_extra,
            template_J=template_J)
        return output

    def update_mass(self, kine_out):
        # template_J: [B, 24 + 5, 3]
        template_J = kine_out.template_J
        BATCH_SIZE = template_J.shape[0]
        DEVICE = template_J.device
        DTYPE = self.dtype

        Inertia = [torch.zeros(BATCH_SIZE, 3, 3, device=DEVICE, dtype=DTYPE)]
        m = [torch.ones(BATCH_SIZE, 1, 1, device=DEVICE, dtype=DTYPE) * 5]
        sum_m = 0
        for k in range(1, len(self.parents_with_leaf)):
            if k > 23 and False:
                Ici = Inertia[self.parents_with_leaf[k]] * 1e-5
                mi = torch.ones_like(template_J[:, 0, 0]) * 1e-5
            else:
                delta_p = template_J[:, k, :] - template_J[:, self.parents_with_leaf[k], :]

                x = delta_p[:, 0]
                y = delta_p[:, 1]
                z = delta_p[:, 2]
                len_i = torch.sqrt(x ** 2 + y ** 2 + z ** 2)
                mi = len_i * self.len_to_mass
                mi = torch.round(mi * 10) / 10

                Ixx = mi * torch.ones_like(x)
                Iyy = mi * torch.ones_like(x)
                Izz = mi * torch.ones_like(x)
                Ixy = torch.zeros_like(x)
                Ixz = torch.zeros_like(x)
                Iyz = torch.zeros_like(x)

                Ici = torch.stack([
                    Ixx, Ixy, Ixz,
                    Ixy, Iyy, Iyz,
                    Ixz, Iyz, Izz
                ], dim=-1).reshape(BATCH_SIZE, 3, 3)

            Inertia.append(Ici)
            m.append(mi[:, None, None])
            sum_m += mi

        kine_out.Inertia = Inertia
        kine_out.m = m

        assert len(m) == 29 and len(Inertia) == 29

        return kine_out

    def get_jacobian(self, kine_out):
        '''
        Jw: list of Bx3x75
        Jv: list of Bx3x75
        Jv_joints: list of Bx3x75

        dJvi_dq: BxNqx3x75
        dJvi_dt = (dJvi_dq * dq_dt).sum(dim=1)
        '''
        BATCH_SIZE = kine_out.joints[0].shape[0]
        DEVICE = kine_out.joints[0].device
        DTYPE = self.dtype
        parents = self.parents_with_leaf

        axes = kine_out.axes
        xyz_joints = kine_out.joints
        xyz_masses = kine_out.masses

        # Jw
        Jw0 = torch.zeros((BATCH_SIZE, 3, 75), device=DEVICE, dtype=DTYPE)
        Jw = [Jw0]
        # Jv_joints
        Jv_joints0 = torch.zeros((BATCH_SIZE, 3, 75), device=DEVICE, dtype=DTYPE)
        Jv_joints0[:, :3, :3] = torch.eye(3, device=DEVICE, dtype=DTYPE)[None, :, :]
        Jv_joints = [Jv_joints0]
        # Jv
        Jv = [Jv_joints0]

        for k in range(1, len(parents)):
            # Jw
            parent_Jwi = Jw[parents[k]]
            rel_Jwi = torch.zeros_like(Jw0)
            rel_Jwi[:, :, (parents[k] + 1) * 3: (parents[k] + 1) * 3 + 3] = torch.eye(3, device=DEVICE, dtype=DTYPE)[None, :, :]
            rel_Jwi = matmul_d(axes[parents[k]], rel_Jwi)
            Jwi = matadd_d(rel_Jwi, parent_Jwi)
            # Jv_joints
            parent_Jv_jointsi = Jv_joints[parents[k]]
            rel_joints = matadd_d(xyz_joints[k], xyz_joints[parents[k]], a2=-1)

            rel_joints = vec2skewmat_d(rel_joints, self.total_variable)
            rel_Jv_jointsi = matmul_d(rel_joints, Jwi, a1=-1)
            Jv_jointsi = matadd_d(rel_Jv_jointsi, parent_Jv_jointsi)
            # Jv
            rel_mass = matadd_d(xyz_masses[k], xyz_joints[parents[k]], a2=-1)

            rel_mass = vec2skewmat_d(rel_mass, self.total_variable)
            rel_Jvi = matmul_d(rel_mass, Jwi, a1=-1)
            Jvi = matadd_d(rel_Jvi, parent_Jv_jointsi)

            Jw.append(Jwi)
            Jv_joints.append(Jv_jointsi)
            Jv.append(Jvi)

        # for k in range(1, parents.shape[0]):
        #     parent_Jv_jointsi = Jv_joints[parents[k]]
        #     rel_p = matadd_d(xyz_joints[k], xyz_joints[parents[k]], a2=-1)
        #     rel_p = vec2skewmat_d(rel_p)
        #     Jwi = Jw[k]
        #     rel_Jv_jointsi = matmul_d(rel_p, Jwi, a1=-1)
        #     Jv_jointsi = matadd_d(rel_Jv_jointsi, parent_Jv_jointsi)
        #     Jv_joints.append(Jv_jointsi)

        # for k in range(1, parents.shape[0]):
        #     parent_Jv_jointsi = Jv_joints[parents[k]]
        #     rel_p = matadd_d(xyz_masses[k], xyz_joints[parents[k]], a2=-1)
        #     rel_p = vec2skewmat_d(rel_p)
        #     Jwi = Jw[k]
        #     rel_Jvi = matmul_d(rel_p, Jwi, a1=-1)
        #     Jvi = matadd_d(rel_Jvi, parent_Jv_jointsi)
        #     Jv.append(Jvi)

        # num_part = len(Jv)
        R = kine_out.rot_mats
        m = kine_out.m

        Inertia = kine_out.Inertia

        # M = 0
        Ic = [Inertia[0]]
        M = m[0] * Jv[0].transpose(1, 2).matmul(Jv[0])
        for i in range(1, len(parents)):
            Ici = R[parents[i]].matmul(Inertia[i]).matmul(R[parents[i]].transpose(1, 2))
            Ic.append(Ici)
            # Ic.append(Inertia[i])

            linear_i = m[i] * Jv[i].transpose(1, 2).matmul(Jv[i])
            angular_i = Jw[i].transpose(1, 2).matmul(Ici).matmul(Jw[i])
            M = M + linear_i + angular_i
            # print(i, 'linear:', torch.max(linear_i), torch.min(linear_i))
            # print(i, 'angular:', torch.max(angular_i), torch.min(angular_i))

        # e, _ = torch.symeig(M)
        # max_e, _ = torch.max(e, dim=1)
        # min_e, _ = torch.min(e, dim=1)
        # print(e.shape)
        # print(max_e / min_e)
        # r = 45
        # U, S, V = torch.svd(M)
        # print(S[0])
        # M2 = M.matmul(V)
        # torch.
        # U, S, V = torch.svd(M2)
        # print(S[0])

        # U = U[:, :, :r]
        # S = S[:, :r]
        # V = V[:, :, :r]
        # U, S, V = torch.svd(M2)
        # print(S[0])

        # print(torch.max(torch.abs(M)), torch.min(torch.abs(M)))
        # inv_M = torch.inverse(M)
        # res = torch.matmul(inv_M, M)
        # print(torch.max(inv_M), torch.min(inv_M))
        # print(torch.max(res), torch.min(res))

        # output = edict(
        #     Jw=Jw,
        #     Jv_joints=Jv_joints,
        #     Jv=Jv
        # )
        kine_out.Jw = Jw
        kine_out.Jv = Jv
        kine_out.Jv_joints = Jv_joints
        kine_out.M = M
        kine_out.Ic = Ic
        return kine_out

    def forward_dynamics(self, kine_out, tau, ext_force, cur_contact, inertia_force=None, inertia_angular=None, g=None):
        '''
        core equation:
            M x dq_dt^2 + dM_dt x dq_dt - dT_dq + dV_dq = \\tau + external_tau

        shape:
            M:      Bx75x75
            q:      Bx75x1
            dq_dt:  Bx75x1
            T:      Bx1x1
            dT_dq:  Bx75(Nq)x1
            dM_dq:  BxNqx75x75

            ext_force_i:        Bx3x1
            inertia_force:      Bx3x1
            inertia_angular:    Bx3x1

            Jwi:            Bx3x75
            Jvi:            Bx3x75
            Jv_joints_i:    Bx3x75
            dJvi_dq:        BxNqx3x75

            cur_contact:    Bx6(Nc)

        derivation:
            M = \\sum_i mi x Jvi.T x Jvi + Jwi.T x Ici x Jwi

            T = 0.5 x dq_dt.T x M x dq_dt

            dM_dt = \\sum_i mi x dJvi_dt.T x Jvi + mi x Jvi.T x dJvi_dt
                        + dJwi_dt.T x Ici x Jwi + Jwi.T x Ici x dJwi_dt + Jwi.T x dIci_dt x Jwi

            Ici = Ri x Ici_0 x Ri.T

            dIci_dt = dRi_dt x Ici_0 x Ri.T + Ri x Ici_0 x dRi_dt.T

            dV_dq = - \\sum_i m_i x Jvi.T x g

            dT_dq = dq_dt.T x dM_dq x dq_dt (multiply on the last dimension)

            # Force
            external_tau_i = Jvi.T x ext_force + Jwi.T x Gwi x ext_force

            Gwi = vec2skewmat(∆p)
        '''

        Jv = kine_out.Jv
        Jv_joints = kine_out.Jv_joints
        Jw = kine_out.Jw
        m = kine_out.m

        M = kine_out.M
        num_part = len(Jv)

        g = g.type(self.dtype)
        ext_force = ext_force.type(self.dtype)
        cur_contact = cur_contact.type(self.dtype)

        dV_dq = 0
        if g is not None:
            for i in range(num_part):
                dV_dq = dV_dq + m[i] * Jv[i].transpose(1, 2).matmul(g)

        if inertia_force is not None:
            inertia_force = inertia_force.type(self.dtype)
            i_tau = Jv[0].transpose(1, 2).matmul(inertia_force * m[0])

            for i in range(1, num_part):
                # delta_pi = kine_out.masses[i] - kine_out.joints[self.parents_with_leaf[i]]
                # Gwi = vec2skewmat(delta_pi)
                # inertia_force * m[i]: F=ma
                # i_tau_i = Jv[i].transpose(1, 2).matmul(inertia_force * m[i]) + \
                #     Jw[i].transpose(1, 2).matmul(Gwi).matmul(inertia_force * m[i])
                i_tau_i = Jv[i].transpose(1, 2).matmul(inertia_force * m[i])

                i_tau = i_tau + i_tau_i
            # M_inv = torch.inverse(M)
            # a = M_inv.matmul(i_tau)
            # print(a.shape, inertia_force[0], a[0][:10], ext_force[0], g[0])
        else:
            i_tau = torch.zeros_like(dV_dq)

        if inertia_angular is not None:
            inertia_angular = inertia_angular.type(self.dtype)
            for i in range(1, num_part):
                delta_pi = kine_out.masses[i] - kine_out.joints[0]
                i_tau_i = Jv[i].transpose(1, 2).matmul(torch.cross(inertia_angular, torch.cross(inertia_angular, delta_pi)) * m[i])

                i_tau = i_tau + i_tau_i

        inv_M = torch.inverse(M)
        tau = 0

        use_list = True
        if use_list:
            dq_dt2_list = []
            e_tau_list = []
            num_contact = cur_contact.shape[1]
            expected_tau = 0
            e_tau_joints = []
            for i, idx in enumerate(self.CONTACT_JOINTS):
                e_tau_i = Jv_joints[idx].transpose(1, 2).matmul(ext_force[:, i])
                e_tau_joints.append(e_tau_i)

            # sum_p = 0
            expected_tau = tau + dV_dq
            if self.training:
                for i, idx in enumerate(self.CONTACT_JOINTS):
                    p_i = cur_contact[:, i]
                    eps1 = torch.rand_like(p_i)
                    eps2 = torch.rand_like(p_i)
                    g1 = torch.pow(-torch.log(eps1), 1 / 10)
                    g2 = torch.pow(-torch.log(eps2), 1 / 10)

                    p_i = p_i / g1 / (p_i / g1 + (1 - p_i) / g2)

                    expected_tau = expected_tau + e_tau_joints[i] * p_i[:, None, None]
            else:
                for i, idx in enumerate(self.CONTACT_JOINTS):
                    p_i = cur_contact[:, i]
                    expected_tau = expected_tau + e_tau_joints[i] * p_i[:, None, None]

            return None, None, i_tau, expected_tau
