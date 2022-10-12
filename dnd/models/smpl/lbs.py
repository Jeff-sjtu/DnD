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

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

import torch
import torch.nn.functional as F
import math


def rot_mat_to_euler(rot_mats):
    ''' Intrinsic
    Z1, Y2, X3
    '''
    # Calculates rotation matrix to euler angles
    # Careful for extreme cases of eular angles like [0.0, pi, 0.0]

    # cy = torch.sqrt(rot_mats[:, 0, 0] * rot_mats[:, 0, 0] + rot_mats[:, 1, 0] * rot_mats[:, 1, 0])

    # singular_mask = cy < 1e-5

    alpha = torch.atan2(rot_mats[:, 2, 1], rot_mats[:, 2, 2] + 1e-8)
    # alpha = torch.atan(rot_mats[:, 2, 1] / (rot_mats[:, 2, 2] + 1e-8))
    check_beta = rot_mats[:, 2, 0]
    if torch.sum(check_beta > 1) + torch.sum(check_beta < -1) > 0:
        print(check_beta > 1, '> 1')
        print(check_beta < -1, '< 1')

    beta = torch.asin(-rot_mats[:, 2, 0].clamp(-1 + 1e-6, 1 - 1e-6))
    gamma = torch.atan2(rot_mats[:, 1, 0], rot_mats[:, 0, 0] + 1e-8)
    # gamma = torch.atan(rot_mats[:, 1, 0] / (rot_mats[:, 0, 0] + 1e-8))

    # alpha_mask = alpha > math.pi
    # beta_mask = beta > math.pi
    # gamma_mask = gamma > math.pi

    # alpha[alpha_mask] = alpha[alpha_mask] - 2 * math.pi
    # beta[beta_mask] = beta[beta_mask] - 2 * math.pi
    # gamma[gamma_mask] = gamma[gamma_mask] - 2 * math.pi

    # alpha_singular = torch.atan2(rot_mats[:, 1, 2], rot_mats[:, 1, 1] + 1e-8)
    # beta_singular = torch.atan2(-rot_mats[:, 2, 0], cy + 1e-8)
    # gamma_singular = torch.zeros_like(gamma)

    # alpha[singular_mask] = alpha_singular[singular_mask]
    # beta[singular_mask] = beta_singular[singular_mask]
    # gamma[singular_mask] = gamma_singular[singular_mask]

    vec = torch.stack([alpha, beta, gamma], dim=1)

    # print('alpha', torch.max(alpha), torch.min(alpha))
    # print('beta', torch.max(beta), torch.min(beta))
    # print('gamma', torch.max(gamma), torch.min(gamma))

    return vec


def rot_mat_to_euler_T(rot_mats):
    ''' Intrinsic
    Z1, Y2, X3
    '''
    # rotmats: [T, K, 3, 3]
    K = rot_mats.shape[1]
    T = rot_mats.shape[0]

    vec_list = []
    prev_vec = None

    for t in range(T):
        # [K, 3, 3]
        rot = rot_mats[t]
        sin_beta = -rot[:, 2, 0].clamp(-1 + 1e-6, 1 - 1e-6)
        beta = torch.asin(sin_beta)
        cos_beta_pos = torch.cos(beta)
        cos_beta_neg = -torch.cos(beta)

        beta_1 = torch.atan2(sin_beta, cos_beta_pos)
        beta_2 = torch.atan2(sin_beta, cos_beta_neg)
        assert torch.allclose(beta_1, beta)
        assert (torch.sign(cos_beta_neg) < 0).all()

        alpha_1 = torch.atan2(rot[:, 2, 1], rot[:, 2, 2] + 1e-6)
        alpha_2 = torch.atan2(-rot[:, 2, 1], -rot[:, 2, 2] + 1e-6)

        gamma_1 = torch.atan2(rot[:, 1, 0], rot[:, 0, 0] + 1e-6)
        gamma_2 = torch.atan2(-rot[:, 1, 0], -rot[:, 0, 0] + 1e-6)

        vec1 = torch.stack([alpha_1, beta_1, gamma_1], dim=1)
        vec2 = torch.stack([alpha_2, beta_2, gamma_2], dim=1)

        if t == 0:
            # [B, 3]
            vec = vec1
            prev_vec = vec
            prev_vec_cs = torch.cat((torch.cos(vec), torch.sin(vec)), dim=1)
        else:
            vec1_cs = torch.cat((torch.cos(vec1), torch.sin(vec1)), dim=1)
            vec2_cs = torch.cat((torch.cos(vec2), torch.sin(vec2)), dim=1)
            dot1 = torch.sum(prev_vec_cs * vec1_cs, dim=1, keepdim=True).expand(K, 3)
            dot2 = torch.sum(prev_vec_cs * vec2_cs, dim=1, keepdim=True).expand(K, 3)
            vec = torch.where(dot1 > dot2, vec1, vec2)

            # neg -> pos
            cond = torch.abs(vec - prev_vec - math.pi * 2) < torch.abs(vec - prev_vec)
            vec = torch.where(cond, vec - math.pi * 2, vec)

            # pos -> neg
            cond = torch.abs(vec - prev_vec + math.pi * 2) < torch.abs(vec - prev_vec)
            vec = torch.where(cond, vec + math.pi * 2, vec)

            prev_vec = vec
            prev_vec_cs = torch.cat((torch.cos(vec), torch.sin(vec)), dim=1)

        vec_list.append(vec)

    # [T, K, 3]
    vec = torch.stack(vec_list, dim=0)
    return vec


rot_mat_to_euler_temporal = rot_mat_to_euler_T


def euler_to_rot_mat(vec, d_namelist, cnt_variable, total_variable):
    # vec: (B, 3)
    alpha, beta, gamma = vec[:, 0:1, None], vec[:, 1:2, None], vec[:, 2:3, None]
    name_alpha, name_beta, name_gamma = d_namelist

    dR_dq = torch.zeros(vec.shape[0], total_variable, 3, 3, device=vec.device, dtype=vec.dtype)
    daxes_dq = torch.zeros(vec.shape[0], total_variable, 3, 3, device=vec.device, dtype=vec.dtype)

    zeros = torch.zeros_like(alpha)
    ones = torch.ones_like(alpha)
    Rx = torch.cat([
        ones, zeros, zeros,
        zeros, torch.cos(alpha), -torch.sin(alpha),
        zeros, torch.sin(alpha), torch.cos(alpha)], dim=1)
    Ry = torch.cat([
        torch.cos(beta), zeros, torch.sin(beta),
        zeros, ones, zeros,
        -torch.sin(beta), zeros, torch.cos(beta)], dim=1)
    Rz = torch.cat([
        torch.cos(gamma), -torch.sin(gamma), zeros,
        torch.sin(gamma), torch.cos(gamma), zeros,
        zeros, zeros, ones], dim=1)

    dRx_da = torch.cat([
        zeros, zeros, zeros,
        zeros, -torch.sin(alpha), -torch.cos(alpha),
        zeros, torch.cos(alpha), -torch.sin(alpha)], dim=1)
    dRy_db = torch.cat([
        -torch.sin(beta), zeros, torch.cos(beta),
        zeros, zeros, zeros,
        -torch.cos(beta), zeros, -torch.sin(beta)], dim=1)
    dRz_dg = torch.cat([
        -torch.sin(gamma), -torch.cos(gamma), zeros,
        torch.cos(gamma), -torch.sin(gamma), zeros,
        zeros, zeros, zeros], dim=1)

    Rx = Rx.reshape(vec.shape[0], 3, 3)
    Ry = Ry.reshape(vec.shape[0], 3, 3)
    Rz = Rz.reshape(vec.shape[0], 3, 3)
    dRx_da = dRx_da.reshape(vec.shape[0], 3, 3)
    dRy_db = dRy_db.reshape(vec.shape[0], 3, 3)
    dRz_dg = dRz_dg.reshape(vec.shape[0], 3, 3)
    # mat = torch.bmm(torch.bmm(Rz, Ry), Rx)
    mat = Rz.matmul(Ry).matmul(Rx)
    dR_dq[:, cnt_variable, :, :] = Rz.matmul(Ry).matmul(dRx_da)
    dR_dq[:, cnt_variable + 1, :, :] = Rz.matmul(dRy_db).matmul(Rx)
    dR_dq[:, cnt_variable + 2, :, :] = dRz_dg.matmul(Ry).matmul(Rx)
    mat.dq = dR_dq

    # (B, 3, 1)
    base_a_x = torch.zeros_like(vec)[:, :, None]
    base_a_y = torch.zeros_like(vec)[:, :, None]
    base_a_z = torch.zeros_like(vec)[:, :, None]
    base_a_x[:, 0, :] = 1
    base_a_y[:, 1, :] = 1
    base_a_z[:, 2, :] = 1

    a_z = base_a_z
    a_y = Rz.matmul(base_a_y)
    a_x = Rz.matmul(Ry.matmul(base_a_x))

    a_y_dg = dRz_dg.matmul(base_a_y)
    a_x_db = Rz.matmul(dRy_db.matmul(base_a_x))
    a_x_dg = dRz_dg.matmul(Ry.matmul(base_a_x))

    axes = torch.cat((a_x, a_y, a_z), dim=2)
    axes_db = torch.zeros_like(axes)
    axes_db[:, :, 0:1] = a_x_db
    axes_dg = torch.zeros_like(axes)
    axes_dg[:, :, 0:1] = a_x_dg
    axes_dg[:, :, 1:2] = a_y_dg

    daxes_dq[:, cnt_variable + 1, :, :] = axes_db
    daxes_dq[:, cnt_variable + 2, :, :] = axes_dg
    axes.dq = daxes_dq
    # setattr(axes, f'd{name_beta}', axes_db)
    # setattr(axes, f'd{name_gamma}', axes_dg)
    # axes.d_set = set([name_beta, name_gamma])

    return mat, axes


def euler_to_rot_mat_quat(vec):
    # vec: (B, 3)
    alpha, beta, gamma = vec[:, 0:1, None], vec[:, 1:2, None], vec[:, 2:3, None]

    zeros = torch.zeros_like(alpha)
    ones = torch.ones_like(alpha)
    Rx = torch.cat([
        ones, zeros, zeros,
        zeros, torch.cos(alpha), -torch.sin(alpha),
        zeros, torch.sin(alpha), torch.cos(alpha)], dim=1)
    Ry = torch.cat([
        torch.cos(beta), zeros, torch.sin(beta),
        zeros, ones, zeros,
        -torch.sin(beta), zeros, torch.cos(beta)], dim=1)
    Rz = torch.cat([
        torch.cos(gamma), -torch.sin(gamma), zeros,
        torch.sin(gamma), torch.cos(gamma), zeros,
        zeros, zeros, ones], dim=1)

    Rx = Rx.reshape(vec.shape[0], 3, 3)
    Ry = Ry.reshape(vec.shape[0], 3, 3)
    Rz = Rz.reshape(vec.shape[0], 3, 3)
    # mat = torch.bmm(torch.bmm(Rz, Ry), Rx)
    mat = Rz.matmul(Ry).matmul(Rx)

    # (B, 3, 1)
    base_a_x = torch.zeros_like(vec)[:, :, None]
    base_a_y = torch.zeros_like(vec)[:, :, None]
    base_a_z = torch.zeros_like(vec)[:, :, None]
    base_a_x[:, 0, :] = 1
    base_a_y[:, 1, :] = 1
    base_a_z[:, 2, :] = 1

    a_z = base_a_z
    a_y = Rz.matmul(base_a_y)
    a_x = Rz.matmul(Ry.matmul(base_a_x))

    axes = torch.cat((a_x, a_y, a_z), dim=2)

    deuler_dquat = get_euler_to_quat_mat(vec)
    quat_axes = axes.matmul(deuler_dquat)

    return mat, axes, quat_axes


def get_euler_to_quat_mat(vec):
    alpha, beta, gamma = vec[:, 0:1, None], vec[:, 1:2, None], vec[:, 2:3, None]
    sa = torch.sin(alpha / 2)
    ca = torch.cos(alpha / 2)
    sb = torch.sin(beta / 2)
    cb = torch.cos(beta / 2)
    sg = torch.sin(gamma / 2)
    cg = torch.cos(gamma / 2)

    da_dq1 = 2 / (ca * sb * sg - sa * cb * cg)
    da_dq2 = 2 / (ca * cb * cg + sa * sb * sg)
    da_dq3 = 2 / (ca * cb * sg - sa * sb * cg)
    da_dq4 = -2 / (sa * cb * sg + ca * sb * cg)

    db_dq1 = 2 / (sa * cb * sg - ca * sb * cg)
    db_dq2 = -2 / (sa * sb * cg + ca * cb * sg)
    db_dq3 = 2 / (ca * cb * cg - sa * sb * sg)
    db_dq4 = -2 / (ca * sb * sg + sa * cb * cg)

    dg_dq1 = 2 / (sa * sb * cg - ca * cb * sg)
    dg_dq2 = -2 / (sa * cb * sg + ca * sb * cg)
    dg_dq3 = 2 / (sa * cb * cg - ca * sb * sg)
    dg_dq4 = 2 / (sa * sb * sg + ca * cb * cg)

    de_dq = torch.cat(
        (
            da_dq1, da_dq2, da_dq3, da_dq4,
            db_dq1, db_dq2, db_dq3, db_dq4,
            dg_dq1, dg_dq2, dg_dq3, dg_dq4
        ), dim=1
    ).reshape(vec.shape[0], 3, 4)
    # print(torch.max((de_dq)), torch.min(de_dq))
    # diff = torch.abs(de_dq[:, 1:2, 3:4] - db_dq4)
    # diff = torch.abs(de_dq[:, 0:1, 0:1] - da_dq1)
    # print('diff:', torch.max(diff))
    return de_dq


def quat_to_rot_mat(quat):
    quat = quat / quat.norm(p=2, dim=1, keepdim=True)
    # quat: (B, 4)
    q0, q1, q2, q3 = quat[:, 0:1, None], quat[:, 1:2, None], quat[:, 2:3, None], quat[:, 3:4, None]

    mat = quat2mat(quat)

    axes = 2 * torch.cat(
        [
            -q1, q0, -q3, q2,
            -q2, q3, q0, -q1,
            -q3, -q2, q1, q0
        ], dim=1
    ).reshape(quat.shape[0], 3, 4)

    return mat, axes


def quat2mat(quat):
    """
    This function is borrowed from https://github.com/MandyMo/pytorch_HMR/blob/master/src/util.py#L50

    Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [batch_size, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [batch_size, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

    batch_size = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack([
        w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz, 2 * wz + 2 * xy,
        w2 - x2 + y2 - z2, 2 * yz - 2 * wx, 2 * xz - 2 * wy, 2 * wx + 2 * yz,
        w2 - x2 - y2 + z2
    ], dim=1).view(batch_size, 3, 3)
    return rotMat


def euler_to_rot_mat2(vec):
    # vec: (B, 3)
    alpha, beta, gamma = vec[:, 0:1, None], vec[:, 1:2, None], vec[:, 2:3, None]

    zeros = torch.zeros_like(alpha)
    ones = torch.ones_like(alpha)
    Rx = torch.cat([
        ones, zeros, zeros,
        zeros, torch.cos(alpha), -torch.sin(alpha),
        zeros, torch.sin(alpha), torch.cos(alpha)], dim=1)
    Ry = torch.cat([
        torch.cos(beta), zeros, torch.sin(beta),
        zeros, ones, zeros,
        -torch.sin(beta), zeros, torch.cos(beta)], dim=1)
    Rz = torch.cat([
        torch.cos(gamma), -torch.sin(gamma), zeros,
        torch.sin(gamma), torch.cos(gamma), zeros,
        zeros, zeros, ones], dim=1)

    Rx = Rx.reshape(vec.shape[0], 3, 3)
    Ry = Ry.reshape(vec.shape[0], 3, 3)
    Rz = Rz.reshape(vec.shape[0], 3, 3)

    mat = Rz.matmul(Ry).matmul(Rx)

    return mat


def euler_to_rot_mat3(vec):
    # vec: (B, 3)
    alpha, beta, gamma = vec[:, 0:1, None], vec[:, 1:2, None], vec[:, 2:3, None]

    zeros = torch.zeros_like(alpha)
    ones = torch.ones_like(alpha)
    Rx = torch.cat([
        ones, zeros, zeros,
        zeros, torch.cos(alpha), -torch.sin(alpha),
        zeros, torch.sin(alpha), torch.cos(alpha)], dim=1)
    Ry = torch.cat([
        torch.cos(beta), zeros, torch.sin(beta),
        zeros, ones, zeros,
        -torch.sin(beta), zeros, torch.cos(beta)], dim=1)
    Rz = torch.cat([
        torch.cos(gamma), -torch.sin(gamma), zeros,
        torch.sin(gamma), torch.cos(gamma), zeros,
        zeros, zeros, ones], dim=1)

    Rx = Rx.reshape(vec.shape[0], 3, 3)
    Ry = Ry.reshape(vec.shape[0], 3, 3)
    Rz = Rz.reshape(vec.shape[0], 3, 3)

    mat = Rz.matmul(Ry).matmul(Rx)

    # (B, 3, 1)
    base_a_x = torch.zeros_like(vec)[:, :, None]
    base_a_y = torch.zeros_like(vec)[:, :, None]
    base_a_z = torch.zeros_like(vec)[:, :, None]
    base_a_x[:, 0, :] = 1
    base_a_y[:, 1, :] = 1
    base_a_z[:, 2, :] = 1

    a_z = base_a_z
    a_y = Rz.matmul(base_a_y)
    a_x = Rz.matmul(Ry.matmul(base_a_x))
    axes = torch.cat((a_x, a_y, a_z), dim=2)

    return mat, axes


def find_dynamic_lmk_idx_and_bcoords(vertices, pose, dynamic_lmk_faces_idx,
                                     dynamic_lmk_b_coords,
                                     neck_kin_chain, dtype=torch.float32):
    ''' Compute the faces, barycentric coordinates for the dynamic landmarks


        To do so, we first compute the rotation of the neck around the y-axis
        and then use a pre-computed look-up table to find the faces and the
        barycentric coordinates that will be used.

        Special thanks to Soubhik Sanyal (soubhik.sanyal@tuebingen.mpg.de)
        for providing the original TensorFlow implementation and for the LUT.

        Parameters
        ----------
        vertices: torch.tensor BxVx3, dtype = torch.float32
            The tensor of input vertices
        pose: torch.tensor Bx(Jx3), dtype = torch.float32
            The current pose of the body model
        dynamic_lmk_faces_idx: torch.tensor L, dtype = torch.long
            The look-up table from neck rotation to faces
        dynamic_lmk_b_coords: torch.tensor Lx3, dtype = torch.float32
            The look-up table from neck rotation to barycentric coordinates
        neck_kin_chain: list
            A python list that contains the indices of the joints that form the
            kinematic chain of the neck.
        dtype: torch.dtype, optional

        Returns
        -------
        dyn_lmk_faces_idx: torch.tensor, dtype = torch.long
            A tensor of size BxL that contains the indices of the faces that
            will be used to compute the current dynamic landmarks.
        dyn_lmk_b_coords: torch.tensor, dtype = torch.float32
            A tensor of size BxL that contains the indices of the faces that
            will be used to compute the current dynamic landmarks.
    '''

    batch_size = vertices.shape[0]

    aa_pose = torch.index_select(pose.view(batch_size, -1, 3), 1,
                                 neck_kin_chain)
    rot_mats = batch_rodrigues(
        aa_pose.view(-1, 3), dtype=dtype).view(batch_size, -1, 3, 3)

    rel_rot_mat = torch.eye(
        3, device=vertices.device, dtype=dtype).unsqueeze_(dim=0).repeat(
        batch_size, 1, 1)
    for idx in range(len(neck_kin_chain)):
        rel_rot_mat = torch.bmm(rot_mats[:, idx], rel_rot_mat)

    y_rot_angle = torch.round(
        torch.clamp(-rot_mat_to_euler(rel_rot_mat) * 180.0 / np.pi,
                    max=39)).to(dtype=torch.long)
    neg_mask = y_rot_angle.lt(0).to(dtype=torch.long)
    mask = y_rot_angle.lt(-39).to(dtype=torch.long)
    neg_vals = mask * 78 + (1 - mask) * (39 - y_rot_angle)
    y_rot_angle = (neg_mask * neg_vals +
                   (1 - neg_mask) * y_rot_angle)

    dyn_lmk_faces_idx = torch.index_select(dynamic_lmk_faces_idx,
                                           0, y_rot_angle)
    dyn_lmk_b_coords = torch.index_select(dynamic_lmk_b_coords,
                                          0, y_rot_angle)

    return dyn_lmk_faces_idx, dyn_lmk_b_coords


def vertices2landmarks(vertices, faces, lmk_faces_idx, lmk_bary_coords):
    ''' Calculates landmarks by barycentric interpolation

        Parameters
        ----------
        vertices: torch.tensor BxVx3, dtype = torch.float32
            The tensor of input vertices
        faces: torch.tensor Fx3, dtype = torch.long
            The faces of the mesh
        lmk_faces_idx: torch.tensor L, dtype = torch.long
            The tensor with the indices of the faces used to calculate the
            landmarks.
        lmk_bary_coords: torch.tensor Lx3, dtype = torch.float32
            The tensor of barycentric coordinates that are used to interpolate
            the landmarks

        Returns
        -------
        landmarks: torch.tensor BxLx3, dtype = torch.float32
            The coordinates of the landmarks for each mesh in the batch
    '''
    # Extract the indices of the vertices for each face
    # BxLx3
    batch_size, num_verts = vertices.shape[:2]
    device = vertices.device

    lmk_faces = torch.index_select(faces, 0, lmk_faces_idx.view(-1)).view(
        batch_size, -1, 3)

    lmk_faces += torch.arange(
        batch_size, dtype=torch.long, device=device).view(-1, 1, 1) * num_verts

    lmk_vertices = vertices.view(-1, 3)[lmk_faces].view(
        batch_size, -1, 3, 3)

    landmarks = torch.einsum('blfi,blf->bli', [lmk_vertices, lmk_bary_coords])
    return landmarks


def joints2bones(joints, parents):
    ''' Decompose joints location to bone length and direction.

        Parameters
        ----------
        joints: torch.tensor Bx24x3
    '''
    assert joints.shape[1] == parents.shape[0]
    bone_dirs = torch.zeros_like(joints)
    bone_lens = torch.zeros_like(joints[:, :, :1])

    for c_id in range(parents.shape[0]):
        p_id = parents[c_id]
        if p_id == -1:
            # Parent node
            bone_dirs[:, c_id] = joints[:, c_id]
        else:
            # Child node
            # (B, 3)
            diff = joints[:, c_id] - joints[:, p_id]
            length = torch.norm(diff, dim=1, keepdim=True) + 1e-8
            direct = diff / length

            bone_dirs[:, c_id] = direct
            bone_lens[:, c_id] = length

    return bone_dirs, bone_lens


def bones2joints(bone_dirs, bone_lens, parents):
    ''' Recover bone length and direction to joints location.

        Parameters
        ----------
        bone_dirs: torch.tensor 1x24x3
        bone_lens: torch.tensor Bx24x1
    '''
    batch_size = bone_lens.shape[0]
    joints = torch.zeros_like(bone_dirs).expand(batch_size, 24, 3)

    for c_id in range(parents.shape[0]):
        p_id = parents[c_id]
        if p_id == -1:
            # Parent node
            joints[:, c_id] = bone_dirs[:, c_id]
        else:
            # Child node
            joints[:, c_id] = joints[:, p_id] + bone_dirs[:, c_id] * bone_lens[:, c_id]

    return joints


def lbs_with_d(betas, rot_mats_list, axes_list, v_template, shapedirs, posedirs, J_regressor, J_regressor_h36m, parents,
               lbs_weights, dtype=torch.float32, leaf_vertex_ids=None):
    ''' Performs Linear Blend Skinning with the given shape and pose parameters

        Parameters
        ----------
        betas : torch.tensor BxNB
            The tensor of shape parameters
        rot_mats_list : torch.tensor [Bx3x3] * K
            The list of rotation matrices
        axes_list : torch.tensor [Bx3x3] * K
            The list of joint axes
        v_template torch.tensor BxVx3
            The template mesh that will be deformed
        shapedirs : torch.tensor 1xNB
            The tensor of PCA shape displacements
        posedirs : torch.tensor Px(V * 3)
            The pose PCA coefficients
        J_regressor : torch.tensor JxV
            The regressor array that is used to calculate the joints from
            the position of the vertices
        parents: torch.tensor J
            The array that describes the kinematic tree for the model
        lbs_weights: torch.tensor N x V x (J + 1)
            The linear blend skinning weights that represent how much the
            rotation matrix of each part affects each vertex
        pose2rot: bool, optional
            Flag on whether to convert the input pose tensor to rotation
            matrices. The default value is True. If False, then the pose tensor
            should already contain rotation matrices and have a size of
            Bx(J + 1)x9
        dtype: torch.dtype, optional

        Returns
        -------
        verts: torch.tensor BxVx3
            The vertices of the mesh after applying the shape and pose
            displacements.
        joints: torch.tensor BxJx3
            The joints of the model
        rot_mats: torch.tensor BxJx3x3
            The rotation matrics of each joints
    '''
    batch_size = max(betas.shape[0], rot_mats_list[0].shape[0])
    device = betas.device

    # Add shape contribution
    v_shaped = v_template + blend_shapes(betas, shapedirs)

    # Get the joints
    # NxJx3 array
    J = vertices2joints(J_regressor, v_shaped)
    leaf_J = torch.index_select(v_shaped, 1, leaf_vertex_ids)
    J = torch.cat((J, leaf_J), dim=1)

    # 3. Add pose blend shapes
    # N x J x 3 x 3
    ident = torch.eye(3, dtype=dtype, device=device)
    rot_mats_tensor = torch.stack(rot_mats_list, dim=1)

    pose_feature = rot_mats_tensor[:, 1:] - ident[None, None, :, :]
    # rot_mats = pose.view(batch_size, -1, 3, 3)

    pose_offsets = torch.matmul(pose_feature.view(batch_size, -1),
                                posedirs).view(batch_size, -1, 3)

    v_posed = pose_offsets + v_shaped

    # 4. Get the global joint location
    J_transformed, mass_transformed, axes_chain, rotmat_chain, A = batch_rigid_transform_with_d(
        rot_mats_list, axes_list, J, parents, dtype=dtype)

    # 5. Do skinning:
    # W is N x V x (J + 1)
    W = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
    # (N x V x (J + 1)) x (N x (J + 1) x 16)
    num_joints = J_regressor.shape[0]
    T = torch.matmul(W, A.view(batch_size, num_joints, 16)) \
        .view(batch_size, -1, 4, 4)

    homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1],
                               dtype=dtype, device=device)
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

    verts = v_homo[:, :, :3, 0]

    J_from_verts = vertices2joints(J_regressor_h36m, verts)

    return verts, J_transformed, mass_transformed, axes_chain, rotmat_chain, J_from_verts, J


def lbs_quat(betas, rot_mats_list, quat_axes_list, v_template, shapedirs, posedirs, J_regressor, J_regressor_h36m, parents,
             lbs_weights, dtype=torch.float32, leaf_vertex_ids=None):
    ''' Performs Linear Blend Skinning with the given shape and pose parameters

        Parameters
        ----------
        betas : torch.tensor BxNB
            The tensor of shape parameters
        rot_mats_list : torch.tensor [Bx3x3] * K
            The list of rotation matrices
        quat_axes_list : torch.tensor [Bx3x3] * K
            The list of joint axes
        v_template torch.tensor BxVx3
            The template mesh that will be deformed
        shapedirs : torch.tensor 1xNB
            The tensor of PCA shape displacements
        posedirs : torch.tensor Px(V * 3)
            The pose PCA coefficients
        J_regressor : torch.tensor JxV
            The regressor array that is used to calculate the joints from
            the position of the vertices
        parents: torch.tensor J
            The array that describes the kinematic tree for the model
        lbs_weights: torch.tensor N x V x (J + 1)
            The linear blend skinning weights that represent how much the
            rotation matrix of each part affects each vertex
        pose2rot: bool, optional
            Flag on whether to convert the input pose tensor to rotation
            matrices. The default value is True. If False, then the pose tensor
            should already contain rotation matrices and have a size of
            Bx(J + 1)x9
        dtype: torch.dtype, optional

        Returns
        -------
        verts: torch.tensor BxVx3
            The vertices of the mesh after applying the shape and pose
            displacements.
        joints: torch.tensor BxJx3
            The joints of the model
        rot_mats: torch.tensor BxJx3x3
            The rotation matrics of each joints
    '''
    batch_size = max(betas.shape[0], rot_mats_list[0].shape[0])
    device = betas.device

    # Add shape contribution
    v_shaped = v_template + blend_shapes(betas, shapedirs)

    # Get the joints
    # NxJx3 array
    J = vertices2joints(J_regressor, v_shaped)
    leaf_J = torch.index_select(v_shaped, 1, leaf_vertex_ids)
    J = torch.cat((J, leaf_J), dim=1)

    # 3. Add pose blend shapes
    # N x J x 3 x 3
    ident = torch.eye(3, dtype=dtype, device=device)
    rot_mats_tensor = torch.stack(rot_mats_list, dim=1)

    pose_feature = rot_mats_tensor[:, 1:] - ident[None, None, :, :]
    # rot_mats = pose.view(batch_size, -1, 3, 3)

    pose_offsets = torch.matmul(pose_feature.view(batch_size, -1),
                                posedirs).view(batch_size, -1, 3)

    v_posed = pose_offsets + v_shaped

    # 4. Get the global joint location
    J_transformed, mass_transformed, axes_chain, rotmat_chain, A = batch_rigid_transform_quat(
        rot_mats_list, quat_axes_list, J, parents, dtype=dtype)

    # 5. Do skinning:
    # W is N x V x (J + 1)
    W = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
    # (N x V x (J + 1)) x (N x (J + 1) x 16)
    num_joints = J_regressor.shape[0]
    T = torch.matmul(W, A.view(batch_size, num_joints, 16)) \
        .view(batch_size, -1, 4, 4)

    homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1],
                               dtype=dtype, device=device)
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

    verts = v_homo[:, :, :3, 0]

    J_from_verts = vertices2joints(J_regressor_h36m, verts)

    return verts, J_transformed, mass_transformed, axes_chain, rotmat_chain, J_from_verts, J


def lbs(betas, pose, v_template, shapedirs, posedirs, J_regressor, J_regressor_h36m, parents,
        lbs_weights, pose2rot=True, dtype=torch.float32, leaf_vertex_ids=None):
    ''' Performs Linear Blend Skinning with the given shape and pose parameters

        Parameters
        ----------
        betas : torch.tensor BxNB
            The tensor of shape parameters
        pose : torch.tensor Bx(J + 1) * 3
            The pose parameters in axis-angle format
        v_template torch.tensor BxVx3
            The template mesh that will be deformed
        shapedirs : torch.tensor 1xNB
            The tensor of PCA shape displacements
        posedirs : torch.tensor Px(V * 3)
            The pose PCA coefficients
        J_regressor : torch.tensor JxV
            The regressor array that is used to calculate the joints from
            the position of the vertices
        parents: torch.tensor J
            The array that describes the kinematic tree for the model
        lbs_weights: torch.tensor N x V x (J + 1)
            The linear blend skinning weights that represent how much the
            rotation matrix of each part affects each vertex
        pose2rot: bool, optional
            Flag on whether to convert the input pose tensor to rotation
            matrices. The default value is True. If False, then the pose tensor
            should already contain rotation matrices and have a size of
            Bx(J + 1)x9
        dtype: torch.dtype, optional

        Returns
        -------
        verts: torch.tensor BxVx3
            The vertices of the mesh after applying the shape and pose
            displacements.
        joints: torch.tensor BxJx3
            The joints of the model
        rot_mats: torch.tensor BxJx3x3
            The rotation matrics of each joints
    '''
    batch_size = max(betas.shape[0], pose.shape[0])
    device = betas.device

    # Add shape contribution
    v_shaped = v_template + blend_shapes(betas, shapedirs)

    # Get the joints
    # NxJx3 array
    J = vertices2joints(J_regressor, v_shaped)
    leaf_J = torch.index_select(v_shaped, 1, leaf_vertex_ids)
    J = torch.cat((J, leaf_J), dim=1)

    # 3. Add pose blend shapes
    # N x J x 3 x 3
    ident = torch.eye(3, dtype=dtype, device=device)
    if pose2rot:
        if pose.numel() == batch_size * 24 * 4:
            rot_mats = quat_to_rotmat(pose.reshape(batch_size * 24, 4)).reshape(batch_size, 24, 3, 3)
        else:
            rot_mats = batch_rodrigues(
                pose.view(-1, 3), dtype=dtype).view([batch_size, -1, 3, 3])

        pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
        # (N x P) x (P, V * 3) -> N x V x 3
        pose_offsets = torch.matmul(pose_feature, posedirs) \
            .view(batch_size, -1, 3)
    else:
        pose_feature = pose[:, 1:].view(batch_size, -1, 3, 3) - ident
        rot_mats = pose.view(batch_size, -1, 3, 3)

        pose_offsets = torch.matmul(pose_feature.view(batch_size, -1),
                                    posedirs).view(batch_size, -1, 3)

    v_posed = pose_offsets + v_shaped
    # 4. Get the global joint location
    J_transformed, rot_mats, A = batch_rigid_transform(
        rot_mats, J, parents, dtype=dtype)

    # 5. Do skinning:
    # W is N x V x (J + 1)
    W = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
    # (N x V x (J + 1)) x (N x (J + 1) x 16)
    num_joints = J_regressor.shape[0]
    T = torch.matmul(W, A.view(batch_size, num_joints, 16)) \
        .view(batch_size, -1, 4, 4)

    homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1],
                               dtype=dtype, device=device)
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

    verts = v_homo[:, :, :3, 0]

    J_from_verts = vertices2joints(J_regressor_h36m, verts)

    return verts, J_transformed, rot_mats, J_from_verts, J


def batch_rigid_transform(rot_mats, joints, parents, dtype=torch.float32):
    """
    Applies a batch of rigid transformations to the joints

    Parameters
    ----------
    rot_mats : torch.tensor BxNx3x3
        Tensor of rotation matrices
    joints : torch.tensor BxNx3
        Locations of joints. (Template Pose)
    parents : torch.tensor BxN
        The kinematic tree of each object
    dtype : torch.dtype, optional:
        The data type of the created tensors, the default is torch.float32

    Returns
    -------
    posed_joints : torch.tensor BxNx3
        The locations of the joints after applying the pose rotations
    rel_transforms : torch.tensor BxNx4x4
        The relative (with respect to the root joint) rigid transformations
        for all the joints
    """
    joints = torch.unsqueeze(joints, dim=-1)
    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]].clone()

    # (B, K + 1, 4, 4)
    eye_mats = torch.eye(3, device=rot_mats.device)[None, None, :, :]
    eye_mats = eye_mats.expand(rot_mats.shape[0], 5, 3, 3)
    rot_mats = torch.cat((rot_mats, eye_mats), dim=1)
    transforms_mat = transform_mat(
        rot_mats.reshape(-1, 3, 3),
        rel_joints.reshape(-1, 3, 1)).reshape(-1, joints.shape[1], 4, 4)

    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, len(parents)):
        # Subtract the joint location at the rest pose
        # No need for rotation, since it's identity when at rest
        # (B, 4, 4) x (B, 4, 4)
        curr_res = torch.matmul(transform_chain[parents[i]],
                                transforms_mat[:, i])
        transform_chain.append(curr_res)

    # (B, K + 1, 4, 4)
    transforms = torch.stack(transform_chain, dim=1)

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]

    joints_homogen = F.pad(joints, [0, 0, 0, 1])

    rel_transforms = transforms - F.pad(
        torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0])

    rotmats = transforms[:, :24, :3, :3]

    return posed_joints, rotmats, rel_transforms[:, :24]


def vertices2joints(J_regressor, vertices):
    ''' Calculates the 3D joint locations from the vertices

    Parameters
    ----------
    J_regressor : torch.tensor JxV
        The regressor array that is used to calculate the joints from the
        position of the vertices
    vertices : torch.tensor BxVx3
        The tensor of mesh vertices

    Returns
    -------
    torch.tensor BxJx3
        The location of the joints
    '''

    return torch.einsum('bik,ji->bjk', [vertices, J_regressor])


def blend_shapes(betas, shape_disps):
    ''' Calculates the per vertex displacement due to the blend shapes


    Parameters
    ----------
    betas : torch.tensor Bx(num_betas)
        Blend shape coefficients
    shape_disps: torch.tensor Vx3x(num_betas)
        Blend shapes

    Returns
    -------
    torch.tensor BxVx3
        The per-vertex displacement due to shape deformation
    '''

    # Displacement[b, m, k] = sum_{l} betas[b, l] * shape_disps[m, k, l]
    # i.e. Multiply each shape displacement by its corresponding beta and
    # then sum them.
    blend_shape = torch.einsum('bl,mkl->bmk', [betas, shape_disps])
    return blend_shape


def batch_rodrigues(rot_vecs, epsilon=1e-8, dtype=torch.float32):
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''

    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat


def transform_mat(R, t):
    ''' Creates a batch of transformation matrices
        Args:
            - R: Bx3x3 array of a batch of rotation matrices
            - t: Bx3x1 array of a batch of translation vectors
        Returns:
            - T: Bx4x4 Transformation matrix
    '''
    # No padding left or right, only add an extra row
    return torch.cat([F.pad(R, [0, 0, 0, 1]),
                      F.pad(t, [0, 0, 0, 1], value=1)], dim=2)


def batch_rigid_transform_with_d(rot_mats_list, axes_list, joints, parents, dtype=torch.float32):
    """
    Applies a batch of rigid transformations to the joints

    Parameters
    ----------
    rot_mats_list : torch.tensor [Bx3x3] * K
        List of rotation matrices
    axes_list : torch.tensor [Bx3x3] * K
        List of joint axes
    joints : torch.tensor BxNx3
        Locations of joints. (Template Pose)
    parents : torch.tensor BxN
        The kinematic tree of each object
    dtype : torch.dtype, optional:
        The data type of the created tensors, the default is torch.float32

    Returns
    -------
    posed_joints : torch.tensor [Bx3x1] * K
        The locations of the joints after applying the pose rotations
    poased_masses : torch.tensor [Bx3x1] * K
        The locations of the mass of each body part after applying the pose rotations
    axes_chain : torch.tensor [Bx3x3] * K
        The axes after applying the pose rotations
    rotmat_chain : torch.tensor [Bx3x1] * K
        The global rotation matrices
    rel_transforms : torch.tensor BxNx4x4
        The relative (with respect to the root joint) rigid transformations
        for all the joints
    """
    joints = torch.unsqueeze(joints, dim=-1)
    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]].clone()
    rel_masses = rel_joints.clone()
    rel_masses[:, 1:] = rel_masses[:, 1:] / 2
    root = rel_joints[:, 0]
    root.dq = torch.zeros(joints.shape[0], 75, 3, 1, device=root.device)

    posed_joints = [root]
    posed_masses = [root]
    rotmat_chain = [rot_mats_list[0]]
    axes_chain = [axes_list[0]]

    for k in range(1, len(parents)):
        curr_joint = matmul_add_d(
            rotmat_chain[parents[k]], rel_joints[:, k], posed_joints[parents[k]])

        curr_mass = matmul_add_d(
            rotmat_chain[parents[k]], rel_masses[:, k], posed_joints[parents[k]])

        if k < 24:
            curr_axes = matmul_d(rotmat_chain[parents[k]], axes_list[k])
            curr_mat = matmul_d(rotmat_chain[parents[k]], rot_mats_list[k])

            axes_chain.append(curr_axes)
            rotmat_chain.append(curr_mat)

        posed_joints.append(curr_joint)
        posed_masses.append(curr_mass)

    # # (B, K + 1, 4, 4)
    # transforms_mat = transform_mat(
    #     rot_mats.reshape(-1, 3, 3),
    #     rel_joints.reshape(-1, 3, 1)).reshape(-1, joints.shape[1], 4, 4)

    # transform_chain = [transforms_mat[:, 0]]
    # for i in range(1, parents.shape[0]):
    #     # Subtract the joint location at the rest pose
    #     # No need for rotation, since it's identity when at rest
    #     # (B, 4, 4) x (B, 4, 4)
    #     curr_res = torch.matmul(transform_chain[parents[i]],
    #                             transforms_mat[:, i])
    #     transform_chain.append(curr_res)

    # (B, K + 1, 4, 4)
    # transforms = torch.stack(transform_chain, dim=1)

    # The last column of the transformations contains the posed joints
    # posed_joints = transforms[:, :, :3, 3]

    # # The last column of the transformations contains the posed joints
    # posed_joints = transforms[:, :, :3, 3]

    # joints_homogen = F.pad(joints, [0, 0, 0, 1])

    # rel_transforms = transforms - F.pad(
    #     torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0])
    rotmat_tensor = torch.stack(rotmat_chain, dim=1)
    posed_joints_tensor = torch.stack(posed_joints, dim=1)

    rel_transforms = transform_relmat(rotmat_tensor, posed_joints_tensor[:, :24, :, :], joints[:, :24, :, :])

    return posed_joints, posed_masses, axes_chain, rotmat_chain, rel_transforms


def batch_rigid_transform_quat(rot_mats_list, quat_axes_list, joints, parents, dtype=torch.float32):
    """
    Applies a batch of rigid transformations to the joints

    Parameters
    ----------
    rot_mats_list : torch.tensor [Bx3x3] * K
        List of rotation matrices
    quat_axes_list : torch.tensor [Bx3x3] * K
        List of joint axes
    joints : torch.tensor BxNx3
        Locations of joints. (Template Pose)
    parents : torch.tensor BxN
        The kinematic tree of each object
    dtype : torch.dtype, optional:
        The data type of the created tensors, the default is torch.float32

    Returns
    -------
    posed_joints : torch.tensor [Bx3x1] * K
        The locations of the joints after applying the pose rotations
    poased_masses : torch.tensor [Bx3x1] * K
        The locations of the mass of each body part after applying the pose rotations
    axes_chain : torch.tensor [Bx3x3] * K
        The axes after applying the pose rotations
    rotmat_chain : torch.tensor [Bx3x1] * K
        The global rotation matrices
    rel_transforms : torch.tensor BxNx4x4
        The relative (with respect to the root joint) rigid transformations
        for all the joints
    """
    joints = torch.unsqueeze(joints, dim=-1)
    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]].clone()
    rel_masses = rel_joints.clone()
    rel_masses[:, 1:] = rel_masses[:, 1:] / 2
    root = rel_joints[:, 0]

    posed_joints = [root]
    posed_masses = [root]
    rotmat_chain = [rot_mats_list[0]]
    axes_chain = [quat_axes_list[0]]

    for k in range(1, len(parents)):
        curr_joint = rotmat_chain[parents[k]].matmul(rel_joints[:, k]) + posed_joints[parents[k]]

        curr_mass = rotmat_chain[parents[k]].matmul(rel_masses[:, k]) + posed_joints[parents[k]]

        if k < 24:
            curr_axes = torch.matmul(rotmat_chain[parents[k]], quat_axes_list[k])
            curr_mat = torch.matmul(rotmat_chain[parents[k]], rot_mats_list[k])

            axes_chain.append(curr_axes)
            rotmat_chain.append(curr_mat)

        posed_joints.append(curr_joint)
        posed_masses.append(curr_mass)

    # # (B, K + 1, 4, 4)
    # transforms_mat = transform_mat(
    #     rot_mats.reshape(-1, 3, 3),
    #     rel_joints.reshape(-1, 3, 1)).reshape(-1, joints.shape[1], 4, 4)

    # transform_chain = [transforms_mat[:, 0]]
    # for i in range(1, parents.shape[0]):
    #     # Subtract the joint location at the rest pose
    #     # No need for rotation, since it's identity when at rest
    #     # (B, 4, 4) x (B, 4, 4)
    #     curr_res = torch.matmul(transform_chain[parents[i]],
    #                             transforms_mat[:, i])
    #     transform_chain.append(curr_res)

    # (B, K + 1, 4, 4)
    # transforms = torch.stack(transform_chain, dim=1)

    # The last column of the transformations contains the posed joints
    # posed_joints = transforms[:, :, :3, 3]

    # # The last column of the transformations contains the posed joints
    # posed_joints = transforms[:, :, :3, 3]

    # joints_homogen = F.pad(joints, [0, 0, 0, 1])

    # rel_transforms = transforms - F.pad(
    #     torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0])
    rotmat_tensor = torch.stack(rotmat_chain, dim=1)
    posed_joints_tensor = torch.stack(posed_joints, dim=1)

    rel_transforms = transform_relmat(rotmat_tensor, posed_joints_tensor[:, :24, :, :], joints[:, :24, :, :])

    return posed_joints, posed_masses, axes_chain, rotmat_chain, rel_transforms


def batch_inverse_kinematics_transform(
        pose_skeleton, global_orient,
        phis,
        rest_pose,
        children, parents, dtype=torch.float32, train=False,
        leaf_thetas=None):
    """
    Applies a batch of inverse kinematics transfoirm to the joints

    Parameters
    ----------
    pose_skeleton : torch.tensor BxNx3
        Locations of estimated pose skeleton.
    global_orient : torch.tensor Bx1x3x3
        Tensor of global rotation matrices
    phis : torch.tensor BxNx2
        The rotation on bone axis parameters
    rest_pose : torch.tensor Bx(N+1)x3
        Locations of rest_pose. (Template Pose)
    children: dict
        The dictionary that describes the kinematic chidrens for the model
    parents : torch.tensor Bx(N+1)
        The kinematic tree of each object
    dtype : torch.dtype, optional:
        The data type of the created tensors, the default is torch.float32

    Returns
    -------
    rot_mats: torch.tensor Bx(N+1)x3x3
        The rotation matrics of each joints
    rel_transforms : torch.tensor Bx(N+1)x4x4
        The relative (with respect to the root joint) rigid transformations
        for all the joints
    """
    batch_size = pose_skeleton.shape[0]
    device = pose_skeleton.device

    rel_rest_pose = rest_pose.clone()
    rel_rest_pose[:, 1:] -= rest_pose[:, parents[1:]].clone()
    rel_rest_pose = torch.unsqueeze(rel_rest_pose, dim=-1)

    # rotate the T pose
    rotate_rest_pose = torch.zeros_like(rel_rest_pose)
    # set up the root
    rotate_rest_pose[:, 0] = rel_rest_pose[:, 0]

    rel_pose_skeleton = torch.unsqueeze(pose_skeleton.clone(), dim=-1).detach()
    rel_pose_skeleton[:, 1:] = rel_pose_skeleton[:, 1:] - rel_pose_skeleton[:, parents[1:]].clone()
    rel_pose_skeleton[:, 0] = rel_rest_pose[:, 0]

    # the predicted final pose
    final_pose_skeleton = torch.unsqueeze(pose_skeleton.clone(), dim=-1)
    final_pose_skeleton = final_pose_skeleton - final_pose_skeleton[:, 0:1] + rel_rest_pose[:, 0:1]

    rel_rest_pose = rel_rest_pose
    rel_pose_skeleton = rel_pose_skeleton
    final_pose_skeleton = final_pose_skeleton
    rotate_rest_pose = rotate_rest_pose

    assert phis.dim() == 3
    phis = phis / (torch.norm(phis, dim=2, keepdim=True) + 1e-8)

    if train:
        global_orient_mat = batch_get_pelvis_orient(
            rel_pose_skeleton.clone(), rel_rest_pose.clone(), parents, children, dtype)
    else:
        global_orient_mat = batch_get_pelvis_orient_svd(
            rel_pose_skeleton.clone(), rel_rest_pose.clone(), parents, children, dtype)

    rot_mat_chain = [global_orient_mat]
    rot_mat_local = [global_orient_mat]
    # leaf nodes rot_mats
    if leaf_thetas is not None:
        leaf_cnt = 0
        leaf_rot_mats = leaf_thetas.view([batch_size, 5, 3, 3])

    for i in range(1, parents.shape[0]):
        if children[i] == -1:
            # leaf nodes
            if leaf_thetas is not None:
                rot_mat = leaf_rot_mats[:, leaf_cnt, :, :]
                leaf_cnt += 1

                rotate_rest_pose[:, i] = rotate_rest_pose[:, parents[i]] + torch.matmul(
                    rot_mat_chain[parents[i]],
                    rel_rest_pose[:, i]
                )

                rot_mat_chain.append(torch.matmul(
                    rot_mat_chain[parents[i]],
                    rot_mat))
                rot_mat_local.append(rot_mat)
        elif children[i] == -3:
            # three children
            rotate_rest_pose[:, i] = rotate_rest_pose[:, parents[i]] + torch.matmul(
                rot_mat_chain[parents[i]],
                rel_rest_pose[:, i]
            )

            spine_child = []
            for c in range(1, parents.shape[0]):
                if parents[c] == i and c not in spine_child:
                    spine_child.append(c)

            # original
            spine_child = []
            for c in range(1, parents.shape[0]):
                if parents[c] == i and c not in spine_child:
                    spine_child.append(c)

            children_final_loc = []
            children_rest_loc = []
            for c in spine_child:
                temp = final_pose_skeleton[:, c] - rotate_rest_pose[:, i]
                children_final_loc.append(temp)

                children_rest_loc.append(rel_rest_pose[:, c].clone())

            rot_mat = batch_get_3children_orient_svd(
                children_final_loc, children_rest_loc,
                rot_mat_chain[parents[i]], spine_child, dtype)

            rot_mat_chain.append(
                torch.matmul(
                    rot_mat_chain[parents[i]],
                    rot_mat)
            )
            rot_mat_local.append(rot_mat)
        else:
            # (B, 3, 1)
            rotate_rest_pose[:, i] = rotate_rest_pose[:, parents[i]] + torch.matmul(
                rot_mat_chain[parents[i]],
                rel_rest_pose[:, i]
            )
            # (B, 3, 1)
            child_final_loc = final_pose_skeleton[:, children[i]] - rotate_rest_pose[:, i]

            if not train:
                orig_vec = rel_pose_skeleton[:, children[i]]
                template_vec = rel_rest_pose[:, children[i]]
                norm_t = torch.norm(template_vec, dim=1, keepdim=True)
                orig_vec = orig_vec * norm_t / torch.norm(orig_vec, dim=1, keepdim=True)

                diff = torch.norm(child_final_loc - orig_vec, dim=1, keepdim=True)
                big_diff_idx = torch.where(diff > 15 / 1000)[0]

                child_final_loc[big_diff_idx] = orig_vec[big_diff_idx]

            child_final_loc = torch.matmul(
                rot_mat_chain[parents[i]].transpose(1, 2),
                child_final_loc)

            child_rest_loc = rel_rest_pose[:, children[i]]
            # (B, 1, 1)
            child_final_norm = torch.norm(child_final_loc, dim=1, keepdim=True)
            child_rest_norm = torch.norm(child_rest_loc, dim=1, keepdim=True)

            child_final_norm = torch.norm(child_final_loc, dim=1, keepdim=True)

            # (B, 3, 1)
            axis = torch.cross(child_rest_loc, child_final_loc, dim=1)
            axis_norm = torch.norm(axis, dim=1, keepdim=True)

            # (B, 1, 1)
            cos = torch.sum(child_rest_loc * child_final_loc, dim=1, keepdim=True) / (child_rest_norm * child_final_norm + 1e-8)
            sin = axis_norm / (child_rest_norm * child_final_norm + 1e-8)

            # (B, 3, 1)
            axis = axis / (axis_norm + 1e-8)

            # Convert location revolve to rot_mat by rodrigues
            # (B, 1, 1)
            rx, ry, rz = torch.split(axis, 1, dim=1)
            zeros = torch.zeros((batch_size, 1, 1), dtype=dtype, device=device)

            K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
                .view((batch_size, 3, 3))
            ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
            rot_mat_loc = ident + sin * K + (1 - cos) * torch.bmm(K, K)

            # Convert spin to rot_mat
            # (B, 3, 1)
            spin_axis = child_rest_loc / child_rest_norm
            # (B, 1, 1)
            rx, ry, rz = torch.split(spin_axis, 1, dim=1)
            zeros = torch.zeros((batch_size, 1, 1), dtype=dtype, device=device)
            K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
                .view((batch_size, 3, 3))
            ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
            # (B, 1, 1)
            cos, sin = torch.split(phis[:, i - 1], 1, dim=1)
            cos = torch.unsqueeze(cos, dim=2)
            sin = torch.unsqueeze(sin, dim=2)
            rot_mat_spin = ident + sin * K + (1 - cos) * torch.bmm(K, K)
            rot_mat = torch.matmul(rot_mat_loc, rot_mat_spin)

            rot_mat_chain.append(torch.matmul(
                rot_mat_chain[parents[i]],
                rot_mat))
            rot_mat_local.append(rot_mat)

    # (B, K + 1, 3, 3)
    rot_mats = torch.stack(rot_mat_local, dim=1)

    return rot_mats, rotate_rest_pose.squeeze(-1)


def batch_get_pelvis_orient_svd(rel_pose_skeleton, rel_rest_pose, parents, children, dtype):
    pelvis_child = [int(children[0])]
    for i in range(1, parents.shape[0]):
        if parents[i] == 0 and i not in pelvis_child:
            pelvis_child.append(i)

    rest_mat = []
    target_mat = []
    for child in pelvis_child:
        rest_mat.append(rel_rest_pose[:, child].clone())
        target_mat.append(rel_pose_skeleton[:, child].clone())

    rest_mat = torch.cat(rest_mat, dim=2)
    target_mat = torch.cat(target_mat, dim=2)
    S = rest_mat.bmm(target_mat.transpose(1, 2))

    mask_zero = S.sum(dim=(1, 2))

    S_non_zero = S[mask_zero != 0].reshape(-1, 3, 3)

    U, _, V = torch.svd(S_non_zero)

    rot_mat = torch.zeros_like(S)
    rot_mat[mask_zero == 0] = torch.eye(3, device=S.device)

    rot_mat_non_zero = torch.bmm(V, U.transpose(1, 2))
    rot_mat[mask_zero != 0] = rot_mat_non_zero

    assert torch.sum(torch.isnan(rot_mat)) == 0, ('rot_mat', rot_mat)

    return rot_mat


def batch_get_pelvis_orient(rel_pose_skeleton, rel_rest_pose, parents, children, dtype):
    batch_size = rel_pose_skeleton.shape[0]
    device = rel_pose_skeleton.device

    assert children[0] == 3
    pelvis_child = [int(children[0])]
    for i in range(1, parents.shape[0]):
        if parents[i] == 0 and i not in pelvis_child:
            pelvis_child.append(i)

    spine_final_loc = rel_pose_skeleton[:, int(children[0])].clone()
    spine_rest_loc = rel_rest_pose[:, int(children[0])].clone()
    spine_norm = torch.norm(spine_final_loc, dim=1, keepdim=True)
    spine_norm = spine_final_loc / (spine_norm + 1e-8)

    rot_mat_spine = vectors2rotmat(spine_rest_loc, spine_final_loc, dtype)

    assert torch.sum(torch.isnan(rot_mat_spine)
                     ) == 0, ('rot_mat_spine', rot_mat_spine)
    center_final_loc = 0
    center_rest_loc = 0
    for child in pelvis_child:
        if child == int(children[0]):
            continue
        center_final_loc = center_final_loc + rel_pose_skeleton[:, child].clone()
        center_rest_loc = center_rest_loc + rel_rest_pose[:, child].clone()
    center_final_loc = center_final_loc / (len(pelvis_child) - 1)
    center_rest_loc = center_rest_loc / (len(pelvis_child) - 1)

    center_rest_loc = torch.matmul(rot_mat_spine, center_rest_loc)

    center_final_loc = center_final_loc - torch.sum(center_final_loc * spine_norm, dim=1, keepdim=True) * spine_norm
    center_rest_loc = center_rest_loc - torch.sum(center_rest_loc * spine_norm, dim=1, keepdim=True) * spine_norm

    center_final_loc_norm = torch.norm(center_final_loc, dim=1, keepdim=True)
    center_rest_loc_norm = torch.norm(center_rest_loc, dim=1, keepdim=True)

    # (B, 3, 1)
    axis = torch.cross(center_rest_loc, center_final_loc, dim=1)
    axis_norm = torch.norm(axis, dim=1, keepdim=True)

    # (B, 1, 1)
    cos = torch.sum(center_rest_loc * center_final_loc, dim=1, keepdim=True) / (center_rest_loc_norm * center_final_loc_norm + 1e-8)
    sin = axis_norm / (center_rest_loc_norm * center_final_loc_norm + 1e-8)

    assert torch.sum(torch.isnan(cos)
                     ) == 0, ('cos', cos)
    assert torch.sum(torch.isnan(sin)
                     ) == 0, ('sin', sin)
    # (B, 3, 1)
    axis = axis / (axis_norm + 1e-8)

    # Convert location revolve to rot_mat by rodrigues
    # (B, 1, 1)
    rx, ry, rz = torch.split(axis, 1, dim=1)
    zeros = torch.zeros((batch_size, 1, 1), dtype=dtype, device=device)

    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))
    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat_center = ident + sin * K + (1 - cos) * torch.bmm(K, K)

    rot_mat = torch.matmul(rot_mat_center, rot_mat_spine)

    return rot_mat


def batch_get_3children_orient_svd(rel_pose_skeleton, rel_rest_pose, rot_mat_chain_parent, children_list, dtype):
    rest_mat = []
    target_mat = []
    for c, child in enumerate(children_list):
        if isinstance(rel_pose_skeleton, list):
            target = rel_pose_skeleton[c].clone()
            template = rel_rest_pose[c].clone()
        else:
            target = rel_pose_skeleton[:, child].clone()
            template = rel_rest_pose[:, child].clone()

        target = torch.matmul(
            rot_mat_chain_parent.transpose(1, 2),
            target)

        target_mat.append(target)
        rest_mat.append(template)

    rest_mat = torch.cat(rest_mat, dim=2)
    target_mat = torch.cat(target_mat, dim=2)
    S = rest_mat.bmm(target_mat.transpose(1, 2))

    U, _, V = torch.svd(S)

    rot_mat = torch.bmm(V, U.transpose(1, 2))
    assert torch.sum(torch.isnan(rot_mat)) == 0, ('3children rot_mat', rot_mat)
    return rot_mat


def vectors2rotmat(vec_rest, vec_final, dtype):
    batch_size = vec_final.shape[0]
    device = vec_final.device

    # (B, 1, 1)
    vec_final_norm = torch.norm(vec_final, dim=1, keepdim=True)
    vec_rest_norm = torch.norm(vec_rest, dim=1, keepdim=True)

    # (B, 3, 1)
    axis = torch.cross(vec_rest, vec_final, dim=1)
    axis_norm = torch.norm(axis, dim=1, keepdim=True)

    # (B, 1, 1)
    cos = torch.sum(vec_rest * vec_final, dim=1, keepdim=True) / (vec_rest_norm * vec_final_norm + 1e-8)
    sin = axis_norm / (vec_rest_norm * vec_final_norm + 1e-8)

    # (B, 3, 1)
    axis = axis / (axis_norm + 1e-8)

    # Convert location revolve to rot_mat by rodrigues
    # (B, 1, 1)
    rx, ry, rz = torch.split(axis, 1, dim=1)
    zeros = torch.zeros((batch_size, 1, 1), dtype=dtype, device=device)

    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))
    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat_loc = ident + sin * K + (1 - cos) * torch.bmm(K, K)

    return rot_mat_loc


def aa_to_rot_mat(aa):
    """

    Args:
        aa : size is [B, 3]
    Returns:
        rotmat : size is [B, 3, 3]
    """
    angle = torch.norm(aa, dim=1, keepdim=True)
    axis = aa / (angle + 1e-9)

    sin = torch.sin(angle)[:, :, None]
    cos = torch.cos(angle)[:, :, None]

    rx, ry, rz = axis[:, 0:1, None], axis[:, 1:2, None], axis[:, 2:3, None]
    zeros = torch.zeros_like(rx)

    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((aa.shape[0], 3, 3))
    ident = torch.eye(3, dtype=rx.dtype, device=rx.device)[None, :, :]

    rotmat = ident + sin * K + (1 - cos) * torch.bmm(K, K)

    return rotmat


def rotmat_to_quat(rotmat):
    """Convert rotation matrix to quaternion coefficients.
    Args:
        rotmat: size is [B, 3, 3]
    Returns:
        Quaternion: size is [B, 4] <===> (w, x, y, z)
    """
    quaternion = torch.zeros([rotmat.size(0), 4], device=rotmat.device)
    trace = rotmat[:, 0, 0] + rotmat[:, 1, 1] + rotmat[:, 2, 2]
    flag = 1 + trace > 0
    s = torch.zeros_like(trace)

    # pos
    s[flag] = 2 * torch.sqrt(1 + trace[flag]) + 1e-16
    s_pos = s[flag]
    quaternion[flag, 0] = s_pos / 4
    quaternion[flag, 1] = (rotmat[flag, 2, 1] - rotmat[flag, 1, 2]) / s_pos
    quaternion[flag, 2] = (rotmat[flag, 0, 2] - rotmat[flag, 2, 0]) / s_pos
    quaternion[flag, 3] = (rotmat[flag, 1, 0] - rotmat[flag, 0, 1]) / s_pos

    # neg
    diag = torch.stack([rotmat[:, 0, 0], rotmat[:, 1, 1], rotmat[:, 2, 2]])
    max_val, max_ind = torch.max(diag, dim=0)

    s[~flag] = 2 * torch.sqrt(1 - trace[~flag] + 2 * max_val[~flag]) + 1e-16

    f0 = ~flag * (max_ind == 0)
    s0 = s[f0]
    quaternion[f0, 0] = (rotmat[f0, 2, 1] - rotmat[f0, 1, 2]) / s0
    quaternion[f0, 1] = s0 / 4
    quaternion[f0, 2] = (rotmat[f0, 0, 1] + rotmat[f0, 1, 0]) / s0
    quaternion[f0, 3] = (rotmat[f0, 0, 2] + rotmat[f0, 2, 0]) / s0

    f1 = ~flag * (max_ind == 1)
    s1 = s[f1]
    quaternion[f1, 0] = (rotmat[f1, 0, 2] - rotmat[f1, 2, 0]) / s1
    quaternion[f1, 1] = (rotmat[f1, 0, 1] + rotmat[f1, 1, 0]) / s1
    quaternion[f1, 2] = s1 / 4
    quaternion[f1, 3] = (rotmat[f1, 1, 2] + rotmat[f1, 2, 1]) / s1

    f2 = ~flag * (max_ind == 2)
    s2 = s[f2]
    quaternion[f2, 0] = (rotmat[f2, 1, 0] - rotmat[f2, 0, 1]) / s2
    quaternion[f2, 1] = (rotmat[f2, 0, 2] + rotmat[f2, 2, 0]) / s2
    quaternion[f2, 2] = (rotmat[f2, 1, 2] + rotmat[f2, 2, 1]) / s2
    quaternion[f2, 3] = s2 / 4

    return quaternion


def quat_to_rotmat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [B, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat / (norm_quat.norm(p=2, dim=1, keepdim=True) + 1e-8)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz,
                          2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx,
                          2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat


def matmul_d(R1, R2, a1=1):
    """ Multiply matrices with derivation

    Args:
        R1 (torch.tensor)
        R2 (torch.tensor)
    Returns:
        R=R1*R2 (torch.tensor)
    """
    R = a1 * R1.matmul(R2)
    dR1_dq = getattr(R1, 'dq', None)
    if dR1_dq is None:
        return R
    dR2_dq = getattr(R2, 'dq', None)
    if dR2_dq is not None:
        R.dq = a1 * R1.dq.matmul(R2[:, None, ...]) + a1 * R1[:, None, ...].matmul(R2.dq)
    else:
        R.dq = a1 * R1.dq.matmul(R2[:, None, ...])
    # d_set1 = R1.d_set
    # d_set2 = getattr(R2, 'd_set', set())
    # d_set = set.union(d_set1, d_set2)

    # for name in d_set:
    #     dR1_dname = getattr(R1, f'd{name}', None)
    #     dR2_dname = getattr(R2, f'd{name}', None)
    #     assert dR1_dname is not None or dR2_dname is not None, (dR1_dname, dR2_dname, d_set, d_set1, d_set2)
    #     dR_dname = None
    #     if dR1_dname is not None:
    #         dR_dname = a1 * dR1_dname.matmul(R2)
    #     if dR2_dname is not None:
    #         if dR_dname is not None:
    #             dR_dname = dR_dname + a1 * R1.matmul(dR2_dname)
    #         else:
    #             dR_dname = a1 * R1.matmul(dR2_dname)
    #     setattr(R, f'd{name}', dR_dname)
    # R.d_set = d_set

    return R


def matadd_d(R1, R2, a1=1, a2=1):
    """ Add matrices with derivation

    Args:
        R1 (torch.tensor)
        R2 (torch.tensor)
    Returns:
        R=R1+R2 (torch.tensor)
    """
    R = a1 * R1 + a2 * R2
    dR1_dq = getattr(R1, 'dq', None)
    if dR1_dq is None:
        return R
    dR2_dq = getattr(R2, 'dq', None)
    if dR2_dq is not None:
        R.dq = a1 * R1.dq + a2 * R2.dq
    else:
        R.dq = a1 * R1.dq
    # d_set1 = R1.d_set
    # d_set2 = getattr(R2, 'd_set', set())
    # d_set = set.union(d_set1, d_set2)

    # for name in d_set:
    #     dR1_dname = getattr(R1, f'd{name}', None)
    #     dR2_dname = getattr(R2, f'd{name}', None)
    #     assert dR1_dname is not None or dR2_dname is not None, (dR1_dname, dR2_dname, d_set, d_set1, d_set2)
    #     dR_dname = None
    #     if dR1_dname is not None:
    #         dR_dname = a1 * dR1_dname
    #     if dR2_dname is not None:
    #         if dR_dname is not None:
    #             dR_dname = dR_dname + a2 * dR2_dname
    #         else:
    #             dR_dname = a2 * dR2_dname
    #     setattr(R, f'd{name}', dR_dname)
    # R.d_set = d_set

    return R


def matmul_add_d(R1, R2, t, a1=1, a2=1):
    """ Multiply and Add matrices with derivation

    Args:
        R1 (torch.tensor)
        R2 (torch.tensor)
        t  (torch.tensor)
    Returns:
        R=R1*R2 + t (torch.tensor)
    """
    R = a1 * R1.matmul(R2) + a2 * t
    dR1_dq = getattr(R1, 'dq', None)
    if dR1_dq is None:
        return R
    dR2_dq = getattr(R2, 'dq', None)
    dt_dq = getattr(t, 'dq', None)
    if dR2_dq is not None:
        R.dq = a1 * R1.dq.matmul(R2[:, None, ...]) + a1 * R1[:, None, ...].matmul(R2.dq)
    else:
        R.dq = a1 * R1.dq.matmul(R2[:, None, ...])
    if dt_dq is not None:
        R.dq = R.dq + a2 * t.dq
    # d_set1 = R1.d_set
    # d_set2 = getattr(R2, 'd_set', set())
    # d_set3 = getattr(t, 'd_set', set())
    # d_set = set.union(d_set1, d_set2, d_set3)

    # for name in d_set:
    #     dR1_dname = getattr(R1, f'd{name}', None)
    #     dR2_dname = getattr(R2, f'd{name}', None)
    #     dt_name = getattr(t, f'd{name}', None)
    #     assert dR1_dname is not None or dR2_dname is not None or dt_name is not None
    #     dR_dname = None
    #     if dR1_dname is not None:
    #         dR_dname = a1 * dR1_dname.matmul(R2)
    #     if dR2_dname is not None:
    #         if dR_dname is not None:
    #             dR_dname = dR_dname + a1 * R1.matmul(dR2_dname)
    #         else:
    #             dR_dname = a1 * R1.matmul(dR2_dname)
    #     if dt_name is not None:
    #         dR_dname = dR_dname + a2 * dt_name
    #     setattr(R, f'd{name}', dR_dname)
    # R.d_set = d_set

    return R


def transform_relmat_list(R_list, t_list):
    """ Creates relative transformation matrices for list of R, t

    Args:
        R_list (List of sympy.Matrix): [Bx3x3] * K
        t_list (List of sympy.Matrix): [Bx3x1] * K

    Returns:
        T_list : [4x4]
    """
    T_list = [transform_relmat(r, t) for r, t in zip(R_list, t_list)]
    return T_list


def transform_relmat(R, p, t):
    """ Creates transformation matrices

    Args:
        R (sympy.Matrix): Bx3x3
        t (sympy.Matrix): Bx3x1
    Returns:
        T (sympy.Matrix): Bx4x4
    """
    t = p - R.matmul(t)
    T = torch.cat([F.pad(R, [0, 0, 0, 1]),
                   F.pad(t, [0, 0, 0, 1], value=1)], dim=-1)
    return T


def vec2skewmat_d(vec, total_variable):
    """
    Args:
        vec torch.tensor: Bx3x1
    """
    kx, ky, kz = torch.split(vec, 1, dim=1)
    # d_set = vec.d_set
    zeros = torch.zeros((vec.shape[0], 1, 1), dtype=vec.dtype, device=vec.device)
    zeros_d = torch.zeros((vec.shape[0], total_variable, 1, 1), dtype=vec.dtype, device=vec.device)

    mat = torch.cat([
        zeros, -kz, ky,
        kz, zeros, -kx,
        -ky, kx, zeros
    ], dim=1).reshape(vec.shape[0], 3, 3)
    dq = getattr(vec, 'dq', None)
    if dq is None:
        return mat
    # dmat_dq = torch.zeros(vec.shape[0], total_variable, 3, 3, device=vec.device)
    dkx, dky, dkz = torch.split(vec.dq, 1, dim=2)
    mat.dq = torch.cat([
        zeros_d, -dkz, dky,
        dkz, zeros_d, -dkx,
        -dky, dkx, zeros_d
    ], dim=2).reshape(vec.shape[0], total_variable, 3, 3)
    # for name in d_set:
    #     dvec_dname = getattr(vec, f'd{name}')
    #     dkx_dname, dky_dname, dkz_dname = torch.split(dvec_dname, 1, dim=1)
    #     dmat_dname = torch.cat([
    #         zeros, -dkz_dname, dky_dname,
    #         dkz_dname, zeros, -dkx_dname,
    #         -dky_dname, dkx_dname, zeros
    #     ], dim=1).reshape(vec.shape[0], 3, 3)
    #     setattr(mat, f'd{name}', dmat_dname)
    # mat.d_set = vec.d_set
    return mat


def vec2skewmat(vec):
    """
    Args:
        vec torch.tensor: Bx3x1
    """
    kx, ky, kz = torch.split(vec, 1, dim=1)
    zeros = torch.zeros((vec.shape[0], 1, 1),
                        dtype=vec.dtype, device=vec.device)

    mat = torch.cat([
        zeros, -kz, ky,
        kz, zeros, -kx,
        -ky, kx, zeros
    ], dim=1).reshape(vec.shape[0], 3, 3)

    return mat
