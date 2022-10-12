
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
import random

import cv2
import numpy as np
import tensorflow as tf
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
from skimage.util.shape import view_as_windows


def get_image(filename):
    image = cv2.imread(filename)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def do_augmentation(scale_factor=0.3, color_factor=0.2):
    scale = random.uniform(1.2, 1.2 + scale_factor)
    # scale = np.clip(np.random.randn(), 0.0, 1.0) * scale_factor + 1.2
    rot = 0  # np.clip(np.random.randn(), -2.0, 2.0) * aug_config.rot_factor if random.random() <= aug_config.rot_aug_rate else 0
    do_flip = False  # aug_config.do_flip_aug and random.random() <= aug_config.flip_aug_rate
    c_up = 1.0 + color_factor
    c_low = 1.0 - color_factor
    color_scale = [random.uniform(c_low, c_up), random.uniform(c_low, c_up), random.uniform(c_low, c_up)]
    return scale, rot, do_flip, color_scale


def trans_point2d(pt_2d, trans):
    src_pt = np.array([pt_2d[0], pt_2d[1], 1.]).T
    dst_pt = np.dot(trans, src_pt)
    return dst_pt[0:2]


def rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)


def gen_trans_from_patch_cv(c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot, inv=False):
    # augment size with scale
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.zeros(2)
    src_center[0] = c_x
    src_center[1] = c_y  # np.array([c_x, c_y], dtype=np.float32)
    # augment rotation
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def generate_patch_image_cv(cvimg, c_x, c_y, bb_width, bb_height, patch_width, patch_height, do_flip, scale, rot):
    img = cvimg.copy()
    img_height, img_width, img_channels = img.shape

    if do_flip:
        img = img[:, ::-1, :]
        c_x = img_width - c_x - 1

    trans = gen_trans_from_patch_cv(c_x, c_y, bb_width, bb_height, patch_width, patch_height, scale, rot, inv=False)

    img_patch = cv2.warpAffine(img, trans, (int(patch_width), int(patch_height)),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    return img_patch, trans


def crop_image(image, kp_2d, center_x, center_y, width, height, patch_width, patch_height, do_augment):

    # get augmentation params
    if do_augment:
        scale, rot, do_flip, _ = do_augmentation()
    else:
        scale, rot, do_flip, _ = 1.3, 0, False, [1.0, 1.0, 1.0]

    # generate image patch
    image, trans = generate_patch_image_cv(
        image,
        center_x,
        center_y,
        width,
        height,
        patch_width,
        patch_height,
        do_flip,
        scale,
        rot
    )

    for n_jt in range(kp_2d.shape[0]):
        kp_2d[n_jt] = trans_point2d(kp_2d[n_jt], trans)

    return image, kp_2d, trans


def transfrom_keypoints(kp_2d, center_x, center_y, width, height, patch_width, patch_height, do_augment, scale=None):

    if do_augment:
        scale_new, rot, _, _ = do_augmentation()
        assert 1 == 2
    else:
        scale_new, rot, _, _ = 1.2, 0, False, [1.0, 1.0, 1.0]

    if scale is None:
        scale = scale_new

    # generate transformation
    trans = gen_trans_from_patch_cv(
        center_x,
        center_y,
        width,
        height,
        patch_width,
        patch_height,
        scale,
        rot,
        inv=False,
    )

    for n_jt in range(kp_2d.shape[0]):
        kp_2d[n_jt] = trans_point2d(kp_2d[n_jt].copy(), trans)

    return kp_2d, trans, scale


def get_image_crops(image_file, bboxes):
    image = cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB)
    crop_images = []
    for bb in bboxes:
        c_y, c_x = (bb[0] + bb[2]) // 2, (bb[1] + bb[3]) // 2
        h, w = bb[2] - bb[0], bb[3] - bb[1]
        w = h = np.where(w / h > 1, w, h)
        crop_image, _ = generate_patch_image_cv(
            cvimg=image.copy(),
            c_x=c_x,
            c_y=c_y,
            bb_width=w,
            bb_height=h,
            patch_width=224,
            patch_height=224,
            do_flip=False,
            scale=1.3,
            rot=0,
        )
        crop_image = convert_cvimg_to_tensor(crop_image)
        crop_images.append(crop_image)

    batch_image = torch.cat([x.unsqueeze(0) for x in crop_images])
    return batch_image


def get_single_image_crop(image, bbox, scale=1.3):
    if isinstance(image, str):
        if os.path.isfile(image):
            image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
        else:
            print(image)
            raise BaseException(image, 'is not a valid file!')
    elif isinstance(image, torch.Tensor):
        image = image.numpy()
    elif not isinstance(image, np.ndarray):
        raise('Unknown type for object', type(image))

    crop_image, _ = generate_patch_image_cv(
        cvimg=image.copy(),
        c_x=bbox[0],
        c_y=bbox[1],
        bb_width=bbox[2],
        bb_height=bbox[3],
        patch_width=224,
        patch_height=224,
        do_flip=False,
        scale=scale,
        rot=0,
    )

    crop_image = convert_cvimg_to_tensor(crop_image)

    return crop_image


def get_video_crop(image_names, bboxes, scale=1.3, color_jitter=False, factor=0.3, erase=None):
    crop_image_list = []
    for image_name, bbox in zip(image_names, bboxes):
        image = cv2.cvtColor(cv2.imread(image_name), cv2.COLOR_BGR2RGB)

        crop_image, _ = generate_patch_image_cv(
            cvimg=image.copy(),
            c_x=bbox[0],
            c_y=bbox[1],
            bb_width=bbox[2],
            bb_height=bbox[3],
            patch_width=224,
            patch_height=224,
            do_flip=False,
            scale=scale,
            rot=0,
        )
        if erase is not None and (np.random.rand() < erase['prob']):
            erased_ratio = np.random.rand() * erase['max_erase_part']
            crop_image = erase['erased_part'](crop_image, erased_ratio)

        crop_image = Image.fromarray(crop_image)
        crop_image_list.append(crop_image)

    if color_jitter:
        crop_image_list = do_color_jitter(crop_image_list, jitter_factor=factor)

    crop_video = [convert_cvimg_to_tensor(img) for img in crop_image_list]

    return crop_video


def do_color_jitter(image_list, jitter_factor=0.3):
    brightness, contrast, saturation, hue = _get_color_params(jitter_factor, jitter_factor, jitter_factor, jitter_factor)

    img_transforms = []

    if brightness is not None:
        img_transforms.append(lambda img: F.adjust_brightness(img, brightness))
    if saturation is not None:
        img_transforms.append(lambda img: F.adjust_saturation(img, saturation))
    if hue is not None:
        img_transforms.append(lambda img: F.adjust_hue(img, hue))
    if contrast is not None:
        img_transforms.append(lambda img: F.adjust_contrast(img, contrast))

    random.shuffle(img_transforms)

    jittered_video = []
    for img in image_list:
        jittered_image = img.copy()
        for func in img_transforms:
            jittered_image = func(jittered_image)
        jittered_video.append(jittered_image)
    return jittered_video


def _get_color_params(brightness, contrast, saturation, hue):
    if brightness > 0:
        brightness_factor = random.uniform(
            max(0, 1 - brightness), 1 + brightness)
    else:
        brightness_factor = None

    if contrast > 0:
        contrast_factor = random.uniform(
            max(0, 1 - contrast), 1 + contrast)
    else:
        contrast_factor = None

    if saturation > 0:
        saturation_factor = random.uniform(
            max(0, 1 - saturation), 1 + saturation)
    else:
        saturation_factor = None

    if hue > 0:
        hue_factor = random.uniform(-hue, hue)
    else:
        hue_factor = None
    return brightness_factor, contrast_factor, saturation_factor, hue_factor


def get_single_image_crop_demo(image, bbox, kp_2d, scale=1.2, crop_size=224):
    if isinstance(image, str):
        if os.path.isfile(image):
            image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
        else:
            print(image)
            raise BaseException(image, 'is not a valid file!')
    elif isinstance(image, torch.Tensor):
        image = image.numpy()
    elif not isinstance(image, np.ndarray):
        raise('Unknown type for object', type(image))

    crop_image, trans = generate_patch_image_cv(
        cvimg=image.copy(),
        c_x=bbox[0],
        c_y=bbox[1],
        bb_width=bbox[2],
        bb_height=bbox[3],
        patch_width=crop_size,
        patch_height=crop_size,
        do_flip=False,
        scale=scale,
        rot=0,
    )

    if kp_2d is not None:
        for n_jt in range(kp_2d.shape[0]):
            kp_2d[n_jt, :2] = trans_point2d(kp_2d[n_jt], trans)

    raw_image = crop_image.copy()

    crop_image = convert_cvimg_to_tensor(crop_image)

    return crop_image, raw_image, kp_2d


def read_image(filename):
    image = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    return convert_cvimg_to_tensor(image)


def convert_cvimg_to_tensor(image):
    transform = get_default_transform()
    image = transform(image)
    return image


def torch2numpy(image):
    image = image.detach().cpu()
    inv_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.255]
    )
    image = inv_normalize(image)
    image = image.clamp(0., 1.)
    image = image.numpy() * 255.
    image = np.transpose(image, (1, 2, 0))
    return image.astype(np.uint8)


def torch_vid2numpy(video):
    video = video.detach().cpu().numpy()
    # video = np.transpose(video, (0, 2, 1, 3, 4)) # NCTHW->NTCHW
    # Denormalize
    mean = np.array([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255])
    std = np.array([1 / 0.229, 1 / 0.224, 1 / 0.255])

    mean = mean[np.newaxis, np.newaxis, ..., np.newaxis, np.newaxis]
    std = std[np.newaxis, np.newaxis, ..., np.newaxis, np.newaxis]

    video = (video - mean) / std  # [:, :, i, :, :].sub_(mean[i]).div_(std[i]).clamp_(0., 1.).mul_(255.)
    video = video.clip(0., 1.) * 255
    video = video.astype(np.uint8)
    return video


def get_bbox_from_kp2d(kp_2d):
    # get bbox
    if len(kp_2d.shape) > 2:
        ul = np.array([kp_2d[:, :, 0].min(axis=1), kp_2d[:, :, 1].min(axis=1)])  # upper left
        lr = np.array([kp_2d[:, :, 0].max(axis=1), kp_2d[:, :, 1].max(axis=1)])  # lower right
    else:
        ul = np.array([kp_2d[:, 0].min(), kp_2d[:, 1].min()])  # upper left
        lr = np.array([kp_2d[:, 0].max(), kp_2d[:, 1].max()])  # lower right

    # ul[1] -= (lr[1] - ul[1]) * 0.10  # prevent cutting the head
    w = lr[0] - ul[0]
    h = lr[1] - ul[1]
    c_x, c_y = ul[0] + w / 2, ul[1] + h / 2
    # to keep the aspect ratio
    w = h = np.where(w / h > 1, w, h)
    w = h = h * 1.1

    bbox = np.array([c_x, c_y, w, h])  # shape = (4,N)
    return bbox


def normalize_2d_kp(kp_2d, crop_size=224, inv=False):
    # Normalize keypoints between -1, 1
    if not inv:
        ratio = 1.0 / crop_size
        kp_2d = 2.0 * kp_2d * ratio - 1.0
    else:
        ratio = 1.0 / crop_size
        kp_2d = (kp_2d + 1.0) / (2 * ratio)

    return kp_2d


def get_default_transform():
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    return transform


def split_into_chunks(db, seqlen, stride, filtered=False):
    vid_names = db['vid_name']
    if 'frame_id' in db.keys():
        frame_ids = db['frame_id']
    video_start_end_indices = []

    video_names, group = np.unique(vid_names, return_index=True)
    perm = np.argsort(group)
    video_names, group = video_names[perm], group[perm]

    indices = np.split(np.arange(0, vid_names.shape[0]), group[1:])

    for idx in range(len(video_names)):
        indexes = indices[idx]
        if indexes.shape[0] < seqlen:
            continue
        chunks = view_as_windows(indexes, (seqlen,), step=stride)
        start_finish = chunks[:, (0, -1)].tolist()
        if filtered:
            start_finish = filter_ids(frame_ids, start_finish, seqlen)
            if 'joints3D' in db.keys():
                start_finish = filter_abnormal_motions(db['joints3D'], start_finish, seqlen)

        video_start_end_indices += start_finish

    return video_start_end_indices


def filter_ids(frame_ids, start_finish, seqlen):
    updated_start_finish = []
    for s_id, e_id in start_finish:
        if frame_ids[s_id] + seqlen - 1 == frame_ids[e_id]:
            updated_start_finish.append([s_id, e_id])

    return updated_start_finish


def filter_abnormal_motions(joints3D, start_finish, seqlen):
    updated_start_finish = []
    for s_id, e_id in start_finish:
        kp_3d = joints3D[s_id:e_id + 1]

        dkp_dt = (kp_3d[1:] - kp_3d[:-1]) * 25
        dkp_dt2 = (dkp_dt[1:] - dkp_dt[:-1]) * 25
        if np.max(dkp_dt2) < 60:
            updated_start_finish.append([s_id, e_id])

    return updated_start_finish


def split_into_chunks_tcmr(vid_names, seqlen, stride, is_train=True, match_vibe=True):
    video_start_end_indices = []
    video_names, group = np.unique(vid_names, return_index=True)
    perm = np.argsort(group)
    video_names, group = video_names[perm], group[perm]

    indices = np.split(np.arange(0, vid_names.shape[0]), group[1:])
    # import pdb; pdb.set_trace()
    for idx in range(len(video_names)):
        indexes = indices[idx]
        if indexes.shape[0] < seqlen:
            continue
        chunks = view_as_windows(indexes, (seqlen,), step=stride)
        start_finish = chunks[:, (0, -1)].tolist()
        if stride != seqlen:
            if match_vibe:
                vibe_chunks = view_as_windows(indexes, (16,), step=16)
                for j in range(1, len(start_finish) + 1):
                    if start_finish[-j][-1] == vibe_chunks[-1][-1]:
                        if j != 1:
                            start_finish = start_finish[:-j + 1]
                        break

            d = start_finish[0][0]
            for j in range(int(seqlen / 2)):
                if is_train:
                    dummy = start_finish[0]
                else:
                    dummy = [d + j, d + j]
                start_finish.insert(j, dummy)
            d = start_finish[-1][0]
            for j in range(int(seqlen / 2 + 0.5) - 1):
                if is_train:
                    dummy = start_finish[-1]
                else:
                    dummy = [d + int(seqlen / 2) + j + 1, d + int(seqlen / 2) + j + 1]
                start_finish.append(dummy)
        video_start_end_indices += start_finish

    return video_start_end_indices


def read_image_tfrec(filename, f_ids):
    desc = {
        'image/encoded': 'byte',
        'image/xys': 'float'
    }

    filename, s_id = filename.decode('ascii').split('tfrecord')
    filename = filename + 'tfrecord'
    s_id = int(s_id.split('-')[-1])
    #  = list(tfrecord.tfrecord_loader(filename, None, desc))
    loader = list(tf.python_io.tf_record_iterator(filename))
    # sess = tf.Session()

    serialized_ex = loader[s_id]
    example = tf.train.Example()
    example.ParseFromString(serialized_ex)
    images_data = example.features.feature['image/encoded'].bytes_list.value
    video = []
    for f_id in f_ids:
        # image = np.expand_dims(sess.run(tf.image.decode_jpeg(images_data[int(f_id)], channels=3)), axis=0)
        image = cv2.cvtColor(cv2.imdecode(np.fromstring(images_data[int(f_id)], dtype=np.uint8), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
        # image = sess.run(tf.image.decode_jpeg(images_data[int(f_id)], channels=3))
        image = convert_cvimg_to_tensor(image)
        # image = torch.from_numpy(sess.run(tf.image.decode_jpeg(images_data[int(f_id)], channels=3)))
        video.append(image)
    return video
    # N = int(example.features.feature['meta/N'].int64_list.value[0])
