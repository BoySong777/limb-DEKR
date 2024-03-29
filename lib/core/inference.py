# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# The code is based on HigherHRNet-Human-Pose-Estimation.
# (https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation)
# Modified by Zigang Geng (zigang@mail.ustc.edu.cn).
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn.functional as F

from dataset.transforms import FLIP_CONFIG
from utils.transforms import up_interpolate


def get_locations(output_h, output_w, device):
    shifts_x = torch.arange(
        0, output_w, step=1,
        dtype=torch.float32, device=device
    )
    shifts_y = torch.arange(
        0, output_h, step=1,
        dtype=torch.float32, device=device
    )
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    locations = torch.stack((shift_x, shift_y), dim=1)

    return locations


def get_reg_poses(offset, num_joints):
    _, h, w = offset.shape
    offset = offset.permute(1, 2, 0).reshape(h*w, num_joints, 2)
    locations = get_locations(h, w, offset.device)
    locations = locations[:, None, :].expand(-1, num_joints, -1)
    poses = locations - offset

    return poses


def offset_to_pose(offset, flip=True, flip_index=None):
    num_offset, h, w = offset.shape[1:]
    num_joints = int(num_offset/2)
    reg_poses = get_reg_poses(offset[0], num_joints)

    if flip:
        reg_poses = reg_poses[:, flip_index, :]
        reg_poses[:, :, 0] = w - reg_poses[:, :, 0] - 1

    reg_poses = reg_poses.contiguous().view(h*w, 2*num_joints).permute(1,0)
    reg_poses = reg_poses.contiguous().view(1,-1,h,w).contiguous()

    return reg_poses

def convert_offset(offset, limbs_offset, limbs_context, keypoints_restore_index):
    n, c, w, h = offset.shape
    # 先把相对偏移的偏移改成绝对位置
    locate_map = offset_to_pose(limbs_offset, flip=False)
    # 接下来把位置值的范围调整到[-1, 1] 公式为：（x,y）*2/((w,h)-1) - 1
    # 由于图片中的坐标(x, y),对应的是tensor中的(h, w)，所以这里要写成[2.0/(h-1), 2.0/(w-1)]
    param = [2.0/(h-1), 2.0/(w-1)]
    param_tensor = torch.tensor(param).reshape(1, 2, 1, 1).repeat(1, 5, 1, 1).cuda(non_blocking=True)
    locate_map = locate_map * param_tensor - 1.0
    point = 0
    # 对肢体逐一进行处理
    for idx, num in enumerate(limbs_context):
        # 把相对于肢体中心点的关键点偏移量挪到肢体中心点对应的人体中心点下，目的是为了方便关键点偏移量和肢体中心点偏移量相加。
        # 之前想把 去除的local_map调换一下位置，因为图像中的坐标(x,y),分别对应着tensor的水平方向和垂直方向，也就是第二维和第一维，
        # 所以调换一下才能取出正确的值，但是后来发现，在这个函数中，它的第一维和第二维和图片中的坐标是对应着的，所以根本不需要调换位置。
        offset[:, point:point+num*2] = F.grid_sample(offset[:, point:point+num*2],
                                                          locate_map[:, idx*2:(idx+1)*2].permute(0, 2, 3, 1), mode='nearest', padding_mode='zeros',
                                                     align_corners=None)
        # 肢体中心点偏移量和关键点偏移量相加，得到关键点相对于人体中心点的偏移量。
        offset[:, point:point + num * 2] = offset[:, point:point+num*2] + limbs_offset[:, idx*2:(idx+1)*2].repeat(1, num, 1, 1)
        point += num*2
    offset = offset.view(n, int(c/2), 2, w, h)
    offset = offset[:, keypoints_restore_index]
    offset = offset.view(n, c, w, h)
    return offset
def get_multi_stage_outputs(cfg, model, image, with_flip=False):
    # forward
    heatmap, offset, limbs_offset = model(image)
    # 将肢体中心点的偏移量和关键点的偏移量做拼接
    # offset = convert_offset(offset, limbs_offset, cfg.DATASET.LIMBS_CONTEXT, cfg.DATASET.KEYPOINTS_RESTORE_INDEX)
    posemap = offset_to_pose(offset, flip=False)

    if with_flip:
        if 'coco' in cfg.DATASET.DATASET:
            flip_index_heat = FLIP_CONFIG['COCO_WITH_ONLY_CENTER']
            flip_index_offset = FLIP_CONFIG['COCO']
        elif 'crowd_pose' in cfg.DATASET.DATASET:
            flip_index_heat = FLIP_CONFIG['CROWDPOSE_WITH_ONLY_CENTER']
            flip_index_offset = FLIP_CONFIG['CROWDPOSE']
        else:
            raise ValueError('Please implement flip_index \
                for new dataset: %s.' % cfg.DATASET.DATASET)

        image = torch.flip(image, [3])
        image[:, :, :, :-3] = image[:, :, :, 3:]
        heatmap_flip, offset_flip, limbs_offset_flip = model(image)

        heatmap_flip = torch.flip(heatmap_flip, [3])
        heatmap = (heatmap + heatmap_flip[:, flip_index_heat, :, :])/2.0

        # offset_flip = convert_offset(offset_flip, limbs_offset_flip, cfg.DATASET.LIMBS_CONTEXT, cfg.DATASET.KEYPOINTS_RESTORE_INDEX)
        posemap_flip = offset_to_pose(offset_flip, flip_index=flip_index_offset)
        posemap = (posemap + torch.flip(posemap_flip, [3]))/2.0

    return heatmap, posemap


def hierarchical_pool(cfg, heatmap):
    pool1 = torch.nn.MaxPool2d(3, 1, 1)
    pool2 = torch.nn.MaxPool2d(5, 1, 2)
    pool3 = torch.nn.MaxPool2d(7, 1, 3)
    map_size = (heatmap.shape[1]+heatmap.shape[2])/2.0
    if map_size > cfg.TEST.POOL_THRESHOLD1:
        maxm = pool3(heatmap[None, :, :, :])
    elif map_size > cfg.TEST.POOL_THRESHOLD2:
        maxm = pool2(heatmap[None, :, :, :])
    else:
        maxm = pool1(heatmap[None, :, :, :])

    return maxm


def get_maximum_from_heatmap(cfg, heatmap):
    # 获取一个经过最大池化处理的热图
    maxm = hierarchical_pool(cfg, heatmap)
    # 两个tensor比较，同一位值，如果值相同，那么为true，否则为false
    maxm = torch.eq(maxm, heatmap).float()
    heatmap = heatmap * maxm
    scores = heatmap.view(-1)
    scores, pos_ind = scores.topk(cfg.DATASET.MAX_NUM_PEOPLE)

    select_ind = (scores > (cfg.TEST.KEYPOINT_THRESHOLD)).nonzero()
    scores = scores[select_ind][:, 0]
    pos_ind = pos_ind[select_ind][:, 0]

    return pos_ind, scores


def aggregate_results(
        cfg, heatmap_sum, poses, heatmap, posemap, scale
):
    """
    Get initial pose proposals and aggregate the results of all scale.

    Args: 
        heatmap (Tensor): Heatmap at this scale (1, 1+num_joints, w, h)
        posemap (Tensor): Posemap at this scale (1, 2*num_joints, w, h)
        heatmap_sum (Tensor): Sum of the heatmaps (1, 1+num_joints, w, h)
        poses (List): Gather of the pose proposals [(num_people, num_joints, 3)]
    """

    ratio = cfg.DATASET.INPUT_SIZE*1.0/cfg.DATASET.OUTPUT_SIZE
    reverse_scale = ratio/scale
    h, w = heatmap[0].size(-1), heatmap[0].size(-2)
    
    heatmap_sum += up_interpolate(
        heatmap,
        size=(int(reverse_scale*w), int(reverse_scale*h)),
        mode='bilinear'
    )

    center_heatmap = heatmap[0,-1:]
    pose_ind, ctr_score = get_maximum_from_heatmap(cfg, center_heatmap)
    posemap = posemap[0].permute(1, 2, 0).view(h*w, -1, 2)
    pose = reverse_scale*posemap[pose_ind]
    ctr_score = ctr_score[:,None].expand(-1,pose.shape[-2])[:,:,None]
    poses.append(torch.cat([pose, ctr_score], dim=2))

    return heatmap_sum, poses
