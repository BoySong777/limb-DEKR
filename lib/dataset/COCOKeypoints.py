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

import logging

import numpy as np
import torch

import pycocotools
from .COCODataset import CocoDataset

logger = logging.getLogger(__name__)


class CocoKeypoints(CocoDataset):
    def __init__(self, cfg, dataset, heatmap_generator=None, offset_generator=None, transforms=None):
        super().__init__(cfg, dataset)
        self.num_joints = cfg.DATASET.NUM_JOINTS
        self.num_joints_with_center = self.num_joints+1

        self.sigma = cfg.DATASET.SIGMA
        self.center_sigma = cfg.DATASET.CENTER_SIGMA
        self.bg_weight = cfg.DATASET.BG_WEIGHT

        self.heatmap_generator = heatmap_generator
        self.offset_generator = offset_generator
        self.transforms = transforms
        
        self.ids = [
            img_id
            for img_id in self.ids
            if len(self.coco.getAnnIds(imgIds=img_id, iscrowd=None)) > 0
        ]
        self.limbs_context = cfg.DATASET.LIMBS_CONTEXT
        self.limbs_keypoints_index = cfg.DATASET.LIMBS_KEYPOINTS_INDEX
        self.keypoints_restore_index = cfg.DATASET.KEYPOINTS_RESTORE_INDEX
        self.limbs_num = cfg.DATASET.LIMBS_NUM
    def __getitem__(self, idx):
        img, anno, image_info = super().__getitem__(idx)

        mask = self.get_mask(anno, image_info)

        anno = [
            obj for obj in anno
            if obj['iscrowd'] == 0 or obj['num_keypoints'] > 0
        ]
        joints, area = self.get_joints(anno)
        
        if self.transforms:
            img, mask_list, joints_list, area = self.transforms(
                img, [mask], [joints], area
            )

        heatmap, ignored = self.heatmap_generator(
            np.concatenate([joints_list[0][:, :self.num_joints, :], joints_list[0][:, self.num_joints_with_center+self.limbs_num-1:, :]], axis=1),
            self.sigma, self.center_sigma, self.bg_weight)
        mask = mask_list[0]*ignored

        offset, offset_weight, limbs_offset, limbs_offset_weight = self.offset_generator(
            joints_list[0], area)

        return img, heatmap, mask, offset, offset_weight, limbs_offset, limbs_offset_weight

    def cal_area_2_torch(self, v):
        w = torch.max(v[:, :, 0], -1)[0] - torch.min(v[:, :, 0], -1)[0]
        h = torch.max(v[:, :, 1], -1)[0] - torch.min(v[:, :, 1], -1)[0]
        return w * w + h * h

    def get_joints(self, anno):
        num_people = len(anno)
        area = np.zeros((num_people, 1))
        # 让joints中增加5个点来存储肢体中心点
        joints = np.zeros((num_people, self.num_joints_with_center+self.limbs_num, 3))

        for i, obj in enumerate(anno):
            joints[i, :self.num_joints, :3] = \
                np.array(obj['keypoints']).reshape([-1, 3])

            area[i, 0] = self.cal_area_2_torch(
                torch.tensor(joints[i:i+1,:,:]))

            if obj['area'] < 32**2:
                joints[i, -1, 2] = 0
                continue
            
            joints_sum = np.sum(joints[i, :self.num_joints_with_center-1, :2], axis=0)
            num_vis_joints = len(np.nonzero(joints[i, :self.num_joints_with_center-1, 2])[0])
            if num_vis_joints <= 0:
                joints[i, -1, :2] = 0
            else:
                joints[i, -1, :2] = joints_sum / num_vis_joints
                joints[i, -1, 2] = 1

            # 用来生成肢体中心点 start
            temp_joint = np.array(obj['keypoints'], dtype=np.float32).reshape([-1, 3])
            temp_joint = temp_joint[self.limbs_keypoints_index]
            point = 0
            for idx, num in enumerate(self.limbs_context):
                joints_group_sum = np.sum(temp_joint[point:point + num, :2], axis=0)
                num_vis_joints = len(np.nonzero(temp_joint[point:point + num, 2])[0])
                if num_vis_joints > 0:
                    joints[i, self.num_joints + idx, :2] = joints_group_sum / num_vis_joints
                    joints[i, self.num_joints + idx, 2] = 1
                point += num
            # 用来生成肢体中心点 end

        return joints, area

    def get_mask(self, anno, img_info):
        m = np.zeros((img_info['height'], img_info['width']))

        for obj in anno:
            if obj['iscrowd']:
                rle = pycocotools.mask.frPyObjects(
                    obj['segmentation'], img_info['height'], img_info['width'])
                m += pycocotools.mask.decode(rle)
            elif obj['num_keypoints'] == 0:
                rles = pycocotools.mask.frPyObjects(
                    obj['segmentation'], img_info['height'], img_info['width'])
                for rle in rles:
                    m += pycocotools.mask.decode(rle)

        return m < 0.5