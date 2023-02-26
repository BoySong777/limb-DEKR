# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn).
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class HeatmapGenerator():
    def __init__(self, output_res, num_joints):
        self.output_res = output_res
        self.num_joints_with_center = num_joints+1

    def get_heat_val(self, sigma, x, y, x0, y0):

        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

        return g

    def __call__(self, joints, sgm, ct_sgm, bg_weight=1.0):
        assert self.num_joints_with_center == joints.shape[1], \
            'the number of joints should be %d' % self.num_joints_with_center

        hms = np.zeros((self.num_joints_with_center, self.output_res, self.output_res),
                       dtype=np.float32)
        ignored_hms = 2*np.ones((self.num_joints_with_center, self.output_res, self.output_res),
                                    dtype=np.float32)

        hms_list = [hms, ignored_hms]

        for p in joints:
            for idx, pt in enumerate(p):
                if idx < 17:
                    sigma = sgm
                else:
                    sigma = ct_sgm
                if pt[2] > 0:
                    x, y = pt[0], pt[1]
                    if x < 0 or y < 0 or \
                            x >= self.output_res or y >= self.output_res:
                        continue

                    ul = int(np.floor(x - 3 * sigma - 1)
                             ), int(np.floor(y - 3 * sigma - 1))
                    br = int(np.ceil(x + 3 * sigma + 2)
                             ), int(np.ceil(y + 3 * sigma + 2))

                    cc, dd = max(0, ul[0]), min(br[0], self.output_res)
                    aa, bb = max(0, ul[1]), min(br[1], self.output_res)

                    joint_rg = np.zeros((bb-aa, dd-cc))
                    for sy in range(aa, bb):
                        for sx in range(cc, dd):
                            joint_rg[sy-aa, sx -
                                     cc] = self.get_heat_val(sigma, sx, sy, x, y)

                    hms_list[0][idx, aa:bb, cc:dd] = np.maximum(
                        hms_list[0][idx, aa:bb, cc:dd], joint_rg)
                    hms_list[1][idx, aa:bb, cc:dd] = 1.

        hms_list[1][hms_list[1] == 2] = bg_weight

        return hms_list


class OffsetGenerator():
    def __init__(self, output_h, output_w, num_joints, radius, limbs_num, limbs_context, limbs_keypoints_index):
        self.num_joints_without_center = num_joints
        self.num_joints_with_center = num_joints+1
        self.output_w = output_w
        self.output_h = output_h
        self.radius = radius
        self.limbs_num = limbs_num
        self.limbs_context = limbs_context
        self.limbs_keypoints_index = limbs_keypoints_index

    def __call__(self, joints, area):
        assert joints.shape[1] == self.num_joints_with_center + self.limbs_num, \
            'the number of joints should be 18, 17 keypoints + 1 center joint.'

        offset_map = np.zeros((self.num_joints_without_center*2, self.output_h, self.output_w),
                              dtype=np.float32)
        weight_map = np.zeros((self.num_joints_without_center*2, self.output_h, self.output_w),
                              dtype=np.float32)
        limbs_offset_map = np.zeros((self.limbs_num * 2, self.output_h, self.output_w),
                              dtype=np.float32)
        limbs_weight_map = np.zeros((self.limbs_num * 2, self.output_h, self.output_w),
                              dtype=np.float32)
        area_map = np.zeros((self.output_h, self.output_w),
                            dtype=np.float32)

        # 处理关键点的顺序
        # temp_joints = joints[:, self.limbs_keypoints_index, :]
        # 进行肢体偏移量的处理
        for person_id, p in enumerate(joints):
            ct_x = int(p[-1, 0])
            ct_y = int(p[-1, 1])
            ct_v = int(p[-1, 2])
            if ct_v < 1 or ct_x < 0 or ct_y < 0 \
                    or ct_x >= self.output_w or ct_y >= self.output_h:
                continue
            for idx, pt in enumerate(p[self.num_joints_without_center:-1]):
                if pt[2] > 0:
                    x, y = pt[0], pt[1]
                    if x < 0 or y < 0 or \
                            x >= self.output_w or y >= self.output_h:
                        continue
                    start_x = max(int(ct_x - self.radius), 0)
                    start_y = max(int(ct_y - self.radius), 0)
                    end_x = min(int(ct_x + self.radius), self.output_w)
                    end_y = min(int(ct_y + self.radius), self.output_h)
                    for pos_x in range(start_x, end_x):
                        for pos_y in range(start_y, end_y):
                            offset_x = pos_x - x
                            offset_y = pos_y - y
                            if limbs_offset_map[idx * 2, pos_y, pos_x] != 0 \
                                    or limbs_offset_map[idx * 2 + 1, pos_y, pos_x] != 0:
                                if area_map[pos_y, pos_x] < area[person_id]:
                                    continue
                            limbs_offset_map[idx * 2, pos_y, pos_x] = offset_x
                            limbs_offset_map[idx * 2 + 1, pos_y, pos_x] = offset_y
                            limbs_weight_map[idx * 2, pos_y, pos_x] = 1. / np.sqrt(area[person_id])
                            limbs_weight_map[idx * 2 + 1, pos_y, pos_x] = 1. / np.sqrt(area[person_id])
                            area_map[pos_y, pos_x] = area[person_id]
        # 肢体偏移量处理完毕

        for person_id, p in enumerate(joints):
            ct_x = int(p[-1, 0])
            ct_y = int(p[-1, 1])
            ct_v = int(p[-1, 2])
            if ct_v < 1 or ct_x < 0 or ct_y < 0 \
                    or ct_x >= self.output_w or ct_y >= self.output_h:
                continue

            for idx, pt in enumerate(p[:self.num_joints_without_center]):
                if pt[2] > 0:
                    x, y = pt[0], pt[1]
                    if x < 0 or y < 0 or \
                            x >= self.output_w or y >= self.output_h:
                        continue

                    start_x = max(int(ct_x - self.radius), 0)
                    start_y = max(int(ct_y - self.radius), 0)
                    end_x = min(int(ct_x + self.radius), self.output_w)
                    end_y = min(int(ct_y + self.radius), self.output_h)

                    for pos_x in range(start_x, end_x):
                        for pos_y in range(start_y, end_y):
                            offset_x = pos_x - x
                            offset_y = pos_y - y
                            if offset_map[idx*2, pos_y, pos_x] != 0 \
                                    or offset_map[idx*2+1, pos_y, pos_x] != 0:
                                if area_map[pos_y, pos_x] < area[person_id]:
                                    continue
                            offset_map[idx*2, pos_y, pos_x] = offset_x
                            offset_map[idx*2+1, pos_y, pos_x] = offset_y
                            weight_map[idx*2, pos_y, pos_x] = 1. / np.sqrt(area[person_id])
                            weight_map[idx*2+1, pos_y, pos_x] = 1. / np.sqrt(area[person_id])
                            area_map[pos_y, pos_x] = area[person_id]
        # # 开始处理关键点偏移量，关键点偏移量是相对于肢体中心点的。
        # for person_id, p in enumerate(joints):
        #     point = 0
        #     # 循环每个人的肢体中心点
        #     for limb_id, kpt_num in enumerate(self.limbs_context):
        #         ct_x = int(p[self.num_joints_without_center+limb_id, 0])
        #         ct_y = int(p[self.num_joints_without_center+limb_id, 1])
        #         ct_v = int(p[self.num_joints_without_center+limb_id, 2])
        #         if ct_v < 1 or ct_x < 0 or ct_y < 0 \
        #                 or ct_x >= self.output_w or ct_y >= self.output_h:
        #             continue
        #         # 循环每个肢体中心点对应的关键点
        #         for idx, pt in enumerate(temp_joints[person_id, point:point+kpt_num]):
        #             if pt[2] > 0:
        #                 x, y = pt[0], pt[1]
        #                 if x < 0 or y < 0 or \
        #                         x >= self.output_w or y >= self.output_h:
        #                     continue
        #
        #                 start_x = max(int(ct_x - self.radius), 0)
        #                 start_y = max(int(ct_y - self.radius), 0)
        #                 end_x = min(int(ct_x + self.radius), self.output_w)
        #                 end_y = min(int(ct_y + self.radius), self.output_h)
        #                 # final_idx 表示在全部关键点里，当前关键点的索引
        #                 final_idx = point + idx
        #                 for pos_x in range(start_x, end_x):
        #                     for pos_y in range(start_y, end_y):
        #                         offset_x = pos_x - x
        #                         offset_y = pos_y - y
        #                         if offset_map[final_idx * 2, pos_y, pos_x] != 0 \
        #                                 or offset_map[final_idx * 2 + 1, pos_y, pos_x] != 0:
        #                             if area_map[pos_y, pos_x] < area[person_id]:
        #                                 continue
        #                         offset_map[final_idx * 2, pos_y, pos_x] = offset_x
        #                         offset_map[final_idx * 2 + 1, pos_y, pos_x] = offset_y
        #                         weight_map[final_idx * 2, pos_y, pos_x] = 1. / np.sqrt(area[person_id])
        #                         weight_map[final_idx * 2 + 1, pos_y, pos_x] = 1. / np.sqrt(area[person_id])
        #                         area_map[pos_y, pos_x] = area[person_id]
        #         point += kpt_num
        return offset_map, weight_map, limbs_offset_map, limbs_weight_map
