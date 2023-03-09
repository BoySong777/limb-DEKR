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

import torch
import torch.nn as nn
import numpy as np

logger = logging.getLogger(__name__)


class HeatmapLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt, mask):
        assert pred.size() == gt.size()
        loss = ((pred - gt)**2) * mask
        loss = loss.mean(dim=3).mean(dim=2).mean(dim=1).mean(dim=0)
        return loss


class OffsetsLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def smooth_l1_loss(self, pred, gt, beta=1. / 9):
        l1_loss = torch.abs(pred - gt)
        cond = l1_loss < beta
        loss = torch.where(cond, 0.5*l1_loss**2/beta, l1_loss-0.5*beta)
        return loss

    def forward(self, pred, gt, weights):
        assert pred.size() == gt.size()
        num_pos = torch.nonzero(weights > 0).size()[0]
        loss = self.smooth_l1_loss(pred, gt) * weights
        if num_pos == 0:
            num_pos = 1.
        loss = loss.sum() / num_pos
        return loss


class OKSLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def oks_loss(self, pred, gt, weights, offset_area, sigmas, beta=1. / 9):
        '''

        Args:
            pred: (N, 2K, W, H)
            gt: (N, 2K, W, H)
            weights: groundtruth中，每个关键点对应人体的面积的倒数：1/s.  (N, 2K, W, H)
            sigmas: 各个关键点的权重:(K,)

        Returns:

        '''
        sigmas = torch.tensor(sigmas, device=gt.device)
        # 计算关键点之间的欧几里得距离
        dist = torch.abs(pred - gt)
        cond = dist < beta
        # loss = (1 - torch.exp(- l1_loss * weight)) * torch.where(cond, 0.5*l1_loss**2/beta, l1_loss-0.5*beta)
        dist = torch.where(cond, 0.5 * dist ** 2 / beta, dist - 0.5 * beta)
        vars = sigmas.unsqueeze(-1).repeat(1, 2).flatten()
        vars = vars[None, :, None, None]
        loss = 1 - torch.exp(-dist * weights / vars)
        return loss

    def forward(self, pred, gt, weights, offset_area, sigmas):
        assert pred.size() == gt.size()
        num_pos = torch.nonzero(weights > 0).size()[0]
        loss = self.oks_loss(pred, gt, weights, offset_area, sigmas)
        if num_pos == 0:
            num_pos = 1.
        loss = loss.sum() / num_pos
        return loss


class OffsetsLossAddWeight(nn.Module):
    '''
        仿照focal loss 给偏移量损失增加权重，具体来说，困难关键点的偏移量的权重应该尽可能地大。
    '''
    def __init__(self):
        super().__init__()

    def smooth_l1_loss(self, pred, gt, weight, beta=1. / 9):
        l1_loss = torch.abs(pred - gt)
        cond = l1_loss < beta
        # loss = (1 - torch.exp(- l1_loss * weight)) * torch.where(cond, 0.5*l1_loss**2/beta, l1_loss-0.5*beta)
        loss = l1_loss * torch.where(cond, 0.5*l1_loss**2/beta, l1_loss-0.5*beta)
        return loss

    def forward(self, pred, gt, weights):
        assert pred.size() == gt.size()
        num_pos = torch.nonzero(weights > 0).size()[0]
        loss = self.smooth_l1_loss(pred, gt, weights) * weights
        if num_pos == 0:
            num_pos = 1.
        loss = loss.sum() / num_pos
        return loss


class MultiLossFactory(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.num_joints = cfg.MODEL.NUM_JOINTS
        self.bg_weight = cfg.DATASET.BG_WEIGHT
        self.sigmas = cfg.DATASET.KEYPOINTS_SIGMAS

        self.heatmap_loss = HeatmapLoss() if cfg.LOSS.WITH_HEATMAPS_LOSS else None
        self.heatmap_loss_factor = cfg.LOSS.HEATMAPS_LOSS_FACTOR

        # self.offset_loss = OffsetsLoss() if cfg.LOSS.WITH_OFFSETS_LOSS else None
        self.offset_loss = OKSLoss() if cfg.LOSS.WITH_OFFSETS_LOSS else None
        # self.offset_loss = OffsetsLossAddWeight() if cfg.LOSS.WITH_OFFSETS_LOSS else None
        self.offset_loss_factor = cfg.LOSS.OFFSETS_LOSS_FACTOR
        # 增加肢体偏移量的损失。
        self.limbs_offset_loss = OffsetsLoss() if cfg.LOSS.WITH_OFFSETS_LOSS else None

    def forward(self, output, poffset, plimbs_offset, heatmap, mask, offset, offset_w, offset_area, limbs_offset, limbs_offset_w):
        if self.heatmap_loss:
            heatmap_loss = self.heatmap_loss(output, heatmap, mask)
            heatmap_loss = heatmap_loss * self.heatmap_loss_factor
        else:
            heatmap_loss = None
        
        if self.offset_loss:
           # offset_loss = self.offset_loss(poffset, offset, offset_w)
            offset_loss = self.offset_loss(poffset, offset, offset_w, offset_area, self.sigmas)
            offset_loss = offset_loss * self.offset_loss_factor
            # 和关键点偏移量类似，获取肢体中心点的偏移量
            limbs_offset_loss = self.limbs_offset_loss(plimbs_offset, limbs_offset, limbs_offset_w)
            limbs_offset_loss = limbs_offset_loss * self.offset_loss_factor

        else:
            offset_loss = None
            limbs_offset_loss = None

        return heatmap_loss, offset_loss, limbs_offset_loss
