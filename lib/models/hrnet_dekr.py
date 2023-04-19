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

import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv_module import HighResolutionModule
from .conv_block import BasicBlock, Bottleneck, AdaptBlock
# from .gc_block import GlobalContextBlock
from .at_block import CBAMBlock

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck,
    'ADAPTIVE': AdaptBlock
}


class PoseHigherResolutionNet(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(PoseHigherResolutionNet, self).__init__()

        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(Bottleneck, 64, 64, 4)

        # build stage
        self.spec = cfg.MODEL.SPEC
        self.stages_spec = self.spec.STAGES
        self.num_stages = self.spec.STAGES.NUM_STAGES
        num_channels_last = [256]
        for i in range(self.num_stages):
            num_channels = self.stages_spec.NUM_CHANNELS[i]
            transition_layer = \
                self._make_transition_layer(num_channels_last, num_channels)
            setattr(self, 'transition{}'.format(i + 1), transition_layer)

            stage, num_channels_last = self._make_stage(
                self.stages_spec, i, num_channels, True
            )
            setattr(self, 'stage{}'.format(i + 2), stage)

        # build head net
        inp_channels = int(sum(self.stages_spec.NUM_CHANNELS[-1]))
        config_heatmap = self.spec.HEAD_HEATMAP
        config_offset = self.spec.HEAD_OFFSET
        self.num_joints = cfg.DATASET.NUM_JOINTS
        self.num_offset = self.num_joints * 2
        self.num_joints_with_center = self.num_joints + 1
        self.offset_prekpt = config_offset['NUM_CHANNELS_PERKPT']

        # 每个肢体内各含有多少关键点
        self.limbs_contxet = cfg.DATASET.LIMBS_CONTEXT
        self.keypoints_restore_index = cfg.DATASET.KEYPOINTS_RESTORE_INDEX

        offset_channels = self.num_joints * self.offset_prekpt
        self.transition_heatmap = self._make_transition_for_head(
            inp_channels, config_heatmap['NUM_CHANNELS'])
        self.transition_offset = self._make_layer(BasicBlock, inp_channels, offset_channels, 2, use_cbam=True)
        self.head_heatmap = self._make_heatmap_head(config_heatmap)
        self.offset_feature_layers, self.offset_final_layer = \
            self._make_separete_regression_head(config_offset)

        self.pretrained_layers = self.spec.PRETRAINED_LAYERS

        self.offset_limbs_feature_layers, self.offset_limbs_final_layer = \
            self._make_limbs_separete_regression_head(config_offset)
        # self.extract_feature_layer = GlobalContextBlock(offset_channels, ratio=0.25)


    def _make_transition_for_head(self, inplanes, outplanes, isoff = False):
        transition_layer = [
            nn.Conv2d(inplanes, outplanes, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(True)
        ]
        return nn.Sequential(*transition_layer)

    def _make_heatmap_head(self, layer_config):
        heatmap_head_layers = []

        feature_conv = self._make_layer(
            blocks_dict[layer_config['BLOCK']],
            layer_config['NUM_CHANNELS'],
            layer_config['NUM_CHANNELS'],
            layer_config['NUM_BLOCKS'],
            dilation=layer_config['DILATION_RATE']
        )
        heatmap_head_layers.append(feature_conv)

        heatmap_conv = nn.Conv2d(
            in_channels=layer_config['NUM_CHANNELS'],
            out_channels=self.num_joints_with_center,
            kernel_size=self.spec.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if self.spec.FINAL_CONV_KERNEL == 3 else 0
        )
        heatmap_head_layers.append(heatmap_conv)

        return nn.ModuleList(heatmap_head_layers)

    def _make_separete_regression_head(self, layer_config):
        offset_feature_layers = []
        offset_final_layer = []

        for num in self.limbs_contxet:
            feature_conv = self._make_layer(
                blocks_dict[layer_config['BLOCK']],
                layer_config['NUM_CHANNELS_PERKPT'] * num,
                layer_config['NUM_CHANNELS_PERKPT'] * num,
                layer_config['NUM_BLOCKS'],
                dilation=layer_config['DILATION_RATE']
            )
            offset_feature_layers.append(feature_conv)

            offset_conv = nn.Conv2d(
                in_channels=layer_config['NUM_CHANNELS_PERKPT'] * num,
                out_channels=2 * num,
                kernel_size=self.spec.FINAL_CONV_KERNEL,
                stride=1,
                padding=1 if self.spec.FINAL_CONV_KERNEL == 3 else 0
            )
            offset_final_layer.append(offset_conv)

        return nn.ModuleList(offset_feature_layers), nn.ModuleList(offset_final_layer)

    def _make_limbs_separete_regression_head(self, layer_config):
        """
        用于生成回归肢体中心点的层，参考：_make_separete_regression_head
        Args:
            layer_config:

        Returns:

        """
        offset_limbs_feature_layers = []
        offset_limbs_final_layer = []

        for num in self.limbs_contxet:
            feature_conv = self._make_layer(
                blocks_dict[layer_config['BLOCK']],
                layer_config['NUM_CHANNELS_PERKPT'] * num,
                layer_config['NUM_CHANNELS_PERKPT'] * num,
                layer_config['NUM_BLOCKS'],
                dilation=layer_config['DILATION_RATE']
            )
            offset_limbs_feature_layers.append(feature_conv)

            offset_conv = nn.Conv2d(
                in_channels=layer_config['NUM_CHANNELS_PERKPT'] * num,
                out_channels=2,
                kernel_size=self.spec.FINAL_CONV_KERNEL,
                stride=1,
                padding=1 if self.spec.FINAL_CONV_KERNEL == 3 else 0
            )
            offset_limbs_final_layer.append(offset_conv)

        return nn.ModuleList(offset_limbs_feature_layers), nn.ModuleList(offset_limbs_final_layer)
    def _make_layer(
            self, block, inplanes, planes, blocks, stride=1, dilation=1, use_cbam=False):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes,
                stride, downsample, dilation=dilation, use_cbam=use_cbam))
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes, dilation=dilation, use_cbam=use_cbam))

        return nn.Sequential(*layers)

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  1,
                                  1,
                                  bias=False),
                        nn.BatchNorm2d(num_channels_cur_layer[i]),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(
                            inchannels, outchannels, 3, 2, 1, bias=False),
                        nn.BatchNorm2d(outchannels),
                        nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_stage(self, stages_spec, stage_index, num_inchannels,
                    multi_scale_output=True):
        num_modules = stages_spec.NUM_MODULES[stage_index]
        num_branches = stages_spec.NUM_BRANCHES[stage_index]
        num_blocks = stages_spec.NUM_BLOCKS[stage_index]
        num_channels = stages_spec.NUM_CHANNELS[stage_index]
        block = blocks_dict[stages_spec['BLOCK'][stage_index]]
        fuse_method = stages_spec.FUSE_METHOD[stage_index]

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    reset_multi_scale_output
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def offset_to_loc(self, offset):
        n, c, w, h = offset.shape
        # 构建位置坐标
        shifts_y = torch.arange(
            0, w, step=1,
            dtype=torch.float32, device=offset.device
        )
        shifts_x = torch.arange(
            0, h, step=1,
            dtype=torch.float32, device=offset.device
        )
        shift_x, shift_y = torch.meshgrid(shifts_x, shifts_y, indexing='xy')
        localtion = torch.stack((shift_x, shift_y))
        localtion = localtion.repeat(int(c / 2), 1, 1)
        localtion = localtion.view(1, c, w, h)

        return localtion - offset

    # 在模型中就把关键点偏移量处理好
    def convert_offset(self, limbs_offset, offset):
        n, c, w, h = offset.shape
        # 先把相对偏移的偏移改成绝对位置
        locate_map = self.offset_to_loc(limbs_offset)
        # 接下来把位置值的范围调整到[-1, 1] 公式为：（x,y）*2/((w,h)-1) - 1
        param = [2.0 / (h - 1), 2.0 / (w - 1)]
        param_tensor = torch.tensor(param).reshape(1, 2, 1, 1).repeat(1, len(self.limbs_contxet), 1, 1).cuda(
            non_blocking=True)
        locate_map = locate_map * param_tensor - 1.0
        point = 0
        final_offset_list = []
        # 对肢体逐一进行处理
        for idx, num in enumerate(self.limbs_contxet):
            # 把相对于肢体中心点的关键点偏移量挪到肢体中心点对应的人体中心点下，目的是为了方便关键点偏移量和肢体中心点偏移量相加。
            # 肢体中心点偏移量和关键点偏移量相加，得到关键点相对于人体中心点的偏移量。
            final_offset_list.append(
                F.grid_sample(offset[:, point:point + num * 2],
                              locate_map[:, idx * 2:(idx + 1) * 2].permute(0, 2, 3, 1),
                              mode='bilinear', padding_mode='zeros', align_corners=True) \
                + limbs_offset[:, idx * 2:(idx + 1) * 2].repeat(1, num, 1, 1)
            )

            point += num * 2
        final_offset = torch.cat(final_offset_list, dim=1).view(n, int(c / 2), 2, w, h)[:, self.keypoints_restore_index] \
            .view(n, c, w, h)
        return final_offset

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        y_list = [x]
        for i in range(self.num_stages):
            x_list = []
            transition = getattr(self, 'transition{}'.format(i + 1))
            for j in range(self.stages_spec['NUM_BRANCHES'][i]):
                if transition[j]:
                    x_list.append(transition[j](y_list[-1]))
                else:
                    x_list.append(y_list[j])
            y_list = getattr(self, 'stage{}'.format(i + 2))(x_list)

        x0_h, x0_w = y_list[0].size(2), y_list[0].size(3)
        x = torch.cat([y_list[0], \
                       F.upsample(y_list[1], size=(x0_h, x0_w), mode='bilinear'), \
                       F.upsample(y_list[2], size=(x0_h, x0_w), mode='bilinear'), \
                       F.upsample(y_list[3], size=(x0_h, x0_w), mode='bilinear')], 1)

        heatmap_feature = self.head_heatmap[0](self.transition_heatmap(x))
        heatmap = self.head_heatmap[1](heatmap_feature)

        final_offset = []
        # 添加肢体中心点偏移量存储
        final_limbs_offset = []

       #  pre_limbs_feature = self.extract_feature_layer(x, heatmap_feature)
        # 获取回归肢体中心点的特征
        limbs_feature = self.transition_offset(x)

        # limbs_feature = self.extract_feature_layer(limbs_feature)

        offset_feature_list = []
        point = 0
        # 求出肢体中心点的输出特征和偏移量
        for idx, num in enumerate(self.limbs_contxet):
            offset_feature_list.append(
                self.offset_limbs_feature_layers[idx](limbs_feature[:, point:point + self.offset_prekpt * num]))
            final_limbs_offset.append(self.offset_limbs_final_layer[idx](offset_feature_list[idx]))

            final_offset.append(
                self.offset_final_layer[idx](
                    self.offset_feature_layers[idx](offset_feature_list[idx])))

            point = point + self.offset_prekpt * num

        offset = torch.cat(final_offset, dim=1)
        limbs_offset = torch.cat(final_limbs_offset, dim=1)

        offset_done = self.convert_offset(limbs_offset, offset)
        return heatmap, offset_done, limbs_offset

    def init_weights(self, pretrained='', verbose=True):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in self.modules():
            if hasattr(m, 'transform_matrix_conv'):
                nn.init.constant_(m.transform_matrix_conv.weight, 0)
                if hasattr(m, 'bias'):
                    nn.init.constant_(m.transform_matrix_conv.bias, 0)
            if hasattr(m, 'translation_conv'):
                nn.init.constant_(m.translation_conv.weight, 0)
                if hasattr(m, 'bias'):
                    nn.init.constant_(m.translation_conv.bias, 0)

        parameters_names = set()
        for name, _ in self.named_parameters():
            parameters_names.add(name)

        buffers_names = set()
        for name, _ in self.named_buffers():
            buffers_names.add(name)

        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained,
                                               map_location=lambda storage, loc: storage)
            logger.info('=> loading pretrained model {}'.format(pretrained))

            need_init_state_dict = {}
            for name, m in pretrained_state_dict.items():
                if name.split('.')[0] in self.pretrained_layers \
                        or self.pretrained_layers[0] == '*':
                    if name in parameters_names or name in buffers_names:
                        if verbose:
                            logger.info(
                                '=> init {} from {}'.format(name, pretrained)
                            )
                        need_init_state_dict[name] = m
            self.load_state_dict(need_init_state_dict, strict=False)


def get_pose_net(cfg, is_train, **kwargs):
    model = PoseHigherResolutionNet(cfg, **kwargs)

    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED, verbose=cfg.VERBOSE)

    return model

