# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from torch import nn

from .batch_norm import FrozenBatchNorm2d


class CNNBlockBase(nn.Module):
    """
    A CNN block is assumed to have input channels, output channels and a stride.
    The input and output of `forward()` method must be NCHW tensors.
    The method can perform arbitrary computation but must match the given
    channels and stride specification.

    假定CNN块具有输入通道，输出通道和stride步幅。 `forward()`方法的输入和输出必须是(N, C, H, W)张量。 该方法可以执行任意计算，但必须匹配给定的通道和步幅规范。

    Attribute:
        in_channels (int):
        out_channels (int):
        stride (int):
    """

    def __init__(self, in_channels, out_channels, stride):
        """
        The `__init__` method of any subclass should also contain these arguments.

        Args:
            in_channels (int):
            out_channels (int):
            stride (int):
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

    def freeze(self):
        """
        Make this block not trainable.
        This method sets all parameters to `requires_grad=False`,
        and convert all BatchNorm layers to FrozenBatchNorm
        冻结此模块的训练
        这个方法会将所有的参数设置为：requires_grad=False
        并会将所有的BN层设置为FrozenBatchNorm

        Returns:
            the block itself
        """
        for p in self.parameters():
            p.requires_grad = False
        FrozenBatchNorm2d.convert_frozen_batchnorm(self)
        return self
