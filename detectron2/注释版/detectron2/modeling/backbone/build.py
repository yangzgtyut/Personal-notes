# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.layers import ShapeSpec
from detectron2.utils.registry import Registry

from .backbone import Backbone

BACKBONE_REGISTRY = Registry("BACKBONE")
BACKBONE_REGISTRY.__doc__ = """
Registry for backbones, which extract feature maps from images

The registered object must be a callable that accepts two arguments:

1. A :class:`detectron2.config.CfgNode`
2. A :class:`detectron2.layers.ShapeSpec`, which contains the input shape specification.

It must returns an instance of :class:`Backbone`.
"""


def build_backbone(cfg, input_shape=None):
    """
    Build a backbone from `cfg.MODEL.BACKBONE.NAME`.

    Returns:
        an instance of :class:`Backbone`
    """
    if input_shape is None:
        # 若没有指定输入大小，则将输入图像的通道数读出来
        input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))

    backbone_name = cfg.MODEL.BACKBONE.NAME
    backbone = BACKBONE_REGISTRY.get(backbone_name)(cfg, input_shape)
    assert isinstance(backbone, Backbone)
    return backbone
