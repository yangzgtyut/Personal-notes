# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
import fvcore.nn.weight_init as weight_init
import torch
import torch.nn.functional as F
from torch import nn

from detectron2.layers import (
    CNNBlockBase,
    Conv2d,
    DeformConv,
    ModulatedDeformConv,
    ShapeSpec,
    get_norm,
)

from .backbone import Backbone
from .build import BACKBONE_REGISTRY

__all__ = [
    "ResNetBlockBase",
    "BasicBlock",
    "BottleneckBlock",
    "DeformBottleneckBlock",
    "BasicStem",
    "ResNet",
    "make_stage",
    "build_resnet_backbone",
]


ResNetBlockBase = CNNBlockBase
"""
Alias for backward compatibiltiy.
"""


class BasicBlock(CNNBlockBase):
    """
    The basic residual block for ResNet-18 and ResNet-34 defined in :paper:`ResNet`, with two 3x3 conv layers and a projection shortcut if needed.
    """

    def __init__(self, in_channels, out_channels, *, stride=1, norm="BN"):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int): Stride for the first conv.
            norm (str or callable): normalization for all conv layers.
                See :func:`layers.get_norm` for supported format.
        """
        super().__init__(in_channels, out_channels, stride)

        if in_channels != out_channels:
            self.shortcut = Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
                norm=get_norm(norm, out_channels),
            )
        else:
            self.shortcut = None

        self.conv1 = Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )

        self.conv2 = Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )

        for layer in [self.conv1, self.conv2, self.shortcut]:
            if layer is not None:  # shortcut can be None
                weight_init.c2_msra_fill(layer)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu_(out)
        out = self.conv2(out)

        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        out += shortcut
        out = F.relu_(out)
        return out


class BottleneckBlock(CNNBlockBase):
    """
    基本残差块
    The standard bottleneck residual block used by ResNet-50, 101 and 152
    defined in :paper:`ResNet`.  It contains 3 conv layers with kernels 1x1,3x3, 1x1, and a projection shortcut if needed.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        *,
        bottleneck_channels,
        stride=1,
        num_groups=1,
        norm="BN",
        stride_in_1x1=False,
        dilation=1,
    ):
        """
        Args:
            bottleneck_channels (int):
                number of output channels for the 3x3 "bottleneck" conv layers.
                bottleneck中 3x3 卷积的输出通道数
            num_groups (int):
                3x3 卷积层的组数
                number of groups for the 3x3 conv layer.
            norm (str or callable):
                normalization for all conv layers.
                See :func:`layers.get_norm` for supported format.
            stride_in_1x1 (bool):
                when stride>1, whether to put stride in the first 1x1 convolution or the bottleneck 3x3 convolution.
            dilation (int):
                the dilation rate of the 3x3 conv layer.
        """
        super().__init__(in_channels, out_channels, stride)

        if in_channels != out_channels:
            # shortcut会和最后的1x1输出的特征图进行相加，若通道数不一致则需要先进性1x1卷积
            self.shortcut = Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
                norm=get_norm(norm, out_channels),
            )
        else:
            self.shortcut = None

        # The original MSRA ResNet models have stride in the first 1x1 conv
        # The subsequent fb.torch.resnet and Caffe2 ResNe[X]t implementations have stride in the 3x3 conv
        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)

        # 1x1
        self.conv1 = Conv2d(
            in_channels,
            bottleneck_channels,  # 3x3的输出通道数
            kernel_size=1,
            stride=stride_1x1,
            bias=False,
            norm=get_norm(norm, bottleneck_channels),
        )

        # 3x3
        self.conv2 = Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride_3x3,
            padding=1 * dilation,
            bias=False,
            groups=num_groups,
            dilation=dilation,
            norm=get_norm(norm, bottleneck_channels),
        )

        # 1x1
        self.conv3 = Conv2d(
            bottleneck_channels,
            out_channels,
            kernel_size=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )

        for layer in [self.conv1, self.conv2, self.conv3, self.shortcut]:
            if layer is not None:  # shortcut can be None
                weight_init.c2_msra_fill(layer)

        # Zero-initialize the last normalization in each residual branch, so that at the beginning, the residual branch starts with zeros, and each residual block behaves like an identity.
        # See Sec 5.1 in "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour":
        # "For BN layers, the learnable scaling coefficient γ is initialized to be 1, except for each residual block's last BN where γ is initialized to be 0."

        # nn.init.constant_(self.conv3.norm.weight, 0)
        # TODO this somehow hurts performance when training GN models from scratch.
        # Add it as an option when we need to use this code to train a backbone.

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu_(out)

        out = self.conv2(out)
        out = F.relu_(out)

        out = self.conv3(out)

        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        out += shortcut
        out = F.relu_(out)
        return out


class DeformBottleneckBlock(ResNetBlockBase):
    """
    Similar to :class:`BottleneckBlock`, but with :paper:`deformable conv <deformconv>`
    in the 3x3 convolution.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        *,
        bottleneck_channels,
        stride=1,
        num_groups=1,
        norm="BN",
        stride_in_1x1=False,
        dilation=1,
        deform_modulated=False,
        deform_num_groups=1,
    ):
        super().__init__(in_channels, out_channels, stride)
        self.deform_modulated = deform_modulated

        if in_channels != out_channels:
            self.shortcut = Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
                norm=get_norm(norm, out_channels),
            )
        else:
            self.shortcut = None

        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)

        self.conv1 = Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=stride_1x1,
            bias=False,
            norm=get_norm(norm, bottleneck_channels),
        )

        if deform_modulated:
            deform_conv_op = ModulatedDeformConv
            # offset channels are 2 or 3 (if with modulated) * kernel_size * kernel_size
            offset_channels = 27
        else:
            deform_conv_op = DeformConv
            offset_channels = 18

        self.conv2_offset = Conv2d(
            bottleneck_channels,
            offset_channels * deform_num_groups,
            kernel_size=3,
            stride=stride_3x3,
            padding=1 * dilation,
            dilation=dilation,
        )
        self.conv2 = deform_conv_op(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride_3x3,
            padding=1 * dilation,
            bias=False,
            groups=num_groups,
            dilation=dilation,
            deformable_groups=deform_num_groups,
            norm=get_norm(norm, bottleneck_channels),
        )

        self.conv3 = Conv2d(
            bottleneck_channels,
            out_channels,
            kernel_size=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )

        for layer in [self.conv1, self.conv2, self.conv3, self.shortcut]:
            if layer is not None:  # shortcut can be None
                weight_init.c2_msra_fill(layer)

        nn.init.constant_(self.conv2_offset.weight, 0)
        nn.init.constant_(self.conv2_offset.bias, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu_(out)

        if self.deform_modulated:
            offset_mask = self.conv2_offset(out)
            offset_x, offset_y, mask = torch.chunk(offset_mask, 3, dim=1)
            offset = torch.cat((offset_x, offset_y), dim=1)
            mask = mask.sigmoid()
            out = self.conv2(out, offset, mask)
        else:
            offset = self.conv2_offset(out)
            out = self.conv2(out, offset)
        out = F.relu_(out)

        out = self.conv3(out)

        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        out += shortcut
        out = F.relu_(out)
        return out


def make_stage(block_class, num_blocks, first_stride, *, in_channels, out_channels, **kwargs):
    """
    创建 stage
    Create a list of blocks just like those in a ResNet stage.

    Args:
        block_class (type):
            a subclass of ResNetBlockBase, e.g. BottleneckBlock
        num_blocks (int):
            该stage中的残差块的数量
        first_stride (int):
            第一个残差块的步长stride
            the stride of the first block. The other blocks will have stride=1.
        in_channels (int):
            input channels of the entire stage.
        out_channels (int):
            output channels of **every block** in the stage.
        kwargs:
            other arguments passed to the constructor of every block.

    Returns:
        list[nn.Module]: a list of block module.
    """
    assert "stride" not in kwargs, "Stride of blocks in make_stage cannot be changed."
    blocks = []
    for i in range(num_blocks):
        blocks.append(
            block_class(
                # * 建立每一个残差块
                in_channels=in_channels,
                out_channels=out_channels,
                stride=first_stride if i == 0 else 1,  # 第一个残差块的步长可以设置，后面的不行
                **kwargs,
            )
        )
        in_channels = out_channels  # 使得下一个残差块的输入通道等于上一个的输出通道
    return blocks


class BasicStem(CNNBlockBase):
    """
    The standard ResNet stem (layers before the first residual block).
    conv1(+norm) + relu_ + max_pool2d
    """

    def __init__(self, in_channels=3, out_channels=64, norm="BN"):
        """
        Args:
            norm (str or callable): norm after the first conv layer.
                See :func:`layers.get_norm` for supported format.
        """
        super().__init__(in_channels, out_channels, 4)
        self.in_channels = in_channels
        self.conv1 = Conv2d(
            in_channels,
            out_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
            norm=get_norm(norm, out_channels),
        )
        weight_init.c2_msra_fill(self.conv1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu_(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        return x


class ResNet(Backbone):
    """
    Implement :paper:`ResNet`.
    """

    def __init__(self, stem, stages, num_classes=None, out_features=None):
        """
        Args:
            stem (nn.Module):
                a stem module
            stages (list[list[CNNBlockBase]]):
                several (typically 4) stages, each contains multiple :class:`CNNBlockBase`.
            num_classes (None or int):
                if None, will not perform classification. Otherwise, will create a linear layer.
                类别数，如果None，则不进行分类
            out_features (list[str]):
                name of the layers whose outputs should be returned in forward. Can be anything in "stem", "linear", or "res2" ...
                If None, will return the output of the last layer.
        """
        super(ResNet, self).__init__()

        # * 建立stem stage
        self.stem = stem
        self.num_classes = num_classes

        current_stride = self.stem.stride  # 当前stage总的步长
        self._out_feature_strides = {"stem": current_stride}  # 存储了每一个stage输出的特征图的总的步长
        self._out_feature_channels = {"stem": self.stem.out_channels}  # 存储了每一个stage输出的特征图的通道数

        # * 建立几个stage的残差块
        self.stages_and_names = []  # [(torch.nn.Sequential, name)]
        for i, blocks in enumerate(stages):
            assert len(blocks) > 0, len(blocks)
            for block in blocks:
                assert isinstance(block, CNNBlockBase), block

            name = "res" + str(i + 2)
            # * 为每个blocks创建 nn.Sequential
            stage = nn.Sequential(*blocks)
            # torch中的顺序容器。模块将按照在构造函数中传递的顺序添加到模块中。

            # torch.nn.module.add_module(name, module) 将子模块添加到当前模块。可以使用给定名称将模块作为属性进行访问。
            self.add_module(name, stage)
            self.stages_and_names.append((stage, name))  # stages_and_names: list

            self._out_feature_strides[name] = current_stride = int(
                current_stride * np.prod([k.stride for k in blocks])
            )  # np.prod 对里面的所有元素进行连乘
            self._out_feature_channels[name] = curr_channels = blocks[-1].out_channels

        # 若要进行分类，则建立下采样和全连接层
        if num_classes is not None:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.linear = nn.Linear(curr_channels, num_classes)

            # Sec 5.1 in "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour":
            # "The 1000-way fully-connected layer is initialized by
            # drawing weights from a zero-mean Gaussian with standard deviation of 0.01."
            nn.init.normal_(self.linear.weight, std=0.01)
            name = "linear"

        if out_features is None:
            # 若没有指定输出的特征图，则默认输出最后一层
            out_features = [name]
        self._out_features = out_features
        assert len(self._out_features)
        children = [x[0] for x in self.named_children()]
        # nn.module.named_children(): Returns an iterator over immediate children modules, yielding both the name of the module as well as the module itself.
        for out_feature in self._out_features:
            assert out_feature in children, "Available children: {}".format(", ".join(children))

    def forward(self, x):
        outputs = {}  # 保存输出的特征图
        x = self.stem(x)
        if "stem" in self._out_features:
            outputs["stem"] = x
        for stage, name in self.stages_and_names:
            x = stage(x)
            if name in self._out_features:
                outputs[name] = x  # * 保存需要保存的stage的输出，并通过dict访问，例如""res4"
        if self.num_classes is not None:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.linear(x)
            if "linear" in self._out_features:
                outputs["linear"] = x
        return outputs

    def output_shape(self):
        '''
        输出是一个字典，keys是要保存的stage名，例如FPN是[res2, res3, res4, res5]，C4是[res4]
        '''
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

    def freeze(self, freeze_at=0):
        """
        Freeze the first several stages of the ResNet. Commonly used in
        fine-tuning.

        Layers that produce the same feature map spatial size are defined as one
        "stage" by :paper:`FPN`.
        在'FPN'产生相同大小特征的层被定义为一个stage

        Args:
            freeze_at (int): number of stages to freeze.
                `1` means freezing the stem. `2` means freezing the stem and
                one residual stage, etc.

        Returns:
            nn.Module: this ResNet itself
        """
        if freeze_at >= 1:
            self.stem.freeze()
        for idx, (stage, _) in enumerate(self.stages_and_names, start=2):
            # start=2: 产生的idx从2开始计数
            if freeze_at >= idx:
                for block in stage.children():
                    block.freeze()
        return self


@BACKBONE_REGISTRY.register()
def build_resnet_backbone(cfg, input_shape):
    """
    Create a ResNet instance from config.

    Returns:
        ResNet: a :class:`ResNet` instance.
    """
    # need registration of new blocks/stems?
    norm = cfg.MODEL.RESNETS.NORM  # 设置BN层：FrozenBN, GN, "SyncBN", "BN"，默认FrozenBN。
    # * 当只有backbone采用BN时，FrozenBN,SyncBN,GN没有明显差异(https://zhuanlan.zhihu.com/p/104069377)
    stem = BasicStem(
        in_channels=input_shape.channels,  # input_shape.channels 即 len(cfg.MODEL.PIXEL_MEAN)
        out_channels=cfg.MODEL.RESNETS.STEM_OUT_CHANNELS,  # 默认64
        norm=norm,
    )

    # fmt: off
    freeze_at = cfg.MODEL.BACKBONE.FREEZE_AT  # 冻结resnet的前几个模块，第一个是普通的卷积层，后面就是一组一组的残差块
    out_features = cfg.MODEL.RESNETS.OUT_FEATURES
    # res4 for C4 backbone
    # FPN：["res2", "res3", "res4", "res5"]
    depth = cfg.MODEL.RESNETS.DEPTH  # 默认50层
    num_groups = cfg.MODEL.RESNETS.NUM_GROUPS  # 1 ==> ResNet; > 1 ==> ResNeXt
    width_per_group = cfg.MODEL.RESNETS.WIDTH_PER_GROUP  # Baseline width of each group.
    bottleneck_channels = num_groups * width_per_group
    in_channels = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
    out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS  # RES2输出的通道数
    stride_in_1x1 = cfg.MODEL.RESNETS.STRIDE_IN_1X1  # 1×1卷积的步长
    # Place the stride 2 conv on the 1x1 filter
    # Use True only for the original MSRA ResNet; use False for C2 and Torch models
    res5_dilation = cfg.MODEL.RESNETS.RES5_DILATION  # RES5应用空洞卷积
    deform_on_per_stage = cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE
    deform_modulated = cfg.MODEL.RESNETS.DEFORM_MODULATED
    deform_num_groups = cfg.MODEL.RESNETS.DEFORM_NUM_GROUPS
    # fmt: on
    assert res5_dilation in {1, 2}, "res5_dilation cannot be {}.".format(res5_dilation)

    num_blocks_per_stage = {
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3],
    }[depth]  # 每个阶段的残差块的数量

    if depth in [18, 34]:
        assert out_channels == 64, "Must set MODEL.RESNETS.RES2_OUT_CHANNELS = 64 for R18/R34"
        assert not any(
            deform_on_per_stage
        ), "MODEL.RESNETS.DEFORM_ON_PER_STAGE unsupported for R18/R34"
        assert res5_dilation == 1, "Must set MODEL.RESNETS.RES5_DILATION = 1 for R18/R34"
        assert num_groups == 1, "Must set MODEL.RESNETS.NUM_GROUPS = 1 for R18/R34"

    stages = []  # 保存stage，每个stage都是一个blocks[残差块]

    # Avoid creating variables without gradients 避免创建没有梯度的变量
    # It consumes extra memory and may cause allreduce to fail
    out_stage_idx = [{"res2": 2, "res3": 3, "res4": 4, "res5": 5}[f]
                     for f in out_features]  # 输出的阶段数

    max_stage_idx = max(out_stage_idx)  # 最后的输出阶段
    for idx, stage_idx in enumerate(range(2, max_stage_idx + 1)):
        # range是个闭开区间，所以要+1
        dilation = res5_dilation if stage_idx == 5 else 1  # 只会设置res5_dilation为空洞卷积
        first_stride = 1 if idx == 0 or (
            stage_idx == 5 and dilation == 2) else 2  # 每个stage中第一个残差块的步长设置，1 or 2
        stage_kargs = {
            "num_blocks": num_blocks_per_stage[idx],  # 每个stage的残差块的数量
            "first_stride": first_stride,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "norm": norm,
        }
        # Use BasicBlock for R18 and R34.
        if depth in [18, 34]:
            stage_kargs["block_class"] = BasicBlock
        else:
            stage_kargs["bottleneck_channels"] = bottleneck_channels
            stage_kargs["stride_in_1x1"] = stride_in_1x1
            stage_kargs["dilation"] = dilation
            stage_kargs["num_groups"] = num_groups
            if deform_on_per_stage[idx]:
                stage_kargs["block_class"] = DeformBottleneckBlock
                stage_kargs["deform_modulated"] = deform_modulated
                stage_kargs["deform_num_groups"] = deform_num_groups
            else:
                # ? 参数何在？
                stage_kargs["block_class"] = BottleneckBlock
        # todo python中的参数
        blocks = make_stage(**stage_kargs)  # 残差块的列表
        in_channels = out_channels
        out_channels *= 2
        bottleneck_channels *= 2
        stages.append(blocks)
    return ResNet(stem, stages, out_features=out_features).freeze(freeze_at)
