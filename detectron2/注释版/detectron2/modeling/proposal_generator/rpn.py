# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
from fvcore.nn import smooth_l1_loss
from torch import nn

from detectron2.config import configurable
from detectron2.layers import ShapeSpec, cat
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.utils.registry import Registry

from ..anchor_generator import build_anchor_generator
from ..box_regression import Box2BoxTransform
from ..matcher import Matcher
from ..sampling import subsample_labels
from .build import PROPOSAL_GENERATOR_REGISTRY
from .proposal_utils import find_top_rpn_proposals

RPN_HEAD_REGISTRY = Registry("RPN_HEAD")
RPN_HEAD_REGISTRY.__doc__ = """
Registry for RPN heads, which take feature maps and perform
objectness classification and bounding box regression for anchors.

The registered object will be called with `obj(cfg, input_shape)`.
The call should return a `nn.Module` object.
"""


"""
Shape shorthand in this module:

    N: number of images in the minibatch
    L: number of feature maps per image on which RPN is run
    A: number of cell anchors (must be the same for all feature maps)
    Hi, Wi: height and width of the i-th feature map
    B: size of the box parameterization

Naming convention:

    objectness:
        refers to the binary classification of an anchor as object vs. not object.

    deltas:
        refers to the 4-d (dx, dy, dw, dh) deltas that parameterize the box2box

    transform (see :class:`box_regression.Box2BoxTransform`), or 5d for rotated boxes.

    pred_objectness_logits:
        预测的分类的分，使用sigmoid得到
        predicted objectness scores in [-inf, +inf];
        use sigmoid(pred_objectness_logits) to estimate P(object).

    gt_labels:
        ground-truth binary classification labels for objectness

    pred_anchor_deltas:
        predicted box2box transform deltas

    gt_anchor_deltas:
        ground-truth box2box transform deltas
"""


def build_rpn_head(cfg, input_shape):
    """
    Build an RPN head defined by `cfg.MODEL.RPN.HEAD_NAME`.
    """
    name = cfg.MODEL.RPN.HEAD_NAME
    return RPN_HEAD_REGISTRY.get(name)(cfg, input_shape)


@RPN_HEAD_REGISTRY.register()
class StandardRPNHead(nn.Module):
    """
    `Faster R-CNN`论文中描述的标准的RPN
    使用一个3x3卷积生成中间隐藏状态特征图，并通过对这个中间隐藏状态特征图两个1x1卷积，分别用于“objectness logits”和“bounding-box deltas”

    Standard RPN classification and regression heads described in :paper:`Faster R-CNN`.
    Uses a 3x3 conv to produce a shared hidden state from which one 1x1 conv predicts objectness logits for each anchor and a second 1x1 conv predicts bounding-box deltas specifying how to deform each anchor into an object proposal.
    """

    @configurable
    def __init__(self, *, in_channels: int, num_anchors: int, box_dim: int = 4):
        """
        NOTE: this interface is experimental.

        Args:
            in_channels (int):
                输入特征的通道数。当使用多个输入特征是，必须拥有相同的通道数
                number of input feature channels. When using multiple input features, they must have the same number of channels.
            num_anchors (int):
                特征图上每个空间位置分配的anchors的数量
                number of anchors to predict for *each spatial position* on the feature map. The total number of anchors for each feature map will be `num_anchors * H * W`.
            box_dim (int):
                box的维度，即box regression为每个anchors预测的数量。一般为4
                dimension of a box, which is also the number of box regression predictions to make for each anchor. An axis aligned box has box_dim=4, while a rotated box has box_dim=5.
        """
        super().__init__()
        # 3x3卷积，输入和输出有相同的通道数
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        # 1x1 conv 用于预测 objectness logits
        # 输出通道数为单个位置上 anchor 的数量：15
        self.objectness_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        # 1x1 conv 用于预测 box2box 转换的 deltas
        # 输出通道数为 num_anchors * box_dim，即 15 * 4 = 60
        self.anchor_deltas = nn.Conv2d(in_channels, num_anchors * box_dim, kernel_size=1, stride=1)

        for l in [self.conv, self.objectness_logits, self.anchor_deltas]:
            # 初始化
            nn.init.normal_(l.weight, std=0.01)  # 使用标准差为0.01进行训练
            nn.init.constant_(l.bias, 0)

    @classmethod
    def from_config(cls, cfg, input_shape):
        # * Standard RPN is shared across levels:
        in_channels = [s.channels for s in input_shape]  # 输入的通道数列表，必须只有一个数字
        assert len(set(in_channels)) == 1, "Each level must have the same channel!"
        in_channels = in_channels[0]

        # RPNHead should take the same input as anchor generator
        # NOTE: it assumes that creating an anchor generator does not have unwanted side effect.
        anchor_generator = build_anchor_generator(cfg, input_shape)
        num_anchors = anchor_generator.num_anchors  # 15
        box_dim = anchor_generator.box_dim  # 4
        assert (
            len(set(num_anchors)) == 1  # python set是一个无限大的不重复元素序列
        ), "Each level must have the same number of anchors per spatial position"
        return {"in_channels": in_channels, "num_anchors": num_anchors[0], "box_dim": box_dim}

    def forward(self, features: List[torch.Tensor]):
        """
        Args:
            features (list[Tensor]):
                list of feature maps 'res4' for C4

        Returns:
            list[Tensor]:
                长度为L（特征图的数量） 的列表。
                每一个元素都是形状为 (N, A, Hi, Wi) 的tensor，代表了对所有anchors预测的结果。
                A list of L elements.
                Element i is a tensor of shape (N, A, Hi, Wi) representing the predicted objectness logits for all anchors. A is the number of cell anchors.
            list[Tensor]:
                长度为L（特征图的数量） 的列表。
                每一个元素都是形状为 (N, A*box_dim, Hi, Wi) 的tensor，代表了对所有anchors预测的"deltas"，用于将其转换为 proposals。
                A list of L elements.
                Element i is a tensor of shape (N, A*box_dim, Hi, Wi) representing the predicted "deltas" used to transform anchors to proposals.
        """
        pred_objectness_logits = []
        pred_anchor_deltas = []
        for x in features:
            t = F.relu(self.conv(x))
            pred_objectness_logits.append(self.objectness_logits(t))
            pred_anchor_deltas.append(self.anchor_deltas(t))
        return pred_objectness_logits, pred_anchor_deltas


@PROPOSAL_GENERATOR_REGISTRY.register()
class RPN(nn.Module):
    """
    默认为RPN
    Region Proposal Network, introduced by :paper:`Faster R-CNN`.
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()

        # fmt: off
        self.min_box_side_len = cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE  # 最小的proposal尺寸大小:0
        self.in_features = cfg.MODEL.RPN.IN_FEATURES
        # RPN网络输入特征图的层数，例如FPN：["res2", "res3", "res4", "res5"], C4：["res4"]
        self.nms_thresh = cfg.MODEL.RPN.NMS_THRESH  # [0.3, 0.7]，会同时取样frontground和background
        self.batch_size_per_image = cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE  # Total number of RPN examples per image
        self.positive_fraction = cfg.MODEL.RPN.POSITIVE_FRACTION  # 前景的取样比例，默认0.5
        self.smooth_l1_beta = cfg.MODEL.RPN.SMOOTH_L1_BETA
        # The transition point from L1 to L2 loss. Set to 0.0 to make the loss simply L1.
        self.loss_weight = cfg.MODEL.RPN.LOSS_WEIGHT  # 1.0
        # fmt: on

        # Map from self.training state to train/test settings
        self.pre_nms_topk = {
            # 在应用NMS之前的保留的top scoring RPN proposals的数量
            # When FPN is used, this is *per FPN level* (not total)
            True: cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN,
            False: cfg.MODEL.RPN.PRE_NMS_TOPK_TEST,
        }
        self.post_nms_topk = {
            # 在应用NMS之后的保留的top scoring RPN proposals的数量
            # When FPN is used, this limit is applied per level and then again to the union of proposals from all levels
            # It means per-image topk here.
            # See "modeling/rpn/rpn_outputs.py" for details.
            True: cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN,
            False: cfg.MODEL.RPN.POST_NMS_TOPK_TEST,
        }
        self.boundary_threshold = cfg.MODEL.RPN.BOUNDARY_THRESH
        # Remove RPN anchors that go outside the image by BOUNDARY_THRESH pixels
        # Set to -1 or a large value, e.g. 100000, to disable pruning anchors

        self.anchor_generator = build_anchor_generator(
            cfg, [input_shape[f] for f in self.in_features]
        )  # "DefaultAnchorGenerator"

        self.box2box_transform = Box2BoxTransform(weights=cfg.MODEL.RPN.BBOX_REG_WEIGHTS)
        # Weights on (dx, dy, dw, dh) for normalizing RPN anchor regression targets
        # 默认：(1.0, 1.0, 1.0, 1.0)

        self.anchor_matcher = Matcher(
            cfg.MODEL.RPN.IOU_THRESHOLDS, cfg.MODEL.RPN.IOU_LABELS, allow_low_quality_matches=True
        )
        self.rpn_head = build_rpn_head(cfg, [input_shape[f] for f in self.in_features])

    def _subsample_labels(self, label):
        """
        对正样本和负样本的子集进行随机抽样，并将未被抽样的样本的标签定义为 (-1)，即ignored。

        Randomly sample a subset of positive and negative examples, and overwrite the label vector to the ignore value (-1) for all elements that are not included in the sample.

        Args:
            labels (Tensor):
                a vector of -1, 0, 1. Will be modified in-place and returned.

        Return:
            label (tensor):
                a vector of -1, 0, 1. 0 代表抽样的负样本，1 代表抽样的正样本
        """
        pos_idx, neg_idx = subsample_labels(
            label, self.batch_size_per_image, self.positive_fraction, 0
        )
        # 尝试返回self.batch_size_per_image个proposals
        # 先用 -1 填充所有的标签，再将正样本的标签设为1，负样本的标签设为0
        # Fill with the ignore label (-1), then set positive and negative labels
        label.fill_(-1)
        label.scatter_(0, pos_idx, 1)  # 0是重写的维度
        label.scatter_(0, neg_idx, 0)
        return label

    @torch.no_grad()
    def label_and_sample_anchors(self, anchors: List[Boxes], gt_instances: List[Instances]):
        """
        Args:
            anchors (list[Boxes]):
                anchors for each feature map.
            gt_instances:
                the ground-truth instances for each image.

        Returns:
            list[Tensor]:
                #img tensors的列表。第 i 个元素是第i张图像的label向量，该向量长度为所有特征图的anchors的总和(sum(Hi * Wi * A))。
                label的值是 {-1, 0, 1}，-1 = ignore; 0 = negative class; 1 = positive class
                List of #img tensors. i-th element is a vector of labels whose length is the total number of anchors across all feature maps (sum(Hi * Wi * A)).
                Label values are in {-1, 0, 1}, with meanings: -1 = ignore; 0 = negative class; 1 = positive class.
            list[Tensor]:
                每一个元素是一个 Nx4 的 tensor，N 是所有特征图的 anchors 的总数，
                值是每一个anchor匹配的GT boxes。
                * 但是未被标记为 1 的anchors的值未定义。
                i-th element is a Nx4 tensor, where N is the total number of anchors across feature maps.
                The values are the matched gt boxes for each anchor.
                Values are undefined for those anchors not labeled as 1.
        """
        anchors = Boxes.cat(anchors)

        # *模型输入的dict包含以下字段：
        # -image: tensor, (C, H, W)
        # -imstance: 包含以下字段：
        #     -gt_boxes: 一个 Boxes 对象，存储了N个box
        #     -gt_classes: tensor, 包含N个label的向量
        #     -image_size(可能不是固有的字段，而是类方法)
        gt_boxes = [x.gt_boxes for x in gt_instances]
        image_sizes = [x.image_size for x in gt_instances]
        del gt_instances

        gt_labels = []
        matched_gt_boxes = []
        for image_size_i, gt_boxes_i in zip(image_sizes, gt_boxes):
            """
            image_size_i (tuple[H, W]):
                (h, w) for the i-th image
            gt_boxes_i (tensor):
                ground-truth boxes for i-th image, 每一个image对应N个gt_boxes
                N是图像中instances的数量？
            """

            # * 进行GT和anchors之间的匹配并分配标签(-1, 0, 1)
            match_quality_matrix = retry_if_cuda_oom(pairwise_iou)(gt_boxes_i, anchors)
            # 计算所有gt_boxes_i和anchors之间的IoU，并返回matched_idxs和gt_labels_i。matched_idxs代表N个anchor相匹配的GT的索引，gt_labels_i代表预测的标签
            # ? 会过滤掉不匹配的anchors吗？应该会在随机取样的时候过滤吧
            matched_idxs, gt_labels_i = retry_if_cuda_oom(self.anchor_matcher)(match_quality_matrix)
            # matched_idxs, gt_labels_i：两个长度为‘N’的tensor（不一定是minibatch），N为跨所有anchors的数量
            # Matching is memory-expensive and may result in CPU tensors. But the result is small
            gt_labels_i = gt_labels_i.to(device=gt_boxes_i.device)
            del match_quality_matrix

            if self.boundary_threshold >= 0:
                # 丢弃超出图像边界的anchors
                # NOTE: This is legacy functionality that is turned off by default in Detectron2
                anchors_inside_image = anchors.inside_box(image_size_i, self.boundary_threshold)
                gt_labels_i[~anchors_inside_image] = -1

            # 对正样本和负样本的子集进行随机抽样，并将未被抽样的样本的标签定义为(-1)，即ignored。
            # A vector of labels (-1, 0, 1) for each anchor
            gt_labels_i = self._subsample_labels(gt_labels_i)

            if len(gt_boxes_i) == 0:
                # ? 当没有实例的情况下
                # These values won't be used anyway since the anchor is labeled as background
                matched_gt_boxes_i = torch.zeros_like(anchors.tensor)
            else:
                # TODO wasted indexing computation for ignored boxes
                matched_gt_boxes_i = gt_boxes_i[matched_idxs].tensor
                # matched_gt_boxes_i：第i张图像的所有anchors所匹配的gt_box（不是索引）

            # i-th个元素：第i张图像的正负样本集合的标签和匹配的gt_boxes，长度均是N，即图像的数量
            # 注意，每个图像所有的anchors（跨不跨特征图不确定）的gt_labels和匹配的gt_boxes（不是索引）都在这儿
            gt_labels.append(gt_labels_i)  # List[AHW], 原: N,AHW
            matched_gt_boxes.append(matched_gt_boxes_i)  # List[Boxes]
        return gt_labels, matched_gt_boxes

    def losses(
        self,
        anchors,
        pred_objectness_logits: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
        pred_anchor_deltas: List[torch.Tensor],
        gt_boxes,
    ):
        """
        Return the losses from a set of RPN predictions and their associated ground-truth.

        Args:
            anchors (list[Boxes or RotatedBoxes], 长度:L):
                anchors for each feature map, each has shape (Hi*Wi*A, B), where B is box dimension (4 or 5).
            pred_objectness_logits (list[Tensor], 长度:L, tensor:(N, Hi*Wi*A)):
                预测的anchors的标签，即objectness_logits输出的标签，
                shape：(N, Hi*Wi*A)，该list的长度为L(每张图像特征图的数量)
            gt_labels (list[Tensor], 长度:N, tensor:sum(Hi * Wi * A)):
                实际的anchors的标签：-1，0，1
                Output of :meth:`label_and_sample_anchors`.
            pred_anchor_deltas (list[Tensor], 长度:L, tensor:(N, Hi*Wi*A, 4)):
                A list of L elements. Element i is a tensor of shape (N, Hi*Wi*A, 4 or 5) representing the predicted "deltas" used to transform anchors to proposals.
            gt_boxes (list[Boxes or RotatedBoxes], 长度:N):
                anchors所匹配的gt_boxes的索引
                Output of :meth:`label_and_sample_anchors`.

        Returns:
            dict[loss name -> loss value]:
                A dict mapping from loss name to loss value.
                Loss names are: `loss_rpn_cls` for objectness classification and
                `loss_rpn_loc` for proposal localization.
        """
        num_images = len(gt_labels)
        gt_labels = torch.stack(gt_labels)  # (N, sum(Hi*Wi*Ai)), 将列表转换为tensor
        anchors = type(anchors[0]).cat(anchors).tensor  # Ax(4 or 5), (L, Hi*Wi*A, B), 默认在第0维进行cat
        gt_anchor_deltas = [self.box2box_transform.get_deltas(anchors, k) for k in gt_boxes]
        gt_anchor_deltas = torch.stack(gt_anchor_deltas)  # (N, sum(Hi*Wi*Ai), 4 or 5), 真实的deltas

        # 记录在训练过程中每张图像positive/negative anchors的数量
        pos_mask = gt_labels == 1  # False和True的列表
        num_pos_anchors = pos_mask.sum().item()  # True（positive）的数量
        num_neg_anchors = (gt_labels == 0).sum().item()  # nagetive的数量
        storage = get_event_storage()
        storage.put_scalar("rpn/num_pos_anchors", num_pos_anchors / num_images)
        storage.put_scalar("rpn/num_neg_anchors", num_neg_anchors / num_images)

        localization_loss = smooth_l1_loss(
            cat(pred_anchor_deltas, dim=1)[pos_mask],
            gt_anchor_deltas[pos_mask],
            self.smooth_l1_beta,
            reduction="sum",
        )
        valid_mask = gt_labels >= 0
        objectness_loss = F.binary_cross_entropy_with_logits(
            cat(pred_objectness_logits, dim=1)[valid_mask],
            gt_labels[valid_mask].to(torch.float32),
            reduction="sum",
        )
        normalizer = self.batch_size_per_image * num_images
        # 一个batch的proposal的总数
        return {
            "loss_rpn_cls": objectness_loss / normalizer,
            "loss_rpn_loc": localization_loss / normalizer,
        }

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        gt_instances: Optional[Instances] = None,
    ):
        """
        Args:
            images (ImageList):
                长度为N的ImageList
                input images of length `N`
            features (dict[str, Tensor]):
                特征图名称str到tensor的映射. 该tensor的形状为 (N, C, H, W)，N为输入数据的数量
                input data as a mapping from feature map name to tensor. Axis 0 represents the number of images `N` in the input data; axes 1-3 are channels, height, and width, which may vary between feature maps (e.g., if a feature pyramid is used).
            gt_instances (list[Instances], optional):
                长度为N的 Instances 列表
                每一个`Instances`都储存了相关联的图像的 ground-truth instances
                Each `Instances` stores ground-truth instances for the corresponding image.

        Returns:
            proposals: list[Instances]:
                contains fields "proposal_boxes", "objectness_logits"
            loss:
                dict[Tensor] or None
        """
        features = [features[f] for f in self.in_features]
        anchors = self.anchor_generator(features)
        # anchors: List[Boxes], i-th element 代表了第i个特征图的Boxes，而每一个Boxes是一个(N, 4)的tensor，N = h x w x (sizes x aspect_ratios)

        pred_objectness_logits, pred_anchor_deltas = self.rpn_head(features)

        # Transpose the Hi*Wi*A dimension to the middle:
        pred_objectness_logits = [
            # Reshape: (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
            score.permute(0, 2, 3, 1).flatten(1)
            # tensor.permute(*dim)，形状转换
            for score in pred_objectness_logits
        ]  # * [tensor(N, Hi*Wi*A)]
        pred_anchor_deltas = [
            # Reshape: (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) -> (N, Hi*Wi*A, B)
            x.view(x.shape[0], -1, self.anchor_generator.box_dim, x.shape[-2], x.shape[-1])
            .permute(0, 3, 4, 1, 2)
            .flatten(1, -2)
            for x in pred_anchor_deltas
        ]  # * [tensor(N, Hi*Wi*A, B)]

        if self.training:
            gt_labels, gt_boxes = self.label_and_sample_anchors(anchors, gt_instances)
            # anchors: List[Boxes], gt_instances: [tensor(N, Hi*Wi*A, B)]
            # gt_labels, gt_boxes长度为N的列表，分别代表了N个图像的跨所有level的特征图的所有(Hi*Wi*A)anchors的labels和deltas
            losses = self.losses(
                anchors, pred_objectness_logits, gt_labels, pred_anchor_deltas, gt_boxes
            )
            losses = {k: v * self.loss_weight for k, v in losses.items()}
        else:
            losses = {}

        proposals = self.predict_proposals(
            anchors, pred_objectness_logits, pred_anchor_deltas, images.image_sizes
        )
        return proposals, losses

    @torch.no_grad()
    def predict_proposals(
        self,
        anchors,
        pred_objectness_logits: List[torch.Tensor],
        pred_anchor_deltas: List[torch.Tensor],
        image_sizes: List[Tuple[int, int]],
    ):
        """
        将所有预测的边框回归deltas编码为proposals。去除过小的边框，并应用NMS算法，找到 top proposals
        Decode all the predicted box regression deltas to proposals. Find the top proposals by applying NMS and removing boxes that are too small.

        Returns:
            proposals (list[Instances]):
                N个Instances的列表。第i个Instances存储了图像i的post_nms_topk个proposals，并通过其 objectness 分数降序排序。
                Instances具有如下字段：
                - proposal_boxes
                - objectness_logits
                - image_size（类方法）
                list of N Instances.
                The i-th Instances stores post_nms_topk object proposals for image i, sorted by their objectness score in descending order.
        """
        # The proposals are treated as fixed for approximate joint training with roi heads.
        # This approach ignores the derivative w.r.t. the proposal boxes’ coordinates that are also network responses, so is approximate.
        pred_proposals = self._decode_proposals(anchors, pred_anchor_deltas)
        # proposals(list[Tensor]):
        #     长度为L的列表. Tensor i has shape(N, Hi * Wi * A, B)
        return find_top_rpn_proposals(
            pred_proposals,
            pred_objectness_logits,
            image_sizes,
            self.nms_thresh,
            self.pre_nms_topk[self.training],
            self.post_nms_topk[self.training],
            self.min_box_side_len,
            self.training,
        )

    def _decode_proposals(self, anchors, pred_anchor_deltas: List[torch.Tensor]):
        """
        Transform anchors into proposals by applying the predicted anchor deltas.
        Args:
            anchors (List[Boxes]):
                i-th 元素代表了第i个特征图的Boxes，而每一个Boxes是一个(N, 4)的tensor，N = h x w x (sizes x aspect_ratios)
            pred_anchor_deltas (List[torch.Tensor]):
                长度为L的列表。
                i-th 元素代表了第i个特征图的每个anchor预测的deltas
                torch.Tensor：(N, Hi*Wi*A, B)

        Returns:
            proposals (list[Tensor]):
                A list of L tensors. Tensor i has shape (N, Hi*Wi*A, B)
        """
        N = pred_anchor_deltas[0].shape[0]  # 图像的数量
        proposals = []
        # For each feature map
        for anchors_i, pred_anchor_deltas_i in zip(anchors, pred_anchor_deltas):
            B = anchors_i.tensor.size(1)  # box的维度
            # (N, Hi*Wi*A, B) -> ((N*Hi*Wi*A, B))
            pred_anchor_deltas_i = pred_anchor_deltas_i.reshape(-1, B)
            anchors_i = anchors_i.tensor.unsqueeze(0).expand(N, -1, -1).reshape(-1, B)
            # (Hi*Wi*A, B) -> (1, Hi*Wi*A, B) -> (N, Hi*Wi*A, B) -> (N, Hi*Wi*A, B)
            proposals_i = self.box2box_transform.apply_deltas(pred_anchor_deltas_i, anchors_i)
            # Append feature map proposals with shape (N, Hi*Wi*A, B)
            proposals.append(proposals_i.view(N, -1, B))
            # torch.tersor.view(): 返回一个新的张量但不具有相同的shape
        return proposals
