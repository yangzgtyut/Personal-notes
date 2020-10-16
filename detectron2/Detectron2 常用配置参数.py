# Detectron2 常用配置参数


_C = CN()
# *版本号
_C.VERSION = 2


# 将要加载到模型的检查点文件的路径（文件路径或URL，如detectron2://.., https://..）。您可以在model_zoo中找到可用的模型。
_C.MODEL.WEIGHTS = ""


# 输入INPUT
_C.INPUT = CN()
# *图像增强
# `True`，如果在训练期间将裁剪用于数据增强
_C.INPUT.CROP = CN({"ENABLED": False})  # 细看


# *dataset数据集
_C.DATASETS = CN()
# *用于训练的数据集名称列表。必须在DatasetCatalog中注册
_C.DATASETS.TRAIN = ()
# 训练过程中产生的proplsal数
_C.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN = 2000
# *用于测试的数据集名称列表。必须在DatasetCatalog中注册
_C.DATASETS.TEST = ()
# 测试过程中产生的proplsal数
_C.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST = 1000


# *DataLoader数据加载器
_C.DATALOADER = CN()
# 数据加载线程数
_C.DATALOADER.NUM_WORKERS = 4
# 如果True, dataloader将会过滤没有标注的图像
_C.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True


# backbone骨干网络
_C.MODEL.BACKBONE = CN()
_C.MODEL.BACKBONE.NAME = "build_resnet_backbone"
# 冻结resnet的前几个阶段
_C.MODEL.BACKBONE.FREEZE_AT = 2
# resnet有五个阶段，The first is a convolution, and the following stages are each group of residual blocks.


# *FPN options


# *Proposal generator options
_C.MODEL.PROPOSAL_GENERATOR = CN()
# 当前的proposal生成器包括 "RPN", "RRPN",  "PrecomputedProposals"
_C.MODEL.PROPOSAL_GENERATOR.NAME = "RPN"


# todo Anchor生成选项
_C.MODEL.ANCHOR_GENERATOR = CN()


# todo RPN options


# todo ROI HEADS options


# todo Box Head


# *ResNe[X]t options (ResNets = {ResNet, ResNeXt}
# Note that parts of a resnet may be used for both the backbone and the head
# These options apply to both


# *Solver超参数


# *test options


# *Misc options其它选项
# 输出文件的输出目录
_C.OUTPUT_DIR = "./output"










# 可能会用到的cfg配置参数

# 用于图像标准化的值（BGR顺序，因为INPUT.FORMAT默认为BGR）。要在不同数量的通道图像上进行训练，只需设置不同的均值和标准差。
# 默认值为ImageNet的平均像素值：[103.53、116.28、123.675]
_C.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675]
_C.MODEL.PIXEL_STD = [1.0, 1.0, 1.0]  # (ImageNet std)
