# Detectron2

> :smiley_cat:
>

## 使用命令行模型训练、测试和推理

内置了一个`tools/{,plain_}train_net.py`可以作为训练的模板。

### 训练

要使用`train_net.py`训练的时候，需要：

1. 先设置成[内置的数据集](https://github.com/facebookresearch/detectron2/blob/master/datasets/README.md)：

   ```
   # coco目标检测和关键点检测
   
   coco/
     annotations/
       instances_{train,val}2017.json
       person_keypoints_{train,val}2017.json
     {train,val}2017/
       # image files that are mentioned in the corresponding json
   ```

2. 运行（需要修改GPU数量）

   ```
   cd tools/
   ./train_net.py --num-gpus 8 \
   	--config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yam
   ```

### 测试模型性能

需要设置`--eval-only MODEL.WEIGHTS /path/to/checkpoint_file`

```
./train_net.py \
	--config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
	--eval-only MODEL.WEIGHTS /path/to/checkpoint_file
```

### 推理

1. 在从[模型Zoo中](https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md)选择一个模型及其配置文件 ，例如`mask_rcnn_R_50_FPN_3x.yaml`。

2. `demo.py`能够运行内置的标准模型：

   ```
   cd demo/
   python demo.py --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml \
     --input input1.jpg input2.jpg \
     [--other-options]
     --opts MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl
   ```

   此命令将运行推断并在OpenCV窗口中显示可视化效果。

   要将输出保存到目录（用于图像）或文件（用于网络摄像头或视频），请使用`--output`。

## 扩展detectron2

两类接口：

1. 带有`cfg`参数的类和方法
2. 带有自定义参数的方法和类，更加灵活
3. 装饰器

## 数据集dataset

### 内置的标准数据集

假定数据集存在于环境变量指定的目录中 `DETECTRON2_DATASETS`。在该目录下，detectron2希望在下面描述的结构中查找数据集。

设置数据集的位置：`export DETECTRON2_DATASETS=/path/to/datasets`

默认位置：工作目录下的`./datasets`

#### coco目标检测和关键点检测

**[x, y, w, h]：[x, y]是目标左上角的坐标**

```
coco/
  annotations/
    instances_{train,val}2017.json
    person_keypoints_{train,val}2017.json
  {train,val}2017/
    # image files that are mentioned in the corresponding json
```

#### Pascal VOC

```
VOC20{07,12}/
  Annotations/
  ImageSets/
    Main/
      trainval.txt
      test.txt
      # train.txt or val.txt, if you use these splits
  JPEGImages/
```

#### 标准数据集字典：list[dict]

将原始数据集加载到`list[dict]`具有类似于COCO json注释的规范中。

每个字典包含有关一个图像的信息。`dict`可能具有以下字段，并且必填字段根据数据加载器或任务的需要而有所不同（请参阅下文）。

- `file_name`: 图像文件的(完整)路径。在拥有扭曲和翻转等此类exif信息时进行扭曲和翻转。

- `height`，`width`

- `image_id`：str或者int，该图像的唯一标识符。

- **`annotations`：==list[dict]==，每一个字典关联一个图像中的对象。**

  **具有空白`annotations`的图像将在训练阶段被默认移除。也可以通过`DATALOADER.FILTER_EMPTY_ANNOTATIONS`利用**。

  包括`bbox`、`bbox_mode`、`category_id`等。

  - `bbox`：list[float]
  - `bbox_mode`：bbox的格式。`BoxMode.XYXY_ABS`或`BoxMode.XYWH_ABS`
  - `category_id`：int，[0, num_categories)。如果适用，num_categories表示背景类。
  - `segmentation`：list[list[float]] or dict。
    - `list[list[float]]`：表示多边形列表，每个多边形表示一个对象。 每个`list[float]`是一个简单的多边形，格式为`[x1，y1，...，xn，yn]`。 Xs和Ys是[0, 1]中的相对坐标，还是绝对坐标，取决于“bbox_mode”。
    - `dict`：per-pixel segmentation mask in COCO’s RLE format
  - `keypoints` ：list[float]
  
- `sem_seg_file_name`：语义分割文件的完整路径。

### 自定义数据集

1. 注册数据集

2. （可选）注册元数据

#### 注册自定义数据集

1. 定义一个函数，返回数据集字典
2. 注册数据集

```python
def my_dataset_function():
  ...
  return list[dict] in the following format

from detectron2.data import DatasetCatalog
DatasetCatalog.register("my_dataset", my_dataset_function)
```

*class* `detectron2.data.DatasetCatalog`

> 一个catalog，存储有关数据集及其获取方式的信息。
>
> 包含从字符串（即标识数据集的名称，例如“ coco_2014_train”）到解析该数据集并以list[dict]格式返回样本的函数的映射。
>
> 如果与 *data/build.py,data/detection_transform.py* 中的数据加载器功能一起使用，则返回的字典应采用Detectron2数据集格式（有关详细信息，请参见DATASETS.md）。
>
> 此catalog的目的是，仅通过使用cfg中的字符串即可轻松选择不同的数据集。
>
> > *static* `register`(*name*, *func*)
> >
> > - Parameters
> >
> >   **name** ([*str*](https://docs.python.org/3.6/library/stdtypes.html#str)) – the name that identifies a dataset, e.g. “coco_2014_train”. 
> >
> >   **func** (*callable*) – a callable which takes no arguments and returns a list of dicts. （可以有参数）
> >
> > *static* `clear`()
> >
> > Remove all registered dataset.

该函数返回的数据集格式：最好是Detectron2的标准数据集字典list[dict]。

##### 新任务的自定义数据集字典

在数据集函数返回的list[dict]中，字典也可以具有任意的自定义数据。 这对于标准数据集字典不支持的额外信息的新任务很有用。

In this case, you need to make sure the downstream code can handle your data correctly. Usually this requires writing a new `mapper` for the dataloader (see [Use Custom Dataloaders](https://detectron2.readthedocs.io/tutorials/data_loading.html)).

需要自定义数据加载器。

#### 注册coco类型的数据集

如果已经有coco标注格式的`json`文件，数据集和元数据可以通过以下方式注册：

```python
from detectron2.data.datasets import register_coco_instances
register_coco_instances("my_dataset_train", {}, "json_annotation_train.json", "path/to/image/dir")
register_coco_instances("my_dataset_val", {}, "json_annotation_val.json", "path/to/image/dir")
```

如果您的数据集为COCO格式，但带有额外的自定义按实例注释，则[load_coco_json](https://detectron2.readthedocs.io/modules/data.html#detectron2.data.datasets.load_coco_json)函数可能会很有用。

#### 获取已注册数据集的数据字典

1. `detectron2.data.get_detection_dataset_dicts` (*dataset_names***,** *filter_empty=True***,** *min_keypoints=0***,** *proposal_files=None*)

   > 加载并准备数据集字典以进行实例检测/分割和语义分割。
   >
   > **Parameters**
   >
   > - **dataset_names** ([*list*](https://docs.python.org/3.6/library/stdtypes.html#list)*[*[*str*](https://docs.python.org/3.6/library/stdtypes.html#str)*]*) – a list of dataset names
   > - **filter_empty** ([*bool*](https://docs.python.org/3.6/library/functions.html#bool)) – whether to filter out images without instance annotations
   > - **min_keypoints** ([*int*](https://docs.python.org/3.6/library/functions.html#int)) – filter out images with fewer keypoints than min_keypoints. Set to 0 to do nothing.
   > - **proposal_files** ([*list*](https://docs.python.org/3.6/library/stdtypes.html#list)*[*[*str*](https://docs.python.org/3.6/library/stdtypes.html#str)*]*) – if given, a list of object proposal files that match each dataset in dataset_names.

2. ```python
   from detectron2.data.datasets import load_coco_json
   dataset_dicts = load_coco_json("my_dataset_train.json", "path/to/images/folder", "my_dataset_train")
   ```


#### 元数据

每个数据集都与一些元数据相关联，这些元数据可以通过`MetadataCatalog.get(dataset_name).some_metadata`访问。 元数据是一个键值映射，其中包含在整个数据集中共享的信息，通常用于解释数据集中的内容，例如，类的名称，类的颜色，文件的根目录等。此信息对于 元数据的结构取决于相应下游代码的需求。

##### 获取元数据

*class* `detectron2.data.MetadataCatalog`

> MetadataCatalog提供了访问给定数据集的接口。
>
> The metadata associated with a certain name is a singleton: once created, the metadata will stay alive and will be returned by future calls to get(name).
>
> It’s like global variables, so don’t abuse it. It’s meant for storing knowledge that’s constant and shared across the execution of the program, e.g.: the class names in COCO.
>
> > *static* `get`(*name*)
> >
> > - Parameters
> >
> >   **name** ([*str*](https://docs.python.org/3.6/library/stdtypes.html#str)) – name of a dataset (e.g. coco_2014_train).
> >
> > - Returns
> >
> >   *Metadata* – The `Metadata` instance associated with this name, or create an empty one if none is available.

##### 设置元数据

如果通过来注册新的数据集`DatasetCatalog.register`，则可能还需要通过来添加其相应的元数据 ，以启用需要该元数据的所有功能。您可以这样做（以元数据键“ `thing_classes`”为例）：

`MetadataCatalog.get(dataset_name).some_key = some_value`

```python
from detectron2.data import MetadataCatalog
MetadataCatalog.get("my_dataset").thing_classes = ["person", "dog"]
```

这是detectron2的内置功能所使用的元数据密钥的列表。如果添加自己的数据集而没有这些元数据，则某些功能可能对您不可用：

- `thing_classes`（list[str]）：由所有实例检测/分段任务使用。每个实例/事物类别的名称列表。如果加载COCO格式的数据集，它将由函数自动设置`load_coco_json`。
- `thing_colors`（list[tuple(r，g，b)]）：每个事物类别的预定义颜色（在[0，255]中）。用于可视化。如果未给出，则使用随机颜色。
- `stuff_classes`（list[str]）：用于语义和全景分割任务。每个物料类别的名称列表。
- `stuff_colors`（list[tuple(r，g，b)]）：每个填充类别的预定义颜色（在[0，255]中）。用于可视化。如果未给出，则使用随机颜色。

一些特定于某些数据集评估的其他元数据（例如COCO）：

- `thing_dataset_id_to_contiguous_id`（dict[int->int]）：由COCO格式的所有实例检测/分割任务使用。从数据集中的实例类ID到[0，#class）范围内的连续ID的映射。该功能将由`load_coco_json`自动设置。
- `stuff_dataset_id_to_contiguous_id`（dict[int->int]）：在生成用于语义/全景分割的预测json文件时使用。从数据集中的语义分段类ID到[0，num_categories）中的连续ID的映射。它仅对评估有用。
- `json_file`：COCO注释json文件。由COCO评估用于COCO格式的数据集。

#### 更新新数据集的config

一旦注册了新数据集，可以在`cfg.DATASETS.{TRAIN,TEST}`中使用数据集的名字，例如“my_dataset”。

对于新数据，可能有一些训练或测试有关的参数需要更改：

- **`MODEL.ROI_HEADS.NUM_CLASSES`  `MODEL.RETINANET.NUM_CLASSES`：the number of thing classes for R-CNN.**
-  `MODEL.RETINANET.NUM_CLASSES`： the number of thing classes for RetinaNet.
- `MODEL.SEM_SEG_HEAD.NUM_CLASSES` sets the number of stuff classes for Semantic FPN & Panoptic FPN.

## 数据加载器

### 现有的数据加载器

Detectron2 提供了两个函数[`build_detection_{train,test}_loader`](https://detectron2.readthedocs.io/modules/data.html#detectron2.data.build_detection_train_loader)，用于从给定的config创建默认的数据加载器。这是其工作原理：

1. 采用已注册数据集的名称，并加载代表该数据集项的list[dict]。这些数据项尚未加载到内存中。
2. 此list中的每个dict都由一个函数（“mapper”）映射：
   - 用户可以通过在`build_detection_{train, test}_loader`中指定“mapper”参数来定制此映射功能。 默认映射器是`DatasetMapper`。
   - The output format of such function can be arbitrary, as long as it is accepted by the consumer of this data loader (usually the model). The outputs of the default mapper, after batching, follow the default model input format documented in [Use Models](https://detectron2.readthedocs.io/tutorials/models.html#model-input-format).
   - `mapper`的作用是将数据集项目的轻量级规范表示转换为可供模型使用的格式（包括例如读取图像，执行随机数据增强并转换为torch张量）。 如果要对数据执行自定义转换，则通常需要自定义映射器。
3. 对mapper的输出进行批量化组合成 batch，通常简单的组合为一个list。
4. 批量化数据即data loader的输出。 通常，它也是 `model.forward()`的input。

### 自定义数据加载器

在`build_detection_{train,test}_loader(mapper=)`中使用一个不同的加载器适用于大多数自定义数据的情况。例如，如果你想将所有图像调整为固定大小以进行Mask R-CNN训练，请编写以下代码：

```python
from detectron2.data import build_detection_train_loader
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils

def mapper(dataset_dict):
	# 实现一个mapper, 和默认的DatasetMapper相似, but with your own customizations
	dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
	image = utils.read_image(dataset_dict["file_name"], format="BGR")
    # 读取图像
	image, transforms = T.apply_transform_gens([T.Resize((800, 800))], image)
    # 大小转变
	dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
    # 将图像转化为tensor

	annos = [
		utils.transform_instance_annotations(obj, transforms, image.shape[:2])
		for obj in dataset_dict.pop("annotations")
		if obj.get("iscrowd", 0) == 0
	]
	instances = utils.annotations_to_instances(annos, image.shape[:2])
	dataset_dict["instances"] = utils.filter_empty_instances(instances)
	return dataset_dict

# 此代码只更改了mapper，没有更改数据加载部分（加载到内存中，生成模型接受的格式，并进行批量化，）
data_loader = build_detection_train_loader(cfg, mapper=mapper)
# use this dataloader instead of the default
```

如果您不仅要更改`mapper`（例如，编写不同的sample或批处理逻辑），还可以编写自己的数据加载器。**数据加载器只是一个python迭代器，它生成模型接受的[格式](https://detectron2.readthedocs.io/tutorials/models.html)。**您可以使用任何喜欢的工具来实现它。

## 模型model

detectron2中的模型及其子模型由`build_model`，`build_backbone`，`build_roi_heads`这样的函数构建。

```python
from detectron2.modeling import build_model
model = build_model(cfg)  # returns a torch.nn.Module
```

`build_model`仅构建模型结构，并用**随机初始化**。

### 加载/保存 checkpoint

```python
# 加载
from detectron2.checkpoint import DetectionCheckpointer
DetectionCheckpointer(model).load(file_path_or_url)  # load a file, usually from cfg.MODEL.WEIGHTS

# 保存
checkpointer = DetectionCheckpointer(model, save_dir="output")
checkpointer.save("model_999")  # save to output/model_999.pth
```

**Detectron2的checkpointer可以识别pytorch的`.pth`格式和model_zoo中的`pkl`格式。**

**可以通过`torch.{load,save}`任意操作`.pth`文件，通过 `pickle.{dump,load}`任意的操作`.pkl`文件。**

### 模型输入/输出

#### 输入格式：list[dict]

用户可以实现支持任何任意输入格式的自定义模型。在这里，我们描述了**detectron2中所有内置模型都支持的标准输入格式**。 **它们都以`list[dict]`作为输入**。 **每个字典对应于有关一个图像的信息。**

该dict应该包括如下keys：

- `"image"`：`tensor`，**(C, H, W)**格式。

  > **==利用`cv2.imread()`读到的格式是(H, W, C)的numpy数组，且是BGR格式。==**
  >
  > **==因此在数据加载器中要对其进行格式转化，将其转成(C, H, W)格式的tensor。==**

  C的含义可以通过`cfg.INPUT.FORMAT`定义。

  图像归一化（如果有），将在模型内部使用`cfg.MODEL.PIXEL_{MEAN, STD}`进行。

- `"instances"`： 一个[Instances](https://detectron2.readthedocs.io/modules/structures.html#detectron2.structures.Instances)对象，拥有以下字段：

  - `"gt_boxes"`：一个[Boxes](https://detectron2.readthedocs.io/modules/structures.html#detectron2.structures.Boxes)对象，存储了N个box，每一个都对应了一个实例。
  - `"gt_classes"`：一个`tensor`，包括N个label的向量，范围是[0, num_classes)
  - image_size(可能不是固有的字段，而是类方法)
  - `"gt_masks"`：一个[PolygonMasks](https://detectron2.readthedocs.io/modules/structures.html#detectron2.structures.PolygonMasks)或者[BitMasks](https://detectron2.readthedocs.io/modules/structures.html#detectron2.structures.BitMasks)，存储了N个mask，每一个都对应了一个实例。
  - `"gt_keypoints"`

- `"height"`, `"width"`：期望输出的分辨率，非必需。若不提供，则输出原始分辨率。

- `"sem_seg"`: `Tensor[int]` in (H, W) format。语义分割真值，值代表从0开始的类别标签。


#### 模型和数据加载器的连接

默认 DatasetMapper 的输出是遵循上述格式的dict。 数据加载器执行批处理后，它将成为内置模型支持的`list[dict]`。

#### 模型输出格式

==在**训练模式**下，内置模型会输出一个包括所有损失的`dict[str-> Scalar Tensor]`。==

在**推理模式**下，内置模型输出`list[dict]`，每个图像一个`dict`。 根据模型正在执行的任务，每个字典可能包含以下字段：

- `"instances"`：[Instances](https://detectron2.readthedocs.io/modules/structures.html#detectron2.structures.Instances)对象，包含以下字段：
  - `"pred_boxes"`：一个[Boxes](https://detectron2.readthedocs.io/modules/structures.html#detectron2.structures.Boxes)对象，存储了N个box，每一个都对应了一个实例。
  - `"scores"`：包含N个scores的向量
  - `"pred_classes"`：`Tensor`，包括N个label的向量，范围是[0, num_classes)
  - `"pred_masks"`: 一个`Tensor`，(N, H, W)，masks for each detected instance.
  - `"pred_keypoints"`
- `"sem_seg"`：`Tensor`，(num_categories, H, W)，语义分割预测
- `proposals`

### 部分的执行模型

有时您可能想在模型内部获得中间张量。 由于通常有数百个中间张量，因此没有API可以为您提供所需的中间结果。 您有以下选择：

1. 写一个(子)模型。

2. 部分地执行一个模型。

   使用自定义的代码执行，而不是`forward()`，For example, the following code obtains mask features before mask head.

   ```python
   images = ImageList.from_tensors(...)  # preprocessed input tensor
   model = build_model(cfg)
   features = model.backbone(images.tensor)
   proposals, _ = model.proposal_generator(images, features)
   instances = model.roi_heads._forward_box(features, proposals)
   mask_features = [features[f] for f in model.roi_heads.in_features]
   mask_features = model.roi_heads.mask_pooler(mask_features, [x.pred_boxes for x in instances])
   ```

每一种方法都需要理解已有的`forward()`代码。

### 自定义模型

在许多情况下，您可能会对修改或扩展现有模型的某些组件感兴趣。因此，我们还提供了一种注册机制，使您可以覆盖标准模型的某些内部组件的行为。

```python
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec

@BACKBONE_REGISTRY.register()
class ToyBackBone(Backbone):
  def __init__(self, cfg, input_shape):
    super().__init__()
    # create your own backbone
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=16, padding=3)

  def forward(self, image):
    return {"conv1": self.conv1(image)}

  def output_shape(self):
    return {"conv1": ShapeSpec(channels=64, stride=16)}
```

**然后，你可以在`build_model(cfg)`中使用`cfg.MODEL.BACKBONE.NAME = 'ToyBackBone"`，`build_model(cfg)`将会调用 `ToyBackBone`。**

作为另一个示例，要**向Generalized R-CNN元体系结构中的ROI头添加新功能**，您可以实现一个新的[ROIHeads](https://detectron2.readthedocs.io/modules/modeling.html#detectron2.modeling.ROIHeads)子类，并将其放在`ROI_HEADS_REGISTRY`中。 有关实现新ROIHeads来执行新任务的示例，请参见[densepose in detectron2](https://github.com/facebookresearch/detectron2/blob/master/projects/DensePose)和[meshrcnn](https://github.com/facebookresearch/meshrcnn)。 [projects/](https://github.com/facebookresearch/detectron2/blob/master/projects/)包含更多实现不同体系结构的示例。

完整的注册表列表可在[API documentation](https://detectron2.readthedocs.io/modules/modeling.html#model-registries)中找到。 您可以在这些注册表中注册组件，以自定义模型的不同部分或整个模型。

### 自定义模型训练

自定义optimizer和训练逻辑。例子：[tools/plain_train_net.py](https://github.com/facebookresearch/detectron2/blob/master/tools/plain_train_net.py)。

我们也提供了一个标准化的 “trainer” abstraction，with a [minimal hook system](https://detectron2.readthedocs.io/modules/engine.html#detectron2.engine.HookBase) that helps simplify the standard types of training.

可以使用 [SimpleTrainer().train()](https://detectron2.readthedocs.io/modules/engine.html#detectron2.engine.SimpleTrainer)，for single-cost single-optimizer single-data-source training. 

**内建的 `train_net.py` 脚本使用 [DefaultTrainer().train()](https://detectron2.readthedocs.io/modules/engine.html#detectron2.engine.defaults.DefaultTrainer)**, 包括了更多的可选的标准默认行为，例如learning rate schedule、日志记录、测试、checkpointing等。这也意味着更小的非默认行为的支持。

*class* `detectron2.engine.defaults.DefaultTrainer`(*cfg*)

> 具有默认训练逻辑的trainer。 与SimpleTrainer相比，它还包含以下逻辑：
>
> 1. 从给定的配置中创建模型，优化器，调度程序，数据加载器。
>
> 2. 当调用resume_or_load时，加载检查点或cfg.MODEL.WEIGHTS（如果存在）。
>
> 3. 注册一些常见的钩子。
>
> It is created to simplify the **standard model training workflow** and reduce code boilerplate for users who only need the standard training workflow, with standard features. It means this class makes *many assumptions* about your training logic that may easily become invalid in a new research. In fact, any assumptions beyond those made in the `SimpleTrainer` are too much for research.
>
> 此类代码已注释了其所作的限制性假设。 当它们对您不起作用时，建议您：
>
> 1. 覆盖此类的方法；
> 2. 使用SimpleTrainer，它仅进行最少的SGD培训，而没有其他操作。 然后，您可以根据需要添加自己的hook‘’
> 3. 编写类似于tools/plain_train_net.py的自己的训练循环。
>
> Examples：
>
> ```python
> trainer = DefaultTrainer(cfg)
> trainer.resume_or_load()  # load last checkpoint or MODEL.WEIGHTS
> trainer.train()
> ```
>
> > `resume_or_load`(*resume=True*)
> >
> > If resume==True, and last checkpoint exists, resume from it, load all checkpointables (eg. optimizer and scheduler) and update iteration counter.
> >
> > Otherwise, load the model specified by the config (skip all checkpointables) and start from the first iteration.
> >
> > `train`()
> >
> > Run training.
> >
> > - Returns
> >
> >   OrderedDict of results, if evaluation is enabled. Otherwise None.

**自定义训练循环**：

1. 如果您的自定义操作类似于`DefaultTrainer`所做的操作，则**可以通过覆盖子类中的方法来更改`DefaultTrainer`的行为**，如[tools/train_net.py](https://github.com/facebookresearch/detectron2/blob/master/tools/train_net.py)。
2. 如果需要一些非常新颖的东西，则可以从[tools/plain_train_net.py](https://github.com/facebookresearch/detectron2/blob/master/tools/plain_train_net.py)开始自己实现它们。

#### 指标记录

**当处于训练模式时，所有的模型都需要在一个`EventStorage`下使用。训练数据将会保存在一个storage：**

```python
from detectron2.utils.events import EventStorage
with EventStorage() as storage:
  losses = model(inputs)
```

在训练期间，指标被保存到一个集中式的[EventStorage](https://detectron2.readthedocs.io/modules/utils.html#detectron2.utils.events.EventStorage)，可以通过以下代码访问和记录：

```python
from detectron2.utils.events import get_event_storage

# inside the model:
if self.training:
  value = # compute the value from inputs  
  storage = get_event_storage()  # 创建storage
  storage.put_scalar("some_accuracy", value)  # 保存
```

然后使用[EventWriter](https://detectron2.readthedocs.io/modules/utils.html#module-detectron2.utils.events)将度量标准保存到各个目标。 DefaultTrainer启用一些具有默认配置的EventWriter。 有关如何自定义它们的信息，请参见上文。

### 测试

评估是一个汇总inputs/outputs的过程，可以通过调用模型手动的汇总评估，也可以使用detectron2中的**[DatasetEvaluator](https://detectron2.readthedocs.io/modules/evaluation.html#detectron2.evaluation.DatasetEvaluator)接口**。

Detectron2包含一些`DatasetEvaluator`，它使用标准的特定于数据集的API（例如COCO，LVIS）来计算指标。 您还可以实现自己的`DatasetEvaluator`，它使用inputs/outputs执行其他一些工作。 例如，要计算在验证集上检测到多少个实例：

```python
class Counter(DatasetEvaluator):
  def reset(self):
    self.count = 0
  def process(self, inputs, outputs):
    for output in outputs:
      self.count += len(output["instances"])
  def evaluate(self):
    # save self.count somewhere, or print it, or return it.
    return {"count": self.count}
```

**一旦有了一些`DatasetEvaluator`，就可以使用[inference_on_dataset](https://detectron2.readthedocs.io/modules/evaluation.html#detectron2.evaluation.inference_on_dataset)运行它。** 例如：

```python
val_results = inference_on_dataset(
    model,  # 模型
    val_data_loader, # 数据加载器（交叉验证集）
    DatasetEvaluators([COCOEvaluator(...), Counter()]))  # 多个Evaluator
```

与使用模型手动运行评估相比，使用[inference_on_dataset](https://detectron2.readthedocs.io/modules/evaluation.html#detectron2.evaluation.inference_on_dataset)的优势在于您可以使用[DatasetEvaluator](https://detectron2.readthedocs.io/modules/evaluation.html#detectron2.evaluation.DatasetEvaluator)将评估器合并在一起。 这样，可以一次性运行所有评估。

**`inference_on_dataset` 还为给定的模型和数据集提供准确的速度基准。**

#### 使用COCO API测试模型AP

```python
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
evaluator = COCOEvaluator("my_dataset_val", cfg, False, output_dir="./output/evaluaition/")
val_loader = build_detection_test_loader(cfg, "my_dataset_val")
inference_on_dataset(trainer.model, val_loader, evaluator)
# another equivalent way is to use trainer.test
```

### 推理预测predictor

**可以通过`outputs = model(inputs)`调用模型。**其中，`inputs`是`list[dict]`。每个字典对应一个图像，所需的keys取决于模型的类型以及模型是训练模式还是测试模式。 例如进行推理时，所有已有的模型都需要`image`参数，以及可选的`height`和`width`参数。

**如果仅想要使用一个已有的模型进行推理，则可以使用[DefaultPredictor](https://detectron2.readthedocs.io/modules/engine.html#detectron2.engine.defaults.DefaultPredictor)**，它是一个装饰器，仅提供基本的功能。包括：**模型加载**，**处理**，**单张图片操作**。

> `detectron2.engine.defaults.DefaultPredictor(cfg)`：
>
> 使用给定的配置创建一个简单的端到端预测器，该配置在单个设备上针对单个输入图像运行。
>
> 与直接使用模型相比，该类做了以下补充：
>
> 1. **==从 cfg.MODEL.WEIGHTS 加载checkpoint。==**
>
>    `cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")`
>
> 2. 始终采用BGR图像作为输入，并由 cfg.INPUT.FORMAT 应用格式转换
>
>    detectron2所有内置的模型的图像标准输入格式是(C, H, W)格式的`tensor`
>
> 3. 应用 cfg.INPUT.{MIN,MAX}_SIZE_TEST 进行resizing
>
> 4. 只能采用单张图像作为输入，而不是一个batch | 可以手动更改

#### 可视化Visualizer

通过`predictor`调用模型进行预测后，可通过 `Visualizer.draw_instance_predictions(predictions)` 对预测进行可视化。同时，`Visualizer`也可以用于数据可视化。

例子：

```python
from detectron2.utils.visualizer import ColorMode
dataset_dicts = get_balloon_dicts("balloon/val")
for d in random.sample(dataset_dicts, 3):
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=balloon_metadata, 
                   scale=0.8, 
                   instance_mode=ColorMode.IMAGE_BW   # 将未分割区域转化为灰度图像
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))  # 定义 Visualizer 之后需要手动的绘制预测实例
    plt.imshow(v.get_image())
```

*class* `detectron2.utils.visualizer.VisImage`(*img***,** *scale=1.0*)

> Parameters
>
> - **img** (*ndarray*) – an RGB image of shape (H, W, 3).
> - **scale** ([*float*](https://docs.python.org/3.6/library/functions.html#float)) – scale the input image
>
> > `save`(*filepath*)
> >
> > - Parameters
> >
> >   **filepath** ([*str*](https://docs.python.org/3.6/library/stdtypes.html#str)) – a string that contains the absolute path, including the file name, where the visualized image will be saved.
> >   
> >   **注意：默认注册的数据集的"file_name"包含（相对）路径，保存（预测）图像时要取路径字符串的最后。**
> >
> > `get_image`()[[source\]](https://detectron2.readthedocs.io/_modules/detectron2/utils/visualizer.html#VisImage.get_image)
> >
> > - Returns
> >
> >   *ndarray* – the visualized image of shape (H, W, 3) (RGB) in uint8 type. The shape is scaled w.r.t the input image using the given scale argument.

*class* `detectron2.utils.visualizer.Visualizer`(*img_rgb***,** *metadata***,** *scale=1.0***,** *instance_mode=*)

**只需要img和metadata就可以可视化**

> 参数：
>
> **img_rgb** – numpy数组(H, W, C)。图片必须为RGB格式，因为这是Matplotlib库的要求。 图像也应在[0，255]范围内。
>
> **metadata** ([*MetadataCatalog*](https://detectron2.readthedocs.io/modules/data.html#detectron2.data.MetadataCatalog)) – 图像元数据。
>
> > **类方法：**
> >
> > `draw_dataset_dict`(*dic*)
> >
> > > Draw annotations/segmentaions in Detectron2 Dataset format.
> > >
> > > Parameters:
> > >
> > > **dic** ([*dict*](https://docs.python.org/3.6/library/stdtypes.html#dict)) – annotation/segmentation data of one image, in Detectron2 Dataset format.
> > >
> > > Returns:
> > >
> > > *output (VisImage)* – image object with visualizations.
> >
> > `draw_instance_predictions`(*predictions*)
> >
> > > 在图像上绘制实例级预测结果。
> > >
> > > **predictions** ([*Instances*](https://detectron2.readthedocs.io/modules/structures.html#detectron2.structures.Instances)) – the output of an instance detection/segmentation model. Following fields will be used to draw: “pred_boxes”, “pred_classes”, “scores”, “pred_masks” (or “pred_masks_rle”).
> > >
> > > 输出：`detectron2.utils.visualizer.VisImage`(*img***,** *scale=1.0*)

*class* `detectron2.data.MetadataCatalog`

> MetadataCatalog提供对给定数据集的“元数据”的访问。
>
> The metadata associated with a certain name is a singleton: once created, the metadata will stay alive and will be returned by future calls to get(name).
>
> 就像全局变量一样，所以不要滥用它。 它用于存储在程序执行过程中恒定且共享的知识，例如：COCO中的类名。
>
> > *static* `get`**(***name*)
> >
> > Parameters
> >
> > **name** ([*str*](https://docs.python.org/3.6/library/stdtypes.html#str)) – name of a dataset (e.g. coco_2014_train).
> >
> > Returns
> >
> > *Metadata* – The `Metadata` instance associated with this name, or create an empty one if none is available.
> >
> > > metadata可以设置
> > >
> > > MetadataCatalog.get("balloon_" + d).set(thing_classes=["balloon"])

## configs

Detectron2提供了基于key-value的config系统，该系统可用于获取标准的常见行为。

**Detectron2的配置系统使用YAML和[yacs](https://github.com/rbgirshick/yacs)。** 除了访问和更新配置的基本操作外，我们还提供以下额外功能：

1. **config可以拥有 `_BASE_: base.yaml` 字段，该字段将首先加载基本配置。 如果存在任何冲突，则基本配置中的值将在子配置中被覆盖。** 我们为标准模型架构提供了一些基本配置。
2. 提供配置**版本控制**，以实现向后兼容。 如果您的配置文件使用诸如`VERSION: 2`这样的配置行进行了版本控制，即使将来我们更改某些keys，detectron2仍然可以识别该文件。

“ Config”是一个非常有限的抽象。 我们不希望Detectron2中的所有功能都可以通过配置使用。 如果您需要的配置空间不可用，请使用detectron2的API编写代码。

### 基本使用

这里显示`CfgNode`对象的一些基本用法。 请参阅[文档](https://detectron2.readthedocs.io/modules/config.html#detectron2.config.CfgNode)中的更多内容。

```python
from detectron2.config import get_cfg
cfg = get_cfg()    # 获取detectron2的默认config
cfg.xxx = yyy      # 为自定义的内容添加新的configs
cfg.merge_from_file("my_cfg.yaml")   # 从一个文件中加载values

cfg.merge_from_list(["MODEL.WEIGHTS", "weights.pth"])   # can also load values from a list of str
print(cfg.dump())  # 输出格式化的configs
```

*class* `detectron2.config.CfgNode`(*init_dict=None***,** *key_list=None***,** *new_allowed=False*)

> The same as fvcore.common.config.CfgNode, but different in:
>
> 1. Use unsafe yaml loading by default. Note that this may lead to arbitrary code execution: you must not load a config file from untrusted sources before manually inspecting the content of the file.
> 2. Support config versioning. When attempting to merge an old config, it will convert the old config automatically.
>
> **方法**：
>
> > `merge_from_file`(*cfg_filename: [str](https://docs.python.org/3.6/library/stdtypes.html#str)***,** *allow_unsafe: [bool](https://docs.python.org/3.6/library/functions.html#bool) = True*) **→ None**
> >
> > `get`()
> >
> > Return the value for key if key is in the dictionary, else default.

detectron2中的许多内置工具都接受命令行配置覆盖：命令行中提供的keys-values将覆盖配置文件中的现有values。 例如，[demo.py](https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py)可以这样使用：

```python
./demo.py --config-file config.yaml [--other-options] \
  --opts MODEL.WEIGHTS /path/to/weights INPUT.MIN_SIZE_TEST 1000
```

查看detectron2中可用的configs列表，参阅[Config References](https://detectron2.readthedocs.io/modules/config.html#config-references)。

### 练习使用configs

1. 将您编写的配置视为“代码”：避免复制； 使用`_BASE_`在不同的配置之间共享公用部分。
2. 简化配置：不要使用不会影响实验设置的keys。
3. **在配置（或基本配置）中保留版本号，例如`VERSION:2`，以实现向后兼容。** 读取没有版本号的配置时，会打印警告。 官方配置不包含版本号，因为它们始终是最新的。

## Model Zoo

在8个NVIDIA V100 GPU上进行训练。

- name列包含cfg配置文件，使用`tools/train_net.py`可以重现结果。
- 推理速度使用`tools/train_net.py --eval-only`或`detectron2.evaluation.``inference_on_dataset`**(***model***,** *data_loader***,** *evaluator*)进行测试，测试时batch大小为1。
- Training curves and other statistics can be found in `metrics` for each model.

### coco训练细节

- 默认的数据扩增使用了水平翻转和尺度抖动scale jittering。
- 对于Mask R-CNN，提供了三个不同的backbone：
  - FPN：ResNet+FPN backbone，对于mask预测，使用标准的卷积头部，对于边框预测，使用标准的全连接FC头部。这提供了最佳的速度和精度均衡，但其他两个更具有研究意义。
  - **C4**: Use a ResNet conv4 backbone with conv5 head. The original baseline in the Faster R-CNN paper.
  - **DC5** (Dilated-C5): Use a ResNet conv5 backbone with dilations in conv5, and standard conv and FC heads for mask and box prediction, respectively. This is used by the Deformable ConvNet paper.
- Most models are trained with the 3x schedule (~37 COCO epochs). Although 1x models are heavily under-trained, we provide some ResNet-50 models with the 1x (~12 COCO epochs) training schedule for comparison when doing quick research iteration.

### ImageNet Pretrained Models训练细节

提供了在ImageNet-1k数据集上的预训练的模型。

- [R-50.pkl](https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/MSRA/R-50.pkl): converted copy of [MSRA's original ResNet-50](https://github.com/KaimingHe/deep-residual-networks) model.
- [R-101.pkl](https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/MSRA/R-101.pkl): converted copy of [MSRA's original ResNet-101](https://github.com/KaimingHe/deep-residual-networks) model.
- [X-101-32x8d.pkl](https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/FAIR/X-101-32x8d.pkl): ResNeXt-101-32x8d model trained with Caffe2 at FB.

通过这个脚本 [script](https://github.com/facebookresearch/detectron2/blob/master/tools/convert-torchvision-to-d2.py)的转换，可以使用Torchvision中的ResNet模型.



## 一些细节

### 实施细节

**边框bbox**

1. 边框坐标的范围是[0, width]或者[0, height]。

2. 边框回归的损失函数默认是L1 loss而非smooth L1 损失。

3. The height and width of a box with corners (x1, y1) and (x2, y2) is now computed more naturally as width = x2 - x1 and height = y2 - y1; In Detectron, a "+ 1" was added both height and width.

   The change in height/width calculations most notably changes:

   - encoding/decoding in bounding box regression.
   - non-maximum suppression. The effect here is very negligible, though.

**Anchors**

- RPN now uses simpler anchors with fewer quantization artifacts.

  In Detectron, the anchors were quantized and [do not have accurate areas](https://github.com/facebookresearch/Detectron/issues/227). In Detectron2, the anchors are center-aligned to feature grid points and not quantized.

**分类id的编写方式**

id范围为[0, K-1]，K为类别数，第K类代表背景类。

### model_zoo的实验细节

- 在训练中使用了 scale augmentation，训练开销更低，同时提升了AP。
- 使用了L1损失而非smooth L1损失，有时会提升目标检测的AP单会影响其他AP
- 使用了`ROIAlignV2`，没有显著的影响AP







