# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import inspect
import numpy as np
import pprint
from typing import Any, List, Optional, Tuple, Union
from fvcore.transforms.transform import Transform, TransformList

"""
Overview of the augmentation system:

We have a design goal that aims at allowing:
    (1) Arbitrary structures of input data (e.g. list[list[boxes]], dict[str, boxes], multiple semantic segmentations for each image, etc) and arbitrary new data types (rotated boxes, 3D meshes, densepose, etc)
    (2) A list of augmentation to be applied sequentially

`Augmentation` defines policies to create a `Transform` object from input data by `get_transform` method.
A `Transform` object usually describes deterministic transformation, in the sense that it can be re-applied on associated data, e.g. the geometry of an image and its segmentation masks need to be transformed in the same way, instead of both being randomly augmented in inconsistent ways.
(If you're sure such re-application is not needed, then determinism is not a crucial requirement.)
An augmentation policy may need to access arbitrary input data to create a `Transform`, so it declares the needed input data by its `input_args` attribute. Users are expected to provide them when calling its `get_transform` method.

`Augmentation` is not able to apply transforms to data: data associated with one sample may be much more than what `Augmentation` gets. For example, >90% of the common augmentation policies only need an image, but the actual input samples can be much more complicated.

`AugInput` manages all inputs needed by `Augmentation` and implements the logic to apply a sequence of augmentation. It defines how the inputs should be modified by a `Transform`, because inputs needed by one `Augmentation` needs to be transformed to become arguments of the next `Augmentation` in the sequence.

`AugInput` does not need to contain all input data, because most augmentation policies only need very few fields (e.g., >90% only need "image"). We provide `StandardAugInput` that only contains "images", "boxes", "sem_seg", that are enough to create transforms for most cases. In this way, users keep the responsibility and flexibility to apply transforms to other (potentially new) data types and structures, e.g. keypoints, proposals boxes.

To extend the system, one can do:
1. To add a new augmentation policy that only needs to use standard inputs ("image", "boxes", "sem_seg"), writing a subclass of `Augmentation` is sufficient.
2. To use new data types or custom data structures, `StandardAugInput` can still be used as long as the new data types or custom data structures are not needed by any augmentation policy.
The new data types or data structures can be transformed using the transforms returned by `AugInput.apply_augmentations`.The way new data types are transformed may need to declared using `Transform.register_type`.
3. (rare) To add new augmentation policies that need new data types or data structures, in addition to implementing new `Augmentation`, a new `AugInput` is needed as well.

增强系统的概述。

我们的设计目标是允许:
    (1)输入数据的任意结构(如list[list[box]]，dict[str，box]，每幅图像的多重语义分割等)和任意新的数据类型(旋转的盒子，3D网格，densepose等)
    (2)依次适用的增量清单。

"增强 "定义了通过 "get_transform "方法从输入数据中创建 "Transform "对象的策略。
`Transform`对象通常描述的是确定性变换，即可以对相关数据进行重新应用，例如图像的几何体及其分割掩模需要以相同的方式进行变换，而不是两者以不一致的方式随机增强。
(如果你确定不需要这种重新应用，那么确定性就不是一个关键的要求)。
一个增强策略可能需要访问任意的输入数据来创建一个`Transform`，所以它通过其`input_args`属性来声明所需的输入数据。用户应该在调用它的`get_transform`方法时提供这些数据。

`Augmentation`无法对数据应用变换：与一个样本相关的数据可能比`Augmentation`得到的数据多得多。例如，>90%的常见增强策略只需要一个图像，但实际输入的样本可能要复杂得多。

`AugInput`管理`Augmentation`所需要的所有输入，并实现应用增强序列的逻辑。它定义了如何通过`Transform`修改输入，因为一个`Augmentation`所需要的输入需要被转换，以成为序列中下一个`Augmentation`的参数。

`AugInput`不需要包含所有的输入数据，因为大多数增强策略只需要很少的字段（例如，>90%只需要 "图像"）。我们提供的`StandardAugInput`只包含 "image"、"box"、"sem_seg"，这些数据足以为大多数情况创建变换。通过这种方式，用户保留了将变换应用于其他（潜在的新）数据类型和结构的责任和灵活性，例如，关键点、建议框。

要扩展系统，可以这样做。
1. 要添加一个新的增强策略，只需要使用标准输入（"图像"、"框"、"sem_seg"），写一个`增强`的子类即可。
2. 要使用新的数据类型或自定义的数据结构，只要新的数据类型或自定义的数据结构不是任何增强策略所需要的，仍然可以使用`StandardAugInput`。
新的数据类型或数据结构可以使用`AugInput.apply_augmentations`返回的transform进行转换，新数据类型的转换方式可能需要使用`Transform.register_type`进行声明。
3. (罕见)如果要增加新的增强策略，需要新的数据类型或数据结构，除了实现新的`Augmentation`外，还需要新建一个`AugInput`。
"""


__all__ = [
    "Augmentation",
    "AugmentationList",
    "TransformGen",
    "apply_transform_gens",
    "AugInput",
    "StandardAugInput",
    "apply_augmentations",
]


def _check_img_dtype(img):
    assert isinstance(img, np.ndarray), "[Augmentation] Needs an numpy array, but got a {}!".format(
        type(img)
    )
    assert not isinstance(img.dtype, np.integer) or (
        img.dtype == np.uint8
    ), "[Augmentation] Got image of type {}, use uint8 or floating points instead!".format(
        img.dtype
    )
    assert img.ndim in [2, 3], img.ndim


def _get_aug_input_args(aug, aug_input) -> List[Any]:
    """
    Get the arguments to be passed to ``aug.get_transform`` from the input ``aug_input``.
    """
    args = []
    for f in aug.input_args:
        try:
            args.append(getattr(aug_input, f))
        except AttributeError:
            raise AttributeError(
                f"Augmentation {aug} needs '{f}', which is not an attribute of {aug_input}!"
            )
    return args


class Augmentation:
    """
    Augmentation定义了从输入数据产生`Transform`的策略
    通常用于输入数据的预处理。策略通常包含随机性，但也可以选择确定性地生成'Transform'。

    一般情况下，生成Transform的“策略”可能需要输入数据的任意信息，以定义要应用的变换。
    因此，每一个'Augmentation'实例使用`input_args`属性，定义了`get_transform`所需要的参数。

    注意：Augmentation定义了创建Transform的策略，但是没有定义如何对这些数据应用实际的转换。

    Augmentation defines policies/strategies to generate :class:`Transform` from data.
    It is often used for pre-processing of input data. A policy typically contains randomness, but it can also choose to deterministically generate a :class:`Transform`.

    A "policy" that generates a :class:`Transform` may, in the most general case, need arbitrary information from input data in order to determine what transforms to apply. Therefore, each :class:`Augmentation` instance defines the arguments needed by its :meth:`get_transform` method with the :attr:`input_args` attribute.
    When called with the positional arguments defined by the :attr:`input_args`, the :meth:`get_transform` method executes the policy.

    Examples:
    ::
        # if a policy needs to know both image and semantic segmentation
        assert aug.input_args == ("image", "sem_seg")
        tfm: Transform = aug.get_transform(image, sem_seg)
        new_image = tfm.apply_image(image)

    To implement a custom :class:`Augmentation`, define its :attr:`input_args` and implement :meth:`get_transform`.

    Note that :class:`Augmentation` defines the policies to create a :class:`Transform`, but not how to apply the actual transform to those data.
    """

    input_args: Tuple[str] = ("image",)
    """
    Attribute of class instances that defines the argument(s) needed by
    :meth:`get_transform`. Default to only "image", because most policies only
    require knowing the image in order to determine the transform.

    Users can freely define arbitrary new args and their types in custom
    :class:`Augmentation`. In detectron2 we use the following convention:

    * image: (H,W) or (H,W,C) ndarray of type uint8 in range [0, 255], or
      floating point in range [0, 1] or [0, 255].
    * boxes: (N,4) ndarray of float32. It represents the instance bounding boxes
      of N instances. Each is in XYXY format in unit of absolute coordinates.
    * sem_seg: (H,W) ndarray of type uint8. Each element is an integer label of pixel.

    We do not specify convention for other types and do not include builtin
    :class:`Augmentation` that uses other types in detectron2.
    """

    def _init(self, params=None):
        if params:
            for k, v in params.items():
                if k != "self" and not k.startswith("_"):
                    setattr(self, k, v)

    def get_transform(self, *args) -> Transform:
        """
        Execute the policy based on input data, and decide what transform to apply to inputs.

        Args:
            arguments must follow what's defined in :attr:`input_args`.

        Returns:
            Transform: Returns the deterministic transform to apply to the input.
        """
        raise NotImplementedError

    def __call__(self, aug_input) -> Transform:
        """
        以输入数据为参数，返回要使用的transform
        在调用时，将应用augmentation。在大多数augmentation中，使用默认操作就行，该默认操作调用`get_transform`用于输入数据。也可以继承实现更加复杂的逻辑

        Augment the given `aug_input` **in-place**, and return the transform that's used.

        This method will be called to apply the augmentation. In most augmentation, it is enough to use the default implementation, which calls :meth:`get_transform` on the inputs. But a subclass can overwrite it to have more complicated logic.

        Args:
            aug_input (AugInput):
                an object that has attributes needed by this augmentation (defined by ``self.input_args``). Its ``transform`` method will be called to in-place transform it.
        Returns:
            Transform:
                the transform that is applied on the input.
        """
        args = _get_aug_input_args(self, aug_input)  # 获得aug_input的参数并使用该参数创建tfm
        tfm = self.get_transform(*args)  # 消除随机性后产生的确定变换
        assert isinstance(tfm, (Transform, TransformList)), (
            f"{type(self)}.get_transform must return an instance of Transform!"
            "Got {type(tfm)} instead."
        )
        aug_input.transform(tfm)  # 对Aug_input中的数据（img、bbox、seg）等进行变换
        return tfm  # 返回应用的确定的tfm，以便应用于其他数据

    def _rand_range(self, low=1.0, high=None, size=None):
        """
        Uniform float random number between low and high.
        """
        if high is None:
            low, high = 0, low
        if size is None:
            size = []
        return np.random.uniform(low, high, size)

    def __repr__(self):
        """
        Produce something like:
        "MyAugmentation(field1={self.field1}, field2={self.field2})"
        """
        try:
            sig = inspect.signature(self.__init__)
            classname = type(self).__name__
            argstr = []
            for name, param in sig.parameters.items():
                assert (
                    param.kind != param.VAR_POSITIONAL and param.kind != param.VAR_KEYWORD
                ), "The default __repr__ doesn't support *args or **kwargs"
                assert hasattr(self, name), (
                    "Attribute {} not found! "
                    "Default __repr__ only works if attributes match the constructor.".format(name)
                )
                attr = getattr(self, name)
                default = param.default
                if default is attr:
                    continue
                attr_str = pprint.pformat(attr)
                if "\n" in attr_str:
                    # don't show it if pformat decides to use >1 lines
                    attr_str = "..."
                argstr.append("{}={}".format(name, attr_str))
            return "{}({})".format(classname, ", ".join(argstr))
        except AssertionError:
            return super().__repr__()

    __str__ = __repr__


def _transform_to_aug(tfm_or_aug):
    """
    Wrap Transform into Augmentation.
    Private, used internally to implement augmentations.
    """
    assert isinstance(tfm_or_aug, (Transform, Augmentation)), tfm_or_aug
    if isinstance(tfm_or_aug, Augmentation):
        return tfm_or_aug
    else:

        class _TransformToAug(Augmentation):
            def __init__(self, tfm: Transform):
                self.tfm = tfm

            def get_transform(self, *args):
                return self.tfm

            def __repr__(self):
                return repr(self.tfm)

            __str__ = __repr__

        return _TransformToAug(tfm_or_aug)


class AugmentationList(Augmentation):
    """
    Apply a sequence of augmentations.
    """

    def __init__(self, augs):
        """
        Args:
            augs (list[Augmentation or Transform]):
        """
        super().__init__()
        self.augs = [_transform_to_aug(x) for x in augs]

    def __call__(self, aug_input) -> Transform:
        tfms = []
        for x in self.augs:
            tfm = x(aug_input)
            tfms.append(tfm)
        return TransformList(tfms)

    def __repr__(self):
        msgs = [str(x) for x in self.augs]
        return "AugmentationList[{}]".format(", ".join(msgs))

    __str__ = __repr__


class AugInput:
    """
    A base class for anything on which a list of :class:`Augmentation` can be applied.
    This class provides input arguments for :class:`Augmentation` to use, and defines how
    to apply transforms to these data.

    An instance of this class must satisfy the following:

    * :class:`Augmentation` declares some data it needs as arguments. A :class:`AugInput`
      must provide access to these data in the form of attribute access (``getattr``).
      For example, if a :class:`Augmentation` to be applied needs "image" and "sem_seg"
      arguments, this class must have the attribute "image" and "sem_seg" whose content
      is as required by the :class:`Augmentation`s.
    * This class must have a ``transform(tfm: Transform) -> None`` method which
      in-place transforms all attributes stored in the class.
    """

    def transform(self, tfm: Transform) -> None:
        raise NotImplementedError

    def apply_augmentations(
        self, augmentations: List[Union[Augmentation, Transform]]
    ) -> TransformList:
        """
        Apply a list of Transform/Augmentation in-place and returned the applied transform.
        Attributes of this class will be modified.

        Returns:
            TransformList:
                returns transformed inputs and the list of transforms applied.
                The TransformList can then be applied to other data associated with the inputs.
        """
        return AugmentationList(augmentations)(self)


class StandardAugInput(AugInput):
    """
    A standard implementation of :class:`AugInput` for the majority of use cases.
    This class provides the following standard attributes that are common to use by Augmentation (augmentation policies). These are chosen because most :class:`Augmentation` won't need anything more to define a augmentation policy.
    After applying augmentations to these special attributes, the returned transforms can then be used to transform other data structures that users have.

    Attributes:
        image (ndarray):
            * image in HW or HWC format. The meaning of C is up to users
        boxes (ndarray or None):
            * Nx4 boxes in XYXY_ABS mode
        sem_seg (ndarray or None):
            HxW semantic segmentation mask

    Examples:
    ::
        input = StandardAugInput(image, boxes=boxes)
        ? tfms = input.apply_augmentations(list_of_augmentations)
        transformed_image = input.image
        transformed_boxes = input.boxes
        transformed_other_data = tfms.apply_other(other_data)

    An extended project that works with new data types may require augmentation policies that need more inputs. An algorithm may need to transform inputs in a way different from the standard approach defined in this class. In those situations, users can implement new subclasses of :class:`AugInput` with different attributes and the :meth:`transform` method.
    """

    # TODO maybe should support more builtin data types here
    def __init__(
        self,
        image: np.ndarray,
        *,
        boxes: Optional[np.ndarray] = None,
        sem_seg: Optional[np.ndarray] = None,
    ):
        """
        Args:
            image:
                (H,W) or (H,W,C) ndarray of type uint8 in range [0, 255], or floating point in range [0, 1] or [0, 255].
            boxes:
                (N,4) ndarray of float32. It represents the instance bounding boxes of N instances. Each is in XYXY format in unit of absolute coordinates.
            sem_seg:
                (H,W) ndarray of type uint8. Each element is an integer label of pixel.
        """
        _check_img_dtype(image)
        self.image = image
        self.boxes = boxes
        self.sem_seg = sem_seg

    def transform(self, tfm: Transform) -> None:
        """
        In-place transform all attributes of this class.
        """
        self.image = tfm.apply_image(self.image)
        if self.boxes is not None:
            self.boxes = tfm.apply_box(self.boxes)
        if self.sem_seg is not None:
            self.sem_seg = tfm.apply_segmentation(self.sem_seg)


def apply_augmentations(augmentations: List[Union[Transform, Augmentation]], inputs):
    """
    Use :meth:`AugInput.apply_augmentations` instead.
    """
    if isinstance(inputs, np.ndarray):
        # handle the common case of image-only Augmentation, also for backward compatibility
        image_only = True
        inputs = StandardAugInput(inputs)
    else:
        image_only = False
    tfms = inputs.apply_augmentations(augmentations)
    return inputs.image if image_only else inputs, tfms


apply_transform_gens = apply_augmentations
"""
Alias for backward-compatibility.
"""

TransformGen = Augmentation
"""
Alias for Augmentation, since it is something that generates :class:`Transform`s
"""
