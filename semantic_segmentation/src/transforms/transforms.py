#  Copyright (c) 2021 PPViT Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
import numpy as np
import cv2
from PIL import Image
from paddle.vision.transforms import functional as F
from src.transforms import functional
from .color_label import color2label


class Compose:
    """
    Do transformation on input data with corresponding pre-processing and 
    augmentation operations. The shape of input data to all operations is 
    [height, width, channels].

    Args:
        transforms (list): A list contains data pre-processing or augmentation. 
        Empty list means only reading images, no transformation.
        to_rgb (bool, optional): If converting image to RGB color space.
        Default: True.

    Raises:
        TypeError: When 'transforms' is not a list.
        ValueError: when the length of 'transforms' is less than 1.
    """

    def __init__(self, transforms, to_rgb=True):
        if not isinstance(transforms, list):
            raise TypeError('The transforms must be a list!')
        self.transforms = transforms
        self.to_rgb = to_rgb  # 记录是否需要将图片转换为RGB

    def __call__(self, img, label=None):
        """
        Args:
            img (str|np.ndarray): It is either image path or image object.
            label (str|np.ndarray): It is either label path or label ndarray.

        Returns:
            (tuple). A tuple including image and label after transformation.
        """
        if isinstance(img, str):
            img = cv2.imread(img).astype('float32')  # 读取方式是bgr
        if isinstance(label, str):
            # 模式“P”为8位彩色图像，它的每个像素用8个bit表示，其对应的彩色值是按照调色板查询出来的 Image.open()默认打开的图片为RGB模式
            # Image.open返回的图片类型为PILImage, 数值类型为uint8，值为0 - 255，尺寸为W * H * C（宽度高度通道数）。通过img = np.array(img)
            # 转为numpy数组后，统一尺寸为H * W * C。

            # 通过pillow打开标签文件，这里使用的pillow原因是因为标注文件有可能是伪彩色标注，使用调色板模式，通过pillow打开
            # 则可以直接获取标注文件每一个像素点值为调色板中的索引，这样就可以直接定义为类别号。这样同时兼容灰度标注与伪彩色标注。
            label = np.asarray(Image.open(label).convert('P'), dtype=np.uint8) #这一句是原始提供的代码
            # print("label_size {},label {}".format(label.shape,label)) #512 512
            # 以下是自定义
            # label = Image.open(label).convert('RGB')  # 标签是彩色图，转一下避免不必要的一些错误
            # label = color2label(label, dataset='ISPRS')  # 这是自己定义的将彩色的label转换为类别标签函数 已经将3通道转换为了单通道的nparry类别
            # label = np.asarray(label, dtype=np.uint8)
            # # label = cv2.cvtColor(np.asarray(label), cv2.COLOR_RGB2BGR)  # 转换成opencv格式在执行相应操作

        if img is None:
            raise ValueError('Can\'t read The image file {}!'.format(img))
        if self.to_rgb:  # 因为opencv打开的图片，像素点排序默认是BGR，这里如果需要可以转换成RGB。
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)

        for op in self.transforms:  # 遍历transforms列表，执行数据预处理与增强。
            outputs = op(img, label)
            img = outputs[0]
            if len(outputs) == 2:
                label = outputs[1]

        # 这里将图像数据的矩阵进行转置，将通道放在高度和宽度之前。 CV2读取的是H,W,C ,PIL读取的是W,H,C
        # 比如一张图片为高度为480，宽度为640，通道数为3代表RGB图像。它的矩阵形状为[480, 640, 3]
        # 经过下面代码的转置操作则变为[3, 480, 640]
        img = np.transpose(img, (2, 0, 1))  # 因为使用cv2读取格式为numpy,而numpy中则是[H, W, Channels], Pytorch中为[Channels, H, W]
        return (img, label)


class RandomHorizontalFlip:
    """
    Flip an image horizontally with a certain probability. 以一定的概率水平翻转一个图像。

    Args:
        prob (float, optional): A probability of horizontally flipping. 水平翻转的概率
    """

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, label=None):
        if random.random() < self.prob:
            img = functional.horizontal_flip(img)
            if label is not None:
                label = functional.horizontal_flip(label)
        if label is None:
            return (img,)
        else:
            return (img, label)


class RandomVerticalFlip:
    """
    Flip an image vertically with a certain probability. 以一定的概率垂直翻转一个图像。

    Args:
        prob (float, optional): A probability of vertical flipping. Default: 0.1.
    """

    def __init__(self, prob=0.1):
        self.prob = prob

    def __call__(self, img, label=None):
        if random.random() < self.prob:
            img = functional.vertical_flip(img)
            if label is not None:
                label = functional.vertical_flip(label)
        if label is None:
            return (img,)
        else:
            return (img, label)


# 根据输入的尺寸调整图片的大小
class Resize:
    """
    Resize an image. If size is a sequence like (h, w), output size will be 
    matched to this. If size is an int, smaller edge of the image will be 
    matched to this number. i.e, if height > width, then image will be 
    rescaled to (size * height / width, size).

    Args:
        target_size (list|tuple|int, optional): The target size of image.
        interp (str, optional): The interpolation mode of resize is consistent  调整大小的插值模式是一致的
        with opencv. ['NEAREST', 'LINEAR', 'CUBIC', 'AREA', 'LANCZOS4', 'RANDOM']. 
        Note that when it is 'RANDOM', a random interpolation mode would be specified.   注意，当它是'RANDOM'时，将指定一个随机插值模式。

    Raises:
        TypeError: When 'target_size' type is neither list nor tuple.
        ValueError: When "interp" is out of pre-defined methods ('NEAREST', 
        'LINEAR', 'CUBIC', 'AREA', 'LANCZOS4', 'RANDOM').
    """

    def __init__(self, target_size=520, interp='LINEAR', keep_ori_size=False):
        self.interp = interp
        self.keep_ori_size = keep_ori_size

        if isinstance(target_size, int):
            assert target_size > 0
        elif isinstance(target_size, list) or isinstance(target_size, tuple):
            if len(target_size) != 2:
                raise ValueError("`target_size` should include 2 elements, "
                                 "but it is {}".format(target_size))
        else:
            raise TypeError(
                "Type of `target_size` is invalid. It should be list or tuple, "
                "but it is {}".format(type(target_size)))
        self.target_size = target_size

    def __call__(self, img, label=None):
        """
        Args:
            img (np.ndarray): The Image data.
            label (np.ndarray, optional): The label data. Default: None.

        Returns:
            (tuple). When label is None, it returns (im, ), otherwise it 
            returns (im, label),

        Raises:
            TypeError: When the 'img' type is not numpy.
            ValueError: When the length of "im" shape is not 3.
        """
        # 需要保证图像的类型为ndarray，通过Opencv读取的默认是该类型，如果是标签图片通过PIL读取
        # 则需要通过asarray等方法转换。
        if not isinstance(img, np.ndarray):
            raise TypeError("Resize: image type is not numpy.")
        if len(img.shape) != 3:  # 图片需要是3阶矩阵，标签图片需要新建一个维度。
            raise ValueError('Resize: image is not 3-dimensional.')
        if self.interp == "RANDOM":
            interp = random.choice(list(self.interp_dict.keys()))
        else:
            interp = self.interp
        if not self.keep_ori_size: #keep_ori_size False值则转换图片
            img = F.resize(img, self.target_size, 'bilinear')
        # 如果传入了标签图片数据，也需要进行缩放，这里注意的是标签图片数据只能使用INTER_NEAREST方法，否则
        # 会影响标签数据的准确性。
        if label is not None:
            label = F.resize(label, self.target_size, 'nearest')

        if label is None:
            return (img,)
        else:
            return (img, label)


# 随机缩放图片
class ResizeStepScaling:
    """
    Scale an image proportionally within a range. 在一个范围内按比例缩放一个图像。

    Args:
        min_scale_factor (float, optional): The minimum scale. Default: 0.75.
        max_scale_factor (float, optional): The maximum scale. Default: 1.25.
        scale_step_size (float, optional): The scale interval. Default: 0.25. 缩放区间

    Raises:
        ValueError: When min_scale_factor is smaller than max_scale_factor.
    """

    def __init__(self,
                 min_scale_factor=0.75,
                 max_scale_factor=1.25,
                 scale_step_size=0.25):
        if min_scale_factor > max_scale_factor:
            raise ValueError(
                "min_scale_factor must be less than max_scale_factor, "
                "but they are {} and {}.".format(min_scale_factor,
                                                 max_scale_factor))
        self.min_scale_factor = min_scale_factor
        self.max_scale_factor = max_scale_factor
        self.scale_step_size = scale_step_size

    def __call__(self, img, label=None):
        """
        Args:
            img (np.ndarray): The Image data.
            label (np.ndarray, optional): The label data. Default: None.

        Returns:
            (tuple). When label is None, it returns (img, ), otherwise it 
            returns (img, label).
        """

        if self.min_scale_factor == self.max_scale_factor:
            scale_factor = self.min_scale_factor

        elif self.scale_step_size == 0:
            scale_factor = np.random.uniform(self.min_scale_factor,
                                             self.max_scale_factor)

        else:
            # option 1  #np.random.random_sample() 随机给出设定的size尺寸的位于[0,1)半开半闭区间上的随机数
            # scale_factor = np.random.random_sample() * (self.max_scale_factor
            #     - self.min_scale_factor) + self.min_scale_factor
            # option 2 #在区间[min,max]之之前，以步长为step生成相应可能的值，在随机选择第一个作为缩放的比例
            num_steps = int((self.max_scale_factor - self.min_scale_factor) / self.scale_step_size + 1)  # 生成多少个可能的值
            scale_factors = np.linspace(self.min_scale_factor, self.max_scale_factor, num_steps).tolist()
            np.random.shuffle(scale_factors)
            scale_factor = scale_factors[0]
        w = int(round(scale_factor * img.shape[1]))
        h = int(round(scale_factor * img.shape[0]))
        img = F.resize(img, (w, h), 'bilinear')
        if label is not None:
            label = F.resize(label, (w, h), 'nearest')
        if label is None:
            return (img,)
        else:
            return (img, label)


class Normalize:
    """
    Normalize an image.

    Args:
        mean (list, optional): The mean value of a dataset. Default: 
        [0.5, 0.5, 0.5].
        std (list, optional): The standard deviation of a dataset. Default: 
        [0.5, 0.5, 0.5].

    Raises:
        ValueError: When mean/std is not list or any value in std is 0.
    """

    def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        self.mean = mean
        self.std = std
        if not (isinstance(self.mean, (list, tuple))
                and isinstance(self.std, (list, tuple))):
            raise ValueError("{}: input type is invalid. It should be list or "
                             "tuple".format(self))
        from functools import reduce
        if reduce(lambda x, y: x * y, self.std) == 0:
            raise ValueError('{}: std is invalid!'.format(self))

    def __call__(self, img, label=None):
        """
        Args:
            img (np.ndarray): The Image data.
            label (np.ndarray, optional): The label data. Default: None.

        Returns:
            (tuple). When label is None, it returns (img, ), otherwise it 
            returns (im, label).
        """
        # 对均值和标准差的维度进行变换，方便与图形进行计算，变换后的维度为[1,1,3]
        mean = np.array(self.mean).reshape(1, -1)
        std = np.array(self.std).reshape(1, -1)
        # option 1
        # img = functional.normalize(img, mean, std)
        # option 2
        img = functional.imnormalize(img, mean, std) #对图像进行标准化处理。
        if label is None:
            return (img,)
        else:
            return (img, label)


class Padding:
    """
    Add bottom-right padding to a raw image or annotation image. 为原始图像或标签图像添加右角-下角的填充。

    Args:
        target_size (list|tuple): The target size after padding.
        im_padding_value (list, optional): The padding value of raw image.
        Default: [127.5, 127.5, 127.5].
        label_padding_value (int, optional): The padding value of annotation
        image. Default: 255.

    Raises:
        TypeError: When target_size is neither list nor tuple.
        ValueError: When the length of target_size is not 2.
    """

    def __init__(self,
                 target_size,
                 im_padding_value=(127.5, 127.5, 127.5),
                 label_padding_value=255):
        if isinstance(target_size, list) or isinstance(target_size, tuple):
            if len(target_size) != 2:
                raise ValueError(
                    "`target_size` should include 2 elements, but it is {}".
                        format(target_size))
        else:
            raise TypeError("Type of target_size is invalid. It should be list "
                            "or tuple, now is {}".format(type(target_size)))
        self.target_size = target_size
        self.im_padding_value = im_padding_value
        self.label_padding_value = label_padding_value

    def __call__(self, img, label=None):
        """
        Args:
            img (np.ndarray): The Image data.
            label (np.ndarray, optional): The label data. Default: None.

        Returns:
            (tuple): When label is None, it returns (img, ), otherwise it 
            returns (img, label).
        """

        img_height, img_width = img.shape[0], img.shape[1]
        if isinstance(self.target_size, int):
            target_height = self.target_size
            target_width = self.target_size
        else:
            target_height = self.target_size[1]
            target_width = self.target_size[0]
        pad_height = target_height - img_height
        pad_width = target_width - img_width
        if pad_height < 0 or pad_width < 0:
            raise ValueError("The size of image should be less than `target_size`, "
                             "but the size of image ({}, {}) is larger than `target_size` "
                             "({}, {})".format(img_width, img_height, target_width, target_height))
        else:
            img = cv2.copyMakeBorder(
                img, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT,
                value=self.im_padding_value)
            if label is not None:
                label = cv2.copyMakeBorder(
                    label, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT,
                    value=self.label_padding_value)
        if label is None:
            return (img,)
        else:
            return (img, label)


class RandomPaddingCrop:
    """
    Crop a sub-image from a raw image and annotation image randomly. If the 
    target cropping siz is larger than original image, then the bottom-right 
    padding will be added.
    从原始图像和注释图像中随机裁剪一个子图像。如果  目标裁剪尺寸大于原始图像，那么下-右角的填充将被添加。
    Args:
        crop_size (tuple, optional): The target cropping size.
        img_padding_value (list, optional): The padding value of raw image.
        Default: (123.675, 116.28, 103.53).
        label_padding_value (int, optional): The padding value of annotation
        image. Default: 255.

    Raises:
        TypeError: When crop_size is neither list nor tuple.
        ValueError: When the length of crop_size is not 2.
    """

    def __init__(self,
                 crop_size=(512, 512),
                 img_padding_value=(123.675, 116.28, 103.53),
                 label_padding_value=255):
        if isinstance(crop_size, list) or isinstance(crop_size, tuple):
            if len(crop_size) != 2:
                raise ValueError("Type of `crop_size` is list or tuple. It "
                                 "should include 2 elements, but it is {}"
                                 .format(crop_size))
        else:
            raise TypeError("The type of `crop_size` is invalid. It should "
                            "be list or tuple, but it is {}"
                            .format(type(crop_size)))
        self.crop_size = crop_size
        self.img_padding_value = img_padding_value
        self.label_padding_value = label_padding_value

    def __call__(self, img, label=None):
        """
        Args:
            img (np.ndarray): The Image data.
            label (np.ndarray, optional): The label data. Default: None.

        Returns:
            (tuple): When label is None, it returns (img, ), otherwise it 
            returns (img, label).
        """
        # 如果传入的crop_size为整型，则需要裁减宽高都为crop_size。
        if isinstance(self.crop_size, int):
            crop_width = self.crop_size
            crop_height = self.crop_size
        # 如果传入的crop_size为整型，则需要裁减宽高都为crop_size。
        else:
            crop_width = self.crop_size[0]
            crop_height = self.crop_size[1]

        img_height = img.shape[0]
        img_width = img.shape[1]
        # 如果图像原始宽高与需要裁减的宽高一致，则直接返回图像，不做任何处理。
        if img_height == crop_height and img_width == crop_width:
            if label is None:
                return (img,)
            else:
                return (img, label)
        else:
            pad_height = max(crop_height - img_height, 0)  # 计算高和宽分别需要填充的长度。
            pad_width = max(crop_width - img_width, 0)
            if (pad_height > 0 or pad_width > 0):  # 如果裁减尺寸大于图像尺寸，则对图像进行填充扩展。
                img = cv2.copyMakeBorder(
                    img, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT,
                    value=self.img_padding_value)
                if label is not None:  # 同样对应的标签图片也需要填充。
                    label = cv2.copyMakeBorder(
                        label, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT,
                        value=self.label_padding_value)
                img_height = img.shape[0]
                img_width = img.shape[1]

            # 如果需要裁剪的尺寸大于0，则在img_height和crop_height的差值之间随机一个整数，作为高度裁剪的起点，
            if crop_height > 0 and crop_width > 0:
                h_off = np.random.randint(img_height - crop_height + 1)
                w_off = np.random.randint(img_width - crop_width + 1)

                img = img[h_off:(crop_height + h_off), w_off:(w_off + crop_width), :]
                if label is not None:
                    label = label[h_off:(crop_height + h_off), w_off:(w_off + crop_width)]
        if label is None:
            return (img,)
        else:
            return (img, label)


class RandomBlur:
    """
    Blurring an image by a Gaussian function with a certain probability.
    用高斯函数以一定的概率模糊一个图像。

    Args:
        prob (float, optional): A probability of blurring an image. Default: 0.1.
    """

    def __init__(self, prob=0.1):
        self.prob = prob

    def __call__(self, img, label=None):
        """
        Args:
            img (np.ndarray): The Image data.
            label (np.ndarray, optional): The label data. Default: None.

        Returns:
            (tuple). When label is None, it returns (img, ), otherwise 
            it returns (img, label).
        """

        if self.prob <= 0:
            n = 0
        elif self.prob >= 1:
            n = 1
        else:
            n = int(1.0 / self.prob)
        if n > 0:
            if np.random.randint(0, n) == 0:
                radius = np.random.randint(3, 10)
                if radius % 2 != 1:
                    radius = radius + 1
                if radius > 9:
                    radius = 9
                img = cv2.GaussianBlur(img, (radius, radius), 0, 0)

        if label is None:
            return (img,)
        else:
            return (img, label)


class RandomRotation:
    """
    Rotate an image randomly with padding. 用填充随机旋转一个图像。

    Args:
        max_rotation (float, optional): The maximum rotation degree. Default: 15.
        img_padding_value (list, optional): The padding value of raw image.
        Default: [127.5, 127.5, 127.5].
        label_padding_value (int, optional): The padding value of annotation
        image. Default: 255.
    """

    def __init__(self,
                 max_rotation=15,
                 img_padding_value=(127.5, 127.5, 127.5),
                 label_padding_value=255):
        self.max_rotation = max_rotation
        self.img_padding_value = img_padding_value
        self.label_padding_value = label_padding_value

    def __call__(self, img, label=None):
        """
        Args:
            img (np.ndarray): The Image data.
            label (np.ndarray, optional): The label data. Default: None.

        Returns:
            (tuple): When label is None, it returns (img, ), otherwise 
            it returns (img, label).
        """

        if self.max_rotation > 0:
            (h, w) = img.shape[:2]
            do_rotation = np.random.uniform(-self.max_rotation,
                                            self.max_rotation)
            pc = (w // 2, h // 2)
            r = cv2.getRotationMatrix2D(pc, do_rotation, 1.0)
            cos = np.abs(r[0, 0])
            sin = np.abs(r[0, 1])

            nw = int((h * sin) + (w * cos))
            nh = int((h * cos) + (w * sin))

            (cx, cy) = pc
            r[0, 2] += (nw / 2) - cx
            r[1, 2] += (nh / 2) - cy
            dsize = (nw, nh)
            img = cv2.warpAffine(
                img, r, dsize=dsize, flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=self.im_padding_value)
            if label is not None:
                label = cv2.warpAffine(
                    label, r, dsize=dsize, flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=self.label_padding_value)

        if label is None:
            return (img,)
        else:
            return (img, label)


class RandomDistort:
    """
    Distort an image with random configurations. 用随机配置扭曲图像。

    Args:
        brightness_range (float, optional): The range of brightness.
        brightness_prob (float, optional): The probability of adjusting brightness.
        contrast_range (float, optional): The range of contrast.
        contrast_prob (float, optional): The probability of adjusting contrast.
        saturation_range (float, optional): The range of saturation.
        saturation_prob (float, optional): The probability of adjusting saturation.
        hue_range (int, optional): The range of hue.
        hue_prob (float, optional): The probability of adjusting hue.
    """

    def __init__(self,
                 brightness_range=0.5,
                 brightness_prob=0.5,
                 contrast_range=0.5,
                 contrast_prob=0.5,
                 saturation_range=0.5,
                 saturation_prob=0.5,
                 hue_range=18,
                 hue_prob=0.5):
        self.brightness_range = brightness_range
        self.brightness_prob = brightness_prob
        self.contrast_range = contrast_range
        self.contrast_prob = contrast_prob
        self.saturation_range = saturation_range
        self.saturation_prob = saturation_prob
        self.hue_range = hue_range
        self.hue_prob = hue_prob

    def __call__(self, img, label=None):
        """
        Args:
            img (np.ndarray): The Image data.
            label (np.ndarray, optional): The label data. Default: None.

        Returns:
            (tuple): When label is None, it returns (img, ), 
            otherwise it returns (img, label).
        """

        brightness_lower = 1 - self.brightness_range
        brightness_upper = 1 + self.brightness_range
        contrast_lower = 1 - self.contrast_range
        contrast_upper = 1 + self.contrast_range
        saturation_lower = 1 - self.saturation_range
        saturation_upper = 1 + self.saturation_range
        hue_lower = -self.hue_range
        hue_upper = self.hue_range
        ops = [
            functional.brightness, functional.contrast, functional.saturation,
            functional.hue
        ]
        random.shuffle(ops)
        params_dict = {
            'brightness': {
                'brightness_lower': brightness_lower,
                'brightness_upper': brightness_upper
            },
            'contrast': {
                'contrast_lower': contrast_lower,
                'contrast_upper': contrast_upper
            },
            'saturation': {
                'saturation_lower': saturation_lower,
                'saturation_upper': saturation_upper
            },
            'hue': {
                'hue_lower': hue_lower,
                'hue_upper': hue_upper
            }
        }
        prob_dict = {
            'brightness': self.brightness_prob,
            'contrast': self.contrast_prob,
            'saturation': self.saturation_prob,
            'hue': self.hue_prob
        }
        img = img.astype('uint8')
        img = Image.fromarray(img)
        for id in range(len(ops)):
            params = params_dict[ops[id].__name__]
            prob = prob_dict[ops[id].__name__]
            params['img'] = img
            if np.random.uniform(0, 1) < prob:
                img = ops[id](**params)
        img = np.asarray(img).astype('float32')
        if label is None:
            return (img,)
        else:
            return (img, label)
