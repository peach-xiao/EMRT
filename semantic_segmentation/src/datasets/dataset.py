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

import os
import paddle
import numpy as np
from PIL import Image
from src.transforms import Compose
import src.transforms.functional as F


class Dataset(paddle.io.Dataset):
    """
    The custom dataset that conforms to the format.

    Args:
        transforms (list): Transforms for image.
        dataset_root (str): The dataset directory.
        num_classes (int): Number of classes.
        mode (str, optional): which part of dataset to use. it is one of 
        ('train', 'val', 'test'). Default: 'train'.
        train_path (str, optional): The train dataset file. When mode is 
        'train', train_path is necessary.
        val_path (str. optional): The evaluation dataset file. When mode 
        is 'val', val_path is necessary. The contents is the same as train_path
        test_path (str, optional): The test dataset file. When mode is 'test', 
        test_path is necessary. 
        ignore_index (int): ignore label, default=255

    """

    def __init__(self,
                 transforms,
                 dataset_root,
                 num_classes,
                 mode='train',
                 train_path=None,
                 val_path=None,
                 test_path=None,
                 ignore_index=255):
        self.dataset_root = dataset_root
        self.transforms = Compose(transforms) #构建数据增强对象
        self.file_list = list() #新建一个保存文件路径的空列表
        mode = mode.lower()
        self.mode = mode
        self.num_classes = num_classes
        self.ignore_index = ignore_index #保存需要忽略的类别编号，一般都是255

        if mode.lower() not in ['train', 'val', 'test']:
            raise ValueError("mode should be 'train', 'val' or 'test', "
                             "but got {}.".format(mode))
        if self.transforms is None: #数据增强对象必须指定，如果未设置，抛出异常。
            raise ValueError("`transforms` is necessary, but it is None.")
        self.dataset_root = dataset_root #如果数据集路径不存在则抛出异常。
        if not os.path.exists(self.dataset_root):
            raise FileNotFoundError("there is not `dataset_root`: {}."
                                    .format(self.dataset_root))

    def __getitem__(self, idx):
        image_path, label_path = self.file_list[idx] # #通过idx下标，在file_list里获取样本图片路径和标签图片路径。
        # 如果是测试模式则返回图片ndarray类型的数据。在transforms中，包含了图片的读取和预处理，不同模式的dataset类的transforms对象是不同的。
        if self.mode == 'test':
            img, _ = self.transforms(img=image_path)
            img = img[np.newaxis, ...]
            return img, image_path
        elif self.mode == 'val': #如果是训练或者验证模式还需要返回样本图片的和标签图片的ndarray的数据类型。
            img, _ = self.transforms(img=image_path)
            label = np.asarray(Image.open(label_path).convert('P'))
            label = label[np.newaxis, :, :]
            return img, label
        else:
            img, label = self.transforms(img=image_path, label=label_path)
            return img, label

    def __len__(self):
        return len(self.file_list)
