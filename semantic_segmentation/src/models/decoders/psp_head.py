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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class PyramidPoolingModule(nn.Layer):
    """PyramidPoolingModule

    VisionTransformerUpHead is the decoder of PSPNet, Ref https://arxiv.org/abs/1612.01105.pdf

    Reference:                                                                                                                                                
        Hengshuang Zhao, et al. *"Pyramid Scene Parsing Network"*
    """

    def __init__(self, pool_scales, in_channels, channels, align_corners=False):
        super(PyramidPoolingModule, self).__init__()
        self.pool_scales = pool_scales  # [1, 2, 3, 6]
        self.in_channels = in_channels  # 768
        self.channels = channels  # 512
        self.align_corners = align_corners
        #创建一个参数属性对象，用户可设置参数的名称、初始化方式、学习率、正则化规则、是否需要训练、梯度裁剪方式、是否做模型平均等属性。
        norm_bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0.0))

        self.pool_branches = nn.LayerList()
        for idx in range(len(self.pool_scales)):
            self.pool_branches.append(nn.Sequential(
                nn.AdaptiveAvgPool2D(self.pool_scales[idx]),
                nn.Conv2D(self.in_channels, self.channels, 1, stride=1, bias_attr=False),
                nn.SyncBatchNorm(self.channels, weight_attr=self.get_norm_weight_attr(), bias_attr=norm_bias_attr),#指定偏置参数属性的对象
                nn.ReLU()))
    # nn.SyncBatchNorm实现了跨卡GPU同步的批归一化(Cross-GPU Synchronized Batch Normalization Layer)的功能，
    # 可用在其他层（类似卷积层和全连接层）之后进行归一化操作。根据所有GPU同一批次的数据按照通道计算的均值和方差进行归一化。
    # weight_attr (ParamAttr|bool, 可选) - 指定权重参数属性的对象。如果设置为 False ，则表示本层没有可训练的权重参数。默认值为None，表示使用默认的权重参数属性。
    # bias_attr(ParamAttr | bool, 可选) - 指定偏置参数属性的对象。如果设置为False ，则表示本层没有可训练的偏置参数。默认值为None，表示使用默认的偏置参数属性。
    def get_norm_weight_attr(self):
        return paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(value=1.0))
    # paddle.nn.initializer.Constant() 该接口为常量初始化函数，用于权重初始化，通过输入的value值初始化输入变量

    def forward(self, x):
        outs = []
        up_resolution = [item for item in x.shape[2:]]
        for _, pool_layer in enumerate(self.pool_branches):
            out = pool_layer(x)
            up_out = F.interpolate(out, up_resolution, mode='bilinear', align_corners=self.align_corners) #align_corners = false
            outs.append(up_out)
        return outs
