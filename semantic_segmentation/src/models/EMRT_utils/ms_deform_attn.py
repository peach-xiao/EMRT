# ------------------------------------------------------------------------
#  Deformable Self-attention
# ------------------------------------------------------------------------
# Modified from Deformable DETR

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import warnings
import math
import paddle
from paddle import nn
import paddle.nn.functional as F
from .initializer import xavier_uniform_, constant_, normal_
from .ms_deform_attn_func import ms_deform_attn_core_pytorch

class MSDeformAttn(nn.Layer):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension         隐变量的维度，也是变换后特征的维度 256
        :param n_levels     number of feature levels 多尺度融合的level数，取决于backbone的输入，比如resnet输出4个stage的输出
        :param n_heads      number of attention heads 多头attention中的head的个数
        :param n_points     number of sampling points per attention head per feature level
        每个query在每个level中每个head中采样的点的个数，也就是说每个query其实采样的点数为 n_leveln_headsn_points, 默认4x4x8=128个点。
        """

        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads

        self.im2col_step = 64

        self.d_model = d_model #256
        self.n_levels = n_levels #4
        self.n_heads = n_heads #8
        self.n_points = n_points #4

        # 每个head为每个level产生n_point个点的偏置， 对应公式里的Delta
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        # 每个位置点的权重，由网络直接生成， 对应公式里的A_{mlqk}
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        # 数据进行变换, 对应W_m'
        self.value_proj = nn.Linear(d_model, d_model)
        # 总体和进行再变换, 对应W_m
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    #conference paddle
    def _reset_parameters(self):
        # 这里初始化不同的权重，采样不同的偏置点时有些特殊，不同的level不同的point初始偏置bias不同
        constant_(self.sampling_offsets.weight)
        thetas = paddle.arange(self.n_heads, dtype=paddle.float32) * (2.0 * math.pi / self.n_heads)

        # grid_init = paddle.stack([thetas.cos(), thetas.sin()*thetas.cos(), thetas.sin()*thetas.sin()], -1)
        # grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 3).repeat(1, self.n_levels, self.n_points, 1)
        # # 相当于每个level每个point偏置对应的head进行编码
        # for i in range(self.n_points):
        #     grid_init[:, :, i, :] *= i + 1 # 对不同的偏置进行编码, 不同点的编码不同但不同level是相同的
        # with paddle.no_grad():
        #  self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))

        grid_init = paddle.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True)
        grid_init = grid_init.reshape([self.n_heads, 1, 1, 2]).tile([1, self.n_levels, self.n_points, 1])
        scaling = paddle.arange(
            1, self.n_points + 1,
            dtype=paddle.float32).reshape([1, 1, -1, 1])
        grid_init *= scaling

        self.sampling_offsets.bias.set_value(grid_init.flatten())
        # attention_weights
        constant_(self.attention_weights.weight)
        constant_(self.attention_weights.bias)
        # proj
        xavier_uniform_(self.value_proj.weight)
        constant_(self.value_proj.bias)
        xavier_uniform_(self.output_proj.weight)
        constant_(self.output_proj.bias)

    # forward:
    # 1.query : query向量 batch_size x query个数 x 表征维度
    # 2.reference_points: batch_size x query个数 x level个数 x 2
    # 表示每个query在每个level中的参考位置，也就是公式中的\phi(pq).归一化的话其实每个level上的reference_points相同
    # 3.input_flatten: batch_size x key的个数 x 特征维度, key包括所有level中的像素位置对应的特征向量。对应公式中的x
    # 4.input_level_shapes: level个数 x 2, 表示每个level的feature map的尺寸，是H x W
    # 5.input_level_start_index: 每个level的key在总体的input_flatten中的初始位置，这个量只在采样时使用
    # 6.input_padding_mask: bool，表示每个input_flatten位置的掩码
    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_padding_mask=None):

        # :param query                       (N, Length_{query}, C)
        # :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
        #                                 or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        # # 每个query在不同的level的参考位置，即公式2的q
        # :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        # # 把不同的level特征flatten一起，所有key的个数，即所有level的像素点个数之和
        # :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        # # 每个level的尺寸
        # :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        # # 每个level的开始索引， 相当于不同的level进行序列排序后的索引
        # :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements
        #                                      [bs, value_length], True for non-padding elements, False for padding elements
        # # bool，每个位置是否mask
        # :return output                     (N, Length_{query}, C)
        #

        N, Len_q, _ = query.shape  # batch size, query的个数
        N, Len_in, _ = input_flatten.shape  # Len_in是所有key的个数
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(input_flatten)  # 数据进行变换, 对应W_m'
        if input_padding_mask is not None:
            # value = value.masked_fill(input_padding_mask[..., None], float(0))
            input_padding_mask = input_padding_mask.astype(value.dtype).unsqueeze(-1)
            value *= input_padding_mask

        #注意：！torch 里面的.view()对于paddle里面的.reshape()
        value = value.reshape([N, Len_in, self.n_heads, self.d_model // self.n_heads])
        # 每个query产生对应不同head不同level的偏置   # 每个head为每个level产生n_point个点的偏置， 对应公式里的Delta
        sampling_offsets = self.sampling_offsets(query).reshape([N, Len_q, self.n_heads, self.n_levels, self.n_points, 2])
        # 每个偏置向量的权重 ,对应公式里的A_{mlqk}
        attention_weights = self.attention_weights(query).reshape([N, Len_q, self.n_heads, self.n_levels * self.n_points])
        # 对属于同一个query的来自与不同level的offset后向量权重在每个head分别归一化
        attention_weights = F.softmax(attention_weights, -1).reshape([N, Len_q, self.n_heads, self.n_levels, self.n_points])

        # N, Len_q, n_heads, n_levels, n_points, 2

        offset_normalizer = paddle.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
        sampling_locations = reference_points[:, :, None, :, None, :] \
                             + sampling_offsets / offset_normalizer[None, None, None, :, None, :]

        # sampling_offsets取值非0，1之间，因此这里相当于归一化后，计算$x_q+\Delta$

        output = ms_deform_attn_core_pytorch(value, input_spatial_shapes, sampling_locations, attention_weights)
        output = self.output_proj(output)
        return output  #output: attention融合之后的每个query的特征向量，长度和输入相同
