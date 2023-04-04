#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import math
import six
import numpy as np
from numbers import Integral

import paddle
import paddle.nn as nn
from paddle import ParamAttr
from paddle import to_tensor
import paddle.nn.functional as F
from paddle.nn.initializer import Normal, Constant, XavierUniform
from paddle.regularizer import L2Decay
from .initializer import xavier_uniform_, constant_

from paddle.vision.ops import DeformConv2D


class DeformableConvV2(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 weight_attr=None, bias_attr=None, lr_scale=1, regularizer=None, skip_quant=False,
                 dcn_bias_regularizer=L2Decay(0.), dcn_bias_lr_scale=2.):
        super(DeformableConvV2, self).__init__()
        self.offset_channel = 2 * kernel_size ** 2
        self.mask_channel = kernel_size ** 2

        if lr_scale == 1 and regularizer is None:
            offset_bias_attr = ParamAttr(initializer=Constant(0.))
        else:
            offset_bias_attr = ParamAttr(initializer=Constant(0.), learning_rate=lr_scale, regularizer=regularizer)

        self.conv_offset = nn.Conv2D(in_channels, 3 * kernel_size ** 2, kernel_size,
                                     stride=stride, padding=(kernel_size - 1) // 2,
                                     weight_attr=ParamAttr(initializer=Constant(0.0)), bias_attr=offset_bias_attr)

        if skip_quant:
            self.conv_offset.skip_quant = True

        if bias_attr:
            # in FCOS-DCN head, specifically need learning_rate and regularizer
            dcn_bias_attr = ParamAttr(initializer=Constant(value=0), regularizer=dcn_bias_regularizer,
                                      learning_rate=dcn_bias_lr_scale)
        else:
            # in ResNet backbone, do not need bias
            dcn_bias_attr = False
        self.conv_dcn = DeformConv2D(in_channels, out_channels, kernel_size,
                                     stride=stride, padding=(kernel_size - 1) // 2 * dilation,
                                     dilation=dilation, groups=groups, weight_attr=weight_attr, bias_attr=dcn_bias_attr)

    def forward(self, x):
        offset_mask = self.conv_offset(x)  # channel = 3 * kernel_size**2,  27
        offset, mask = paddle.split(offset_mask, num_or_sections=[self.offset_channel, self.mask_channel],
                                    axis=1)  # 18,9
        # self.offset_channel = 2 * kernel_size**2 self.mask_channel = kernel_size**2
        mask = F.sigmoid(mask)
        y = self.conv_dcn(x, offset, mask=mask)
        return y


class ConvNormLayer(nn.Layer):
    def __init__(self, ch_in, ch_out, filter_size, stride, groups=1, norm_type='bn', norm_decay=0.,
                 norm_groups=32, use_dcn=False, bias_on=False, lr_scale=1.,
                 freeze_norm=False, initializer=Normal(mean=0., std=0.01),
                 skip_quant=False, dcn_lr_scale=2., dcn_regularizer=L2Decay(0.)):
        super(ConvNormLayer, self).__init__()
        assert norm_type in ['bn', 'sync_bn', 'gn', None]

        if bias_on:
            bias_attr = ParamAttr(initializer=Constant(value=0.), learning_rate=lr_scale)
        else:
            bias_attr = False

        if not use_dcn:
            self.conv = nn.Conv2D(in_channels=ch_in, out_channels=ch_out, kernel_size=filter_size,
                                  stride=stride, padding=(filter_size - 1) // 2,
                                  groups=groups, weight_attr=ParamAttr(initializer=initializer, learning_rate=1.),
                                  bias_attr=bias_attr)
            if skip_quant:
                self.conv.skip_quant = True
        else:
            # in FCOS-DCN head, specifically need learning_rate and regularizer
            self.conv = DeformableConvV2(in_channels=ch_in, out_channels=ch_out,
                                         kernel_size=filter_size, stride=stride, padding=(filter_size - 1) // 2,
                                         groups=groups,
                                         weight_attr=ParamAttr(initializer=initializer, learning_rate=1.),
                                         bias_attr=True, lr_scale=dcn_lr_scale, regularizer=dcn_regularizer,
                                         dcn_bias_regularizer=dcn_regularizer,
                                         dcn_bias_lr_scale=dcn_lr_scale, skip_quant=skip_quant)

        norm_lr = 0. if freeze_norm else 1.
        param_attr = ParamAttr(learning_rate=norm_lr,
                               regularizer=L2Decay(norm_decay) if norm_decay is not None else None)
        bias_attr = ParamAttr(learning_rate=norm_lr,
                              regularizer=L2Decay(norm_decay) if norm_decay is not None else None)
        if norm_type in ['bn', 'sync_bn']:
            self.norm = nn.BatchNorm2D(ch_out, weight_attr=param_attr, bias_attr=bias_attr)
        elif norm_type == 'gn':
            self.norm = nn.GroupNorm(num_groups=norm_groups, num_channels=ch_out, weight_attr=param_attr,
                                     bias_attr=bias_attr)
        else:
            self.norm = None

    def forward(self, inputs):
        out = self.conv(inputs)
        if self.norm is not None:
            out = self.norm(out)
        return out


def _convert_attention_mask(attn_mask, dtype):
    """
    Convert the attention mask to the target dtype we expect.
    Parameters:
        attn_mask (Tensor, optional): A tensor used in multi-head attention
                to prevents attention to some unwanted positions, usually the
                paddings or the subsequent positions. It is a tensor with shape
                broadcasted to `[batch_size, n_head, sequence_length, sequence_length]`.
                When the data type is bool, the unwanted positions have `False`
                values and the others have `True` values. When the data type is
                int, the unwanted positions have 0 values and the others have 1
                values. When the data type is float, the unwanted positions have
                `-INF` values and the others have 0 values. It can be None when
                nothing wanted or needed to be prevented attention to. Default None.
        dtype (VarType): The target type of `attn_mask` we expect.
    Returns:
        Tensor: A Tensor with shape same as input `attn_mask`, with data type `dtype`.
    """
    return nn.layer.transformer._convert_attention_mask(attn_mask, dtype)


class MultiHeadAttention(nn.Layer):
    """
    Attention mapps queries and a set of key-value pairs to outputs, and
    Multi-Head Attention performs multiple parallel attention to jointly attending
    to information from different representation subspaces.

    Please refer to `Attention Is All You Need <https://arxiv.org/pdf/1706.03762.pdf>`_
    for more details.

    Parameters:
        embed_dim (int): The expected feature size in the input and output.
        num_heads (int): The number of heads in multi-head attention.
        dropout (float, optional): The dropout probability used on attention
            weights to drop some attention targets. 0 for no dropout. Default 0
        kdim (int, optional): The feature size in key. If None, assumed equal to
            `embed_dim`. Default None.
        vdim (int, optional): The feature size in value. If None, assumed equal to
            `embed_dim`. Default None.
        need_weights (bool, optional): Indicate whether to return the attention
            weights. Default False.

    Examples:

        .. code-block:: python

            import paddle

            # encoder input: [batch_size, sequence_length, d_model]
            query = paddle.rand((2, 4, 128))
            # self attention mask: [batch_size, num_heads, query_len, query_len]
            attn_mask = paddle.rand((2, 2, 4, 4))
            multi_head_attn = paddle.nn.MultiHeadAttention(128, 2)
            output = multi_head_attn(query, None, None, attn_mask=attn_mask)  # [2, 4, 128]
    """

    def __init__(self, embed_dim, num_heads, dropout=0., kdim=None, vdim=None, need_weights=False):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.need_weights = need_weights

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim:
            self.in_proj_weight = self.create_parameter(
                shape=[embed_dim, 3 * embed_dim],
                attr=None,
                dtype=self._dtype,
                is_bias=False)
            self.in_proj_bias = self.create_parameter(
                shape=[3 * embed_dim],
                attr=None,
                dtype=self._dtype,
                is_bias=True)
        else:
            self.q_proj = nn.Linear(embed_dim, embed_dim)
            self.k_proj = nn.Linear(self.kdim, embed_dim)
            self.v_proj = nn.Linear(self.vdim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self._type_list = ('q_proj', 'k_proj', 'v_proj')

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
            else:
                constant_(p)

    def compute_qkv(self, tensor, index):
        if self._qkv_same_embed_dim:
            tensor = F.linear(
                x=tensor,
                weight=self.in_proj_weight[:, index * self.embed_dim:(index + 1)
                                                                     * self.embed_dim],
                bias=self.in_proj_bias[index * self.embed_dim:(index + 1) *
                                                              self.embed_dim]
                if self.in_proj_bias is not None else None)
        else:
            tensor = getattr(self, self._type_list[index])(tensor)
        tensor = tensor.reshape(
            [0, 0, self.num_heads, self.head_dim]).transpose([0, 2, 1, 3])
        return tensor

    def forward(self, query, key=None, value=None, attn_mask=None):
        r"""
        Applies multi-head attention to map queries and a set of key-value pairs
        to outputs.

        Parameters:
            query (Tensor): The queries for multi-head attention. It is a
                tensor with shape `[batch_size, query_length, embed_dim]`. The
                data type should be float32 or float64.
            key (Tensor, optional): The keys for multi-head attention. It is
                a tensor with shape `[batch_size, key_length, kdim]`. The
                data type should be float32 or float64. If None, use `query` as
                `key`. Default None.
            value (Tensor, optional): The values for multi-head attention. It
                is a tensor with shape `[batch_size, value_length, vdim]`.
                The data type should be float32 or float64. If None, use `query` as
                `value`. Default None.
            attn_mask (Tensor, optional): A tensor used in multi-head attention
                to prevents attention to some unwanted positions, usually the
                paddings or the subsequent positions. It is a tensor with shape
                broadcasted to `[batch_size, n_head, sequence_length, sequence_length]`.
                When the data type is bool, the unwanted positions have `False`
                values and the others have `True` values. When the data type is
                int, the unwanted positions have 0 values and the others have 1
                values. When the data type is float, the unwanted positions have
                `-INF` values and the others have 0 values. It can be None when
                nothing wanted or needed to be prevented attention to. Default None.

        Returns:
            Tensor|tuple: It is a tensor that has the same shape and data type \
                as `query`, representing attention output. Or a tuple if \
                `need_weights` is True or `cache` is not None. If `need_weights` \
                is True, except for attention output, the tuple also includes \
                the attention weights tensor shaped `[batch_size, num_heads, query_length, key_length]`. \
                If `cache` is not None, the tuple then includes the new cache \
                having the same type as `cache`, and if it is `StaticCache`, it \
                is same as the input `cache`, if it is `Cache`, the new cache \
                reserves tensors concatanating raw tensors with intermediate \
                results of current query.
        """
        key = query if key is None else key
        value = query if value is None else value
        # compute q ,k ,v
        q, k, v = (self.compute_qkv(t, i)
                   for i, t in enumerate([query, key, value]))

        # scale dot product attention
        product = paddle.matmul(x=q, y=k, transpose_y=True)
        scaling = float(self.head_dim) ** -0.5
        product = product * scaling

        if attn_mask is not None:
            # Support bool or int mask
            attn_mask = _convert_attention_mask(attn_mask, product.dtype)
            product = product + attn_mask
        weights = F.softmax(product)
        if self.dropout:
            weights = F.dropout(
                weights,
                self.dropout,
                training=self.training,
                mode="upscale_in_train")

        out = paddle.matmul(weights, v)

        # combine heads
        out = paddle.transpose(out, perm=[0, 2, 1, 3])
        out = paddle.reshape(x=out, shape=[0, 0, out.shape[2] * out.shape[3]])

        # project to output
        out = self.out_proj(out)

        outs = [out]
        if self.need_weights:
            outs.append(weights)
        return out if len(outs) == 1 else tuple(outs)


class MultiHeadAttention_end(nn.Layer):
    def __init__(self, dim=256, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super(MultiHeadAttention_end, self).__init__()
        self.num_heads = num_heads
        self.attn_head_size = int(dim / self.num_heads)  # 256/8
        self.all_head_size = self.attn_head_size * self.num_heads  # 256

        self.scale = self.attn_head_size ** -0.5

        self.query = nn.Linear(dim, dim, bias_attr=qkv_bias)
        self.key = nn.Linear(dim, dim, bias_attr=qkv_bias)
        self.value = nn.Linear(dim, dim, bias_attr=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def transpose_multihead(self, x):
        new_shape = x.shape[:-1] + [self.num_heads, self.attn_head_size]
        x = x.reshape(new_shape)
        # print("transpose_multihead", x.shape) #[8, 1024, 8, 32]
        x = x.transpose([0, 2, 1, 3])
        return x

    def forward(self, psp_query, encoder_x):
        # print("psp,x",psp_query.shape,encoder_x.shape) #[8, 256, 32, 32] [8, 256, 32, 32]
        bs, c, h, w = encoder_x.shape
        x1 = psp_query.flatten(2).transpose([0, 2, 1])  # [8, 1024, 256]
        x2 = encoder_x.flatten(2).transpose([0, 2, 1])  # [8, 1024, 256]

        B, N, C = x2.shape

        mixed_q = self.query(x1)  # .reshape([B, self.num_heads, N, C // self.num_heads])
        mixed_k = self.key(x2)  # .reshape([B, self.num_heads, N, C // self.num_heads])
        mixed_v = self.value(x2)  # .reshape([B, self.num_heads, N, C // self.num_heads])

        q = self.transpose_multihead(mixed_q)
        k = self.transpose_multihead(mixed_k)
        v = self.transpose_multihead(mixed_v)

        # print("q.k.v", q.shape, k.shape, v.shape)# [8, 8, 1024, 32] [8, 8, 1024, 32] [8, 8, 1024, 32]

        attn = (q @ k.transpose([0, 1, 3, 2])) * self.scale
        attn = F.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)
        # print("attn", attn.shape)#[8, 8, 1024, 1024]

        x = (attn @ v).reshape([B, N, C])
        # print("x", x.shape) #[8, 1024, 256]

        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.transpose([0, 2, 1]).reshape([bs, c, h, w])  # # [8, 256, 32, 32]

        return x
