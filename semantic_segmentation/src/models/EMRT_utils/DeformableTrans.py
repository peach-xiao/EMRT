# ------------------------------------------------------------------------
# Deformable Transformer
# ------------------------------------------------------------------------
# Modified from Deformable DETR

import copy
from typing import Optional, List
import math
import paddle
import paddle.nn.functional as F
from paddle import nn
from .initializer import xavier_uniform_, constant_, normal_, linear_init_
from .ms_deform_attn import MSDeformAttn
from .position_encoding import build_position_encoding


class DeformableTransformer(nn.Layer):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", num_feature_levels=4, enc_n_points=4):
        super().__init__()

        self.d_model = d_model #256
        self.nhead = nhead

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        #self.level_embed:
        # DETR仅用了单尺度特征，于是对于特征点位置信息的编码，使用的是三角函数，不同位置的特征点会对应不同的编码值
        # 但是，这仅能区分位于单尺度特征点的位置！而在多尺度特征中，位于不同特征层的特征点可能拥有相同的(h,w)坐标，这样就无法区分它们的位置编码了。
        # 针对这个问题，作者增加使用一个称之为'scale-level embedding'的东东，它仅用于区分不同的特征层，
        # 也就是同一特征层中的所有特征点会对应相同的scale-level embedding，于是有几层特征就使用几个不同的scale-level embedding
        # 另外，不同于三角函数那种固定地利用公式计算出来的编码方式，这个scale-level embedding是随机初始化并且是随网络一起训练的、是可学习的
        self.level_embed = nn.Embedding(num_feature_levels, d_model)

        #self.level_embed -> 对4个特征图每个附加256-dim的embedding,目的是为了区分各query对应到那个特征层。
        # 这个 scale-level embedding 与基于三角函数公式计算的 position embedding 相加在一起作为位置信息的嵌入
        #注意，位于同一个特征图的所有query都会对应到相通的scale-level embedding

        # self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model)) #torch方式

        self.reference_points = nn.Linear(d_model, 2)

        self._reset_parameters()

    def _reset_parameters(self):
        # for p in self.parameters():
        #     if p.dim() > 1:
        #         # nn.init.xavier_uniform_(p)
        #         xavier_uniform_(p)
        # for m in self.modules():
        #     if isinstance(m, MSDeformAttn):
        #         m._reset_parameters() #调用ms_deform.attn.py 里面的函数执行初始化参数
        # normal_(self.level_embed)
        xavier_uniform_(self.reference_points.weight)
        constant_(self.reference_points.bias)
        normal_(self.level_embed.weight)

    def get_valid_ratio(self, mask):
        mask = mask.astype(paddle.float32)
        _, H, W = mask.shape
        valid_H = paddle.sum(mask[:, :, 0], 1)
        valid_W = paddle.sum(mask[:, 0, :], 1)
        valid_ratio_h = valid_H / H #valid_H.float() --> 'Tensor' object has no attribute 'float'
        valid_ratio_w = valid_W / W
        valid_ratio = paddle.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds):

        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        # print("srcs",srcs)
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            # print("src",src) #shape=[8, 256, 32, 32]
            src = src.flatten(2).transpose([0, 2, 1])  # (bs,c,h,w) -> (bs,c,hw) -> (bs,hw,c)
            # print("src2", src) #shape=[8, 1024, 256],
            # print("mask",mask) #shape=[8, 32, 32],
            # print("src.dtype",lvl, src.dtype)
            mask = mask.astype(src.dtype).flatten(1)  # bs x hw
            # print("mask2", mask) # shape=[8, 1024]

            pos_embed = pos_embed.flatten(2).transpose([0, 2, 1])  # (bs,c,h,w) -> (bs,c,hw) -> (bs,hw,c)
            lvl_pos_embed = pos_embed +  self.level_embed.weight[lvl].reshape([1, 1, -1])   # (bs,hw,c) + (1,1,c) 每一level提供一个可学习的编码

            lvl_pos_embed_flatten.append(lvl_pos_embed)  # 分别flatten之后append，方便encoder调用，即所有的keys
            src_flatten.append(src)
            mask_flatten.append(mask)

        src_flatten = paddle.concat(src_flatten, 1)
        mask_flatten = paddle.concat(mask_flatten, 1)
        lvl_pos_embed_flatten = paddle.concat(lvl_pos_embed_flatten, 1)

        spatial_shapes = paddle.to_tensor(spatial_shapes, dtype='int64')
        # level_start_index = paddle.concat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        #这个 level_start_index究竟有什么用？

        valid_ratios = paddle.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder
        memory = self.encoder(src_flatten, spatial_shapes, valid_ratios, lvl_pos_embed_flatten, mask_flatten)

        return memory


class DeformableTransformerEncoderLayer(nn.Layer):
    def __init__(self,
                 d_model=256, dim_feedforward=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        linear_init_(self.linear1)
        linear_init_(self.linear2)
        xavier_uniform_(self.linear1.weight)
        xavier_uniform_(self.linear2.weight)

    # 当某个方法不需要用到对象中的任何资源,将这个方法改为一个静态方法, 加一个@staticmethod
    # 加上之后, 这个方法就和普通的函数没有什么区别了, 只不过写在了一个类中, 可以使用这个类的对象调用
    @staticmethod
    def with_pos_embed(tensor, pos): #注意：放在函数前（该函数不传入self或者cls），所以不能访问类属性和实例属性
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, padding_mask=None):
        # pos: 每个query的特征，在encoder里是每一个level中每个位置点的特征
        # src: backbone的输出，可能是多个stage，是cat之后再flatten的结果
        # reference_points: batch_size x query个数 x level个数 x 2 ，每个query每个level的位置点，归一化之后的点，encoder里是每个level的位置点归一化之后的位置
        # spatial_shapes: 每个level的featmap尺寸
        # level_start_index 每个level在flatten的特征向量集上的起始索引
        # padding_mask 考虑所有level，每个位置是否mask的标志

        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src


class DeformableTransformerEncoder(nn.Layer):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios):  # spatial_shapes: backbone提取出的几个特征图的大小(h，w)
        valid_ratios = valid_ratios.unsqueeze(1)
        reference_points = []
        for i, (H, W) in enumerate(spatial_shapes.tolist()):
            # reference_points的定义,其实就是计算每个level中每个网格点的位置，这里的位置采用的是网格的中心点
            ref_y, ref_x = paddle.meshgrid(
                paddle.linspace(0.5, H - 0.5, H),
                paddle.linspace(0.5, W - 0.5, W))
            ref_y = ref_y.flatten().unsqueeze(0) / (valid_ratios[:, :, i, 1] * H)
            ref_x = ref_x.flatten().unsqueeze(0) / (valid_ratios[:, :, i, 0] * W)

            reference_points.append(paddle.stack((ref_x, ref_y), axis=-1))
        reference_points = paddle.concat(reference_points, 1).unsqueeze(2)
        reference_points = reference_points * valid_ratios
        return reference_points

    # valid_ratios 需要解释一下，query的个数是所有的像素位置，包括不同的level， 那么每个query都需要在不同的level上采点，
    # 所以需要每个reference_point在每个level上映射后的点，所以这里的valid_ratios在计算时就是公式2里的\phi函数。
    # 于是reference_points的size为 BatchSize x \sum_l^L (H_l x W_l) x L x 2，总共有\sum_l^L (H_l x W_l)个queries
    def forward(self, src, spatial_shapes, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, padding_mask)

        return output


def _get_clones(module, N):
    return nn.LayerList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


