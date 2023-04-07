from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
import warnings

from .layers import MultiHeadAttention
from .position_encoding import PositionEmbedding
from .utils import _get_clones, deformable_attention_core_func
from .initializer import linear_init_, constant_, xavier_uniform_, normal_

from ..backbones.swin_transformer import Identity, DropPath, Mlp

__all__ = ['EncoderDecoder']

class MSDeformableAttention(nn.Layer):
    def __init__(self, embed_dim=256, num_heads=8, num_levels=4, num_points=4, lr_mult=0.1):
        """
        Multi-Scale Deformable Attention Module
        """
        super(MSDeformableAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.total_points = num_heads * num_levels * num_points

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2,
                                          weight_attr=ParamAttr(learning_rate=lr_mult),
                                          bias_attr=ParamAttr(learning_rate=lr_mult))

        self.attention_weights = nn.Linear(embed_dim, self.total_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight)  # 0
        thetas = paddle.arange(self.num_heads, dtype=paddle.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = paddle.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True)
        grid_init = grid_init.reshape([self.num_heads, 1, 1, 2]).tile([1, self.num_levels, self.num_points, 1])
        scaling = paddle.arange(1, self.num_points + 1, dtype=paddle.float32).reshape([1, 1, -1, 1])
        grid_init *= scaling

        self.sampling_offsets.bias.set_value(grid_init.flatten())
        # attention_weights
        constant_(self.attention_weights.weight)
        constant_(self.attention_weights.bias)

        xavier_uniform_(self.value_proj.weight)
        constant_(self.value_proj.bias)
        xavier_uniform_(self.output_proj.weight)
        constant_(self.output_proj.bias)

    def forward(self, query, reference_points, value, value_spatial_shapes, value_mask=None):
        """
        Args:
            query (Tensor): [bs, query_length, C]
            reference_points (Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area
            value (Tensor): [bs, value_length, C]
            value_spatial_shapes (Tensor): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            value_mask (Tensor): [bs, value_length], True for non-padding elements, False for padding elements

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """

        bs, Len_q = query.shape[:2]
        Len_v = value.shape[1]
        assert int(value_spatial_shapes.prod(1).sum()) == Len_v

        value = self.value_proj(value)  # nn.Linear(embed_dim, embed_dim)
        if value_mask is not None:
            value_mask = value_mask.astype(value.dtype).unsqueeze(-1)
            value *= value_mask

        value = value.reshape([bs, Len_v, self.num_heads, self.head_dim])
        sampling_offsets = self.sampling_offsets(query).reshape(
            [bs, Len_q, self.num_heads, self.num_levels, self.num_points, 2])

        attention_weights = self.attention_weights(query).reshape(
            [bs, Len_q, self.num_heads, self.num_levels * self.num_points])

        attention_weights = F.softmax(attention_weights, -1).reshape(
            [bs, Len_q, self.num_heads, self.num_levels, self.num_points])

        offset_normalizer = value_spatial_shapes.flip([1]).reshape(
            [1, 1, 1, self.num_levels, 1, 2])  # [1, 1, 1, 3, 1, 2]

        sampling_locations = reference_points.reshape([bs, Len_q, 1, self.num_levels, 1, 2]) \
                             + sampling_offsets / offset_normalizer

        output = deformable_attention_core_func(value, value_spatial_shapes, sampling_locations, attention_weights)

        output = self.output_proj(output)
        return output

class TransformerEncoderLayer(nn.Layer):
    def __init__(self, d_model=256, n_head=8, dim_feedforward=1024, dropout=0.1, activation="relu", n_levels=4,
                 n_points=4, weight_attr=None, bias_attr=None):
        super(TransformerEncoderLayer, self).__init__()
        # self attention
        self.self_attn = MSDeformableAttention(d_model, n_head, n_levels, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward, weight_attr, bias_attr)
        self.activation = getattr(F, activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, weight_attr, bias_attr)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        self.conv0 = nn.Sequential(
            nn.Conv2D(d_model, d_model, kernel_size=3, stride=1, padding=1, bias_attr=False),
            # nn.BatchNorm2D(d_model),
            # nn.ReLU())
            nn.GroupNorm(32, d_model),
            nn.GELU())

        self.conv1 = nn.Sequential(
            nn.Conv2D(d_model, d_model, kernel_size=3, stride=1, padding=1, bias_attr=False),
            # nn.BatchNorm2D(d_model),
            # nn.ReLU())
            nn.GroupNorm(32, d_model),
            nn.GELU())

        self.conv2 = nn.Sequential(
            nn.Conv2D(d_model, d_model, kernel_size=3, stride=1, padding=1, bias_attr=False),
            # nn.BatchNorm2D(d_model),
            # nn.ReLU())
            nn.GroupNorm(32, d_model),
            nn.GELU())

        self._reset_parameters()

    def _reset_parameters(self):
        linear_init_(self.linear1)
        linear_init_(self.linear2)
        xavier_uniform_(self.linear1.weight)
        xavier_uniform_(self.linear2.weight)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def seq2_2D(self, src, spatial_shapes):
        bs, hw, c = src.shape
        h = w = int(math.sqrt(hw))

        x0_index = spatial_shapes[0][0].numpy() * spatial_shapes[0][1].numpy()  # 32 * 32 = 1024
        x1_index = spatial_shapes[1][0].numpy() * spatial_shapes[1][1].numpy()  # 16 * 16 = 256
        x2_index = spatial_shapes[2][0].numpy() * spatial_shapes[2][1].numpy()  # 8 * 8 =64

        x0_index = x0_index[0]
        x1_index = x1_index[0]
        x2_index = x2_index[0]

        x0 = src[:, 0:x0_index].transpose([0, 2, 1]).reshape(
            [bs, c, spatial_shapes[0][0], spatial_shapes[0][1]])  # shape=[4, 256, 32, 32]
        x1 = src[:, x0_index:x0_index + x1_index].transpose([0, 2, 1]).reshape(
            [bs, c, spatial_shapes[1][0], spatial_shapes[1][1]])  # shape=[4, 256, 16, 16]
        x2 = src[:, x0_index + x1_index::].transpose([0, 2, 1]).reshape(
            [bs, c, spatial_shapes[2][0], spatial_shapes[2][1]])  # shape=[4, 256, 8, 8]

        return x0, x1, x2

    def forward(self, src, reference_points, spatial_shapes, src_mask=None, pos_embed=None):
        x0, x1, x2 = self.seq2_2D(src, spatial_shapes)

        src0 = self.conv0(x0) + x0
        src1 = self.conv1(x1) + x1
        src2 = self.conv2(x2) + x2

        src0 = src0.flatten(2).transpose([0, 2, 1])  # (bs,c,h,w) ->(bs,h*w,c)
        src1 = src1.flatten(2).transpose([0, 2, 1])  # (bs,c,h,w) ->(bs,h*w,c)
        src2 = src2.flatten(2).transpose([0, 2, 1])  # (bs,c,h,w) ->(bs,h*w,c)

        src_flatten = [src0, src1, src2]
        src_flatten = paddle.concat(src_flatten, 1)  # [4, 5376, 256]

        src2 = self.self_attn(self.with_pos_embed(src, pos_embed), reference_points, src, spatial_shapes, src_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # ffn
        src = self.forward_ffn(src)
        src = src + src_flatten
        return src


class TransformerEncoder(nn.Layer):
    def __init__(self, encoder_layer, num_layers):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios):
        valid_ratios = valid_ratios.unsqueeze(1)
        reference_points = []
        for i, (H, W) in enumerate(spatial_shapes.tolist()):
            ref_y, ref_x = paddle.meshgrid(
                paddle.linspace(0.5, H - 0.5, H),
                paddle.linspace(0.5, W - 0.5, W))

            ref_y = ref_y.flatten().unsqueeze(0) / (valid_ratios[:, :, i, 1] * H)
            ref_x = ref_x.flatten().unsqueeze(0) / (valid_ratios[:, :, i, 0] * W)
            reference_points.append(paddle.stack((ref_x, ref_y), axis=-1))

        reference_points = paddle.concat(reference_points, 1).unsqueeze(2)
        reference_points = reference_points * valid_ratios
        return reference_points

    def forward(self, src, spatial_shapes, src_mask=None, pos_embed=None, valid_ratios=None):
        output = src
        if valid_ratios is None:
            valid_ratios = paddle.ones([src.shape[0], spatial_shapes.shape[0], 2])  # [bs,num_features,2]

        reference_points = self.get_reference_points(spatial_shapes, valid_ratios)
        for layer in self.layers:
            output = layer(output, reference_points, spatial_shapes, src_mask, pos_embed)

        return output


class TransformerDecoderLayer(nn.Layer):
    def __init__(self, d_model=256, n_head=8, dim_feedforward=1024, dropout=0.1, activation="relu", n_levels=3,
                 n_points=4, weight_attr=None,
                 bias_attr=None):
        super(TransformerDecoderLayer, self).__init__()

        # self attention
        self.self_attn = MultiHeadAttention(d_model, n_head, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # cross attention
        self.cross_attn = MSDeformableAttention(d_model, n_head, n_levels, n_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward, weight_attr, bias_attr)
        self.activation = getattr(F, activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, weight_attr, bias_attr)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        linear_init_(self.linear1)
        linear_init_(self.linear2)
        xavier_uniform_(self.linear1.weight)
        xavier_uniform_(self.linear2.weight)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, reference_points, memory, memory_spatial_shapes, memory_mask=None, query_pos_embed=None):
        q = k = self.with_pos_embed(tgt, query_pos_embed)
        tgt2 = self.self_attn(q, k, value=tgt)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos_embed), reference_points, memory,
                               memory_spatial_shapes, memory_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)
        return tgt


class TransformerDecoder(nn.Layer):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, H, W):
        valid_ratios = valid_ratios.unsqueeze(1)
        reference_points = []
        ref_y, ref_x = paddle.meshgrid(paddle.linspace(0.5, H - 0.5, H),
                                       paddle.linspace(0.5, W - 0.5, W))
        ref_y = ref_y.flatten().unsqueeze(0) / H
        ref_x = ref_x.flatten().unsqueeze(0) / W
        ref = paddle.stack((ref_x, ref_y), axis=-1)

        reference_points = ref.unsqueeze(2)
        reference_points = reference_points * valid_ratios

        return reference_points

    def forward(self, tgt, memory, reference_points, memory_spatial_shapes, memory_mask=None, query_pos_embed=None,
                valid_ratios=None):

        output = tgt  # tgt: --> nn.Embedding(num_queries, hidden_dim)
        intermediate = []

        for lid, layer in enumerate(self.layers):
            output = layer(output, reference_points, memory, memory_spatial_shapes, memory_mask, query_pos_embed)
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return paddle.stack(intermediate)

        return output.unsqueeze(0)


class EncoderDecoder(nn.Layer):
    __shared__ = ['hidden_dim']

    def __init__(self, num_queries=110, position_embed_type='sine', return_intermediate_dec=False,
                 backbone_num_channels=[512, 1024, 2048], num_feature_levels=3, nclass=6,
                 num_encoder_points=4, num_decoder_points=4, hidden_dim=256, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=4, dim_feedforward=1024, dropout=0.1, activation="relu", lr_mult=0.1,
                 weight_attr=None, bias_attr=None):

        super(EncoderDecoder, self).__init__()
        assert position_embed_type in ['sine', 'learned'], \
            f'ValueError: position_embed_type not supported {position_embed_type}!'
        assert len(backbone_num_channels) <= num_feature_levels

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.num_feature_levels = num_feature_levels

        encoder_layer = TransformerEncoderLayer(hidden_dim, nhead, dim_feedforward, dropout, activation,
                                                          num_feature_levels, num_encoder_points, weight_attr,
                                                          bias_attr)

        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)


        decoder_layer = TransformerDecoderLayer(hidden_dim, nhead, dim_feedforward, dropout, activation,
                                                          num_feature_levels, num_decoder_points, weight_attr,
                                                          bias_attr)

        self.decoder =TransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)
        self.level_embed = nn.Embedding(num_feature_levels, hidden_dim)
        self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_embed = nn.Embedding(num_queries, hidden_dim)

        self.reference_points = nn.Linear(hidden_dim, 2, weight_attr=ParamAttr(learning_rate=lr_mult),
                                          bias_attr=ParamAttr(learning_rate=lr_mult))
        self.input_proj = nn.LayerList()
        for in_channels in backbone_num_channels:
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2D(in_channels, hidden_dim, kernel_size=1, weight_attr=weight_attr, bias_attr=bias_attr),
                    nn.GroupNorm(32, hidden_dim)))

        in_channels = backbone_num_channels[-1]
        for _ in range(num_feature_levels - len(backbone_num_channels)):
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2D(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1,
                              weight_attr=weight_attr, bias_attr=bias_attr),
                    nn.GroupNorm(32, hidden_dim)))
            in_channels = hidden_dim

        self.position_embedding = PositionEmbedding(hidden_dim // 2,
                                                    normalize=True if position_embed_type == 'sine' else False,
                                                    embed_type=position_embed_type, offset=-0.5)
        self._reset_parameters()

    def _reset_parameters(self):
        normal_(self.level_embed.weight)
        normal_(self.tgt_embed.weight)
        normal_(self.query_pos_embed.weight)
        xavier_uniform_(self.reference_points.weight)
        constant_(self.reference_points.bias)
        for l in self.input_proj:
            xavier_uniform_(l[0].weight)
            constant_(l[0].bias)

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'backbone_num_channels': [i.channels for i in input_shape], }

    def _get_valid_ratio(self, mask):
        mask = mask.astype(paddle.float32)
        _, H, W = mask.shape
        valid_ratio_h = paddle.sum(mask[:, :, 0], 1) / H
        valid_ratio_w = paddle.sum(mask[:, 0, :], 1) / W
        valid_ratio = paddle.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, src_feats, src_psp, src_mask=None):
        srcs = []
        for i in range(len(src_feats)):
            srcs.append(self.input_proj[i](src_feats[i]))

        if self.num_feature_levels > len(srcs):
            len_srcs = len(srcs)
            for i in range(len_srcs, self.num_feature_levels):
                if i == len_srcs:
                    srcs.append(self.input_proj[i](src_feats[-1]))
                else:
                    srcs.append(self.input_proj[i](srcs[-1]))

        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        valid_ratios = []
        for level, src in enumerate(srcs):
            bs, c, h, w = src.shape
            spatial_shapes.append([h, w])
            src = src.flatten(2).transpose([0, 2, 1])  # (bs,c,h,w) ->(bs,h*w,c)
            src_flatten.append(src)
            if src_mask is not None:
                mask = F.interpolate(src_mask.unsqueeze(0).astype(src.dtype), size=(h, w))[0].astype('bool')
            else:
                mask = paddle.ones([bs, h, w], dtype='bool')

            valid_ratios.append(self._get_valid_ratio(mask))

            # position_embedding = sine + cosine
            pos_embed = self.position_embedding(mask).flatten(2).transpose([0, 2, 1])  # (bs,c,h,w) ->(bs,h*w,c)
            lvl_pos_embed = pos_embed + self.level_embed.weight[level].reshape([1, 1, -1])  # (bs,h*w,c)+(1,1,256)

            lvl_pos_embed_flatten.append(lvl_pos_embed)
            mask = mask.astype(src.dtype).flatten(1)  # (bs,h,w)->(bs,h*w)
            mask_flatten.append(mask)

        src_flatten = paddle.concat(src_flatten, 1)  # [4, 5376, 256]
        mask_flatten = paddle.concat(mask_flatten, 1)
        lvl_pos_embed_flatten = paddle.concat(lvl_pos_embed_flatten, 1)  # position + level embed
        spatial_shapes = paddle.to_tensor(spatial_shapes, dtype='int64')

        valid_ratios = paddle.stack(valid_ratios, 1)

        memory = self.encoder(src_flatten, spatial_shapes, mask_flatten, lvl_pos_embed_flatten, valid_ratios)

        bs, _, c = memory.shape  # [8, 5376, 256]
        query_embed = self.query_pos_embed.weight.unsqueeze(0).tile([bs, 1, 1])

        reference_points = F.sigmoid(self.reference_points(query_embed))  # [4, 110, 256]
        reference_points_input = reference_points.unsqueeze(2) * valid_ratios.unsqueeze(1)

        src_psp = src_psp.transpose([0, 2, 1])
        hs = self.decoder(src_psp, memory, reference_points_input, spatial_shapes, mask_flatten, query_embed,
                          valid_ratios)

        return hs, memory
