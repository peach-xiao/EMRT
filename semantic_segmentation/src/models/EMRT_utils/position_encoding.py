"""
Positional encodings for the transformer.
"""
import math
import paddle
from paddle import nn
from typing import Optional

# Featuremap的位置编码，position_embedding 的前向如下：
# 利用三角函数的方式获取position_embedding，输入是NestedTensor={tensor,mask}, 输出最终pos的size为[1,2,256,7,8]
class PositionEmbeddingSine(nn.Layer):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        bs, c, h, w = x.shape
        mask = paddle.zeros(shape=[bs, h, w], dtype="bool").cuda()
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype='float32')
        x_embed = not_mask.cumsum(2, dtype='float32')
        if self.normalize:
            eps = 1e-6
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self.scale

        # num_pos_feats = 128
        ## 0~127 self.num_pos_feats=128,因为前面输入向量是256，编码是一半sin，一半cos
        dim_t = 2 * (paddle.arange(self.num_pos_feats) // 2).astype('float32')
        dim_t = self.temperature ** (dim_t / self.num_pos_feats)

        ## 输出shape=b,h,w,128
        # pos_x = x_embed[:, :, :, None] / dim_t
        # pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = x_embed.unsqueeze(-1) / dim_t
        pos_y = y_embed.unsqueeze(-1) / dim_t
        pos_x = paddle.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), axis=4).flatten(3)
        pos_y = paddle.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), axis=4).flatten(3)
        pos = paddle.concat((pos_y, pos_x), axis=3).transpose([0, 3, 1, 2])
        # 每个特征图的xy位置都编码成256的向量，其中前128是y方向编码，而128是x方向编码
        return pos
        ## b,n=256,h,w


def build_position_encoding(mode, hidden_dim):
    N_steps = hidden_dim // 2
    if mode in ('v2', 'sine'):
        position_embedding = PositionEmbeddingSine(num_pos_feats=N_steps, normalize=True)
    else:
        raise ValueError(f"not supported {mode}")

    return position_embedding
