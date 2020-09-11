import torch
from torch import nn
from torch.nn import functional as F

import distributed as dist_fn


# Copyright 2018 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================


# Borrowed from https://github.com/deepmind/sonnet and ported it to PyTorch


class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim  # 量化空间中每个张量的维数，一般与encoder的最后一层输出的channel数相同，假设为64
        self.n_embed = n_embed  # 量化空间中向量的个数, 假设为512
        self.decay = decay  # 移动平均线的衰减
        self.eps = eps

        embed = torch.randn(dim, n_embed)  # (64, 512)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))  # 512个向量均初始化为0，作为N^t,t=0
        self.register_buffer("embed_avg", embed.clone())  # 512个向量均初始化为0，作为m^t,t=0

    def forward(self, input):
        flatten = input.reshape(-1, self.dim)  # (WxH, 64)

        # 计算这些编码（H * W）与 Embedding 中 K 个矢量（K=512，表示矢量量化编码的矢量个数，每个矢量为dim维)之间的距离，这里用的是欧氏距离的平方
        dist = (  # dist = (WxH, 1) - (WxH, 512) + (1, 512) = (WxH, 512)
            flatten.pow(2).sum(1, keepdim=True)  # 在第1维上缩减，横向压缩，keepdim=True时，输出与输入维度相同，只是第1维依然被压缩维1，所以最后的维度为(WxH, 1)
            - 2 * flatten @ self.embed  # @是一个操作符，表示矩阵-向量乘法
            + self.embed.pow(2).sum(0, keepdim=True)  # 在第0维上缩减，纵向压缩，keepdim=True时，输出与输入维度相同，只是第0维依然被压缩维1，所以最后的维度为(1, 512)
        )
        _, embed_ind = (-dist).max(1)  # 比较同为第0维度上不同第1维度值的大小，即横向压缩，返回(WxH,)的WxH个最大值，(WxH,)的最大值的第1维度坐标
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)  # 返回(WxH, 512)的onehot矩阵，onehot的第0维表示编码个数(类似于样本个数)，第1维表示矢量量化编码的矢量个数
        embed_ind = embed_ind.view(*input.shape[:-1])  # 把原先tensor中的数据按照行优先的顺序排成一个一维的数据，长度为WxH
        quantize = self.embed_code(embed_ind)  # (W, H, 64)，矢量化输出后，大小与flatten相同

        if self.training:

            # 这里和原版的vqvae的第二项损失不同，使用的是码本的指数移动平均（Exponential Moving Average，也叫权重移动平均Weighted Moving Average）更新,作为码本损失的替代，
            # EMA本身不做梯度传播，但其更新过程作用与梯度传播相似，可通过公式证明，因此网络中的损失只剩第一项reconstruction loss(在train.py中)和第三项commitment loss
            embed_onehot_sum = embed_onehot.sum(0)  # (512,)，纵向压缩，即在将每一个向量上的WXH个值做onehot个数累加，即计算512个向量在WXH个编码的各个编码中出现的个数

            # 这里乘embed_onehot是计算512个向量每个向量下64个维度被激活的位置(hot个数)做一个累加(累加的最大个数为WXH)，
            # 最后得到在WXH个样本里512个向量里各个维度出现的总值，通过除cluster_size即，512个向量在WXH个编码的各个编码中出现的个数(每个64维的向量出现的个数)，来做平均
            embed_sum = flatten.transpose(0, 1) @ embed_onehot  # (64, WxH) dot (WxH, 512)= (64, 512)；

            dist_fn.all_reduce(embed_onehot_sum)
            dist_fn.all_reduce(embed_sum)

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay  # (512)
            )
            self.embed_avg.data.mul_(self.decay).add_(
                embed_sum, alpha=1 - self.decay)  # (64, 512)

            # 对cluster_size作拉普拉斯平滑
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n  # (512)
            )

            # 更新后的e_i
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)  # (64, 512) / (512) = (64, 512)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean()  # .detach(): 这里就是原文中的stop gradient。diff是损失函数的第三项

        # bp的时候z_q_x(quantize)的梯度直接copy给z_e_x(inputs), 而不给codebook里的embedding,
        # quantize是decoder的输入，但quantize又可以是inputs的一个加减法映射，这里是不让quantize的值做梯度传播(因为会传播给embedding)，
        # 但inputs中不属于quantize的还是要传播的，所以要将quantize=inputs-(inputs-quantize)
        quantize = input + (quantize - input).detach()  # 参与第一项损失

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        '''
        这里的embedding操作为：
        先将embed的两个维度交换，得到(512, 64)的embed表
        输入进来的embed_id维度为(1, CxB)，代表CxB个编码分别在embed表中的矢量id，即每一个编码与embed表的512个矢量（每个矢量长度为64）中距离最近的一个矢量的下标位置id
        再根据这CxB个id在embed表的512个矢量里找对应位置的值，即(1,64)的数据，组合起来一共为(CxB, 64)
        '''
        return F.embedding(embed_id, self.embed.transpose(0, 1))  # embed的两个维度交换，这里与numpy的transpose不同


class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out


class Encoder(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()

        if stride == 4:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
            ]

        elif stride == 2:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 3, padding=1),
            ]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class Decoder(nn.Module):
    def __init__(
        self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride
    ):
        super().__init__()

        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        if stride == 4:
            blocks.extend(
                [
                    nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        channel // 2, out_channel, 4, stride=2, padding=1
                    ),
                ]
            )

        elif stride == 2:
            blocks.append(
                nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1)
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class VQVAE(nn.Module):
    def __init__(
        self,
        in_channel=3,
        channel=128,
        n_res_block=2,
        n_res_channel=32,
        embed_dim=64,
        n_embed=512,
        decay=0.99,
    ):
        super().__init__()

        self.enc_b = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=4)
        self.enc_t = Encoder(channel, channel, n_res_block, n_res_channel, stride=2)
        self.quantize_conv_t = nn.Conv2d(channel, embed_dim, 1)
        self.quantize_t = Quantize(embed_dim, n_embed)
        self.dec_t = Decoder(
            embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=2
        )
        self.quantize_conv_b = nn.Conv2d(embed_dim + channel, embed_dim, 1)
        self.quantize_b = Quantize(embed_dim, n_embed)
        self.upsample_t = nn.ConvTranspose2d(
            embed_dim, embed_dim, 4, stride=2, padding=1
        )
        self.dec = Decoder(
            embed_dim + embed_dim,
            in_channel,
            channel,
            n_res_block,
            n_res_channel,
            stride=4,
        )

    def forward(self, input):
        quant_t, quant_b, diff, _, _ = self.encode(input)
        dec = self.decode(quant_t, quant_b)

        return dec, diff

    def encode(self, input):
        enc_b = self.enc_b(input)
        enc_t = self.enc_t(enc_b)

        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1)
        quant_t, diff_t, id_t = self.quantize_t(quant_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        diff_t = diff_t.unsqueeze(0)

        dec_t = self.dec_t(quant_t)
        enc_b = torch.cat([dec_t, enc_b], 1)

        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)
        quant_b, diff_b, id_b = self.quantize_b(quant_b)
        quant_b = quant_b.permute(0, 3, 1, 2)
        diff_b = diff_b.unsqueeze(0)

        return quant_t, quant_b, diff_t + diff_b, id_t, id_b

    def decode(self, quant_t, quant_b):
        upsample_t = self.upsample_t(quant_t)
        quant = torch.cat([upsample_t, quant_b], 1)
        dec = self.dec(quant)

        return dec

    def decode_code(self, code_t, code_b):
        quant_t = self.quantize_t.embed_code(code_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        quant_b = self.quantize_b.embed_code(code_b)
        quant_b = quant_b.permute(0, 3, 1, 2)

        dec = self.decode(quant_t, quant_b)

        return dec
