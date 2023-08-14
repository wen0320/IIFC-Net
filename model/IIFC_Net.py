
import math
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pad_packed_sequence
import os
import numpy as np
import torch
import copy
from torch.nn import functional as F
from torch.nn.modules.module import Module
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.container import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.rnn import LSTM, GRU
from torch.nn.modules.normalization import LayerNorm
from functools import partial


class STFT(nn.Module):
    def __init__(self, n_fft=512, hop_length=100, window_length=400):
        super(STFT, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window_length = window_length

    def forward(self, x):
        x = x.squeeze(1)
        x = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.window_length,
                       return_complex=False)[:, :-1, :, :]
        c = x.permute(0, 3, 2, 1).contiguous()
        return c


class ISTFT(nn.Module):
    def __init__(self, n_fft=512, hop_length=100, window_length=400):
        super(ISTFT, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window_length = window_length
        self.pad = torch.nn.ZeroPad2d((0, 1, 0, 0))

    def forward(self, x):
        out = self.pad(x)
        out = out.permute(0, 3, 2, 1)  # [B,F,T,C]
        out = torch.istft(out, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.window_length,
                          return_complex=False)
        out = out.unsqueeze(1)
        return out


class SwitchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.9, using_moving_average=True, using_bn=True,
                 last_gamma=False):
        super(SwitchNorm2d, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.using_bn = using_bn
        self.last_gamma = last_gamma
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        if self.using_bn:
            self.mean_weight = nn.Parameter(torch.ones(3))
            self.var_weight = nn.Parameter(torch.ones(3))
        else:
            self.mean_weight = nn.Parameter(torch.ones(2))
            self.var_weight = nn.Parameter(torch.ones(2))
        if self.using_bn:
            self.register_buffer('running_mean', torch.zeros(1, num_features, 1))
            self.register_buffer('running_var', torch.zeros(1, num_features, 1))

        self.reset_parameters()

    def reset_parameters(self):
        if self.using_bn:
            self.running_mean.zero_()
            self.running_var.zero_()
        if self.last_gamma:
            self.weight.data.fill_(0)
        else:
            self.weight.data.fill_(1)
        self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        N, C, H, W = x.size()
        x = x.view(N, C, -1)
        mean_in = x.mean(-1, keepdim=True)
        var_in = x.var(-1, keepdim=True)

        mean_ln = mean_in.mean(1, keepdim=True)
        temp = var_in + mean_in ** 2
        var_ln = temp.mean(1, keepdim=True) - mean_ln ** 2

        if self.using_bn:
            if self.training:
                mean_bn = mean_in.mean(0, keepdim=True)
                var_bn = temp.mean(0, keepdim=True) - mean_bn ** 2
                if self.using_moving_average:
                    self.running_mean.mul_(self.momentum)
                    self.running_mean.add_((1 - self.momentum) * mean_bn.data)
                    self.running_var.mul_(self.momentum)
                    self.running_var.add_((1 - self.momentum) * var_bn.data)
                else:
                    self.running_mean.add_(mean_bn.data)
                    self.running_var.add_(mean_bn.data ** 2 + var_bn.data)
            else:
                mean_bn = torch.autograd.Variable(self.running_mean)
                var_bn = torch.autograd.Variable(self.running_var)

        softmax = nn.Softmax(0)
        mean_weight = softmax(self.mean_weight)
        var_weight = softmax(self.var_weight)

        if self.using_bn:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln + mean_weight[2] * mean_bn
            var = var_weight[0] * var_in + var_weight[1] * var_ln + var_weight[2] * var_bn
        else:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln
            var = var_weight[0] * var_in + var_weight[1] * var_ln

        x = (x - mean) / (var + self.eps).sqrt()
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias


def chunkwise(xs, N_l, N_c, N_r):
    """Slice input frames chunk by chunk.

    Args:
        xs (FloatTensor): `[B, T, input_dim]`
        N_l (int): number of frames for left context
        N_c (int): number of frames for current context
        N_r (int): number of frames for right context
    Returns:
        xs (FloatTensor): `[B * n_chunks, N_l + N_c + N_r, input_dim]`
            where n_chunks = ceil(T / N_c)
    """
    bs, xmax, idim = xs.size()
    n_chunks = math.ceil(xmax / N_c)
    c = N_l + N_c + N_r
    s_index = torch.arange(0, xmax, N_c).unsqueeze(-1)
    c_index = torch.arange(0, c)
    index = s_index + c_index  # (xmax,c)
    xs_pad = torch.cat([xs.new_zeros(bs, N_l, idim),
                        xs,
                        xs.new_zeros(bs, N_c * n_chunks - xmax + N_r, idim)], dim=1)  # B,C+T-1,D
    xs_chunk = xs_pad[:, index].contiguous().view(bs * n_chunks, N_l + N_c + N_r, idim)  # B*T,C,D
    return xs_chunk


class MHLocalDenseSynthesizerAttention(nn.Module):
    """Multi-Head Local Dense Synthesizer attention layer
    In this implementation, the calculation of multi-head mechanism is similar to that of self-attention,
    but it takes more time for training. We provide an alternative multi-head mechanism implementation
    that can achieve competitive results with less time.

    :param int n_head: the number of heads
    :param int n_feat: the dimension of features
    :param float dropout_rate: dropout rate
    :param int context_size: context size
    :param bool use_bias: use bias term in linear layers
    """

    def __init__(self, n_head, n_feat, dropout_rate, context_size=3, use_bias=False):
        super().__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.c = context_size
        self.w1 = nn.Linear(n_feat, n_feat, bias=use_bias)
        # self.w2 = nn.Linear(n_feat, n_head * self.c, bias=use_bias)
        self.w2 = nn.Conv1d(in_channels=n_feat, out_channels=n_head * self.c, kernel_size=1,
                            groups=n_head)
        self.w3 = nn.Linear(n_feat, n_feat, bias=use_bias)
        self.w_out = nn.Linear(n_feat, n_feat, bias=use_bias)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, query, key, value, mask):
        """Forward pass.

                :param torch.Tensor query: (batch, time, size)
                :param torch.Tensor key: (batch, time, size) dummy
                :param torch.Tensor value: (batch, time, size)
                :param torch.Tensor mask: (batch, time, time) dummy
                :return torch.Tensor: attentioned and transformed `value` (batch, time, d_model)
                """
        bs, time = query.size()[: 2]
        query = self.w1(query)  # [B, T, d]
        # [B, T, d] --> [B, d, T] --> [B, H*c, T]
        weight = self.w2(torch.relu(query).transpose(1, 2))
        # [B, H, c, T] --> [B, T, H, c] --> [B*T, H, 1, c]
        weight = weight.view(bs, self.h, self.c, time).permute(0, 3, 1, 2) \
            .contiguous().view(bs * time, self.h, 1, self.c)
        value = self.w3(value)  # [B, T, d]
        # [B*T, c, d] --> [B*T, c, H, d_k] --> [B*T, H, c, d_k]
        value_cw = chunkwise(value, (self.c - 1) // 2, 1, (self.c - 1) // 2) \
            .view(bs * time, self.c, self.h, self.d_k).transpose(1, 2)
        self.attn = torch.softmax(weight, dim=-1)
        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, value_cw)
        x = x.contiguous().view(bs, -1, self.h * self.d_k)  # [B, T, d]
        x = self.w_out(x)  # [B, T, d]
        return x


class DPH(nn.Module):
    """
    Deep duaL-path RNN.
    args:
        rnn_type: string, select from 'RNN', 'LSTM' and 'GRU'.
        input_size: int, dimension of the input feature. The input should have shape
                    (batch, seq_len, input_size).
        hidden_size: int, dimension of the hidden state.
        output_size: int, dimension of the output size.
        dropout: float, dropout ratio. Default is 0.
        num_layers: int, number of stacked RNN layers. Default is 1.
        bidirectional: bool, whether the RNN layers are bidirectional. Default is False.
    """

    def __init__(self, input_size, output_size, dropout=0, num_layers=1):
        super(DPH, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.input = nn.Sequential(
            nn.Conv2d(input_size, input_size // 2, kernel_size=1),
            nn.PReLU()
        )

        # dual-path RNN
        self.row_trans = nn.ModuleList([])  # local
        self.col_trans = nn.ModuleList([])  # global
        self.row_norm = nn.ModuleList([])
        self.col_norm = nn.ModuleList([])
        for i in range(num_layers):
            self.row_trans.append(
                HOIIFormer(d_model=input_size // 2, nhead=4, dropout=dropout, bidirectional=True,
                                               gnconv_dim=input_size // 2, gnconv_order=i+2, T=True))
            self.col_trans.append(
                HOIIFormer(d_model=input_size // 2, nhead=4, dropout=dropout, bidirectional=True,
                                               gnconv_dim=input_size // 2, gnconv_order=i+2, T=False))
            self.row_norm.append(nn.GroupNorm(1, input_size // 2, eps=1e-8))  # 将input_size//2放入一组
            self.col_norm.append(nn.GroupNorm(1, input_size // 2, eps=1e-8))

        # output layer
        self.output = nn.Sequential(nn.PReLU(),
                                    nn.Conv2d(input_size // 2, output_size, 1)  # inchannels=32 , outchannels=32
                                    )

    def forward(self, input):
        #  input --- [b,  c,  num_frames, frame_size]  --- [b, c, dim2, dim1]
        b, c, dim2, dim1 = input.shape
        output = self.input(input)
        for i in range(len(self.row_trans)):
            row_output = self.row_trans[i](output)  # [dim1, b*dim2, c]
            row_output = row_output.view(dim1, b, dim2, -1).permute(1, 3, 2, 0).contiguous()  # [b, c, dim2, dim1]
            row_output = self.row_norm[i](row_output)  # [b, c, dim2, dim1]
            output = output + row_output  # [b, c, dim2, dim1]

            col_output = self.col_trans[i](output)  # [dim2, b*dim1, c]
            col_output = col_output.view(dim2, b, dim1, -1).permute(1, 3, 0, 2).contiguous()  # [b, c, dim2, dim1]
            col_output = self.col_norm[i](col_output)  # [b, c, dim2, dim1]
            output = output + col_output  # [b, c, dim2, dim1]

        del row_output, col_output
        output = self.output(output)  # [b, c, dim2, dim1]

        return output

def get_dwconv(dim, kernel, bias):
    return nn.Conv2d(dim, dim, kernel_size=kernel, padding=(kernel - 1) // 2, bias=bias, groups=dim)

class HOII(nn.Module):
    def __init__(self, dim, order=5, s=1.0):
        super().__init__()
        self.order = order
        self.dims = [dim // 2 ** i for i in range(order)]
        self.dims.reverse()
        self.proj_in = nn.Conv2d(dim, 2 * dim, 1)

        self.dwconv = get_dwconv(sum(self.dims), 7, True)
        self.proj_out = nn.Conv2d(dim, dim, 1)

        self.pws = nn.ModuleList(
            [nn.Conv2d(self.dims[i], self.dims[i + 1], 1) for i in range(order - 1)]
        )

        self.scale = s

    def forward(self, x):  # [2,32,641,128]

        fused_x = self.proj_in(x)  # [2,64,641,128]

        pwa, abc = torch.split(fused_x, (self.dims[0], sum(self.dims)), dim=1)  # [2,16,641,128], [2,48,641,128]

        dw_abc = self.dwconv(abc) * self.scale  # [2,48,641,128]

        dw_list = torch.split(dw_abc, self.dims, dim=1)
        x = pwa * dw_list[0]
        for i in range(self.order - 1):
            x = self.pws[i](x) * dw_list[i + 1]

        x = self.proj_out(x)

        return x

class HOIIFormer(Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead, bidirectional=True, dropout=0, activation="relu", gnconv_dim=32, gnconv_order=2,rate=1, T=True):
        super(HOIIFormer, self).__init__()
        self.HOII = HOII(dim=gnconv_dim, order=gnconv_order, s=1.0 / 3.0)
        self.T = T

        # Implementation of Feedforward model
        self.gru = GRU(d_model, d_model*2, 1, bidirectional=bidirectional)  # orginal
        self.dropout = Dropout(dropout)

        if bidirectional:
            self.linear2 = Linear(d_model*2*2, d_model)
        else:
            self.linear2 = Linear(d_model * 2, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        self.mhldsa = MHLocalDenseSynthesizerAttention(nhead, d_model, dropout_rate=dropout, context_size=3)
        self.norm3 = LayerNorm(d_model)
        self.dropout3 = Dropout(dropout)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(HOIIFormer, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequnce to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """


        b, c, dim2, dim1 = src.shape

        if self.T:
            src2 = self.HOII(src)
            src = src + self.dropout1(src2)
            src = src.permute(3, 0, 2, 1).contiguous().view(dim1, b * dim2, -1)
        else:
            src = src.permute(0, 1, 3, 2).contiguous()
            src2 = self.HOII(src)
            src = src + self.dropout1(src2)
            src = src.permute(0, 1, 3, 2).contiguous()
            src = src.permute(2, 0, 3, 1).contiguous().view(dim2, b * dim1, -1)  # [dim2, b*dim1, c]

        src = self.norm1(src)

        src3 = self.mhldsa(src, src, src, mask=src_mask)
        src = src + self.dropout3(src3)
        src = self.norm3(src)
        # orginal gru
        self.gru.flatten_parameters()
        out, h_n = self.gru(src)  # [641,256,128]
        del h_n

        src4 = self.linear2(self.dropout(self.activation(out)))
        src = src + self.dropout2(src4)
        src = self.norm2(src)
        return src


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


class FSFB(nn.Module):
    def __init__(self, N):
        super(FSFB, self).__init__()
        # Hyper-parameter
        self.N = N
        self.linear3 = nn.Conv2d(2 * N, N, kernel_size=(1, 1), bias=False)

    def forward(self, stft_feature, conv_feature):
        fusion_feature = self.linear3(torch.cat([stft_feature, conv_feature], dim=1))
        ratio_mask1 = torch.sigmoid(fusion_feature)
        ratio_mask2 = 1 - ratio_mask1
        conv_out = conv_feature * ratio_mask1
        stft_out = stft_feature * ratio_mask2
        fusion_out = conv_out + stft_out
        out = F.relu(stft_feature + conv_feature + fusion_out)

        return out


class DepthwiseSeparable(nn.Module):
    """
    Depthwise Separable Convolution is Depthwise Convolution + Pointwise Convolution.
        Depthwise Convolution : Convolution over each channel independently
            Divide input channels into "in_channel" groups and then apply convolution over each
            Group independently : Depth is not used
        Pointwise Convolution : Normal Convolution with kernel Size (1,1)
            Only depth Used.
    In Xception Architecture the Order of operation is different:
        Pointwise Convolution + Depthwise Convolution
    groups : No of groups the input channel should be divided into
             For depthwise convolution = in_channel
    padding = default: "same" (1 for kernel_size = 3)
    """

    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=1):
        super(DepthwiseSeparable, self).__init__()

        self.pointwise = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1)
        self.depthwise = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=out_channel)

    def forward(self, x):
        x = self.pointwise(x)
        x = self.depthwise(x)
        # x = self.pointwise(x)

        return x

# Convlution-Unit
class CU33(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(CU33, self).__init__()
        self.DWConv = DepthwiseSeparable(inchannel, outchannel, (3, 3), padding=1)
        # self.conv3x3 = nn.Conv2d(inchannel, outchannel, (3, 3), padding=1)
        self.sn = SwitchNorm2d(outchannel)
        self.act1 = nn.PReLU()
        self.shortcut = nn.Conv2d(inchannel, outchannel, (1, 1), bias=False)
        self.pointwise = nn.Conv2d(inchannel, outchannel, (1, 1))
        self.act2 = nn.Sequential(
            SwitchNorm2d(outchannel),
            nn.PReLU())

    def forward(self, x):
        out3x3 = self.DWConv(x)
        identity = self.shortcut(x)
        identity2 = self.pointwise(x)
        out = self.act2(out3x3 + identity + identity2)
        return out


class CU55(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(CU55, self).__init__()
        self.DWConv = DepthwiseSeparable(inchannel, outchannel, (5, 5), padding=2)
        # self.conv3x3 = nn.Conv2d(inchannel, outchannel, (5, 5), padding=2)
        self.sn = SwitchNorm2d(outchannel)
        self.act1 = nn.PReLU()
        self.shortcut = nn.Conv2d(inchannel, outchannel, (1, 1), bias=False)
        self.pointwise = nn.Conv2d(inchannel, outchannel, (1, 1))
        self.act2 = nn.Sequential(
            SwitchNorm2d(outchannel),
            nn.PReLU())

    def forward(self, x):
        out5x5 = self.DWConv(x)
        # out = self.act1(self.sn(out3x3))
        identity = self.shortcut(x)
        identity2 = self.pointwise(x)
        out = self.act2(out5x5 + identity + identity2)
        return out


class MSFE(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(MSFE, self).__init__()
        self.cu33 = CU33(inchannel=inchannel, outchannel=outchannel)
        self.cu55 = CU55(inchannel=inchannel, outchannel=outchannel)
        self.fsfb = FSFB(outchannel)
        # self.sf = AFF_Im(channels=outchannel)
        self.norm = SwitchNorm2d(outchannel)
        self.prelu = nn.PReLU(outchannel)

    def forward(self, x):
        x_33 = self.cu33(x)
        x_55 = self.cu55(x)
        out = self.fsfb(x_33, x_55)
        out = self.prelu(self.norm(out))
        return out


class DownSampleLayer(nn.Module):
    def __init__(self, channel):
        super(DownSampleLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=(1, 3), stride=(1, 2),
                              padding=(0, 1))
        self.norm = SwitchNorm2d(channel)
        self.prelu = nn.PReLU(channel)

    def forward(self, x):
        return self.prelu(self.norm(self.conv(x)))


class UpSampleLayer(nn.Module):
    def __init__(self, channel):
        super(UpSampleLayer, self).__init__()
        self.conv = SPConvTranspose2d(in_channels=channel, out_channels=channel, kernel_size=(1, 1), r=2)
        self.norm = SwitchNorm2d(channel)
        self.prelu = nn.PReLU(channel)

    def forward(self, x):
        return self.prelu(self.norm(self.conv(x)))


class SPConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, r=1):
        # upconvolution only along second dimension of image
        # Upsampling using sub pixel layers
        super(SPConvTranspose2d, self).__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels * r, kernel_size=kernel_size, stride=(1, 1))
        self.r = r

    def forward(self, x):
        out = self.conv(x)
        batch_size, nchannels, H, W = out.shape
        out = out.view((batch_size, self.r, nchannels // self.r, H, W))
        out = out.permute(0, 2, 3, 4, 1)
        out = out.contiguous().view((batch_size, nchannels // self.r, H, -1))
        return out

def GlobLN(nOut):
    return nn.GroupNorm(1, nOut, eps=1e-8)


class ConvNorm(nn.Module):
    """
    This class defines the convolution layer with normalization and PReLU activation
    """

    def __init__(self, nIn, nOut, kSize, stride=1, groups=1, bias=True):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        """
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(
            nIn, nOut, kSize, stride=stride, padding=padding, bias=bias, groups=groups
        )
        self.norm = GlobLN(nOut)
        # self.act = nn.PReLU(nOut)

    def forward(self, input):
        output = self.conv(input)
        output = self.norm(output)
        return output


class FC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, groups=1):
        super(FC, self).__init__()
        self.Conv1 = ConvNorm(in_channels, out_channels, kernel_size, groups=groups, bias=True)
        self.act = nn.Sigmoid()
        self.upsample = UpSampleLayer(out_channels)

    def forward(self, x_r, x):
        x1 = self.act(self.Conv1(x))
        out = x_r * x1
        out = self.upsample(out)
        return out


class FCi(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, groups=1):
        super(FCi, self).__init__()
        self.Conv1 = ConvNorm(in_channels, out_channels, kernel_size, groups=groups, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x_r, x):
        x1 = self.act(self.Conv1(x))
        out = x_r * x1
        return out


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.endbconv1 = MSFE(2, 16)
        self.endbconv2 = MSFE(16, 32)
        self.endbconv3 = MSFE(32, 48)
        self.endbconv = MSFE(48, 64)
        self.dsl3 = DownSampleLayer(64)
        self.DPH = DPH(input_size=64, output_size=64, num_layers=4)
        self.dedbconv = MSFE(128, 48)
        self.dedbconv1 = MSFE(96, 32)
        self.dedbconv2 = MSFE(64, 16)
        self.dedbconv3 = MSFE(32, 2)

        self.Up = UpSampleLayer(64)
        self.FC4 = FC(in_channels=64, out_channels=64, kernel_size=3, groups=64)
        self.FC3 = FCi(in_channels=48, out_channels=48, kernel_size=3, groups=48)
        self.FC2 = FCi(in_channels=32, out_channels=32, kernel_size=3, groups=32)
        self.FC1 = FCi(in_channels=16, out_channels=16, kernel_size=3, groups=16)
        self.down_C1 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=48,kernel_size=1),
            nn.GroupNorm(1, 48, eps=1e-8)
        )
        self.down_C2 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=32,kernel_size=1),
            nn.GroupNorm(1, 32, eps=1e-8)
        )
        self.down_C3 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=16,kernel_size=1),
            nn.GroupNorm(1, 16, eps=1e-8)
        )

        self.stft = STFT()
        self.istft = ISTFT()

        #init
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)


    def forward(self, x):
        x = self.stft(x)  # [1,2,321,256]
        x1 = self.endbconv1(x)  # [1,16,641,256]
        x2 = self.endbconv2(x1)  # [1,32,641,256]
        x3 = self.endbconv3(x2)
        x4 = self.dsl3(self.endbconv(x3))  # [1,64,641,128]
        x5 = self.DPH(x4)  # [1,64,641,128]
        x_dpt = self.Up(x5)
        x6 = self.dedbconv(torch.cat((self.FC4(x4, x5), x_dpt), dim=1))  # [2,48,321,128]
        x7 = self.dedbconv1(torch.cat((self.FC3(x3, self.down_C1(x_dpt)), x6), dim=1))
        x8 = self.dedbconv2(torch.cat((self.FC2(x2, self.down_C2(x_dpt)), x7), dim=1)) # [2,16,321,256]
        x9 = self.dedbconv3(torch.cat((self.FC1(x1, self.down_C3(x_dpt)), x8), dim=1))  # [2,2,321,256]
        out = self.istft(x9)
        return out



if __name__ == '__main__':
    x = torch.randn(1, 1, 16000)
    model = Net()
    out2 = model(x)
    print(out2.shape)

    # test Calculate the force
    from ptflops.flops_counter import get_model_complexity_info
    flops2, params2 = get_model_complexity_info(model, (1, 16000), print_per_layer_stat=True)
    print("%s %s" % (flops2, params2))



