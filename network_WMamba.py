import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
# from models.vim.models_mamba import VisionMamba
from mamba_simple import Mamba
from einops import rearrange
import numbers

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, C,H, W = x.shape
    x = x.view(B, C,H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 1, 2, 4, 3, 5).contiguous().view(-1,C, window_size, window_size)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, C, H, W)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B,-1, H // window_size, W // window_size, window_size, window_size)
    x = x.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, -1,H, W)
    return x

class LayerNorm_GASU(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.num_feat * 3 * 9
        return flops

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 0)

def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    out_batch, out_channel, out_height, out_width = int(in_batch/(r**2)),in_channel, r * in_height, r * in_width
    #提取四种x分量
    x1 = x[0:out_batch, :, :] / 2
    x2 = x[out_batch:out_batch * 2, :, :, :] / 2
    x3 = x[out_batch * 2:out_batch * 3, :, :, :] / 2
    x4 = x[out_batch * 3:out_batch * 4, :, :, :] / 2
    h = torch.zeros([out_batch, out_channel, out_height,
                     out_width]).float().to(x.device)

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h

def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)

class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)


class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)

class WMB(nn.Module):
    def __init__(self, channel,H,W,LayerNorm_type='WithBias'):
        super(WMB, self).__init__()
        self.DWT = DWT()
        self.IWT = IWT()
        self.norm1 = LayerNorm(channel, LayerNorm_type)
        self.process1 = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channel, channel, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channel, channel, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True))
        self.norm2 = LayerNorm(channel, LayerNorm_type)
        self.GSAU = GSAU(channel)
        self.VisionMamba1 = SingleMambaBlock(channel, H, W)

    def forward(self, x):
        n, c, h, w = x.shape
        shortcut = x
        x_norm1 = self.norm1(x)
        input_dwt = self.DWT(data_transform(x_norm1))
        input_LL, input_high = input_dwt[:n, ...], input_dwt[n:, ...]

        B, C, H, W = input_high.shape
        high_rearrange = rearrange(input_high, 'b c h w -> b (h w) c', h=H, w=W)
        high_mamba = self.VisionMamba1(high_rearrange)
        output_high = high_mamba.view(B, C, H, W)

        output_LL = self.process1(input_LL)
        x_w = inverse_data_transform(self.IWT(torch.cat((output_LL, output_high), dim=0)))
        x_res = x_w+shortcut
        shortcut_2 = x_res
        output = shortcut_2 + self.GSAU(self.norm2(x_res))

        return output

class GSAU(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        i_feats = n_feats * 2

        self.Conv1 = nn.Conv2d(n_feats, i_feats, 1, 1, 0)
        self.DWConv1 = nn.Conv2d(n_feats, n_feats, 7, 1, 7 // 2, groups=n_feats)
        self.Conv2 = nn.Conv2d(n_feats, n_feats, 1, 1, 0)

        self.norm = LayerNorm_GASU(n_feats, data_format='channels_first')
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)

    def forward(self, x):
        shortcut = x.clone()

        x = self.Conv1(self.norm(x))
        a, x = torch.chunk(x, 2, dim=1)
        x = x * self.DWConv1(a)
        x = self.Conv2(x)

        return x * self.scale + shortcut

class SingleMambaBlock(nn.Module):
    def __init__(self, dim, H, W):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.block = Mamba(dim, expand=1, d_state=8, bimamba_type='v6',
                           if_devide_out=True, use_norm=True, input_h=H, input_w=W)

    def forward(self, input):
        # input: (B, N, C)
        skip = input
        input = self.norm(input)
        output = self.block(input)
        return output + skip

class Mamba_Block(nn.Module):
    def __init__(self, channel,H,W, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.WMB = WMB(channel,H,W)

    def forward(self, x):
        output = self.WMB(x)
        return output



class WMamba(nn.Module):
    def __init__(self, patch_size=16, in_chans=1,embed_dim=192):
        super(WMamba, self).__init__()
        self.patch_size = patch_size
        embed_dim_temp = int(embed_dim / 2)
        if in_chans == 3 or in_chans == 6:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            rgbrgb_mean = (0.4488, 0.4371, 0.4040, 0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
            self.mean_in = torch.Tensor(rgbrgb_mean).view(1, 6, 1, 1)
        else:
            pass

        #####################################################################################################
        ################################### 1, shallow feature extraction ###################################
        self.conv_first1_A = nn.Conv2d(in_chans, embed_dim_temp, 3, 2, 1)
        self.conv_first1_B = nn.Conv2d(in_chans, embed_dim_temp, 3, 2, 1)
        self.conv_first2_A = nn.Conv2d(embed_dim_temp, embed_dim, 3, 1, 1)
        self.conv_first2_B = nn.Conv2d(embed_dim_temp, embed_dim, 3, 1, 1)
        self.conv_first3_A = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.conv_first3_B = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        N=192
        H=64//2
        W=64//2
        self.mamba_A1 = Mamba_Block(channel=N,H=H,W=W)
        self.mamba_B1 = Mamba_Block(channel=N,H=H,W=W)
        self.mamba_A2 = Mamba_Block(channel=N,H=H,W=W)
        self.mamba_B2 = Mamba_Block(channel=N,H=H,W=W)
        self.mamba_AB1 = Mamba_Block(channel=N,H=H,W=W)
        self.mamba_AB2 = Mamba_Block(channel=N,H=H,W=W)
        self.mamba_BA1 = Mamba_Block(channel=N,H=H,W=W)
        self.mamba_BA2 = Mamba_Block(channel=N,H=H,W=W)

        self.group_norm1 = nn.GroupNorm(num_groups=32, num_channels=embed_dim)
        self.group_norm2 = nn.GroupNorm(num_groups=32, num_channels=embed_dim)
        self.grop_norm_re = nn.GroupNorm(num_groups=32, num_channels=64)
        self.relu = nn.ReLU()

        self.upsample = UpsampleOneStep(scale = 2, num_feat=2*N,num_out_ch=2*N)
        self.conv_re_1 = nn.Conv2d(2*N,N,3,1,1)
        self.conv_re_2 = nn.Conv2d(N,64,3,1,1)
        self.conv_re_3 = nn.Conv2d(64,1,3,1,1)
        self.conv_cat_A = nn.Conv2d(2*N,N,1,1)
        self.conv_cat_B = nn.Conv2d(2 * N, N, 1, 1)

        self.sigmoid_final = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, A, B):  # A:[B,1,H,W]
        x_A = self.lrelu(self.conv_first1_A(A))
        x_A_2 = self.lrelu(self.conv_first2_A(x_A)) # (B,C,H,W)->(B,192,H/2,W/2)
        x_A_temp = self.lrelu(self.conv_first3_A(x_A_2))
        x_B = self.lrelu(self.conv_first1_B(B))
        x_B_2 = self.lrelu(self.conv_first2_B(x_B)) # (B,C,H,W)->(B,192,H/2,W/2)
        x_B_temp = self.lrelu(self.conv_first3_B(x_B_2))

        x_A_Mamba_1 = self.mamba_A1(x_A_temp)
        x_A_Mamba_2 = self.mamba_A2(x_A_Mamba_1)

        x_B_Mamba_1 = self.mamba_B1(x_B_temp)
        x_B_Mamba_2 = self.mamba_B2(x_B_Mamba_1)

        x_A_Mamba_3=x_A_Mamba_2+x_A_temp  #(B,192,H/2,W/2)
        x_B_Mamba_3=x_B_Mamba_2+x_B_temp  #(B,192,H/2,W/2)
        x_A_Mamba_3 = self.group_norm1(x_A_Mamba_3)
        x_A_Mamba_3 = self.relu(x_A_Mamba_3)
        x_B_Mamba_3 = self.group_norm1(x_B_Mamba_3)
        x_B_Mamba_3 = self.relu(x_B_Mamba_3)

        x_A_Mamba_3_temp = x_A_Mamba_3
        x_B_Mamba_3_temp = x_B_Mamba_3

        x_AB_conv = torch.cat([x_A_Mamba_3, x_B_Mamba_3], dim=1)
        x_AB_conv = self.conv_cat_A(x_AB_conv)
        x_BA_conv = torch.cat([x_B_Mamba_3, x_A_Mamba_3], dim=1)
        x_BA_conv = self.conv_cat_B(x_BA_conv)

        x_AB_1 = self.mamba_AB1(x_AB_conv)
        x_BA_1 = self.mamba_BA1(x_BA_conv)

        x_AB_2 = self.mamba_AB2(x_AB_1)
        x_BA_2 = self.mamba_BA2(x_BA_1)

        x_AB_4=x_AB_2+x_A_Mamba_3_temp
        x_BA_4=x_BA_2+x_B_Mamba_3_temp
        x_AB_4 = self.group_norm2(x_AB_4)
        x_AB_4 = self.relu(x_AB_4)
        x_BA_4 = self.group_norm2(x_BA_4)
        x_BA_4 = self.relu(x_BA_4)

        #[B,192,H/2,W/2]->[B,384,H/2,W/2]
        ABBA = torch.cat([x_AB_4,x_BA_4],dim=1)
        ABBA = self.upsample(ABBA)
        ABBA_conv1 = self.lrelu(self.conv_re_1(ABBA))

        ABBA_conv2 = self.lrelu(self.conv_re_2(ABBA_conv1))

        ABBA_conv3 = self.lrelu(self.conv_re_3(ABBA_conv2))
        output = self.sigmoid_final(ABBA_conv3)
        return output

    def flops(self):
        flops = 0
        return flops


if __name__ == '__main__':
    upscale = 4
    window_size = 8
    height = (1024 // upscale // window_size + 1) * window_size
    width = (720 // upscale // window_size + 1) * window_size
    model = WMamba()
    print(model)
    x = torch.randn((1, 3, height, width))
    x = model(x)
    print(x.shape)
