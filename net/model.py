## AdaIR: Adaptive All-in-One Image Restoration via Frequency Mining and Modulation
## Yuning Cui, Syed Waqas Zamir, Salman Khan, Alois Knoll, Mubarak Shah, and Fahad Shahbaz Khan
## https://arxiv.org/abs/2403.14614
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
from einops import rearrange
import copy

from numpy.ma.core import indices


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)


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
        return x / torch.sqrt(sigma+1e-5) * self.weight
    

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
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class DCAM(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.scale = c ** -0.5
        self.norm_l = LayerNorm2d(c)
        self.norm_r = LayerNorm2d(c)
        self.l_proj1 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.r_proj1 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        # self.r_proj2 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.l_proj2 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)

    def forward(self, x_l, x_r, mask, mode="ii"):
        if mode == 1:
            # x_l_m = x_l

            # x_l_m = torch.mul(x_r, 1 - mask)
            # x_r_m = torch.mul(x_r, mask)

            # x_l_m = torch.mul(x_r, mask)
            # x_r_m = x_l

            x_l_m = torch.mul(x_r, 1-mask)
            x_r_m = x_l


        elif mode == 2:
            x_l_m = x_l
            x_r_m = x_r
        elif mode == 3:
            x_l_m = torch.mul(x_l, mask)
            x_r_m = torch.mul(x_r, 1-mask)

            # x_l_m = x_l
            # x_r_m = x_r
        elif mode == 4:
            # x_l_m = torch.mul(x_l, mask)
            x_l_m = x_l
            x_r_m = x_r

        Q_l = self.l_proj1(self.norm_l(x_l_m)).permute(0, 2, 3, 1).contiguous()  # B, H, W, c
        Q_r_T = self.r_proj1(self.norm_r(x_r_m)).permute(0, 2, 1, 3).contiguous()  # B, H, c, W (transposed)
        # (B, H, W, c) x (B, H, c, W) -> (B, H, W, W)
        attention = torch.matmul(Q_l, Q_r_T) * self.scale
        # V_r = self.r_proj2(x_r).permute(0, 2, 3, 1).contiguous()  # B, H, W, c
        V_r = self.l_proj2(x_l).permute(0, 2, 3, 1).contiguous()  # B, H, W, c
        F_r2l = torch.matmul(torch.softmax(attention, dim=-1), V_r)  # B, H, W, c
        F_r2l = F_r2l.permute(0, 3, 1, 2).contiguous() * self.beta
        # return x_l + F_r2l
        return F_r2l

class DSAM(nn.Module):
    def __init__(self, d_dim=128, input_dim=384):
        super(DSAM, self).__init__()
        # degraded represent
        self.DR = None
        self.channel_align_in = None

        self.semantic_extra = None
        self.norm_1 = nn.LayerNorm(d_dim)
        self.norm_2 = nn.LayerNorm(d_dim)
        self.channel_align_out = nn.Conv2d(d_dim, input_dim, 1, 1, 0)

        # self.norm = nn.BatchNorm2d(1)

    def forward(self, x, inp_img=None):
        '''
            sharing semantic extraction for consistence to each encoder
        '''
        x_algin = self.channel_align_in(x)
        B, C, H, W = x_algin.shape
        x_semantic = self.semantic_extra(x_algin).permute(0, 2, 3, 1)  # B, ddim, H, W  -> B, H, W, ddim

        # B, H, W, ddim * ddim N -> B ,H, W, N -> B, N, H, W
        '''
            calculate cosine similarity and generation degradation features for c2d process
        '''
        x_semantic = x_semantic.flatten(1, 2)  # B H*W ddim
        x_semantic = self.norm_1(x_semantic)
        N, _ = self.DR.shape
        dr = self.DR.unsqueeze(0).repeat(B, 1, 1)  # n, ddim -> B, n, ddim
        dr = self.norm_2(dr).permute(0, 2, 1)  # B, ddim, n
        # dr = dr.permute(0, 2, 1)  # B, ddim, n
        c_u = torch.matmul(self.norm_1(x_semantic), dr)
        c_d_l = torch.sqrt(torch.sum(torch.pow(x_semantic, 2), dim=-1, keepdim=True)).repeat(1, 1, N)  # B HW 1
        c_d_r = torch.sqrt(torch.sum(torch.pow(dr, 2), dim=1, keepdim=True)).repeat(1, H * W, 1)  # B 1 N
        sim = torch.div(c_u, torch.mul(c_d_l, c_d_r))  # B H*W N

        # sim = torch.pow(1 + sim, 2)  ##
        sim_soft = F.softmax(sim, dim=-1)

        sim_soft_2D = sim_soft.permute(0, 2, 1).view(B, -1, H, W)
        degrad_feat = torch.matmul(sim_soft, dr.transpose(-2, -1)).permute(0, 2, 1).view(
            x_algin.shape).contiguous()  # B HW N * B N C -> B HW C  -> B H W C

        # print(torch.mean(sim_soft_2D.flatten(-2,-1),dim=-1))
        # print(torch.mean(sim,dim=1))
        '''
            sharing mask generation for consistence to each pixel probability
        '''
        # mask_v = torch.pow((1 + sim_soft_2D), 2)  ## 2
        mask_v = torch.pow((sim_soft_2D), 3)  ## 2
        # mask_v = sim_soft_2D  ##
        mask_v, mask_type = torch.max(mask_v, dim=1, keepdim=True)  # B, N, H, W -> B, 1, H, W
        # print("mask_v:", mask_v.shape)
        # print("mask_type:", mask_type.shape, mask_type)
        #
        # import matplotlib
        # import matplotlib.pyplot as plt
        # matplotlib.use('TkAgg')
        # fig, ax = plt.subplots()
        # vis_type = mask_type[0] / N
        # vis_type = vis_type.cpu().detach().numpy()
        # ax.imshow(vis_type[0, :, :], cmap="bwr")
        # plt.xticks([])
        # plt.yticks([])
        # plt.axis("off")
        # plt.show()
        # plt.close()
        # sys.exit()


        min_val, _ = torch.min(mask_v.view(B, -1), dim=1, keepdim=True)  # B, H*W
        max_val, _ = torch.max(mask_v.view(B, -1), dim=1, keepdim=True)
        min_val = min_val.unsqueeze(-1).unsqueeze(-1)
        max_val = max_val.unsqueeze(-1).unsqueeze(-1)

        # mask_v = (mask_v - min_val) / (max_val - min_val)
        # min_val, _ = torch.min(mask_v.view(B, -1), dim=1, keepdim=True)
        # max_val, _ = torch.max(mask_v.view(B, -1), dim=1, keepdim=True)
        mask_v = (mask_v - min_val) / (max_val - min_val)

        mask_v = 1 - mask_v
        # self.show_prob2D(mask_v)
        # self.train_show_mask(mask_v, inp_img)


        max_values, indices = torch.max(sim_soft, dim=2)
        _, top_indices = torch.topk(max_values, int(sim_soft.shape[1] / 50), dim=1, largest=True, sorted=False)

        results = sim_soft[:, top_indices, :].squeeze(1)
        # print(sim_soft.shape, results.shape)

        return mask_v, self.channel_align_out(degrad_feat), results

##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


##########################################################################
## Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x, x2=None):
        if x2 is not None:
            x = x + self.attn(self.norm1(x)) + x2
        else:
            x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x
    


##########################################################################
##---------- DGIR -----------------------

class DGIR(nn.Module):
    def __init__(self,
        inp_channels=3,
        out_channels=3,
        dim = 48,
        # dim = 24,
        num_blocks = [4,6,6,8],
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',
        is_train = True
    ):
        super(DGIR, self).__init__()
        self.num_blocks = num_blocks
        # ddim, dl = 256, 24
        ddim, dl = dim*4, 72
        '''
            sharing module
        '''
        self.SEM1 = nn.Sequential(  # semantic extraction module
            nn.Conv2d(ddim, ddim // 2, kernel_size=3, padding=2, dilation=2),
            nn.PReLU(),
            nn.Conv2d(ddim // 2, ddim // 2, kernel_size=3, padding=4, dilation=4),
            nn.PReLU(),
            nn.Conv2d(ddim // 2, ddim, kernel_size=1, stride=1, padding=0),
        )
        self.SEM2 = nn.Sequential(  # semantic extraction module
            nn.Conv2d(ddim, ddim // 2, kernel_size=3, padding=2, dilation=2),
            nn.PReLU(),
            nn.Conv2d(ddim // 2, ddim // 2, kernel_size=3, padding=4, dilation=4),
            nn.PReLU(),
            nn.Conv2d(ddim // 2, ddim, kernel_size=1, stride=1, padding=0),
        )

        self.DR = nn.Parameter(torch.zeros(dl, ddim), requires_grad=True)  # degradation representation

        '''
            encoder-decoder
        '''
        # encoder layer 1
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.encoder_level1 = get_clones(TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type), self.num_blocks[0])
        self.encoder_cross1 = get_clones(DCAM(dim), self.num_blocks[0])

        # encoder layer 2
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = get_clones(TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type), self.num_blocks[1])
        self.encoder_cross2 = get_clones(DCAM(int(dim*2**1)), self.num_blocks[1])

        # encoder layer 3
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = get_clones(TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type), self.num_blocks[2])
        self.encoder_cross3 = get_clones(DCAM(int(dim * 2 ** 2)), self.num_blocks[2])

        # encoder layer 4
        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = get_clones(TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type), self.num_blocks[3])
        self.encoder_cross4 = get_clones(DCAM(int(dim * 2 ** 3)), self.num_blocks[3])
        '''
            decoder d2c
        '''
        # decoder layer 1
        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = get_clones(TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type), self.num_blocks[2])
        self.decoder_cross3dc = get_clones(DCAM(int(dim * 2 ** 2)), self.num_blocks[2])

        # decoder layer 2
        self.up3_2 = Upsample(int(dim*2**2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = get_clones(TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type), self.num_blocks[1])
        self.decoder_cross2dc = get_clones(DCAM(int(dim * 2 ** 1)), self.num_blocks[1])

        # decoder layer 3
        self.up2_1 = Upsample(int(dim*2**1))
        self.decoder_level1 = get_clones(TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type), self.num_blocks[0])
        self.decoder_cross1dc = get_clones(DCAM(int(dim * 2 ** 1)), self.num_blocks[0])

        # last layer
        self.refinement_dc = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        self.output_dc = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        # self.norm = LayerNorm2d(dim)

        '''
            decoder c2d
        '''
        # decoder layer 1
        self.up4_3cd = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.reduce_chan_level3cd = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        # self.decoder_level3cd = get_clones(
        #     TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
        #                      bias=bias, LayerNorm_type=LayerNorm_type), self.num_blocks[2])

        # decoder layer 2
        self.up3_2cd = Upsample(int(dim * 2 ** 2))
        self.reduce_chan_level2cd = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        # self.decoder_level2cd = get_clones(
        #     TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
        #                      bias=bias, LayerNorm_type=LayerNorm_type), self.num_blocks[1])

        # decoder layer 3
        self.up2_1cd = Upsample(int(dim * 2 ** 1))
        # self.decoder_level1cd = get_clones(
        #     TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
        #                      bias=bias, LayerNorm_type=LayerNorm_type), self.num_blocks[0])


        self.decoder_cross3cd = get_clones(DCAM(int(dim * 2 ** 2)), self.num_blocks[2])
        self.decoder_cross2cd = get_clones(DCAM(int(dim * 2 ** 1)), self.num_blocks[1])
        self.decoder_cross1cd = get_clones(DCAM(int(dim * 2 ** 1)), self.num_blocks[0])
        self.refinement_cd = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        self.output_cd = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        '''
            d2c
        '''
        self.SFM1 = DSAM(ddim, dim)
        self.SFM2 = DSAM(ddim, int(dim * 2 ** 1))
        self.SFM3 = DSAM(ddim, int(dim * 2 ** 2))
        self.SFM4 = DSAM(ddim, int(dim * 2 ** 3))

        self.CA1 = nn.Sequential(
            nn.Conv2d(dim, ddim * 2, 1, 1, 0),
            nn.PReLU(),
            nn.Conv2d(ddim * 2, ddim, 1, 1, 0),
        )
        self.CA2 = nn.Sequential(
            nn.Conv2d(int(dim * 2 ** 1), ddim * 2, 1, 1, 0),
            nn.PReLU(),
            nn.Conv2d(ddim * 2, ddim, 1, 1, 0),
        )
        self.CA3 = nn.Sequential(
            nn.Conv2d(int(dim * 2 ** 2), ddim * 2, 1, 1, 0),
            nn.PReLU(),
            nn.Conv2d(ddim * 2, ddim, 1, 1, 0),
        )
        self.CA4 = nn.Sequential(
            nn.Conv2d(int(dim * 2 ** 3), ddim * 2, 1, 1, 0),
            nn.PReLU(),
            nn.Conv2d(ddim * 2, ddim, 1, 1, 0),
        )

        self.SFM1.DR = self.DR
        self.SFM1.channel_align_in = self.CA1
        self.SFM1.semantic_extra = self.SEM1

        self.SFM2.DR = self.DR
        self.SFM2.channel_align_in = self.CA2
        self.SFM2.semantic_extra = self.SEM1

        self.SFM3.DR = self.DR
        self.SFM3.channel_align_in = self.CA3
        self.SFM3.semantic_extra = self.SEM1

        self.SFM4.DR = self.DR
        self.SFM4.channel_align_in = self.CA4
        self.SFM4.semantic_extra = self.SEM1
        '''
            c2d
        '''
        self.df1_down = nn.Conv2d(dim, int(dim * 2 ** 1), 1, 1, 0)  # channel align

        self.is_train = is_train
        # self._reset_parameters()

    def tb(self, tbblock, crossblock, feats, feats2, mask=None, mode=1, grad=True):
        for (tblayer, crosslayer) in zip(tbblock, crossblock):
            feat1 = crosslayer(feats, feats2, mask, mode)
            if grad:
                feats = tblayer(feats, feat1)
            else:
                with torch.no_grad():
                    feats = tblayer(feats, feat1)
        return feats

    def tb2(self, tbblock, feats):
        for tblayer in tbblock:
            feats = tblayer(feats)
        return feats



    def _reset_parameters(self):
        for p in self.parameters():
            # print(p)
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.normal_(p)

    def train_forward(self, inp_img, clean_img):
        # inp_img (B, C, H, W)

        d2c_semantic_list = []
        c2d_semantic_list = []

        # for i in range(self.num_blocks[0]):
        #     self.encoder_level1[i].requires_grad=False
        # for i in range(self.num_blocks[1]):
        #     self.encoder_level2[i].requires_grad = False
        # for i in range(self.num_blocks[2]):
        #     self.encoder_level3[i].requires_grad=False
        # for i in range(self.num_blocks[3]):
        #     self.latent[i].requires_grad = False

        '''
            degradation to clean encoder
        '''
        # encoder layer 1
        ie_dc1 = self.patch_embed(inp_img)  # (B, 48, H, W)
        mask1, df1, sim1 = self.SFM1(ie_dc1, inp_img)  # find degradation field and feature
        # oe_dc1 = self.tb(self.encoder_level1, self.encoder_cross1, ie_dc1, ie_dc1, mask1, 1, False)
        oe_dc1 = self.tb(self.encoder_level1, self.encoder_cross1, ie_dc1, ie_dc1, mask1, 1, True)
        d2c_s1 = self.SEM2(self.CA1(oe_dc1))
        d2c_semantic_list.append(d2c_s1)

        # encoder layer 2
        ie_dc2 = self.down1_2(oe_dc1)  # (B, 96, H//2, W//2)
        mask2, df2, sim2 = self.SFM2(ie_dc2)
        # oe_dc2 = self.tb(self.encoder_level2, self.encoder_cross2, ie_dc2, ie_dc2, mask2, 1, False)
        oe_dc2 = self.tb(self.encoder_level2, self.encoder_cross2, ie_dc2, ie_dc2, mask2, 1, True)
        d2c_s2 = self.SEM2(self.CA2(oe_dc2))
        d2c_semantic_list.append(d2c_s2)

        # encoder layer 3
        ie_dc3 = self.down2_3(oe_dc2)  # (B, 192, H//4, W//4)
        mask3, df3, sim3 = self.SFM3(ie_dc3)
        # oe_dc3 = self.tb(self.encoder_level3, self.encoder_cross3, ie_dc3, ie_dc3, mask3, 1, False)
        oe_dc3 = self.tb(self.encoder_level3, self.encoder_cross3, ie_dc3, ie_dc3, mask3, 1, True)
        d2c_s3 = self.SEM2(self.CA3(oe_dc3))
        d2c_semantic_list.append(d2c_s3)

        # encoder layer 4
        ie_dc4 = self.down3_4(oe_dc3)  # (B, 384, H//8, W//8)
        mask4, df4, sim4 = self.SFM4(ie_dc4)
        # latent_d2c = self.tb(self.latent, self.encoder_cross4, ie_dc4, ie_dc4, mask4, 1, False)
        latent_d2c = self.tb(self.latent, self.encoder_cross4, ie_dc4, ie_dc4, mask4, 1, True)
        d2c_s4 = self.SEM2(self.CA4(latent_d2c))
        d2c_semantic_list.append(d2c_s4)

        '''
                clean to degradation
        '''
        # for i in range(self.num_blocks[0]):
        #     self.encoder_level1[i].requires_grad=True
        # for i in range(self.num_blocks[1]):
        #     self.encoder_level2[i].requires_grad=True
        # for i in range(self.num_blocks[2]):
        #     self.encoder_level3[i].requires_grad=True
        # for i in range(self.num_blocks[3]):
        #     self.latent[i].requires_grad = True
        #
        # for i in range(self.num_blocks[2]):
        #     self.decoder_level3[i].requires_grad=False
        # for i in range(self.num_blocks[1]):
        #     self.decoder_level2[i].requires_grad = False
        # for i in range(self.num_blocks[0]):
        #     self.decoder_level1[i].requires_grad=False


        # encoder layer 1
        ie_cd1 = self.patch_embed(clean_img)  # (B, 48, H, W)
        # oe_cd1 = self.tb(self.encoder_level1, self.encoder_cross1, ie_cd1, ie_cd1, mask1, 2, True)
        oe_cd1 = self.tb2(self.encoder_level1, ie_cd1)
        c2d_s1 = self.SEM2(self.CA1(oe_cd1))
        c2d_semantic_list.append(c2d_s1)

        # encoder layer 2
        ie_cd2 = self.down1_2(oe_cd1)  # (B, 96, H//2, W//2)
        # oe_cd2 = self.tb(self.encoder_level2, self.encoder_cross2, ie_cd2, ie_cd2, mask2, 2, True)
        oe_cd2 = self.tb2(self.encoder_level2, ie_cd2)
        c2d_s2 = self.SEM2(self.CA2(oe_cd2))
        c2d_semantic_list.append(c2d_s2)

        # encoder layer 3
        ie_cd3 = self.down2_3(oe_cd2)  # (B, 192, H//4, W//4)
        # oe_cd3 = self.tb(self.encoder_level3, self.encoder_cross3, ie_cd3, ie_cd3, mask3, 2, True)
        oe_cd3 = self.tb2(self.encoder_level3, ie_cd3)
        c2d_s3 = self.SEM2(self.CA3(oe_cd3))
        c2d_semantic_list.append(c2d_s3)

        # encoder layer 4
        ie_cd4 = self.down3_4(oe_cd3)  # (B, 384, H//8, W//8)
        # latent_c2d = self.tb(self.latent, self.encoder_cross4, ie_cd4, ie_cd4, mask4, 2, True)
        latent_c2d = self.tb2(self.latent, ie_cd4)
        c2d_s4 = self.SEM2(self.CA4(latent_c2d))
        c2d_semantic_list.append(c2d_s4)

        # decoder layer 1
        id_cd3 = self.up4_3cd(latent_c2d)
        id_cd3 = torch.cat([id_cd3, oe_cd3], 1)
        id_cd3 = self.reduce_chan_level3cd(id_cd3)
        od_cd3 = self.tb(self.decoder_level3, self.decoder_cross3cd, df3, id_cd3, 1 - mask3, 4, False)
        # od_cd3 = self.tb(self.decoder_level3cd, self.decoder_cross3cd, df3, id_cd3, 1 - mask3, 4, True)

        # decoder layer 2
        id_cd2 = self.up3_2cd(od_cd3)
        id_cd2 = torch.cat([id_cd2, oe_cd2], 1)
        id_cd2 = self.reduce_chan_level2cd(id_cd2)
        od_cd2 = self.tb(self.decoder_level2, self.decoder_cross2cd, df2, id_cd2, 1 - mask2, 4, False)
        # od_cd2 = self.tb(self.decoder_level2cd, self.decoder_cross2cd, df2, id_cd2, 1 - mask2, 4, True)

        # decoder layer 3
        id_cd1 = self.up2_1cd(od_cd2)
        id_cd1 = torch.cat([id_cd1, oe_cd1], 1)
        od_cd1 = self.tb(self.decoder_level1, self.decoder_cross1cd, self.df1_down(df1), id_cd1, 1 - mask1, 4, False)
        # od_cd1 = self.tb(self.decoder_level1cd, self.decoder_cross1cd, self.df1_down(df1), id_cd1, 1 - mask1, 4, True)
        # last layer
        od_cd1 = self.refinement_cd(od_cd1)
        c2d = self.output_cd(od_cd1) + clean_img

        '''
            degradation to clean decoder
        '''

        # for i in range(self.num_blocks[2]):
        #     self.decoder_level3[i].requires_grad=True
        # for i in range(self.num_blocks[1]):
        #     self.decoder_level2[i].requires_grad = True
        # for i in range(self.num_blocks[0]):
        #     self.decoder_level1[i].requires_grad=True

        # decoder layer 1
        id_dc3 = self.up4_3(latent_d2c)
        id_dc3 = torch.cat([id_dc3, ie_dc3], 1)
        id_dc3 = self.reduce_chan_level3(id_dc3)
        od_dc3 = self.tb(self.decoder_level3, self.decoder_cross3dc, id_dc3, df3, 1 - mask3, 4, True)

        # decoder layer 2
        id_dc2 = self.up3_2(od_dc3)
        id_dc2 = torch.cat([id_dc2, ie_dc2], 1)
        id_dc2 = self.reduce_chan_level2(id_dc2)
        od_dc2 = self.tb(self.decoder_level2, self.decoder_cross2dc, id_dc2, df2, 1 - mask2, 4, True)

        # decoder layer 3
        id_dc1 = self.up2_1(od_dc2)
        id_dc1 = torch.cat([id_dc1, ie_dc1], 1)
        od_dc1 = self.tb(self.decoder_level1, self.decoder_cross1dc, id_dc1, self.df1_down(df1), 1 - mask1, 4, True)

        # last layer
        od_dc1 = self.refinement_dc(od_dc1)
        d2c = self.output_dc(od_dc1) + inp_img


        for ind in range(len(d2c_semantic_list)):
            d2c_semantic_list[ind] = nn.AvgPool2d(d2c_semantic_list[ind].size()[2:])(d2c_semantic_list[ind])
        for ind in range(len(c2d_semantic_list)):
            c2d_semantic_list[ind] = nn.AvgPool2d(c2d_semantic_list[ind].size()[2:])(c2d_semantic_list[ind])

        return d2c, c2d, d2c_semantic_list, c2d_semantic_list
        # return d2c, d2c_semantic_list, c2d_semantic_list, mask1

    # def show_prob2D(self, x):
    #     b, c, w, h = x.shape
    #     matplotlib.use('TkAgg')
    #     fig, ax = plt.subplots()
    #     # x.clip_(0.4, 1.0)
    #     vis_prob = x[0].cpu().detach().numpy()
    #     # ax.imshow(vis_prob[0, w//20:w-w//20, h//20:h-h//20], cmap="bwr")
    #     ax.imshow(vis_prob[0, :, :], cmap="bwr")
    #     plt.show()
    #     # sys.exit()
    def test_forward(self, inp_img):
        '''
                    degradation to clean encoder
                '''
        # encoder layer 1
        ie_dc1 = self.patch_embed(inp_img)  # (B, 48, H, W)
        mask1, df1, sim1 = self.SFM1(ie_dc1, inp_img)  # find degradation field and feature
        oe_dc1 = self.tb(self.encoder_level1, self.encoder_cross1, ie_dc1, ie_dc1, mask1, 1)

        # encoder layer 2
        ie_dc2 = self.down1_2(oe_dc1)  # (B, 96, H//2, W//2)
        mask2, df2, sim2 = self.SFM2(ie_dc2)
        oe_dc2 = self.tb(self.encoder_level2, self.encoder_cross2, ie_dc2, ie_dc2, mask2, 1)

        # encoder layer 3
        ie_dc3 = self.down2_3(oe_dc2)  # (B, 192, H//4, W//4)
        mask3, df3, sim3 = self.SFM3(ie_dc3)
        oe_dc3 = self.tb(self.encoder_level3, self.encoder_cross3, ie_dc3, ie_dc3, mask3, 1)

        # encoder layer 4
        ie_dc4 = self.down3_4(oe_dc3)  # (B, 384, H//8, W//8)
        mask4, df4, sim4 = self.SFM4(ie_dc4)
        latent_d2c = self.tb(self.latent, self.encoder_cross4, ie_dc4, ie_dc4, mask4, 1)

        # decoder layer 1
        id_dc3 = self.up4_3(latent_d2c)
        id_dc3 = torch.cat([id_dc3, ie_dc3], 1)
        id_dc3 = self.reduce_chan_level3(id_dc3)
        od_dc3 = self.tb(self.decoder_level3, self.decoder_cross3dc, id_dc3, df3, 1 - mask3, 4)

        # decoder layer 2
        id_dc2 = self.up3_2(od_dc3)
        id_dc2 = torch.cat([id_dc2, ie_dc2], 1)
        id_dc2 = self.reduce_chan_level2(id_dc2)
        od_dc2 = self.tb(self.decoder_level2, self.decoder_cross2dc, id_dc2, df2, 1 - mask2, 4)

        # decoder layer 3
        id_dc1 = self.up2_1(od_dc2)
        id_dc1 = torch.cat([id_dc1, ie_dc1], 1)
        od_dc1 = self.tb(self.decoder_level1, self.decoder_cross1dc, id_dc1, self.df1_down(df1), 1 - mask1, 4)

        # last layer
        od_dc1 = self.refinement_dc(od_dc1)
        d2c = self.output_dc(od_dc1) + inp_img

        masks = [mask1, mask2, mask3]
        sims = [sim1, sim2, sim3, sim4]
        return d2c, masks, sims
        # return d2c


    def forward(self, inp_img, clean_img=None):
        if self.is_train:
            return self.train_forward(inp_img, clean_img)
        else:
            return self.test_forward(inp_img)



