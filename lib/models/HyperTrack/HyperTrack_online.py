from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm.models.vision_transformer
from einops import rearrange
from timm.models.layers import DropPath, Mlp

from lib.utils.misc import is_main_process
from lib.models.HyperTrack.head import build_box_head
from lib.models.HyperTrack.utils import to_2tuple
from lib.utils.box_ops import box_xyxy_to_cxcywh, box_cxcywh_to_xyxy
from lib.models.HyperTrack.pos_utils import get_2d_sincos_pos_embed
from lib.models.HyperTrack.score_decoder import ScoreDecoder


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2).contiguous()  # BCHW -> BNC
        x = self.norm(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.qkv_mem = None

    def forward(self, x, x_hsi, t_h, t_w, s_h, s_w):
        """
        x is a concatenated vector of template and search region features.
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv_hsi = self.qkv(x_hsi).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        q_hsi, k_hsi, v_hsi = qkv_hsi.unbind(0)

        q_mt, q_s = torch.split(q, [t_h * t_w * 2, s_h * s_w], dim=2)
        k_mt, k_s = torch.split(k, [t_h * t_w * 2, s_h * s_w], dim=2)
        v_mt, v_s = torch.split(v, [t_h * t_w * 2, s_h * s_w], dim=2)

        _, q_s_hsi = torch.split(q_hsi, [t_h * t_w * 2, s_h * s_w], dim=2)
        _, k_s_hsi = torch.split(k_hsi, [t_h * t_w * 2, s_h * s_w], dim=2)
        _, v_s_hsi = torch.split(v_hsi, [t_h * t_w * 2, s_h * s_w], dim=2)


        # asymmetric mixed attention
        attn = (q_mt @ k_mt.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_mt = (attn @ v_mt).transpose(1, 2).reshape(B, t_h*t_w*2, C)

        attn = (q_s @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_s = (attn @ v).transpose(1, 2).reshape(B, s_h*s_w, C)


        attn = (q_s_hsi @ k_hsi.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_s_hsi = (attn @ v_hsi).transpose(1, 2).reshape(B, s_h*s_w, C)



        x = torch.cat([x_mt, x_s], dim=1)
        x = self.proj(x)
        x = self.proj_drop(x)


        x_hsi = torch.cat([x_mt, x_s_hsi], dim=1)
        x_hsi = self.proj(x_hsi)
        x_hsi = self.proj_drop(x_hsi)

        return x, x_hsi

    def forward_test(self, x, x_hsi, s_h, s_w):
        B, N, C = x.shape
        qkv_s = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv_s_hsi = self.qkv(x_hsi).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q_s, _, _ = qkv_s.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        q_s_hsi, _, _ = qkv_s_hsi.unbind(0)

        qkv = torch.cat([self.qkv_mem, qkv_s], dim=3)
        qkv_hsi = torch.cat([self.qkv_mem, qkv_s_hsi], dim=3)

        _, k, v = qkv.unbind(0)
        _, k_hsi, v_hsi = qkv_hsi.unbind(0)

        attn = (q_s @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, s_h*s_w, C)

        attn = (q_s_hsi @ k_hsi.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_hsi = (attn @ v_hsi).transpose(1, 2).reshape(B, s_h*s_w, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        x_hsi = self.proj(x_hsi)
        x_hsi = self.proj_drop(x_hsi)
        return x, x_hsi

    def set_online(self, x, t_h, t_w):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        self.qkv_mem = qkv
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple) [B, num_heads, N, C//num_heads]

        # asymmetric mixed attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)   # 这是一个完整的anttention的block
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, x_hsi, t_h, t_w, s_h, s_w):
        x_temp, x_hsi_temp = x, x_hsi
        x, x_hsi = self.attn(self.norm1(x), self.norm1(x_hsi), t_h, t_w, s_h, s_w)

        x = x_temp + self.drop_path1(x)
        x_hsi = x_hsi_temp + self.drop_path1(x_hsi)

        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        x_hsi = x_hsi + self.drop_path2(self.mlp(self.norm2(x_hsi)))
        return x, x_hsi

    def forward_test(self, x, x_hsi, s_h, s_w):
        x_temp, x_hsi_temp = x, x_hsi
        x, x_hsi = self.attn.forward_test(self.norm1(x), self.norm1(x_hsi), s_h, s_w)

        x = x_temp + self.drop_path1(x)
        x_hsi = x_hsi_temp + self.drop_path1(x_hsi)


        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        x_hsi = x_hsi + self.drop_path2(self.mlp(self.norm2(x_hsi)))

        return x, x_hsi

    def set_online(self, x, t_h, t_w):
        x = x + self.drop_path1(self.attn.set_online(self.norm1(x), t_h, t_w))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, img_size_s=256, img_size_t=128, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768,
                 depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., weight_init='', embed_layer=PatchEmbed, norm_layer=None, act_layer=None):
        super(VisionTransformer, self).__init__(img_size=224, patch_size=patch_size, in_chans=in_chans,
                                                num_classes=num_classes, embed_dim=embed_dim, depth=depth,
                                                num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                                drop_path_rate=drop_path_rate, weight_init=weight_init,
                                                norm_layer=norm_layer, act_layer=act_layer)

        self.patch_embed = embed_layer(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                norm_layer=norm_layer) for i in range(depth)])

        self.grid_size_s = img_size_s // patch_size
        self.grid_size_t = img_size_t // patch_size
        self.num_patches_s = self.grid_size_s ** 2
        self.num_patches_t = self.grid_size_t ** 2
        self.pos_embed_s = nn.Parameter(torch.zeros(1, self.num_patches_s, embed_dim), requires_grad=False)
        self.pos_embed_t = nn.Parameter(torch.zeros(1, self.num_patches_t, embed_dim), requires_grad=False)

        self.init_pos_embed()

        if weight_init != 'skip':
            self.init_weights(weight_init)

    def init_pos_embed(self):
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed_t = get_2d_sincos_pos_embed(self.pos_embed_t.shape[-1], int(self.num_patches_t ** .5),
                                            cls_token=False)
        self.pos_embed_t.data.copy_(torch.from_numpy(pos_embed_t).float().unsqueeze(0))

        pos_embed_s = get_2d_sincos_pos_embed(self.pos_embed_s.shape[-1], int(self.num_patches_s ** .5),
                                              cls_token=False)
        self.pos_embed_s.data.copy_(torch.from_numpy(pos_embed_s).float().unsqueeze(0))

    def forward(self, x_t, x_ot, x_s, x_s_his):
        """
        :param x_t: (batch, c, 128, 128)
        :param x_s: (batch, c, 288, 288)
        :return:
        """
        x_t = self.patch_embed(x_t)  # BCHW-->BNC
        x_ot = self.patch_embed(x_ot)
        x_s = self.patch_embed(x_s)
        x_s_his = self.patch_embed(x_s_his)
        B, C = x_t.size(0), x_t.size(-1)
        H_s = W_s = self.grid_size_s
        H_t = W_t = self.grid_size_t

        x_s = x_s + self.pos_embed_s
        x_s_his = x_s_his + self.pos_embed_s

        x_t = x_t + self.pos_embed_t
        x_ot = x_ot + self.pos_embed_t

        x = torch.cat([x_t, x_ot, x_s], dim=1)
        x_hsi = torch.cat([x_t, x_ot, x_s_his], dim=1)

        x = self.pos_drop(x)
        x_hsi = self.pos_drop(x_hsi)

        for blk in self.blocks:
            x, x_hsi = blk(x, x_hsi, H_t, W_t, H_s, W_s)

        x_t, x_ot, x_s = torch.split(x, [H_t*W_t, H_t*W_t, H_s*W_s], dim=1)
        _, _, x_s_his = torch.split(x_hsi, [H_t*W_t, H_t*W_t, H_s*W_s], dim=1)

        x_t_2d = x_t.transpose(1, 2).reshape(B, C, H_t, W_t)
        x_ot_2d = x_ot.transpose(1, 2).reshape(B, C, H_t, W_t)
        x_s_2d = x_s.transpose(1, 2).reshape(B, C, H_s, W_s)
        x_s_2d_hsi = x_s_his.transpose(1, 2).reshape(B, C, H_s, W_s)

        return x_t_2d, x_ot_2d, x_s_2d, x_s_2d_hsi

    def forward_test(self, x, x_hsi):
        x = self.patch_embed(x)
        x_hsi = self.patch_embed(x_hsi)
        H_s = W_s = self.grid_size_s

        x = x + self.pos_embed_s
        x_hsi = x_hsi + self.pos_embed_s
        x = self.pos_drop(x)
        x_hsi = self.pos_drop(x_hsi)

        for blk in self.blocks:
            x, x_hsi = blk.forward_test(x, x_hsi, H_s, W_s)

        x = rearrange(x, 'b (h w) c -> b c h w', h=H_s, w=H_s)
        x_hsi = rearrange(x_hsi, 'b (h w) c -> b c h w', h=H_s, w=H_s)
        return self.template, x, x_hsi

    def set_online(self, x_t, x_ot):
        x_t = self.patch_embed(x_t)
        x_ot = self.patch_embed(x_ot)

        H_t = W_t = self.grid_size_t

        x_t = x_t + self.pos_embed_t
        x_ot = x_ot + self.pos_embed_t
        x_ot = x_ot.reshape(1, -1, x_ot.size(-1))  # [1, num_ot * H_t * W_t, C]
        x = torch.cat([x_t, x_ot], dim=1)

        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk.set_online(x, H_t, W_t)

        x_t = x[:, :H_t * W_t]
        x_t = rearrange(x_t, 'b (h w) c -> b c h w', h=H_t, w=W_t)

        self.template = x_t


def get_hypertrack_vit(config, train):
    img_size_s = config.DATA.SEARCH.SIZE
    img_size_t = config.DATA.TEMPLATE.SIZE
    if config.MODEL.VIT_TYPE == 'large_patch16':
        vit = VisionTransformer(
            img_size_s=img_size_s, img_size_t=img_size_t,
            patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_path_rate=0.1)
    elif config.MODEL.VIT_TYPE == 'base_patch16':
        vit = VisionTransformer(
            img_size_s=img_size_s, img_size_t=img_size_t,
            patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_path_rate=0.1)
    else:
        raise KeyError(f"VIT_TYPE shoule set to 'large_patch16' or 'base_patch16'")

    if config.MODEL.BACKBONE.PRETRAINED and train:
        ckpt_path = config.MODEL.BACKBONE.PRETRAINED_PATH
        ckpt = torch.load(ckpt_path, map_location='cpu')['model']
        new_dict = {}
        for k, v in ckpt.items():
            if 'pos_embed' not in k and 'mask_toke' not in k:    # use fixed pos embed
                new_dict[k] = v
        missing_keys, unexpected_keys = vit.load_state_dict(new_dict, strict=False)
        if is_main_process():
            print("Load pretrained backbone checkpoint from:", ckpt_path)
            print("missing keys:", missing_keys)
            print("unexpected keys:", unexpected_keys)
            print("Loading pretrained ViT done.")

    return vit


class hypertrackOnlineScore(nn.Module):
    """ hypertrack tracking with score prediction module, whcih jointly perform feature extraction and interaction. """
    def __init__(self, backbone, box_head, transband_red, transband_vis, transband_nir, bandsgate, bands2three, datatype, score_branch=None, head_type="CORNER"):
        """ Initializes the model.
        """
        super().__init__()
        self.datatype = datatype
        self.bandsgate = bandsgate
        self.transband_rednir = transband_red
        self.transband_vis = transband_vis
        self.transband_nir = transband_nir
        self.backbone = backbone
        self.box_head = box_head
        self.score_branch = score_branch
        self.head_type = head_type
        self.bands2three = bands2three




    def forward(self, template, online_template, search, run_score_head=True, gt_bboxes=None):
        # search: (b, c, h, w)

        if template.dim() == 5:
            template = template.squeeze(0)
        if online_template.dim() == 5:
            online_template = online_template.squeeze(0)
        if search.dim() == 5:
            search = search.squeeze(0)

        batch_size, num_channels, height, width = template.size()
        rank_template = torch.zeros(batch_size, num_channels, height, width).cuda()
        rank_online_template = torch.zeros(batch_size, num_channels, height, width).cuda()
        size_t = template.shape[2]
        size_s = search.shape[2]
        # zero padding
        if self.datatype == "hypertrack_rednir":
            template = template[:, :15, :, :]
            online_template = online_template[:, :15, :, :]

            zero_template = torch.zeros(batch_size, 10, size_t, size_t).cuda()
            zero_search = torch.zeros(batch_size, 10, size_s, size_s).cuda()
            template = torch.cat((template, zero_template), dim=1)
            online_template = torch.cat((online_template, zero_template), dim=1)
            search = torch.cat((search, zero_search), dim=1)
            rank_template = torch.zeros(batch_size, num_channels-1, height, width).cuda()
            rank_online_template = torch.zeros(batch_size, num_channels-1, height, width).cuda()

        elif self.datatype == "hypertrack_vis":
            zero_template = torch.zeros(batch_size, 9, size_t, size_t).cuda()
            zero_search = torch.zeros(batch_size, 9, size_s, size_s).cuda()
            template = torch.cat((template, zero_template), dim=1)
            online_template = torch.cat((online_template, zero_template), dim=1)
            search = torch.cat((search, zero_search), dim=1)
            rank_template = torch.zeros(batch_size, num_channels, height, width).cuda()
            rank_online_template = torch.zeros(batch_size, num_channels, height, width).cuda()

        search_selected, search_plug, indice_topall = self.bandsgate(search, self.datatype)
        if self.datatype == "hypertrack_nir":
            for i in range(batch_size):
                permutation = indice_topall[i].long()
                rank_template[i] = template[i, permutation, :, :]
                rank_online_template[i] = template[i, permutation, :, :]
            template = self.bands2three(rank_template[:, :-15, :, :])
            online_template = self.bands2three(rank_online_template[:, :-15, :, :])
            search_plug = self.transband_nir(search_plug)
        elif self.datatype == "hypertrack_vis":
            for i in range(batch_size):
                permutation = indice_topall[i].long()
                rank_template[i] = template[i, permutation, :, :]
                rank_online_template[i] = template[i, permutation, :, :]
            template = self.bands2three(rank_template[:, :-6, :, :])
            online_template = self.bands2three(rank_online_template[:, :-6, :, :])
            search_plug = self.transband_vis(search_plug)
        elif self.datatype == "hypertrack_rednir":
            for i in range(batch_size):
                permutation = indice_topall[i].long()
                rank_template[i] = template[i, permutation, :, :]
                rank_online_template[i] = template[i, permutation, :, :]
            template = self.bands2three(rank_online_template[:, :-5, :, :])
            online_template = self.bands2three(rank_online_template[:, :-5, :, :])
            search_plug = self.transband_rednir(search_plug)

        search_selected = self.bands2three(search_selected)
        template, online_template, search, search_plug = self.backbone(template, online_template, search_selected, search_plug)

        abs_tensor1 = torch.abs(search)
        abs_tensor2 = torch.abs(search_plug)

        # 找到绝对值的最大值
        max_abs = torch.max(abs_tensor1, abs_tensor2)

        # 根据最大值找到对应位置的原始值
        search = torch.where(abs_tensor1 >= abs_tensor2, search, search_plug)



        # search = torch.max(search, search_HSI)
        # search shape: (b, 384, 20, 20)
        # Forward the corner head and score head
        out, outputs_coord_new = self.forward_head(search, template, run_score_head, gt_bboxes)

        return out, outputs_coord_new

    def forward_test(self, search, topk_instance, run_score_head=True, gt_bboxes=None):
        # search: (b, c, h, w)
        if search.dim() == 5:
            search = search.squeeze(0)
        batch_size, num_channels, height, width = search.size()
        rank_search = torch.zeros(batch_size, num_channels, height, width).cuda()
        if self.datatype == "hypertrack_rednir":
            search = search[:, :-1, :, :]
            num_channels -= 1
            rank_search = torch.zeros(batch_size, num_channels, height, width).cuda()
        for i in range(batch_size):
            permutation = topk_instance[i].long()
            rank_search[i] = search[i, permutation, :, :]



        if self.datatype == "hypertrack_rednir":
            search = self.bands2three(rank_search[:, :-5, :, :])
            search_plug = self.transband_rednir(rank_search[:, 10:, :, :])
        elif self.datatype == "hypertrack_vis":
            search = self.bands2three(rank_search[:, :-6, :, :])
            search_plug = self.transband_vis(rank_search[:, 10:, :, :])
        elif self.datatype == "hypertrack_nir":
            search = self.bands2three(rank_search[:, :-15, :, :])
            search_plug = self.transband_nir(rank_search[:, 10:, :, :])

        # fig, axs = plt.subplots(1, 3, figsize=(288, 288))
        # search = search.cpu()
        # search = search.squeeze(0).numpy()
        # for i in range(3):
        #     axs[i].imshow(search[i])
        #     axs[i].set_title(f'Channel {i + 1}')
        # plt.tight_layout()
        # plt.show()


        template, search, search_plug = self.backbone.forward_test(search, search_plug)

        abs_tensor1 = torch.abs(search)
        abs_tensor2 = torch.abs(search_plug)


        search = torch.where(abs_tensor1 >= abs_tensor2, search, search_plug)


        #search = torch.max(search, search_HSI)
        # Forward the corner head and score head
        out, outputs_coord_new = self.forward_head(search, template, run_score_head, gt_bboxes)

        return out, outputs_coord_new

    def set_online(self, template, online_template):
        if template.dim() == 5:
            template = template.squeeze(0)
        if online_template.dim() == 5:
            online_template = online_template.squeeze(0)
        batch_size, num_channels, height, width = template.size()
        rank_template = torch.zeros(batch_size, num_channels, height, width).cuda()
        rank_online_template = torch.zeros(batch_size, num_channels, height, width).cuda()



        # zero padding
        if self.datatype == "hypertrack_rednir":
            template = template[:, :15, :, :]
            online_template = online_template[:, :15, :, :]
            zero_template = torch.zeros(batch_size, 10, height, width).cuda()
            template = torch.cat((template, zero_template), dim=1)
            online_template = torch.cat((online_template, zero_template), dim=1)
            rank_template = torch.zeros(batch_size, num_channels - 1, height, width).cuda()
            rank_online_template = torch.zeros(batch_size, num_channels - 1, height, width).cuda()

        elif self.datatype == "hypertrack_vis":
            zero_template = torch.zeros(batch_size, 9, height, width).cuda()
            template = torch.cat((template, zero_template), dim=1)
            online_template = torch.cat((online_template, zero_template), dim=1)
            rank_template = torch.zeros(batch_size, num_channels, height, width).cuda()
            rank_online_template = torch.zeros(batch_size, num_channels, height, width).cuda()

        rank_online_template, _, indice_topall = self.bandsgate(online_template, self.datatype)

        if self.datatype == "hypertrack_nir":
            for i in range(batch_size):
                permutation = indice_topall[i].long()
                rank_template[i] = template[i, permutation, :, :]
            template = self.bands2three(rank_template[:, :-15, :, :])
            online_template = self.bands2three(rank_online_template)


        elif self.datatype == "hypertrack_vis":
            for i in range(batch_size):
                permutation = indice_topall[i].long()
                rank_template[i] = template[i, permutation, :, :]
            template = self.bands2three(rank_template[:, :-6, :, :])
            online_template = self.bands2three(rank_online_template)


        elif self.datatype == "hypertrack_rednir":
            for i in range(batch_size):
                permutation = indice_topall[i].long()
                rank_template[i] = template[i, permutation, :, :]
            template = self.bands2three(rank_template[:, :-5, :, :])
            online_template = self.bands2three(rank_online_template)

        self.backbone.set_online(template, online_template)
        return indice_topall

    def online_update(self, template, online_template, instance):
        if template.dim() == 5:
            template = template.squeeze(0)
        if online_template.dim() == 5:
            online_template = online_template.squeeze(0)
        batch_size, num_channels, height, width = online_template.size()
        instance = instance
        # zero padding
        if self.datatype == "hypertrack_rednir":
            online_template = online_template[:, :-1, :, :]
            online_template_1 = online_template[0, instance, :, :]
            #online_template_2 = torch.cat((online_template[1, :, :, :].unsqueeze(0), zero_template), dim=1)
            online_template_1 = self.bands2three(online_template_1[:, :-5, :, :])
            template = template[:, :-1, :, :]
            template = self.bands2three(template[:, instance[0], :, :][:, :-5, :, :])
            #online_template_2, _, indice_topall = self.bandsgate(online_template_2, self.datatype)
            online_template_2 = online_template[1, instance, :, :]                  #  change
            online_template_2 = self.bands2three(online_template_2[:, :-5, :, :])        # change


        elif self.datatype == "hypertrack_vis":
            zero_template = torch.zeros(1, 9, height, width).cuda()
            online_template_1 = online_template[0, instance, :, :]
            online_template_2 = torch.cat((online_template[1, :, :, :].unsqueeze(0), zero_template), dim=1)

            online_template_1 = self.bands2three(online_template_1[:, :-6, :, :])
            template = self.bands2three(template[:, instance[0], :, :][:, :-6, :, :])
            #online_template_2, _, indice_topall = self.bandsgate(online_template_2, self.datatype)

            online_template_2 = online_template[1, instance, :, :]                  #  change
            online_template_2 = self.bands2three(online_template_2[:, :-6, :, :])        # change

        elif self.datatype == "hypertrack_nir":

            online_template_1 = online_template[0, instance, :, :]
            #online_template_2 = online_template[1, :, :, :].unsqueeze(0)
            online_template_1 = self.bands2three(online_template_1[:, :-15, :, :])
            template = self.bands2three(template[:, instance[0], :, :][:, :-15, :, :])
            #online_template_2, _, indice_topall = self.bandsgate(online_template_2, self.datatype)
            online_template_2 = online_template[1,instance, :, :]
            online_template_2 = self.bands2three(online_template_2[:, :-15, :, :])

        # if self.datatype == "hypertrack_nir":
        #
        #     online_template_2 = self.bands2three(online_template_2)
        #
        # elif self.datatype == "hypertrack_vis":
        #     online_template_2 = self.bands2three(online_template_2)
        #
        #
        # elif self.datatype == "hypertrack_rednir":
        #
        #     online_template_2 = self.bands2three(online_template_2)

        online_template = torch.cat((online_template_1, online_template_2), dim=0)
        self.backbone.set_online(template, online_template)
        return instance

    def forward_head(self, search, template, run_score_head=True, gt_bboxes=None):
        """
        :param search: (b, c, h, w)
        :return:
        """
        out_dict = {}
        out_dict_box, outputs_coord = self.forward_box_head(search)
        out_dict.update(out_dict_box)
        if run_score_head:
            # forward the classification head
            if gt_bboxes is None:
                gt_bboxes = box_cxcywh_to_xyxy(outputs_coord.clone().view(-1, 4))
            # (b,c,h,w) --> (b,h,w)
            out_dict.update({'pred_scores': self.score_branch(search, template, gt_bboxes).view(-1)})

        return out_dict, outputs_coord


    def forward_box_head(self, search):
        """
        :param search: (b, c, h, w)
        :return:
        """
        if "CORNER" in self.head_type:
            # run the corner head
            b = search.size(0)
            outputs_coord = box_xyxy_to_cxcywh(self.box_head(search))
            outputs_coord_new = outputs_coord.view(b, 1, 4)
            out = {'pred_boxes': outputs_coord_new}
            return out, outputs_coord_new
        else:
            raise KeyError


def build_hypertrack_vit_online_score(cfg, datatype, settings=None, train=True) -> hypertrackOnlineScore:

    bandsgate = Bandsgate(input_dim=25, output_dim=25, hidden_dim=50)
    bands2three = Bands2three()

    bandsembedding_nir = Transband(input_dim=15, output_dim=3)
    bandsembedding_vis = Transband(input_dim=6, output_dim=3)
    bandsembedding_red = Transband(input_dim=5, output_dim=3)

    backbone = get_hypertrack_vit(cfg, train)  # backbone without positional encoding and attention mask
    score_branch = ScoreDecoder(pool_size=4, hidden_dim=cfg.MODEL.HIDDEN_DIM, num_heads=cfg.MODEL.HIDDEN_DIM//64)  # the proposed score prediction module (SPM)
    box_head = build_box_head(cfg)  # a simple corner head
    model = hypertrackOnlineScore(
        datatype=datatype,
        bands2three=bands2three,
        bandsgate=bandsgate,
        transband_nir=bandsembedding_nir,
        transband_red=bandsembedding_red,
        transband_vis=bandsembedding_vis,
        backbone=backbone,
        box_head=box_head,
        score_branch=score_branch,
        head_type=cfg.MODEL.HEAD_TYPE
    )

    if cfg.MODEL.PRETRAINED_STAGE1 and train:
        ckpt_path = settings.stage1_model
        ckpt = torch.load(ckpt_path, map_location='cpu')
        # if "optimizer" in ckpt:
        #     del ckpt["optimizer"]
        missing_keys, unexpected_keys = model.load_state_dict(ckpt['net'], strict=False)
        if is_main_process():
            print("missing keys:", missing_keys)
            print("unexpected keys:", unexpected_keys)
            print("Loading pretrained hypertrack weights done.")

    return model


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

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


class Transband(nn.Module):
    def __init__(self, input_dim=25, output_dim=3, layer_scale_init_value=1e-6, drop_path=0):
        super().__init__()
        self.dwconv = nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=1, padding=1, groups=input_dim)
        self.norm = LayerNorm(input_dim, eps=1e-6)

        self.pwconv_1 = nn.Linear(input_dim, 4*input_dim, bias=False)
        self.act = nn.GELU()
        self.pwconv_2 = nn.Linear(input_dim*4, input_dim, bias=False)
        self.pwconv_3 = nn.Linear(input_dim, output_dim, bias=False)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((output_dim)),
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)     # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv_1(x)
        x = self.act(x)
        x = self.pwconv_2(x)
        x = self.act(x)
        x = self.pwconv_3(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        return x

    def forward_test(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv_1(x)
        x = self.act(x)
        x = self.pwconv_2(x)
        x = self.act(x)
        x = self.pwconv_3(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        return x


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        return nn.functional.avg_pool2d(x, kernel_size=x.size()[2:])


class Bands2three(nn.Module):
    def __init__(self, input_dim=25, output_dim=25, hidden_dim=50):
        super().__init__()
        self.relu = nn.ReLU()
        self.pwconv = nn.Conv2d(10, 5, kernel_size=1)
        self.pwconv_1 = nn.Conv2d(5, 3, kernel_size=1)

    def forward(self, x):

        x = self.pwconv(x)
        x = self.relu(x)
        x = self.pwconv_1(x)

        return x

class Bandsgate(nn.Module):
    def __init__(self, input_dim=25, output_dim=25, hidden_dim=50):
        super().__init__()
        self.dwconv = nn.Conv2d(input_dim, input_dim, kernel_size=7, stride=1, padding=1, groups=input_dim)
        self.dwconv_1 = nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=1, padding=1, groups=input_dim, bias=False)
        self.act = nn.GELU()
        self.gap = GlobalAvgPool2d()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.epsilon = 1e-8
        self.sgmod = nn.Sigmoid()
        # self.gate = nn.Parameter(torch.ones((output_dim)),requires_grad=True)
        #self.gate = nn.Parameter(torch.ones(1,25))
        # self.pwconv = nn.Linear(10, 20)
        # self.pwconv_1 = nn.Linear(20, 3)

    def forward(self, x, bandstype):
        input_x = x
        x = self.dwconv(x)
        x = self.act(x)
        x = self.dwconv_1(x)

        x = self.gap(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sgmod(x)
        #x = x + self.gate
        batch_size, num_channels, height, width = input_x.size()

        gate = (x**2)/(x**2+self.epsilon)
        #gate = x
        gate = gate.view(batch_size, num_channels, 1, 1)
        input_x = input_x * gate.expand_as(gate)
        #x = x.squeeze(3).squeeze(2)
        if bandstype == "hypertrack_nir":
            rank_x = torch.zeros(batch_size, num_channels, height, width, requires_grad=False).cuda()
            _, topall_indices = torch.topk(x, k=25)
            for i in range(batch_size):
                permutation = topall_indices[i].long()
                rank_x[i] = input_x[i, permutation, :, :]
            x_selected = rank_x[:, :-15, :, :]
            x_plug = rank_x[:, 10:, :, :]
        elif bandstype == "hypertrack_vis":
            input_x = input_x[:, :-9, :, :]
            num_channels = 16
            rank_x = torch.zeros(batch_size, num_channels, height, width, requires_grad=False).cuda()
            _, topall_indices = torch.topk(x[:, :16], k=16)
            for i in range(batch_size):
                permutation = topall_indices[i].long()
                rank_x[i] = input_x[i, permutation, :, :]
            x_selected = input_x[:, :-6, :, :]
            x_plug = input_x[:, 10:, :, :]
        elif bandstype == "hypertrack_rednir":
            input_x = input_x[:, :-10, :, :]
            num_channels = 15
            rank_x = torch.zeros(batch_size, num_channels, height, width, requires_grad=False).cuda()
            _, topall_indices = torch.topk(x[:, :15], k=15)
            for i in range(batch_size):
                permutation = topall_indices[i].long()
                rank_x[i] = input_x[i, permutation, :, :]
            x_selected = input_x[:, :-5, :, :]
            x_plug = input_x[:, 10:, :, :]
        #x_selected = input_x[:, topall_indices[1][:-15], :, :]
        # x_selected = self.pwconv(x_selected)
        # self.act(x_selected)
        # x_selected = self.pwconv_1(x_selected)

        #x_plug = input_x[:, topall_indices[1][10:], :, :]
        return x_selected, x_plug, topall_indices

    def forward_test(self, x):
        input_x = x
        x = self.dwconv(x)
        x = self.act(x)
        x = self.dwconv_1(x)

        x = self.gap(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        _, topall_indices = torch.topk(x, k=25)
        x_selected = input_x[:, topall_indices[1][:-15], :, :]

        x_plug = input_x[:, topall_indices[1][10:], :, :]
        return x_selected, x_plug, topall_indices


#
# bandsgate = Bandsgate()
# #
# input_tensor = torch.randn(2, 25, 192, 192)
# #
# output_1, output_2 = bandsgate.forward_test(input_tensor)
#
#
# print(output_1.shape)
# print(output_2.shape)