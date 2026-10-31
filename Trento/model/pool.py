import torch
from timm.models.layers import DropPath, trunc_normal_
from torch import nn
from torch.nn import GroupNorm


class PoolAttn(nn.Module):
    """
    Implementation of pooling for PoolAttnFormer
    --pool_size: pooling size
    """

    def __init__(self, dim=32, norm_layer=GroupNorm):
        super().__init__()
        self.patch_pool1 = nn.AdaptiveAvgPool2d((None, 1))
        self.patch_pool2 = nn.AdaptiveAvgPool2d((1, None))

        self.embdim_pool1 = nn.AdaptiveAvgPool2d((None, 1))
        self.embdim_pool2 = nn.AdaptiveAvgPool2d((1, None))

        self.norm = norm_layer(dim, num_channels=32)
        self.proj0 = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x_patch_attn1 = self.patch_pool1(x)
        x_patch_attn2 = self.patch_pool2(x)
        x_patch_attn = x_patch_attn1 @ x_patch_attn2
        x_patch_attn = self.proj0(x_patch_attn)

        x1 = x.view(B, C, H * W).transpose(1, 2).view(B, H * W, 4, -1)
        x_embdim_attn1 = self.embdim_pool1(x1)
        x_embdim_attn2 = self.embdim_pool2(x1)
        x_embdim_attn = x_embdim_attn1 @ x_embdim_attn2

        x_embdim_attn = x_embdim_attn.view(B, H * W, C).transpose(1, 2).view(B, C, H, W)
        x_embdim_attn = self.proj1(x_embdim_attn)

        x_out = self.norm(x_patch_attn + x_embdim_attn)
        x_out = self.proj2(x_out)
        return x_out


class nchwAttentionLePE(nn.Module):
    """
    Attention with LePE, takes nchw input
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., side_dwconv=5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.proj_drop = nn.Dropout(proj_drop)
        self.lepe = (
            nn.Conv2d(dim, dim, kernel_size=side_dwconv, stride=1,
                      padding=side_dwconv // 2, groups=dim)
            if side_dwconv > 0 else lambda x: torch.zeros_like(x)
        )

    def forward(self, x: torch.Tensor):
        """
        args:
            x: NCHW tensor
        return:
            NCHW tensor
        """
        B, C, H, W = x.size()
        q, k, v = self.qkv.forward(x).chunk(3, dim=1)  # B, C, H, W

        attn = (q.view(B, self.num_heads, self.head_dim, H * W).transpose(-1, -2)
                @ k.view(B, self.num_heads, self.head_dim, H * W))
        attn = torch.softmax(attn * self.scale, dim=-1)
        attn = self.attn_drop(attn)

        # (B, nhead, HW, HW) @ (B, nhead, HW, head_dim) -> (B, nhead, HW, head_dim)
        output: torch.Tensor = attn @ v.view(B, self.num_heads, self.head_dim, H * W).transpose(-1, -2)
        output = output.permute(0, 1, 3, 2).reshape(B, C, H, W)
        output = output + self.lepe(v)

        output = self.proj_drop(self.proj(output))

        return output


class Mlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PoolFormerBlock(nn.Module):
    def __init__(self, dim, pool_size=3, mlp_ratio=4.,
                 act_layer=nn.GELU, norm_layer=GroupNorm,
                 drop=0., drop_path=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5):
        super().__init__()

        self.norm1 = norm_layer(dim, num_channels=32)
        self.token_mixer = PoolAttn(dim=dim, norm_layer=norm_layer)
        self.norm2 = norm_layer(dim, num_channels=32)

        self.normm = norm_layer(dim, num_channels=32)

        self.attention = nchwAttentionLePE(
            dim, num_heads=8, qkv_bias=False, qk_scale=None,
            attn_drop=0., proj_drop=0., side_dwconv=5)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

        # The following two techniques are useful to train deep PoolAttnFormer.
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_m = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_m.unsqueeze(-1).unsqueeze(-1)
                * self.attention(self.normm(x)))

            x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
                * self.token_mixer(self.norm1(x)))

            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)
                * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.attention(self.normm(x)))
            x = x + self.drop_path(self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
