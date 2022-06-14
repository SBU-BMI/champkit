"""Implementation of CTransPath.

Jakub Kaczmarzyk downloaded the authors' modified timm library
(https://drive.google.com/file/d/1JV7aj9rKqGedXY1TdDfi3dP07022hcgZ/view),
isolated the differences to the swin_transformer.py file and the SwinTransformer class,
and then copied over the SwinTransformer implementation and made the changes. The only
change is using ConvStem instead of PatchEmbed in the SwinTransformer __init__.

```diff
diff --git a/timm/timm/models/swin_transformer.py b/timm-0.5.4/timm/models/swin_transformer.py
old mode 100644
new mode 100755
index 584d564..6cf9122
--- a/timm/timm/models/swin_transformer.py
+++ b/timm-0.5.4/timm/models/swin_transformer.py
@@ -448,7 +448,7 @@ class SwinTransformer(nn.Module):
                  embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
                  window_size=7, mlp_ratio=4., qkv_bias=True,
                  drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
-                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
+                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,embed_layer=PatchEmbed,
                  use_checkpoint=False, weight_init='', **kwargs):
         super().__init__()

@@ -461,7 +461,7 @@ class SwinTransformer(nn.Module):
         self.mlp_ratio = mlp_ratio

         # split image into non-overlapping patches
-        self.patch_embed = PatchEmbed(
+        self.patch_embed = embed_layer(
             img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
             norm_layer=norm_layer if self.patch_norm else None)
         num_patches = self.patch_embed.num_patches
```
"""

import math

import timm
from timm.models.layers import to_ntuple
import torch
from torch import nn


class ConvStem(nn.Module):
    """ConvStem for CTransPath model."""

    def __init__(
        self,
        img_size=224,
        patch_size=4,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
    ):
        super().__init__()

        assert patch_size == 4
        assert embed_dim % 8 == 0

        img_size = to_ntuple(2)(img_size)
        patch_size = to_ntuple(2)(patch_size)
        assert len(img_size) == 2 and isinstance(img_size, tuple)
        assert len(patch_size) == 2 and isinstance(patch_size, tuple)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        stem = []
        input_dim, output_dim = 3, embed_dim // 8
        for l in range(2):
            stem.append(
                nn.Conv2d(
                    input_dim,
                    output_dim,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            )
            stem.append(nn.BatchNorm2d(output_dim))
            stem.append(nn.ReLU(inplace=True))
            input_dim = output_dim
            output_dim *= 2
        stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*stem)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class SwinTransformerForCTransPath(nn.Module):
    r"""CTransPath.

    This is a copy of the Swin Transformer.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Modified SwinTransformer for use with CTransPath. The only change is enabling
    choice of embed_layer. CTransPath uses ConvStem. SwinTransformer uses PatchEmbed.

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        head_dim (int, tuple(int)):
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        embed_layer (nn.Module): Embedding layer. Default is ConvStem.
    """

    def __init__(
        self,
        img_size=224,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        global_pool="avg",
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        head_dim=None,
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        weight_init="",
        embed_layer=None,
        **kwargs,
    ):
        super().__init__()

        from timm.models.swin_transformer import BasicLayer, PatchMerging

        assert global_pool in ("", "avg")
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))

        # split image into non-overlapping patches
        self.embed_layer = embed_layer or ConvStem
        self.patch_embed = self.embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None,
        )
        num_patches = self.patch_embed.num_patches
        self.patch_grid = self.patch_embed.grid_size

        # absolute position embedding
        self.absolute_pos_embed = (
            nn.Parameter(torch.zeros(1, num_patches, embed_dim)) if ape else None
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        # build layers
        if not isinstance(embed_dim, (tuple, list)):
            embed_dim = [int(embed_dim * 2**i) for i in range(self.num_layers)]
        embed_out_dim = embed_dim[1:] + [None]
        head_dim = to_ntuple(self.num_layers)(head_dim)
        window_size = to_ntuple(self.num_layers)(window_size)
        mlp_ratio = to_ntuple(self.num_layers)(mlp_ratio)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule
        layers = []
        for i in range(self.num_layers):
            layers += [
                BasicLayer(
                    dim=embed_dim[i],
                    out_dim=embed_out_dim[i],
                    input_resolution=(
                        self.patch_grid[0] // (2**i),
                        self.patch_grid[1] // (2**i),
                    ),
                    depth=depths[i],
                    num_heads=num_heads[i],
                    head_dim=head_dim[i],
                    window_size=window_size[i],
                    mlp_ratio=mlp_ratio[i],
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                    norm_layer=norm_layer,
                    downsample=PatchMerging if (i < self.num_layers - 1) else None,
                )
            ]
        self.layers = nn.Sequential(*layers)

        self.norm = norm_layer(self.num_features)
        self.head = (
            nn.Linear(self.num_features, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

        if weight_init != "skip":
            self.init_weights(weight_init)

    @torch.jit.ignore
    def init_weights(self, mode=""):
        from timm.models.helpers import named_apply
        from timm.models.layers import trunc_normal_
        from timm.models.vision_transformer import get_init_weights_vit

        assert mode in ("jax", "jax_nlhb", "moco", "")
        if self.absolute_pos_embed is not None:
            trunc_normal_(self.absolute_pos_embed, std=0.02)
        head_bias = -math.log(self.num_classes) if "nlhb" in mode else 0.0
        named_apply(get_init_weights_vit(mode, head_bias=head_bias), self)

    @torch.jit.ignore
    def no_weight_decay(self):
        nwd = {"absolute_pos_embed"}
        for n, _ in self.named_parameters():
            if "relative_position_bias_table" in n:
                nwd.add(n)
        return nwd

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r"^absolute_pos_embed|patch_embed",  # stem and embed
            blocks=r"^layers\.(\d+)"
            if coarse
            else [
                (r"^layers\.(\d+).downsample", (0,)),
                (r"^layers\.(\d+)\.\w+\.(\d+)", None),
                (r"^norm", (99999,)),
            ],
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        for l in self.layers:
            l.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ("", "avg")
            self.global_pool = global_pool
        self.head = (
            nn.Linear(self.num_features, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.absolute_pos_embed is not None:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x = self.layers(x)
        x = self.norm(x)  # B L C
        return x

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool == "avg":
            x = x.mean(dim=1)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


@timm.models.register_model
def ctranspath_ssl(pretrained=False, **kwargs):
    """Swin-T @ 224x224, trained on histopathology images."""

    from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    from timm.models.helpers import build_model_with_cfg
    from timm.models.vision_transformer import checkpoint_filter_fn

    def _checkpoint_filter_fn(state_dict, model):
        state_dict = checkpoint_filter_fn(state_dict=state_dict, model=model)
        state_dict["head.weight"] = torch.zeros(1000, 768)
        state_dict["head.bias"] = torch.zeros(1000)
        return state_dict

    if kwargs.get("pretrained_cfg", None) is None:
        kwargs["pretrained_cfg"] = {
            # This file is from
            # https://drive.google.com/file/d/1DoDx_70_TLj98gTf6YTXnu4tFhsFocDX/view
            # TODO: once the model weights are available online for download, update
            # this to the url. See https://github.com/Xiyue-Wang/TransPath/issues/14.
            "file": "ctranspath.pth",
            # The head of this model is meant to be overridden.
            "num_classes": 1000,
            "input_size": (3, 224, 224),
            "pool_size": None,
            "crop_pct": 0.9,
            "interpolation": "bicubic",
            "fixed_input_size": True,
            "mean": IMAGENET_DEFAULT_MEAN,
            "std": IMAGENET_DEFAULT_STD,
            "first_conv": "patch_embed.proj",
            "classifier": "head",
        }

    model_kwargs = dict(
        patch_size=4,
        window_size=7,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        embed_layer=ConvStem,
        **kwargs,
    )

    model = build_model_with_cfg(
        SwinTransformerForCTransPath,
        "ctranspath_ssl",
        pretrained,
        pretrained_filter_fn=_checkpoint_filter_fn,
        **model_kwargs,
    )

    return model
