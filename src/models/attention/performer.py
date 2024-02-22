# Adapted from https://github.com/HazyResearch/fly/blob/master/src/models/attention/performer_attention.py
# and https://github.com/lucidrains/performer-pytorch/blob/main/performer_pytorch/performer_pytorch.py


import torch
from torch import nn
from einops import rearrange

from functools import partial

from ..model_utils.rff_utils import (
    softmax_kernel,
    linear_attention,
    gaussian_orthogonal_random_matrix,
)
from math import log
from fast_transformers.feature_maps import Favor as BaseFavor


class Favor(BaseFavor):
    def forward(self, x, offset):
        u = x.unsqueeze(-2).matmul(self.omega).squeeze(-2)
        offset = - 0.5 * log(self.n_dims) + offset

        exp_u1 = torch.exp(u + offset)
        exp_u2 = torch.exp(-u + offset)
        phi = torch.cat([exp_u1, exp_u2], dim=-1)
        return phi


class PerformerAttention(nn.Module):
    def __init__(
        self,
        ortho_scaling=0,
        softmax_temp=None,
        softmax_eps=1e-6,
        normalization_eps=1e-6,
        **kwargs,
    ):
        super().__init__()
        self.dim_per_head = kwargs["h_dim"]
        self.num_heads = kwargs["num_heads"]
        self.nb_features = kwargs["nb_features"]
        self.pe_type = kwargs["pe_type"]
        self.num_w_per_dist = kwargs["num_w_per_dist"]
        self.out_linear = nn.Linear(self.num_heads * self.dim_per_head, self.dim_per_head)

        if self.pe_type == "rpe":
            self.kernel = Favor(self.dim_per_head + kwargs["coords_dim"], self.nb_features, orthogonal=True)
            self.kernel.new_feature_map(None)
        else:
            self.ortho_scaling = ortho_scaling
            self.create_projection = partial(
                gaussian_orthogonal_random_matrix, nrows=self.nb_features, ncols=self.dim_per_head, scaling=ortho_scaling
            )
            projection_matrix = self.create_projection()
            self.register_buffer("projection_matrix", projection_matrix)

        self.softmax_temp = softmax_temp
        self.softmax_eps = softmax_eps
        self.normalization_eps = normalization_eps

    def forward(self, query, key, value, **kwargs):
        key_padding_mask = kwargs["key_padding_mask"]
        query = rearrange(query, "b n (h d) -> b h n d", h=self.num_heads, d=self.dim_per_head)
        key = rearrange(key, "b n (h d) -> b h n d", h=self.num_heads, d=self.dim_per_head)
        value = rearrange(value, "b n (h d) -> b h n d", h=self.num_heads, d=self.dim_per_head)

        if self.pe_type == "rpe":
            w = rearrange(kwargs["w_rpe"].weight, "(h d) (r k) -> h d r k", h=self.num_heads, d=self.dim_per_head, k=self.num_w_per_dist)
            qw = w.sum(dim=1).clamp(max=50).exp().sum(dim=-1)
            new_qw_expand_dim = torch.cat([qw[:, :1], qw], dim=-1)
            sqrt_w_r = torch.sqrt(2 * new_qw_expand_dim)[None] * kwargs["pe"][:, :, None]
            sqrt_w_r = rearrange(sqrt_w_r, "b n h d -> b h n d")
            query_sq = -0.5 * (query ** 2).sum(dim=-1, keepdim=True)
            key_sq = -0.5 * (key ** 2).sum(dim=-1, keepdim=True)
            w_r = -(sqrt_w_r ** 2).sum(dim=-1, keepdim=True)
            query_offset = query_sq + w_r
            key_offset = key_sq + w_r
            query = torch.cat([query, sqrt_w_r], dim=-1)
            key = torch.cat([key, sqrt_w_r], dim=-1)
            query = self.kernel(query, query_offset)
            key = self.kernel(key, key_offset)
        else:
            create_kernel = partial(
                softmax_kernel,
                projection_matrix=self.projection_matrix,
                softmax_temp=self.softmax_temp,
                eps=self.softmax_eps,
            )
            query = create_kernel(query, is_query=True)
            key = create_kernel(key, is_query=False)

        if key_padding_mask is not None and not key_padding_mask.all_ones:
            key = key.masked_fill(rearrange(~key_padding_mask.bool_matrix, "b s -> b 1 s 1"), 0.0)

        out, attn = linear_attention(query, key, value, eps=self.normalization_eps)
        out = rearrange(out, "b h s d -> b s (h d)")
        out = self.out_linear(out)
        return out
