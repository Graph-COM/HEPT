# https://arxiv.org/pdf/2302.01925.pdf
# Adapted from https://github.com/HazyResearch/fly/blob/master/src/models/attention/performer_attention.py
# and https://github.com/lucidrains/performer-pytorch/blob/main/performer_pytorch/performer_pytorch.py


from torch import nn
from einops import rearrange

from functools import partial

from ..model_utils.rff_utils import (
    softmax_kernel,
    linear_attention,
    gaussian_orthogonal_random_matrix,
)

from fast_transformers.feature_maps import RandomFourierFeatures
import torch
import math


class RFF(RandomFourierFeatures):
    def forward(self, x, gamma=1.0):
        x = x * math.sqrt(gamma)
        u = x.unsqueeze(-2).matmul(self.omega).squeeze(-2)
        phi = torch.cat([torch.cos(u), torch.sin(u)], dim=-1)
        return phi * math.sqrt(2 / self.n_dims)


class FLTAttention(nn.Module):
    def __init__(
        self,
        coords_dim,
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
        self.nb_features_inner = kwargs["nb_features_inner"]
        self.out_linear = nn.Linear(self.num_heads * self.dim_per_head, self.dim_per_head)
        self.num_w_per_dist = kwargs["num_w_per_dist"]

        self.rff_kernel_dAngle = RFF(query_dimensions=1, n_dims=self.nb_features_inner, orthogonal=True)
        self.rff_kernel_dAngle.new_feature_map(None)
        self.rff_kernel_dR = RFF(query_dimensions=2, n_dims=self.nb_features_inner, orthogonal=True)
        self.rff_kernel_dR.new_feature_map(None)

        self.ortho_scaling = ortho_scaling
        ncols = self.dim_per_head + coords_dim * self.nb_features_inner
        self.create_projection = partial(
            gaussian_orthogonal_random_matrix, nrows=self.nb_features, ncols=ncols, scaling=ortho_scaling
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

        if self.softmax_temp is None:
            self.softmax_temp = 1 / math.sqrt(query.shape[-1])

        coords = kwargs["coords"]

        w = rearrange(
            kwargs["w_rpe"].weight,
            "(h d) (r c k) -> c h d r k",
            h=self.num_heads,
            d=self.dim_per_head,
            c=2,
            k=int(self.num_w_per_dist // 2),
        )
        alpha, qw = w.sum(dim=2).clamp(max=50).exp().sum(dim=-1)
        new_qw_expand_dim = torch.cat([qw[:, :1], qw], dim=-1)
        sqrt_w_r = torch.sqrt(new_qw_expand_dim)[:, None] * coords[:, None]
        dR = sqrt_w_r[..., :2][..., None, :]
        dAngle = sqrt_w_r[..., 2:][..., None]

        phi_dR = self.rff_kernel_dR(dR)
        phi_dAngle = self.rff_kernel_dAngle(dAngle)
        phi_coords = torch.cat([phi_dR, phi_dAngle], dim=-2) * torch.sqrt(alpha)[:, None, ..., None]
        phi_coords = rearrange(phi_coords, "b h n c d -> b h n (c d)")

        query = torch.cat([query * math.sqrt(self.softmax_temp), phi_coords], dim=-1)
        key = torch.cat([key * math.sqrt(self.softmax_temp), phi_coords], dim=-1)

        create_kernel = partial(
            softmax_kernel,
            projection_matrix=self.projection_matrix,
            softmax_temp=1.0,
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
