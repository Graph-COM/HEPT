# https://github.com/mit-han-lab/flatformer

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange


def qkv_res(s_query, s_key, s_value):
    q_sq_05 = -0.5 * (s_query**2).sum(dim=-1, keepdim=True)
    k_sq_05 = -0.5 * (s_key**2).sum(dim=-1, keepdim=True)

    clustered_dists = torch.einsum("...id,...jd->...ij", s_query, s_key)
    clustered_dists = (clustered_dists + q_sq_05 + k_sq_05.transpose(-1, -2)).clamp(max=0.0).exp()

    denom = clustered_dists.sum(dim=-1, keepdim=True) + (1e-20)
    qk = clustered_dists / (denom)

    so = torch.einsum("...ij,...jd->...id", qk, s_value)
    return denom, so


def prep_qk(query, key, w, coords, num_groups, group_size):
    qw = w.sum(dim=1).clamp(max=50).exp().sum(dim=-1)
    new_qw_expand_dim = torch.cat([qw[:, :1], qw], dim=-1)

    sqrt_w_r = torch.sqrt(2 * new_qw_expand_dim)[None] * coords[:, None]
    sqrt_w_r = rearrange(sqrt_w_r, "(b n) h d -> b h n d", b=num_groups, n=group_size)
    q_hat = torch.cat([query, sqrt_w_r], dim=-1)
    k_hat = torch.cat([key, sqrt_w_r], dim=-1)
    return q_hat, k_hat


class GroupAttention(nn.Module):
    def __init__(self, in_channels: int, num_heads: int, group_size: int, num_w_per_dist, softmax_temp=None) -> None:
        super().__init__()
        self.group_size = group_size
        self.w_q = nn.Linear(in_channels, in_channels * num_heads, bias=False)
        self.w_k = nn.Linear(in_channels, in_channels * num_heads, bias=False)
        self.w_v = nn.Linear(in_channels, in_channels * num_heads, bias=False)
        self.dim_per_head = in_channels
        self.num_heads = num_heads
        self.num_w_per_dist = num_w_per_dist
        self.softmax_temp = softmax_temp
        self.out_linear = nn.Linear(self.num_heads * self.dim_per_head, self.dim_per_head)

    def forward(self, x, pe, w_rpe, pe_type):
        size = x.shape[0]
        num_groups = int(math.ceil(size / self.group_size))

        v = x
        if pe_type == "rpe":
            q = k = x
        else:
            q = k = x + pe
        query, key, value = self.w_q(q), self.w_k(k), self.w_v(v)

        if pe_type == "rpe":
            query = rearrange(query, "(b n) (h d) -> b h n d", b=num_groups, n=self.group_size, h=self.num_heads)
            key = rearrange(key, "(b n) (h d) -> b h n d", b=num_groups, n=self.group_size, h=self.num_heads)
            value = rearrange(value, "(b n) (h d) -> b h n d", b=num_groups, n=self.group_size, h=self.num_heads)

            w_rpe = rearrange(w_rpe.weight, "(h d) (r k) -> h d r k", h=self.num_heads, d=self.dim_per_head, k=self.num_w_per_dist)
            q_hat, k_hat = prep_qk(query, key, w_rpe, pe, num_groups, self.group_size)
            _, x = qkv_res(q_hat, k_hat, value)
            x = rearrange(x, "b h n d -> (b n) (h d)")
        else:
            query = rearrange(query, "(b n) (h d) -> b n h d", b=num_groups, n=self.group_size, h=self.num_heads)
            key = rearrange(key, "(b n) (h d) -> b n h d", b=num_groups, n=self.group_size, h=self.num_heads)
            value = rearrange(value, "(b n) (h d) -> b n h d", b=num_groups, n=self.group_size, h=self.num_heads)

            softmax_temp = self.softmax_temp or 1 / math.sqrt(query.shape[-1])
            query = query * softmax_temp
            QK = torch.einsum("bthe,bshe->bhts", query, key)
            attn = torch.softmax(QK, dim=-1)
            x = torch.einsum("bhts,bshd->bthd", attn, value)
            x = rearrange(x, "b n h d -> (b n) (h d)")

        x = self.out_linear(x)
        return x


class BasicLayer(nn.Module):
    def __init__(self, in_channels, num_heads, activation, group_size, num_w_per_dist) -> None:
        super().__init__()
        self.attn = GroupAttention(in_channels, num_heads, group_size, num_w_per_dist)

        self.fc1 = nn.Linear(in_channels, 2 * in_channels)
        self.fc2 = nn.Linear(2 * in_channels, in_channels)

        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)

        self.act = _get_activation_fn(activation)

    def forward(self, src, pe, w_rpe, pe_type):
        src = self.norm1(src + self.attn(src, pe, w_rpe, pe_type))
        src = self.norm2(src + self.fc2(self.act(self.fc1(src))))

        return src


class FlatformerAttention(nn.Module):
    def __init__(
        self,
        activation="relu",
        **kwargs,
    ) -> None:
        super().__init__()
        num_heads = kwargs["num_heads"]
        in_channels = kwargs["h_dim"]
        group_size = kwargs["group_size"]
        self.pe_type = kwargs["pe_type"]

        self.block = nn.ModuleList()
        for _ in range(4):
            layer = BasicLayer(
                in_channels,
                num_heads,
                activation,
                group_size,
                num_w_per_dist=kwargs["num_w_per_dist"],
            )
            self.block.append(layer)

    def forward(self, x: torch.Tensor, pe: torch.Tensor, mappings: Dict[str, Any], **kwargs) -> torch.Tensor:
        all_x = []
        for k, name in enumerate(["x", "x_shift", "y", "y_shift"]):
            indices = mappings[name]
            x[indices] = self.block[k](
                x[indices][mappings["flat2win"]],
                pe[indices][mappings["flat2win"]],
                kwargs["w_rpe"],
                self.pe_type,
            )[mappings["win2flat"]]
            all_x.append(x)
        return x, all_x


def _get_activation_fn(activation):
    if activation == "relu":
        return torch.nn.functional.relu
    if activation == "gelu":
        return torch.nn.functional.gelu
    if activation == "glu":
        return torch.nn.functional.glu
