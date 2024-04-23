import math
import torch
from torch import nn
from torch_geometric.nn import MLP
from hept import HEPTAttention
from einops import rearrange
from hept_utils import quantile_partition, get_regions


def bit_shift(base, shift_idx):
    max_base = base.max(dim=1, keepdim=True).values
    num_bits = torch.ceil(torch.log2(max_base + 1)).long()
    return (shift_idx << num_bits) | base


def pad_and_unpad(batch, block_size, region_indices, raw_sizes):
    padded_sizes = ((raw_sizes + block_size - 1) // block_size) * block_size
    pad_sizes = padded_sizes - raw_sizes

    pad_cumsum = padded_sizes.cumsum(0)
    pad_seq = torch.arange(pad_cumsum[-1], device=batch.device)
    unpad_seq = torch.ones(pad_cumsum[-1], device=batch.device).bool()

    sorted_region_indices = region_indices.argsort()
    for i in range(len(raw_sizes)):
        idx_to_fill = pad_cumsum[i] - block_size - pad_sizes[i] + torch.arange(pad_sizes[i], device=batch.device)
        if i >= 1:
            pad_seq[pad_cumsum[i - 1] :] -= pad_sizes[i - 1]
            idx_to_fill -= pad_sizes[:i].sum()
        pad_seq[pad_cumsum[i] - pad_sizes[i] : pad_cumsum[i]] = sorted_region_indices[idx_to_fill]
        unpad_seq[pad_cumsum[i] - pad_sizes[i] : pad_cumsum[i]] = False
    return pad_seq, unpad_seq


def prepare_input(x, coords, batch, helper_params):
    kwargs = {}
    regions = rearrange(helper_params["regions"], "c a h -> a (c h)")
    with torch.no_grad():
        block_size, num_heads = helper_params["block_size"], helper_params["num_heads"]
        graph_sizes = batch.bincount()
        graph_size_cumsum = graph_sizes.cumsum(0)

        region_indices_eta, region_indices_phi = [], []
        for graph_idx in range(len(graph_size_cumsum)):
            start_idx = 0 if graph_idx == 0 else graph_size_cumsum[graph_idx - 1]
            end_idx = graph_size_cumsum[graph_idx]
            sorted_eta_idx = torch.argsort(coords[start_idx:end_idx, 0], dim=-1)
            sorted_phi_idx = torch.argsort(coords[start_idx:end_idx, 1], dim=-1)

            region_indices_eta.append(quantile_partition(sorted_eta_idx, regions[0][:, None]))
            region_indices_phi.append(quantile_partition(sorted_phi_idx, regions[1][:, None]))
        region_indices_eta = torch.cat(region_indices_eta, dim=-1)
        region_indices_phi = torch.cat(region_indices_phi, dim=-1)

        combined_shifts = bit_shift(region_indices_eta.long(), region_indices_phi.long())
        combined_shifts = bit_shift(combined_shifts, batch[None])
        combined_shifts = rearrange(combined_shifts, "(c h) n -> c h n", h=num_heads)

        pad_seq, unpad_seq = pad_and_unpad(batch, block_size, combined_shifts[0, 0], graph_sizes)
        x = x[pad_seq]
        kwargs["combined_shifts"] = combined_shifts[..., pad_seq]
        kwargs["coords"] = coords[pad_seq]
    return x, kwargs, unpad_seq


class Transformer(nn.Module):
    def __init__(self, in_dim, coords_dim, num_classes, dropout=0.1, **kwargs):
        super().__init__()
        self.n_layers = kwargs["n_layers"]
        self.h_dim = kwargs["h_dim"]
        self.num_classes = num_classes

        self.feat_encoder = nn.Sequential(
            nn.Linear(in_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
        )

        self.attns = nn.ModuleList()
        for _ in range(self.n_layers):
            self.attns.append(Attn(coords_dim, **kwargs))

        self.dropout = nn.Dropout(dropout)
        self.W = nn.Linear(self.h_dim * (self.n_layers + 1), int(self.h_dim // 2), bias=False)

        self.mlp_out = MLP(
            in_channels=int(self.h_dim // 2),
            out_channels=int(self.h_dim // 2),
            hidden_channels=256,
            num_layers=5,
            norm="layer_norm",
            act="tanh",
            norm_kwargs={"mode": "node"},
        )

        self.helper_params = {}

        self.helper_params["block_size"] = kwargs["block_size"]
        self.regions = nn.Parameter(
            get_regions(kwargs["num_regions"], kwargs["n_hashes"], kwargs["num_heads"]), requires_grad=False
        )
        self.helper_params["regions"] = self.regions
        self.helper_params["num_heads"] = kwargs["num_heads"]

        if self.num_classes:
            self.out_proj = nn.Linear(int(self.h_dim // 2), num_classes)

    def forward(self, x, coords, batch):
        x, kwargs, unpad_seq = prepare_input(x, coords, batch, self.helper_params)

        encoded_x = self.feat_encoder(x)
        all_encoded_x = [encoded_x]
        for i in range(self.n_layers):
            encoded_x = self.attns[i](encoded_x, kwargs)
            all_encoded_x.append(encoded_x)

        encoded_x = self.W(torch.cat(all_encoded_x, dim=-1))
        out = encoded_x + self.dropout(self.mlp_out(encoded_x))

        if self.num_classes:
            out = self.out_proj(out)
        return out[unpad_seq]


class Attn(nn.Module):
    def __init__(self, coords_dim, **kwargs):
        super().__init__()
        self.dim_per_head = kwargs["h_dim"]
        self.num_heads = kwargs["num_heads"]

        self.w_q = nn.Linear(self.dim_per_head, self.dim_per_head * self.num_heads, bias=False)
        self.w_k = nn.Linear(self.dim_per_head, self.dim_per_head * self.num_heads, bias=False)
        self.w_v = nn.Linear(self.dim_per_head, self.dim_per_head * self.num_heads, bias=False)

        # +2 for data.pos
        self.attn = HEPTAttention(self.dim_per_head + coords_dim, **kwargs)

        self.dropout = nn.Dropout(0.1)
        self.norm1 = nn.LayerNorm(self.dim_per_head)
        self.norm2 = nn.LayerNorm(self.dim_per_head)
        self.ff = nn.Sequential(
            nn.Linear(self.dim_per_head, self.dim_per_head),
            nn.ReLU(),
            nn.Linear(self.dim_per_head, self.dim_per_head),
        )

        # eta/phi from data.pos use the same weights as they are used to calc dR
        self.w_rpe = nn.Linear(kwargs["num_w_per_dist"] * (coords_dim - 1), self.num_heads * self.dim_per_head)

    def forward(self, x, kwargs):
        x_normed = self.norm1(x)
        q, k, v = self.w_q(x_normed), self.w_k(x_normed), self.w_v(x_normed)
        aggr_out = self.attn(q, k, v, pe=kwargs["coords"], w_rpe=self.w_rpe, **kwargs)

        x = x + self.dropout(aggr_out)
        ff_output = self.ff(self.norm2(x))
        x = x + self.dropout(ff_output)

        return x
