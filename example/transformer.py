import torch
from torch import nn
from torch_geometric.nn import MLP
from hept import HEPTAttention
from torch.utils.checkpoint import checkpoint
from einops import rearrange
from hept_utils import quantile_partition, get_regions, pad_to_multiple


def prepare_input(x, coords, edge_index, batch, attn_type, helper_funcs):
    kwargs = {}
    assert batch.max() == 0
    key_padding_mask = None
    mask = None
    kwargs["key_padding_mask"] = key_padding_mask
    kwargs["edge_index"] = edge_index
    kwargs["coords"] = coords

    with torch.no_grad():
        block_size = helper_funcs["block_size"]
        kwargs["raw_size"] = x.shape[0]
        x = pad_to_multiple(x, block_size, dims=0)
        kwargs["coords"] = pad_to_multiple(kwargs["coords"], block_size, dims=0, value=float("inf"))
        sorted_eta_idx = torch.argsort(kwargs["coords"][..., 0], dim=-1)
        sorted_phi_idx = torch.argsort(kwargs["coords"][..., 1], dim=-1)
        regions = helper_funcs["regions"]
        regions_h = rearrange(regions, "c a h -> a (c h)")
        region_indices_eta = quantile_partition(sorted_eta_idx, regions_h[0][:, None])
        region_indices_phi = quantile_partition(sorted_phi_idx, regions_h[1][:, None])
        kwargs["region_indices"] = [region_indices_eta, region_indices_phi]
        kwargs["regions_h"] = regions_h
        kwargs["coords"][kwargs["raw_size"] :] = 0.0
    return x, mask, kwargs


class Transformer(nn.Module):
    def __init__(self, attn_type, in_dim, coords_dim, task, **kwargs):
        super().__init__()
        self.attn_type = attn_type
        self.n_layers = kwargs["n_layers"]
        self.h_dim = kwargs["h_dim"]
        self.task = task
        self.use_ckpt = kwargs.get("use_ckpt", False)

        # discrete feature to embedding
        if self.task == "pileup":
            self.pids_enc = nn.Embedding(7, 10)
            in_dim = in_dim - 1 + 10

        self.feat_encoder = nn.Sequential(
            nn.Linear(in_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
        )

        self.attns = nn.ModuleList()
        for _ in range(self.n_layers):
            self.attns.append(Attn(attn_type, coords_dim, **kwargs))

        self.dropout = nn.Dropout(0.1)
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

        self.helper_funcs = {}

        self.helper_funcs["block_size"] = kwargs["block_size"]
        self.regions = nn.Parameter(get_regions(kwargs["num_regions"], kwargs["n_hashes"], kwargs["num_heads"]), requires_grad=False)
        self.helper_funcs["regions"] = self.regions

        if self.task == "pileup":
            self.out_proj = nn.Linear(int(self.h_dim // 2), 1)

    def forward(self, data):
        if isinstance(data, dict):
            x, edge_index, coords, batch, self.use_ckpt = data["x"], data["edge_index"], data["coords"], data["batch"], False
        else:
            x, edge_index, coords, batch = data.x, data.edge_index, data.coords, data.batch

        # discrete feature to embedding
        if self.task == "pileup":
            pids_emb = self.pids_enc(x[..., -1].long())
            x = torch.cat((x[..., :-1], pids_emb), dim=-1)

        x, mask, kwargs = prepare_input(x, coords, edge_index, batch, self.attn_type, self.helper_funcs)

        encoded_x = self.feat_encoder(x)
        all_encoded_x = [encoded_x]
        for i in range(self.n_layers):
            if self.use_ckpt:
                encoded_x = checkpoint(self.attns[i], encoded_x, kwargs)
            else:
                encoded_x = self.attns[i](encoded_x, kwargs)
            all_encoded_x.append(encoded_x)

        encoded_x = self.W(torch.cat(all_encoded_x, dim=-1))
        out = encoded_x + self.dropout(self.mlp_out(encoded_x))

        if kwargs.get("raw_size", False):
            out = out[: kwargs["raw_size"]]

        if mask is not None:
            out = out[mask]

        if self.task == "pileup":
            out = self.out_proj(out)
            out = torch.sigmoid(out)

        return out


class Attn(nn.Module):
    def __init__(self, attn_type, coords_dim, **kwargs):
        super().__init__()
        self.attn_type = attn_type
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
        pe = kwargs["coords"]
        x_normed = self.norm1(x)
        q, k, v = self.w_q(x_normed), self.w_k(x_normed), self.w_v(x_normed)
        aggr_out = self.attn(q, k, v, pe=pe, w_rpe=self.w_rpe, **kwargs)

        x = x + self.dropout(aggr_out)
        ff_output = self.ff(self.norm2(x))
        x = x + self.dropout(ff_output)

        return x
