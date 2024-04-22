import torch
from torch import nn
from torch_geometric.nn import MLP
from ..attention import (
    PerformerAttention,
    HEPTAttention,
    ReformerAttention,
    SmyrfAttention,
    SBAttention,
    FLTAttention,
    PCTAttention,
    FlatformerAttention,
)
from ..model_utils.mask_utils import FullMask
from ..model_utils.hash_utils import pad_to_multiple, get_regions, quantile_partition
from ..model_utils.window_utils import discretize_coords, FlattenedWindowMapping, get_pe_func
from torch_geometric.utils import to_dense_batch
from torch.utils.checkpoint import checkpoint
from einops import rearrange


def prepare_input(x, coords, edge_index, batch, attn_type, helper_funcs):
    kwargs = {}
    if attn_type not in ["pct", "flatformer", "hept"]:
        edge_index = None
        x, mask = to_dense_batch(x, batch)
        coords = to_dense_batch(coords, batch)[0]
        key_padding_mask = FullMask(mask)
    else:
        assert batch.max() == 0
        key_padding_mask = None
        mask = None
    kwargs["key_padding_mask"] = key_padding_mask
    kwargs["edge_index"] = edge_index
    kwargs["coords"] = coords

    if attn_type in ["flatformer"]:
        discretized_coords = torch.zeros((x.shape[0], 4), device=x.device)
        discretized_coords[:, -2:] = discretize_coords(coords[:, :2], B=helper_funcs["B"])
        mappings = helper_funcs["mapping"](discretized_coords, batch_size=1)
        kwargs["mappings"] = mappings

    if attn_type in ["hept"]:
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

    if attn_type in ["smyrf"] and "rpe" in helper_funcs["pe_type"]:
        rpe_ones_shape = (*x.shape[:-1], helper_funcs["num_heads"], 1)
        kwargs["rpe_ones"] = torch.ones(rpe_ones_shape, device=x.device)

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
        if self.attn_type == "flatformer":
            self.helper_funcs["B"] = kwargs["B"]
            self.helper_funcs["mapping"] = FlattenedWindowMapping(**kwargs)
            self.W = nn.Linear(self.h_dim * (self.n_layers * 4 + 1), int(self.h_dim // 2), bias=False)
        elif self.attn_type == "hept":
            self.helper_funcs["block_size"] = kwargs["block_size"]
            self.regions = nn.Parameter(get_regions(kwargs["num_regions"], kwargs["n_hashes"], kwargs["num_heads"]), requires_grad=False)
            self.helper_funcs["regions"] = self.regions

        elif self.attn_type == "smyrf":
            self.helper_funcs["num_heads"], self.helper_funcs["pe_type"] = kwargs["num_heads"], kwargs["pe_type"]

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
            if self.attn_type in ["flatformer"]:
                encoded_x, all_shift_x = self.attns[i](encoded_x, kwargs)
                all_encoded_x = all_encoded_x + all_shift_x
            else:
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

        if self.attn_type not in ["pct", "flatformer"]:
            self.w_q = nn.Linear(self.dim_per_head, self.dim_per_head * self.num_heads, bias=False)
            self.w_k = nn.Linear(self.dim_per_head, self.dim_per_head * self.num_heads, bias=False)
            self.w_v = nn.Linear(self.dim_per_head, self.dim_per_head * self.num_heads, bias=False)

        if attn_type == "hept":
            # +2 for data.pos
            self.attn = HEPTAttention(self.dim_per_head + coords_dim, **kwargs)
        elif attn_type == "performer":
            self.attn = PerformerAttention(coords_dim=coords_dim, **kwargs)
        elif attn_type == "reformer":
            self.attn = ReformerAttention(**kwargs)
        elif attn_type == "smyrf":
            self.attn = SmyrfAttention(**kwargs)
        elif attn_type == "sb":
            self.attn = SBAttention(**kwargs)
        elif attn_type == "flt":
            # eta/phi from data.pos use the same weights as they are used to calc dR
            self.attn = FLTAttention(coords_dim - 1, **kwargs)
        elif attn_type == "pct":
            self.attn = PCTAttention(coords_dim, **kwargs)
            self.w_q = nn.Linear(self.dim_per_head, self.dim_per_head * self.num_heads, bias=False)
        elif attn_type == "flatformer":
            self.attn = FlatformerAttention(**kwargs)
        else:
            raise NotImplementedError

        if self.attn_type not in ["flatformer"]:
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
        self.pe_func = get_pe_func(kwargs["pe_type"], coords_dim, kwargs)

    def forward(self, x, kwargs):
        pe = kwargs["coords"] if self.pe_func is None else self.pe_func(kwargs["coords"])
        if self.attn_type not in ["pct", "flatformer"]:
            x_pe = x + pe if self.pe_func is not None else x
            x_normed = self.norm1(x_pe)
            q, k, v = self.w_q(x_normed), self.w_k(x_normed), self.w_v(x_normed)
            aggr_out = self.attn(q, k, v, pe=pe, w_rpe=self.w_rpe, **kwargs)

            x = x + self.dropout(aggr_out)
            ff_output = self.ff(self.norm2(x))
            x = x + self.dropout(ff_output)

        elif self.attn_type == "pct":
            aggr_out = self.attn(self.w_q(self.norm1(x)), w_rpe=self.w_rpe, **kwargs)
            x = x + self.dropout(aggr_out)
            ff_output = self.ff(self.norm2(x))
            x = x + self.dropout(ff_output)

        elif self.attn_type == "flatformer":
            x = self.attn(x, pe=pe, w_rpe=self.w_rpe, **kwargs)

        return x
