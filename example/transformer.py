import torch
from torch import nn
from torch_geometric.nn import MLP
from hept import HEPTAttention
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from einops import rearrange


class PE(nn.Module):
    def __init__(self, pos_temperature=10000, normalize_pos=False, **kwargs):
        super().__init__()
        self.feat_dim = kwargs["h_dim"]
        self.normalize_pos = normalize_pos
        self.pos_temperature = pos_temperature

    def forward(self, coors):
        assert not self.normalize_pos
        dtype = torch.float32

        dis_coords = discretize_coords(coors[..., :2], B=1000)
        x, y = dis_coords[..., 0], dis_coords[..., 1]

        inv_freq = self.inv_freq

        # [num_tokens, pos_length]
        pex = x[..., None] / inv_freq(coors.device)[None, :]
        pey = y[..., None] / inv_freq(coors.device)[None, :]

        # [num_tokens, pos_length]
        pex = torch.stack([pex[..., ::2].sin(), pex[..., 1::2].cos()], dim=-1).flatten(-2)
        pey = torch.stack([pey[..., ::2].sin(), pey[..., 1::2].cos()], dim=-1).flatten(-2)
        pe = torch.cat([pex, pey], dim=-1).to(dtype)

        gap = self.feat_dim - pe.size(-1)
        if gap > 0:
            pe_shape = list(pe.shape)
            pe_shape[-1] = gap
            pe_p = torch.zeros(pe_shape, dtype=dtype, device=coors.device)
            pe = torch.cat([pe, pe_p], dim=1)

        return pe

    def inv_freq(self, device):
        ndim = 2
        pos_length = (self.feat_dim // (ndim * 2)) * 2

        # [pos_length]
        inv_freq = torch.arange(pos_length, dtype=torch.float32, device=device)
        inv_freq = self.pos_temperature ** (2 * (inv_freq // 2) / pos_length)
        return inv_freq


class PELearned(nn.Module):
    """
    https://github.com/Haiyang-W/DSVT
    Absolute pos embedding, learned.
    """

    def __init__(self, input_channel, **kwargs):
        super().__init__()
        num_pos_feats = kwargs["h_dim"]
        self.position_embedding_head = nn.Sequential(
            nn.Linear(input_channel, num_pos_feats),
            nn.LayerNorm(num_pos_feats),
            nn.ReLU(),
            nn.Linear(num_pos_feats, num_pos_feats),
        )

    def forward(self, xyz):
        position_embedding = self.position_embedding_head(xyz)
        return position_embedding


def get_pe_func(pe_type, coords_size, kwargs):
    if pe_type == "learned":
        return PELearned(input_channel=coords_size, **kwargs)
    elif pe_type == "fixed":
        return PE(**kwargs)
    else:
        return None



def discretize_coords(coords, B):
    min_vals, _ = torch.min(coords, dim=-2)
    max_vals, _ = torch.max(coords, dim=-2)
    ranges = max_vals - min_vals
    bucket_size = ranges / B

    # Normalize each dimension to [0, 1] and then scale to [0, B)
    # Subtract min, divide by range, then multiply by number of buckets
    # Floor the values to get the bucket indices
    buckets = torch.floor((coords - min_vals) / bucket_size)

    # Ensure the maximum value falls into the last bucket
    coords = torch.clamp(buckets, 0, B - 1)
    return coords


def quantile_binning(sorted_indices, num_bins):
    total_elements = sorted_indices.shape[-1]
    bin_size = torch.ceil(total_elements / num_bins)
    inverse_indices = torch.argsort(sorted_indices, dim=-1)

    base = torch.arange(total_elements, device=sorted_indices.device)[None]
    bin_indices = base // bin_size + 1
    reassigned_bins = bin_indices[:, inverse_indices]
    return reassigned_bins


def get_bins(num_buckets, num_or_hashes, num_heads, num_and_hashes=2):
    lb = 2
    ub = 2 * num_buckets ** (1 / num_and_hashes) - lb
    bins = []
    for _ in range(num_or_hashes * num_heads):
        bin = []
        for _ in range(num_and_hashes):
            a = torch.rand(1).item() * (ub - lb) + lb
            bin.append(a)
        bins.append(bin)
    bins = torch.tensor(bins)
    bins = (num_buckets / bins.prod(dim=1, keepdim=True)) ** (1 / num_and_hashes) * bins

    bins = torch.round(bins * 3) / 3
    return rearrange(bins, "(h c) a -> c a h", h=num_heads)


def pad_to_multiple(tensor, multiple, dims=-1, value=0):
    try:
        dims = list(dims)  # If dims is an iterable (e.g., List, Tuple)
    except:
        dims = [dims]
    # convert dims from negative to positive
    dims = [d if d >= 0 else tensor.ndim + d for d in dims]
    padding = [0] * (2 * tensor.ndim)
    for d in dims:
        size = tensor.size(d)
        # Pytorch's JIT doesn't like divmod
        # m, remainder = divmod(size, multiple)
        m = size // multiple
        remainder = size - m * multiple
        if remainder != 0:
            padding[2 * (tensor.ndim - d - 1) + 1] = multiple - remainder
    if all(p == 0 for p in padding):
        return tensor
    else:
        return F.pad(tensor, tuple(padding), value=value)


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
        bins = helper_funcs["bins"]
        bins_h = rearrange(bins, "c a h -> a (c h)")
        bin_indices_eta = quantile_binning(sorted_eta_idx, bins_h[0][:, None])
        bin_indices_phi = quantile_binning(sorted_phi_idx, bins_h[1][:, None])
        kwargs["bin_indices"] = [bin_indices_eta, bin_indices_phi]
        kwargs["bins_h"] = bins_h
        kwargs["coords"][kwargs["raw_size"]:] = 0.0


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
        self.bins = nn.Parameter(get_bins(kwargs["num_buckets"], kwargs["n_hashes"], kwargs["num_heads"]), requires_grad=False)
        self.helper_funcs["bins"] = self.bins

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
            out = out[:kwargs["raw_size"]]

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

        # eta/phi from data.pos use the same weights as they are used to calc dR
        self.w_rpe = nn.Linear(kwargs["num_w_per_dist"] * (coords_dim - 1), self.num_heads * self.dim_per_head)
        self.pe_func = get_pe_func(kwargs["pe_type"], coords_dim, kwargs)

    def forward(self, x, kwargs):
        pe = kwargs["coords"] if self.pe_func is None else self.pe_func(kwargs["coords"])

        x_pe = x + pe if self.pe_func is not None else x
        x_normed = self.norm1(x_pe)
        q, k, v = self.w_q(x_normed), self.w_k(x_normed), self.w_v(x_normed)
        aggr_out = self.attn(q, k, v, pe=pe, w_rpe=self.w_rpe, **kwargs)

        x = x + self.dropout(aggr_out)
        ff_output = self.ff(self.norm2(x))
        x = x + self.dropout(ff_output)

        return x
