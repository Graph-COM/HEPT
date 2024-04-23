import torch
import torch.nn as nn
from einops import rearrange
from hept_utils import E2LSH, invert_permutation, lsh_mapping, sort_to_buckets, unsort_from_buckets


def qkv_res(s_query, s_key, s_value):
    q_sq_05 = -0.5 * (s_query**2).sum(dim=-1, keepdim=True)
    k_sq_05 = -0.5 * (s_key**2).sum(dim=-1, keepdim=True)

    clustered_dists = torch.einsum("...id,...jd->...ij", s_query, s_key)
    clustered_dists = (clustered_dists + q_sq_05 + k_sq_05.transpose(-1, -2)).clamp(max=0.0).exp()

    denom = clustered_dists.sum(dim=-1, keepdim=True) + (1e-20)
    qk = clustered_dists

    so = torch.einsum("...ij,...jd->...id", qk, s_value)
    return denom, so


def prep_qk(query, key, w, coords):
    qw = w.sum(dim=1).clamp(max=50).exp().sum(dim=-1)
    new_qw_expand_dim = torch.cat([qw[:, :1], qw], dim=-1)

    sqrt_w_r = torch.sqrt(2 * new_qw_expand_dim)[None] * coords[:, None]
    q_hat = torch.cat([query, sqrt_w_r], dim=-1)
    k_hat = torch.cat([key, sqrt_w_r], dim=-1)
    return q_hat, k_hat


class HEPTAttention(nn.Module):
    def __init__(self, hash_dim, **kwargs):
        super().__init__()
        self.dim_per_head = kwargs["h_dim"]
        self.num_heads = kwargs["num_heads"]
        self.out_linear = nn.Linear(self.num_heads * self.dim_per_head, self.dim_per_head)

        self.block_size = kwargs["block_size"]
        self.n_hashes = kwargs["n_hashes"]
        self.num_w_per_dist = kwargs["num_w_per_dist"]
        self.e2lsh = E2LSH(n_hashes=self.n_hashes, n_heads=self.num_heads, dim=hash_dim)

    def forward(self, query, key, value, **kwargs):
        query = query.view(-1, self.num_heads, self.dim_per_head)
        key = key.view(-1, self.num_heads, self.dim_per_head)
        value = value.view(-1, self.num_heads, self.dim_per_head)

        w = rearrange(
            kwargs["w_rpe"].weight,
            "(h d) (r k) -> h d r k",
            h=self.num_heads,
            d=self.dim_per_head,
            k=self.num_w_per_dist,
        )
        q_hat, k_hat = prep_qk(query, key, w, kwargs["coords"])

        q_hat = rearrange(q_hat, "n h d -> h n d")
        k_hat = rearrange(k_hat, "n h d -> h n d")
        value = rearrange(value, "n h d -> h n d")

        q_hashed, k_hashed, hash_shift = lsh_mapping(self.e2lsh, q_hat, k_hat)

        combined_shifts = kwargs["combined_shifts"] * hash_shift
        q_hashed = q_hashed + combined_shifts
        k_hashed = k_hashed + combined_shifts

        q_positions = q_hashed.argsort(dim=-1)
        k_positions = k_hashed.argsort(dim=-1)

        s_query = sort_to_buckets(q_hat, q_positions, self.block_size)
        s_key = sort_to_buckets(k_hat, k_positions, self.block_size)
        s_value = sort_to_buckets(value, k_positions, self.block_size)

        denom, so = qkv_res(s_query, s_key, s_value)

        q_rev_positions = invert_permutation(q_positions)
        o = unsort_from_buckets(so, q_rev_positions)
        logits = unsort_from_buckets(denom, q_rev_positions)
        out = o.sum(dim=0) / logits.sum(dim=0)
        out = self.out_linear(rearrange(out, "h n d -> n (h d)"))
        return out
