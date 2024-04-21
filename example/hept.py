import torch
import torch.nn as nn
from typing import List
from einops import rearrange


def uniform(a, b, shape, device='cpu'):
    '''
        Draws shape samples from a uniform distribution U(a, b).

    '''
    return (b - a) * torch.rand(shape, device=device) + a


class E2LSH(nn.Module):
    def __init__(self, n_hashes, n_heads, dim, r=1):
        super(E2LSH, self).__init__()

        self.alpha = nn.Parameter(torch.normal(0, 1, (n_heads, dim, n_hashes)))
        self.beta = nn.Parameter(uniform(0, r, shape=(1, n_hashes)))
        self.alpha.requires_grad = False
        self.beta.requires_grad = False

    def forward(self, vecs):
        projection = torch.bmm(vecs, self.alpha)
        return projection.permute(2, 0, 1)


def invert_permutation(perm: torch.Tensor) -> torch.Tensor:
    """
    Params:
        perm: (..., n)
    Return:
        inverse_perm: (..., n)
    """
    # This is simpler but has complexity O(n log n)
    # return torch.argsort(perm, dim=-1)
    # This is more complicated but has complexity O(n)
    arange = torch.arange(perm.shape[-1], device=perm.device).expand_as(perm)
    return torch.empty_like(perm).scatter_(-1, perm, arange)


@torch.no_grad()
def lsh_clustering(e2lsh, queries, keys, block_size, r=1):
    """
        LSH clustering based on Euclidean distance.
    """
    queries_hashed = e2lsh(queries)
    keys_hashed = e2lsh(keys)
    hash_shift = max(queries_hashed.max(), keys_hashed.max()) - min(queries_hashed.min(), keys_hashed.min())
    return queries_hashed, keys_hashed, hash_shift


def batched_index_select(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """
    Params:
        values: (1 or n_hashes, batch, seqlen, dim)
        indices: (n_hashes, batch, seqlen)
    Return:
        (n_hashes, batch, seqlen, dim)
    """
    last_dim = values.shape[-1]
    indices_expanded = rearrange(indices, '... -> ... 1').expand(*indices.shape, last_dim)
    return values.expand(*indices_expanded.shape[:-2],
                         *values.shape[-2:]).gather(-2, indices_expanded)


def sort_to_buckets(x, perm, bucketsz):
    return rearrange(
        batched_index_select(rearrange(x, "b s d -> 1 b s d"), perm),
        "h b (nbuckets bucketsz) d -> h b nbuckets bucketsz d",
        bucketsz=bucketsz,
    )


def unsort_from_buckets(s_x, perm_inverse):
    b_x = rearrange(s_x, "h b nbuckets bucketsz d -> h b (nbuckets bucketsz) d")
    return batched_index_select(b_x, perm_inverse)


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


@torch.no_grad()
def get_geo_shift(bins_h: List[List[int]], hash_shift, bin_indices):
    bin_indices_eta, bin_indices_phi = bin_indices

    q_hash_shift_eta = bin_indices_eta * hash_shift
    k_hash_shift_eta = bin_indices_eta * hash_shift

    q_hash_shift_phi = bin_indices_phi * hash_shift * (torch.ceil(bins_h[0][:, None]) + 1)
    k_hash_shift_phi = bin_indices_phi * hash_shift * (torch.ceil(bins_h[0][:, None]) + 1)
    res = torch.stack([q_hash_shift_phi + q_hash_shift_eta, k_hash_shift_phi + k_hash_shift_eta], dim=0)
    return rearrange(res, "a (c h) n -> a c h n", c=3)


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
        # TODO: support batched inputs
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
        q_hat[:, kwargs["raw_size"] :] = 0.0
        k_hat[:, kwargs["raw_size"] :] = 0.0
        value[:, kwargs["raw_size"] :] = 0.0

        q_hashed, k_hashed, hash_shift = lsh_clustering(self.e2lsh, q_hat, k_hat, self.block_size, r=1)
        q_hashed[..., kwargs["raw_size"]:] = float("inf")
        k_hashed[..., kwargs["raw_size"]:] = float("inf")

        q_shifts, k_shifts = get_geo_shift(kwargs["bins_h"], hash_shift, kwargs["bin_indices"])

        q_hashed = q_hashed + q_shifts
        k_hashed = k_hashed + k_shifts

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
