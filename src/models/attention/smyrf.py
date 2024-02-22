# https://github.com/HazyResearch/fly/blob/master/src/models/attention/smyrf_attention.py
# Adapted from https://github.com/giannisdaras/smyrf/blob/master/smyrf/torch/attn.py
import math
import torch
import torch.nn as nn

from einops import rearrange, repeat
from ..model_utils.hash_utils import batched_index_select, invert_permutation, pad_to_multiple, XBOXPLUS, LSH, uniform
from ..model_utils.mask_utils import LengthMask, pad_mask


def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max


def lsh_clustering(queries, keys, n_hashes, r=1, key_padding_mask=None):
    """
    LSH clustering based on Euclidean distance.
    """

    e2lsh = E2LSH(n_hashes=n_hashes, dim=queries.shape[-1], r=r, device=queries.device)
    queries_hashed = e2lsh(queries)
    keys_hashed = e2lsh(keys)
    if key_padding_mask is not None:
        keys_hashed.masked_fill_(~key_padding_mask, float("inf"))
        # HACK: if queries and keys have the same length, we assume it's self-attention.
        # By right we shouldn't change queries_hashed, but the original SMYRF code does it.
        if queries.shape[-2] == key_padding_mask.shape[-1]:
            queries_hashed.masked_fill_(~key_padding_mask, float("inf"))
    return queries_hashed.argsort(dim=-1), keys_hashed.argsort(dim=-1)


class E2LSH(LSH):
    def __init__(self, n_hashes, dim, r, device="cuda"):
        super(E2LSH, self).__init__()
        self.alpha = torch.normal(0, 1, (dim, n_hashes), device=device)
        self.beta = uniform(0, r, shape=(1, n_hashes), device=device)
        self.dim = dim
        self.r = r

    def __call__(self, vecs):
        """
        L2 Sensitive Hashing based on p-stable distributions.
        Also known as E2LSH.
        Args:
            vecs: (bs, N, dim) (dtype: torch.float32)
        Output:
            buckets: (n_hashes, bs, N) (dtype: torch.int32)
        """
        projection = vecs @ self.alpha
        projection_shift = projection + self.beta
        projection_rescale = projection_shift / self.r
        return projection_shift.permute(2, 0, 1)


class SmyrfAttention(nn.Module):
    def __init__(
        self,
        r=1,
        softmax_temp=None,
        attention_dropout=0.0,
        **kwargs,
    ):
        super().__init__()
        self.dim_per_head = kwargs["h_dim"]
        self.num_heads = kwargs["num_heads"]
        self.bucket_size = kwargs["bucket_size"]
        self.n_hashes = kwargs["n_hashes"]
        self.pe_type = kwargs["pe_type"]
        self.num_w_per_dist = kwargs["num_w_per_dist"]
        self.out_linear = nn.Linear(self.num_heads * self.dim_per_head, self.dim_per_head)

        self.q_cluster_size = self.bucket_size
        self.k_cluster_size = self.bucket_size
        self.softmax_temp = softmax_temp
        self.dropout = nn.Dropout(attention_dropout)
        self.hash_fn = XBOXPLUS()
        self.clustering_params = {"r": r, "n_hashes": self.n_hashes}

    def hash_vectors(self, query, key, key_padding_mask=None, pe_type=None):
        # XBOX+ transform
        if pe_type == "rpe":
            query_t = query
            key_t = key
        else:
            self.hash_fn.set_norms(query, key)
            query_t = self.hash_fn.Q(query)
            key_t = self.hash_fn.K(key)

        num_clusters = query_t.shape[-2] // self.q_cluster_size
        assert num_clusters == (
            key_t.shape[-2] // self.k_cluster_size
        ), "Unequal number of clusters for query and key."
        q_positions, k_positions = lsh_clustering(
            query_t, key_t, **self.clustering_params, key_padding_mask=key_padding_mask
        )
        return q_positions, k_positions

    def forward(self, query, key, value, **kwargs):
        key_padding_mask = kwargs["key_padding_mask"]
        query = rearrange(query, "b n (h d) -> b n h d", h=self.num_heads, d=self.dim_per_head)
        key = rearrange(key, "b n (h d) -> b n h d", h=self.num_heads, d=self.dim_per_head)
        value = rearrange(value, "b n (h d) -> b n h d", h=self.num_heads, d=self.dim_per_head)

        if "rpe" in self.pe_type:
            w = rearrange(kwargs["w_rpe"].weight, "(h d) (r k) -> h d r k", h=self.num_heads, d=self.dim_per_head, k=self.num_w_per_dist)
            qw = w.sum(dim=1).clamp(max=50).exp().sum(dim=-1)
            new_qw_expand_dim = torch.cat([qw[:, :1], qw], dim=-1)
            sqrt_w_r = torch.sqrt(2 * new_qw_expand_dim)[None] * kwargs["pe"][:, :, None]
            query = torch.cat([query, sqrt_w_r], dim=-1)
            key = torch.cat([key, sqrt_w_r], dim=-1)
            kwargs["rpe_ones"] = pad_to_multiple(kwargs["rpe_ones"], self.k_cluster_size, dims=1)
            kwargs["rpe_ones"] = rearrange(kwargs["rpe_ones"], "b s h d -> (b h) s d")
            self.softmax_temp = 1.0

        _, q_seqlen_og, _, _ = query.shape
        _, k_seqlen_og, _, _ = key.shape
        query = pad_to_multiple(query, self.q_cluster_size, dims=1)
        key = pad_to_multiple(key, self.k_cluster_size, dims=1)
        value = pad_to_multiple(value, self.k_cluster_size, dims=1)

        # Extract some shapes and compute the temperature
        B, T, H, E = query.shape
        _, S, _, D = value.shape
        softmax_temp = self.softmax_temp or 1 / math.sqrt(E)

        # pad the masks
        if S > k_seqlen_og:
            if key_padding_mask is None:
                key_padding_mask = LengthMask(key.new_full((key.shape[0],), k_seqlen_og, dtype=torch.long), max_len=S)
            else:
                key_padding_mask = pad_mask(key_padding_mask, pad_length=S - k_seqlen_og, left=False, value=False)

        query = rearrange(query, "b t h e -> (b h) t e")
        key = rearrange(key, "b t h e -> (b h) t e")
        value = rearrange(value, "b s h d -> (b h) s d")
        bs = query.shape[0]

        if key_padding_mask is not None and not key_padding_mask.all_ones:
            # Repeat for all heads
            key_padding_mask_bool = repeat(key_padding_mask.bool_matrix, "b s -> (b h) s", h=H)
        else:
            key_padding_mask_bool = None

        with torch.no_grad():
            q_positions, k_positions = self.hash_vectors(
                query,
                key,
                rearrange(key_padding_mask_bool, "b s -> 1 b s") if key_padding_mask_bool is not None else None,
                self.pe_type,
            )

        # sort queries, keys, values
        def sort_to_buckets(x, perm, bucketsz):
            return rearrange(
                batched_index_select(rearrange(x, "b s d -> 1 b s d"), perm),
                "h b (nbuckets bucketsz) d -> h b nbuckets bucketsz d",
                bucketsz=bucketsz,
            )

        if self.pe_type == "rpe":
            query_sq = -0.5 * (query ** 2).sum(dim=-1, keepdim=True)
            key_sq = -0.5 * (key ** 2).sum(dim=-1, keepdim=True)
            query = torch.cat([query, kwargs["rpe_ones"], query_sq], dim=-1)
            key = torch.cat([key, key_sq, kwargs["rpe_ones"]], dim=-1)

        s_query = sort_to_buckets(query, q_positions, self.q_cluster_size)
        s_key = sort_to_buckets(key, k_positions, self.k_cluster_size)
        s_value = sort_to_buckets(value, k_positions, self.k_cluster_size)

        if "rpe" in self.pe_type:
            inner = torch.einsum("...id,...jd->...ij", s_query, s_key).clamp(max=0.0) * softmax_temp
        else:
            inner = torch.einsum("...id,...jd->...ij", s_query, s_key) * softmax_temp

        masked_value = max_neg_value(inner)
        # mask out attention to padded tokens
        if key_padding_mask is not None and not key_padding_mask.all_ones:
            s_key_padding_mask = sort_to_buckets(
                rearrange(key_padding_mask_bool, "b s -> b s 1"), k_positions, self.k_cluster_size
            )
            s_key_padding_mask = rearrange(s_key_padding_mask, "... bucketsz 1 -> ... 1 bucketsz")
            inner.masked_fill_(~s_key_padding_mask, masked_value)

        q_rev_positions = invert_permutation(q_positions)
        # free memory
        del q_positions, k_positions

        # softmax denominator
        dots_logsumexp = torch.logsumexp(inner, dim=-1, keepdim=True)
        # softmax
        dots = torch.exp(inner - dots_logsumexp)
        # If the whole row within this bucket is masked out, then inner is the uniform distribution.
        # We actually want it to be zero.
        if key_padding_mask is not None and not key_padding_mask.all_ones:
            full_row_mask = (inner <= masked_value).all(dim=-1, keepdim=True)
            dots = dots.masked_fill(full_row_mask, 0.0)

        # dropout
        dropped_dots = self.dropout(dots)

        # n_hashes outs
        so = torch.einsum("...ij,...jd->...id", dropped_dots, s_value)

        # undo sort
        def unsort_from_buckets(s_x, perm_inverse):
            b_x = rearrange(s_x, "h b nbuckets bucketsz d -> h b (nbuckets bucketsz) d")
            return batched_index_select(b_x, perm_inverse)

        o = unsort_from_buckets(so, q_rev_positions)
        logits = unsort_from_buckets(dots_logsumexp, q_rev_positions)

        # free memory
        del q_rev_positions

        probs = torch.exp(logits - torch.logsumexp(logits, dim=0, keepdim=True))
        out = torch.sum(o * probs, dim=0)
        out = rearrange(out, "(b h) t d -> b t (h d)", h=H)
        out = out[:, :q_seqlen_og]

        out = self.out_linear(out)
        return out
