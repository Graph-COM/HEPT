# https://github.com/HazyResearch/fly/blob/master/src/models/attention/sbsmyrf_attention.py
# Adapted from https://github.com/giannisdaras/smyrf/blob/master/smyrf/torch/attn.py
import math
import torch
import torch.nn as nn

from einops import rearrange, repeat

from ..model_utils.feature_maps_sb import SBPerformerFeatures
from ..model_utils.hash_utils import batched_index_select, invert_permutation, pad_to_multiple, XBOXPLUS, LSH, uniform
from ..model_utils.mask_utils import LengthMask, pad_mask
from ..model_utils.rff_utils import linear_attention, linear_attention_normalization


def chunked_sum(tensor, chunks=1):
    *orig_size, last_dim = tensor.shape
    tensor = tensor.reshape(-1, last_dim)
    summed_tensors = [c.sum(dim=-1) for c in tensor.chunk(chunks, dim=0)]
    return torch.cat(summed_tensors, dim=0).reshape(orig_size)


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


class SBAttention(nn.Module):
    def __init__(
        self,
        r=1,  # LSH clustering
        ortho_scaling=0,
        softmax_eps=1e-6,  # Performer
        softmax_temp=None,
        attention_dropout=0.0,
        **kwargs,
    ):
        super().__init__()
        self.dim_per_head = kwargs["h_dim"]
        self.num_heads = kwargs["num_heads"]
        self.nb_features = kwargs["nb_features"]
        self.bucket_size = kwargs["bucket_size"]
        self.n_hashes = kwargs["n_hashes"]
        self.out_linear = nn.Linear(self.num_heads * self.dim_per_head, self.dim_per_head)

        self.feature_map = SBPerformerFeatures(
            self.dim_per_head,
            self.nb_features,
            ortho_scaling=ortho_scaling,
            softmax_temp=softmax_temp,
            eps=softmax_eps,
        )
        self.q_cluster_size = self.bucket_size
        self.k_cluster_size = self.bucket_size
        self.softmax_temp = softmax_temp
        self.dropout = nn.Dropout(attention_dropout)
        self.hash_fn = XBOXPLUS()
        self.clustering_params = {"r": r, "n_hashes": self.n_hashes}

    def hash_vectors(self, query, key, key_padding_mask=None):
        # XBOX+ transform
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
            )

        self.feature_map.new_feature_map(query.device)
        q_prime, q_prime_log_scale = self.feature_map.forward_queries(query)
        k_prime, k_prime_log_scale = self.feature_map.forward_keys(key)

        prime_log_scale = q_prime_log_scale + k_prime_log_scale
        m = q_prime.shape[-1]
        if key_padding_mask_bool is not None:
            k_prime.masked_fill_(~rearrange(key_padding_mask_bool, "b s -> b s 1"), 0.0)

        q_prime_k_prime_1 = linear_attention_normalization(q_prime, k_prime)
        q_prime_k_prime_v, attn_prime = linear_attention(q_prime, k_prime, value)

        # sort queries, keys, values
        def sort_to_buckets(x, perm, bucketsz):
            return rearrange(
                batched_index_select(rearrange(x, "b s d -> 1 b s d"), perm),
                "h b (nbuckets bucketsz) d -> h b nbuckets bucketsz d",
                bucketsz=bucketsz,
            )

        s_query = sort_to_buckets(query, q_positions, self.q_cluster_size)
        s_key = sort_to_buckets(key, k_positions, self.k_cluster_size)
        s_value = sort_to_buckets(value, k_positions, self.k_cluster_size)
        sq_prime = sort_to_buckets(q_prime, q_positions, self.q_cluster_size)
        sk_prime = sort_to_buckets(k_prime, k_positions, self.k_cluster_size)
        # sq_prime, sq_prime_log_scale = kernel_fn(s_queries, is_query=True)
        # sk_prime, sk_prime_log_scale = kernel_fn(s_keys, is_query=False)
        # k_prime_log_scale doesn't depend on the index of the token
        sprime_log_scale = sort_to_buckets(prime_log_scale, q_positions, self.q_cluster_size)
        # sprime_log_scale = sq_prime_log_scale + sk_prime_log_scale

        inner = torch.einsum("...id,...jd->...ij", s_query, s_key) * softmax_temp
        dots_prime = torch.einsum("...im,...jm->...ij", sq_prime, sk_prime)

        masked_value = max_neg_value(inner)
        # mask out attention to padded tokens
        if key_padding_mask is not None and not key_padding_mask.all_ones:
            s_key_padding_mask = sort_to_buckets(
                rearrange(key_padding_mask_bool, "b s -> b s 1"), k_positions, self.k_cluster_size
            )
            s_key_padding_mask = rearrange(s_key_padding_mask, "... bucketsz 1 -> ... 1 bucketsz")
            inner.masked_fill_(~s_key_padding_mask, masked_value)
            dots_prime.masked_fill_(~s_key_padding_mask, 0.0)

        q_rev_positions = invert_permutation(q_positions)

        # Don't double-count query-key pairs across multiple rounds of hashing.
        # Count how many times a query-key pair is repeated, and to lower its log-prob
        # correspondingly at each repetition.
        if self.n_hashes > 1:
            k_rev_positions = invert_permutation(k_positions)
            q_bucket_idx = rearrange(q_rev_positions // self.q_cluster_size, "h b seqlen -> b seqlen h")
            k_bucket_idx = rearrange(k_rev_positions // self.k_cluster_size, "h b seqlen -> b seqlen h")
            s_q_bucket_idx = sort_to_buckets(q_bucket_idx, q_positions, self.q_cluster_size)
            s_k_bucket_idx = sort_to_buckets(k_bucket_idx, k_positions, self.k_cluster_size)
            dup_counts = rearrange(s_q_bucket_idx, "... bk_size h -> ... bk_size 1 h") == rearrange(
                s_k_bucket_idx, "... bk_size h -> ... 1 bk_size h"
            )
            # for memory considerations, chunk summation of last dimension for counting duplicates
            dup_counts = chunked_sum(dup_counts, chunks=(self.n_hashes * bs))
            dup_counts = dup_counts.detach()
            assert dup_counts.shape == inner.shape
            inner = inner - torch.log(dup_counts.float())
            dots_prime = dots_prime / dup_counts

        # free memory
        del q_positions, k_positions

        # softmax denominator
        # TD: Even though we call this dots_logsumexp, it can be of arbitrary value and the
        # computation would still be correct (assuming infinite precision), since it's just an
        # arbitrary scaling of @dots.
        # Here we choose it for numerical stability: we want torch.exp(inner - dots_logsumexp) <= 1.0
        # and torch.exp(spring_log_scale - dots_logsumexp) <= 1.0
        # dots_logsumexp = torch.logsumexp(inner, dim=-1, keepdim=True)
        dots_logsumexp = torch.maximum(torch.amax(inner, dim=-1, keepdim=True), sprime_log_scale)
        # TD: dots and dots_sum has log scale dots_logsumexp
        # TD: No longer need this because we pick dots_logsumexp to not be -inf
        # dots_prime_scale = torch.exp(sprime_log_scale - dots_logsumexp)
        # nan_q_indices = dots_prime_scale.isinf()
        # # dots_logsumexp[nan_q_indices] = 0.0
        # dots_logsumexp = torch.where(nan_q_indices, torch.tensor(0.0, device=dots_logsumexp.device),
        #                              dots_logsumexp)
        dots_prime_scale = torch.exp(sprime_log_scale - dots_logsumexp)
        dots = torch.exp(inner - dots_logsumexp) - dots_prime * dots_prime_scale
        # TD: No longer need this because we pick dots_logsumexp to not be -inf
        # If the whole row within this bucket is masked out, then inner is the uniform distribution.
        # We actually want it to be zero.
        # if key_padding_mask is not None and not key_padding_mask.all_ones:
        #     full_row_mask = (inner <= masked_value).all(dim=-1, keepdim=True)
        #     dots = dots.masked_fill(full_row_mask, 0.0)
        dots_sum = dots.sum(dim=-1, keepdim=True)

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
        dots_sum_unsort = unsort_from_buckets(dots_sum, q_rev_positions)

        # free memory
        del q_rev_positions

        normalization_log_scale = torch.logsumexp(logits, dim=0)
        probs = torch.exp(logits - rearrange(normalization_log_scale, "... -> 1 ..."))
        out_lsh = torch.sum(o * probs, dim=0)

        prime_scale = torch.exp(prime_log_scale - normalization_log_scale)
        out = out_lsh + q_prime_k_prime_v * prime_scale

        normalization = (dots_sum_unsort * probs).sum(dim=0) + q_prime_k_prime_1.unsqueeze(-1) * prime_scale
        out_normalized = out / normalization.clamp_min(1e-6)
        out_normalized = (rearrange(out_normalized, "(b h) t d -> b t (h d)", h=H))[:, :q_seqlen_og]

        out_normalized = self.out_linear(out_normalized)
        return out_normalized
