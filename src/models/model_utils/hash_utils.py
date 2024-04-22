# Adapted from https://github.com/giannisdaras/smyrf/blob/master/smyrf/torch/utils.py
""" Utility functions for smyrf """
import torch
import torch.nn.functional as F
from collections import defaultdict, Counter
import numpy as np
from tqdm import tqdm
import random

import torch.nn as nn
from einops import rearrange


def quantile_partition(sorted_indices, num_regions):
    total_elements = sorted_indices.shape[-1]
    region_size = torch.ceil(total_elements / num_regions)
    inverse_indices = torch.argsort(sorted_indices, dim=-1)

    base = torch.arange(total_elements, device=sorted_indices.device)[None]
    region_indices = base // region_size + 1
    reassigned_regions = region_indices[:, inverse_indices]
    return reassigned_regions


def get_regions(num_regions, num_or_hashes, num_heads, num_and_hashes=2):
    lb = 2
    ub = 2 * num_regions ** (1 / num_and_hashes) - lb
    regions = []
    for _ in range(num_or_hashes * num_heads):
        region = []
        for _ in range(num_and_hashes):
            a = torch.rand(1).item() * (ub - lb) + lb
            region.append(a)
        regions.append(region)
    regions = torch.tensor(regions)
    regions = (num_regions / regions.prod(dim=1, keepdim=True)) ** (1 / num_and_hashes) * regions

    regions = torch.round(regions * 3) / 3
    return rearrange(regions, "(h c) a -> c a h", h=num_heads)


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


# Adapted from https://github.com/lucidrains/reformer-pytorch/blob/master/reformer_pytorch/autopadder.py
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


def batched_index_select(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """
    Params:
        values: (1 or n_hashes, batch, seqlen, dim)
        indices: (n_hashes, batch, seqlen)
    Return:
        (n_hashes, batch, seqlen, dim)
    """
    last_dim = values.shape[-1]
    indices_expanded = rearrange(indices, "... -> ... 1").expand(*indices.shape, last_dim)
    return values.expand(*indices_expanded.shape[:-2], *values.shape[-2:]).gather(-2, indices_expanded)


def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max


def random_flip(x):
    flips = torch.ceil((torch.rand(x.shape, device=x.device) - 0.5)).to(torch.uint8)
    return flips * x


def sign_randomness(fn):
    def do(*args, **kwargs):
        return random_flip(fn(*args, **kwargs))

    return do


@sign_randomness
def hadamard_transform(u, normalize=False):
    batch_size, n = u.shape
    m = int(np.log2(n))
    assert n == 1 << m, "n must be a power of 2"
    x = u[..., np.newaxis]
    for d in range(m)[::-1]:
        x = torch.cat((x[..., ::2, :] + x[..., 1::2, :], x[..., ::2, :] - x[..., 1::2, :]), dim=-1)
    return x.squeeze(-2) / 2 ** (m / 2) if normalize else x.squeeze(-2)


def inversion_number(arr1, arr2):
    """
    Counts "relative" mistakes.
    """
    mapping = {}
    count = 0
    not_found = 0

    for i, elem in enumerate(arr2):
        mapping[elem] = i

    for i, elem_a in enumerate(arr1):
        if not elem_a in mapping:
            not_found += 1
            count += len(arr1[i + 1 :])
            continue

        for elem_b in arr1[i + 1 :]:
            mapped_a = mapping[elem_a]
            if not elem_b in mapping:
                count += 1
                continue
            mapped_b = mapping[elem_b]
            if mapped_a > mapped_b:
                count += 1
    return count, not_found


def two_dimensional(fn):
    def do(self, x, *args, **kwargs):
        if len(x.shape) == 2:
            return fn(self, x, *args, **kwargs)
        else:
            x = x.reshape(-1, x.shape[-1])
            return fn(self, x, *args, **kwargs)

    return do


def sort_key_val(t1, t2, dim=-1, n_buckets=1):
    """
    Sort t2 based on t1.
    """
    values, indices = t1.sort(dim=dim)
    t2 = t2.expand_as(t1)
    return values, t2.gather(dim, indices)


def uniform(a, b, shape, device="cpu"):
    """
    Draws shape samples from a uniform distribution U(a, b).

    """
    return (b - a) * torch.rand(shape, device=device) + a


"""                   Preprocessing functions for ALSH                      """


class AsymmetricTransform:

    def Q(self, *args, **kwargs):
        raise NotImplementedError("Query transform not implemented")

    def K(self, *args, **kwargs):
        raise NotImplementedError("Key transform not implemented")


class L2LSH(AsymmetricTransform):

    def K(self, vec):
        # Normalize x = vec / max_norm
        norms = vec.norm(p=2, dim=-1).unsqueeze(-1)
        max_norm = torch.max(norms, dim=0)[0]
        x = vec / max_norm

        # compute new_norms
        norms = x.norm(p=2, dim=-1).unsqueeze(-1)

        # transform: x = [x; norm_x**2, norm_x**4]
        return torch.cat((x, norms**2, norms**4, norms**8), -1)

    def Q(self, vec):
        # normalize queries
        x = (vec - vec.mean(dim=-1).unsqueeze(-1)) / vec.std(dim=-1).unsqueeze(-1)
        device = vec.device
        ext = torch.empty(x.shape[:-1] + (1,), device=device).fill_(0.5)
        return torch.cat((x, ext, ext, ext), -1)


class XBOX(AsymmetricTransform):

    def K(self, x):
        norms = x.norm(p=2, dim=-1).unsqueeze(-1)
        max_norm = torch.max(norms, dim=1).values.unsqueeze(-1)
        ext = torch.sqrt(max_norm**2 - norms**2)
        return torch.cat((x, ext), -1)

    def Q(self, x):
        zero = torch.tensor([0.0], device=x.device).repeat(x.shape[:-1], 1).unsqueeze(-1)
        return torch.cat((x, zero), -1)


class XBOXPLUS(AsymmetricTransform):

    def set_norms(self, queries, keys):
        self.q_norm_sq = queries.norm(p=2, dim=-1, keepdim=True).square()
        self.k_norm_sq = keys.norm(p=2, dim=-1, keepdim=True).square()
        MQ_sq = torch.amax(self.q_norm_sq, dim=-2, keepdim=True)
        MK_sq = torch.amax(self.k_norm_sq, dim=-2, keepdim=True)
        self.MQ_sq_MK_sq = MQ_sq + MK_sq

    def K(self, x):
        ext = (self.MQ_sq_MK_sq - self.k_norm_sq).sqrt()
        return torch.cat([x, ext, torch.zeros_like(ext)], dim=-1)

    def Q(self, x):
        ext = (self.MQ_sq_MK_sq - self.q_norm_sq).sqrt()
        return torch.cat([x, torch.zeros_like(ext), ext], dim=-1)


class XBOXMax(AsymmetricTransform):

    def set_norms(self, queries, keys):
        self.q_norm_sq = queries.norm(p=2, dim=-1, keepdim=True).square()
        self.k_norm_sq = keys.norm(p=2, dim=-1, keepdim=True).square()
        MQ_sq = torch.amax(self.q_norm_sq, dim=-2, keepdim=True)
        MK_sq = torch.amax(self.k_norm_sq, dim=-2, keepdim=True)
        self.MQ_sq_MK_sq_max = torch.maximum(MQ_sq, MK_sq)

    def K(self, x):
        ext = (self.MQ_sq_MK_sq_max - self.k_norm_sq).sqrt()
        return torch.cat([x, ext, torch.zeros_like(ext)], dim=-1)

    def Q(self, x):
        ext = (self.MQ_sq_MK_sq_max - self.k_norm_sq).sqrt()
        return torch.cat([x, torch.zeros_like(ext), ext], dim=-1)


class H2LSH(AsymmetricTransform):
    """
    "Advanced" xbox for queries. Technique: H2-ALSH.
    Based on paper: Accurate and Fast ALSH (KDD 2018)
    """

    def K(self, x):
        norms = x.norm(p=2, dim=-1).unsqueeze(-1)
        max_norm = torch.max(norms, dim=0)[0]
        self.max_norm = max_norm
        ext = torch.sqrt(max_norm**2 - norms**2)
        return torch.cat((x, ext), -1)

    def Q(self, x):
        assert hasattr(self, "max_norm"), "Max norm not set"
        zero = torch.tensor([0.0], device=x.device).repeat(x.shape[0], 1)
        res = torch.cat((self.max_norm * x, zero), -1)
        del self.max_norm
        return res


"""                              Hashing                                     """


class LSH:
    def __call__(self, *args, **kwargs):
        raise NotImplementedError("LSH scheme not implemented")

    def compute_hash_agreement(self, q_hash, k_hash):
        return (q_hash == k_hash).min(dim=-1)[0].sum(dim=-1)


class VoronoiLSH(LSH):
    def __init__(self, L, K, dim, device="cuda"):
        """
        We repeat L times the following process.
        Choose K gaussians. Compute the inner product, keep the index of
        the maximum.

        L: increases the probability of collision for near ones.
        K: decreases the probability of collision for far ones.

        Suggested values:
            -> K = ln(N) / ln(2)
            -> L = sqrt(N)
        """
        self.gaussians = torch.randn(dim, K * L, device=device)
        self.K = K
        self.L = L
        self.dim = dim

    def __call__(self, vecs):
        products = vecs @ self.gaussians
        return torch.argmax(products.reshape(-1, self.L, self.K), dim=-1)


class CrossPolytopeLSH(LSH):
    def __init__(self, L, K, dim, device="cuda"):
        self.L = L
        self.K = K
        self.dim = dim

    def __call__(self, vecs):
        x = vecs.repeat([self.L * self.K, 1])
        x = hadamard_transform(x, normalize=True)
        x = hadamard_transform(x)
        x = x.reshape(self.L, self.K, -1, vecs.shape[-1])
        indices = torch.argmax(x, dim=-1).permute(2, 0, 1)
        return indices


@torch.no_grad()
def lsh_mapping(e2lsh, queries, keys):
    queries_hashed = e2lsh(queries)
    keys_hashed = e2lsh(keys)
    hash_shift = max(queries_hashed.max(), keys_hashed.max()) - min(queries_hashed.min(), keys_hashed.min())
    return queries_hashed, keys_hashed, hash_shift


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


class QLSH(LSH):
    def __init__(self, L, K, dim, r=4, device="cuda"):
        self.alpha = torch.normal(0, 1, (dim, L * K), device=device)
        self.dim = dim
        self.L = L
        self.K = K
        self.r = r

    @two_dimensional
    def __call__(self, queries, keys):
        q_projection = (queries @ self.alpha).reshape(-1, self.L, self.K)
        k_projection = (keys @ self.alpha).reshape(-1, self.L, self.K)

        return self.compute_hash_agreement(q_projection, k_projection)

    def compute_hash_agreement(self, q_projection, k_projection):
        diff = k_projection - q_projection
        left_part = diff >= (-self.r / 2)
        right_part = diff <= (self.r / 2)
        truth_table = (left_part * right_part).min(dim=-1)[0].sum(dim=-1)
        return truth_table
