# Adapted from https://github.com/lucidrains/performer-pytorch/blob/main/performer_pytorch/performer_pytorch.py
import math
import torch
from torch import nn
from torch.cuda.amp import autocast
from einops import rearrange, repeat

from functools import partial
from contextlib import contextmanager

try:
    from apex import amp
    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False


# helpers

@contextmanager
def null_context():
    yield


def linear_attention_normalization(q, k, causal=False):
    if not causal:
        return torch.einsum('...nm,...m->...n', q, k.sum(dim=-2))
    else:
        return torch.einsum('...nm,...nm->...n', q, k.cumsum(dim=-2))


def gaussian_orthogonal_random_matrix(nrows, ncols, scaling=0, device=None, dtype=None):
    factory_kwargs = {'device': device, 'dtype': dtype}
    nblocks = int(math.ceil(nrows / ncols))
    # TD [2021-10-28]: Sometimes QR fails on CUDA
    unstructured_blocks = torch.randn((nblocks, ncols, ncols), device='cpu')
    q, r = torch.linalg.qr(unstructured_blocks)
    # To make sure Q is uniform from the Haar distribution https://arxiv.org/pdf/math-ph/0609050.pdf
    q *= rearrange(torch.diagonal(r, dim1=-2, dim2=-1).sign(), 'b c -> b 1 c')
    q = q.to(**factory_kwargs)
    # TD [2021-10-28] Idk why the transpose is necessary. I suspect it isn't.
    # https://github.com/google-research/google-research/blob/ea313c6e96acce6c863de41615c6cf4079b8ca94/performer/fast_attention/jax/fast_attention.py#L362
    q = rearrange(q, 'b c c1 -> b c1 c')
    g_ortho = rearrange(q, 'b c1 c -> (b c1) c')[:nrows]

    if scaling == 0:
        multiplier = torch.randn((nrows, ncols), **factory_kwargs).norm(dim=1)
        return rearrange(multiplier, 'r -> r 1') * g_ortho
    elif scaling == 1:
        return math.sqrt(ncols) * g_ortho
    else:
        raise ValueError(f'Invalid scaling {scaling}')


# kernel functions

# transcribed from jax to pytorch from
# https://github.com/google-research/google-research/blob/master/performer/fast_attention/jax/fast_attention.py

def softmax_kernel(data, *, projection_matrix, is_query, softmax_temp=None, eps=1e-4):
    """For key, we expect shape (b, h, s, d) where s is the sequence dimension
    """
    b, h, _, d = data.shape

    if softmax_temp is None:
        softmax_temp = 1 / math.sqrt(d)
    data_normalizer = math.sqrt(softmax_temp)

    ratio = (projection_matrix.shape[0] ** -0.5)

    projection = repeat(projection_matrix, 'j d -> b h j d', b = b, h = h)
    projection = projection  #.type_as(data)

    data_dash = torch.einsum('...id,...jd->...ij', (data_normalizer * data), projection)

    diag_data = data ** 2
    diag_data = torch.sum(diag_data, dim=-1)
    diag_data = (diag_data / 2.0) * (data_normalizer ** 2)
    diag_data = diag_data.unsqueeze(dim=-1)

    if is_query:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data -
                    torch.amax(data_dash, dim=-1, keepdim=True)) + eps)
    else:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data - torch.amax(data_dash, dim=(-1, -2), keepdim=True)) + eps)

    return data_dash  #.type_as(data)


# linear attention classes with softmax kernel

# non-causal linear attention
# By default Performer uses eps=0.0 here
def linear_attention(q, k, v, eps=0.0, need_weights=False):
    k_cumsum = k.sum(dim=-2)
    D_inv = 1. / (torch.einsum('...nd,...d->...n', q, k_cumsum.type_as(q)) + eps)
    context = torch.einsum('...nd,...ne->...de', k, v)
    out = torch.einsum('...de,...nd,...n->...ne', context, q, D_inv)
    attn = None if not need_weights else torch.einsum('...te,...se,...s->...ts', q, k, D_inv)
    return out, attn
