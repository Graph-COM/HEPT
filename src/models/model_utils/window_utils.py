# https://github.com/mit-han-lab/flatformer
# https://github.com/tusen-ai/SST

import math
from typing import Any, Dict, Optional, Tuple

import torch
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


def get_pe_func(pe_type, coords_size, kwargs):
    if pe_type == "learned":
        return PELearned(input_channel=coords_size, **kwargs)
    elif pe_type == "fixed":
        return PE(**kwargs)
    else:
        return None


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


@torch.no_grad()
def get_window_coors(coors, sparse_shape, window_shape, do_shift):
    if len(window_shape) == 2:
        win_shape_x, win_shape_y = window_shape
        win_shape_z = sparse_shape[-1]
    else:
        win_shape_x, win_shape_y, win_shape_z = window_shape

    sparse_shape_x, sparse_shape_y, sparse_shape_z = sparse_shape
    assert sparse_shape_z < sparse_shape_x, "Usually holds... in case of wrong order"

    max_num_win_x = int(np.ceil((sparse_shape_x / win_shape_x)) + 1)  # plus one here to meet the needs of shift.
    max_num_win_y = int(np.ceil((sparse_shape_y / win_shape_y)) + 1)  # plus one here to meet the needs of shift.
    max_num_win_z = int(np.ceil((sparse_shape_z / win_shape_z)) + 1)  # plus one here to meet the needs of shift.
    max_num_win_per_sample = max_num_win_x * max_num_win_y * max_num_win_z

    if do_shift:
        shift_x, shift_y, shift_z = win_shape_x // 2, win_shape_y // 2, win_shape_z // 2
    else:
        shift_x, shift_y, shift_z = win_shape_x, win_shape_y, win_shape_z

    # compatibility between 2D window and 3D window
    if sparse_shape_z == win_shape_z:
        shift_z = 0

    shifted_coors_x = coors[:, 3] + shift_x
    shifted_coors_y = coors[:, 2] + shift_y
    shifted_coors_z = coors[:, 1] + shift_z

    win_coors_x = shifted_coors_x // win_shape_x
    win_coors_y = shifted_coors_y // win_shape_y
    win_coors_z = shifted_coors_z // win_shape_z

    if len(window_shape) == 2:
        assert (win_coors_z == 0).all()

    batch_win_inds = (
        coors[:, 0] * max_num_win_per_sample
        + win_coors_x * max_num_win_y * max_num_win_z
        + win_coors_y * max_num_win_z
        + win_coors_z
    )

    coors_in_win_x = shifted_coors_x % win_shape_x
    coors_in_win_y = shifted_coors_y % win_shape_y
    coors_in_win_z = shifted_coors_z % win_shape_z
    coors_in_win = torch.stack([coors_in_win_z, coors_in_win_y, coors_in_win_x], dim=-1)
    # coors_in_win = torch.stack([coors_in_win_x, coors_in_win_y], dim=-1)

    return batch_win_inds, coors_in_win


@torch.no_grad()
def make_continuous_inds(inds):
    ### make batch_win_inds continuous
    dtype = inds.dtype
    device = inds.device

    unique_inds, _ = torch.sort(torch.unique(inds))
    num_valid_inds = len(unique_inds)
    max_origin_inds = unique_inds.max().item()
    canvas = -torch.ones((max_origin_inds + 1,), dtype=dtype, device=device)
    canvas[unique_inds] = torch.arange(num_valid_inds, dtype=dtype, device=device)

    conti_inds = canvas[inds]

    return conti_inds


@torch.no_grad()
def get_flat2win_inds(batch_win_inds, voxel_drop_lvl, drop_info, debug=True):
    """
    Args:
        batch_win_inds: shape=[N, ]. Indicates which window a voxel belongs to. Window inds is unique is the whole batch.
        voxel_drop_lvl: shape=[N, ]. Indicates batching_level of the window the voxel belongs to.
    Returns:
        flat2window_inds_dict: contains flat2window_inds of each voxel, shape=[N,]
            Determine the voxel position in range [0, num_windows * max_tokens) of each voxel.
    """
    device = batch_win_inds.device

    flat2window_inds_dict = {}

    for dl in drop_info:  # dl: short for drop level
        dl_mask = voxel_drop_lvl == dl
        if not dl_mask.any():
            continue

        conti_win_inds = make_continuous_inds(batch_win_inds[dl_mask])

        max_tokens = drop_info[dl]["max_tokens"]

        inner_win_inds = get_inner_win_inds(conti_win_inds)

        flat2window_inds = conti_win_inds * max_tokens + inner_win_inds

        flat2window_inds_dict[dl] = (flat2window_inds, torch.where(dl_mask))

        if debug:
            num_windows = len(torch.unique(conti_win_inds))
            assert (
                inner_win_inds.max() < max_tokens
            ), f"Max inner inds({inner_win_inds.max()}) larger(equal) than {max_tokens}"
            assert (flat2window_inds >= 0).all()
            max_ind = flat2window_inds.max().item()
            assert (
                max_ind < num_windows * max_tokens
            ), f"max_ind({max_ind}) larger than upper bound({num_windows * max_tokens})"
            assert (
                max_ind >= (num_windows - 1) * max_tokens
            ), f"max_ind({max_ind}) less than lower bound({(num_windows-1) * max_tokens})"

    return flat2window_inds_dict


def get_flat2win_inds_v2(batch_win_inds, voxel_drop_lvl, drop_info, debug=True):
    transform_dict = get_flat2win_inds(batch_win_inds, voxel_drop_lvl, drop_info, debug)
    # add voxel_drop_lvl and batching_info into transform_dict for better wrapping
    transform_dict["voxel_drop_level"] = voxel_drop_lvl
    transform_dict["batching_info"] = drop_info
    return transform_dict


try:
    import ingroup_indices
except:
    pass
from torch.autograd import Function


class IngroupIndicesFunction(Function):
    @staticmethod
    def forward(ctx, group_inds):
        out_inds = torch.zeros_like(group_inds) - 1

        ingroup_indices.forward(group_inds, out_inds)

        ctx.mark_non_differentiable(out_inds)

        return out_inds

    @staticmethod
    def backward(ctx, g):
        return None


get_inner_win_inds = IngroupIndicesFunction.apply


def window2flat(feat_3d_dict, inds_dict):
    flat_feat_list = []

    num_all_voxel = 0
    for dl in inds_dict:
        num_all_voxel += inds_dict[dl][0].shape[0]

    dtype = feat_3d_dict[list(feat_3d_dict.keys())[0]].dtype

    device = feat_3d_dict[list(feat_3d_dict.keys())[0]].device
    feat_dim = feat_3d_dict[list(feat_3d_dict.keys())[0]].shape[-1]

    all_flat_feat = torch.zeros((num_all_voxel, feat_dim), device=device, dtype=dtype)
    # check_feat = -torch.ones((num_all_voxel,), device=device, dtype=torch.long)

    for dl in feat_3d_dict:
        feat = feat_3d_dict[dl]
        feat_dim = feat.shape[-1]
        inds, flat_pos = inds_dict[dl]
        feat = feat.reshape(-1, feat_dim)
        flat_feat = feat[inds]
        all_flat_feat[flat_pos] = flat_feat
        # check_feat[flat_pos] = 0
        # flat_feat_list.append(flat_feat)
    # assert (check_feat == 0).all()

    return all_flat_feat


def window2flat_v2(feat_3d_dict, inds_dict):
    inds_v1 = {k: inds_dict[k] for k in inds_dict if not isinstance(k, str)}
    return window2flat(feat_3d_dict, inds_v1)


def flat2window(feat, voxel_drop_lvl, flat2win_inds_dict, drop_info, padding=0):
    """
    Args:
        feat: shape=[N, C], N is the voxel num in the batch.
        voxel_drop_lvl: shape=[N, ]. Indicates drop_level of the window the voxel belongs to.
    Returns:
        feat_3d_dict: contains feat_3d of each drop level. Shape of feat_3d is [num_windows, num_max_tokens, C].

    drop_info:
    {1:{'max_tokens':50, 'range':(0, 50)}, }
    """
    dtype = feat.dtype
    device = feat.device
    feat_dim = feat.shape[-1]

    feat_3d_dict = {}

    for dl in drop_info:
        dl_mask = voxel_drop_lvl == dl
        if not dl_mask.any():
            continue

        feat_this_dl = feat[dl_mask]

        this_inds = flat2win_inds_dict[dl][0]

        max_tokens = drop_info[dl]["max_tokens"]
        num_windows = (this_inds // max_tokens).max().item() + 1
        padding = torch.tensor(padding, dtype=dtype, device=device)
        feat_3d = torch.ones((num_windows * max_tokens, feat_dim), dtype=dtype, device=device) * padding
        # if this_inds.max() >= num_windows * max_tokens:
        #     set_trace()
        feat_3d[this_inds] = feat_this_dl
        feat_3d = feat_3d.reshape((num_windows, max_tokens, feat_dim))
        feat_3d_dict[dl] = feat_3d

    return feat_3d_dict


def flat2window_v2(feat, inds_dict, padding=0):
    assert "voxel_drop_level" in inds_dict, "voxel_drop_level should be in inds_dict in v2 function"
    inds_v1 = {k: inds_dict[k] for k in inds_dict if not isinstance(k, str)}
    batching_info = inds_dict["batching_info"]
    return flat2window(feat, inds_dict["voxel_drop_level"], inds_v1, batching_info, padding=padding)


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


def get_window_coors_shift(coords, sparse_shape, window_shape, shifted):
    n, m, _ = sparse_shape
    n2, m2, _ = window_shape

    n1 = int(np.ceil(n / n2) + 1)  # plus one here to meet the needs of shift.
    m1 = int(np.ceil(m / m2) + 1)  # plus one here to meet the needs of shift.

    if shifted:
        shift_x, shift_y = (n2 // 2, m2 // 2)
        x = coords[:, 3] + shift_x
        y = coords[:, 2] + shift_y
    else:
        x = coords[:, 3]
        y = coords[:, 2]

    x1 = x // n2
    y1 = y // m2
    x2 = x % n2
    y2 = y % m2

    return 2 * n2, 2 * m2, 2 * n1, 2 * m1, x1, y1, x2, y2


class FlattenedWindowMapping(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        B, N = kwargs["B"], kwargs["num_slices_per_axis"]
        self.sparse_shape = (B, B, 1)
        self.window_shape = (B // N, B // N, 1)
        self.group_size = kwargs["group_size"]

    def forward(self, coords: torch.Tensor, batch_size: int) -> Dict[str, torch.Tensor]:
        coords = coords.long()

        _, num_per_batch = torch.unique(coords[:, 0], sorted=False, return_counts=True)
        batch_start_indices = F.pad(torch.cumsum(num_per_batch, dim=0), (1, 0))
        num_per_batch_p = (
            torch.div(
                batch_start_indices[1:] - batch_start_indices[:-1] + self.group_size - 1,
                self.group_size,
                rounding_mode="trunc",
            )
            * self.group_size
        )
        batch_start_indices_p = F.pad(torch.cumsum(num_per_batch_p, dim=0), (1, 0))
        flat2win = torch.arange(batch_start_indices_p[-1]).to(coords.device)
        win2flat = torch.arange(batch_start_indices[-1]).to(coords.device)
        for i in range(batch_size):
            win2flat[batch_start_indices[i] : batch_start_indices[i + 1]] += (
                batch_start_indices_p[i] - batch_start_indices[i]
            )
            if num_per_batch[i] != num_per_batch_p[i]:
                flat2win[
                    batch_start_indices_p[i + 1]
                    - self.group_size
                    + (num_per_batch[i] % self.group_size) : batch_start_indices_p[i + 1]
                ] = flat2win[
                    batch_start_indices_p[i + 1]
                    - 2 * self.group_size
                    + (num_per_batch[i] % self.group_size) : batch_start_indices_p[i + 1]
                    - self.group_size
                ]
            flat2win[batch_start_indices_p[i] : batch_start_indices_p[i + 1]] -= (
                batch_start_indices_p[i] - batch_start_indices[i]
            )

        mappings = {"flat2win": flat2win, "win2flat": win2flat}
        for shifted in [False, True]:
            (
                n2,
                m2,
                n1,
                m1,
                x1,
                y1,
                x2,
                y2,
            ) = get_window_coors_shift(coords, self.sparse_shape, self.window_shape, shifted=shifted)
            vx = (n1 * y1 + (-1) ** y1 * x1) * n2 * m2 + (-1) ** y1 * (m2 * x2 + (-1) ** x2 * y2)
            vx += coords[:, 0] * self.sparse_shape[0] * self.sparse_shape[1] * 10
            vy = (m1 * x1 + (-1) ** x1 * y1) * m2 * n2 + (-1) ** x1 * (n2 * y2 + (-1) ** y2 * x2)
            vy += coords[:, 0] * self.sparse_shape[0] * self.sparse_shape[1] * 10
            _, mappings["x" + ("_shift" if shifted else "")] = torch.sort(vx)
            _, mappings["y" + ("_shift" if shifted else "")] = torch.sort(vy)

        return mappings
