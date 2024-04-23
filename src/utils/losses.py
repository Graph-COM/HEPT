import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, segment_csr
from .metrics import pair_filter


class InfoNCELoss(nn.Module):
    def __init__(self, tau, dist_metric):
        super().__init__()
        self.tau = tau
        self.dist_metric = dist_metric

    def forward(self, x, point_pairs, cluster_ids, recons, pts, **kwargs):
        all_pos_pair_mask = cluster_ids[point_pairs[0]] == cluster_ids[point_pairs[1]]

        extra_pos_pair_mask = pair_filter(cluster_ids, point_pairs, recons, pts, pt_thres=0.9)
        all_pos_pair_mask = all_pos_pair_mask & extra_pos_pair_mask
        all_neg_pair_mask = ~all_pos_pair_mask

        if self.dist_metric == "cosine":
            similarity = F.cosine_similarity(x[point_pairs[0]], x[point_pairs[1]], dim=-1)
        elif self.dist_metric == "l2_rbf":
            # l2_dist = torch.linalg.norm(x[point_pairs[0]] - x[point_pairs[1]], ord=2, dim=-1)
            l2_dist = batched_point_distance(x, point_pairs, batch_size=5000)
            sigma = 0.75
            similarity = torch.exp(-l2_dist / (2 * sigma**2))
        elif self.dist_metric == "l2_inverse":
            l2_dist = torch.linalg.norm(x[point_pairs[0]] - x[point_pairs[1]], ord=2, dim=-1)
            similarity = 1.0 / (l2_dist + 1.0)
        else:
            raise NotImplementedError

        loss_per_pos_pair = self.calc_info_nce(x, similarity, point_pairs, all_pos_pair_mask, all_neg_pair_mask)
        new_labels = cluster_ids[point_pairs[0][all_pos_pair_mask]]  # [topk_mask]
        unique_new_labels, new_labels = torch.unique(new_labels, return_inverse=True)
        loss_per_pos_pair = deterministic_scatter(loss_per_pos_pair, new_labels, reduce="mean")

        return torch.mean(loss_per_pos_pair)

    def calc_info_nce(self, x, similarity, all_pairs, all_pos_pair_mask, all_neg_pair_mask):
        max_sim = (similarity / self.tau).max()
        exp_sim = torch.exp(similarity / self.tau - max_sim)

        pos_exp_sim = exp_sim[all_pos_pair_mask]
        neg_exp_sim = exp_sim[all_neg_pair_mask]

        numerator = pos_exp_sim
        group_indices = all_pairs[0][all_neg_pair_mask]
        denominator = deterministic_scatter(neg_exp_sim, group_indices, reduce="sum").clamp(min=0)

        denominator = denominator[all_pairs[0][all_pos_pair_mask]]
        loss_per_pos_pair = -torch.log(numerator / (numerator + denominator))
        return loss_per_pos_pair

    def calc_triplet(self, x, dists, all_pairs, all_pos_pair_mask, all_neg_pair_mask):
        group_indices = all_pairs[0][all_neg_pair_mask]
        neg_pair_dists = torch.zeros(x.shape[0], device=x.device)
        neg_pair_dists = scatter_mean(dists[all_neg_pair_mask], group_indices, out=neg_pair_dists)
        neg_pair_dists = neg_pair_dists[all_pairs[0][all_pos_pair_mask]]

        loss_per_pos_pair = torch.clamp(dists[all_pos_pair_mask] - neg_pair_dists + self.margin, min=0.0)
        return loss_per_pos_pair


def deterministic_scatter(src, index, reduce):
    sorted_arg = torch.argsort(index)
    sorted_index = index[sorted_arg]
    sorted_src = src[sorted_arg]
    unique_groups, counts = torch.unique_consecutive(sorted_index, return_counts=True)
    indptr = torch.zeros(len(unique_groups) + 1, device=src.device)
    indptr[1:] = torch.cumsum(counts, dim=0)
    output = segment_csr(sorted_src, indptr.long(), reduce=reduce)
    return output


def batched_point_distance(x, point_pairs, batch_size=1000):
    """
    Compute the L2 norm between points in x specified by point_pairs in batches.

    :param x: Tensor of shape (n, d)
    :param point_pairs: Tensor of shape (2, E)
    :param batch_size: Size of the batch for processing
    :return: Tensor of distances
    """
    num_pairs = point_pairs.size(1)
    distances = []

    for i in range(0, num_pairs, batch_size):
        batch_pairs = point_pairs[:, i : i + batch_size]
        diff = x[batch_pairs[0]] - x[batch_pairs[1]]
        batch_distances = torch.linalg.norm(diff, ord=2, dim=-1)
        distances.append(batch_distances)

    return torch.cat(distances)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-BCE_loss)  # prevents nans when probability 0
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()
