import torch
import torch.nn.functional as F
import numpy as np
from numba import jit
from torch_scatter import scatter_mean


def pair_filter(cluster_ids, point_pairs, recons, pts, pt_thres):
    # Have been taken care of when generating the dataset
    # non_zero_pid_point_pairs = (cluster_ids[point_pairs[0]] != 0) & (cluster_ids[point_pairs[1]] != 0)

    reconstructable_point_pairs = (recons[point_pairs[0]] != 0) & (recons[point_pairs[1]] != 0)
    high_pt_point_pairs = (pts[point_pairs[0]] > pt_thres) & (pts[point_pairs[1]] > pt_thres)
    mask = reconstructable_point_pairs & high_pt_point_pairs
    return mask


def point_filter(cluster_ids, recons, pts, pt_thres):
    mask = (cluster_ids != 0) & (recons != 0) & (pts > pt_thres)
    return mask


@torch.no_grad()
def acc_and_pr_at_k(embeddings, cluster_ids, mask, dist_metric, K=19, batch_size=None):
    cluster_ids = cluster_ids.cpu().numpy()
    mask = mask.cpu().numpy()

    num_points = embeddings.shape[0]
    if batch_size is None:
        batch_size = num_points

    unique_clusters, counts = np.unique(cluster_ids, return_counts=True)
    cluster_sizes = dict(zip(unique_clusters.tolist(), counts.tolist()))

    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    for start_index in range(0, num_points, batch_size):
        end_index = min(start_index + batch_size, num_points)
        batch_mask = mask[start_index:end_index]
        batch_embeddings = embeddings[start_index:end_index][batch_mask]
        batch_cluster_ids = cluster_ids[start_index:end_index][batch_mask]

        if "l2" in dist_metric:
            dist_mat_batch = torch.cdist(batch_embeddings, embeddings, p=2.0)
        elif dist_metric == "cosine":
            dist_mat_batch = 1 - F.cosine_similarity(batch_embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1)
        else:
            raise NotImplementedError

        # Each point should have (cluster_size - 1) true neighbors
        k_list = np.array([cluster_sizes[each_cluster_id] - 1 for each_cluster_id in batch_cluster_ids])
        assert max(k_list) <= K, f"K is too small, max k is {max(k_list)}"
        indices = dist_mat_batch.topk(K + 1, dim=1, largest=False, sorted=True)[1].cpu().numpy()

        acc, prec, recall, filtered_cluster_ids = calc_scores(K, k_list, indices, cluster_ids, batch_cluster_ids)

        accuracy_scores.extend(acc)
        precision_scores.extend(prec)
        recall_scores.extend(recall)

    return np.mean(accuracy_scores), np.mean(precision_scores), np.mean(recall_scores)


@jit(nopython=True)
def calc_scores(K, k_list, indices, cluster_ids, batch_cluster_ids):
    acc = []
    prec = []
    recall = []
    filtered_cluster_ids = []
    for i, k in enumerate(k_list):
        if k == 0:
            continue

        # slice the k nearest neighbors
        neighbors = indices[i, 1 : K + 1]

        # Retrieve the labels of the k nearest neighbors
        neighbor_labels = cluster_ids[neighbors]

        # check if neighbor labels match the expanded labels (precision)
        matches = neighbor_labels == batch_cluster_ids[i]

        accuracy = matches[:k].sum() / k
        precision_at_K = matches.sum() / K
        recall_at_K = matches.sum() / k

        acc.append(accuracy)
        prec.append(precision_at_K)
        recall.append(recall_at_K)
        filtered_cluster_ids.append(batch_cluster_ids[i])

    return acc, prec, recall, filtered_cluster_ids


def calculate_node_classification_metrics(pred, target, mask):
    pred = pred[mask]
    target = target[mask]
    acc = (pred == target).sum().item() / mask.sum().item()
    return acc
