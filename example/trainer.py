import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch_geometric.utils import unbatch, to_undirected
from torchmetrics import MeanMetric
import numpy as np
from numba import jit
from tqdm import tqdm


def train_one_batch(model, optimizer, criterion, data, lr_s):
    model.train()
    embeddings = model(data)
    loss = criterion(embeddings, data.point_pairs_index, data.particle_id, data.reconstructable, data.pt)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if lr_s is not None and isinstance(lr_s, LambdaLR):
        lr_s.step()
    return loss.item(), embeddings.detach(), data.particle_id.detach()


@torch.no_grad()
def eval_one_batch(model, optimizer, criterion, data, lr_s):
    model.eval()
    embeddings = model(data)
    loss = criterion(embeddings, data.point_pairs_index, data.particle_id, data.reconstructable, data.pt)
    return loss.item(), embeddings.detach(), data.particle_id.detach()


def process_data(data, phase, device, epoch, p=0.2):
    data = data.to(device)
    if phase == "train":
        # pairs_to_add = add_random_edge(data.point_pairs_index, p=p, batch=data.batch, force_undirected=True)
        num_aug_pairs = int(data.point_pairs_index.size(1) * p / 2)
        pairs_to_add = to_undirected(torch.randint(0, data.num_nodes, (2, num_aug_pairs), device=device))
        data.point_pairs_index = torch.cat([data.point_pairs_index, pairs_to_add], dim=1)
    return data


def run_one_epoch(model, optimizer, criterion, data_loader, phase, epoch, device, metrics, lr_s):
    run_one_batch = train_one_batch if phase == "train" else eval_one_batch
    phase = "test " if phase == "test" else phase
    pbar = tqdm(data_loader)
    for idx, data in enumerate(pbar):
        if phase == "train" and model.attn_type == "None":
            torch.cuda.empty_cache()
        data = process_data(data, phase, device, epoch)

        batch_loss, batch_embeddings, batch_cluster_ids = run_one_batch(model, optimizer, criterion, data, lr_s)
        batch_acc = update_metrics(metrics, data, batch_embeddings, batch_cluster_ids, criterion.dist_metric)
        metrics["loss"].update(batch_loss)

        desc = f"[Epoch {epoch}] {phase}, loss: {batch_loss:.4f}, acc: {batch_acc:.4f}"
        if idx == len(data_loader) - 1:
            metric_res = compute_metrics(metrics)
            loss, acc = (metric_res["loss"], metric_res["accuracy@0.9"])
            prec, recall = metric_res["precision@0.9"], metric_res["recall@0.9"]
            desc = f"[Epoch {epoch}] {phase}, loss: {loss:.4f}, acc: {acc:.4f}, prec: {prec:.4f}, recall: {recall:.4f}"
            reset_metrics(metrics)
        pbar.set_description(desc)
    return metric_res


def reset_metrics(metrics):
    for metric in metrics.values():
        if isinstance(metric, MeanMetric):
            metric.reset()


def compute_metrics(metrics):
    return {
        f"{name}@{pt}": metrics[f"{name}@{pt}"].compute().item()
        for name in ["accuracy", "precision", "recall"]
        for pt in metrics["pt_thres"]
    } | {"loss": metrics["loss"].compute().item()}


def update_metrics(metrics, data, batch_embeddings, batch_cluster_ids, dist_metric):
    embeddings = unbatch(batch_embeddings, data.batch)
    cluster_ids = unbatch(batch_cluster_ids, data.batch)

    for pt in metrics["pt_thres"]:
        batch_mask = point_filter(batch_cluster_ids, data.reconstructable, data.pt, pt_thres=pt)
        mask = unbatch(batch_mask, data.batch)

        res = [acc_and_pr_at_k(embeddings[i], cluster_ids[i], mask[i], dist_metric) for i in range(len(embeddings))]
        res = torch.tensor(res)
        metrics[f"accuracy@{pt}"].update(res[:, 0])
        metrics[f"precision@{pt}"].update(res[:, 1])
        metrics[f"recall@{pt}"].update(res[:, 2])
        if pt == 0.9:
            acc_09 = res[:, 0].mean().item()
    return acc_09


def init_metrics(dataset_name):
    assert "tracking" in dataset_name
    pt_thres = [0, 0.5, 0.9]
    metric_names = ["accuracy", "precision", "recall"]
    metrics = {f"{name}@{pt}": MeanMetric(nan_strategy="error") for name in metric_names for pt in pt_thres}
    metrics["loss"] = MeanMetric(nan_strategy="error")
    metrics["pt_thres"] = pt_thres
    return metrics


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
