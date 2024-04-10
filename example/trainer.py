from tqdm import tqdm
import torch
from torchmetrics import MeanMetric
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR, StepLR
from torch_geometric.utils import unbatch, remove_self_loops, to_undirected, subgraph

from utils.metrics import acc_and_pr_at_k, point_filter


def get_new_idx_split(dataset):
    sorted_evtid = dataset.evtid.argsort()
    dataset_len = len(dataset)

    split = {"train": 0.8, "valid": 0.1, "test": 0.1}
    n_train = int(dataset_len * split["train"])
    n_train = n_train - n_train % 10
    n_valid = int(dataset_len * split["valid"])

    idx = sorted_evtid
    train_idx = idx[:n_train]
    valid_idx = idx[n_train : n_train + n_valid]
    test_idx = idx[n_train + n_valid :]
    return {"train": train_idx, "valid": valid_idx, "test": test_idx}


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


def get_new_idx_split(dataset):
    sorted_evtid = dataset.evtid.argsort()
    dataset_len = len(dataset)

    split = {"train": 0.8, "valid": 0.1, "test": 0.1}
    n_train = int(dataset_len * split["train"])
    n_train = n_train - n_train % 10
    n_valid = int(dataset_len * split["valid"])

    idx = sorted_evtid
    train_idx = idx[:n_train]
    valid_idx = idx[n_train : n_train + n_valid]
    test_idx = idx[n_train + n_valid :]
    return {"train": train_idx, "valid": valid_idx, "test": test_idx}
