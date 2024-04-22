import nni
import yaml
import argparse
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from datetime import datetime

import numpy as np
import torch
from torchmetrics import MeanMetric
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR, StepLR
from torch_geometric.utils import unbatch, remove_self_loops, to_undirected, subgraph
from torch_geometric.nn import knn_graph, radius_graph

from utils import set_seed, get_optimizer, log, get_lr_scheduler, add_random_edge, get_loss
from utils.get_data import get_data_loader, get_dataset
from utils.get_model import get_model
from utils.metrics import acc_and_pr_at_k, point_filter


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

    pbar = tqdm(data_loader, disable=__name__ != "__main__")
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


def run_one_seed(config):
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(config["num_threads"])

    dataset_name = config["dataset_name"]
    model_name = config["model_name"]
    dataset_dir = Path(config["data_dir"]) / dataset_name.split("-")[0]
    log(f"Device: {device}, Model: {model_name}, Dataset: {dataset_name}, Note: {config['note']}")

    time = datetime.now().strftime("%m_%d-%H_%M_%S.%f")[:-4]
    rand_num = np.random.randint(10, 100)
    log_dir = dataset_dir / "logs" / f"{time}{rand_num}_{model_name}_{config['seed']}_{config['note']}"
    log(f"Log dir: {log_dir}")
    log_dir.mkdir(parents=True, exist_ok=False)
    writer = SummaryWriter(log_dir) if config["log_tensorboard"] else None

    set_seed(config["seed"])
    dataset = get_dataset(dataset_name, dataset_dir)
    loaders = get_data_loader(dataset, dataset.idx_split, batch_size=config["batch_size"])

    model = get_model(model_name, config["model_kwargs"], dataset)
    if config.get("only_flops", False):
        raise RuntimeError
    if config.get("resume", False):
        log(f"Resume from {config['resume']}")
        model_path = dataset_dir / "logs" / (config["resume"] + "/best_model.pt")
        checkpoint = torch.load(model_path, map_location="cpu")
        model.load_state_dict(checkpoint, strict=True)
    model = model.to(device)

    opt = get_optimizer(model.parameters(), config["optimizer_name"], config["optimizer_kwargs"])
    config["lr_scheduler_kwargs"]["num_training_steps"] = config["num_epochs"] * len(loaders["train"])
    lr_s = get_lr_scheduler(opt, config["lr_scheduler_name"], config["lr_scheduler_kwargs"])
    criterion = get_loss(config["loss_name"], config["loss_kwargs"])

    main_metric = config["main_metric"]
    pt_thres = [0, 0.5, 0.9]
    metric_names = ["accuracy", "precision", "recall"]
    metrics = {f"{name}@{pt}": MeanMetric(nan_strategy="error") for name in metric_names for pt in pt_thres}
    metrics["loss"] = MeanMetric(nan_strategy="error")
    metrics["pt_thres"] = pt_thres

    coef = 1 if config["mode"] == "max" else -1
    best_epoch, best_train = 0, {metric: -coef * float("inf") for metric in metrics.keys()}
    best_valid, best_test = deepcopy(best_train), deepcopy(best_train)

    if writer is not None:
        layout = {
            "Gap": {
                "loss": ["Multiline", ["train/loss", "valid/loss", "test/loss"]],
                "acc@0.9": ["Multiline", ["train/accuracy@0.9", "valid/accuracy@0.9", "test/accuracy@0.9"]],
            }
        }
        writer.add_custom_scalars(layout)

    for epoch in range(config["num_epochs"]):
        if not config.get("only_eval", False):
            train_res = run_one_epoch(model, opt, criterion, loaders["train"], "train", epoch, device, metrics, lr_s)
        valid_res = run_one_epoch(model, opt, criterion, loaders["valid"], "valid", epoch, device, metrics, lr_s)
        test_res = run_one_epoch(model, opt, criterion, loaders["test"], "test", epoch, device, metrics, lr_s)

        if lr_s is not None:
            if isinstance(lr_s, ReduceLROnPlateau):
                lr_s.step(valid_res[config["lr_scheduler_metric"]])
            elif isinstance(lr_s, StepLR):
                lr_s.step()

        if (valid_res[main_metric] * coef) > (best_valid[main_metric] * coef):
            best_epoch, best_train, best_valid, best_test = epoch, train_res, valid_res, test_res
            torch.save(model.state_dict(), log_dir / "best_model.pt")

        print(
            f"[Epoch {epoch}] Best epoch: {best_epoch}, train: {best_train[main_metric]:.4f}, "
            f"valid: {best_valid[main_metric]:.4f}, test: {best_test[main_metric]:.4f}"
        )
        print("=" * 50), print("=" * 50)

        if writer is not None:
            writer.add_scalar("lr", opt.param_groups[0]["lr"], epoch)
            for phase, res in zip(["train", "valid", "test"], [train_res, valid_res, test_res]):
                for k, v in res.items():
                    writer.add_scalar(f"{phase}/{k}", v, epoch)
            for phase, res in zip(["train", "valid", "test"], [best_train, best_valid, best_test]):
                for k, v in res.items():
                    writer.add_scalar(f"best_{phase}/{k}", v, epoch)

def main():
    parser = argparse.ArgumentParser(description="Train a model for tracking.")
    parser.add_argument("-m", "--model", type=str, default="hept")
    args = parser.parse_args()

    if args.model in ["gcn", "gatedgnn", "dgcnn", "gravnet"]:
        config_dir = Path(f"./configs/tracking/tracking_gnn_{args.model}.yaml")
    else:
        config_dir = Path(f"./configs/tracking/tracking_trans_{args.model}.yaml")
    config = yaml.safe_load(config_dir.open("r").read())
    run_one_seed(config)


if __name__ == "__main__":
    main()
