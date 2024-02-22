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
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR

from utils import set_seed, get_optimizer, log, get_lr_scheduler, get_loss
from utils.get_data import get_data_loader, get_dataset
from utils.get_model import get_model
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score


def train_one_batch(model, optimizer, criterion, data, lr_s):
    model.train()
    embeddings = model(data)
    loss = criterion(embeddings[data.is_neu], data.y[data.is_neu].unsqueeze(-1).float())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if lr_s is not None and isinstance(lr_s, LambdaLR):
        lr_s.step()
    return loss.item(), embeddings.detach()


@torch.no_grad()
def eval_one_batch(model, optimizer, criterion, data, lr_s):
    model.eval()
    embeddings = model(data)
    loss = criterion(embeddings[data.is_neu], data.y[data.is_neu].unsqueeze(-1).float())
    return loss.item(), embeddings.detach()


def run_one_epoch(model, optimizer, criterion, data_loader, phase, epoch, device, metrics, lr_s):
    run_one_batch = train_one_batch if phase == "train" else eval_one_batch
    phase = "test " if phase == "test" else phase

    pbar = tqdm(data_loader, disable=__name__ != "__main__")
    for idx, data in enumerate(pbar):
        data = data.to(device)
        batch_loss, batch_embeddings = run_one_batch(model, optimizer, criterion, data, lr_s)
        batch_auc = update_metrics(metrics, data, batch_embeddings)
        metrics["loss"].update(batch_loss)

        desc = f"[Epoch {epoch}] {phase}, loss: {batch_loss:.4f}, auc: {batch_auc:.4f}"
        if idx == len(data_loader) - 1:
            metric_res = compute_metrics(metrics)
            loss, auc, f1, roc = metric_res["loss"], metric_res["auc"], metric_res["f1"], metric_res["roc"]
            desc = f"[Epoch {epoch}] {phase}, loss: {loss:.5f}, auc: {auc:.4f}, f1: {f1:.4f}, roc: {roc:.4f}"
            reset_metrics(metrics)
        pbar.set_description(desc)
    return metric_res


def reset_metrics(metrics):
    for metric in metrics.values():
        if isinstance(metric, MeanMetric):
            metric.reset()


def compute_metrics(metrics):
    return {f"{name}": metrics[f"{name}"].compute().item() for name in ["auc", "f1", "roc"]} | {
        "loss": metrics["loss"].compute().item()
    }


def update_metrics(metrics, data, embeddings):
    pred = (embeddings > 0.5).int()[data.is_neu].cpu()
    label = data.y[data.is_neu].cpu()
    embeddings = embeddings[data.is_neu].cpu()

    auc = average_precision_score(label, embeddings)
    roc = roc_auc_score(label, embeddings)
    f1 = f1_score(label, pred)

    metrics["auc"].update(auc)
    metrics["f1"].update(f1)
    metrics["roc"].update(roc)
    return auc


def run_one_seed(config, tune=False):
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(config["num_threads"])

    dataset_name = config["dataset_name"]
    model_name = config["model_name"]
    dataset_dir = Path(config["data_dir"]) / dataset_name
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
    metric_names = ["auc", "f1", "roc"]
    metrics = {f"{name}": MeanMetric() for name in metric_names}
    metrics["loss"] = MeanMetric()

    coef = 1 if config["mode"] == "max" else -1
    best_epoch, best_train = 0, {metric: -coef * float("inf") for metric in metrics.keys()}
    best_valid, best_test = deepcopy(best_train), deepcopy(best_train)

    if writer is not None:
        layout = {
            "Gap": {
                "loss": ["Multiline", ["train/loss", "valid/loss", "test/loss"]],
                "auc": ["Multiline", ["train/auc", "valid/auc", "test/auc"]],
            }
        }
        writer.add_custom_scalars(layout)

    for epoch in range(config["num_epochs"]):
        if not config.get("only_eval", False):
            train_res = run_one_epoch(model, opt, criterion, loaders["train"], "train", epoch, device, metrics, lr_s)
        valid_res = run_one_epoch(model, opt, criterion, loaders["valid"], "valid", epoch, device, metrics, lr_s)
        test_res = run_one_epoch(model, opt, criterion, loaders["test"], "test", epoch, device, metrics, lr_s)

        if lr_s is not None and isinstance(lr_s, ReduceLROnPlateau):
            lr_s.step(valid_res[config["lr_scheduler_metric"]])

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
    parser = argparse.ArgumentParser(description="Train a model for pileup.")
    parser.add_argument("-m", "--model", type=str, default="gcn")
    args = parser.parse_args()

    if args.model in ["gcn", "gatedgnn", "dgcnn", "gravnet"]:
        config_dir = Path(f"./configs/pileup/pileup_gnn_{args.model}.yaml")
    else:
        config_dir = Path(f"./configs/pileup/pileup_trans_{args.model}.yaml")
    config = yaml.safe_load(config_dir.open("r").read())
    run_one_seed(config)


if __name__ == "__main__":
    main()
