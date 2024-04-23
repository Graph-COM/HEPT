import math
import socket
import random
import torch
import numpy as np
from datetime import datetime
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR, StepLR
from torch_geometric.utils import batched_negative_sampling

from typing import Optional, Tuple
from torch import Tensor

from utils.losses import InfoNCELoss, FocalLoss


def compute_edge_weight(data):
    node_positions = data.pos
    node_indices = data.edge_index
    dist = torch.sum((node_positions[node_indices[0]] - node_positions[node_indices[1]]) ** 2, dim=1)
    dist = torch.unsqueeze(dist, dim=-1)
    edge_weights = -dist  # Calculate torch.exp(-dist / w) in the forward process
    return edge_weights


def log(*args):
    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]', *args)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_loss(loss_name, loss_kwargs):
    if loss_name == "infonce":
        return InfoNCELoss(**loss_kwargs)
    elif loss_name == "crossentropy":
        return torch.nn.BCELoss()
    elif loss_name == "focal":
        return FocalLoss()
    else:
        raise NotImplementedError


def get_optimizer(parameters, optimizer_name, optimizer_kwargs):
    if optimizer_name == "adam":
        return Adam(parameters, **optimizer_kwargs)
    elif optimizer_name == "adamw":
        return AdamW(parameters, **optimizer_kwargs)
    else:
        raise ValueError(f"Optimizer {optimizer_name} not supported!")


def get_lr_scheduler(optimizer, lr_scheduler_name, lr_scheduler_kwargs):
    if lr_scheduler_name is None:
        return None
    elif lr_scheduler_name == "impatient":
        del lr_scheduler_kwargs["num_training_steps"]
        return ReduceLROnPlateau(optimizer, **lr_scheduler_kwargs)
    elif lr_scheduler_name == "cosine":
        return get_cosine_schedule_with_warmup(optimizer, **lr_scheduler_kwargs)
    elif lr_scheduler_name == "step":
        del lr_scheduler_kwargs["num_training_steps"]
        return StepLR(optimizer, **lr_scheduler_kwargs)
    else:
        raise ValueError(f"LR scheduler {lr_scheduler_name} not supported!")


def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, eta_min, num_cycles=0.5, last_epoch=-1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        num_cycles (:obj:`float`, `optional`, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(eta_min, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def find_first_available_port(starting_port):
    port = starting_port
    while True:
        try:
            # Attempt to create a socket and bind it to the port
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", port))
                # If successful, return the port
                return port
        except OSError:
            # If the port is already in use, try the next one
            port += 1


# https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/utils/augmentation.html#add_random_edge
def add_random_edge(
    edge_index,
    p: float = 0.5,
    force_undirected: bool = False,
    batch: Optional[Tensor] = None,
    training: bool = True,
) -> Tuple[Tensor, Tensor]:
    r"""Randomly adds edges to :obj:`edge_index`.

    The method returns (1) the retained :obj:`edge_index`, (2) the added
    edge indices.

    Args:
        edge_index (LongTensor): The edge indices.
        p (float): Ratio of added edges to the existing edges.
            (default: :obj:`0.5`)
        force_undirected (bool, optional): If set to :obj:`True`,
            added edges will be undirected.
            (default: :obj:`False`)
        num_nodes (int, Tuple[int], optional): The overall number of nodes,
            *i.e.* :obj:`max_val + 1`, or the number of source and
            destination nodes, *i.e.* :obj:`(max_src_val + 1, max_dst_val + 1)`
            of :attr:`edge_index`. (default: :obj:`None`)
        training (bool, optional): If set to :obj:`False`, this operation is a
            no-op. (default: :obj:`True`)

    :rtype: (:class:`LongTensor`, :class:`LongTensor`)

    Examples:

        >>> # Standard case
        >>> edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
        ...                            [1, 0, 2, 1, 3, 2]])
        >>> edge_index, added_edges = add_random_edge(edge_index, p=0.5)
        >>> edge_index
        tensor([[0, 1, 1, 2, 2, 3, 2, 1, 3],
                [1, 0, 2, 1, 3, 2, 0, 2, 1]])
        >>> added_edges
        tensor([[2, 1, 3],
                [0, 2, 1]])

        >>> # The returned graph is kept undirected
        >>> edge_index, added_edges = add_random_edge(edge_index, p=0.5,
        ...                                           force_undirected=True)
        >>> edge_index
        tensor([[0, 1, 1, 2, 2, 3, 2, 1, 3, 0, 2, 1],
                [1, 0, 2, 1, 3, 2, 0, 2, 1, 2, 1, 3]])
        >>> added_edges
        tensor([[2, 1, 3, 0, 2, 1],
                [0, 2, 1, 2, 1, 3]])

        >>> # For bipartite graphs
        >>> edge_index = torch.tensor([[0, 1, 2, 3, 4, 5],
        ...                            [2, 3, 1, 4, 2, 1]])
        >>> edge_index, added_edges = add_random_edge(edge_index, p=0.5,
        ...                                           num_nodes=(6, 5))
        >>> edge_index
        tensor([[0, 1, 2, 3, 4, 5, 3, 4, 1],
                [2, 3, 1, 4, 2, 1, 1, 3, 2]])
        >>> added_edges
        tensor([[3, 4, 1],
                [1, 3, 2]])
    """
    if p < 0.0 or p > 1.0:
        raise ValueError(f"Ratio of added edges has to be between 0 and 1 " f"(got '{p}')")

    device = edge_index.device
    if not training or p == 0.0:
        edge_index_to_add = torch.tensor([[], []], device=device)
        return edge_index_to_add

    edge_index_to_add = batched_negative_sampling(
        edge_index=edge_index,
        batch=batch,
        num_neg_samples=round(edge_index.size(1) * p / (batch.max().item() + 1)),
        force_undirected=force_undirected,
    )

    # edge_index = torch.cat([edge_index, edge_index_to_add], dim=1)

    return edge_index_to_add
