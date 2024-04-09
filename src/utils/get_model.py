import torch
from models.baselines import Transformer, GNNStack
from fvcore.nn import FlopCountAnalysis, flop_count_table


def get_model(model_name, model_kwargs, dataset, test_N=10000, test_k=100):
    model_type = model_name.split("_")[0]
    if model_type == "trans":
        model = Transformer(
            attn_type=model_name.split("_")[1],
            in_dim=dataset.x_dim,
            coords_dim=dataset.coords_dim,
            task=dataset.dataset_name,
            **model_kwargs,
        )
    elif model_type == "gnn":
        model = GNNStack(
            model_name=model_name.split("_")[1],
            in_dim=dataset.x_dim,
            h_dim=model_kwargs["hidden_dim"],
            n_layers=model_kwargs["num_layers"],
            task=dataset.dataset_name,
            **model_kwargs
        )
    else:
        raise NotImplementedError
    model.model_name = model_name
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params}")
    # count_flops_and_params(model, dataset, test_N, test_k)
    return model


@torch.no_grad()
def count_flops_and_params(model, dataset, N, k):
    E = k * N
    x = torch.randn((N, dataset.x_dim))
    edge_index = torch.randint(0, N, (2, E))
    coords = torch.randn((N, dataset.coords_dim))
    pos = coords[..., :2]
    batch = torch.zeros(N, dtype=torch.long)
    edge_weight = torch.randn((E, 1))

    if dataset.dataset_name == "pileup":
        x[..., -2:] = 0.0

    data = {"x": x, "edge_index": edge_index, "coords": coords, "pos": pos, "batch": batch, "edge_weight": edge_weight}
    print(flop_count_table(FlopCountAnalysis(model, data), max_depth=1))
