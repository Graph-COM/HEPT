from torch_geometric.loader import DataLoader
from datasets import Tracking, TrackingTransform, Pileup, PileupTransform


def get_data_loader(dataset, idx_split, batch_size):
    train_loader = DataLoader(
        dataset[idx_split["train"]],
        batch_size=batch_size,
        shuffle=True,
    )
    valid_loader = DataLoader(
        dataset[idx_split["valid"]],
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        dataset[idx_split["test"]],
        batch_size=batch_size,
        shuffle=False,
    )
    return {"train": train_loader, "valid": valid_loader, "test": test_loader}


def get_dataset(dataset_name, data_dir):
    if "tracking" in dataset_name:
        dataset = Tracking(data_dir, transform=TrackingTransform(), dataset_name=dataset_name)
    elif dataset_name == "pileup":
        dataset = Pileup(data_dir, transform=PileupTransform())
    else:
        raise NotImplementedError
    dataset.dataset_name = dataset_name
    return dataset
