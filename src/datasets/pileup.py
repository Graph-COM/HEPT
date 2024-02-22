import sys

sys.path.append("../")

import os
import shutil
import uproot
import awkward as ak
import os.path as osp
from tqdm import tqdm
from joblib import Parallel, delayed

import numpy as np
import torch
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.nn import knn_graph
from torch_geometric.utils import to_undirected
from utils import set_seed, compute_edge_weight
from utils import download_url, extract_zip, decide_download

class PileupTransform(BaseTransform):
    def __call__(self, data):
        # data.edge_weight = compute_edge_weight(data)  # uncomment this line for GCN
        data.coords = torch.cat([data.pos, data.x[:, :2]], dim=-1)
        return data


class Pileup(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, seed=42):
        set_seed(seed)
        self.url_processed = "https://zenodo.org/records/10694703/files/pileup-10k-processed.zip"
        super(Pileup, self).__init__(root, transform, pre_transform)
        self.data, self.slices, self.idx_split = torch.load(self.processed_paths[0])
        self.x_dim = self._data.x.shape[1]
        self.coords_dim = 2 + 2

    @property
    def raw_file_names(self):
        return ["testTTbar_part1.root", "testTTbar_part2.root"]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        warning = "This dataset would need 11 GB of space after extraction. Do you want to continue? (y/n)\n"
        if osp.exists(self.processed_paths[0]):
            return
        if decide_download(self.url_processed) and input(warning).lower() == "y":
            path = download_url(self.url_processed, self.root)
            extract_zip(path, str(self.root) + "/processed")
            os.unlink(path)
        else:
            print("Stop downloading.")
            shutil.rmtree(self.root)
            exit(-1)

    def process(self):
        train_data = prepare_dataset(self.raw_paths[0])
        val_test_data = prepare_dataset(self.raw_paths[1])
        data, slices = self.collate(train_data + val_test_data)

        idx_split = self.get_idx_split(train_data, val_test_data)
        torch.save((data, slices, idx_split), self.processed_paths[0])

    def get_idx_split(self, train_data, val_test_data):
        n_train = len(train_data)
        n_valid = len(val_test_data) // 2
        n_test = len(val_test_data) - n_valid
        dataset_len = n_train + n_valid + n_test

        idx = np.arange(dataset_len)
        np.random.shuffle(idx[n_train:])
        train_idx = idx[:n_train]
        valid_idx = idx[n_train : n_train + n_valid]
        test_idx = idx[n_train + n_valid :]
        return {"train": train_idx, "valid": valid_idx, "test": test_idx}


def prepare_dataset(datadir):
    features = [
        "PF/PF.PT",
        "PF/PF.Eta",
        "PF/PF.Phi",
        "PF/PF.Charge",
        "PF/PF.IsPU",
        "PF/PF.PID",
        "PF/PF.Rapidity",
        "PF/PF.E",
        "PF/PF.Px",
        "PF/PF.Py",
    ]
    tree = uproot.open(datadir)["Delphes"]
    num_entries = tree.num_entries
    particles = tree.arrays(features, entry_start=0, entry_stop=0 + num_entries)
    data_list = Parallel(n_jobs=32, prefer="threads")(
        delayed(process_one_event)(particles[i]) for i in tqdm(range(num_entries))
    )
    return data_list


def process_one_event(event):
    nparticles = len(event["PF/PF.PT"])

    pt = ak.to_numpy(event["PF/PF.PT"])
    chg = ak.to_numpy(event["PF/PF.Charge"])
    eta = ak.to_numpy(event["PF/PF.Eta"])
    phi = ak.to_numpy(event["PF/PF.Phi"])
    pids = ak.to_numpy(event["PF/PF.PID"])
    px, py = ak.to_numpy(event["PF/PF.Px"]), ak.to_numpy(event["PF/PF.Py"])
    rapidity, E = ak.to_numpy(event["PF/PF.Rapidity"]), ak.to_numpy(event["PF/PF.E"])

    pids[chg != 0] = 0
    pids[pids == 22] = 1
    pids[pids == 130] = 2
    pids[pids == 310] = 3
    pids[np.abs(pids) == 2112] = 4
    pids[np.abs(pids) == 3122] = 5
    pids[np.abs(pids) == 3322] = 6
    assert np.abs(pids).max() == 6

    node_features = torch.from_numpy(np.stack((eta, phi, px, py, pt, E, rapidity, pids), axis=1)).float()
    label = torch.from_numpy(ak.to_numpy(event["PF/PF.IsPU"] == 0)).long()
    is_neu = torch.from_numpy((chg == 0) & (pt > 0.9))
    pos = torch.from_numpy(np.stack([eta, phi], axis=1))

    # In case all positive points are put at the beginning
    perm = np.random.permutation(nparticles)
    node_features = node_features[perm]
    label = label[perm]
    is_neu = is_neu[perm]
    pos = pos[perm]
    edge_index = to_undirected(knn_graph(pos, k=60, loop=True))

    data = Data(x=node_features, edge_index=edge_index, y=label, pos=pos, is_neu=is_neu)
    return data


if __name__ == "__main__":
    root = "../../data/pileup"
    dataset = Pileup(root)
