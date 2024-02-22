import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as pyg_nn
from torch_geometric.nn import MLP
from math import pi
from torch_geometric.nn import GCNConv, DynamicEdgeConv, GravNetConv
from torch_cluster import knn
from torch_geometric.typing import Adj, OptTensor, PairOptTensor, PairTensor
from typing import Callable, Optional, Union
from torch import Tensor

try:
    knn = torch.compiler.disable(knn)
except AttributeError:
    pass


class GNNStack(torch.nn.Module):
    def __init__(self, in_dim, h_dim, n_layers, model_name, task, **kwargs):
        super(GNNStack, self).__init__()
        conv_model = self.build_conv_model(model_name)
        self.task = task
        self.h_dim = h_dim
        self.n_layers = n_layers

        # discrete feature to embedding
        w_out = kwargs["out_dim"]
        if self.task == "pileup":
            self.pids_enc = nn.Embedding(7, 10)
            in_dim = in_dim - 1 + 10
            w_out = int(self.h_dim // 2)
            self.out_proj = nn.Linear(int(self.h_dim // 2), kwargs["out_dim"])

        self.feat_encoder = nn.Sequential(
            nn.Linear(in_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
        )

        self.convs = nn.ModuleList()
        self.norm1s = nn.ModuleList()
        self.norm2s = nn.ModuleList()
        self.ffs = nn.ModuleList()
        self.pre_ffs = nn.ModuleList()
        for _ in range(self.n_layers):
            self.pre_ffs.append(
                nn.Sequential(
                    nn.LayerNorm(self.h_dim),
                    nn.Linear(self.h_dim, self.h_dim),
                )
            )

            self.convs.append(conv_model(self.h_dim, self.h_dim, **kwargs))
            self.norm1s.append(nn.LayerNorm(self.h_dim))
            self.norm2s.append(nn.LayerNorm(self.h_dim))
            self.ffs.append(
                nn.Sequential(
                    nn.Linear(self.h_dim, self.h_dim),
                    nn.ReLU(),
                    nn.Linear(self.h_dim, self.h_dim),
                )
            )

        self.W = nn.Linear(self.h_dim * (self.n_layers + 1), w_out, bias=False)
        self.mlp_out = MLP(
            in_channels=w_out,
            out_channels=w_out,
            hidden_channels=256,
            num_layers=5,
            norm="layer_norm",
            act="tanh",
            norm_kwargs={"mode": "node"},
        )
        self.attn_type = model_name

    def build_conv_model(self, model_type):
        if model_type == 'gatedgnn':
            return Gated_model
        elif model_type == 'gcn':
            return CustomGCNConv
        elif model_type == 'dgcnn':
            return CustomDGCNNConv
        elif model_type == 'gravnet':
            return CustomGravNetConv
        else:
            raise NotImplementedError('conv model type not found')

    def forward(self, data):
        if isinstance(data, dict):
            x, edge_index, coords, edge_weight = data["x"], data["edge_index"], data["coords"], data["edge_weight"]
        else:
            x, edge_index, coords, edge_weight = data.x, data.edge_index, data.coords, data.edge_weight

        # For Gated GNN only
        eta = coords[:, 0]
        phi = coords[:, 1]

        # discrete feature to embedding
        if self.task == "pileup":
            pids_emb = self.pids_enc(x[..., -1].long())
            x = torch.cat((x[..., :-1], pids_emb), dim=-1)

        encoded_x = self.feat_encoder(x)
        all_encoded_x = [encoded_x]
        for idx, layer in enumerate(self.convs):
            aggr_out = layer(self.pre_ffs[idx](encoded_x), edge_index, eta, phi, edge_weight)

            encoded_x = encoded_x + F.dropout(aggr_out, p=0.1, training=self.training)
            ff_output = self.ffs[idx](self.norm2s[idx](encoded_x))
            encoded_x = encoded_x + F.dropout(ff_output, p=0.1, training=self.training)

            all_encoded_x.append(encoded_x)

        encoded_x = self.W(torch.cat(all_encoded_x, dim=-1))
        out = encoded_x + F.dropout(self.mlp_out(encoded_x), p=0.1, training=self.training)

        if self.task == "pileup":
            out = self.out_proj(out)
            out = torch.sigmoid(out)
        return out


class Gated_model(pyg_nn.MessagePassing):
    def __init__(self, in_channels, out_channels, normalize_embedding=True, **kwargs):
        super(Gated_model, self).__init__(aggr='mean')
        # last sclar = d_eta, d_phi, d_R, append x_i, x_j, x_g so * 3，+1 for log count
        new_x_input = 3 * (in_channels) + 3 + 1
        self.x_dim = new_x_input
        self.lin_m2 = torch.nn.Linear(new_x_input, 1)
        # also append x and x_g, so + 2 * in_channels, +1 for log count in the global node
        self.lin_m5 = torch.nn.Linear(new_x_input + 2 * in_channels + 1, 1)
        self.lin_m5_g1 = torch.nn.Linear(in_channels, out_channels)
        self.lin_m5_g2 = torch.nn.Linear(new_x_input, out_channels)
        self.edge_weight_w = nn.Parameter(torch.randn([1, 1]))

    def forward(self, x, edge_index, eta, phi, edge_weight=None):
        num_nodes = x.size(0)
        x = torch.cat((x, eta.view(-1, 1)), dim=1)
        x = torch.cat((x, phi.view(-1, 1)), dim=1)
        return self.propagate(edge_index, size=(num_nodes, num_nodes), x=x)

    def message(self, x_j, x_i, edge_index, size, x):
        self.node_count = torch.tensor(size[0]).type(torch.float32).to(x.device)
        # eta at position 0
        dif_eta_phi = x_j[:, -2: x_j.size()[1]] - x_i[:, -2: x_i.size()[1]]

        # make sure delta within 2pi
        indices = dif_eta_phi[:, 1] > pi
        temp = torch.ceil((dif_eta_phi[:, 1][indices] - pi) / (2 * pi)) * (2 * pi)
        dif_eta_phi[:, 1][indices] = dif_eta_phi[:, 1][indices] - temp

        delta_r = torch.sum(dif_eta_phi ** 2, dim=1).reshape(-1, 1)
        delta_r = torch.exp(-delta_r / self.edge_weight_w.exp())

        x = x[:, 0:-2]
        x_i = x_i[:, 0:-2]
        x_j = x_j[:, 0:-2]
        x_g = torch.mean(x, dim=0)
        log_count = torch.log(self.node_count)
        log_count = log_count.repeat(x_i.size()[0], 1)
        x_g = x_g.repeat(x_i.size()[0], 1)
        x_j = torch.cat((x_j, x_i, x_g, dif_eta_phi, delta_r, log_count), dim=1)
        M_1 = self.lin_m2(x_j)
        M_2 = torch.sigmoid(M_1)
        x_j = x_j * M_2
        return x_j

    def update(self, aggr_out, x):
        x = x[:, 0:-2]
        x_g = torch.mean(x, dim=0)
        log_count = torch.log(self.node_count)
        log_count = log_count.repeat(x.size()[0], 1)
        x_g = x_g.repeat(x.size()[0], 1)
        aggr_out_temp = aggr_out
        aggr_out = torch.cat((aggr_out, x, x_g, log_count), dim=1)
        aggr_out = torch.sigmoid(self.lin_m5(aggr_out))
        aggr_out = F.relu(aggr_out * self.lin_m5_g1(x) + (1 - aggr_out) * self.lin_m5_g2(aggr_out_temp))
        return aggr_out


class CustomGCNConv(GCNConv):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CustomGCNConv, self).__init__(in_channels, out_channels, **kwargs)
        self.edge_weight_w = nn.Parameter(torch.randn([1, 1]))

    def forward(self, x, edge_index, eta, phi, edge_weight=None):
        edge_weight = torch.exp(edge_weight / self.edge_weight_w.exp())
        x = super(CustomGCNConv, self).forward(x, edge_index, edge_weight)
        return x


class CustomDGCNNConv(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CustomDGCNNConv, self).__init__()
        self.model = CustomDynamicEdgeConv(in_channels, out_channels, **kwargs)

    def forward(self, x, edge_index, eta, phi, edge_weight=None):
        x = self.model(x, edge_index, eta, phi, edge_weight)
        return x


class CustomDynamicEdgeConv(DynamicEdgeConv):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CustomDynamicEdgeConv, self).__init__(nn=nn.Sequential(
            nn.Linear(2 * in_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.ReLU()),
            k=kwargs['k'],
            aggr='mean')

        self.lin_s = nn.Linear(in_channels, kwargs['knn_dim'])

    # def forward(self, x, edge_index, eta, phi, edge_weight=None):
    #     x = super(CustomDynamicEdgeConv, self).forward(x)
    #     return x

    '''
    The only difference between this implementation and the original implementation
    in pyg library is that we use a linear layer (self.lin_s) to project the x to
    a specific dimension (knn_dim) before calculating the knn graph.
    '''
    def forward(
            self, x: Union[Tensor, PairTensor],
            edge_index=None,
            eta=None,
            phi=None,
            edge_weight: OptTensor = None,
            batch: Union[OptTensor, Optional[PairTensor]] = None) -> Tensor:
        # type: (Tensor, OptTensor) -> Tensor  # noqa
        # type: (PairTensor, Optional[PairTensor]) -> Tensor  # noqa

        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        if x[0].dim() != 2:
            raise ValueError("Static graphs not supported in DynamicEdgeConv")

        b: PairOptTensor = (None, None)
        if isinstance(batch, Tensor):
            b = (batch, batch)
        elif isinstance(batch, tuple):
            assert batch is not None
            b = (batch[0], batch[1])

        s_l: Tensor = self.lin_s(x[0])
        s_r: Tensor = self.lin_s(x[1])
        edge_index = knn(s_l, s_r, self.k, b[0], b[1]).flip([0])

        # propagate_type: (x: PairTensor)
        return self.propagate(edge_index, x=x, size=None)


class CustomGravNetConv(GravNetConv):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CustomGravNetConv, self).__init__(in_channels,
                                                out_channels,
                                                space_dimensions=kwargs['knn_dim'],
                                                propagate_dimensions=32,
                                                k=kwargs['k'])
        self.edge_weight_w = nn.Parameter(torch.randn([1]))

    def forward(self, x, edge_index, eta, phi, edge_weight=None, batch=None):
        is_bipartite: bool = True
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)
            is_bipartite = False

        if x[0].dim() != 2:
            raise ValueError("Static graphs not supported in 'GravNetConv'")

        b: PairOptTensor = (None, None)
        if isinstance(batch, Tensor):
            b = (batch, batch)
        elif isinstance(batch, tuple):
            assert batch is not None
            b = (batch[0], batch[1])

        h_l: Tensor = self.lin_h(x[0])

        s_l: Tensor = self.lin_s(x[0])
        s_r: Tensor = self.lin_s(x[1]) if is_bipartite else s_l

        edge_index = knn(s_l, s_r, self.k, b[0], b[1]).flip([0])

        edge_weight = (s_l[edge_index[0]] - s_r[edge_index[1]]).pow(2).sum(-1)
        edge_weight = torch.exp(-edge_weight * self.edge_weight_w.exp())  # 10 gives a better spread

        # propagate_type: (x: OptPairTensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=(h_l, None),
                             edge_weight=edge_weight,
                             size=(s_l.size(0), s_r.size(0)))

        return self.lin_out1(x[1]) + self.lin_out2(out)
