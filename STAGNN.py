import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv
import numpy as np
from torch_geometric.nn import GCNConv


class TemporalConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super(TemporalConv, self).__init__()
        padding = (0, kernel_size // 2)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size), padding=padding)
        self.conv_2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size), padding=padding)
        self.conv_3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size), padding=padding)

    def forward(self, X: torch.FloatTensor) -> torch.FloatTensor:
        X = X.permute(0, 3, 2, 1)
        P = self.conv_1(X)
        Q = torch.sigmoid(self.conv_2(X))
        PQ = P * Q
        H = F.relu(PQ + self.conv_3(X))
        H = F.dropout(H, p=0.5, training=self.training)
        H = H.permute(0, 3, 2, 1)
        return H

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=0, alpha=0.7,
                 dropout=0.5, save_mem=True, use_bn=True, use_resi=True):
        super(GCN, self).__init__()

        self.convs = nn.ModuleList()
        # self.convs.append(
        #     GCNConv(in_channels, hidden_channels, cached=not save_mem, normalize=not save_mem))
        #self.convs.append(
        #    GCNConv(hidden_channels, hidden_channels, cached=not save_mem))

        self.bns = nn.ModuleList()
        #self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.conv_0 = GCNConv(in_channels, hidden_channels, cached=not save_mem)
        self.bn_0 = nn.BatchNorm1d(hidden_channels)
        self.conv_1 = GCNConv(hidden_channels, out_channels, cached=not save_mem)
        self.bn_1 = nn.BatchNorm1d(out_channels)
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        for _ in range(num_layers):
            # self.convs.append(
            #     GCNConv(hidden_channels, hidden_channels, cached=not save_mem, normalize=not save_mem))
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=not save_mem))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        # self.convs.append(
        #     GCNConv(hidden_channels, out_channels, cached=not save_mem, normalize=not save_mem))
        # self.convs.append(
        #     GCNConv(hidden_channels, hidden_channels, cached=not save_mem))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.use_res = use_resi
        self.alpha = alpha
        #self.fc = nn.Linear(64*30, out_channels)
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        #edge_weight=data.graph['edge_weight'] if 'edge_weight' in data.graph else None
        edge_weight = data.edge_attr
        layer_ = []

        #x = self.fc1(x)

        x = self.conv_0(x, edge_index, edge_weight)
        if self.use_bn:
            x = self.bn_0(x)
        x = self.activation(x)
        x = F.dropout(x, p=0.5, training=self.training)
        layer_.append(x)
        for i in range(len(self.convs)):
            conv = self.convs[i]
            if edge_weight is None:
                x = conv(x, edge_index)
            else:
                x = conv(x, edge_index, edge_weight)
            if self.use_res:
                #x = self.alpha * x + (1 - self.alpha) * layer_[i]
                x =  x + layer_[0]
            if self.use_bn:
                x = self.bns[i](x)
            x = self.activation(x)
            #x = F.dropout(x, p=self.dropout, training=self.training)
            layer_.append(x)
        x = self.conv_1(x, edge_index, edge_weight)
        if self.use_bn:
            x = self.bn_1(x)
        x = self.activation(x)
        #x = F.dropout(x, p=0.5, training=self.training)
        layer_.append(x)
        # for i, conv in enumerate(self.convs):
        #     if edge_weight is None:
        #         x = conv(x, edge_index)
        #     else:
        #         x=conv(x,edge_index,edge_weight)
        #     if self.use_res:
        #         x = self.alpha * x + (1 - self.alpha) * layer_[i]
        #     if self.use_bn:
        #         x = self.bns[i](x)
        #     x = self.activation(x)
        #     #x = F.dropout(x, p=self.dropout, training=self.training)
        #     layer_.append(x)
        #x = self.convs[-1](x, edge_index)
        # x = self.activation(x)
        # x = F.dropout(x, p=self.dropout, training=self.training)
        # x = torch.flatten(x, start_dim=1)
        # x = x.view(-1, 64 * 30)
        # x = self.fc(x)
        return x

class STConv(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        kernel_size: int,
        K: int,
        normalization: str = "sym",
        bias: bool = True,
    ):
        super(STConv, self).__init__()
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.K = K
        self.normalization = normalization
        self.bias = bias
        self.fc = nn.Linear(6384, self.out_channels * 2048)
        # self.fc1 = nn.Linear(2048, 1024)
        # self.fc2 = nn.Linear(1024, 1024)
        # self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(2048, 300)
        self.params = list(self.fc.parameters())
        # self.params.extend(list(self.fc1.parameters()))
        # self.params.extend(list(self.fc2.parameters()))
        # self.params.extend(list(self.fc3.parameters()))
        self.params.extend(list(self.fc4.parameters()))

        self._temporal_conv1 = TemporalConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
        )
        self.gcn = GCN(20, hidden_channels, hidden_channels)
        self._graph_conv = ChebConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            K=K,
            normalization=normalization,
            bias=bias,
        )

        self._temporal_conv2 = TemporalConv(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
        )

        self._batch_norm = nn.BatchNorm2d(num_nodes)
    def forward(
        self,
        data
    ) -> torch.FloatTensor:
        x_yuanshi=data.x,
        X = data.x.reshape(-1, 20, self.num_nodes, self.in_channels)
        #edge_index = edge_index.to(X.device)
        #edge_index = data.edge_index#.cpu().numpy()
        #edge_index = torch.from_numpy(edge_index).to(X.device)
        edge_weight = None
        T_0 = self._temporal_conv1(X)
        # T_0 = F.dropout(T_0, p=0.5, training=self.training)
        gcn_feature = self.gcn(data)
        gcn_feature = gcn_feature.permute(1, 0).reshape(-1, 1, self.num_nodes, self.hidden_channels)
        T = F.relu(T_0)
        T = torch.cat((T, gcn_feature), dim=1)
        T = F.relu(T)
        T = self._temporal_conv2(T)
        T = T.permute(0, 2, 1, 3)
        T = self._batch_norm(T)
        T = T.permute(0, 2, 1, 3)
        T = T.reshape(T.size(0), -1)
        T = F.relu(self.fc(T))
        # T = F.relu(self.fc1(T))
        # T = F.relu(self.fc2(T))
        # T = F.relu(self.fc3(T))
        T = self.fc4(T)
        #T = F.relu(T)
        #T = T.reshape(-1, self.out_channels*288*6)
        return T
