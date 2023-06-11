# -*- coding: utf-8 -*-
import torch
import numpy as np
from torch import nn
from torch_scatter import scatter_add, scatter_mean, scatter_max
import torch_geometric.nn as gnn
import torch_geometric.utils as utils
from einops import rearrange
from .utils import pad_batch, unpad_batch
from .gnn_layers import get_simple_gnn_layer, EDGE_GNN_TYPES
import torch.nn.functional as F




class StructureExtractor(nn.Module):#我就假如它主要是进行三次gcn操作
    r""" K-subtree structure extractor. Computes the structure-aware node embeddings using the
    k-hop subtree centered around each node.

    Args:
    ----------
    embed_dim (int):        the embeding dimension
    gnn_type (str):         GNN type to use in structure extractor. (gcn, gin, pna, etc)
    num_layers (int):       number of GNN layers
    batch_norm (bool):      apply batch normalization or not
    concat (bool):          whether to concatenate the initial edge features
    khopgnn (bool):         whether to use the subgraph instead of subtree
    """

    def __init__(self, embed_dim, gnn_type="gcn", num_layers=2,#这个要走
                 batch_norm=True, concat=True, khopgnn=False, **kwargs):#128,gcn,none
        super().__init__()
        self.num_layers = num_layers#2
        self.khopgnn = khopgnn#false
        self.concat = concat#true
        self.gnn_type = gnn_type#gcn
        layers = []
        for _ in range(num_layers):
            layers.append(get_simple_gnn_layer(gnn_type, embed_dim, **kwargs))#gcn,128,none
        self.gcn = nn.ModuleList(layers)

        self.relu = nn.ReLU()
        self.batch_norm = batch_norm
        inner_dim = (num_layers + 1) * embed_dim if concat else embed_dim
#512
        if batch_norm:
            self.bn = nn.BatchNorm1d(inner_dim)

        self.out_proj = nn.Linear(inner_dim, embed_dim)#512,128

    def forward(self, x, edge_index, edge_attr=None,
            subgraph_indicator_index=None, agg="sum"):#（260，128），（2，586），none


        x_cat = [x]
        for gcn_layer in self.gcn:
            # if self.gnn_type == "attn":
            #     x = gcn_layer(x, edge_index, None, edge_attr=edge_attr)
            if self.gnn_type in EDGE_GNN_TYPES:#走这个
                if edge_attr is None:#走这个
                    x = self.relu(gcn_layer(x, edge_index))#（260，128），（2，586 ）
                else:
                    x = self.relu(gcn_layer(x, edge_index, edge_attr=edge_attr))
            else:
                x = self.relu(gcn_layer(x, edge_index))

            if self.concat:
                x_cat.append(x)

        if self.concat:
            x = torch.cat(x_cat, dim=-1)

        if self.khopgnn:#false不走这个
            if agg == "sum":
                x = scatter_add(x, subgraph_indicator_index, dim=0)
            elif agg == "mean":
                x = scatter_mean(x, subgraph_indicator_index, dim=0)
            return x

        if self.num_layers > 0 and self.batch_norm:
            x = self.bn(x)

        x = self.out_proj(x)#linnear操作
        return x#260,128


