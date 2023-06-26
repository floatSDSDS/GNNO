# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
# import networkx as nx
from scipy.sparse import eye
import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree

from torch_geometric.utils import to_dense_adj


"""
u1\ti1,i2,i3\n
u2\ti1,i3\n
"""

def build_WITG_from_trainset(dataset, use_renorm=True, use_scale=False, user_seq=False):
    seqs = []
    item_set = set()
    for record in dataset.items():
        if user_seq:
            items = record[1]
        else:
            items = record[1]['item_id']
        seqs.append(items)
        item_set = item_set | set(items)
    max_item = max(item_set)
    num_node = max_item + 1

    relation = []
    adj = [dict() for _ in range(num_node)]

    for i in range(len(seqs)):
        data = seqs[i]
        for k in range(1, 4):
            for j in range(len(data) - k):
                relation.append([data[j], data[j + k], k])
                relation.append([data[j + k], data[j], k])
    for temp in relation:
        if temp[1] in adj[temp[0]].keys():
            adj[temp[0]][temp[1]] += 1 / temp[2]
        else:
            adj[temp[0]][temp[1]] = 1 / temp[2]

    adj_pyg = []
    weight_pyg = []

    for t in range(1, num_node):
        x = [v for v in sorted(adj[t].items(), reverse=True, key=lambda x: x[1])]
        adj_pyg += [[t, v[0]] for v in x]
        if use_scale:
            t_sum = 0
            for v in x:
                t_sum += v[1]
            weight_pyg += [v[1] / t_sum for v in x]
        else:
            weight_pyg += [v[1] for v in x]

    adj_np = np.array(adj_pyg)
    adj_np = adj_np.transpose()
    edge_np = np.array([adj_np[0, :], adj_np[1, :]])
    x = torch.arange(0, num_node).long().view(-1, 1)  # torch.int64[n_node, 1], item entity index
    edge_attr = torch.from_numpy(np.array(weight_pyg)).view(-1, 1)  # torch.float64[n_edge, 1]
    edge_index = torch.from_numpy(edge_np).long()  # torch.int64[2, n_edge]
    Graph_data = Data(x, edge_index, edge_attr=edge_attr)
    print(Graph_data)
    if use_renorm:
        row, col = Graph_data.edge_index[0], Graph_data.edge_index[1]
        row_deg = 1. / degree(row, num_node, Graph_data.edge_attr.dtype)
        col_deg = 1. / degree(col, num_node, Graph_data.edge_attr.dtype)
        deg = row_deg[row] + col_deg[col]
        new_att = edge_attr * deg.view(-1, 1)
        Graph_data.edge_attr = new_att

    # torch.save(Graph_data, datapath + 'witg.pt')
    return Graph_data


def js(la1, la2):
    combined = la1 & la2
    union = la1 | la2
    return len(combined), len(combined) / len(union)

# def js(la1, la2):
#     # combined = torch.cat((la1, la2))
#     # union, counts = combined.unique(return_counts=True)
#     # intersection = union[counts > 1]
#     # return intersection.shape[0], torch.numel(intersection) / torch.numel(union)


def extract_circles_count(graph, sizes=tuple([3, 4, 5]), normalize=True):
    """
    :param graph:
    :param sizes: tuple(count size), size in [3, 5]

    :return: tensor[n_node, |size|]
    """
    adj = to_dense_adj(graph.edge_index).squeeze(0)
    count_lst = [extract_circle_count(adj, size) for size in sizes]
    counts = torch.cat(count_lst, dim=1)
    # normalization
    if normalize:
        norm = torch.nn.BatchNorm1d(len(sizes)).to(adj.device)
        counts = norm(counts.float())
    return counts


def extract_circle_count(adj, size):
    """
    :param graph: networkx graph
    :param size: int larger than 2
    :return: torch.tensor[n_v]
    """
    print(f'extract C{size}')
    # adj = nx.adjacency_matrix(graph)
    # adj = adj + eye(adj.shape[0])
    adj = adj + torch.eye(adj.shape[0], device=adj.device)
    katz_adj = [adj]
    for i in range(size-1):
        katz_adj.append(katz_adj[-1].matmul(adj))
    ring_last = katz_adj[-1] - katz_adj[-2]
    c_size = ring_last.diagonal()
    return torch.tensor(c_size).view(-1, 1)


if __name__ == '__main__':
    # build_WITG_from_trainset(datapath='home/')
    # build_WITG_from_trainset(datapath='goodreads/poetry/')
    build_WITG_from_trainset(datapath='/data2/fanlu/GCL4SR-IJCAI22/datasets/goodreads/comics_graphic/')
