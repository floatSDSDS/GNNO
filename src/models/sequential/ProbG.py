import copy
# import pickle
import random
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset

from torch_geometric.data import Data

import dgl
from dgl.sampling import sample_neighbors
import netrd.distance as nd

import networkx as nx

from utils import layers, utils
import utils.build_witg as bw


def flatten(lst):
    return [item for sublist in lst for item in sublist]


def dgl_to_weighted_adj(g_dgl, key_weight='w'):
    weight, indices = (g_dgl.edata[key_weight].squeeze(-1), g_dgl.edges())
    indices = (indices[0].view(-1, 1), indices[1].view(-1, 1))
    indices = torch.cat(indices, dim=1)
    shape = g_dgl.adj().shape
    adj = torch.sparse_coo_tensor(indices.t(), weight, shape)
    return adj


def collect_list_dict(list_of_dict, comment=''):
    keys = list_of_dict[0].keys()
    rst = dict()
    for key in keys:
        tmp = [d[key] for d in list_of_dict]
        rst[key] = torch.cat(tmp)
        print(f'{comment} {key}: {rst[key].mean():.4f}')
    return rst


""" Topo Analysis """
class ProbGraph(nn.Module):
    def __init__(self, args, model, data_dict):
        """
        construct a graph in recommendation for topology analysis
        """
        super().__init__()
        self.args = args
        self.key = args.key_g
        self.min_seq_len = args.min_seq_len
        self.short_len = args.short_len
        self.k0 = args.k0
        self.kt = args.kt

        self.model = model
        self.device = self.model.device
        self.data_dict = data_dict
        self.corpus = data_dict['train'].corpus

        self.trainset = None
        self.user_seqs, self.seq_lst_sort = None, None
        self.short_seqs = None
        self.seq_sort, self.target_sort = None, None

        self.hard_neg_dict = dict()
        self.hard_neg_vec_dict = dict()
        self.target_seq_dict = dict()
        self.dl = None

        self.n_users = self.corpus.n_users
        self.n_items = self.corpus.n_items

        self.graph = None
        self.g_dgl = None
        self.g_nx = None
        self.ego_nodes, self.nei_nodes, self.seq_g = None, None, None
        self.seq_g_nx, self.ego_nodes_nx = None, None
        self.g_adj, self.g_adj_w = None, None

        self.ind_sort = None
        self.i_vectors = None
        self.data_dict_sort = None

        self._pre_cook_dataset()
        self.construct_graph()
        self._post_cook_dataset()

        # analysis
        self.nd_dist = dict()
        self.analysis_rst = dict()
        self.gplus_batch = dict()
        self.analyze_tt()
        self.analyze_ct()
        print()

    def construct_graph(self, key=None):
        key_g = key if key else self.key
        if key_g == 'witg':
            self._construct_witg()
        elif key_g == 'ui':
            self._construct_ui()
        self.g_dgl = self.g_dgl.to(self.device)

    def _pre_cook_dataset(self):
        self.data_dict['train'].actions_before_epoch()
        self.data_dict['train'].phase = 'test'
        self.data_dict['train'].prepare()
        self.data_dict['train'].phase = 'train'
        self.trainset = self.data_dict['train'].buffer_dict
        self._sort_train()

        # user seqs
        df_train = self.corpus.data_df['train']
        df_train = df_train.sort_values(['user_id', 'time'])
        self.user_seqs = df_train.groupby('user_id')['item_id'].apply(np.array).to_dict()
        self._sliding_short_seq(len=self.short_len)

        # target seq dict
        self.target_seq_dict = {}
        target_sort = self.target_sort.numpy()
        self.target_seq_dict = {i: [] for i in self.target_sort.unique().numpy()}
        for i, seq in enumerate(self.seq_lst_sort):
            self.target_seq_dict[target_sort[i]].append(seq)

        # hard neg with emb
        self.i_vectors = self.model.i_embeddings(
            torch.arange(1, self.model.item_num, device=self.device))

    def _post_cook_dataset(self):
        print('calculate neighbors')
        nodes = self.g_dgl.nodes()

        self.g_adj_w = dgl_to_weighted_adj(self.g_dgl, 'w').to_dense()
        self.g_adj = self.g_dgl.adj().to_dense().to(self.device)

        self.ego_nodes = [dgl.khop_in_subgraph(self.g_dgl, n, k=1) for n in tqdm(nodes)]
        # self.ego_nodes_nx = [nx.from_numpy_matrix(
        #     g[0].adj().to_dense().numpy()) for g in tqdm(self.ego_nodes)]
        self.nei_nodes = [ego[0].dstdata['_ID'].cpu() for ego in tqdm(self.ego_nodes)]

        self._retrieve_hard_neg()

        print("pre calculate subgraphs")
        # self.seq_g = [dgl.khop_in_subgraph(self.g_dgl, seq, k=0)[0] for seq in tqdm(self.seq_lst_sort)]
        # self.seq_g_nx = [g.cpu().to_networkx() for g in tqdm(self.seq_g)]
        # self.seq_g_nx = [nx.from_numpy_matrix(
        #     g.adj().to_dense().numpy()) for g in tqdm(self.seq_g)]

    def _retrieve_hard_neg(self, key=None):
        """key: [item, seq, topo]"""
        if key:
            getattr(self, f'_retrieve_hard_{key}')
        else:
            self._retrieve_hardneg_item(key='item_fn', k0=1, kt=3)
            self._retrieve_hardneg_item(key='item', k0=self.k0, kt=self.kt)
            self._retrieve_hardneg_seq(key='seq_fn', k0=1, kt=3)
            self._retrieve_hardneg_seq(key='seq', k0=self.k0, kt=self.kt)
            self._retrieve_hardneg_nei()
            self._retrieve_hardneg_nei_w()
        for c, v in self.hard_neg_dict.items():
            self.data_dict['train'].data[f'hn_{c}'] = v
        self.dl = DataLoader(
            self.data_dict['train'], batch_size=self.model.batch_size,
            shuffle=False, num_workers=self.model.num_workers,
            collate_fn=self.data_dict['train'].collate_batch,
            pin_memory=self.model.pin_memory)

    def _retrieve_hardneg_item(self, key='item', k0=1, kt=3):
        dl = DataLoader(
            self.target_sort, batch_size=self.model.batch_size,
            shuffle=False, num_workers=self.model.num_workers,
            pin_memory=self.model.pin_memory)
        hard_neg_lst = []
        for target_batch in tqdm(dl, leave=False, ncols=100, mininterval=1):
            target_batch = target_batch.to(self.device)
            target_vectors = self.model.i_embeddings(target_batch)
            hard_neg_batch = self.model.topk_ct(target_vectors, self.i_vectors, k0, kt)
            hard_neg_batch = hard_neg_batch + 1
            hard_neg_lst.append(hard_neg_batch)
        self.hard_neg_dict[key] = torch.cat(hard_neg_lst).cpu().numpy()

    def _retrieve_hardneg_seq(self, key='seq', k0=1, kt=3):
        hard_neg_lst = []
        for batch in tqdm(self.dl, leave=False, ncols=100, mininterval=1):
            batch = utils.batch_to_gpu(batch, self.device)
            out_dict = self.model(batch)
            feat = out_dict['features'][:, 0, :]
            hard_neg_batch = self.model.topk_ct(feat, self.i_vectors, k0, kt)
            hard_neg_batch = hard_neg_batch + 1
            hard_neg_lst.append(hard_neg_batch)
        self.hard_neg_dict[key] = torch.cat(hard_neg_lst).cpu().numpy()

    def _retrieve_hardneg_nei(self):
        n_neg = self.kt - self.k0 - 1
        nei = [np.random.choice(self.nei_nodes[n], n_neg, replace=True) for n in self.target_sort]
        self.hard_neg_dict['nei'] = np.stack(nei)

    def _retrieve_hardneg_nei_w(self):
        print("sample neighbors with weight")
        n_neg = self.kt - self.k0 - 1
        dl = DataLoader(
            self.target_sort, batch_size=self.model.batch_size,
            shuffle=False, num_workers=self.model.num_workers,
            pin_memory=self.model.pin_memory)
        hard_neg_lst = []
        for target_batch in tqdm(dl, leave=False, ncols=100, mininterval=1):
            target_batch = target_batch.to(self.device)
            weight_batch = self.g_adj_w[target_batch]
            hard_neg_batch = torch.multinomial(weight_batch, n_neg, replacement=True)
            hard_neg_lst.append(hard_neg_batch)
        # nei = [sample_neighbors(
        #     self.g_dgl.cpu(), n, n_neg, prob='w', replace=True) for n in tqdm(self.target_sort)]
        self.hard_neg_dict['nei_w'] = torch.cat(hard_neg_lst).cpu().numpy()

    def _sliding_short_seq(self, len=3):
        # for u, seq in tqdm(self.user_seqs.items()):
        #     print()
        return

    def _sort_train(self, min_len=None):
        min_len = min_len if min_len else self.min_seq_len

        train_seq_lst = [r[1]['history_items'] for r in self.trainset.items() if r[1]['lengths']]
        train_seq = pad_sequence([torch.from_numpy(x) for x in train_seq_lst], batch_first=True)
        train_target = torch.tensor([r[1]['item_id'][0] for r in self.trainset.items() if r[1]['lengths']])
        train_len = [r[1]['lengths'] for r in self.trainset.items() if r[1]['lengths']]

        ind_sort_target = np.argsort(train_target.clone().detach()).numpy()
        ind_keep = [v for i, v in enumerate(ind_sort_target) if train_len[i] >= min_len]

        train_seq_sort_lst = [train_seq_lst[i] for i in ind_keep]
        train_seq = train_seq[ind_keep]
        train_target = train_target[ind_keep]

        self.ind_sort = ind_keep
        self.seq_lst_sort = train_seq_sort_lst
        self.seq_sort = train_seq
        self.target_sort = train_target

        # update data_dict['train'] and create
        for key in self.data_dict['train'].data.keys():
            self.data_dict['train'].data[key] = self.data_dict['train'].data[key][self.ind_sort]
        self.dl = DataLoader(
            self.data_dict['train'], batch_size=self.model.batch_size,
            shuffle=False, num_workers=self.model.num_workers,
            collate_fn=self.data_dict['train'].collate_batch, pin_memory=self.model.pin_memory)

    def _construct_ui(self):
        # x: torch.int64[n_node, 1], edge_index: torch.int64[2, n_edge]
        df_train = self.corpus.data_df['train']
        # x = torch.arange(0, num_node).long().view(-1, 1)  # torch.int64[n_node, 1], item entity index
        # Graph_data = Data(x, edge_index, edge_attr=edge_attr)
        return

    def _construct_witg(self, use_renorm=True, use_scale=False):
        witg = bw.build_WITG_from_trainset(
            self.user_seqs, use_renorm=use_renorm, use_scale=use_scale, user_seq=True)
        edge_index = witg.edge_index.detach().cpu().numpy()
        g_dgl = dgl.graph((edge_index[0], edge_index[1]))
        g_dgl.edata['w'] = witg.edge_attr
        g_nx = g_dgl.to_networkx()

        self.graph = witg
        self.g_dgl = g_dgl
        self.g_nx = g_nx

    def analyze_tt(self, key=''):
        rst_cn = []
        rst_jaccard = []
        for batch in tqdm(self.dl, leave=False, ncols=100, mininterval=1):
            batch = utils.batch_to_gpu(batch, self.model.device)
            cn, jaccard = self._analyze_overlap_tt(batch)
            rst_cn.append(cn)
            rst_jaccard.append(jaccard)
        self.analysis_rst['rst_cn'] = collect_list_dict(rst_cn, 'cn')
        self.analysis_rst['rst_jaccard'] = collect_list_dict(rst_jaccard, 'jaccard')

    def _analyze_overlap_tt(self, batch):
        """"""
        target = batch['item_id'][:, 0].view(-1, 1)
        neg_rand = batch['item_id'][:, 1:]
        hn_item, hn_seq, = batch['hn_item'], batch['hn_seq']
        fn_item, fn_seq, = batch['hn_item_fn'], batch['hn_seq_fn']
        hn_nei, hn_nei_w = batch['hn_nei'], batch['hn_nei_w']

        target_nei = self.g_adj[target].float()
        rand_nei = self.g_adj[neg_rand].float()
        fn_item_nei = self.g_adj[fn_item].float()
        hn_item_nei = self.g_adj[hn_item].float()
        fn_seq_nei = self.g_adj[fn_seq].float()
        hn_seq_nei = self.g_adj[hn_seq].float()
        hn_nei_nei = self.g_adj[hn_nei].float()
        hn_nei_nei_w = self.g_adj[hn_nei_w].float()

        cn_target_rand = (target_nei.expand_as(rand_nei) * rand_nei).sum(dim=-1)
        cn_target_fn_item = (target_nei.expand_as(fn_item_nei) * fn_item_nei).sum(dim=-1)
        cn_target_hn_item = (target_nei.expand_as(hn_item_nei) * hn_item_nei).sum(dim=-1)
        cn_target_fn_seq = (target_nei.expand_as(fn_seq_nei) * fn_seq_nei).sum(dim=-1)
        cn_target_hn_seq = (target_nei.expand_as(hn_seq_nei) * hn_seq_nei).sum(dim=-1)
        cn_target_hn_nei = (target_nei.expand_as(hn_nei_nei) * hn_nei_nei).sum(dim=-1)
        cn_target_hn_nei_w = (target_nei.expand_as(hn_nei_nei_w) * hn_nei_nei_w).sum(dim=-1)

        union_target_rand = ((target_nei.expand_as(rand_nei) + rand_nei) != 0).sum(dim=-1)
        union_target_fn_item = ((target_nei.expand_as(fn_item_nei) + fn_item_nei) != 0).sum(dim=-1)
        union_target_hn_item = ((target_nei.expand_as(hn_item_nei) + hn_item_nei) != 0).sum(dim=-1)
        union_target_fn_seq = ((target_nei.expand_as(fn_seq_nei) + fn_seq_nei) != 0).sum(dim=-1)
        union_target_hn_seq = ((target_nei.expand_as(hn_seq_nei) + hn_seq_nei) != 0).sum(dim=-1)
        union_target_hn_nei = ((target_nei.expand_as(hn_nei_nei) + hn_nei_nei) != 0).sum(dim=-1)
        union_target_hn_nei_w = ((target_nei.expand_as(hn_nei_nei_w) + hn_nei_nei_w) != 0).sum(dim=-1)

        jaccard_target_rand = cn_target_rand / union_target_rand
        jaccard_target_fn_item = cn_target_fn_item / union_target_fn_item
        jaccard_target_hn_item = cn_target_hn_item / union_target_hn_item
        jaccard_target_fn_seq = cn_target_fn_seq / union_target_fn_seq
        jaccard_target_hn_seq = cn_target_hn_seq / union_target_hn_seq
        jaccard_target_hn_nei = cn_target_hn_nei / union_target_hn_nei
        jaccard_target_hn_nei_w = cn_target_hn_nei_w / union_target_hn_nei_w

        rst_cn = dict()
        rst_jaccard = dict()

        rst_cn['rand'] = cn_target_rand.mean(dim=1)
        rst_cn['fn_item'] = cn_target_fn_item.mean(dim=1)
        rst_cn['hn_item'] = cn_target_hn_item.mean(dim=1)
        rst_cn['fn_seq'] = cn_target_fn_seq.mean(dim=1)
        rst_cn['hn_seq'] = cn_target_hn_seq.mean(dim=1)
        rst_cn['hn_nei'] = cn_target_hn_nei.mean(dim=1)
        rst_cn['hn_nei_w'] = cn_target_hn_nei_w.mean(dim=1)

        rst_jaccard['rand'] = jaccard_target_rand.mean(dim=1)
        rst_jaccard['fn_item'] = jaccard_target_fn_item.mean(dim=1)
        rst_jaccard['hn_item'] = jaccard_target_hn_item.mean(dim=1)
        rst_jaccard['fn_seq'] = jaccard_target_fn_seq.mean(dim=1)
        rst_jaccard['hn_seq'] = jaccard_target_hn_seq.mean(dim=1)
        rst_jaccard['hn_nei'] = jaccard_target_hn_nei.mean(dim=1)
        rst_jaccard['hn_nei_w'] = jaccard_target_hn_nei_w.mean(dim=1)
        return rst_cn, rst_jaccard

    def _seq_subg_2nx(self, seq):
        """(source node, target node)"""
        g_dgl = dgl.khop_in_subgraph(self.g_dgl, seq, k=0)[0]
        edges = g_dgl.edges()
        np_edges = np.concatenate([e.view(-1, 1).cpu() for e in edges], axis=1)
        g_nx = nx.from_edgelist(np_edges)
        # adj = g_dgl.adj()
        # g_nx = nx.from_numpy_matrix(adj.to_dense().numpy())
        return g_nx

    def analyze_ct(self, key=''):

        dl = DataLoader(
            self.data_dict['train'], batch_size=self.model.batch_size,
            shuffle=True, num_workers=self.model.num_workers,
            collate_fn=self.data_dict['train'].collate_batch, pin_memory=self.model.pin_memory)

        for batch in tqdm(dl, leave=False, ncols=100, mininterval=1):
            batch = utils.batch_to_gpu(batch, self.device)
            seq_batch = batch['history_items']
            target = batch['item_id'][:, 0:1]
            neg_rand = batch['item_id'][:, 1:]
            hn_item, fn_item = batch['hn_item'], batch['hn_item_fn']
            hn_seq, fn_seq = batch['hn_seq'], batch['hn_seq_fn']

            seq_target = torch.cat([target, seq_batch], dim=1)
            seq_rand = torch.cat([neg_rand, seq_batch], dim=1)
            seq_fn_item = torch.cat([fn_item, seq_batch], dim=1)
            seq_hn_item = torch.cat([hn_item, seq_batch], dim=1)
            seq_fn_seq = torch.cat([fn_seq, seq_batch], dim=1)
            seq_hn_seq = torch.cat([hn_seq, seq_batch], dim=1)

            subg_seq = [self._seq_subg_2nx(seq) for seq in seq_batch]
            subg_target = [self._seq_subg_2nx(seq) for seq in seq_target]
            subg_rand = [self._seq_subg_2nx(seq) for seq in seq_rand]
            subg_fn_item = [self._seq_subg_2nx(seq) for seq in seq_fn_item]
            subg_hn_item = [self._seq_subg_2nx(seq) for seq in seq_hn_item]
            subg_fn_seq = [self._seq_subg_2nx(seq) for seq in seq_hn_seq]
            subg_hn_seq = [self._seq_subg_2nx(seq) for seq in seq_fn_seq]

            # key = 'LaplacianSpectral'
            # k = 3
            # print('LaplacianSpectral3')
            # gplus_batch['LaplacianSpectral3']['target'].append(self._analyze_freq_ct(subg_seq, subg_target, key, k))
            # gplus_batch['LaplacianSpectral3']['fn_item'].append(self._analyze_freq_ct(subg_seq, subg_fn_item, key, k))
            # gplus_batch['LaplacianSpectral3']['hn_item'].append(self._analyze_freq_ct(subg_seq, subg_hn_item, key, k))
            # gplus_batch['LaplacianSpectral3']['fn_seq'].append(self._analyze_freq_ct(subg_seq, subg_fn_seq, key, k))
            # gplus_batch['LaplacianSpectral3']['hn_seq'].append(self._analyze_freq_ct(subg_seq, subg_hn_seq, key, k))
            # # gplus_batch[f'{key}{k}']['neg_rand'].append(self._analyze_freq_ct(subg_seq, subg_rand, key, k))
            #
            # key = 'LaplacianSpectral'
            # print(key)
            # gplus_batch[key]['target'].append(self._analyze_freq_ct(subg_seq, subg_target, key))
            # gplus_batch[key]['fn_item'].append(self._analyze_freq_ct(subg_seq, subg_fn_item, key))
            # gplus_batch[key]['hn_item'].append(self._analyze_freq_ct(subg_seq, subg_hn_item, key))
            # gplus_batch[key]['fn_seq'].append(self._analyze_freq_ct(subg_seq, subg_fn_seq, key))
            # gplus_batch[key]['hn_seq'].append(self._analyze_freq_ct(subg_seq, subg_hn_seq, key))
            # # gplus_batch[key]['neg_rand'].append(self._analyze_freq_ct(subg_seq, subg_rand, key))

            # key = 'IpsenMikhailov'
            print(key)
            # gplus_batch[key]['target'].append(self._analyze_freq_ct(subg_seq, subg_target, key))
            # gplus_batch[key]['neg_rand'].append(self._analyze_freq_ct(subg_seq, subg_rand, key))
            # gplus_batch[key]['hn_item'].append(self._analyze_freq_ct(subg_seq, subg_hn_item, key))
            # gplus_batch[key]['hn_seq'].append(self._analyze_freq_ct(subg_seq, subg_hn_seq, key))

    def _init_dist_obj(self, key_selected=None):

        distance_measure = [
            "CommunicabilityJSD", "DegreeDivergence",  "DeltaCon", "DistributionalNBD",
            "dkSeries", "DMeasure", "Frobenius", "GraphDiffusion", "Hamming",
            "HammingIpsenMikhailov", "IpsenMikhailov", "JaccardDistance",
            "LaplacianSpectral", "NonBacktrackingSpectral", 'NetLSD', "NetSimile", "OnionDivergence",
            "PolynomialDissimilarity", "PortraitDivergence", "QuantumJSD", "ResistancePerturbation"
        ]
        keys = key_selected if key_selected else distance_measure
        for key in keys:
            self.nd_dist[key] = getattr(nd, key)

        self.gplus_batch['LaplacianSpectral'] = dict()
        self.gplus_batch['LaplacianSpectral3'] = dict()
        self.gplus_batch['IpsenMikhailov'] = dict()
        for key in ['target', 'neg_rand', 'fn_item', 'hn_item', 'fn_seq', 'hn_seq']:
            self.gplus_batch['LaplacianSpectral'][key] = []
            self.gplus_batch['LaplacianSpectral3'][key] = []
            self.gplus_batch['IpsenMikhailov'][key] = []

        # self.nd_dist['DistributionalNBD'] = nd.DistributionalNBD()
        # self.nd_dist['LaplacianSpectral'] = nd.LaplacianSpectral()
        # self.nd_dist['IpsenMikhailov'] = nd.IpsenMikhailov()

    def _analyze_freq_ct(self, subg_batch, gplus_batch, key='LaplacianSpectral', k=None):
        dist = self.nd_dist[key]
        n = len(subg_batch)
        rst = np.zeros(n)
        for i in tqdm(range(n)):
            try:
                rst[i] = dist(subg_batch[i], gplus_batch[i], k)
            except:
                rst[i] = -1
        return rst
