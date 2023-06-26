# -*- coding: UTF-8 -*-

import os
import sys
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import logging
import argparse
import numpy as np
import pandas as pd
import networkx as nx
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence

import dgl
from dgl.sampling import sample_neighbors

from helpers import *
from models.general import *
from models.sequential import *
from models.developing import *
from utils import utils
import utils.build_witg as bw


def parse_global_args(parser):
    parser.add_argument('--gpu', type=str, default='',
                        help='Set CUDA_VISIBLE_DEVICES, default for CPU only')
    parser.add_argument('--verbose', type=int, default=logging.INFO,
                        help='Logging Level, 0, 10, ..., 50')
    parser.add_argument('--log_file', type=str, default='',
                        help='Logging file path')
    parser.add_argument('--random_seed', type=int, default=0,
                        help='Random seed of numpy and pytorch')
    parser.add_argument('--load', type=int, default=0,
                        help='Whether load model and continue to train')
    parser.add_argument('--train', type=int, default=1,
                        help='To train the model or not.')
    parser.add_argument('--regenerate', type=int, default=0,
                        help='Whether to regenerate intermediate files')
    return parser


def visualize_batch(batch_logits):
    batch_size = batch_logits.shape[0]
    batch_show = batch_logits - torch.eye(batch_size)
    batch_show = batch_show - batch_show.min()
    batch_show = batch_show / (batch_show.max() - batch_show.min())
    plt.imshow(batch_show)
    plt.show()


def flatten(lst):
    return [item for sublist in lst for item in sublist]


def check_ct_distance(g_dgl, G0, dataset, model, k=2):
    dist_target, dist_hard, dist_random = [], [], []
    len_train = len(dataset.buffer_dict)
    n_node = len(G0.nodes)
    # shortest_path_len = [nx.shortest_path_length(G0, i) for i in tqdm(G0.nodes,total=n_node)]
    shortest_path_len = pickle.load(open('grocery_shortest_pl.pkl', 'rb'))
    # n = len(shortest_path_len)
    # pl = torch.zeros((n, n), device=model.device).int()
    # for i in tqdm(range(1, n)):
    #     for j in range(n):
    #         try:
    #             pl[i, j] = shortest_path_len[i][j]
    #         except:
    #             pl[i, j] = 10
    model.eval()
    i_vectors = model.i_embeddings(torch.arange(1, model.item_num, device=model.device))
    dl = DataLoader(
        dataset, batch_size=model.batch_size,
        shuffle=True, num_workers=model.num_workers,
        collate_fn=dataset.collate_batch, pin_memory=model.pin_memory)
    for batch in tqdm(dl, leave=False, ncols=100, mininterval=1):
        batch = utils.batch_to_gpu(batch, model.device)
        out_dict = model(batch)
        target = batch['item_id'][:, 0]
        neg_rand = batch['item_id'][:, 1]
        feat = out_dict['features'][:, 0, :]
        neg_hardset = model.topk_ct(feat, i_vectors, k)
        target_mask = (neg_hardset[:, 0] != target)
        neg_hard = neg_hardset[:, 0] * target_mask + neg_hardset[:, 1] * ~target_mask
        seq = batch['history_items']
        print()


# def check_cc_eig(g_dgl, dataset, model, k=2):
#
#     dist_target, dist_hard, dist_random = [], [], []
#     len_train = len(dataset.buffer_dict)
#     n_node = len(G0.nodes)
#     shortest_path_len = [nx.shortest_path_length(G0, i) for i in tqdm(G0.nodes,total=n_node)]
#     model.eval()
#     i_vectors = model.i_embeddings(torch.arange(1, model.item_num, device=model.device))
#     dl = DataLoader(
#         dataset, batch_size=model.batch_size,
#         shuffle=True, num_workers=model.num_workers,
#         collate_fn=dataset.collate_batch, pin_memory=model.pin_memory)
#     for batch in tqdm(dl, leave=False, ncols=100, mininterval=1):
#         batch = utils.batch_to_gpu(batch, model.device)
#         out_dict = model(batch)
#         target = batch['item_id'][:, 0]
#         neg_rand = batch['item_id'][:, 1]
#         feat = out_dict['features'][:, 0, :]
#         neg_hardset = model.topk_ct(feat, i_vectors, k)
#         target_mask = (neg_hardset[:, 0] != target)
#         neg_hard = neg_hardset[:, 0] * target_mask + neg_hardset[:, 1] * ~target_mask
#
#         gs_1hop = [sample_neighbors(g_dgl, seq, -1) for seq in seq_batch]


# def check_ct_distance(g_dgl, train_seq, train_target, G0, model):
#     dist_target, dist_random, dist_hard = [], [], []
#     len_train = train_seq.shape[0]
#     neg_random = np.random.choice(g_dgl.nodes(), len_train, replace=True)
#     ew = g_dgl.edata['w']
#     ew = 1/ew
#     i_vectors = model.i_embeddings(torch.arange(1, model.item_num, device=model.device))
#
#     for seq, target, neg_random in tqdm(zip(train_seq, train_target, neg_random), total=len_train):
#         his_vectors = model.i_embeddings(seq.to(model.device))
#         seq = [int(i) for i in seq if i!=0]
#         lengths = len(seq)
#         his_vector = model.encoder(his_vectors, lengths)
#         neg_hard = model.topk_ct(his_vector, i_vectors, k=1)
#         try:
#             dist_target.extend([min([nx.shortest_path_length(
#                 # G0, int(target), i, weight=ew, method='bellman-ford') for i in seq])])
#                 G0, int(target), i) for i in seq])])
#         except:
#             dist_target.extend([10])
#         try:
#             dist_hard.extend([min([nx.shortest_path_length(
#                 G0, int(neg_hard), i) for i in seq])])
#         except:
#             dist_hard.extend([10])
#         try:
#             dist_random.extend([min([nx.shortest_path_length(
#                 # G0, int(neg_random), i, weight=ew, method='bellman-ford') for i in seq])])
#                 G0, int(neg_random), i) for i in seq])])
#         except:
#             dist_random.extend([10])
#         pass
#     print(np.array(dist_target).mean())
#     print(np.array(dist_hard).mean())
#     print(np.array(dist_random).mean())


def visual_cc_jaccard(train_dataloader_target, g_dgl, data_dict, k=8):
    data_dict.actions_before_epoch()
    model = data_dict.model
    dl = model.cc_dl
    for batch in dl:
        seq_batch = batch['history_items']
        batch_size = seq_batch.shape[0]
        gs_1hop = [sample_neighbors(g_dgl, seq, -1) for seq in seq_batch]
        adjs_1hop = [g.adj().coalesce().indices() for g in gs_1hop]
        node_unique_1hop = [set(adj.unique().numpy()) for adj in adjs_1hop]

        batch_jaccard = torch.zeros(batch_size, batch_size)
        batch_cn = torch.zeros(batch_size, batch_size)
        for i in tqdm(range(batch_size)):
            for j in range(batch_size):
                batch_cn[i, j], batch_jaccard[i, j] = bw.js(node_unique_1hop[i], node_unique_1hop[j])

        batch = utils.batch_to_gpu(batch, model.device)
        target_batch = batch['item_id'][:, 0]
        seq_vec = model(batch)['features'][:, 0, :]
        cc_relation = torch.matmul(seq_vec, seq_vec.transpose(0, 1))
        mask = torch.eq(target_batch.unsqueeze(1), target_batch.unsqueeze(1).transpose(0, 1))
        cc_relation = cc_relation * ~mask
        topk_hard = torch.topk(cc_relation, k)

        # target_mask = (neg_hardset[:, 0] != target)
        # neg_hard = neg_hardset[:, 0] * target_mask + neg_hardset[:, 1] * ~target_mask

        print()
    for seq_batch, target_batch in train_dataloader_target:
        batch_size = target_batch.shape[0]
        gs_1hop = [sample_neighbors(g_dgl, seq, -1) for seq in seq_batch]
        adjs_1hop = [g.adj().coalesce().indices() for g in gs_1hop]
        node_unique_1hop = [adj.unique() for adj in adjs_1hop]
        batch_jaccard = torch.zeros(batch_size, batch_size)
        batch_cn = torch.zeros(batch_size, batch_size)
        for i in tqdm(range(batch_size)):
            for j in range(batch_size):
                batch_cn[i, j], batch_jaccard[i, j] = bw.js(node_unique_1hop[i], node_unique_1hop[j])
        visualize_batch(batch_cn)
        visualize_batch(batch_jaccard)


def check_cc_iso_count(g, train_dataloader_target):
    lgp_feature = bw.extract_circles_count(g)
    lgp_feat_pad = torch.cat([torch.zeros(1, 3), lgp_feature])
    for seq_batch, target_batch in train_dataloader_target:
        seq_len = (seq_batch != 0).sum(1)
        lgp_feat = lgp_feat_pad[seq_batch - 1]
        lgp_feat = lgp_feat.sum(1)
        lgp_feat = lgp_feat / seq_len.unsqueeze(1).expand_as(lgp_feat)
        sim = lgp_feat.matmul(lgp_feat.t())
        # sim = torch.cosine_similarity(lgp_feat.unsqueeze(1), lgp_feat.unsqueeze(0), dim=-1)
        visualize_batch(sim)


def analyze_topo():
    trainset, model, data_dict = load_train_data()
    # train_uid = [r[1]['user_id'] for r in trainset.items()]
    train_seq = [r[1]['history_items'] for r in trainset.items()]
    train_seq = pad_sequence([torch.from_numpy(x) for x in train_seq], batch_first=True)
    train_target = torch.tensor([r[1]['item_id'][0] for r in trainset.items()])
    ind_sort_target = torch.tensor(np.argsort(train_target.clone().detach()))
    train_seq = train_seq[ind_sort_target]
    train_target = train_target[ind_sort_target]
    trainset_sort_target = TensorDataset(train_seq, train_target)
    train_dataloader_target = DataLoader(
        trainset_sort_target, shuffle=False,
        batch_size=args.batch_size, pin_memory=True, num_workers=8)

    witg = bw.build_WITG_from_trainset(trainset)
    edge_index = witg.edge_index.detach().cpu().numpy()
    g_dgl = dgl.graph((edge_index[0], edge_index[1]))
    g_dgl.edata['w'] = witg.edge_attr
    g_nx = g_dgl.to_networkx()
    # g_nx_undi = nx.to_undirected(g_nx)
    # Gcc = sorted(nx.connected_components(g_nx_undi), key=len, reverse=True)
    # G0 = g_nx_undi.subgraph(Gcc[0])

    # check_ct_distance(g_dgl, train_seq, train_target, g_nx, model)

    # check_ct_distance(g_dgl, g_nx, data_dict, model)
    visual_cc_jaccard(train_dataloader_target, g_dgl, data_dict)
    check_cc_iso_count(witg, train_dataloader_target)
    return


def load_train_data():
    logging.info('-' * 45 + ' BEGIN: ' + utils.get_time() + ' ' + '-' * 45)
    exclude = ['check_epoch', 'log_file', 'model_path', 'path', 'pin_memory', 'load',
               'regenerate', 'sep', 'train', 'verbose', 'metric', 'test_epoch', 'buffer']
    logging.info(utils.format_arg_str(args, exclude_lst=exclude))

    # Random seed
    utils.init_seed(args.random_seed)

    # GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device('cpu')
    if args.gpu != '' and torch.cuda.is_available():
        args.device = torch.device('cuda')
    logging.info('Device: {}'.format(args.device))

    # Read data
    corpus_path = os.path.join(args.path, args.dataset, model_name.reader + '.pkl')
    logging.info('Load corpus from {}'.format(corpus_path))
    corpus = pickle.load(open(corpus_path, 'rb'))

    # Define model
    model = model_name(args, corpus).to(args.device)
    model.load_model()
    # model_path = "../model/STRec/STRec__Grocery_and_Gourmet_Food__0__lr=0.0001__l2=1e-06__num_neg=1__batch_size=4096__gamma=1.pt"
    # model.load_model(model_path)
    logging.info('#params: {}'.format(model.count_variables()))
    logging.info(model)

    # Run model
    data_dict = dict()
    for phase in ['train', 'dev', 'test']:
        data_dict[phase] = model_name.Dataset(model, corpus, phase)
        data_dict[phase].prepare()
    data_dict['train'].actions_before_epoch()
    data_dict['train'].phase = 'test'
    data_dict['train'].prepare()
    data_dict['train'].phase = 'train'

    train_set = data_dict['train'].buffer_dict
    return train_set, model, data_dict['train']


if __name__ == '__main__':
    init_parser = argparse.ArgumentParser(description='Model')
    init_parser.add_argument('--model_name', type=str, default='STRec', help='Choose a model to run.')
    init_args, init_extras = init_parser.parse_known_args()
    model_name = eval('{0}.{0}'.format(init_args.model_name))
    reader_name = eval('{0}.{0}'.format(model_name.reader))  # model chooses the reader
    runner_name = eval('{0}.{0}'.format(model_name.runner))  # model chooses the runner

    # Args
    parser = argparse.ArgumentParser(description='')
    parser = parse_global_args(parser)
    parser = reader_name.parse_data_args(parser)
    parser = runner_name.parse_runner_args(parser)
    parser = model_name.parse_model_args(parser)
    args, extras = parser.parse_known_args()

    # Logging configuration
    log_args = [init_args.model_name, args.dataset, str(args.random_seed)]
    for arg in ['lr', 'l2'] + model_name.extra_log_args:
        log_args.append(arg + '=' + str(eval('args.' + arg)))
    log_file_name = '__'.join(log_args).replace(' ', '__')
    if args.log_file == '':
        args.log_file = '../log/{}/{}.txt'.format(init_args.model_name, log_file_name)
    if args.model_path == '':
        args.model_path = '../model/{}/{}.pt'.format(init_args.model_name, log_file_name)

    utils.check_dir(args.log_file)
    logging.basicConfig(filename=args.log_file, level=args.verbose)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(init_args)

    analyze_topo()
