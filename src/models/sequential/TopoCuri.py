import logging
import random
from tqdm import tqdm
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler
from torch.nn.utils.rnn import pad_sequence

import dgl
import numpy as np

import matplotlib.pyplot as plt

from models.BaseModel import SequentialModel
from utils import layers, utils
import utils.build_witg as bw

from models.sequential.ContraRec import BERT4RecEncoder, ContraLoss

dataset_name = dict(
    Beauty='Beauty',
    Cell_Phones_and_Accessories='Phone',
    Grocery_and_Gourmet_Food='Grocery',
    Toys_and_Games='Toys'
)

class TopoCuri(SequentialModel):
    reader = 'SeqReader'
    runner = 'BaseRunner'
    extra_log_args = [
        'num_neg', 'batch_size',
        'hard_init',
        'gamma',
        'f_hn', 'n_hn', 'hn0',
        'c_pace', 'max_hard', 'fn_split', 'weight'
    ]

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--gamma', type=float, default=1,
                            help='Coefficient of the contrastive loss.')
        parser.add_argument('--beta_a', type=int, default=3,
                            help='Parameter of the beta distribution for sampling.')
        parser.add_argument('--beta_b', type=int, default=3,
                            help='Parameter of the beta distribution for sampling.')
        parser.add_argument('--ctc_temp', type=float, default=1,
                            help='Temperature in context-target contrastive loss.')
        parser.add_argument('--ccc_temp', type=float, default=0.2,
                            help='Temperature in context-context contrastive loss.')

        parser.add_argument('--gamma_st', type=float, default=0.1,
                            help='Coefficient of the contrastive loss with constructed trainset.')
        parser.add_argument('--min_st_freq', type=int, default=8, help='min st frequency.')

        parser.add_argument('--f_hn', type=str, default='jaccard',
                            help='how to retrieve hard negatives. '
                                 '[rand, dns, nei, jaccard] '
                            )
        parser.add_argument('--n_hn', type=int, default=3, help='number of hard negatives.')
        parser.add_argument('--fn_split', type=int, default=15, help='where to split false negative.')
        parser.add_argument('--hn0', type=int, default=5, help='when to start add hn.')
        parser.add_argument('--hard_init', type=float, default=0.8, help='hardness increment pace.')
        parser.add_argument('--c_pace', type=float, default=0.01, help='hardness increment pace.')
        parser.add_argument('--max_hard', type=float, default=0.7, help='maximum hardenss.')
        parser.add_argument('--weight', type=str, default='bool', help='[bool, weight], whether to adopt weight.')

        return SequentialModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.emb_size = args.emb_size
        self.max_his = args.history_max
        self.gamma = args.gamma
        self.beta_a = args.beta_a
        self.beta_b = args.beta_b
        self.ctc_temp = args.ctc_temp
        self.ccc_temp = args.ccc_temp
        self.mask_token = corpus.n_items

        self.f_hn = args.f_hn
        self.n_hn = args.n_hn
        self.fn_split = args.fn_split
        self.hn0 = args.hn0
        self.weight = args.weight

        self.hardeness = 0.0
        self.c_pace = args.c_pace
        self.max_hard = args.max_hard
        # self.hardeness = args.hard_init
        # self.c_pace = 0.0
        # self.max_hard = args.hard_init

        self.gamma_st = args.gamma_st
        self.min_st_freq = args.min_st_freq
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.pin_memory = args.pin_memory

        self.epoch = 0
        self.loss_last, self.loss_current = None, None

        self._define_params()
        self.apply(self.init_weights)

    def _define_params(self):
        self.i_embeddings = nn.Embedding(self.item_num + 1, self.emb_size)
        self.encoder = BERT4RecEncoder(self.emb_size, self.max_his, num_layers=2, num_heads=2)
        self.ccc_loss = ContraLoss(self.device, temperature=self.ccc_temp)

    def forward(self, feed_dict):
        self.check_list = []
        i_ids = feed_dict['item_id']  # bsz, n_candidate
        history = feed_dict['history_items']  # bsz, history_max
        lengths = feed_dict['lengths']  # bsz

        his_vectors = self.i_embeddings(history)
        his_vector = self.encoder(his_vectors, lengths)
        i_vectors = self.i_embeddings(i_ids)
        prediction = (his_vector[:, None, :] * i_vectors).sum(-1)
        out_dict = {'prediction':prediction}

        if feed_dict['phase'] == 'train':
            history_a = feed_dict['history_items_a']
            his_a_vectors = self.i_embeddings(history_a)
            his_a_vector = self.encoder(his_a_vectors, lengths)
            history_b = feed_dict['history_items_b']
            his_b_vectors = self.i_embeddings(history_b)
            his_b_vector = self.encoder(his_b_vectors, lengths)
            features = torch.stack([his_a_vector, his_b_vector], dim=1)  # bsz, 2, emb
            features = F.normalize(features, dim=-1)
            out_dict['features'] = features  # bsz, 2, emb
            out_dict['labels'] = i_ids[:, 0]  # bsz

            if self.epoch > self.hn0:
                if self.f_hn == 'dns_':
                    ids_fn = feed_dict['hn']
                    ids_fn_neg = feed_dict['fn_neg']
                    i_vec_fn = self.i_embeddings(ids_fn)
                    i_vec_fn_neg = self.i_embeddings(ids_fn_neg)
                    out_dict['pred_fn'] = (his_vector[:, None, :] * i_vec_fn).sum(-1)
                    out_dict['pred_fn_neg'] = (his_vector[:, None, :] * i_vec_fn_neg).sum(-1)
        return out_dict

    def loss(self, out_dict):
        predictions = out_dict['prediction'] / self.ctc_temp
        logits_neg = predictions[:, 1:]
        logits_pos = predictions[:, 0:1].expand_as(logits_neg)
        # bpr
        ctc_loss = -torch.log(10e-8 + torch.sigmoid(logits_pos - logits_neg)).mean()
        # softmax
        # pre_softmax = (predictions - predictions.max()).softmax(dim=1)
        # ctc_loss = - self.ctc_temp * pre_softmax[:, 0].log().mean()

        ccc_loss = self.ccc_loss(out_dict['features'], labels=out_dict['labels'])
        loss = ctc_loss + self.gamma * ccc_loss

        if self.epoch > self.hn0:
            if self.f_hn == 'dns_':
                pred_fn_neg = out_dict['pred_fn_neg'] / self.ctc_temp
                pred_fn = out_dict['pred_fn'].unsqueeze(1).expand_as(pred_fn_neg) / self.ctc_temp
                ctc_loss_fn = -torch.log(10e-8 + torch.sigmoid(pred_fn - pred_fn_neg)).mean()
                loss += ctc_loss_fn
        return loss

    def topk_ct(self, seq_batch, item_candidates, k=None):
        """return [bsz, k] index"""
        k = k if k else self.fn_split
        sim = torch.matmul(seq_batch, item_candidates.transpose(0, 1))
        rst = torch.topk(sim, k, dim=1)[1]
        return rst

    class Dataset(SequentialModel.Dataset):
        def reorder_op(self, seq):
            ratio = np.random.beta(a=self.model.beta_a, b=self.model.beta_b)
            select_len = int(len(seq) * ratio)
            start = np.random.randint(0, len(seq) - select_len + 1)
            idx_range = np.arange(len(seq))
            np.random.shuffle(idx_range[start: start + select_len])
            return seq[idx_range]

        def mask_op(self, seq):
            ratio = np.random.beta(a=self.model.beta_a, b=self.model.beta_b)
            selected_len = int(len(seq) * ratio)
            mask = np.full(len(seq), False)
            mask[:selected_len] = True
            np.random.shuffle(mask)
            seq[mask] = self.model.mask_token
            return seq

        def augment(self, seq):
            aug_seq = np.array(seq).copy()
            if np.random.rand() > 0.5:
                return self.mask_op(aug_seq)
            else:
                return self.reorder_op(aug_seq)

        def _get_feed_dict(self, index):
            feed_dict = super()._get_feed_dict(index)
            if self.phase == 'train':
                history_items_a = self.augment(feed_dict['history_items'])
                history_items_b = self.augment(feed_dict['history_items'])
                feed_dict['history_items_a'] = history_items_a
                feed_dict['history_items_b'] = history_items_b
                feed_dict['index'] = index
                if self.model.epoch > self.model.hn0:
                    feed_dict['item_id'] = np.concatenate(
                        [feed_dict['item_id'], self.data['hn'][index]]
                    )
                    feed_dict['hn'] = self.data['hn'][index]
                    feed_dict['fn_neg'] = self.data['fn_neg'][index]
            return feed_dict

        def prepare(self):
            super().prepare()
            if self.model.buffer and self.phase == 'train':
                self._construct_graph()
                self.model.loss_last = torch.zeros(self.data['user_id'].shape[0]).to(self.model.device)
                self.model.loss_current = torch.zeros(self.data['user_id'].shape[0]).to(self.model.device)
            # self.model.load_model()

        def actions_before_epoch(self):
            super().actions_before_epoch()
            self.i_vectors = self.model.i_embeddings(
                torch.arange(0, self.model.item_num, device=self.model.device)).detach()
            if self.model.epoch >= self.model.hn0:
                self._update_hn_nei()
                if self.model.hardeness < self.model.max_hard:
                    hardeness = self.model.hardeness + self.model.c_pace
                    self.model.hardeness = min(hardeness, self.model.max_hard)
            if self.model.epoch > 0:
                self.model.loss_last = self.model.loss_current.detach()
                self.model.loss_current = torch.zeros(self.data['user_id'].shape[0]).to(self.model.device)
            self.model.epoch += 1

        def _construct_graph(self):
            df = self.corpus.data_df['train']
            user_seqs = df.groupby('user_id')['item_id'].apply(np.array).to_dict()
            self.g = bw.build_WITG_from_trainset(
                user_seqs, use_renorm=True, use_scale=True, user_seq=True)
            edge_index = self.g.edge_index.detach().cpu().numpy()
            self.g_dgl = dgl.graph((edge_index[0], edge_index[1]))
            self.g_dgl.edata['w'] = self.g.edge_attr
            nodes = self.g_dgl.nodes()
            ego_nodes = [dgl.khop_in_subgraph(self.g_dgl, n, k=1) for n in tqdm(nodes)]
            self.nei_nodes = [ego[0].dstdata['_ID'].cpu() for ego in tqdm(ego_nodes)]

            if self.model.weight == 'bool':
                self.g_adj = self.g_dgl.adj().to_dense()
            else:
                self.g_adj = dgl_to_weighted_adj(self.g_dgl, 'w').to_dense()

            g_cn = torch.matmul(self.g_adj, self.g_adj.t())
            g_deg = g_cn.diag().expand_as(g_cn)
            g_union = g_deg + g_deg.t() - g_cn
            self.g_jaccard = (g_cn / g_union).nan_to_num()
            # self.g_jaccard = self.g_jaccard.nan_to_num().to(self.model.device)
            # del g_cn, g_deg, g_union
            # self.g_adj_w = dgl_to_weighted_adj(self.g_dgl, 'w').to_dense()
            print("graph constructed")

        def _update_hn_nei(self):
            targets = self.data['item_id']
            dl = DataLoader(
                targets, batch_size=self.model.batch_size,
                shuffle=False, num_workers=self.model.num_workers,
                pin_memory=self.model.pin_memory)
            hn_lst = []
            dist_lst = []
            jaccard_lst = []
            for target_batch in tqdm(dl, leave=False, ncols=100, mininterval=1):
                neg_batch = getattr(self, f'_gen_hn_{self.model.f_hn}')(target_batch)
                hn_lst.append(neg_batch)
                if self.model.epoch in [0, 20, 50, 100, 150]:
                    dist_lst.append(self._calculate_dist(target_batch).detach().cpu())
                    jaccard_lst.append(self.g_jaccard[target_batch])
            self.data['hn'] = torch.cat(hn_lst).detach().cpu().numpy()
            self.data['fn_neg'] = torch.randint(
                1, self.model.item_num - 1,
                (self.data['hn'].shape[0], self.model.num_neg, self.data['hn'].shape[1])).numpy()

        def _calculate_dist(self, target_batch):
            target_batch = target_batch.to(self.model.device)
            target_vectors = self.i_vectors[target_batch]
            target_vectors = torch.nn.functional.normalize(target_vectors, p=2, dim=-1)
            i_vectors = torch.nn.functional.normalize(self.i_vectors, p=2, dim=-1)
            dist = torch.matmul(target_vectors, i_vectors.transpose(0, 1))
            return dist

        def _gen_hn_jaccard(self, target_batch):
            jaccard_batch = self.g_jaccard[target_batch].to(self.model.device)
            target_batch = target_batch.to(self.model.device)
            mask = self.g_jaccard <= self.model.hardeness
            mask_batch = mask[target_batch].to(self.model.device)
            # jaccard_batch = self.g_jaccard[target_batch]
            jaccard_batch_softmax = jaccard_batch.softmax(dim=1)
            jaccard_batch_softmax = jaccard_batch_softmax * mask_batch
            neg_batch = torch.multinomial(jaccard_batch_softmax, self.model.n_hn, replacement=True)
            return neg_batch

        def _gen_hn_rand(self, target_batch):
            neg_batch = torch.randint(1, self.model.item_num - 1,
                (target_batch.shape[0], self.model.n_hn), device=self.model.device)
            return neg_batch

        def _gen_hn_dns(self, target_batch):
            topk_until = self.model.fn_split + self.model.n_hn
            target_batch = target_batch.to(self.model.device)
            target_vectors = self.i_vectors[target_batch]
            target_vectors = torch.nn.functional.normalize(target_vectors, p=2, dim=-1)
            i_vectors = torch.nn.functional.normalize(self.i_vectors, p=2, dim=-1)
            dist = torch.matmul(target_vectors, i_vectors.transpose(0, 1))
            dist[:, 0] = 0
            hn_topk = torch.topk(dist, topk_until, dim=1)[1]
            select_hn = hn_topk[:, self.model.fn_split:topk_until]
            select_hn = select_hn - 1
            # select_hn = torch.randint(1, self.model.item_num - 1,
            #     (target_batch.shape[0], self.model.n_hn), device=self.model.device)
            return select_hn

        def _gen_hn_nei(self, target_batch):
            nei = [self.nei_nodes[n] for n in target_batch]
            nei_padded = pad_sequence(nei, batch_first=True)
            nei_padded = nei_padded.to(self.model.device)

            n_sample = self.model.fn_split + self.model.n_hn
            k = min([nei_padded.shape[1], n_sample])

            target_batch = target_batch.to(self.model.device)
            target_vectors = self.i_vectors[target_batch]
            target_vectors = torch.nn.functional.normalize(target_vectors, p=2, dim=-1)
            i_vectors = torch.nn.functional.normalize(self.i_vectors, p=2, dim=-1)
            dist = torch.matmul(target_vectors, i_vectors.transpose(0, 1))
            dist[:, 0] = 0

            dist_nei = torch.gather(dist, 1, nei_padded)
            self.model.fn_split = min([k-1, self.model.fn_split])
            hn_topk = torch.topk(dist_nei, k, dim=1, largest=False)[1]
            hn_range = hn_topk[:, self.model.fn_split:k]
            select_hn = torch.gather(nei_padded, 1, hn_range)
            select_hn[select_hn == 0] = torch.randint(
                1, self.model.item_num - 1,
                select_hn[select_hn == 0].shape, device=select_hn.device)
            return select_hn


def collect_list_dict(list_of_dict, comment=''):
    keys = list_of_dict[0].keys()
    rst = dict()
    for key in keys:
        tmp = [d[key] for d in list_of_dict]
        rst[key] = torch.cat(tmp)
        print(f'{comment} {key}: {rst[key].mean():.4f}')
    return rst


def dgl_to_weighted_adj(g_dgl, key_weight='w'):
    weight, indices = (g_dgl.edata[key_weight].squeeze(-1), g_dgl.edges())
    indices = (indices[0].view(-1, 1), indices[1].view(-1, 1))
    indices = torch.cat(indices, dim=1)
    shape = g_dgl.adj().shape
    adj = torch.sparse_coo_tensor(indices.t(), weight, shape)
    return adj


