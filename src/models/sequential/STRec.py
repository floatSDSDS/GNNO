import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler

import dgl
import numpy as np

import matplotlib.pyplot as plt

from models.BaseModel import SequentialModel
from utils import layers, utils

from models.sequential.ContraRec import BERT4RecEncoder, ContraLoss


class STRec(SequentialModel):
    reader = 'SeqReader'
    runner = 'BaseRunner'
    extra_log_args = [
        'num_neg', 'batch_size',
        'gamma',
        # 'gamma_st', 'min_st_freq',
        # 'ctc_temp', 'ccc_temp',
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
        parser.add_argument('--min_st_freq', type=int, default=8,
                            help='min st frequency.')

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

        self.cc_dl = None
        self.cc_dataset = None
        self.gamma_st = args.gamma_st
        self.min_st_freq = args.min_st_freq
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.pin_memory = args.pin_memory

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
        out_dict = {'prediction': prediction}

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

        return out_dict

    def loss(self, out_dict):
        predictions = out_dict['prediction'] / self.ctc_temp
        pre_softmax = (predictions - predictions.max()).softmax(dim=1)
        ctc_loss = - self.ctc_temp * pre_softmax[:, 0].log().mean()
        ccc_loss = self.ccc_loss(out_dict['features'], labels=out_dict['labels'])

        # cc_dict_batch = self.cc_dl.next()
        # cc_dict_batch = utils.batch_to_gpu(cc_dict_batch, self.device)
        # cc_out = self(cc_dict_batch)
        # ccc_st_loss = self.ccc_loss(cc_out['features'], labels=cc_out['labels'])

        loss = ctc_loss + self.gamma * ccc_loss # + self.gamma_st * ccc_st_loss
        return loss

    def topk_ct(self, seq_batch, item_candidates, k=4):
        """return [bsz, k] index"""
        sim = torch.matmul(seq_batch, item_candidates.transpose(0, 1))
        rst = torch.topk(sim, k, dim=1)[1]
        return rst

    def topk_cc(self, seq_source, seq_target, k=4):
        """return [bsz, k] index"""
        sim = torch.matmul(seq_source, seq_target.transpose(0, 1))
        rst = torch.topk(sim, k, dim=0)[1]
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
            return feed_dict

        def actions_before_epoch(self):
            super().actions_before_epoch()
            self.model.cc_dl = self.construct_cc_dl()

        def construct_cc_dl(self):
            if not self.model.cc_dataset:
                self.model.cc_dataset = copy.deepcopy(self)
            dataset_copy = self.model.cc_dataset

            item_ids = dataset_copy.data['item_id']
            min_st_freq = self.model.min_st_freq
            n = len(item_ids)
            n_sample = int(n/min_st_freq) + 1

            item_counts = np.unique(item_ids, return_counts=True)
            item_pool, item_freq = item_counts[0], item_counts[1]
            ind_item = {i: np.where(item_ids == i)[0] for i in item_pool}
            p_item = item_freq / sum(item_freq)
            item_selected = np.random.choice(
                item_pool, size=n_sample, replace=True, p=p_item)
            ind_set = [
                np.random.choice(ind_item[i], min_st_freq, replace=True) for i in item_selected]
            ind = np.stack(ind_set).flatten()[:n]
            for key in dataset_copy.data.keys():
                dataset_copy.data[key] = dataset_copy.data[key][ind]

            # dl = SequentialSampler(data_dicts)
            dl = DataLoader(
                dataset_copy, batch_size=self.model.batch_size,
                shuffle=False, num_workers=self.model.num_workers,
                collate_fn=self.collate_batch, pin_memory=self.model.pin_memory)
            dl = dl.__iter__()
            # ind_keep = item_freq > min_st_freq
            # item_freq = item_freq[item_freq > min_st_freq]
            # plt.boxplot(item_freq[item_freq>5])
            # plt.show()
            return dl
