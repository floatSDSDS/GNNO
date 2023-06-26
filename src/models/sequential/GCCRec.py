import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler

import numpy as np

import matplotlib.pyplot as plt

from models.BaseModel import SequentialModel
from utils import layers, utils

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler

import numpy as np

import matplotlib.pyplot as plt

from models.BaseModel import SequentialModel
from utils import layers, utils, build_witg

from models.sequential.ContraRec import BERT4RecEncoder, ContraLoss


class STRec(SequentialModel):
    reader = 'SeqReader'
    runner = 'BaseRunner'
    extra_log_args = [
        'num_neg', 'batch_size',
        'gamma', 'gamma_st', 'min_st_freq',
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

        parser.add_argument('--gamma_st', type=float, default=0.5,
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
