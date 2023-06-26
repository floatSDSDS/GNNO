# -*- coding: UTF-8 -*-
# @Author  : Chenyang Wang
# @Email   : THUwangcy@gmail.com

""" CFKG
Reference:
    "Learning over Knowledge-Base Embeddings for Recommendation"
    Yongfeng Zhang et al., SIGIR'2018.
Note:
    In the built-in dataset, we have four kinds of relations: buy, category, complement, substitute, where 'buy' is
    a special relation indexed by 0. And there are three kinds of nodes in KG: user, item, category, among which
    users are placed ahead of other entities when indexing.
CMD example:
    python main.py --model_name CFKG --emb_size 64 --margin 1 --include_attr 1 --lr 1e-4 --l2 1e-6 \
    --dataset 'Grocery_and_Gourmet_Food'
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from utils import utils
from models.BaseModel import GeneralModel
from helpers.KGReader import KGReader
from utils.c5_trainer import C5Trainer


class BaseRecommenderConfig:
    def __init__(
            self,
            hidden_size=64,
            use_c5=False,
            cluster_sizes=None,
            num_codebook_layers=0,
            layer_connect=True,
            commitment_cost=0.25,
            **kwargs,
    ):
        self.hidden_size = hidden_size
        self.use_c5 = use_c5
        self.cluster_sizes = cluster_sizes
        self.num_codebook_layers = num_codebook_layers
        self.layer_connect = layer_connect
        self.commitment_cost = commitment_cost


class CFKGC5(GeneralModel):
    reader = 'KGReader'
    runner = 'BaseRunner'
    extra_log_args = ['emb_size', 'margin', 'include_attr',
                      'c1', 'c2', 'cost', 'cz', 'layer_cz'
                      ]

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--margin', type=float, default=0,
                            help='Margin in hinge loss.')
        parser.add_argument('--c1', type=float, default=0.1, help='embedding coef.')
        parser.add_argument('--c2', type=float, default=0.75, help='embedding coef.')
        parser.add_argument('--cost', type=float, default=0.25, help='commitment_cost.')
        parser.add_argument('--cz', type=int, default=500, help='cluster size.')
        parser.add_argument('--layer_cz', type=int, default=1, help='commitment_cost.')
        return GeneralModel.parse_model_args(parser)

    def __init__(self, args, corpus: KGReader):
        super().__init__(args, corpus)
        self.emb_size = args.emb_size
        self.margin = args.margin
        self.relation_num = corpus.n_relations
        self.entity_num = corpus.n_entities

        self.c1 = args.c1
        self.c2 = args.c2
        self.cost = args.cost
        self.cz = args.cz
        self.layer_cz = args.layer_cz
        cluster_size = []
        cz = args.cz
        for i in range(self.layer_cz):
            cluster_size.append(int(cz / np.power(5, i)))

        self._define_params()

        self.cfg = BaseRecommenderConfig()
        self.c5 = C5Trainer(
                embed_dim=self.cfg.hidden_size,
                vocab_size=1,
                cluster_sizes=cluster_size,
                num_layers=self.cfg.num_codebook_layers,
                layer_connect=self.cfg.layer_connect,
                layer_loss=False,
                commitment_cost=self.cost,
            )

        self.apply(self.init_weights)

    def _define_params(self):
        self.e_embeddings = nn.Embedding(self.user_num + self.entity_num, self.emb_size)
        # ↑ user and entity embeddings, user first
        self.r_embeddings = nn.Embedding(self.relation_num, self.emb_size)
        # ↑ relation embedding: 0 is used for "buy" between users and items
        self.loss_function = nn.MarginRankingLoss(margin=self.margin)

    def forward(self, feed_dict):
        self.check_list = []
        head_ids = feed_dict['head_id']  # [batch_size, -1]
        tail_ids = feed_dict['tail_id']  # [batch_size, -1]
        relation_ids = feed_dict['relation_id']  # [batch_size, -1]

        head_vectors = self.e_embeddings(head_ids)
        tail_vectors = self.e_embeddings(tail_ids)
        relation_vectors = self.r_embeddings(relation_ids)

        quantized = self.c5.quantize(head_vectors, with_loss=True)
        head_vectors = head_vectors + self.c1 * quantized.mean
        loss1 = quantized.loss
        quantized = self.c5.quantize(tail_vectors, with_loss=True)
        tail_vectors = self.e_embeddings(tail_ids) + self.c1 * quantized.mean
        loss2 = quantized.loss

        prediction = -((head_vectors + relation_vectors - tail_vectors)**2).sum(-1)
        return {
            'prediction': prediction.view(feed_dict['batch_size'], -1),
            'loss_q': loss1 + loss2
        }

    def loss(self, out_dict):
        predictions = out_dict['prediction']
        batch_size = predictions.shape[0]
        pos_pred, neg_pred = predictions[:, :2].flatten(), predictions[:, 2:].flatten()
        target = torch.from_numpy(np.ones(batch_size * 2, dtype=np.float32)).to(self.device)
        loss = self.loss_function(pos_pred, neg_pred, target)

        return loss + self.c2 * out_dict['loss_q']

    class Dataset(GeneralModel.Dataset):
        def __init__(self, model, corpus, phase):
            super().__init__(model, corpus, phase)
            if self.phase == 'train':
                interaction_df = pd.DataFrame({
                    'head': self.data['user_id'],
                    'tail': self.data['item_id'],
                    'relation': np.zeros_like(self.data['user_id'])  # "buy" relation
                })
                self.data = utils.df_to_dict(pd.concat((self.corpus.relation_df, interaction_df), axis=0))
                self.neg_heads = np.zeros(len(self), dtype=int)
                self.neg_tails = np.zeros(len(self), dtype=int)

        def _get_feed_dict(self, index):
            if self.phase == 'train':
                head, tail = self.data['head'][index], self.data['tail'][index]
                relation = self.data['relation'][index]
                head_id = np.array([head, head, head, self.neg_heads[index]])
                tail_id = np.array([tail, tail, self.neg_tails[index], tail])
                relation_id = np.array([relation] * 4)
                if relation > 0:  # head is not a user
                    head_id = head_id + self.corpus.n_users
            else:
                target_item = self.data['item_id'][index]
                if self.model.test_all:
                    neg_items = np.arange(1, self.corpus.n_items)
                else:
                    neg_items = self.data['neg_items'][index]
                tail_id = np.concatenate([[target_item], neg_items])
                head_id = self.data['user_id'][index] * np.ones_like(tail_id)
                relation_id = np.zeros_like(tail_id)
            tail_id += self.corpus.n_users  # tail must be a non-user entity

            feed_dict = {'head_id': head_id, 'tail_id': tail_id, 'relation_id': relation_id}
            return feed_dict

        def actions_before_epoch(self):
            for i in range(len(self)):
                head, tail, relation = self.data['head'][i], self.data['tail'][i], self.data['relation'][i]
                self.neg_tails[i] = np.random.randint(1, self.corpus.n_items)
                if relation == 0:  # "buy" relation
                    self.neg_heads[i] = np.random.randint(1, self.corpus.n_users)
                    while self.neg_tails[i] in self.corpus.train_clicked_set[head]:
                        self.neg_tails[i] = np.random.randint(1, self.corpus.n_items)
                    while tail in self.corpus.train_clicked_set[self.neg_heads[i]]:
                        self.neg_heads[i] = np.random.randint(1, self.corpus.n_users)
                else:
                    self.neg_heads[i] = np.random.randint(1, self.corpus.n_entities)
                    while (head, relation, self.neg_tails[i]) in self.corpus.triplet_set:
                        self.neg_tails[i] = np.random.randint(1, self.corpus.n_entities)
                    while (self.neg_heads[i], relation, tail) in self.corpus.triplet_set:
                        self.neg_heads[i] = np.random.randint(1, self.corpus.n_entities)
