# -*- coding: UTF-8 -*-
# @Author  : Chenyang Wang
# @Email   : THUwangcy@gmail.com

""" LightGCN
Reference:
    "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation"
    He et al., SIGIR'2020.
CMD example:
    python main.py --model_name LightGCN --emb_size 64 --n_layers 3 --lr 1e-3 --l2 1e-8 \
    --dataset 'Grocery_and_Gourmet_Food'
"""

import torch
import numpy as np
import torch.nn as nn
import scipy.sparse as sp
import numpy as np


from models.BaseModel import GeneralModel
# from utils.base_recommender import BaseRecommender
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


class LightGCNC5(GeneralModel):
    reader = 'BaseReader'
    runner = 'BaseRunner'
    extra_log_args = ['emb_size', 'n_layers',
                      'c1', 'c2', 'cost', 'cz', 'layer_cz'
                      ]

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64, help='Size of embedding vectors.')
        parser.add_argument('--n_layers', type=int, default=3, help='Number of LightGCN layers.')
        parser.add_argument('--c1', type=float, default=0.1, help='embedding coef.')
        parser.add_argument('--c2', type=float, default=0.75, help='embedding coef.')
        parser.add_argument('--cost', type=float, default=0.25, help='commitment_cost.')
        parser.add_argument('--cz', type=int, default=500, help='cluster size.')
        parser.add_argument('--layer_cz', type=int, default=1, help='commitment_cost.')
        return GeneralModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.emb_size = args.emb_size
        self.n_layers = args.n_layers

        self.c1 = args.c1
        self.c2 = args.c2
        self.cost = args.cost
        self.cz = args.cz
        self.layer_cz = args.layer_cz

        self.norm_adj = self.build_adjmat(corpus.n_users, corpus.n_items, corpus.train_clicked_set)

        self._define_params()
        self.apply(self.init_weights)

    @staticmethod
    def build_adjmat(user_count, item_count, train_mat, selfloop_flag=False):
        R = sp.dok_matrix((user_count, item_count), dtype=np.float32)
        for user in train_mat:
            for item in train_mat[user]:
                R[user, item] = 1
        R = R.tolil()

        adj_mat = sp.dok_matrix((user_count + item_count, user_count + item_count), dtype=np.float32)
        adj_mat = adj_mat.tolil()

        adj_mat[:user_count, user_count:] = R
        adj_mat[user_count:, :user_count] = R.T
        adj_mat = adj_mat.todok()

        def normalized_adj_single(adj):
            # D^-1/2 * A * D^-1/2
            rowsum = np.array(adj.sum(1)) + 1e-10

            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

            bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
            return bi_lap.tocoo()

        if selfloop_flag:
            norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        else:
            norm_adj_mat = normalized_adj_single(adj_mat)

        return norm_adj_mat.tocsr()

    def _define_params(self):
        self.encoder = LGCNEncoder(
            self.user_num, self.item_num,
            self.emb_size, self.norm_adj,
            self.n_layers,
            self.c1, self.c2, self.cost, self.cz, self.layer_cz
        )

    def forward(self, feed_dict):
        self.check_list = []
        user, items = feed_dict['user_id'], feed_dict['item_id']

        u_embed, i_embed, loss = self.encoder(user, items)

        prediction = (u_embed[:, None, :] * i_embed).sum(dim=-1)
        out_dict = {'prediction': prediction, 'loss_q': loss}
        return out_dict

    def loss(self, out_dict):
        predictions = out_dict['prediction']
        pos_pred, neg_pred = predictions[:, 0], predictions[:, 1:]
        neg_softmax = (neg_pred - neg_pred.max()).softmax(dim=1)
        loss = -((pos_pred[:, None] - neg_pred).sigmoid() * neg_softmax).sum(dim=1).log().mean()

        loss2 = out_dict['loss_q']
        return loss + loss2


class LGCNEncoder(nn.Module):
    def __init__(self, user_count, item_count, emb_size, norm_adj, n_layers=3,
                 c1=0, c2=0, cost=0.25, cz=500, layer_cz=1):
        super(LGCNEncoder, self).__init__()
        self.user_count = user_count
        self.item_count = item_count
        self.emb_size = emb_size
        self.layers = [emb_size] * n_layers
        self.norm_adj = norm_adj

        self.c1 = c1
        self.c2 = c2
        self.cost = cost
        self.cz = cz
        self.layer_cz = layer_cz
        cluster_size = []
        cz = cz
        for i in range(self.layer_cz):
            cluster_size.append(int(cz / np.power(5, i)))

        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.norm_adj).cuda()
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

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.user_count, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.item_count, self.emb_size))),
        })
        return embedding_dict

    @staticmethod
    def _convert_sp_mat_to_sp_tensor(X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def forward(self, users, items):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = [ego_embeddings]

        for k in range(len(self.layers)):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            all_embeddings += [ego_embeddings]

        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)

        user_all_embeddings = all_embeddings[:self.user_count, :]
        item_all_embeddings = all_embeddings[self.user_count:, :]

        user_embeddings = user_all_embeddings[users, :]
        item_embeddings = item_all_embeddings[items, :]

        quantized = self.c5.quantize(item_embeddings, with_loss=True)
        item_embeddings = item_embeddings + self.c1 * quantized.mean
        return user_embeddings, item_embeddings, self.c2 * quantized.loss
