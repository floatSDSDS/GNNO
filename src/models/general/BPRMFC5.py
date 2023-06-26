# -*- coding: UTF-8 -*-
# @Author  : Chenyang Wang
# @Email   : THUwangcy@gmail.com

""" BPRMF
Reference:
    "Bayesian personalized ranking from implicit feedback"
    Rendle et al., UAI'2009.
CMD example:
    python main.py --model_name BPRMF --emb_size 64 --lr 1e-3 --l2 1e-6 --dataset 'Grocery_and_Gourmet_Food'
"""

import torch.nn as nn
import numpy as np

from models.BaseModel import GeneralModel
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


class BPRMFC5(GeneralModel):
    reader = 'BaseReader'
    runner = 'BaseRunner'
    extra_log_args = ['emb_size',
                      'c1', 'c2', 'cost', 'cz', 'layer_cz'
                      ]

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64, help='Size of embedding vectors.')
        parser.add_argument('--c1', type=float, default=0.1, help='embedding coef.')
        parser.add_argument('--c2', type=float, default=0.75, help='embedding coef.')
        parser.add_argument('--cost', type=float, default=0.25, help='commitment_cost.')
        parser.add_argument('--cz', type=int, default=500, help='cluster size.')
        parser.add_argument('--layer_cz', type=int, default=1, help='commitment_cost.')
        return GeneralModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.emb_size = args.emb_size
        self._define_params()
        self.apply(self.init_weights)

        self.c1 = args.c1
        self.c2 = args.c2
        self.cost = args.cost
        self.cz = args.cz
        self.layer_cz = args.layer_cz
        cluster_size = []
        cz = args.cz
        for i in range(self.layer_cz):
            cluster_size.append(int(cz / np.power(5, i)))

        self.cfg = BaseRecommenderConfig()
        self.c5 = C5Trainer(
                embed_dim=self.cfg.hidden_size,
                vocab_size=1,
                # cluster_sizes=[100, 10],
                cluster_sizes=cluster_size,
                num_layers=self.cfg.num_codebook_layers,
                layer_connect=self.cfg.layer_connect,
                layer_loss=False,
                commitment_cost=self.cost,
            )

    def _define_params(self):
        self.u_embeddings = nn.Embedding(self.user_num, self.emb_size)
        self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)

    def forward(self, feed_dict):
        self.check_list = []
        u_ids = feed_dict['user_id']  # [batch_size]
        i_ids = feed_dict['item_id']  # [batch_size, -1]

        cf_u_vectors = self.u_embeddings(u_ids)
        cf_i_vectors = self.i_embeddings(i_ids)

        quantized = self.c5.quantize(cf_i_vectors, with_loss=True)
        item_embeddings = cf_i_vectors + self.c1 * quantized.mean

        prediction = (cf_u_vectors[:, None, :] * item_embeddings).sum(dim=-1)  # [batch_size, -1]
        return {'prediction': prediction.view(feed_dict['batch_size'], -1),
                'loss_q': quantized.loss
                }

    def loss(self, out_dict):
        predictions = out_dict['prediction']
        pos_pred, neg_pred = predictions[:, 0], predictions[:, 1:]
        neg_softmax = (neg_pred - neg_pred.max()).softmax(dim=1)
        loss = -((pos_pred[:, None] - neg_pred).sigmoid() * neg_softmax).sum(dim=1).log().mean()

        loss2 = out_dict['loss_q']
        return loss + self.c2 * loss2
