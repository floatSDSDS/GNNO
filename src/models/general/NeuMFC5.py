# -*- coding: UTF-8 -*-
# @Author  : Chenyang Wang
# @Email   : THUwangcy@gmail.com

""" NeuMF
Reference:
    "Neural Collaborative Filtering"
    Xiangnan He et al., WWW'2017.
Reference code:
    The authors' tensorflow implementation https://github.com/hexiangnan/neural_collaborative_filtering
CMD example:
    python main.py --model_name NeuMF --emb_size 64 --layers '[64]' --lr 5e-4 --l2 1e-7 --dropout 0.2 \
    --dataset 'Grocery_and_Gourmet_Food'
"""

import torch
import torch.nn as nn

from models.BaseModel import GeneralModel
from utils.c5_trainer import C5Trainer

import numpy as np


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


class NeuMFC5(GeneralModel):
    reader = 'BaseReader'
    runner = 'BaseRunner'
    extra_log_args = ['emb_size', 'layers',
                      'c1', 'c2', 'cost', 'cz', 'layer_cz'
                      ]

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--layers', type=str, default='[64]',
                            help="Size of each layer.")
        parser.add_argument('--c1', type=float, default=0.1, help='embedding coef.')
        parser.add_argument('--c2', type=float, default=0.75, help='embedding coef.')
        parser.add_argument('--cost', type=float, default=0.25, help='commitment_cost.')
        parser.add_argument('--cz', type=int, default=500, help='cluster size.')
        parser.add_argument('--layer_cz', type=int, default=1, help='commitment_cost.')
        return GeneralModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.emb_size = args.emb_size
        self.layers = eval(args.layers)

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
        self.apply(self.init_weights)
        self.cfg = BaseRecommenderConfig()
        self.c5_mf = C5Trainer(
                embed_dim=self.cfg.hidden_size,
                vocab_size=1,
                cluster_sizes=cluster_size,
                num_layers=self.cfg.num_codebook_layers,
                layer_connect=True,
                layer_loss=False,
                commitment_cost=self.cost,
            )

        self.c5_mlp = C5Trainer(
            embed_dim=self.cfg.hidden_size,
            vocab_size=1,
            cluster_sizes=cluster_size,
            num_layers=self.cfg.num_codebook_layers,
            layer_connect=True,
            layer_loss=False,
            commitment_cost=self.cost,
        )

    def _define_params(self):
        self.mf_u_embeddings = nn.Embedding(self.user_num, self.emb_size)
        self.mf_i_embeddings = nn.Embedding(self.item_num, self.emb_size)
        self.mlp_u_embeddings = nn.Embedding(self.user_num, self.emb_size)
        self.mlp_i_embeddings = nn.Embedding(self.item_num, self.emb_size)

        self.mlp = nn.ModuleList([])
        pre_size = 2 * self.emb_size
        for i, layer_size in enumerate(self.layers):
            self.mlp.append(nn.Linear(pre_size, layer_size))
            pre_size = layer_size
        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.prediction = nn.Linear(pre_size + self.emb_size, 1, bias=False)

    def forward(self, feed_dict):
        self.check_list = []
        u_ids = feed_dict['user_id']  # [batch_size]
        i_ids = feed_dict['item_id']  # [batch_size, -1]

        u_ids = u_ids.unsqueeze(-1).repeat((1, i_ids.shape[1]))  # [batch_size, -1]

        mf_u_vectors = self.mf_u_embeddings(u_ids)
        mf_i_vectors = self.mf_i_embeddings(i_ids)
        mlp_u_vectors = self.mlp_u_embeddings(u_ids)
        mlp_i_vectors = self.mlp_i_embeddings(i_ids)

        quantized_mf = self.c5_mf.quantize(mf_i_vectors, with_loss=True)
        final_mf_i_vectors = mf_i_vectors + quantized_mf.mean * self.c1

        quantized_mlp = self.c5_mlp.quantize(mlp_i_vectors, with_loss=True)
        final_mlp_i_vectors = mlp_i_vectors + quantized_mlp.mean * self.c1

        mf_vector = mf_u_vectors * final_mf_i_vectors

        mlp_vector = torch.cat([mlp_u_vectors, final_mlp_i_vectors], dim=-1)
        for layer in self.mlp:
            mlp_vector = layer(mlp_vector).relu()
            mlp_vector = self.dropout_layer(mlp_vector)

        output_vector = torch.cat([mf_vector, mlp_vector], dim=-1)
        prediction = self.prediction(output_vector)
        return {
            'prediction': prediction.view(feed_dict['batch_size'], -1),
            'loss_q': quantized_mf.loss + quantized_mlp.loss,
        }

    def loss(self, out_dict):
        predictions = out_dict['prediction']
        pos_pred, neg_pred = predictions[:, 0], predictions[:, 1:]
        neg_softmax = (neg_pred - neg_pred.max()).softmax(dim=1)
        loss = -((pos_pred[:, None] - neg_pred).sigmoid() * neg_softmax).sum(dim=1).log().mean()

        loss2 = out_dict['loss_q']
        return loss + self.c2 * loss2
