# -*- coding: UTF-8 -*-
# @Author  : Chenyang Wang
# @Email   : THUwangcy@gmail.com

""" GRU4Rec
Reference:
    "Session-based Recommendations with Recurrent Neural Networks"
    Hidasi et al., ICLR'2016.
CMD example:
    python main.py --model_name GRU4Rec --emb_size 64 --hidden_size 128 --lr 1e-3 --l2 1e-4 --history_max 20 \
    --dataset 'Grocery_and_Gourmet_Food'
"""
from tqdm import tqdm

import dgl
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SequentialSampler
from torch.nn.utils.rnn import pad_sequence

from models.BaseModel import SequentialModel
from utils import layers
import utils.build_witg as bw


class BERT4Rec(SequentialModel):
    reader = 'SeqReader'
    runner = 'BaseRunner'
    extra_log_args = [
        'emb_size', 'hidden_size',
        'f_hn', 'fn_split', 'n_hn', 'hn0',
        'c_pace', 'max_hard'
    ]

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--hidden_size', type=int, default=64,
                            help='Size of hidden vectors in Bert.')
        parser.add_argument('--f_hn', type=str, default='rand',
                            help='how to retrieve hard negatives. '
                                 '[rand, dns, nei, jaccard] ')
        parser.add_argument('--n_hn', type=int, default=3, help='number of hard negatives.')
        parser.add_argument('--fn_split', type=int, default=15, help='where to split false negative.')
        parser.add_argument('--hn0', type=int, default=5, help='when to start add hn.')
        parser.add_argument('--c_pace', type=float, default=0.02, help='hardness increment pace.')
        parser.add_argument('--max_hard', type=float, default=0.7, help='maximum hardenss.')
        return SequentialModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.emb_size = args.emb_size
        self.hidden_size = args.hidden_size
        self.max_his = args.history_max
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.pin_memory = args.pin_memory

        self.f_hn = args.f_hn
        self.n_hn = args.n_hn
        self.fn_split = args.fn_split
        self.hn0 = args.hn0

        self.hardeness = 0.0
        self.c_pace = args.c_pace
        self.max_hard = args.max_hard

        self.epoch = 0
        self._define_params()
        self.apply(self.init_weights)

    def _define_params(self):
        self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)
        self.encoder = BERT4RecEncoder(self.emb_size, self.max_his, num_layers=2, num_heads=2)
        # self.pred_embeddings = nn.Embedding(self.item_num, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.emb_size)

    def forward(self, feed_dict):
        self.check_list = []
        i_ids = feed_dict['item_id']  # [batch_size, -1]
        history = feed_dict['history_items']  # [batch_size, history_max]
        lengths = feed_dict['lengths']  # [batch_size]

        his_vectors = self.i_embeddings(history)

        # RNN
        his_vector = self.encoder(his_vectors, lengths)
        i_vectors = self.i_embeddings(i_ids)
        prediction = (his_vector[:, None, :] * i_vectors).sum(-1)
        out_dict = {'prediction': prediction}
        # return {'prediction': prediction.view(feed_dict['batch_size'], -1)}
        return out_dict

    class Dataset(SequentialModel.Dataset):
        def _get_feed_dict(self, index):
            feed_dict = super()._get_feed_dict(index)
            if self.phase == 'train':
                feed_dict['index'] = index
                if self.model.epoch > self.model.hn0:
                    feed_dict['item_id'] = np.concatenate(
                        [feed_dict['item_id'], self.data['hn'][index]]
                    )
            return feed_dict

        def prepare(self):
            super().prepare()
            if self.model.buffer and self.phase == 'train':
                self._construct_graph()

        def actions_before_epoch(self):
            super().actions_before_epoch()
            self.i_vectors = self.model.i_embeddings(
                torch.arange(0, self.model.item_num, device=self.model.device)).detach()
            if self.model.epoch >= self.model.hn0:
                self._update_hn_nei()
                if self.model.hardeness < self.model.max_hard:
                    hardeness = self.model.hardeness + self.model.c_pace
                    self.model.hardeness = min(hardeness, self.model.max_hard)
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

            self.g_adj = self.g_dgl.adj().to_dense()
            g_cn = torch.matmul(self.g_adj, self.g_adj.t())
            g_deg = g_cn.diag().expand_as(g_cn)
            g_union = g_deg + g_deg.t() - g_cn
            self.g_jaccard = g_cn / g_union
            self.g_jaccard = self.g_jaccard.nan_to_num().to(self.model.device)
            print("graph constructed")

        def _update_hn_nei(self):
            targets = self.data['item_id']
            dl = DataLoader(
                targets, batch_size=self.model.batch_size,
                shuffle=False, num_workers=self.model.num_workers,
                pin_memory=self.model.pin_memory)
            hn_lst = []
            for target_batch in tqdm(dl, leave=False, ncols=100, mininterval=1):
                neg_batch = getattr(self, f'_gen_hn_{self.model.f_hn}')(target_batch)
                hn_lst.append(neg_batch)
            self.data['hn'] = torch.cat(hn_lst).detach().cpu().numpy()

        def _gen_hn_jaccard(self, target_batch):
            target_batch = target_batch.to(self.model.device)
            mask = self.g_jaccard <= self.model.hardeness
            mask_batch = mask[target_batch]
            jaccard_batch = self.g_jaccard[target_batch]
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


class BERT4RecEncoder(nn.Module):
    def __init__(self, emb_size, max_his, num_layers=2, num_heads=2):
        super().__init__()
        self.p_embeddings = nn.Embedding(max_his + 1, emb_size)
        self.transformer_block = nn.ModuleList([
            layers.TransformerLayer(d_model=emb_size, d_ff=emb_size, n_heads=num_heads)
            for _ in range(num_layers)
        ])

    def forward(self, seq, lengths):
        batch_size, seq_len = seq.size(0), seq.size(1)
        len_range = torch.from_numpy(np.arange(seq_len)).to(seq.device)
        valid_mask = len_range[None, :] < lengths[:, None]

        # Position embedding
        position = len_range[None, :] * valid_mask.long()
        pos_vectors = self.p_embeddings(position)
        seq = seq + pos_vectors

        # Self-attention
        attn_mask = valid_mask.view(batch_size, 1, 1, seq_len)
        for block in self.transformer_block:
            seq = block(seq, attn_mask)
        seq = seq * valid_mask[:, :, None].float()

        his_vector = seq[torch.arange(batch_size), lengths - 1]
        return his_vector
