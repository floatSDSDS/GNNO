import time

import torch
from torch import nn
from tqdm import tqdm

from c5_trainer import C5Trainer
from common import LayerConnector, ConnectNode, DecoderLayer, C5Module, C5Quantization, C5Classification
from printer import printer, Color


class C5Booster(C5Module):
    def __init__(self, trainer: C5Trainer, item_embeddings, broaden_ratio=0.2):
        super().__init__()

        self.initializing = True
        self.num_layers = trainer.num_layers
        self.vocab_size = trainer.vocab_size
        self.embed_dim = trainer.embed_dim
        self.transform_layer = trainer.transform_layer

        self.embedding_tables = [item_embeddings, *trainer.codebooks]
        self.decoders = [trainer.decoder_layer, *trainer.codebook_decoders]

        self.print = printer[(self.__class__.__name__, '-', Color.MAGENTA)]
        self.print(f'c5 connector constructing with broaden ratio {broaden_ratio} ...')
        self.connector = LayerConnector(
            embedding_tables=self.embedding_tables,
            broaden_ratio=broaden_ratio,
        )
        self.connector.visualize()

        self.print('c5 tree constructing ...')
        self.nodes = self.connector.nodes  # type: list[list[ConnectNode]]
        self.root = self.construct_tree()

        self.print('c5 item pre quantization ...')
        self.quantized_embeddings = self.prepare_quantization()

        self.initializing = False

    def construct_tree(self):
        root = ConnectNode(device=self.embedding_tables[0].weight.device)

        for i in range(len(self.embedding_tables) - 1):
            embeddings = self.embedding_tables[i].weight  # [8, d]
            self.print(f'Constructing layer {i+1} with {len(embeddings)} available leaves '
                       f'and {len(self.nodes[i])} nodes ...')

            decoder = self.decoders[i]  # type: DecoderLayer

            for node in self.nodes[i]:
                for j in node.leaves:
                    leaf = j if i == 0 else self.nodes[i - 1][j]
                    node.add_leaf_node(
                        leaf_node=leaf,
                        leaf_embedding=embeddings[j],  # [d]
                        leaf_decoder_linear=decoder.decoder.weight[j],
                        leaf_decoder_bias=decoder.bias[j],
                    )

        # self.print(f'Constructing root layer with {len(self.nodes[-1])} available leaves ...')
        for i, leaf in enumerate(self.nodes[-1]):
            if not leaf.leaves:
                continue
            root.add_leaf(i, None)
            root.add_leaf_node(
                leaf_node=leaf,
                leaf_embedding=leaf.center,
                leaf_decoder_linear=self.decoders[-1].decoder.weight[i],
                leaf_decoder_bias=self.decoders[-1].bias[i],
            )

        root.finish_adding_leaves(layer=0)
        return root

    def quantize(
            self,
            embeds,  # [D]
            with_loss=False,
    ) -> C5Quantization:
        start_ = time.time()

        qindices, qembeds = self.root.quantize(embeds)
        qindices = qindices[1:]  # [L]
        qembeds = qembeds[1:]  # [L, D]

        end_ = time.time()
        if not self.initializing:
            self.timer.append('quantize', end_ - start_)

        # return C5Quantization(qembeds, indices=qindices)
        return C5Quantization(qembeds, indices=qindices)

    def batch_quantize(
            self,
            embeds,  # [B, S, D]
    ):
        shape = embeds.shape
        embeds = embeds.view(-1, shape[-1])  # [B * S, D]

        qindices, qembeds = self.root.batch_quantize(embeds)
        qindices = torch.tensor(qindices, dtype=torch.long, device=embeds.device)[:, 1:]
        qembeds = torch.stack([torch.stack(row) for row in qembeds], dim=0)[:, 1:]

        qindices = qindices.view(shape[:-1] + (-1,))  # [B, S, L]
        qembeds = qembeds.view(shape[:-1] + (-1, shape[-1]))  # [B, S, L, D]

        return C5Quantization(qembeds, indices=qindices)

    def track_quantize(self, embeds):
        shape = embeds.shape  # [B, ..., D]
        embeds = embeds.view(-1, self.embed_dim)  # [B * ..., D]
        paths = []
        for embed in embeds:  # [D]
            path = torch.stack(self.root.track_quantize(embed), dim=0)
            paths.append(path)
        paths = torch.stack(paths, dim=0).view(shape[:-1] + (-1,))  # [B, ..., L]
        return paths

    def classify(
            self,
            embeds,
            indices=False,
    ) -> C5Classification:
        start_ = time.time()
        embeds = self.transform_layer(embeds)  # [D]

        indices = self.root.classify(embeds)

        end_ = time.time()
        self.timer.append('classify', end_ - start_)
        return C5Classification(
            scores=None,
            indices=indices,
        )

    def batch_classify(
            self,
            embeds,  # [B, D]
    ):
        embeds = self.transform_layer(embeds)  # [B, D]
        indices = self.root.batch_classify(embeds)  # [B, Vx], [B, Vx]

        return C5Classification(
            scores=None,
            indices=indices,
        )

    def track_classify(self, embed):
        embed = self.transform_layer(embed)  # [D]
        return self.root.track_classify(embed)

    def visualize(self):
        last_level_nodes = [self.root]
        next_level_nodes = []
        level = 0

        while True:
            last_level_nodes = list(set(last_level_nodes))
            self.print(f'level {level}: {len(last_level_nodes)} nodes')
            if not isinstance(last_level_nodes[0], ConnectNode):
                break
            for node in last_level_nodes:  # type: ConnectNode
                next_level_nodes.extend(node.leaf_nodes)
            last_level_nodes = next_level_nodes
            next_level_nodes = []
            level += 1

    def track(self, item_id):
        print(f'tracking item {item_id} ...')

        last_track_ids = {item_id}
        next_track_ids = set()
        level = self.num_layers

        nodes_list = [*self.nodes]

        while level:
            nodes = nodes_list[level - 1]
            for i, node in enumerate(nodes):
                for track_id in last_track_ids:
                    if track_id in node.leaves:
                        print(f'level {level}: {track_id} -> {i}')
                        next_track_ids.add(i)

            last_track_ids = next_track_ids
            next_track_ids = set()
            level -= 1

    def prepare_quantization(self):
        quantization = []
        item_embeddings = self.embedding_tables[0]
        item_embeddings = item_embeddings.weight.detach()
        for embed in tqdm(item_embeddings):
            quantization.append(self.quantize(embed).mean)

        quantization = torch.stack(quantization, dim=0)
        return nn.Embedding.from_pretrained(quantization, freeze=True)
