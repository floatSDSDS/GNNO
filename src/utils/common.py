from typing import Optional, Union

import torch
from torch import nn
from tqdm import tqdm
from transformers.activations import ACT2FN


class C5Quantization:
    def __init__(self, embeds, loss=None, indices=None):
        self.embeds = embeds
        self.loss = loss
        self.indices = indices
        if len(self.embeds) > 0:
            if isinstance(self.embeds, torch.Tensor):
                self.mean = torch.mean(self.embeds, dim=2)
            else:
                self.mean = torch.mean(torch.stack(self.embeds), dim=0)
        else:
            self.mean = 0


class C5Classification:
    def __init__(self, scores, layer_loss=None, indices=None):
        self.scores = scores
        self.indices = indices
        self.layer_loss = layer_loss


class C5Module(nn.Module):
    def quantize(
            self,
            embeds,
            with_loss=False,
    ) -> C5Quantization:
        raise NotImplementedError

    def classify(
            self,
            embeds,
            indices=None,
    ) -> C5Classification:
        raise NotImplementedError


class TransformLayer(nn.Module):
    """
    Transform layer for Classifier
    """

    def __init__(
            self,
            embed_dim,
            activation_function,
            layer_norm_eps=None,
    ):
        super(TransformLayer, self).__init__()
        self.transform = nn.Linear(embed_dim, embed_dim)
        self.transform_act_fn = ACT2FN[activation_function]
        self.layer_norm = nn.LayerNorm(embed_dim, eps=layer_norm_eps or 1e-5)

    def forward(self, hidden_states) -> torch.Tensor:
        hidden_states = self.transform(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


class DecoderLayer(nn.Module):
    """
    Decoder layer for Classifier, projecting hidden states to vocab_size
    """

    def __init__(
            self,
            embed_dim,
            vocab_size,
    ):
        super(DecoderLayer, self).__init__()
        self.decoder = nn.Linear(embed_dim, vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(vocab_size), requires_grad=True)
        self.decoder.bias = self.bias

    def forward(self, hidden_states) -> torch.Tensor:
        return self.decoder(hidden_states)

    def set_values(self, decoder_weights: torch.Tensor, decoder_bias: torch.Tensor):
        self.decoder.weight.data = decoder_weights.data
        self.bias.data = decoder_bias.data
        self.decoder.bias = self.bias


class ConnectNode:
    def __init__(self, center: torch.Tensor = None, device=None):
        self.center = center  # [d]
        self.device = device or center.device

        self.leaves = []  # type: Union[torch.Tensor, list]
        self.leaves_set = set()
        self.radius = 0

        self.leaf_embeddings = []  # type: Union[torch.Tensor, list]
        self.leaf_nodes = []  # type: list[Union[ConnectNode, int]]

        self.leaf_decoder_linear = []  # type: Union[torch.Tensor, list]
        self.leaf_decoder_bias = []  # type: Union[torch.Tensor, list]
        self.leaf_decoder = None  # type: Optional[DecoderLayer]

    def add_leaf(self, index, leaf: Optional[torch.Tensor]):
        """
        :param index: index of the leaf
        :param leaf: [d]
        """
        assert index not in self.leaves_set
        self.leaves.append(index)
        self.leaves_set.add(index)
        if leaf is not None:
            self.radius = max(self.radius, torch.dist(self.center, leaf, p=2).item())

    def add_leaf_node(
            self,
            leaf_node: 'ConnectNode',
            leaf_embedding: torch.Tensor,  # [d]
            leaf_decoder_linear: torch.Tensor,
            leaf_decoder_bias: torch.Tensor
    ):
        self.leaf_nodes.append(leaf_node)
        self.leaf_embeddings.append(leaf_embedding)

        self.leaf_decoder_linear.append(leaf_decoder_linear)
        self.leaf_decoder_bias.append(leaf_decoder_bias)

    def finish_adding_leaves(self, layer):
        self.leaves = torch.tensor(self.leaves, dtype=torch.long).to(self.device)

        if isinstance(self.leaf_embeddings, list):
            if not self.leaf_embeddings:
                print(f'leaf_embeddings is empty at layer {layer}')
            self.leaf_embeddings = torch.stack(self.leaf_embeddings).to(self.device)

            self.leaf_decoder_linear = torch.stack(self.leaf_decoder_linear).to(self.device)
            self.leaf_decoder_bias = torch.stack(self.leaf_decoder_bias).to(self.device)
            self.leaf_decoder = DecoderLayer(
                embed_dim=self.leaf_embeddings.shape[1],
                vocab_size=self.leaf_embeddings.shape[0],
            ).to(self.device)
            self.leaf_decoder.set_values(
                decoder_weights=self.leaf_decoder_linear,
                decoder_bias=self.leaf_decoder_bias,
            )

            if self.leaf_nodes and isinstance(self.leaf_nodes[0], ConnectNode):
                for node in self.leaf_nodes:
                    node.finish_adding_leaves(layer + 1)

    def quantize(self, embed: torch.Tensor) -> (list, list):
        """
        Get the leaf searching path
        :param embed: [d]
        """
        dist = torch.cdist(embed.unsqueeze(dim=0), self.leaf_embeddings, p=2)
        dist = dist.squeeze(dim=0)
        min_index = torch.argmin(dist)
        min_embed = self.leaf_embeddings[min_index]
        leaf_node = self.leaf_nodes[min_index]
        indices, embeds = [], []
        if isinstance(leaf_node, ConnectNode):
            indices, embeds = leaf_node.quantize(embed)
        indices.append(min_index)
        embeds.append(min_embed)
        return indices, embeds

    def batch_quantize(self, embeds: torch.Tensor) -> (list, list):
        batch_size = embeds.shape[0]

        dist = torch.cdist(embeds, self.leaf_embeddings, p=2)
        min_indices = torch.argmin(dist, dim=1)
        min_embeds = self.leaf_embeddings[min_indices]

        batch_indices, batch_embeds = [], []
        for _ in range(batch_size):
            batch_indices.append([])
            batch_embeds.append([])

        values = torch.unique(min_indices)
        for value in values:
            indices = torch.where(min_indices == value)[0]
            if isinstance(self.leaf_nodes[value], ConnectNode):
                leaf_indices, leaf_embeds = self.leaf_nodes[value].batch_quantize(embeds[indices])
                for i in range(len(indices)):
                    batch_indices[indices[i]] += leaf_indices[i]
                    batch_embeds[indices[i]] += leaf_embeds[i]

        min_indices = min_indices.tolist()
        for i in range(batch_size):
            batch_indices[i].append(min_indices[i])
            batch_embeds[i].append(min_embeds[i])
        return batch_indices, batch_embeds

    def batch_classify(self, embeds: torch.Tensor):
        batch_size = embeds.shape[0]

        scores = self.leaf_decoder(embeds)  # [B, V]
        max_indices = torch.argmax(scores, dim=1)

        batch_indices = [None] * batch_size

        values = torch.unique(max_indices)
        is_leaf = True
        for value in values:
            indices = torch.where(max_indices == value)[0]
            if isinstance(self.leaf_nodes[value], ConnectNode):
                is_leaf = False
                leaf_indices = self.leaf_nodes[value].batch_classify(embeds[indices])
                for i in range(len(indices)):
                    batch_indices[indices[i]] = leaf_indices[i]
                    
        if is_leaf:
            orders = torch.argsort(scores, dim=1, descending=True)
            # indices = torch.index_select(self.leaves, 0, orders)
            return self.leaves[orders]

        return batch_indices

    def classify(self, embed: torch.Tensor):
        """
        Classify the embedding
        :param embed: [d]
        """
        scores = self.leaf_decoder(embed.unsqueeze(dim=0)).squeeze(dim=0)  # [V]

        # get the max score index
        max_index = torch.argmax(scores)
        leaf_node = self.leaf_nodes[max_index]
        if isinstance(leaf_node, ConnectNode):
            return leaf_node.classify(embed)

        order = torch.argsort(scores, descending=True)
        indices = torch.index_select(self.leaves, 0, order)  # [V]
        return indices  # [V], [V]

    def track_classify(self, embed: torch.Tensor):
        scores = self.leaf_decoder(embed.unsqueeze(dim=0)).squeeze(dim=0)
        max_index = torch.argmax(scores)
        leaf_node = self.leaf_nodes[max_index]
        if isinstance(leaf_node, ConnectNode):
            return [max_index] + leaf_node.track_classify(embed)
        return [max_index]

    def track_quantize(self, embed: torch.Tensor):
        dist = torch.cdist(embed.unsqueeze(dim=0), self.leaf_embeddings, p=2)
        dist = dist.squeeze(dim=0)

        min_index = torch.argmin(dist)
        leaf_node = self.leaf_nodes[min_index]

        if isinstance(leaf_node, ConnectNode):
            return [min_index] + leaf_node.track_quantize(embed)
        return [min_index]


class LayerConnector:
    def __init__(self, embedding_tables, broaden_ratio):

        self.embedding_tables = embedding_tables  # [[8, d], [4, d], [2, d]]
        self.broaden_ratio = broaden_ratio
        self.nodes = []  # self.nodes: [[4*ConnectNode], [2*ConnectNode]]

        self.initialize_nodes()
        self.connect_nodes()

        self.succ_broaden_count = 0
        if self.broaden_ratio > 0:
            self.broaden_nodes()
        else:
            self.print('Skip broadening nodes.')

        # self.remove_empty_nodes()

    def initialize_nodes(self):
        self.print('Constructing nodes...')
        for table in self.embedding_tables[1:]:  # type: nn.Embedding # [4, d], [2, d]
            nodes = []
            for embedding in table.weight:  # [d]
                nodes.append(ConnectNode(embedding))
            self.nodes.append(nodes)

    def connect_nodes(self):
        self.print('Connecting nodes...')
        for i in range(len(self.embedding_tables) - 1):
            last_table = self.embedding_tables[i].weight  # type: torch.Tensor
            next_table = self.embedding_tables[i + 1].weight  # type: torch.Tensor

            dists = torch.cdist(last_table, next_table, p=2)
            indices = torch.argmin(dists, dim=1)  # [8]

            for j, index in enumerate(indices):  # j: 0-7, index: 0-3
                if i and not self.nodes[i - 1][j].leaves:  # if the last layer is empty, continue
                    continue
                self.nodes[i][index].add_leaf(j, last_table[j])

    def _broaden(self, o: ConnectNode, t: ConnectNode, leaf_embeddings):
        """
        if node o and node t are close enough, add leaves from o to t
        :param o: origin node
        :param t: target node
        :param leaf_embeddings: [n, d]
        """
        if o is t:
            return
        if torch.dist(o.center, t.center, p=2) > o.radius + t.radius:  # prune
            return

        indices = torch.tensor(o.leaves, dtype=torch.long).to(o.device)
        leaf_embeddings = torch.index_select(leaf_embeddings, 0, indices)

        dists = torch.cdist(leaf_embeddings, t.center.unsqueeze(dim=0), p=2)
        dists = dists.squeeze(dim=1) <= t.radius * self.broaden_ratio  # type: torch.Tensor

        for i, dist in enumerate(dists):
            if dist and o.leaves[i] not in t.leaves_set:
                t.add_leaf(o.leaves[i], None)
                self.succ_broaden_count += 1

    def broaden_nodes(self):
        """
        Broaden the nodes at the same level
        """
        self.print('Broadening nodes...')
        for index, nodes in tqdm(enumerate(self.nodes)):
            leaf_embeddings = self.embedding_tables[index].weight
            for node in tqdm(nodes):  # type: ConnectNode
                dists = torch.cdist(leaf_embeddings, node.center.unsqueeze(dim=0), p=2)
                dists = dists.squeeze(dim=1) <= node.radius * self.broaden_ratio

                for i, dist in enumerate(dists):
                    if dist and i not in node.leaves_set:
                        if index and not self.nodes[index - 1][i].leaves:
                            continue
                        node.add_leaf(i, None)
                        self.succ_broaden_count += 1
        self.print('Broaden nodes done. {} leaves added.'.format(self.succ_broaden_count))

    def visualize(self):
        for i, nodes in enumerate(self.nodes):
            self.print(f'Layer {i}: {len(nodes)} nodes')
            leaves = 0
            for node in nodes:
                leaves += len(node.leaves)
            self.print(f'Layer {i}: {leaves} leaves')


    # def remove_empty_nodes(self):
    #     self.print('Removing empty nodes...')
    #     for nodes in self.nodes:
    #         for node in nodes:  # type: ConnectNode
    #             if len(node.leaves) == 0:
    #                 nodes.remove(node)
