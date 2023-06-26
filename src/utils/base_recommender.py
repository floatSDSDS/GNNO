import torch
from torch import nn

from loader.global_setting import Setting
from model.c5.c5_trainer import C5Trainer
from model.utils.column_map import ColumnMap
from loader.embedding_manager import EmbeddingManager
from utils.printer import printer, Color


class BaseRecommenderConfig:
    def __init__(
            self,
            hidden_size,
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


class BaseRecommender(nn.Module):
    config_class = BaseRecommenderConfig

    def __init__(
            self,
            config: BaseRecommenderConfig,
            column_map: ColumnMap,
            embedding_manager: EmbeddingManager,
    ):
        super().__init__()

        self.config = config  # type: BaseRecommenderConfig
        self.print = printer[(self.__class__.__name__, '|', Color.MAGENTA)]

        self.column_map = column_map  # type: ColumnMap
        self.embedding_manager = embedding_manager
        self.embedding_table = embedding_manager.get_table()

        self.user_col = column_map.user_col
        self.item_col = column_map.item_col
        self.label_col = column_map.label_col

        if self.config.use_c5:
            self.c5 = C5Trainer(
                embed_dim=self.config.hidden_size,
                vocab_size=1,
                cluster_sizes=[100, 10],
                num_layers=self.config.num_codebook_layers,
                layer_connect=self.config.layer_connect,
                layer_loss=False,
                commitment_cost=self.config.commitment_cost,
            )

    def forward(self, batch):
        labels = batch[self.label_col].to(Setting.device)

        item_embedding = self.embedding_manager(self.item_col)(batch[self.item_col].to(Setting.device))
        user_embedding = self.embedding_manager(self.user_col)(batch[self.user_col].to(Setting.device))

        additional_loss = 0
        if self.config.use_c5:
            quantized = self.c5.quantize(item_embedding, with_loss=True)
            final_item_embedding = item_embedding + quantized.mean
            additional_loss = quantized.loss
        else:
            final_item_embedding = item_embedding

        results = self.predict(user_embedding, final_item_embedding, labels)
        return results, additional_loss

    def predict(self, user_embedding, candidates, batch) -> [torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return self.__str__()
