import json
import os.path as osp
import random
import re

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import optim
from torch.nn import Linear
from torch.utils.data import random_split
from torch_geometric.data import Data, Dataset, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.nn.models import GAT, GCN
from torchmetrics.functional import kendall_rank_corrcoef
from torchvision.ops import MLP

from vulnerability_networks.config import PATH_MODELS, PATH_PROCESSED_DATA

# seed = 42
# L.seed_everything(seed, workers=True)
# torch.manual_seed(seed)
# np.random.seed(seed)
# torch.backends.cudnn.benchmark = False
# torch.use_deterministic_algorithms(True)


# def seed_worker(worker_id):
#     worker_seed = torch.initial_seed() % 2**32
#     np.random.seed(worker_seed)
#     random.seed(worker_seed)


# g = torch.Generator()
# g.manual_seed(seed)
# %%


class NetworkDataset(InMemoryDataset):
    def __init__(self, root, data_list, transform=None):
        self.data_list = data_list
        super().__init__(root, transform)
        self.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return "data.pt"

    def process(self):
        self.save(self.data_list, self.processed_paths[0])


# %%
def map_at_k(pred_edges, target_edges, index_edges, k=0.4):
    graph_ids, count_edges = index_edges.unique(return_counts=True)
    K = (k * count_edges).floor().int()
    average_precisions = []
    for graph_id, k in zip(graph_ids, K):
        # find relevant edges of a graph
        target_graph_edges = target_edges[index_edges == graph_id]
        # I wont use the top k relevant edges because there may be ties, find the min score and set all edges with at least
        # that score as relevant
        min_score_at_k = torch.topk(target_graph_edges, k.item()).values.min()
        relevant_graph_edges = torch.nonzero(target_graph_edges >= min_score_at_k, as_tuple=True)[0]
        # find the ranking of edge predictions
        pred_graph_edges = pred_edges[index_edges == graph_id]
        rank_edge_predictions = torch.topk(pred_graph_edges, k.item()).indices
        rank_edge_predictions = torch.isin(rank_edge_predictions, relevant_graph_edges).cumsum(dim=0)
        positions = torch.arange(1, k.item() + 1, device=rank_edge_predictions.device)
        average_precision = torch.div(rank_edge_predictions, positions).mean()
        average_precisions.append(average_precision)
    return torch.tensor(average_precisions).mean()

def rank_loss(preds, targets, phi_fn):
    targets = targets.repeat(len(targets)).view(len(targets),-1)
    targets = targets < targets.T

    preds = preds.repeat(len(preds)).view(len(preds),-1)
    preds = preds.T - preds
    loss = phi_fn(preds[targets]).sum()
    return loss

# def soft_kendall_tau(preds, targets, temperature=1.0):
#     """Differentiable approximation of Kendall's Tau"""
#     n = preds.size(0)
    
#     # Create all pairs of indices
#     idx_i, idx_j = torch.triu_indices(n, n, offset=1)
    
#     # Get pairs of predictions and ground truth
#     pred_pairs_i, pred_pairs_j = preds[idx_i], preds[idx_j]
#     true_pairs_i, true_pairs_j = targets[idx_i], targets[idx_j]
    
#     # Compute signs of the pairs (using soft sign with sigmoid)
#     pred_sign = torch.sigmoid((pred_pairs_i - pred_pairs_j) / temperature)
#     true_sign = torch.sigmoid((true_pairs_i - true_pairs_j) / temperature)
    
#     # Concordant pairs contribute positively, discordant pairs negatively
#     # Using smooth L1 or MSE to measure the agreement
#     pair_loss = torch.nn.functional.mse_loss(pred_sign, true_sign, reduction='none')
    
#     # Compute the mean loss over all pairs
#     return torch.mean(pair_loss)

class RRN(torch.nn.Module):
    def __init__(self, num_x_features, embedding_size, num_layers_msg, num_layers_mlp, msg_passing="GCN", dropout=0):
        super(RRN, self).__init__()
        if msg_passing == "GCN":
            self.msg_block = GCN(
                in_channels=num_x_features, hidden_channels=embedding_size, num_layers=num_layers_msg, dropout=dropout
            )
        elif msg_passing == "GAT":
            self.msg_block = GAT(
                in_channels=num_x_features, hidden_channels=embedding_size, num_layers=num_layers_msg, dropout=dropout
            )

        self.edge_mlp = MLP(
            in_channels=2 * embedding_size + 1,
            hidden_channels=[embedding_size, embedding_size, embedding_size, 1],
            dropout=dropout,
        )

    def forward(self, x, edge_index, edge_weight):
        # Create node embeddings using the message block
        node_embeddings = self.msg_block(x, edge_index, edge_weight)
        # Create edge embeddings (by concatenating). An edge is node(tail) ---> node(head)
        tails, heads = edge_index
        edge_weight_reshaped = edge_weight.view(-1, 1)
        edge_embeddings = torch.cat([node_embeddings[tails], node_embeddings[heads], edge_weight_reshaped], dim=1)
        edge_logits = self.edge_mlp(edge_embeddings).squeeze(-1)
        return edge_logits
        # return F.sigmoid(edge_scores)


class LightningEdgeCriticality(L.LightningModule):
    def __init__(
        self,
        num_x_features: int,
        embedding_size: int,
        num_layers_msg: int,
        num_layers_mlp: int,
        msg_passing: str,
        loss_fn: str,
        dropout: float,
        loss_phi_fn = None,
        lr = 1e-3
    ):
        super().__init__()
        assert msg_passing in ["GCN", "GAT"]
        assert loss_fn in ["MSE", "RANK"]
        if loss_fn == "RANK":
            assert loss_phi_fn in ["logistic", "exponential"]
            if loss_phi_fn == "logistic":
                self.loss_phi_fn = lambda z: torch.log(1 + 1/torch.exp(z))
            elif loss_phi_fn == "exponential":
                self.loss_phi_fn = lambda z: (1/torch.exp(z))
        else:
            self.loss_phi_fn = None

        self.save_hyperparameters()
        self.model = RRN(num_x_features, embedding_size, num_layers_msg, num_layers_mlp, msg_passing, dropout)
        self.loss_fn = loss_fn
        self.learning_rate = lr
        # self.register_buffer("some_tensor", torch.tensor([...]))

    def loss(self, *args, **kwargs):
        if self.loss_fn == "MSE":
            preds = F.sigmoid(kwargs["logits"])
            return F.mse_loss(preds, kwargs["targets"])
        elif self.loss_fn == "RANK":
            return rank_loss(kwargs["logits"], kwargs["targets"], self.loss_phi_fn)

    def training_step(self, batch, batch_idx):
        edge_logits = self.model(batch.x, batch.edge_index, batch.edge_attr)
        loss = self.loss(logits=edge_logits, targets=batch.edge_y)
        self.log(f"train_loss_{self.loss_fn}", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        edge_logits = self.model(batch.x, batch.edge_index, batch.edge_attr)
        val_loss = self.loss(logits=edge_logits, targets=batch.edge_y)
        kendal_corr = kendall_rank_corrcoef(edge_logits, batch.edge_y, variant="b")
        map_at_k_perc = map_at_k(edge_logits, batch.edge_y, batch.edge_attr_batch)
        self.log_dict(
            {f"val_loss_{self.loss_fn}": val_loss, "val_kendall_corr": kendal_corr, "val_map_at_k": map_at_k_perc},
            batch_size=batch.batch_size,
        )
        return val_loss

    def test_step(self, batch, batch_idx):
        edge_logits = self.model(batch.x, batch.edge_index, batch.edge_attr)
        test_loss = self.loss(logits=edge_logits, targets=batch.edge_y)
        kendal_corr = kendall_rank_corrcoef(edge_logits, batch.edge_y)
        map_at_k_perc = map_at_k(edge_logits, batch.edge_y, batch.edge_attr_batch)
        self.log_dict(
            {f"test_loss_{self.loss_fn}": test_loss, "test_kendall_corr": kendal_corr, "test_map_at_k": map_at_k_perc},
            batch_size=batch.batch_size,
        )
        return test_loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# %%

index_accessibility = "independent_path"

assert index_accessibility in ["global_efficiency", "independent_path"]

full_dataset = NetworkDataset(
    PATH_PROCESSED_DATA/f"{index_accessibility}_old", []
)

dataset_size = len(full_dataset)
train_size = int(0.8 * dataset_size)  # 70% for training
val_size = int(0.1 * dataset_size)  # 15% for validation
test_size = dataset_size - train_size - val_size  # 15% for testing

# Split the dataset
train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

# Create DataLoaders
batch_size = 32

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    follow_batch=["edge_attr"],
    shuffle=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    follow_batch=["edge_attr"],
    shuffle=False,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    follow_batch=["edge_attr"],
    shuffle=False,
)

# %%
lit_model = LightningEdgeCriticality(
    num_x_features=full_dataset.num_features,
    embedding_size=64,
    num_layers_msg=3,
    num_layers_mlp=3,
    msg_passing="GAT",
    loss_fn="RANK",
    loss_phi_fn="logistic",
    dropout=0.2,
)


#%%


tb_logger = L.pytorch.loggers.TensorBoardLogger(save_dir=PATH_MODELS, name=index_accessibility)
csv_logger = L.pytorch.loggers.CSVLogger(save_dir=PATH_MODELS, name=index_accessibility, version=tb_logger.version)
# Train the model
trainer = L.Trainer(max_epochs=100, accelerator="auto", logger=[tb_logger, csv_logger])
trainer.fit(lit_model, train_loader, val_loader)

# Test the model
trainer.test(lit_model, test_loader)


# %%
# Version 9 es NIP
# Versión 10 es GE
edge_scores = torch.randint(1, 10, (18,))
indices = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
# Calcular cuántos elementos hay por grupo
unique_ids, counts = indices.unique(return_counts=True)
K = (0.4 * counts).floor().int()
targets = []
for uid, k in zip(unique_ids, K):
    edge_scores_uid = edge_scores[indices == uid]
    targets_uid = edge_scores_uid >= torch.topk(edge_scores_uid, k.item()).values.min()
    targets.append(targets_uid)
targets = torch.cat(targets)


# k = int(len(edge_scores)*0.4)
# targets = edge_scores >= torch.topk(edge_scores, k).values.min()


# %%
# out = RRN(num_x_features=3, embedding_size=64, num_layers_msg=3, num_layers_mlp=3)(batch.x, batch.edge_index, batch.edge_attr)
# print(out.shape)
# class GCN(torch.nn.Module):
#     def __init__(self, hidden_channels):
#         super(GCN, self).__init__()
#         self.conv1 = GCNConv(3, hidden_channels)
#         self.conv2 = GCNConv(hidden_channels, hidden_channels)
#         self.conv3 = GCNConv(hidden_channels, hidden_channels)

#         # Edge scoring module
#         self.edge_mlp = torch.nn.Sequential(
#             Linear(
#                 2 * hidden_channels + 1, hidden_channels
#             ),  # Edge embeddings(i.e., 2 x node embeddings) + edge weight
#             torch.nn.ReLU(),
#             torch.nn.Dropout(0.2),
#             Linear(hidden_channels, hidden_channels // 2),
#             torch.nn.ReLU(),
#             Linear(hidden_channels // 2, 1),
#         )

#     def forward(self, x, edge_index, edge_weight, batch):
#         # 1. Obtain node embeddings
#         node_embeddings = self.conv1(x, edge_index, edge_weight)
#         node_embeddings = node_embeddings.relu()
#         node_embeddings = self.conv2(node_embeddings, edge_index, edge_weight)
#         node_embeddings = node_embeddings.relu()
#         node_embeddings = self.conv3(node_embeddings, edge_index, edge_weight)

#         # 2. Readout layer
#         # x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

#         # 3. Apply a final classifier
#         # x = F.dropout(x, p=0.5, training=self.training)

#         # Create edge embeddings (by concatenating). An edge is node(tail) ---> node(head)
#         tails, heads = edge_index
#         edge_weight_reshaped = edge_weight.view(-1, 1)
#         edge_embeddings = torch.cat([node_embeddings[tails], node_embeddings[heads], edge_weight_reshaped], dim=1)

#         edge_scores = self.edge_mlp(edge_embeddings).squeeze(-1)

#         return F.sigmoid(edge_scores)

# class LightningEdgeCriticality(L.LightningModule):
#     def __init__(self, model):
#         super().__init__()
#         self.model = model
#         self.optimizer = None
#         self.scheduler = None
#         self.kendall = KendallRankCorrCoef(variant="b")  # variant b give importance to ties

#     def mse_loss(self, edge_hat, edge_score):
#         return F.mse_loss(edge_hat, edge_score)

#     def kendall_tau_rank(self, edge_hat, edge_score):
#         return self.kendall(edge_hat, edge_score)

#     def training_step(self, batch, batch_idx):
#         edge_hat = self.model(batch.x, batch.edge_index, batch.edge_attr)
#         loss = self.mse_loss(edge_hat, batch.edge_y)
#         self.log("train_loss", loss)
#         return loss

#     def validation_step(self, batch, batch_idx):
#         edge_hat = self.model(batch.x, batch.edge_index, batch.edge_attr)
#         val_loss = self.mse_loss(edge_hat, batch.edge_y)
#         kendal_corr = self.kendall(edge_hat, batch.edge_y)
#         self.log_dict({"val_loss": val_loss, "kendall_corr": kendal_corr}, batch_size=batch.batch_size)
#         return val_loss

#     def test_step(self, batch, batch_idx):
#         edge_hat = self.model(batch.x, batch.edge_index, batch.edge_attr)
#         test_loss = self.mse_loss(edge_hat, batch.edge_y)
#         kendal_corr = self.kendall(edge_hat, batch.edge_y)
#         self.log_dict({"test_loss": test_loss, "kendall_corr": kendal_corr}, batch_size=batch.batch_size)
#         return test_loss

# def configure_optimizers(self):
#     """
#     Initializes the optimizer and learning rate scheduler

#     :return: output - Initialized optimizer and scheduler
#     """
#     self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
#     self.scheduler = {
#         "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
#             self.optimizer,
#             mode="min",
#             factor=0.2,
#             patience=2,
#             min_lr=1e-6,
#             verbose=True,
#         ),
#         "monitor": "val_loss",
#     }
#     return [self.optimizer], [self.scheduler]


#%%
# x = torch.tensor([21, 10, 30]).repeat(3).view(3,-1)
# torch.triu(x)
# x = torch.tensor(["edge1", "edge2", "edge3"], dtype="str")
phi_fn = F.sigmoid()
targets = torch.tensor([1,2,3])
preds = torch.tensor([1,2,3])

def rank_loss(preds, targets, phi_fn):
    targets = targets.repeat(len(targets)).view(len(targets),-1)
    targets = targets < targets.T

    preds = preds.repeat(len(preds)).view(len(preds),-1)
    preds = preds.T - preds
    loss = phi_fn(preds[targets]).sum()
    return loss

def logistic(z):
    return torch.log(1 + 1/torch.exp(z))
