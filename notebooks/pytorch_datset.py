import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import networkx as nx
    import matplotlib.pyplot as plt
    from torch.nn import Linear
    from torch import optim
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv
    from torch_geometric.nn import global_mean_pool
    import torch
    import numpy as np
    from torch_geometric.data import Data, Dataset
    import os.path as osp
    import random
    from torch_geometric.loader import DataLoader
    import pandas as pd
    import json
    from vulnerability_networks.config import PATH_INTERIM_DATA, PATH_RAW_DATA
    import re
    import lightning as L
    from torch_geometric.data import InMemoryDataset
    from torch.utils.data import random_split

    return (
        Data,
        DataLoader,
        F,
        GCNConv,
        InMemoryDataset,
        L,
        Linear,
        PATH_INTERIM_DATA,
        PATH_RAW_DATA,
        json,
        nx,
        optim,
        pd,
        random_split,
        re,
        torch,
    )


@app.cell
def _():

    # G = nx.DiGraph()
    # G.add_edge(1, 2, weight=0.4)
    # G.add_edge(1, 3, weight=0.3)
    # G.add_edge(1, 5, weight=0.1)
    # G.add_edge(2, 3, weight=0.111)
    # G.add_edge(3, 4, weight=0.4)
    # G.add_edge(4, 5, weight=0.56)

    # # explicitly set positions
    # pos = {1: (0, 0), 2: (-1, 0.3), 3: (2, 0.17), 4: (4, 0.255), 5: (5, 0.03)}

    # nx.draw_networkx_edges(G, pos, width=6)
    # nx.draw_networkx_nodes(G, pos, node_size=700)

    # # node labels
    # nx.draw_networkx_labels(G, pos, font_size=20, font_family="sans-serif")
    # # edge weight labels
    # edge_labels = nx.get_edge_attributes(G, "weight")
    # nx.draw_networkx_edge_labels(G, pos, edge_labels)

    # # Set margins for the axes so that nodes aren't clipped
    # ax = plt.gca()
    # ax.margins(0.20)
    # plt.axis("off")
    # plt.show()
    return


@app.cell
def _():

    # G2 = nx.DiGraph()
    # G2.add_edge(1, 2, weight=0.4)
    # G2.add_edge(1, 3, weight=0.3)
    # G2.add_edge(1, 5, weight=0.1)
    # G2.add_edge(2, 3, weight=0.5)

    # # explicitly set positions
    # pos2 = {1: (0, 0), 2: (-1, 0.3), 3: (2, 0.17), 4: (4, 0.255), 5: (5, 0.03)}

    # nx.draw_networkx_edges(G2, pos2, width=6)
    # nx.draw_networkx_nodes(G2, pos2, node_size=700)

    # # node labels
    # nx.draw_networkx_labels(G2, pos2, font_size=20, font_family="sans-serif")
    # # edge weight labels
    # edge_labels2 = nx.get_edge_attributes(G2, "weight")
    # nx.draw_networkx_edge_labels(G2, pos2, edge_labels2)

    # # Set margins for the axes so that nodes aren't clipped
    # ax2 = plt.gca()
    # ax2.margins(0.20)
    # plt.axis("off")
    # plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Convert to PyG Dataset""")
    return


@app.cell
def _(PATH_INTERIM_DATA, PATH_RAW_DATA, json, pd, re):
    # load catalog
    with open(PATH_RAW_DATA / "network_catalog.json", "r") as f:
        catalog = json.load(f)

    df_catalog = pd.DataFrame(catalog)
    df_catalog["path"] = df_catalog["path"].apply(lambda network_path: PATH_RAW_DATA / network_path / "graph.gml")

    # load criticality scores
    # nip scores
    nip_scores = []
    for path in (PATH_INTERIM_DATA/"independent_paths_ratiolinks0_4_max_scenarios10000").rglob("*.json"):
        with open(path, "r") as f:
            network_results = json.load(f)
            if network_results["link_scores"]:
                match = re.search(r'network_(\d+)', network_results["network_path"])
                network_id = int(match.group(1)) if match else None
                if network_id is None:
                    raise Exception("No se econtro el id de la red")
                nip_scores.append({"network_id": network_id, "nip_criticality_scores": network_results["link_scores"]}) 
    df_nip_scores = pd.DataFrame(nip_scores)       

    # ge scores
    ge_scores = []
    for path in (PATH_INTERIM_DATA/"global_efficiency_ratiolinks0_4_max_scenarios10000").rglob("*.json"):
        with open(path, "r") as f:
            network_results = json.load(f)
            if network_results["link_scores"]:
                match = re.search(r'network_(\d+)', network_results["network_path"])
                network_id = int(match.group(1)) if match else None
                if network_id is None:
                    raise Exception("No se econtro el id de la red")
                ge_scores.append({"network_id": network_id, "ge_criticality_scores": network_results["link_scores"]}) 
    df_ge_scores = pd.DataFrame(ge_scores)

    df_scores = df_ge_scores.merge(df_nip_scores,how="inner").merge(df_catalog, how="inner")
    return (df_scores,)


@app.cell
def _(Data, nx, torch):
    def add_weights_to_existing_edges(
        G: nx.Graph,
        edges: list[list[list[str], float]], edge_type: str) -> None:
        """
        Adds weights only to existing edges in the graph.

        Args:
            G (nx.Graph): Existing NetworkX graph.
            edges (list): List of [[[u, v], weight]].
        """
        for (u, v), w in edges:
            #u, v = int(u), int(v)
            if G.has_edge(u, v):
                G[u][v][edge_type] = w
            else:
                # Optionally log or warn here
                raise Exception(f"Edge ({u}, {v}) not found in graph. Skipping.")

    # Function to convert a single NetworkX graph to PyTorch Geometric Data
    def convert_nx_to_pyg(nx_graph, normalize_scores=False):
        # Get node indices
        # Get node features
        one_hot = {"source": [1, 0, 0], "terminal": [0, 1, 0], "regular": [0, 0, 1]}
        x = []
        node_indices = {}
        for i, node in enumerate( nx_graph.nodes()):
            node_indices[node] = i
            # If you have node features, extract them here
            # This is just a placeholder; replace with your actual node features
            node_data = nx_graph.nodes[node]
            if node_data:
                # Example: Extract 'feature' attribute if it exists
                if 'profile' in node_data:
                    x.append(one_hot[node_data['profile']])
                else:
                    raise Exception("No tiene caracteristica específica")
            else:
                raise Exception("No tiene features")

        # Get edge indices (2 x num_edges)
        edge_index = []
        edge_attr = []
        edge_criticality_scores = []

        for u, v, data in nx_graph.edges(data=True):
            edge_index.append([node_indices[u], node_indices[v]])
            # If you have edge attributes, extract them here
            if data:
                edge_attr.append(data["weight"])
                edge_criticality_scores.append(data["criticality_score"])

            else:
                raise Exception("No hay data en el enlace")

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        x = torch.tensor(x, dtype=torch.float)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        edge_y = torch.tensor(edge_criticality_scores, dtype=torch.float) # ground of truth
        if normalize_scores:
            edge_y = (edge_y - edge_y.min())/(edge_y.max() - edge_y.min())
        # Create PyG Data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, edge_y = edge_y)

        return data

    # G_ = nx.read_gml(df_scores.iloc[0]["path"])
    # add_weights_to_existing_edges(G_, df_scores.iloc[0]["ge_criticality_scores"], "criticality_score")
    # data_ = convert_nx_to_pyg(G_, normalize_scores=True)
    return add_weights_to_existing_edges, convert_nx_to_pyg


@app.cell
def _():
    # data = convert_nx_to_pyg(G)
    # data.x = F.one_hot(data.x.long().flatten()).float()

    # data2 = convert_nx_to_pyg(G2)
    # data2.x = F.one_hot(data2.x.long().flatten()).float()
    return


@app.cell
def _(add_weights_to_existing_edges, convert_nx_to_pyg, df_scores, nx):
    networks_ge, networks_nip = [], []
    for _, row in df_scores.iterrows():
        G_ge = nx.read_gml(row["path"])
        G_nip = G_ge.copy()
        if len(row["ge_criticality_scores"]) == row["edges"]: # TODO: revisar por que no se generan puntajes para todos los indices
            add_weights_to_existing_edges(G_ge, row["ge_criticality_scores"], "criticality_score")
            data = convert_nx_to_pyg(G_ge, normalize_scores=False)
            networks_ge.append(data)

        if len(row["nip_criticality_scores"]) == row["edges"]:
            add_weights_to_existing_edges(G_nip, row["nip_criticality_scores"], "criticality_score")
            data = convert_nx_to_pyg(G_nip, normalize_scores=False)
            networks_nip.append(data)
    return networks_ge, networks_nip


@app.cell
def _(InMemoryDataset):
    class NetworkDataset(InMemoryDataset):
        def __init__(self, root, data_list, transform=None):
            self.data_list = data_list
            super().__init__(root, transform)
            self.load(self.processed_paths[0])

        @property
        def processed_file_names(self):
            return 'data.pt'

        def process(self):
            self.save(self.data_list, self.processed_paths[0])
    return (NetworkDataset,)


@app.cell
def _(NetworkDataset, networks_ge, networks_nip):
    ds_ge = NetworkDataset("/home/danielaguirre/University/vulnerability-networks/data/processed/global_efficiency", networks_ge)
    ds_nip = NetworkDataset("/home/danielaguirre/University/vulnerability-networks/data/processed/independent_path", networks_nip)
    return (ds_nip,)


@app.cell
def _(F, GCNConv, L, Linear, optim, torch):
    class GCN(torch.nn.Module):
        def __init__(self, hidden_channels):
            super(GCN, self).__init__()
            torch.manual_seed(12345)
            self.conv1 = GCNConv(3, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
            self.conv3 = GCNConv(hidden_channels, hidden_channels)

            # Edge scoring module
            self.edge_mlp = torch.nn.Sequential(
                Linear(2 * hidden_channels + 1, hidden_channels),  # Edge embeddings(i.e., 2 x node embeddings) + edge weight
                torch.nn.ReLU(),
                torch.nn.Dropout(0.2),
                Linear(hidden_channels, hidden_channels // 2),
                torch.nn.ReLU(),
                Linear(hidden_channels // 2, 1)
            )

        def forward(self, x, edge_index, edge_weight, batch):
            # 1. Obtain node embeddings 
            node_embeddings = self.conv1(x, edge_index, edge_weight)
            node_embeddings = node_embeddings.relu()
            node_embeddings = self.conv2(node_embeddings, edge_index, edge_weight)
            node_embeddings = node_embeddings.relu()
            node_embeddings = self.conv3(node_embeddings, edge_index, edge_weight)

            # 2. Readout layer
            # x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

            # 3. Apply a final classifier
            # x = F.dropout(x, p=0.5, training=self.training)

            # Create edge embeddings (by concatenating). An edge is node(tail) ---> node(head)
            tails, heads = edge_index
            edge_weight_reshaped = edge_weight.view(-1, 1)
            edge_embeddings = torch.cat([node_embeddings[tails], node_embeddings[heads], edge_weight_reshaped], dim=1)

            edge_scores = self.edge_mlp(edge_embeddings).squeeze(-1)
        
            return F.sigmoid(edge_scores)


    class LitGCN(L.LightningModule):
        def __init__(self, model):
            super().__init__()
            self.model = model
        
        def training_step(self, batch, batch_idx):
            edge_hat = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = F.mse_loss(edge_hat, batch.edge_y)
            self.log("train_loss", loss)
            return loss
    
        def validation_step(self, batch, batch_idx):
            edge_hat = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            val_loss = F.mse_loss(edge_hat, batch.edge_y)
            self.log("val_loss", val_loss)
            return val_loss
    
        def test_step(self, batch, batch_idx):
            edge_hat = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            test_loss = F.mse_loss(edge_hat, batch.edge_y)
            self.log("test_loss", test_loss)
            return test_loss
        
        def configure_optimizers(self):
            optimizer = optim.Adam(self.parameters(), lr=1e-3)
            return optimizer

    return GCN, LitGCN


@app.cell
def _(DataLoader, ds_nip, random_split, torch):
    full_dataset = ds_nip

    dataset_size = len(full_dataset)
    train_size = int(0.8 * dataset_size)  # 70% for training
    val_size = int(0.1 * dataset_size)   # 15% for validation
    test_size = dataset_size - train_size - val_size  # 15% for testing

    # Split the dataset
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # Set seed for reproducibility
    )

    # Create DataLoaders
    batch_size = 32

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    return test_loader, train_loader, val_loader


@app.cell
def _(GCN, L, LitGCN, test_loader, train_loader, val_loader):
    # Initialize model
    hidden_channels = 64
    model = GCN(hidden_channels)
    lit_model = LitGCN(model)

    # Train the model
    trainer = L.Trainer(max_epochs=100, accelerator="auto")
    trainer.fit(lit_model, train_loader, val_loader)

    # Test the model
    trainer.test(lit_model, test_loader)
    return (model,)


@app.cell
def _(dl_nip, model):
    for d in dl_nip:
        out = model(d.x, d.edge_index, d.edge_attr, d.batch)
        print(out.shape, d.edge_y.shape)
        break
    return d, out


app._unparsable_cell(
    r"""
        d
    """,
    name="_"
)


@app.cell
def _(d):
    d.edge_index
    return


@app.cell
def _(d):
    d.x
    return


@app.cell
def _(d):
    d.batch
    return


@app.cell
def _(d):
    for i in d.edge_index:
        print(list(i))
    return


@app.cell
def _(out):
    out.shape
    return


@app.cell
def _(d):
    tails, heads = d.edge_index
    return heads, tails


@app.cell
def _(heads, out):
    out[heads]
    return


@app.cell
def _(heads, out, tails, torch):
    torch.concat([out[tails], out[heads]], dim=1).shape
    return


@app.cell
def _(torch):
    ex = torch.Tensor([[1,1,1,1],
                  [2,2,2,2],
                  [3,3,3,3],
                  [4,4,4,4],
                  [5,5,5,5],
                 ])
    return (ex,)


@app.cell
def _(ex):
    ex
    return


@app.cell
def _(ex):
    ex[[0, 2, 0]]
    return


@app.cell
def _():
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
