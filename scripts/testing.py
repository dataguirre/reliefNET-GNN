import joblib
from vulnerability_networks.config import PATH_PROCESSED_DATA, PATH_MODELS, PATH_RAW_DATA, PATH_INTERIM_DATA
from vulnerability_networks.modeling.train import NetworkDataModule, LightningRankEdgeNet
from vulnerability_networks.metrics import map_at_k, ndcg_at_k, kendall_tau_b_corrcoef
from vulnerability_networks.algorithms.rank_links import process_network
from vulnerability_networks.algorithms.functionality_based import global_efficiency, number_independent_paths
import json
import re
import pandas as pd
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import itertools
import timeit
from collections import defaultdict
import random
import math
import numpy as np
#%%

dm_ge = NetworkDataModule(PATH_PROCESSED_DATA / "global_efficiency_old", batch_size=1_000_000_000)
dm_ge.setup()
gnn_ge = LightningRankEdgeNet.load_from_checkpoint(PATH_MODELS/"validate/global_efficiency/version_7/checkpoints/epoch=99-step=2500.ckpt", map_location="cpu")
# gnn_ge = gnn_ge.to("cpu")
test_ge = next(iter(dm_ge.test_dataloader())).to(gnn_ge.device)

dm_ip = NetworkDataModule(PATH_PROCESSED_DATA / "independent_path_old", batch_size=1_000_000_000)
dm_ip.setup()
gnn_ip = LightningRankEdgeNet.load_from_checkpoint(PATH_MODELS/"validate/independent_path/version_3/checkpoints/epoch=99-step=1600.ckpt", map_location="cpu")
# gnn_ip = gnn_ip.to("cpu")
test_ip = next(iter(dm_ip.test_dataloader())).to(gnn_ip.device)


#%%
gnn_ge.eval()
gnn_ip.eval()
start = timeit.default_timer()
# gnn_ge_edge_hat = gnn_ge(test_ge)
y_hat = gnn_ge(test_ge)
y_hat = gnn_ip(test_ge)
end = timeit.default_timer()
print("FINISH", end-start)
#%%
start = timeit.default_timer()

print("MAP_AT_K")
print("GNN_GE with TEST_GE", map_at_k(gnn_ge(test_ge), test_ge.edge_y, index_edges = test_ge.edge_attr_batch).round(decimals=2))#map_at_k(gnn_ge_edge_hat
print("GNN_GE with TEST_IP", map_at_k(gnn_ge(test_ip), test_ip.edge_y, index_edges = test_ip.edge_attr_batch).round(decimals=2))#map_at_k(gnn_ge_edge_hat
print("GNN_IP with TEST_GE", map_at_k(gnn_ip(test_ge), test_ge.edge_y, index_edges = test_ge.edge_attr_batch).round(decimals=2))#map_at_k(gnn_ge_edge_hat
print("GNN_IP with TEST_IP", map_at_k(gnn_ip(test_ip), test_ip.edge_y, index_edges = test_ip.edge_attr_batch).round(decimals=2))#map_at_k(gnn_ge_edge_hat

print("NDCG")
print("GNN_GE with TEST_GE", ndcg_at_k(gnn_ge(test_ge), test_ge.edge_y, index_edges = test_ge.edge_attr_batch).round(decimals=2))#ndcg_at_k(gnn_ge_edge_hat
print("GNN_GE with TEST_IP", ndcg_at_k(gnn_ge(test_ip), test_ip.edge_y, index_edges = test_ip.edge_attr_batch).round(decimals=2))#ndcg_at_k(gnn_ge_edge_hat
print("GNN_IP with TEST_GE", ndcg_at_k(gnn_ip(test_ge), test_ge.edge_y, index_edges = test_ge.edge_attr_batch).round(decimals=2))#ndcg_at_k(gnn_ge_edge_h
print("GNN_IP with TEST_IP", ndcg_at_k(gnn_ip(test_ip), test_ip.edge_y, index_edges = test_ip.edge_attr_batch).round(decimals=2))#ndcg_at_k(gnn_ge_edge_hat

print("KENDALL TAU B")
print("GNN_GE with TEST_GE", kendall_tau_b_corrcoef(gnn_ge(test_ge), test_ge.edge_y, graph_ids = test_ge.edge_attr_batch).round(decimals=2))#kendall_tau_b_corrcoef(gnn_ge_edge_hat
print("GNN_GE with TEST_IP", kendall_tau_b_corrcoef(gnn_ge(test_ip), test_ip.edge_y, graph_ids = test_ip.edge_attr_batch).round(decimals=2))#kendall_tau_b_corrcoef(gnn_ge_edge_hat
print("GNN_IP with TEST_GE", kendall_tau_b_corrcoef(gnn_ip(test_ge), test_ge.edge_y, graph_ids = test_ge.edge_attr_batch).round(decimals=2))#kendall_tau_b_corrcoef(gnn_ge_edge_hat
print("GNN_IP with TEST_IP", kendall_tau_b_corrcoef(gnn_ip(test_ip), test_ip.edge_y, graph_ids = test_ip.edge_attr_batch).round(decimals=2))#map_at
end = timeit.default_timer()
print("FINISH", end-start)


#%%
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
df_scores["number_of_st"] = df_scores["sources"].str.len() + df_scores["terminals"].str.len()

#%%

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
            edge_criticality_scores.append(0)
            # edge_criticality_scores.append(data["criticality_score"])

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

def pyg_to_nx(data):
    G = nx.DiGraph()
    for i in range(data.edge_index.shape[1]):
        u, v = data.edge_index[:, i]
        predicted_gnn_ge = data.gnn_ge[i]
        predicted_gnn_ip = data.gnn_ip[i]
        weight = data.edge_attr[i]

        # Add nodes with profile attribute if they don't exist yet
        for node in [u, v]:
            if node.item() not in G:  # Convert tensor to item for use as node identifier
                if all(data.x[node] == torch.tensor([1, 0, 0])):
                    profile = "source"
                elif all(data.x[node] == torch.tensor([0, 1, 0])):
                    profile = "terminal"
                elif all(data.x[node] == torch.tensor([0, 0, 1])):
                    profile = "regular"

                G.add_node(node.item(), profile=profile)

        # Add edge with attributes
        G.add_edge(u.item(), v.item(), gnn_ge=predicted_gnn_ge.item(), gnn_ip = predicted_gnn_ip.item(),
                   weight=weight.item())
    return G

def create_distribution_plan(G):
    distribution_plan = defaultdict(dict)
    sources, terminals = [], []
    for node, profile in nx.get_node_attributes(G, "profile").items():
        if profile == "source":
            sources.append(node)
        elif profile == "terminal":
            terminals.append(node)
    if not sources or not terminals:
        raise Exception("There must be at least 1 source and 1 terminal")
    for s, t in itertools.product(sources, terminals):
        if not nx.has_path(G, s, t):
            raise Exception("no hay camino entre fuente y terminal")
        distribution_plan[s][t] = random.randint(10, 300)
    return distribution_plan



def gii(G, ranking, distribution_plan, normalize=False):
    sources, terminals = [], []
    for node, profile in nx.get_node_attributes(G, "profile").items():
        if profile == "source":
            sources.append(node)
        elif profile == "terminal":
            terminals.append(node)
    if not sources or not terminals:
        raise Exception("There must be at least 1 source and 1 terminal")

    global_importances = []
    initial_costs = dict(nx.all_pairs_dijkstra_path_length(G))
    G_ = G.copy()
    max_disrupted_cost = 0
    for i, edge in enumerate(ranking, start=1):
        perc = i/len(G.edges)
        n_edges_removed = i
        G_.remove_edge(*edge)
    # for perc in drop_perc:
    #     G_ = G.copy()
    #     n_edges_removed = round(perc*len(G.edges))
    #     G_.remove_edges_from(ranking[:n_edges_removed])
        disrupted_costs = dict(nx.all_pairs_dijkstra_path_length(G_))
        # breakpoint()
        gii_val, max_disrupted_cost = _compute_gii(sources, terminals, distribution_plan, initial_costs,
                                                   disrupted_costs, max_disrupted_cost)
        # if gii_val == math.inf:
        #     break
        if global_importances and gii_val < global_importances[-1]["gii"]:
            raise Exception("ERRRROOOORR")
        global_importances.append({"perc": perc, "edges_removed": n_edges_removed, "gii": gii_val})
    x, y = [], []
    for glob_imp in global_importances:
            x.append(glob_imp["perc"])
            y.append(glob_imp["gii"])
    x = np.array(x)
    y = np.array(y)
    if normalize:
        y = (y-min(y))/(max(y)-min(y))
    result = np.array([x, y], dtype=np.float32)
    return result

def _compute_gii(sources, terminals, distribution_plan, costs, disrupted_costs, max_disrupted_cost):

    
    unsatisfied_demand = 0
    importance_score = 0
    total_demand = 0
    for s, t in itertools.product(sources, terminals):
        total_demand += distribution_plan[s][t]
        # breakpoint()
        if not (s in disrupted_costs and t in disrupted_costs[s]):
             unsatisfied_demand += distribution_plan[s][t]
        elif s in disrupted_costs and t in disrupted_costs[s]:
            importance_score += distribution_plan[s][t]*(disrupted_costs[s][t] - costs[s][t])
            max_disrupted_cost = max(max_disrupted_cost, disrupted_costs[s][t] - costs[s][t])
        else:
            breakpoint()
    alpha = 1.1 * max_disrupted_cost
    # if unsatisfied_demand == total_demand:
    #     return (math.inf, max_disrupted_cost)
    result = (importance_score + alpha*unsatisfied_demand)/total_demand
        
    return result, max_disrupted_cost



#%%

# Create a directed graph
G = nx.DiGraph()

# Add nodes with profile attributes
G.add_node(0, profile="source")
G.add_node(1, profile="regular") 
G.add_node(2, profile="regular")
G.add_node(3, profile="regular")
G.add_node(4, profile="terminal")

# Add edges with weights
G.add_edge(0, 4, weight=5)  # Direct path (shortest initially)
G.add_edge(0, 1, weight=2)  # Part of alternative path 1
G.add_edge(1, 2, weight=2)  # Part of alternative path 1
G.add_edge(2, 4, weight=2)  # Part of alternative path 1
G.add_edge(0, 3, weight=4)  # Part of alternative path 2
G.add_edge(3, 4, weight=5)  # Part of alternative path 2

ranking = [(0,2), (0,1), (1,2)]
distribution_plan = create_distribution_plan(G)

#%%
gii(G, ranking, [], distribution_plan)


#%%
for idx, group in df_scores.query("edges<1000").groupby(["topology", "nodes", "edges"]):
    print(idx, len(group))

#%%
# GII
# GII = {"id": {"1": {"ge":[[(x1, y1), (x2, y2), ...], ...],
#                     "gnn_ge": [[]],
#                    "gnn_ip": [[(x1, y1), (x2, y2)]],
#                    "ip": {},
#               }
#        } # network 17
GII = {}
#.groupby(["topology", "nodes", "edges"]).sample(5)
for _, row in df_scores.query("edges<1000").groupby(["topology", "nodes", "edges"]).sample(5).iterrows():
    # for scenarios in (1, 1000, 5000, 10_000):
    #     pass
    network_id = row["network_id"]
    G = nx.read_gml(row["path"])
    data = convert_nx_to_pyg(G)
    data.gnn_ge = gnn_ge(data)
    data.gnn_ip = gnn_ip(data)
    G_ = pyg_to_nx(data)
    edges_attr = list(G_.edges(data = True))
    rank_gnn_ge = [((u, v), attrs["gnn_ge"]) for u, v, attrs in edges_attr]
    rank_gnn_ge = sorted(rank_gnn_ge, key=lambda x: x[1], reverse=True)
    rank_gnn_ge = [edge for edge, score in rank_gnn_ge]
    rank_gnn_ip = [((u, v), attrs["gnn_ip"]) for u, v, attrs in edges_attr]
    rank_gnn_ip = sorted(rank_gnn_ip, key=lambda x: x[1], reverse=True)
    rank_gnn_ip = [edge for edge, score in rank_gnn_ip]

    distribution_plan = create_distribution_plan(G_)
    gii_gnn_ge = gii(G_.copy(), rank_gnn_ge, distribution_plan)
    gii_gnn_ip = gii(G_.copy(), rank_gnn_ip, distribution_plan)
    GII[network_id] = defaultdict(list)
    GII[network_id]["gnn_ge"] = gii_gnn_ge
    GII[network_id]["gnn_ip"] = gii_gnn_ip
    for max_disruption_scenarios in [1, 1000, 5000, 10000]:
        rank_ge = process_network(G_.copy(), global_efficiency, max_links_in_disruption=0.4,
                                  max_disruption_scenarios=max_disruption_scenarios, workers=16)["link_scores"]
        rank_ge = sorted(rank_ge, key=lambda x: x[1], reverse=True)
        rank_ge = [edge for edge, score in rank_ge]
        rank_ip = process_network(G_.copy(), number_independent_paths, max_links_in_disruption=0.4,
                                  max_disruption_scenarios=max_disruption_scenarios, workers=16)["link_scores"]
        rank_ip = sorted(rank_ip, key=lambda x: x[1], reverse=True)
        rank_ip = [edge for edge, score in rank_ip]
        gii_ge = gii(G_.copy(), rank_ge, distribution_plan)
        gii_ip = gii(G_.copy(), rank_ip, distribution_plan)
        GII[network_id][f"ge_{max_disruption_scenarios}"] = gii_ge
        GII[network_id][f"ip_{max_disruption_scenarios}"] = gii_ip

    # print("NIP:")

# joblib.dump(GII, PATH_PROCESSED_DATA/"GII_V3.pkl")

#%%

GII2 = joblib.load(PATH_PROCESSED_DATA/"GII.pkl")


#%%
from tqdm import tqdm
times = defaultdict(list)

for _, row in tqdm(df_scores.drop_duplicates(subset=["edges"]).sort_values(by="edges").iterrows()):
    edges = row["edges"]
    G = nx.read_gml(row["path"])
    data = convert_nx_to_pyg(G)
    start = timeit.default_timer()
    logits = gnn_ge(data)
    end = timeit.default_timer()
    times["gnn_ge"].append((edges, end-start))

    start = timeit.default_timer()
    logits = gnn_ip(data)
    end = timeit.default_timer()
    times["gnn_ip"].append((edges, end-start))


    start = timeit.default_timer()
    logits = process_network(G, global_efficiency, max_links_in_disruption=0.4, max_disruption_scenarios=1)
    end = timeit.default_timer()
    times["ge"].append((edges, end-start))


    start = timeit.default_timer()
    logits = process_network(G, number_independent_paths, max_links_in_disruption=0.4, max_disruption_scenarios=1)
    end = timeit.default_timer()
    times["ip"].append((edges, end-start))

    start = timeit.default_timer()
    logits = process_network(G, global_efficiency, max_links_in_disruption=0.4, max_disruption_scenarios=1, workers=16)
    end = timeit.default_timer()
    times["ge_parallel"].append((edges, end-start))

    start = timeit.default_timer()
    logits = process_network(G, number_independent_paths, max_links_in_disruption=0.4, max_disruption_scenarios=1, workers=16)
    end = timeit.default_timer()
    times["ip_parallel"].append((edges, end-start))


#%%
from tqdm import tqdm
times_st = defaultdict(list)
G_ = nx.read_gml(df_scores.query("topology=='grid' and nodes==81").iloc[0]["path"])
combs = list(itertools.combinations_with_replacement(range(len(G.nodes)//2), 2))
def assign_source_terminal(nodes: list, n_sources: int, n_terminals: int):
    """Assign sources and terminals to nodes"""
    random.shuffle(nodes)

    sources = nodes[:n_sources]
    remaining_nodes = nodes[n_sources:]
    terminals = remaining_nodes[:n_terminals]

    node_profile = {}
    for node in nodes:
        if node in sources:
            node_profile[node] = "source"
        elif node in terminals:
            node_profile[node] = "terminal"
        else:
            node_profile[node] = "regular"

    return node_profile

for num_s, num_t in tqdm(combs):
    if not num_s or not num_t:
        continue
    G = G_.copy()
    for _, attr in G_.nodes(data=True):
        attr.clear()
    assignment = assign_source_terminal(list(G.nodes), num_s, num_t)
    nx.set_node_attributes(G, assignment, "profile")

    data = convert_nx_to_pyg(G)
    start = timeit.default_timer()
    logits = gnn_ge(data)
    end = timeit.default_timer()
    times_st["gnn_ge"].append((num_s + num_t, end-start))

    start = timeit.default_timer()
    logits = gnn_ip(data)
    end = timeit.default_timer()
    times_st["gnn_ip"].append((num_s + num_t, end-start))


    start = timeit.default_timer()
    logits = process_network(G, global_efficiency, max_links_in_disruption=0.4, max_disruption_scenarios=1)
    end = timeit.default_timer()
    times_st["ge"].append((num_s + num_t, end-start))


    start = timeit.default_timer()
    logits = process_network(G, number_independent_paths, max_links_in_disruption=0.4, max_disruption_scenarios=1)
    end = timeit.default_timer()
    times_st["ip"].append((num_s + num_t, end-start))

    start = timeit.default_timer()
    logits = process_network(G, global_efficiency, max_links_in_disruption=0.4, max_disruption_scenarios=1, workers=16)
    end = timeit.default_timer()
    times_st["ge_parallel"].append((num_s + num_t, end-start))

    start = timeit.default_timer()
    logits = process_network(G, number_independent_paths, max_links_in_disruption=0.4, max_disruption_scenarios=1, workers=16)
    end = timeit.default_timer()
    times_st["ip_parallel"].append((num_s + num_t, end-start))



#%%
joblib.dump(times_st, PATH_PROCESSED_DATA/"times_st.pkl")


#%%
from collections import defaultdict

# Correct way to create a nested defaultdict
times_by_scenario = defaultdict(lambda: defaultdict(list))

for _, row in (df_scores.query("edges<30").sample(frac=1)).iterrows():
    G = nx.read_gml(row["path"])
    data = convert_nx_to_pyg(G)

    for max_disruption_scenarios in [1000, 5000, 10_000]:
        start = timeit.default_timer()
        logits = gnn_ge(data)
        end = timeit.default_timer()
        times_by_scenario[max_disruption_scenarios]["gnn_ge"].append(end-start)

        start = timeit.default_timer()
        logits = gnn_ip(data)
        end = timeit.default_timer()
        times_by_scenario[max_disruption_scenarios]["gnn_ip"].append(end-start)


        start = timeit.default_timer()
        logits = process_network(G, global_efficiency, max_links_in_disruption=0.4, max_disruption_scenarios=max_disruption_scenarios)
        end = timeit.default_timer()
        times_by_scenario[max_disruption_scenarios]["ge"].append(end-start)


        start = timeit.default_timer()
        logits = process_network(G, number_independent_paths, max_links_in_disruption=0.4, max_disruption_scenarios=max_disruption_scenarios)
        end = timeit.default_timer()
        times_by_scenario[max_disruption_scenarios]["ip"].append(end-start)

        start = timeit.default_timer()
        logits = process_network(G, global_efficiency, max_links_in_disruption=0.4, max_disruption_scenarios=max_disruption_scenarios, workers=16)
        end = timeit.default_timer()
        times_by_scenario[max_disruption_scenarios]["ge_parallel"].append(end-start)

        start = timeit.default_timer()
        logits = process_network(G, number_independent_paths, max_links_in_disruption=0.4, max_disruption_scenarios=max_disruption_scenarios, workers=16)
        end = timeit.default_timer()
        times_by_scenario[max_disruption_scenarios]["ip_parallel"].append(end-start)


#%%

joblib.dump(dict(times_by_scenario), PATH_PROCESSED_DATA/"times_by_scenario.pkl")
