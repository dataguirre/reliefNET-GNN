import json
import pandas as pd
import concurrent.futures
from pathlib import Path
import networkx as nx
import numpy as np
from functools import partial
from tqdm import tqdm
import time
from vulnerability_networks.algorithms.functionality_based import global_efficiency
from vulnerability_networks.algorithms.utils import generate_disruption_scenarios
import math

def get_total_combinations(edges: list, percentage_links: float):
    R = int(len(edges) * percentage_links)
    total = sum(math.comb(len(edges), r) for r in range(1, R + 1))
    return total

def evaluate_disruption(disruption, G, sources, terminals, accessibility_index, P_0):
    """Evaluate a single disruption scenario"""
    G_disrupted = G.copy()

    for edge in disruption:
        G_disrupted.remove_edge(*edge)
    try:
        P_disruption = accessibility_index(G_disrupted, sources, terminals)
        delta = P_0 - P_disruption
        # Return the disruption and its impact
        return disruption, delta
    except Exception as e:
        # Return None if calculation fails
        return disruption, None

def parallel_rank_links(G, disruptions, accessibility_index, max_workers=None):
    """Parallelized version of rank_links"""
    assert all_edges_have_attr(G, "weight")
    
    # Identify sources and terminals
    sources, terminals = [], []
    for node, profile in nx.get_node_attributes(G, "profile").items():
        if profile == "source":
            sources.append(node)
        elif profile == "terminal":
            terminals.append(node)
    if not sources or not terminals:
        raise Exception("There must be at least 1 source and 1 terminal")
    
    # Calculate initial network performance
    P_0 = accessibility_index(G, sources, terminals)
    
    # Initialize result container
    link_performance = {}
    
    # Process disruptions in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Create a partial function for parallel execution
        evaluate_func = partial(evaluate_disruption, G=G, sources=sources, 
                               terminals=terminals, accessibility_index=accessibility_index, P_0=P_0)

        size = min(get_total_combinations(list(G.edges), 0.4), 1_000_000)
        # Use tqdm to show a progress bar
        results = list(tqdm(executor.map(evaluate_func, disruptions), total=size))
        
    # Process results
    for disruption, delta in results:
        if delta is not None:
            for edge in disruption:
                link_performance[edge] = link_performance.get(edge, 0) + delta * (1 / len(disruption))
    
    # Sort and return results
    return sorted(link_performance.items(), key=lambda x: x[1], reverse=True)

def all_edges_have_attr(G: nx.Graph, attr: str) -> bool:
    """Check if all edges in the graph have a specific attribute."""
    return all(attr in data for _, _, data in G.edges(data=True))


# Main execution for testing
# List to store all results
results = []

# For timing comparison
start_time = time.time()

# Define paths to your network files
path_raw =  Path("/home/danielaguirre/University/vulnerability-networks/data/raw")


with open(path_raw/"network_catalog.json", "r") as f:
    catalog = json.load(f)

df = pd.DataFrame(catalog)
df["path"] = df["path"].apply(lambda x: path_raw/x/'graph.gml')

# For demonstration, process only a subset of networks
df_example = df.sort_values(by="nodes")[:10]

for path in df_example["path"]:
    print(f"Processing network: {path}")
    try:
        G = nx.read_gml(path)
        disruptions = generate_disruption_scenarios(list(G.edges), max_links_in_disruption=0.4, max_scenarios=1_000_000)
        size = min(get_total_combinations(list(G.edges), 0.4), 1_000_000)
        print(f"Network has {len(G.nodes)} nodes and {len(G.edges)} edges")
        print(f"Testing with {size} disruption scenarios")

        # Call the parallel ranking function
        critical_links = parallel_rank_links(G, disruptions, global_efficiency, max_workers=16)

        # Store results
        results.append({
            'network_path': str(path),
            'nodes': len(G.nodes),
            'edges': len(G.edges),
            'critical_links': critical_links[:10]  # Top 10 most critical links
        })

        print(f"Top 5 critical links: {critical_links[:5]}")
        print("-" * 50)
        break

    except Exception as e:
        print(f"Error processing network {path}: {e}")

print(f"Total processing time: {time.time() - start_time:.2f} seconds")
print(f"Processed {len(results)} networks")

# Now results contains the critical links for each network
print("\nSummary of results:")
for i, result in enumerate(results):
    print(f"Network {i+1}: {result['network_path']}")
    print(f"  Top 3 critical links: {result['critical_links'][:3]}")
