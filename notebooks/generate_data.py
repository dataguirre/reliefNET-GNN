import itertools
import math
import random
from collections import defaultdict
from math import comb
from typing import Callable
import pandas as pd
import networkx as nx
import numpy as np

from vulnerability_networks.algorithms.functionality_based import global_efficiency, number_independent_paths
from vulnerability_networks.algorithms.rank_links import rank_links


def generate_disruption_scenarios(edges: list, percentage_links: float, max_scenarios: int = 2_000_000):
    """
    Generate disruption scenarios by selecting combinations of edges.

    Args:
        edges: List of edges in the network
        percentage_links: Percentage of links to consider for disruption (0.0 to 1.0)
        max_scenarios: Maximum number of scenarios to generate

    Returns:
        Generator yielding disruption scenarios
    """
    R = int(len(edges) * percentage_links)

    # Calculate total number of combinations for all r values
    total_combinations = 0
    combinations_by_r = {}

    for r in range(1, R + 1):
        try:
            n_combinations = comb(len(edges), r)
            combinations_by_r[r] = n_combinations
            total_combinations += n_combinations
        except OverflowError:
            # For extremely large combinations
            combinations_by_r[r] = float("inf")
            total_combinations = float("inf")
            break

    # If total combinations don't exceed max_scenarios, generate all combinations
    if total_combinations <= max_scenarios:
        for r in range(1, R + 1):
            yield from itertools.combinations(edges, r)
        return

    # Always include all single-edge failures first
    yield from itertools.combinations(edges, 1)
    scenarios_generated = combinations_by_r.get(1, len(edges))

    # Distribute remaining scenarios across r = 2 to R
    if scenarios_generated >= max_scenarios:
        return

    remaining_scenarios = max_scenarios - scenarios_generated

    # Allocate remaining scenarios proportionally across r values
    for r in range(2, R + 1):
        if remaining_scenarios <= 0:
            break

        n_combinations = combinations_by_r.get(r, 0)

        if n_combinations <= remaining_scenarios:
            # If we can generate all combinations for this r, do so
            yield from itertools.combinations(edges, r)
            remaining_scenarios -= n_combinations
        else:
            # Otherwise, sample from the combinations
            if n_combinations > 10000:  # For very large combination spaces, use sampling
                edge_indices = list(range(len(edges)))
                sampled_indices_sets = set()  # To avoid duplicates

                while len(sampled_indices_sets) < remaining_scenarios:
                    sampled_indices = tuple(sorted(random.sample(edge_indices, r)))
                    if sampled_indices not in sampled_indices_sets:
                        sampled_indices_sets.add(sampled_indices)
                        yield tuple(edges[i] for i in sampled_indices)
            else:
                # For smaller spaces, generate combinations with a limit
                for i, combo in enumerate(itertools.combinations(edges, r)):
                    if i >= remaining_scenarios:
                        break
                    yield combo

            break  # We've used up all remaining scenarios

def get_total_combinations(edges: list, percentage_links: float):
    R = int(len(edges) * percentage_links)
    print("Total de combinaciones a evaluar:", sum(math.comb(len(edges), r) for r in range(1, R + 1)))



#%%
import random

def assign_source_terminal(elements, num_sources, num_terminals):
    """
    Randomly assign elements as either 'source' or 'terminal'
    
    Args:
        elements: List of elements (can be any type)
        num_sources: Number of elements to mark as 'source'
        num_terminals: Number of elements to mark as 'terminal'
    
    Returns:
        Dictionary mapping each element to either 'source' or 'terminal'
    """
    # Check if we have enough elements
    if num_sources + num_terminals > len(elements):
        raise ValueError(f"Cannot assign {num_sources + num_terminals} roles when only {len(elements)} elements exist")
    
    # Randomly select indices for sources and terminals
    selected_indices = random.sample(range(len(elements)), num_sources + num_terminals)
    source_indices = selected_indices[:num_sources]
    terminal_indices = selected_indices[num_sources:num_sources + num_terminals]
    
    # Create the dictionary assignment
    assignment = {}
    for i in source_indices:
        assignment[elements[i]] = "source"
    for i in terminal_indices:
        assignment[elements[i]] = "terminal"
    
    return assignment

def assign_realistic_weights(G, weight_type="travel_time", **kwargs):
    """
    Assign realistic weights to a transportation network
    
    Args:
        G: NetworkX directed graph
        weight_type: Type of weight to assign (travel_time, distance, cost, multi)
        kwargs: Additional parameters for specific weight types
    
    Returns:
        Graph with weights assigned to edges
    """
    # Check if it's a geometric graph with positions
    has_positions = all('pos' in G.nodes[n] for n in G.nodes())
    
    if weight_type == "travel_time":
        # Parameters
        base_speed = kwargs.get('base_speed', 60)  # km/h
        speed_variation = kwargs.get('speed_variation', 0.3)  # ±30%
        
        for u, v in G.edges():
            # Calculate distance (if positions available)
            if has_positions:
                pos_u = G.nodes[u]['pos']
                pos_v = G.nodes[v]['pos']
                distance = ((pos_u[0] - pos_v[0])**2 + (pos_u[1] - pos_v[1])**2)**0.5
                distance *= 10  # Scale factor to convert to km
            else:
                distance = 1.0  # Default distance
            
            # Randomize speed for this road segment
            speed = base_speed * (1 + random.uniform(-speed_variation, speed_variation))
            
            # Travel time in minutes
            G[u][v]['weight'] = distance / speed * 60
    
    elif weight_type == "distance":
        # Simple physical distance
        for u, v in G.edges():
            if has_positions:
                pos_u = G.nodes[u]['pos']
                pos_v = G.nodes[v]['pos']
                distance = ((pos_u[0] - pos_v[0])**2 + (pos_u[1] - pos_v[1])**2)**0.5
                distance *= 10  # Scale factor
            else:
                # Use degree-based importance for abstract networks
                avg_degree = (G.degree[u] + G.degree[v]) / 2
                distance = 10 / (avg_degree + 1)  # Important connections are shorter
                
            G[u][v]['weight'] = distance
    
    elif weight_type == "cost":
        # Economic cost (base + distance component)
        base_cost = kwargs.get('base_cost', 1.0)
        distance_cost = kwargs.get('distance_cost', 0.1)
        
        for u, v in G.edges():
            if has_positions:
                pos_u = G.nodes[u]['pos']
                pos_v = G.nodes[v]['pos']
                distance = ((pos_u[0] - pos_v[0])**2 + (pos_u[1] - pos_v[1])**2)**0.5 * 10
            else:
                distance = 1.0
                
            G[u][v]['weight'] = base_cost + distance * distance_cost
    
    elif weight_type == "multi":
        # Multi-criteria weight combining time, distance and cost
        time_factor = kwargs.get('time_factor', 0.6)
        distance_factor = kwargs.get('distance_factor', 0.3)
        cost_factor = kwargs.get('cost_factor', 0.1)
        
        # First calculate individual components
        assign_realistic_weights(G, "travel_time")
        travel_times = {(u, v): G[u][v]['weight'] for u, v in G.edges()}
        
        assign_realistic_weights(G, "distance")
        distances = {(u, v): G[u][v]['weight'] for u, v in G.edges()}
        
        assign_realistic_weights(G, "cost")
        costs = {(u, v): G[u][v]['weight'] for u, v in G.edges()}
        
        # Normalize each component
        max_time = max(travel_times.values())
        max_dist = max(distances.values())
        max_cost = max(costs.values())
        
        for u, v in G.edges():
            norm_time = travel_times[(u, v)] / max_time
            norm_dist = distances[(u, v)] / max_dist
            norm_cost = costs[(u, v)] / max_cost
            
            G[u][v]['weight'] = (
                time_factor * norm_time + 
                distance_factor * norm_dist + 
                cost_factor * norm_cost
            )
    
    elif weight_type == "random":
        # Simple random weights
        min_weight = kwargs.get('min_weight', 1)
        max_weight = kwargs.get('max_weight', 10)
        
        for u, v in G.edges():
            G[u][v]['weight'] = random.uniform(min_weight, max_weight)
    
    return G

def generate_network(method, size, n_OD = 10, n_weights=10, ratio_sources = 0.15, ratio_terminals = 0.4, **kwargs):
    match method:
        case "grid":
            G = nx.grid_2d_graph(**kwargs)
            mapping = {node: i for i, node in enumerate(G.nodes())}
            G = nx.relabel_nodes(G, mapping)

    G = nx.DiGraph(G)
    N = len(G.nodes)
    rows = []
    for n_od in range(n_OD):
        n_sources = random.randint(1, int(N*ratio_sources))
        n_terminals = random.randint(1, int(N*ratio_terminals))
        assignment_sources_terminals = assign_source_terminal(list(G.nodes), n_sources, n_terminals)
        G2 = G.copy()
        nx.set_node_attributes(G2, assignment_sources_terminals, "profile")
        for n_w in range(n_weights):
            G3 = assign_realistic_weights(G2.copy(), weight_type="multi")
            rows.append((method, size, n_od, n_w, G3))
    df = pd.DataFrame(rows, columns = ["method", "size", "od", "w", "graph"])
    return df



#%%
import networkx as nx
import pandas as pd
import random
import numpy as np
import os
import json


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

def generate_base_graph(topology, size_category, param_set, seed):
    """Generate a base graph with the specified topology"""
    random.seed(seed)
    np.random.seed(seed)
    
    # Size ranges
    if size_category == "small":
        n_nodes = random.randint(10, 100)
    elif size_category == "medium":
        n_nodes = random.randint(100, 500)
    else:  # large
        n_nodes = random.randint(500, 2000)
    
    # Generate graph based on topology
    if topology == "grid":
        # Square grid or rectangular grid
        if param_set == 0:
            m = int(np.sqrt(n_nodes))
            n = m
        else:
            m = max(int(np.sqrt(n_nodes * 0.75)), 2)
            n = max(int(np.sqrt(n_nodes * 1.25)), 2)
        G = nx.grid_2d_graph(m, n)
        mapping = {node: i for i, node in enumerate(G.nodes())}
        G = nx.relabel_nodes(G, mapping)
    
    elif topology == "cycle":
        G = nx.cycle_graph(n_nodes)
    
    elif topology == "star":
        # Star graphs are limited in size
        n_nodes = min(n_nodes, 500)
        G = nx.star_graph(n_nodes - 1)
    
    elif topology == "wheel":
        # For wheel graphs, limit size for practicality
        n_nodes = min(n_nodes, 500)
        G = nx.wheel_graph(n_nodes)
    
    elif topology == "frucht":
        # Fixed size graph
        G = nx.frucht_graph()
    
    elif topology == "heawood":
        # Fixed size graph
        G = nx.heawood_graph()
    
    elif topology == "barbell":
        if param_set == 0:
            m1 = max(int(n_nodes/2), 3)
            m2 = m1
        else:
            m1 = max(int(n_nodes/3), 3)
            m2 = max(int(n_nodes*2/3), 3)
        G = nx.barbell_graph(m1, m2)
    
    elif topology == "watts_strogatz":
        k_values = [4, 6, 8] if param_set < 3 else [3, 5, 7]
        p_values = [0.1, 0.3, 0.5]
        k = k_values[param_set % 3]
        p = p_values[param_set % 3]
        G = nx.watts_strogatz_graph(n_nodes, k, p)
    
    elif topology == "random_geometric":
        radius_values = [0.1, 0.15, 0.2]
        radius = radius_values[param_set % 3]
        G = nx.random_geometric_graph(n_nodes, radius)
    
    elif topology == "scale_free":
        alpha_values = [0.05, 0.1, 0.15]
        beta_values = [0.9, 0.85, 0.8]
        gamma_values = [0.05, 0.05, 0.05]
        alpha = alpha_values[param_set % 3]
        beta = beta_values[param_set % 3]
        gamma = gamma_values[param_set % 3]
        G = nx.scale_free_graph(n_nodes, alpha=alpha, beta=beta, gamma=gamma)
        G = nx.Graph(G)  # Convert to undirected
    
    elif topology == "barabasi_albert":
        m_values = [1, 2, 3]
        m = m_values[param_set % 3]
        G = nx.barabasi_albert_graph(n_nodes, m)
    
    else:
        raise ValueError(f"Unknown topology: {topology}")
    
    # Convert to directed graph for transportation networks
    G = nx.DiGraph(G)

    # Add positions for layouts if not present
    if not all('pos' in G.nodes[n] for n in G.nodes()):
        if topology in ["grid", "random_geometric"]:
            # These already have positions
            pass
        elif topology in ["frucht", "heawood"]:
            pos = nx.spring_layout(G, seed=seed)
            nx.set_node_attributes(G, {n: {'pos': tuple(p)} for n, p in pos.items()})
        else:
            pos = nx.spring_layout(G, seed=seed)
            nx.set_node_attributes(G, {n: {'pos': tuple(p)} for n, p in pos.items()})
    return G

def assign_realistic_weights(G, weight_type="multi", **kwargs):
    """
    Assign realistic weights to a transportation network
    
    Args:
        G: NetworkX directed graph
        weight_type: Type of weight to assign (travel_time, distance, cost, multi)
        kwargs: Additional parameters for specific weight types
    
    Returns:
        Graph with weights assigned to edges
    """
    G = G.copy()
    # Check if it's a geometric graph with positions
    has_positions = all('pos' in G.nodes[n] for n in G.nodes())
    
    if weight_type == "travel_time":
        # Parameters
        base_speed = kwargs.get('base_speed', 60)  # km/h
        speed_variation = kwargs.get('speed_variation', 0.3)  # ±30%
        
        for u, v in G.edges():
            # Calculate distance (if positions available)
            if has_positions:
                pos_u = G.nodes[u]['pos']
                pos_v = G.nodes[v]['pos']
                distance = ((pos_u[0] - pos_v[0])**2 + (pos_u[1] - pos_v[1])**2)**0.5
                distance *= 10  # Scale factor to convert to km
            else:
                distance = 1.0  # Default distance
            
            # Randomize speed for this road segment
            speed = base_speed * (1 + random.uniform(-speed_variation, speed_variation))
            
            # Travel time in minutes
            G[u][v]['weight'] = distance / speed * 60
    
    elif weight_type == "distance":
        # Simple physical distance
        for u, v in G.edges():
            if has_positions:
                pos_u = G.nodes[u]['pos']
                pos_v = G.nodes[v]['pos']
                distance = ((pos_u[0] - pos_v[0])**2 + (pos_u[1] - pos_v[1])**2)**0.5
                distance *= 10  # Scale factor
            else:
                # Use degree-based importance for abstract networks
                avg_degree = (G.degree[u] + G.degree[v]) / 2
                distance = 10 / (avg_degree + 1)  # Important connections are shorter
                
            G[u][v]['weight'] = distance
    
    elif weight_type == "cost":
        # Economic cost (base + distance component)
        base_cost = kwargs.get('base_cost', 1.0)
        distance_cost = kwargs.get('distance_cost', 0.1)
        
        for u, v in G.edges():
            if has_positions:
                pos_u = G.nodes[u]['pos']
                pos_v = G.nodes[v]['pos']
                distance = ((pos_u[0] - pos_v[0])**2 + (pos_u[1] - pos_v[1])**2)**0.5 * 10
            else:
                distance = 1.0
                
            G[u][v]['weight'] = base_cost + distance * distance_cost
    
    elif weight_type == "multi":
        # First calculate individual components
        G_time = assign_realistic_weights(G, "travel_time")
        G_dist = assign_realistic_weights(G, "distance")
        G_cost = assign_realistic_weights(G, "cost")
        
        # Mix the weights
        time_factor = kwargs.get('time_factor', 0.6)
        distance_factor = kwargs.get('distance_factor', 0.3)
        cost_factor = kwargs.get('cost_factor', 0.1)
        
        # Normalize within each type
        time_values = [G_time[u][v]['weight'] for u, v in G_time.edges()]
        dist_values = [G_dist[u][v]['weight'] for u, v in G_dist.edges()]
        cost_values = [G_cost[u][v]['weight'] for u, v in G_cost.edges()]
        
        max_time = max(time_values) if time_values else 1
        max_dist = max(dist_values) if dist_values else 1
        max_cost = max(cost_values) if cost_values else 1
        
        for u, v in G.edges():
            norm_time = G_time[u][v]['weight'] / max_time
            norm_dist = G_dist[u][v]['weight'] / max_dist
            norm_cost = G_cost[u][v]['weight'] / max_cost
            
            # Combined weight
            G[u][v]['weight'] = (
                time_factor * norm_time + 
                distance_factor * norm_dist + 
                cost_factor * norm_cost
            )
    
    elif weight_type == "random":
        # Simple random weights
        min_weight = kwargs.get('min_weight', 1)
        max_weight = kwargs.get('max_weight', 10)
        
        for u, v in G.edges():
            G[u][v]['weight'] = random.uniform(min_weight, max_weight)
    
    return G

def generate_networks_dataset(base_path="data/raw"):
    """Generate and save the complete dataset of networks"""
    os.makedirs(base_path, exist_ok=True)
    
    # Define parameters
    topologies = [
        "grid", "cycle", "star", "wheel", "frucht", "heawood", 
        "barbell", "watts_strogatz", "random_geometric", 
        "scale_free", "barabasi_albert"
    ]
    
    size_categories = ["small", "medium", "large"]  # Add "large" later if needed
    param_variations = 3
    seed_variations = 2
    od_patterns = 5
    weight_patterns = 5
    
    catalog = []
    network_id = 0
    
    for topology in topologies:
        print(f"Generating {topology} networks...")
        
        for size_category in size_categories:
            # Skip some combinations for fixed-size graphs
            if topology in ["frucht", "heawood"] and size_category != "small":
                continue
                
            for param_set in range(param_variations):
                for seed in range(seed_variations):
                    # Generate base graph
                    try:
                        G_base = generate_base_graph(topology, size_category, param_set, seed=seed+42)
                        
                        # For each OD pattern
                        for od_pattern in range(od_patterns):
                            # Assign sources and terminals
                            n_nodes = G_base.number_of_nodes()
                            n_sources = max(1, int(n_nodes * random.uniform(0.05, 0.15)))
                            n_terminals = max(1, int(n_nodes * random.uniform(0.2, 0.4)))
                            
                            assignment = assign_source_terminal(
                                list(G_base.nodes), n_sources, n_terminals
                            )
                            
                            G_with_od = G_base.copy()
                            nx.set_node_attributes(G_with_od, assignment, "profile")
                            
                            # For each weight pattern
                            for weight_pattern in range(weight_patterns):
                                # Apply weights
                                G_final = assign_realistic_weights(
                                    G_with_od, 
                                    weight_type="multi",
                                    time_factor=0.5 + random.uniform(-0.2, 0.2),
                                    distance_factor=0.3 + random.uniform(-0.1, 0.1),
                                    cost_factor=0.2 + random.uniform(-0.1, 0.1)
                                )
                                
                                # Create network directory
                                network_dir = os.path.join(base_path, f"network_{network_id}")
                                os.makedirs(network_dir, exist_ok=True)
                                
                                # Save the graph
                                nx.write_gml(G_final, os.path.join(network_dir, "graph.gml"))
                                # Calculate network properties
                                try:
                                    transitivity = nx.transitivity(nx.Graph(G_final))
                                except:
                                    transitivity = 0
                                    
                                density = nx.density(G_final)
                                
                                # Create metadata
                                sources = [str(n) for n, attr in G_final.nodes(data=True) 
                                           if attr.get("profile") == "source"]
                                terminals = [str(n) for n, attr in G_final.nodes(data=True) 
                                             if attr.get("profile") == "terminal"]
                                
                                metadata = {
                                    "network_id": network_id,
                                    "topology": topology,
                                    "size_category": size_category,
                                    "parameters": {
                                        "param_set": param_set,
                                        "seed": seed+42
                                    },
                                    "nodes": G_final.number_of_nodes(),
                                    "edges": G_final.number_of_edges(),
                                    "sources": sources,
                                    "terminals": terminals,
                                    "od_pattern": od_pattern,
                                    "weight_pattern": weight_pattern,
                                    "density": density,
                                    "transitivity": transitivity,
                                    "path": f"network_{network_id}/"
                                }
                                
                                # Save metadata
                                with open(os.path.join(network_dir, "metadata.json"), 'w') as f:
                                    json.dump(metadata, f, indent=2)
                                
                                # Add to catalog
                                catalog.append(metadata)
                                
                                network_id += 1
                                if network_id % 10 == 0:
                                    print(f"Generated {network_id} networks so far...")
                    
                    except Exception as e:
                        print(f"Error generating {topology} network with param_set={param_set}, seed={seed}: {e}")
                        continue
    
    # Save the catalog
    with open(os.path.join(base_path, "network_catalog.json"), 'w') as f:
        json.dump(catalog, f, indent=2)
    
    print(f"Generated and saved a total of {network_id} networks")
    return network_id


