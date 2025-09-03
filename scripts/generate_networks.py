"""
Runnable script to reproduce dataset generation (Algorithm 1 from paper). It saves the generated networks
in the data/raw/{dataset} folder.

    python generate_networks.py --dataset example \
                                --topologies grid,cycle,wheel,star,watts_strogatz \
                                --min_nodes 20 \
                                --max_nodes 100 \
                                --n_samples_per_topology 10 \
                                --n_st_patterns 8 \
                                --n_weight_patterns 8\
                                --seed 45

"""

import argparse
import json
import os
import random

import networkx as nx
import numpy as np
from loguru import logger

from reliefnet_gnn.config import PATH_DATA_RAW


def generate_networks_dataset(
    path_dataset: str,
    topologies: list[str],
    min_nodes: int,
    max_nodes: int,
    n_samples_per_topology: int,
    n_st_patterns: int,
    n_weight_patterns: int,
) -> None:
    """Algorithm 1: Dataset Generation

    Parameters
    ----------
    dataset : str
        name of the dataset to be saved in raw data
    topologies : list[str]
        list of topologies to generate
    min_nodes : int
        minimum number of nodes
    max_nodes : int
        maximum number of nodes
    n_samples_per_topology : int
        number of samples per topology
    n_patterns_st : int
        number of source-terminal patterns
    n_patterns_weights : int
        number of weight patterns

    """

    network_id = 0
    for i, topology in enumerate(topologies, start=1):
        logger.info(f"Generating {topology} networks... [{i}/{len(topologies)}]")
        for _ in range(n_samples_per_topology):
            nodes = random.randint(min_nodes, max_nodes)
            base_graph = create_backbone_network(topology, nodes)
            for _ in range(n_st_patterns):
                G_st = assign_source_terminal_pattern(base_graph.copy())
                for _ in range(n_weight_patterns):
                    G_st_w = assign_weight_pattern(
                        G_st.copy(),
                        time_factor=0.5 + random.uniform(-0.2, 0.2),
                        distance_factor=0.3 + random.uniform(-0.1, 0.1),
                        cost_factor=0.2 + random.uniform(-0.1, 0.1),
                    )
                    add_graph_to_dataset(path_dataset, network_id, G_st_w, topology)
                    network_id += 1


def create_backbone_network(topology: str, n_nodes: int) -> nx.DiGraph:
    """Generate a base graph with the specified topology"""
    match topology:
        # regular structures
        case "grid":
            # Square grid or rectangular grid
            if random.random() < 0.33:
                m = int(np.sqrt(n_nodes))
                n = m
            else:
                m = max(int(np.sqrt(n_nodes * 0.75)), 2)
                n = max(int(np.sqrt(n_nodes * 1.25)), 2)
            G = nx.grid_2d_graph(m, n)
            mapping = {node: i for i, node in enumerate(G.nodes())}
            G = nx.relabel_nodes(G, mapping)

        case "cycle":
            G = nx.cycle_graph(n_nodes)
        case "wheel":
            G = nx.wheel_graph(n_nodes)
        case "star":
            G = nx.star_graph(n_nodes)

        # complex network models
        case "watts_strogatz":
            k_values = [3, 4, 5, 6, 7, 8]
            p_values = [0.1, 0.3, 0.5]
            k = random.choice(k_values)
            p = random.choice(p_values)
            G = nx.watts_strogatz_graph(n_nodes, k, p)
        case "random_geometric":
            radius_values = [0.1, 0.15, 0.2]
            radius = random.choice(radius_values)
            G = nx.random_geometric_graph(n_nodes, radius)
        case "scale_free":
            alpha_values = [0.05, 0.1, 0.15]
            beta_values = [0.9, 0.85, 0.8]
            alpha = random.choice(alpha_values)
            beta = random.choice(beta_values)
            gamma = 0.05
            G = nx.scale_free_graph(n_nodes, alpha=alpha, beta=beta, gamma=gamma)
            G = nx.Graph(G)
        case "barabasi_albert":
            m_values = [1, 2, 3]
            m = random.choice(m_values)
            G = nx.barabasi_albert_graph(n_nodes, m)

        # specialized networks
        case "frucht":
            G = nx.frucht_graph()
        case "heawood":
            G = nx.heawood_graph()
        case "barbell":
            if random.random() < 0.33:
                m1 = max(int(n_nodes / 2), 3)
                m2 = m1
            else:
                m1 = max(int(n_nodes / 3), 3)
                m2 = max(int(n_nodes * 2 / 3), 3)
            G = nx.barbell_graph(m1, m2)

    # Convert to directed graph for transportation networks
    G = nx.DiGraph(G)

    # Add positions for layouts if not present
    if not all("pos" in G.nodes[n] for n in G.nodes()):
        if topology in ["grid", "random_geometric"]:
            # These already have positions
            pass
        elif topology in ["frucht", "heawood"]:
            pos = nx.spring_layout(G)
            nx.set_node_attributes(G, {n: {"pos": tuple(p)} for n, p in pos.items()})
        else:
            pos = nx.spring_layout(G)
            nx.set_node_attributes(G, {n: {"pos": tuple(p)} for n, p in pos.items()})
    return G


def assign_source_terminal_pattern(graph: nx.DiGraph) -> nx.DiGraph:
    """Assign source-tarminal pattern to graph nodes."""
    nodes = list(graph.nodes)
    random.shuffle(nodes)

    # Centralized distribution: 5-10% source nodes serving 30-40% terminals
    # Decentralized distribution: 15-20% sources serving 20-30% terminals
    distribution_type = random.choice(["centralized", "decentralized"])
    if distribution_type == "centralized":
        n_sources = max(1, int(len(nodes) * random.uniform(0.05, 0.10)))
        n_terminals = max(1, int(len(nodes) * random.uniform(0.3, 0.4)))

    else:
        n_sources = max(1, int(len(nodes) * random.uniform(0.15, 0.20)))
        n_terminals = max(1, int(len(nodes) * random.uniform(0.2, 0.3)))

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
    nx.set_node_attributes(graph, node_profile, "profile")
    return graph


def assign_weight_pattern(G, weight_type="travel_time", **kwargs) -> nx.DiGraph:
    """
    Assign realistic weights to a transportation network
    """
    # Check if it's a geometric graph with positions
    has_positions = all("pos" in G.nodes[n] for n in G.nodes())

    if weight_type == "travel_time":
        # Parameters
        base_speed = kwargs.get("base_speed", 60)  # km/h
        speed_variation = kwargs.get("speed_variation", 0.3)  # ±30%

        for u, v in G.edges():
            # Calculate distance (if positions available)
            if has_positions:
                pos_u = G.nodes[u]["pos"]
                pos_v = G.nodes[v]["pos"]
                distance = ((pos_u[0] - pos_v[0]) ** 2 + (pos_u[1] - pos_v[1]) ** 2) ** 0.5
                distance *= 10  # Scale factor to convert to km
            else:
                distance = 1.0  # Default distance

            # Randomize speed for this road segment
            speed = base_speed * (1 + random.uniform(-speed_variation, speed_variation))

            # Travel time in minutes
            G[u][v]["weight"] = distance / speed * 60

    elif weight_type == "distance":
        # Simple physical distance
        for u, v in G.edges():
            if has_positions:
                pos_u = G.nodes[u]["pos"]
                pos_v = G.nodes[v]["pos"]
                distance = ((pos_u[0] - pos_v[0]) ** 2 + (pos_u[1] - pos_v[1]) ** 2) ** 0.5
                distance *= 10  # Scale factor
            else:
                # Use degree-based importance for abstract networks
                avg_degree = (G.degree[u] + G.degree[v]) / 2
                distance = 10 / (avg_degree + 1)  # Important connections are shorter

            G[u][v]["weight"] = distance

    elif weight_type == "cost":
        # Economic cost (base + distance component)
        base_cost = kwargs.get("base_cost", 1.0)
        distance_cost = kwargs.get("distance_cost", 0.1)

        for u, v in G.edges():
            if has_positions:
                pos_u = G.nodes[u]["pos"]
                pos_v = G.nodes[v]["pos"]
                distance = ((pos_u[0] - pos_v[0]) ** 2 + (pos_u[1] - pos_v[1]) ** 2) ** 0.5 * 10
            else:
                distance = 1.0

            G[u][v]["weight"] = base_cost + distance * distance_cost

    elif weight_type == "multi":
        # Multi-criteria weight combining time, distance and cost
        time_factor = kwargs.get("time_factor", 0.6)
        distance_factor = kwargs.get("distance_factor", 0.3)
        cost_factor = kwargs.get("cost_factor", 0.1)

        # First calculate individual components
        assign_weight_pattern(G, "travel_time")
        travel_times = {(u, v): G[u][v]["weight"] for u, v in G.edges()}

        assign_weight_pattern(G, "distance")
        distances = {(u, v): G[u][v]["weight"] for u, v in G.edges()}

        assign_weight_pattern(G, "cost")
        costs = {(u, v): G[u][v]["weight"] for u, v in G.edges()}

        # Normalize each component
        max_time = max(travel_times.values())
        max_dist = max(distances.values())
        max_cost = max(costs.values())

        for u, v in G.edges():
            norm_time = travel_times[(u, v)] / max_time
            norm_dist = distances[(u, v)] / max_dist
            norm_cost = costs[(u, v)] / max_cost

            G[u][v]["weight"] = time_factor * norm_time + distance_factor * norm_dist + cost_factor * norm_cost

    elif weight_type == "random":
        # Simple random weights
        min_weight = kwargs.get("min_weight", 1)
        max_weight = kwargs.get("max_weight", 10)

        for u, v in G.edges():
            G[u][v]["weight"] = random.uniform(min_weight, max_weight)

    return G


def add_graph_to_dataset(path_dataset: str, network_id: int, graph: nx.DiGraph, topology: str) -> None:
    """Add configured graph to the dataset (as .gml format) with metadata"""
    # Create network directory
    network_dir = os.path.join(path_dataset, f"network_{network_id}")
    os.makedirs(network_dir, exist_ok=True)

    # Save the graph
    nx.write_gml(graph, os.path.join(network_dir, "graph.gml"))
    # Calculate network properties
    try:
        transitivity = nx.transitivity(nx.Graph(graph))
    except:
        transitivity = 0

    density = nx.density(graph)

    # Create metadata
    sources = [str(n) for n, attr in graph.nodes(data=True) if attr.get("profile") == "source"]
    terminals = [str(n) for n, attr in graph.nodes(data=True) if attr.get("profile") == "terminal"]

    metadata = {
        "network_id": network_id,
        "topology": topology,
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "sources": sources,
        "terminals": terminals,
        "density": density,
        "transitivity": transitivity,
        "path": f"network_{network_id}/",
    }

    # Save metadata
    with open(os.path.join(network_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Generate network datasets for ReliefNet GNN",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset to be saved in raw data")

    parser.add_argument(
        "--topologies",
        type=str,
        required=True,
        help="Comma-separated list of topologies to generate (e.g., 'grid,cycle,wheel,star,watts_strogatz')",
    )

    parser.add_argument("--min_nodes", type=int, default=20, help="Minimum number of nodes")

    parser.add_argument("--max_nodes", type=int, default=100, help="Maximum number of nodes")

    parser.add_argument("--n_samples_per_topology", type=int, default=10, help="Number of samples per topology")

    parser.add_argument("--n_st_patterns", type=int, default=8, help="Number of source-terminal patterns")

    parser.add_argument("--n_weight_patterns", type=int, default=8, help="Number of weight patterns")

    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")

    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()

    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    # Parse topologies from comma-separated string
    topologies = [t.strip() for t in args.topologies.split(",")]

    # Create dataset path
    path_dataset = os.path.join(PATH_DATA_RAW, args.dataset)
    os.makedirs(path_dataset, exist_ok=True)

    logger.info(f"Starting dataset generation: {args.dataset}")
    logger.info(f"Topologies: {topologies}")
    logger.info(f"Node range: {args.min_nodes} - {args.max_nodes}")
    logger.info(f"Samples per topology: {args.n_samples_per_topology}")
    logger.info(f"ST patterns: {args.n_st_patterns}")
    logger.info(f"Weight patterns: {args.n_weight_patterns}")
    logger.info(f"Output path: {path_dataset}")

    # Generate the dataset
    generate_networks_dataset(
        path_dataset=path_dataset,
        topologies=topologies,
        min_nodes=args.min_nodes,
        max_nodes=args.max_nodes,
        n_samples_per_topology=args.n_samples_per_topology,
        n_st_patterns=args.n_st_patterns,
        n_weight_patterns=args.n_weight_patterns,
    )

    logger.info("Dataset generation completed!")
