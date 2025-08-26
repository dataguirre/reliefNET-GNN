"""
Runnable script to reproduce dataset generation (Algorithm 1 from paper)

    python generate_networks.py --dataset example \
                                --topologies grid,cycle,wheel,star,watts_strogatz \
                                --min_nodes 20 \
                                --max_nodes 100 \
                                --n_samples_per_topology 10 \
                                --n_st_patterns 8 \
                                --n_weight_patterns 8

"""

import random

import networkx as nx
from loguru import logger

from reliefnet_gnn.config import PATH_DATA_RAW


def generate_networks_dataset(
    dataset: str,
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
            st_patterns = generate_random_source_target_patterns(base_graph, n_patterns=n_st_patterns)
            weight_patterns = generate_random_weight_patterns(base_graph, n_patterns=n_weight_patterns)
            for st in st_patterns:
                G_st = assign_source_terminal_pattern(base_graph.copy(), st)
                for weight in weight_patterns:
                    G_st_w = assign_weight_pattern(G_st.copy(), weight)
                    add_graph_to_dataset(dataset, network_id, G_st_w, topology, st, weight, st)
                    network_id += 1


def create_backbone_network(topology: str, n_nodes: int) -> nx.DiGraph:
    """Generate a base graph with the specified topology"""
    match topology:
        # regular structures
        case "grid":
            pass
        case "cycle":
            pass
        case "wheel":
            pass
        case "star":
            pass
        # complex network models
        case "watts_strogatz":
            pass
        case "random_geometric":
            pass
        case "scale_free":
            pass
        case "barabasi_albert":
            pass
        # specialized networks
        case "frucht":
            pass
        case "heawood":
            pass
        case "barbell":
            pass

    return None


def generate_random_source_target_patterns(graph, n_patterns: int) -> list[dict[int, str]]:
    """Generate random source-target node patterns."""

    pass


def generate_random_weight_patterns(graph, n_patterns: int):
    """Generate random edge weight patterns."""
    # Implementation for generating weight patterns
    pass


def assign_source_terminal_pattern(graph, pattern):
    """Assign source-tarminal pattern to graph nodes."""
    # Implementation for applying ST pattern
    pass


def assign_weight_pattern(graph, pattern):
    """Assign weight pattern to graph edges."""
    # Implementation for applying weight pattern
    pass


def add_graph_to_dataset(
    graph,
):
    """Add configured graph to the dataset with metadata."""
    # Implementation for dataset storage
    pass


if __name__ == "__main__":
    generate_networks_dataset()
