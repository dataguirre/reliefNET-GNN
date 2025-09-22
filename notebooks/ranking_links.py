import concurrent.futures
import json
from pathlib import Path

import networkx as nx
import pandas as pd
from tqdm import tqdm

from vulnerability_networks.algorithms.functionality_based import global_efficiency
from vulnerability_networks.algorithms.rank_links import parallel_rank_links, rank_links
from vulnerability_networks.algorithms.utils import generate_disruption_scenarios

#%%

path_raw =  Path("/home/danielaguirre/University/vulnerability-networks/data/raw")


with open(path_raw/"network_catalog.json", "r") as f:
    catalog = json.load(f)

df = pd.DataFrame(catalog)
df["path"] = df["path"].apply(lambda x: path_raw/x/'graph.gml')
# df["graph"] = df["path"].apply(lambda path: nx.read_gml(path))


#%%
# import re

# for path in df["path"]:
#     with open(path, "r") as f:
#         content = f.read()
#         fixed_content = re.sub(r'NP\.FLOAT64\(([-0-9.eE+-]+)\)', r'\1', content)
#         with open(path, "w") as f:
#             f.write(fixed_content)


#%%
# import os

# for path in df["path"]:
#     os.remove(path.parent/"graph_fixed.gml")
    

#%%
# De forma normal, tiempo:
df_example = df.sort_values(by="nodes")[:100]
critical_links = []
for path in tqdm(df_example["path"]):
    G = nx.read_gml(path)
    disruptions = generate_disruption_scenarios(list(G.edges), 0.4, max_scenarios=1_000_000)
    ranking = rank_links(G, disruptions, global_efficiency)
    critical_links.append(ranking)


#%%

def process_network(path):
    """Process a single network file"""
    G = nx.read_gml(path)
    disruptions = generate_disruption_scenarios(list(G.edges), 0.4, max_scenarios=1_000_000)
    critical_links = parallel_rank_links(G, disruptions, global_efficiency)
    return path, critical_links

# Main execution
results = []
# Use ProcessPoolExecutor for CPU-bound tasks
with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
    futures = []
    for path in Path("raw_data").rglob("*.gml"):
        # Submit each network processing job to the executor
        futures.append(executor.submit(process_network, path))

    # Process results as they complete
    for future in concurrent.futures.as_completed(futures):
        try:
            result = future.result()
            # Handle/save the result
            results.append(result)
        except Exception as e:
            print(f"Error processing network: {e}")

#%%
for path in Path(raw_data).rglob("*.gml"):
    G = nx.read_gml(path)
    disruptions = generate_disruption_scenarios(list(G.edges), 0.4, max_scenarios=1_000_000)
    critical_links = rank_links(G, disruptions, global_efficiency)
