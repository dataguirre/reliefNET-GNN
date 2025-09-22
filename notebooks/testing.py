import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import joblib
    from vulnerability_networks.config import PATH_PROCESSED_DATA, PATH_FIGURES, PATH_DATA
    from collections import defaultdict
    import numpy as np
    import matplotlib.pyplot as plt
    import scienceplots
    from matplotlib.ticker import PercentFormatter
    plt.style.use(['science','ieee'])
    rc_fonts = {
        "font.size": 9,
        #'figure.figsize': (4, 3),
    }
    plt.rcParams.update(rc_fonts)

    return (
        PATH_DATA,
        PATH_FIGURES,
        PATH_PROCESSED_DATA,
        PercentFormatter,
        defaultdict,
        joblib,
        np,
        plt,
    )


@app.cell
def _(PATH_PROCESSED_DATA, defaultdict, joblib, np):
    GII2 = joblib.load(PATH_PROCESSED_DATA/"GII_V3.pkl")

    #%%
    all_x_points = set([0])

    for gii_dicc in GII2.values():
        for method, values in gii_dicc.items():
            all_x_points = all_x_points.union(values[0, :].tolist())
    all_x_points =np.array(sorted(all_x_points))

    results_by_method = defaultdict(list)
    for net_id, gii_dicc in GII2.items():
        execute = True
        for method, values in gii_dicc.items():
            x, y = values
            x, y = np.hstack([[0], x]), np.hstack([[0], y])
            if y.max() == y.min():
                execute = False
        if not execute:
            continue
        for method, values in gii_dicc.items():
            x, y = values
            x, y = np.hstack([[0], x]), np.hstack([[0], y])
            y_norm = (y - y.min()) / (y.max() - y.min())
            new_y = np.interp(all_x_points, x, y_norm)
            gii_interp = np.array([all_x_points, new_y])
            results_by_method[method].append(gii_interp)
    return (results_by_method,)


@app.cell
def _():
    # GII2[2229]["gnn_ge"]
    return


@app.cell
def _():
    show_until = 1922
    return (show_until,)


@app.cell
def _(np, results_by_method, show_until):
    gnn_ge = np.array(results_by_method["gnn_ge"]).transpose(1, 0, 2).mean(axis=1)[:, :show_until]
    gnn_ip = np.array(results_by_method["gnn_ip"]).transpose(1, 0, 2).mean(axis=1)[:, :show_until]
    ge_1 = np.array(results_by_method["ge_1"]).transpose(1, 0, 2).mean(axis=1)[:, :show_until]
    cantillo = np.mean(np.array([gnn_ip[1], ge_1[1]]), axis=0)
    ge_1000 = np.array(results_by_method["ge_1000"]).transpose(1, 0, 2).mean(axis=1)[:, :show_until]
    cantillo_1000 = np.mean(np.array([gnn_ip[1], ge_1000[1]]), axis=0)
    ge_5000 = np.array(results_by_method["ge_5000"]).transpose(1, 0, 2).mean(axis=1)[:, :show_until]
    cantillo_5000 = np.mean(np.array([gnn_ip[1], ge_5000[1]]), axis=0)
    ge_10000 = np.array(results_by_method["ge_10000"]).transpose(1, 0, 2).mean(axis=1)[:, :show_until]
    cantillo_10000 = np.mean(np.array([gnn_ip[1], ge_10000[1]]), axis=0)

    ip_1 = np.array(results_by_method["ip_1"]).transpose(1, 0, 2).mean(axis=1)[:, :show_until]
    ip_1000 = np.array(results_by_method["ip_1000"]).transpose(1, 0, 2).mean(axis=1)[:, :show_until]
    ip_5000 = np.array(results_by_method["ip_5000"]).transpose(1, 0, 2).mean(axis=1)[:, :show_until]
    ip_10000 = np.array(results_by_method["ip_10000"]).transpose(1, 0, 2).mean(axis=1)[:, :show_until]
    return (
        cantillo,
        cantillo_1000,
        cantillo_10000,
        cantillo_5000,
        ge_1,
        ge_1000,
        ge_10000,
        ge_5000,
        gnn_ge,
        gnn_ip,
        ip_1,
        ip_1000,
        ip_10000,
        ip_5000,
    )


@app.cell
def _():
    return


@app.cell
def _(
    PATH_FIGURES,
    PercentFormatter,
    cantillo,
    ge_1,
    gnn_ge,
    gnn_ip,
    ip_1,
    np,
    plt,
):
    plt.plot(ip_1[0], ip_1[1], label=f"$$IP ({round(np.trapezoid(ip_1[1]))})$$", alpha=.3)
    plt.plot(gnn_ip[0], gnn_ip[1], label=f"$$GNN_{{IP}} ({round(np.trapezoid(gnn_ip[1]))})$$")
    plt.plot(gnn_ge[0], gnn_ge[1], label=f"$$GNN_{{GE}} ({round(np.trapezoid(gnn_ge[1]))})$$")
    plt.plot(gnn_ge[0], cantillo, label=f"$$Cantillo ({round(np.trapezoid(cantillo))})$$", alpha=.3)
    plt.plot(ge_1[0], ge_1[1], label=f"$$GE ({round(np.trapezoid(ge_1[1]))})$$", alpha=.2)
    plt.legend()
    plt.gca().xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    plt.xlabel("Disrupted links")
    plt.ylabel("Normalized GII")
    plt.title("Single link disruptions")
    plt.savefig(PATH_FIGURES/"gii_curve_single_presentation.pdf", transparent=True)
    plt.show()
    return


@app.cell
def _(
    PATH_FIGURES,
    PercentFormatter,
    cantillo_1000,
    ge_1000,
    gnn_ge,
    gnn_ip,
    ip_1000,
    np,
    plt,
):
    plt.plot(ip_1000[0], ip_1000[1], label=f"$$IP ({round(np.trapezoid(ip_1000[1]))})$$", alpha=.3)
    plt.plot(gnn_ip[0], gnn_ip[1], label=f"$$GNN_{{IP}} ({round(np.trapezoid(gnn_ip[1]))})$$")
    plt.plot(gnn_ge[0], gnn_ge[1], label=f"$$GNN_{{GE}} ({round(np.trapezoid(gnn_ge[1]))})$$")
    plt.plot(gnn_ge[0], cantillo_1000, label=f"$$Cantillo ({round(np.trapezoid(cantillo_1000))})$$", alpha=.3)
    plt.plot(ge_1000[0], ge_1000[1], label=f"$$GE ({round(np.trapezoid(ge_1000[1]))})$$", alpha=.2)
    handles_1, labels_1 = plt.gca().get_legend_handles_labels()
    order_1 = [1,0,2,3,4]
    plt.legend([handles_1[idx] for idx in order_1],[labels_1[idx] for idx in order_1])
    plt.gca().xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    plt.xlabel("Disrupted links")
    plt.ylabel("Normalized GII")
    plt.title("1,000 disruptions")
    plt.savefig(PATH_FIGURES/"gii_curve_1000_presentation.pdf", transparent=True)
    plt.show()
    return


@app.cell
def _(
    PATH_FIGURES,
    PercentFormatter,
    cantillo_5000,
    ge_5000,
    gnn_ge,
    gnn_ip,
    ip_5000,
    np,
    plt,
):
    plt.plot(ip_5000[0], ip_5000[1], label=f"$$IP ({round(np.trapezoid(ip_5000[1]))})$$", alpha=.3)
    plt.plot(gnn_ip[0], gnn_ip[1], label=f"$$GNN_{{IP}} ({round(np.trapezoid(gnn_ip[1]))})$$")
    plt.plot(gnn_ge[0], gnn_ge[1], label=f"$$GNN_{{GE}} ({round(np.trapezoid(gnn_ge[1]))})$$")
    plt.plot(gnn_ge[0], cantillo_5000, label=f"$$Cantillo ({round(np.trapezoid(cantillo_5000))})$$", alpha=.3)
    plt.plot(ge_5000[0], ge_5000[1], label=f"$$GE ({round(np.trapezoid(ge_5000[1]))})$$", alpha=.2)
    handles_2, labels_2 = plt.gca().get_legend_handles_labels()
    order_2 = [1,0,2,3,4]
    plt.legend([handles_2[idx] for idx in order_2],[labels_2[idx] for idx in order_2])
    plt.gca().xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    plt.xlabel("Disrupted links")
    plt.ylabel("Normalized GII")
    plt.title("5,000 disruptions")
    plt.savefig(PATH_FIGURES/"gii_curve_5000_presentation.pdf", transparent=True)

    plt.show()
    return


@app.cell
def _(
    PATH_FIGURES,
    PercentFormatter,
    cantillo_10000,
    ge_10000,
    gnn_ge,
    gnn_ip,
    ip_10000,
    np,
    plt,
):
    plt.plot(ip_10000[0], ip_10000[1], label=f"$$IP ({round(np.trapezoid(ip_10000[1]))})$$", alpha=.3)
    plt.plot(gnn_ip[0], gnn_ip[1], label=f"$$GNN_{{IP}} ({round(np.trapezoid(gnn_ip[1]))})$$")
    plt.plot(gnn_ge[0], gnn_ge[1], label=f"$$GNN_{{GE}} ({round(np.trapezoid(gnn_ge[1]))})$$")
    plt.plot(gnn_ge[0], cantillo_10000, label=f"$$Cantillo ({round(np.trapezoid(cantillo_10000))})$$", alpha=.3)
    plt.plot(ge_10000[0], ge_10000[1], label=f"$$GE ({round(np.trapezoid(ge_10000[1]))})$$", alpha=.2)
    plt.legend()
    plt.gca().xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    plt.xlabel("Disrupted links")
    plt.ylabel("Normalized GII")
    plt.title("10,000 disruptions")
    plt.savefig(PATH_FIGURES/"gii_curve_10000_presentation.pdf", transparent=True)
    plt.show()
    return


@app.cell
def _(PATH_FIGURES, PATH_PROCESSED_DATA, joblib, np, plt):
    times_edges = joblib.load(PATH_PROCESSED_DATA/"N_edges_timing.pkl")
    times_ = {}
    for meth_, time in times_edges.items():
        x_, y_ = list(map(list, zip(*time)))
        times_[meth_] = {"x": x_, "y": y_}
    #plt.plot(times_["ip"]["x"], signal.savgol_filter(times_["ip"]["y"], window_length=3, polyorder=1))
    # plt.plot(times_["ip"]["x"], gaussian_filter(times_["ip"]["y"], 2))
    plt.plot(times_["ip"]["x"], np.array([times_["ip"]["y"], times_["ge"]["y"]]).mean(axis=0), label="Sequential")
    plt.plot(times_["ip_parallel"]["x"], np.array([times_["ip_parallel"]["y"], times_["ge_parallel"]["y"]]).mean(axis=0), label="Parallel")
    plt.plot(times_["ip_parallel"]["x"], np.array([times_["gnn_ip"]["y"], times_["gnn_ge"]["y"]]).mean(axis=0), label="GNN")
    # plt.plot(times_["ge_parallel"]["x"], times_["gnn_ge"]["y"], label="$$GNN_{GE}$$")
    plt.xlabel("Number of edges")
    plt.legend(loc="lower center", bbox_to_anchor=(0.25, 1, 0.5, 0.5), ncols=3)

    plt.ylabel(r"$$\log_{10} (\text{seconds})$$")
    # plt.ylabel("Seconds")
    plt.yscale("log")
    handles_, labels = plt.gca().get_legend_handles_labels()
    plt.savefig(PATH_FIGURES/"n_edges_timing_log_presentation.pdf", transparent=True)

    plt.show()
    return


@app.cell
def _(PATH_FIGURES, PATH_PROCESSED_DATA, defaultdict, joblib, np, plt):
    times_st = joblib.load(PATH_PROCESSED_DATA/"times_st.pkl")
    new_times_st = {}
    for method_st, values_ in times_st.items():
        x_y = defaultdict(list)
        for x_st, y_st in values_:
            x_y[x_st].append(y_st)
        new_list = []
        for x_st, y_st in x_y.items():
            new_list.append((x_st, np.mean(y_st)))
        x_st, y_st = list(map(list, zip(*new_list)))
        new_times_st[method_st] = {"x": x_st, "y": y_st}

    plt.plot(new_times_st["ip"]["x"], np.array([new_times_st["ip"]["y"], new_times_st["ge"]["y"]]).mean(axis=0), label="Sequential")
    plt.plot(new_times_st["ip_parallel"]["x"], np.array([new_times_st["ip_parallel"]["y"], new_times_st["ge_parallel"]["y"]]).mean(axis=0), label="Parallel")
    plt.plot(new_times_st["ip_parallel"]["x"], np.array([new_times_st["gnn_ip"]["y"], new_times_st["gnn_ge"]["y"]]).mean(axis=0), label="GNN")
    # plt.plot(times_["ge_parallel"]["x"], times_["gnn_ge"]["y"], label="$$GNN_{GE}$$")
    plt.xlabel("Number of source + terminal nodes")
    plt.legend(loc="lower center", bbox_to_anchor=(0.25, 1, 0.5, 0.5), ncols=3)

    plt.ylabel(r"Seconds")
    plt.ylabel(r"$$\log_{10} (\text{seconds})$$")
    plt.yscale("log")
    handles_st, labels_st = plt.gca().get_legend_handles_labels()
    plt.savefig(PATH_FIGURES/"n_st_timing_log_presentation.pdf", transparent=True)
    plt.show()
    return


@app.cell
def _(random):
    import pandas as pd
    import networkx as nx
    from vulnerability_networks.modeling.train import NetworkDataModule, LightningRankEdgeNet
    from vulnerability_networks.config import PATH_MODELS
    from torch_geometric.data import Data
    import torch
    from vulnerability_networks.algorithms.rank_links import process_network
    from vulnerability_networks.algorithms.functionality_based import global_efficiency, number_independent_paths
    import itertools

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
                    print(f"Nodo {node} mapeado a {i} con profile {node_data['profile']}")
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


    model_gnn_ge = LightningRankEdgeNet.load_from_checkpoint(PATH_MODELS/"validate/global_efficiency/version_7/checkpoints/epoch=99-step=2500.ckpt", map_location="cpu")
    model_gnn_ge.eval()
    model_gnn_ip = LightningRankEdgeNet.load_from_checkpoint(PATH_MODELS/"validate/independent_path/version_3/checkpoints/epoch=99-step=1600.ckpt", map_location="cpu")
    model_gnn_ip.eval()
    return (
        convert_nx_to_pyg,
        global_efficiency,
        itertools,
        model_gnn_ge,
        model_gnn_ip,
        number_independent_paths,
        nx,
        pd,
        process_network,
        pyg_to_nx,
    )


@app.cell
def _(PATH_DATA, pd):
    net= pd.read_csv(PATH_DATA/'external/sioux_falls_net.csv')
    net['edge']=net.index+1
    return (net,)


@app.cell
def _():
    return


@app.cell
def _():
    #net.to_csv(PATH_DATA/"external/sioux_falls_net.csv", index=False)
    return


@app.cell
def _(net, nx):
    G = nx.from_pandas_edgelist(net, 'init_node', 'term_node', 'weight',create_using=nx.DiGraph())
    sources = [1, 10, 20]
    terminals = [5, 6, 7, 12, 13, 14, 15, 18, 22, 23]
    assignment = {}
    for node in G.nodes:
        if node in sources:
            assignment[node] = "source"
        elif node in terminals:
            assignment[node] = "terminal"
        else:
            assignment[node] = "regular"

    nx.set_node_attributes(G, assignment, "profile")
    return (G,)


@app.cell
def _(data, model_gnn_ge):
    model_gnn_ge(data)
    return


@app.cell
def _(
    G,
    convert_nx_to_pyg,
    global_efficiency,
    model_gnn_ge,
    model_gnn_ip,
    nx,
    process_network,
    pyg_to_nx,
):
    data = convert_nx_to_pyg(G)
    data.gnn_ge = model_gnn_ge(data)
    data.gnn_ip = model_gnn_ip(data)
    G_ = pyg_to_nx(data)
    print(nx.get_node_attributes(G_, "profile"))
    edges_attr = list(G_.edges(data = True))
    rank_gnn_ge = [((u, v), attrs["gnn_ge"]) for u, v, attrs in edges_attr]
    rank_gnn_ge = sorted(rank_gnn_ge, key=lambda x: x[1], reverse=True)
    rank_gnn_ge = [edge for edge, score in rank_gnn_ge]
    rank_gnn_ip = [((u, v), attrs["gnn_ip"]) for u, v, attrs in edges_attr]
    rank_gnn_ip = sorted(rank_gnn_ip, key=lambda x: x[1], reverse=True)
    rank_gnn_ip = [edge for edge, score in rank_gnn_ip]
    rank_ge = process_network(G_.copy(), global_efficiency, max_links_in_disruption=0.4,
                              max_disruption_scenarios=10_000, workers=16)["link_scores"]
    rank_ge = sorted(rank_ge, key=lambda x: x[1], reverse=True)
    rank_ge = [edge for edge, score in rank_ge]

    return G_, data, rank_ge, rank_gnn_ge, rank_gnn_ip


@app.cell
def _(G_, number_independent_paths, process_network):
    rank_ip = process_network(G_.copy(), number_independent_paths, max_links_in_disruption=0.4,
                              max_disruption_scenarios=10_000, workers=16)["link_scores"]
    rank_ip = sorted(rank_ip, key=lambda x: x[1], reverse=True)
    rank_ip = [edge for edge, score in rank_ip]
    return (rank_ip,)


@app.cell
def _(itertools, np, nx):
    def compute_gii(sources, terminals, distribution_plan, costs, disrupted_costs, max_disrupted_cost):


        unsatisfied_demand = 0
        importance_score = 0
        total_demand = 0
        for s, t in itertools.product(sources, terminals):
            try:
                demand_st = distribution_plan[s][t]
            except:
                demand_st = 0
            total_demand +=demand_st
            # breakpoint()
            if not (s in disrupted_costs and t in disrupted_costs[s]):
                 unsatisfied_demand += demand_st
            elif s in disrupted_costs and t in disrupted_costs[s]:
                importance_score += demand_st*(disrupted_costs[s][t] - costs[s][t])
                max_disrupted_cost = max(max_disrupted_cost, disrupted_costs[s][t] - costs[s][t])
            else:
                breakpoint()
        alpha = 1.1 * max_disrupted_cost
        # if unsatisfied_demand == total_demand:
        #     return (math.inf, max_disrupted_cost)
        result = (importance_score + alpha*unsatisfied_demand)/total_demand

        return result, max_disrupted_cost

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
            gii_val, max_disrupted_cost = compute_gii(sources, terminals, distribution_plan, initial_costs,
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


    return (gii,)


@app.cell
def _(G_):
    G_[3]
    return


@app.cell
def _(G_, gii, np, rank_ge, rank_gnn_ge, rank_gnn_ip, rank_ip):
    distribution_plan = {0: {6: 100, 5: 100, 17: 100, 16: 100}, 
                        13: {6: 100, 3: 200, 10: 200, 5: 100, 14: 100},
                        22: {17: 100, 16: 100, 14: 100, 22: 200, 21: 200, 19: 200}
                        }

    gii_gnn_ge = gii(G_.copy(), rank_gnn_ge, distribution_plan)
    x1, y1 = gii_gnn_ge[0], gii_gnn_ge[1]
    x1, y1 = np.hstack([[0], x1]), np.hstack([[0], y1])
    y_norm1 = (y1 - y1.min()) / (y1.max() - y1.min())
    gii_gnn_ge = np.array([x1, y_norm1])[:, :31]

    gii_gnn_ip = gii(G_.copy(), rank_gnn_ip, distribution_plan)
    x2, y2 = gii_gnn_ip[0], gii_gnn_ip[1]
    x2, y2 = np.hstack([[0], x2]), np.hstack([[0], y2])
    y_norm2 = (y2 - y2.min()) / (y2.max() - y2.min())
    gii_gnn_ip = np.array([x2, y_norm2])[:, :31]

    gii_ge = gii(G_.copy(), rank_ge, distribution_plan)
    x3, y3 = gii_ge[0], gii_ge[1]
    x3, y3 = np.hstack([[0], x3]), np.hstack([[0], y3])
    y_norm3 = (y3 - y3.min()) / (y3.max() - y3.min())
    gii_ge = np.array([x3, y_norm3])[:, :31]

    gii_ip = gii(G_.copy(), rank_ip, distribution_plan)
    x4, y4 = gii_ip[0], gii_ip[1]
    x4, y4 = np.hstack([[0], x4]), np.hstack([[0], y4])
    y_norm4 = (y4 - y4.min()) / (y4.max() - y4.min())
    gii_ip = np.array([x4, y_norm4])[:, :31]

    return gii_ge, gii_gnn_ge, gii_gnn_ip, gii_ip


@app.cell
def _(
    PATH_FIGURES,
    PercentFormatter,
    gii_ge,
    gii_gnn_ge,
    gii_gnn_ip,
    gii_ip,
    np,
    plt,
):
    plt.plot(gii_ip[0], gii_ip[1], label=f"$$IP ({round(np.trapezoid(gii_ip[1]), 2)})$$", alpha=.3)
    plt.plot(gii_gnn_ip[0], gii_gnn_ip[1], label=f"$$GNN_{{IP}} ({round(np.trapezoid(gii_gnn_ip[1]), 2)})$$")
    plt.plot(gii_gnn_ge[0], gii_gnn_ge[1], label=f"$$GNN_{{GE}} ({round(np.trapezoid(gii_gnn_ge[1]), 2)})$$")
    plt.plot(gii_ge[0], gii_ge[1], label=f"$$GE ({round(np.trapezoid(gii_ge[1]), 2)})$$", alpha=.2)
    handles_six, labels_six = plt.gca().get_legend_handles_labels()
    #order_six = [1,0,2,3,4]
    plt.legend()
    plt.gca().xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    plt.xlabel("Disrupted links")
    plt.ylabel("Normalized GII")
    #plt.title("10,000 disruptions")
    plt.savefig(PATH_FIGURES/"gii_curve_10000_sioux_falls.pdf", transparent=True)
    plt.show()
    return


if __name__ == "__main__":
    app.run()
