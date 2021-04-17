import sys

sys.path.insert(0, '..')
sys.path.insert(0, '../..')
sys.path.insert(0, '.')

import os
import networkx as nx

data_path = "../../DATA1/er_SET2/200_200/alpha_0.75_setParam_100/train/"

# TODO

for num, filename in enumerate(os.listdir(data_path)):
    print(num)
    # Get graph.
    graph = nx.read_gpickle(data_path + filename)

    # Make graph directed.
    graph = nx.convert_node_labels_to_integers(graph)
    graph = graph.to_undirected() if nx.is_directed(graph) else graph

    graph_new = nx.Graph()

    matrices_vv_cv_1 = []
    matrices_vv_vc_2 = []

    matrices_cc_vc_1 = []
    matrices_cc_cv_2 = []

    matrices_vc_cc_1 = []
    matrices_vc_vv_2 = []

    matrices_cv_vv_1 = []
    matrices_cv_cc_2 = []

    c = 0
    for i, (u, v) in enumerate(graph.edges):
        if graph.nodes[u]['bipartite'] == 0:
            graph_new.add_node((u, v), type="VC", first=u, second=v, num=c)
            c += 1
            graph_new.add_node((v, u), type="CV", first=v, second=u, num=c)
            c += 1

    for i, v in enumerate(graph.nodes):
         if graph.nodes[v]['bipartite'] == 0:
             graph_new.add_node((v,v), type="VV", first=v, second=v, num=c)
             c += 1
         elif graph.nodes[v]['bipartite'] == 1:
             graph_new.add_node((v,v), type="CC", first=v, second=v, num=c)
             c += 1

    for i, (v, data) in enumerate(graph_new.nodes(data=True)):
        first = data["first"]
        second = data["second"]
        num = data["num"]

        for n in graph.neighbors(first):

            if graph_new.nodes[v]["type"] == "VV":
                if graph.has_edge(n, second):
                    matrices_vv_cv_1.append([num, graph_new.nodes[(n, second)]["num"]])
            if graph_new.nodes[v]["type"] == "CC":
                if graph.has_edge(n, second):
                    matrices_cc_vc_1.append([num, graph_new.nodes[(n, second)]["num"]])
            if graph_new.nodes[v]["type"] == "VC":
                if graph.has_edge(n, second):
                    matrices_vc_cc_1.append([num, graph_new.nodes[(n, second)]["num"]])
            if graph_new.nodes[v]["type"] == "CV":
                if graph.has_edge(n, second):
                    matrices_cv_vv_1.append([num, graph_new.nodes[(n, second)]["num"]])


    print(graph_new.number_of_nodes())


        # for n in graph.neighbors(second):
        #
        #     if graph_new.nodes[v]["type"] == "VV":
        #         matrices_vv_vc_2.append([num, graph_new.nodes[(first, n)]["num"]])
        #     if graph_new.nodes[v]["type"] == "CC":
        #         matrices_cc_cv_2.append([num, graph_new.nodes[(first, n)]["num"]])
        #     if graph_new.nodes[v]["type"] == "VC":
        #         matrices_vc_vv_2.append([num, graph_new.nodes[(first, n)]["num"]])
        #     if graph_new.nodes[v]["type"] == "CV":
        #         matrices_cv_cc_2.append([num, graph_new.nodes[(first, n)]["num"]])




