import networkx as nx
import igraph as ig
from time import time_ns
import numpy as np
import torch
from itertools import combinations
from GQN import GATCQNetwork
from Utils import Utils
import torch_geometric
import csv
from MCTS import MCTSAgent
import math




def detect_cycle(G: ig.Graph, chain_length: int):
    """Returns the color of a cycle of length chain_length if it exists, otherwise None"""
    cliques = list(G.cliques(min=chain_length))
    cliques = [i for i in cliques if len(i) == chain_length]
    if len(cliques) < 1:
        return None
    else:
        for chain in cliques:
            chain_edges = [*[G[chain[node], chain[node + 1]] for node in range(chain_length - 1)],
                           *[G[chain[0], chain[-1]]]]
            if len(set(chain_edges)) == 1:
                return chain_edges[0]
    return None

def detect_cycle2(G: ig.Graph, chain_length: int):
    """Returns the color of a cycle of length chain_length if it exists, otherwise None"""
    cliques = list(G.cliques(min=chain_length))
    cliques = [i for i in cliques if len(i) == chain_length]
    if len(cliques) < 1:
        return None
    else:
        for chain in cliques:
            edges = combinations(chain,2)
            chain_edges = [G[c[0],c[1]] for c in edges]
            if len(set(chain_edges)) == 1:
                return chain_edges[0]
    return None
f = Utils.make_graph(6)
for edge in [(1,2),(2,3),(3,4),(4,1),(3,1),(2,4)]:
    f = Utils.transition(f,'Red',edge)
for edge in []:
    f = Utils.transition(f,'Blue',edge)
print(detect_cycle2(f,4))
# g = nx.Graph()
# g.add_nodes_from([i for i in range(6)])
# t = GQNetwork(6,100)
# print(torch_geometric.utils.from_networkx(g))
# g.add_edge(1, 2, color=['red'])
# g.add_edge(2, 3, color=['red'])
# g.add_edge(3, 1, color=['red'])
# g.add_edge(5, 4, color=['blue'])
# i = ig.Graph()
# i.add_vertices([i for i in range(6)])
# i.es['weight'] = 0.0
# print(Utils.graph_to_data(i,'Red'))
# i.add_edges([(1, 2), (2, 3), (3, 1), (5, 4)])
# i.es['weight'] = 0.0
# i[1, 2] = 1
# i[2, 3] = 1
# i[3, 1] = 1
# i[5, 4] = -1
# # print(time_ns())
# # y = torch.tensor(list(i.get_adjacency(type=ig.GET_ADJACENCY_UPPER,attribute='weight')))[np.triu_indices(6,k=1)]
# # print(time_ns())
# # print(y)
# #
# # # print(detect_cycle2(i, 3))
# # print(time_ns())
# # x = 1
# # ans = 0
# # count = 6-1
# # while x > count and count > 0:
# #     x -= count
# #     count -= 1
# # print((5-count),x+(5-count))
# # print(time_ns())
#
# # print([*[list(e) for e in i.get_edgelist()],*[list(list(e).__reversed__()) for e in i.get_edgelist()]])
# # x = list(i.get_adjacency(attribute='weight'))
# # print(x)
# # for r in range(len(x)):
# #     del x[r][r]
# # print(x)
#
# g = GQNetwork(6,100)
f = Utils.make_graph(6)
for edge in [(1,2),(2,4)]:
    f = Utils.transition(f,'Red',edge)
for edge in [(3,2),(1,5)]:
    f = Utils.transition(f,'Blue',edge)
#
# print(list(f.cliques(min=3)))

# print('Starting')
# start = time_ns()
# m = GATCQNetwork(6,100)
# d = Utils.graph_to_data(f,'Red',torch.device('cpu'))
# e_types = [f[e[0],e[1]] for e in f.get_edgelist()]
# print(m.forward(d))
# print(Utils.get_uncolored_edges(f))
# print((start-time_ns())*1e-9)


