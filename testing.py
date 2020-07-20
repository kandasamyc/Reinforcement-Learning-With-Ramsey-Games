import networkx as nx
import igraph as ig
from time import time_ns
import numpy as np
import torch

def detect_cycle(G: nx.Graph, chain_length: int):
    """Returns the color of a cycle of length chain_length if it exists, otherwise None"""
    if nx.graph_clique_number(G) < chain_length:
        return None
    chains = [i for i in list(nx.enumerate_all_cliques(G)) if len(i) == chain_length]
    print(chains)
    for chain in chains:
        chain_edges = [*[G[chain[node]][chain[node + 1]] for node in range(chain_length - 1)],
                       *[G[chain[0]][chain[-1]]]]
        if all([c['color'] == chain_edges[0]['color'] for c in chain_edges]):
            return chain_edges[0]['color']
    return None

def detect_cycle2(G:ig.Graph,chain_length:int):
    cliques = list(G.cliques(min=chain_length))
    print(cliques)
    if len(cliques) < 1:
        return None
    else:
        for chain in cliques:
            chain_edges = [*[G[chain[node],chain[node + 1]] for node in range(chain_length - 1)],
                       *[G[chain[0],chain[-1]]]]
            print(chain_edges)
            if len(set(chain_edges)) == 1:
                return chain_edges[0]
    return None

g = nx.Graph()
g.add_nodes_from([i for i in range(6)])
g.add_edge(1,2,color=['red'])
g.add_edge(2,3,color=['red'])
g.add_edge(3,1,color=['red'])
g.add_edge(5,4,color=['blue'])
i = ig.Graph()
i.add_vertices([i for i in range(6)])
i.add_edges([(1,2),(2,3),(3,1),(5,4)])
i.es['weight'] = 1
i[1,2] = 2
i[2,3] = 2
i[3,1] = 1
i[5,4] = -1
print(detect_cycle2(i,3))