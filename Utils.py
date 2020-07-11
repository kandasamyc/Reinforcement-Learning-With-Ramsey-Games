import networkx as nx
import matplotlib.pyplot as plt
from copy import deepcopy

class Utils:

    @staticmethod
    def detect_cycle(G: nx.Graph(),chain_length: int):
        """Returns the color of a cycle of length chain_length if it exists, otherwise None"""
        chains = [i for i in list(nx.enumerate_all_cliques(G)) if len(i)==chain_length]
        for chain in chains:
            chain_edges = [*[G[chain[node]][chain[node + 1]] for node in range(chain_length-1)],*[G[chain[0]][chain[-1]]]]
            if all([c['color'] == chain_edges[0]['color'] for c in chain_edges]):
                return chain_edges[0]['color']
        return None

    @staticmethod
    def display_graph(G:nx.Graph()):
        """Draws the graph with colored edges"""
        all_edges = G.edges
        colors = [G[u][v]['color'] for u,v in all_edges]
        pos = nx.circular_layout(G)
        nx.draw(G,pos=pos,edges=all_edges,edge_color=colors,node_color=["gray"]*nx.number_of_nodes(G),with_labels=True)
        plt.show()

    @staticmethod
    def new_edge(G: nx.Graph(),color: str,edge: tuple):
        """Adds an edge on G if the edge doesn't already exist, otherwise None"""
        if edge in list(G.edges):
            return None
        G.add_edge(*edge,color=color)
        return G

    @staticmethod
    def reward(G: nx.Graph(),chain_length: int, player_color: str,):
        """Returns the reward for a state"""
        cycle = Utils.detect_cycle(G,chain_length)
        if cycle is None:
            return 0.0
        else:
            if cycle == player_color:
                return 1.0
            else:
                return -1.0

    @staticmethod
    def transition(G:nx.Graph(),color:str,edge:tuple):
        """Returns a copy of G with the edge added"""
        new_G = deepcopy(G)
        new_G = Utils.new_edge(new_G,color,edge)
        return new_G

    @staticmethod
    def get_uncolored_edges(G:nx.Graph()):
        """Returns the edges in the graph that have not been colored"""
        complete_G = nx.complete_graph(nx.number_of_nodes(G))
        uncolored_edges = set(complete_G.edges) - set(G.edges)
        return uncolored_edges

print(Utils.get_uncolored_edges(nx.star_graph(5)))