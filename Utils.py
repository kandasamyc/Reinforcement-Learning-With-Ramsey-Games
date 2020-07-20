import networkx as nx
import matplotlib.pyplot as plt
from Agent import Agent
import numpy as np
import ax
from tqdm import tqdm
import torch
from torch import nn
from itertools import permutations


# noinspection PyUnresolvedReferences
class Utils:

    def __init__(self, player: Agent, adversary: Agent, num_of_games: int = 3000):
        super().__init__()
        self.player = player
        self.adversary = adversary
        self.number_of_games = num_of_games

    @staticmethod
    def detect_cycle(G: nx.Graph, chain_length: int):
        """Returns the color of a cycle of length chain_length if it exists, otherwise None"""
        if nx.graph_clique_number(G) < chain_length:
            return None
        chains = [i for i in list(nx.enumerate_all_cliques(G)) if len(i) == chain_length]
        for chain in chains:
            chain_edges = [*[G[chain[node]][chain[node + 1]] for node in range(chain_length - 1)],
                           *[G[chain[0]][chain[-1]]]]
            if all([c['color'] == chain_edges[0]['color'] for c in chain_edges]):
                return chain_edges[0]['color']
        return None

    @staticmethod
    def display_graph(G: nx.Graph, text: bool = True):
        """Draws the graph with colored edges, If text is true it returns a modified adjacency matrix, otherwise shows it graphically and returns None"""
        if not text:
            all_edges = G.edges
            colors = [G[u][v]['color'] for u, v in all_edges]
            pos = nx.circular_layout(G)
            nx.draw(G, pos=pos, edges=all_edges, edge_color=colors, node_color=["gray"] * nx.number_of_nodes(G),
                    with_labels=True)
            plt.show()
            return None
        else:
            return list(Utils.weighted_adj(G))

    @staticmethod
    def weighted_adj(G: nx.Graph,color):
        adj = nx.adjacency_matrix(G).tocoo()
        w_adj = np.zeros((nx.number_of_nodes(G), nx.number_of_nodes(G)), dtype=float)
        for i, j in zip(adj.row, adj.col):
            w_adj[i][j] = 1 if G.get_edge_data(i, j)['color'] == color else (
                -1 if G.get_edge_data(i, j)['color'] else 0)
        return torch.tensor(w_adj,dtype=torch.float)

    @staticmethod
    def new_edge(G: nx.Graph, color: str, edge: tuple):
        """Adds an edge on G if the edge doesn't already exist, otherwise None"""
        edge = list(edge)
        edge.sort()
        edge = tuple(edge)
        G.add_edge(*edge, color=color)
        return G

    @staticmethod
    def reward(G: nx.Graph, chain_length: int, player_color: str, ):
        """Returns the reward for a state"""
        cycle = Utils.detect_cycle(G, chain_length)
        if cycle is None:
            return 0.0
        else:
            if cycle == player_color:
                return 1.0
            else:
                return -1.0

    @staticmethod
    def make_graph(nodes: int):
        G = nx.Graph()
        G.add_nodes_from([i for i in range(0, nodes)])
        return G

    @staticmethod
    def transition(G: nx.Graph, color: str, edge: tuple):
        """Returns a copy of G with the edge added"""
        new_G = G.copy()
        new_G = Utils.new_edge(new_G, color, edge)
        return new_G

    @staticmethod
    def get_uncolored_edges(G: nx.Graph):
        """Returns the edges in the graph that have not been colored"""
        uncolored_edges = set(permutations([i for i in range(nx.number_of_nodes(G))],2)) - set(G.edges)
        return uncolored_edges

    @staticmethod
    def weight_initialization(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def train(self, parametrization=None):
        """Given two Agents, the method will train them against each other until number of games is reached, by default 3000 games"""
        self.player.hard_reset()
        self.adversary.hard_reset()
        if parametrization is not None:
            self.player.hyperparameters = parametrization
            self.adversary.hyperparameters = parametrization
            self.player.update_writer(parametrization)
            self.player.update_writer(parametrization)
        for game_num in tqdm(range(self.number_of_games)):
            finished = False
            while not finished:
                finished = self.player.move(self.adversary)  # player makes move
                if finished:
                    break
                finished = self.adversary.move(self.player)  # adversary makes move
            self.player.epoch += 1
            self.adversary.epoch += 1
            self.player.write_info()
            self.adversary.write_info()
            # self.player.write_network_info(
            #     [self.player.q_network.input,self.player.q_network.l2,self.player.q_network.output],
            #     ['Input','Hidden','Output']
            # )
            # self.adversary.write_network_info(
            #     [self.adversary.q_network.input, self.adversary.q_network.l2, self.adversary.q_network.output],
            #     ['Input', 'Hidden', 'Output']
            # )
            self.player.reset()
            self.adversary.reset()
        return self.player.wins / self.player.epoch

    def optimize_training(self, params):
        best_parameters, values, experiment, model = ax.optimize(
            parameters=params,
            evaluation_function=self.train,
            minimize=False,
        )
        return best_parameters

    @staticmethod
    def play(player: Agent, goes_first: bool = True):
        """Allows the user to play a game against an agent, agent will go first by default"""
        finished = False
        while not finished:
            if goes_first:
                finished = player.move()
            else:
                good_input = False
                while not good_input:
                    starting_node = input("Enter a starting node for an edge: ")
                    ending_node = input("Enter an ending node for an edge: ")

        # TODO: When finished with one Q-Learning model, use those methods to write this
