from Agent import Agent
import ax
from tqdm import tqdm
import torch
from torch import nn
from itertools import combinations
import igraph as ig
import numpy as np
import datetime

colors = {'Red':1,'Blue':-1}

class Utils:

    def __init__(self, player: Agent, adversary: Agent, num_of_games: int = 3000):
        super().__init__()
        self.player = player
        self.adversary = adversary
        self.number_of_games = num_of_games

    @staticmethod
    def detect_cycle(G: ig.Graph, chain_length: int):
        """Returns the color of a cycle of length chain_length if it exists, otherwise None"""
        cliques = list(G.cliques(min=chain_length))
        if len(cliques) < 1:
            return None
        else:
            for chain in cliques:
                chain_edges = [*[G[chain[node], chain[node + 1]] for node in range(chain_length - 1)],
                               *[G[chain[0], chain[-1]]]]
                if len(set(chain_edges)) == 1:
                    return chain_edges[0]
        return None

    @staticmethod
    def display_graph(G: ig.Graph, text: bool = False):
        """Draws the graph with colored edges, If text is true it returns a modified adjacency matrix, otherwise shows it graphically and returns None"""
        if not text:
            layout = G.layout('circle')
            G.vs['label'] = G.vs['name']
            color_dict = {1:'red',-1:'blue'}
            G.es['color'] = [color_dict[weight] for weight in G.es['weight']]
            ig.plot(G,f'games/{datetime.datetime.now()}',layout=layout)
        else:
            return G.summary()

    @staticmethod
    def weighted_adj(G: ig.Graph,color:str):
        w_adj = G.get_adjacency(attribute='weight',type=ig.GET_ADJACENCY_UPPER)
        return torch.tensor(list(w_adj),dtype=torch.float)[np.triu_indices(G.vcount(),1)]*colors[color]

    @staticmethod
    def new_edge(G: ig.Graph, color: str, edge: tuple):
        """Adds an edge on G if the edge doesn't already exist, otherwise None"""
        G.add_edge(*edge,weight=colors[color])
        return G

    @staticmethod
    def reward(G: ig.Graph, chain_length: int, player_color: str, ):
        """Returns the reward for a state"""
        cycle = Utils.detect_cycle(G, chain_length)
        if cycle is None:
            return 0.0
        else:
            if cycle == colors[player_color]:
                return 1.0
            else:
                return -1.0

    @staticmethod
    def make_graph(nodes: int):
        G = ig.Graph()
        G.es['weight'] = 0.0
        G.add_vertices([i for i in range(0, nodes)])
        return G

    @staticmethod
    def transition(G: ig.Graph, color: str, edge: tuple):
        """Returns a copy of G with the edge added"""
        new_G = G.copy()
        new_G = Utils.new_edge(new_G, color, edge)
        return new_G

    @staticmethod
    def get_uncolored_edges(G: ig.Graph):
        """Returns the edges in the graph that have not been colored"""
        uncolored_edges = set(combinations([i for i in range(G.vcount())],2)) - set(G.get_edgelist())
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
            self.adversary.update_writer(parametrization)
        for game_num in tqdm(range(self.number_of_games)):
            finished = False
            while not finished:
                finished = self.player.move(self.adversary) # player makes move
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
