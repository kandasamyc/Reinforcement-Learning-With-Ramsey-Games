import networkx as nx
import matplotlib.pyplot as plt
from copy import deepcopy
from Agent import Agent
import numpy as np

class Utils:

    @staticmethod
    def detect_cycle(G: nx.Graph,chain_length: int):
        """Returns the color of a cycle of length chain_length if it exists, otherwise None"""
        chains = [i for i in list(nx.enumerate_all_cliques(G)) if len(i)==chain_length]
        for chain in chains:
            chain_edges = [*[G[chain[node]][chain[node + 1]] for node in range(chain_length-1)],*[G[chain[0]][chain[-1]]]]
            if all([c['color'] == chain_edges[0]['color'] for c in chain_edges]):
                return chain_edges[0]['color']
        return None

    @staticmethod
    def display_graph(G:nx.Graph, text: bool = True):
        """Draws the graph with colored edges, If text is true it returns a modified adjacency matrix, otherwise shows it graphically and returns None"""
        if not text:
            all_edges = G.edges
            colors = [G[u][v]['color'] for u,v in all_edges]
            pos = nx.circular_layout(G)
            nx.draw(G,pos=pos,edges=all_edges,edge_color=colors,node_color=["gray"]*nx.number_of_nodes(G),with_labels=True)
            plt.show()
            return None
        else:
            adj = nx.adjacency_matrix(G).tocoo()
            w_adj = np.zeros((nx.number_of_nodes(G),nx.number_of_nodes(G)),dtype=int)
            for i,j in zip(adj.row,adj.col):
                w_adj[i][j] = 1 if G.get_edge_data(i,j)['color'] == 'red' else(-1 if G.get_edge_data(i,j)['color'] else 0)
            return w_adj

    @staticmethod
    def new_edge(G: nx.Graph,color: str,edge: tuple):
        """Adds an edge on G if the edge doesn't already exist, otherwise None"""
        if edge in Utils.get_uncolored_edges(G):
            G.add_edge(*edge,color=color)
            return G
        return None

    @staticmethod
    def reward(G: nx.Graph,chain_length: int, player_color: str,):
        """Returns the reward for a state"""
        cycle = Utils.detect_cycle(G,chain_length)
        if cycle is None:
            return 0.0
        else:
            if cycle == player_color:
                return 1.0
            else:
                return -1.0
        return None

    @staticmethod
    def transition(G:nx.Graph,color:str,edge:tuple):
        """Returns a copy of G with the edge added"""
        new_G = deepcopy(G)
        new_G = Utils.new_edge(new_G,color,edge)
        return new_G

    @staticmethod
    def get_uncolored_edges(G:nx.Graph):
        """Returns the edges in the graph that have not been colored"""
        complete_G = nx.complete_graph(nx.number_of_nodes(G))
        uncolored_edges = set(complete_G.edges) - set(G.edges)
        return uncolored_edges

    @staticmethod
    def train(player: Agent,adversary: Agent, number_of_games: int = 3000):
        """Given two Agents, the method will train them against each other until number of games is reached, by default 3000 games"""
        for game_num in range(number_of_games):
            print("Game " + game_num+ ":")
            finished = False
            while not finished:
                finished = player.move()#player makes move
                if finished:
                    break
                finished = adversary.move()#adversary makes move
            player.reset()
            adversary.reset()
        #TODO: Look into tensorboard live reporting for real-time training feedback
        #TODO: Add Ax values to tune, i.e. win rate and total loss

    

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
                
        #TODO: When finished with one Q-Learning model, use those methods to write this

    #TODO: Write Ax optimizing method
        
g = nx.Graph()
g.add_nodes_from([i for i in range(6)])
g = Utils.new_edge(g,'red',(1,2))
g = Utils.new_edge(g,'blue',(3,4))
print(Utils.display_graph(g))
