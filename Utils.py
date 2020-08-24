from Agent import Agent
import ax
from tqdm import tqdm
import torch
from torch import nn
from itertools import combinations
import igraph as ig
import numpy as np
import time
import torch_geometric
from random import choice

colors = {'Red': 1, 'Blue': -1}


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

    @staticmethod
    def display_graph(G: ig.Graph, text: bool = False):
        """Draws the graph with colored edges, If text is true it returns a modified adjacency matrix, otherwise shows it graphically and returns None"""
        if not text:
            layout = G.layout('circle')
            G.vs['label'] = G.vs['name']
            G.vs['color'] = ['grey' for i in range(G.vcount())]
            color_dict = {1: 'red', -1: 'blue'}
            G.es['color'] = [color_dict[weight] for weight in G.es['weight']]
            ig.plot(G, layout=layout)
            ig.plot(G, f'games/{time.strftime("%Y %m %d-%H %M %S")}.png', layout=layout)
        else:
            return G.summary()

    @staticmethod
    def weighted_adj(G: ig.Graph, color: str):
        w_adj = G.get_adjacency(attribute='weight', type=ig.GET_ADJACENCY_UPPER)
        return torch.tensor(list(w_adj), dtype=torch.float)[np.triu_indices(G.vcount(), 1)] * colors[color]

    @staticmethod
    def new_edge(G: ig.Graph, color: str, edge: tuple):
        """Adds an edge on G if the edge doesn't already exist, otherwise None"""
        G.add_edge(*edge, weight=colors[color])
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
        uncolored_edges = set(combinations([i for i in range(G.vcount())], 2)) - set(G.get_edgelist())
        return uncolored_edges

    @staticmethod
    def weight_initialization(m):
        """Initializes a Linear layer with xavier uniform initialization"""
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    @staticmethod
    def graph_to_data(G: ig.Graph, color: str, device: torch.device):
        """Converts a graph to a pytorch Data object"""
        edge_index = torch.tensor(
            [*[list(e) for e in G.get_edgelist()], *[list(e).__reversed__() for e in G.get_edgelist()]],
            dtype=torch.long).t().contiguous()
        if edge_index.size()[0] == 0:
            edge_index = torch.tensor([[], []], dtype=torch.long)
        x = list(G.get_adjacency(attribute='weight'))
        for r in range(len(x)):
            del x[r][r]
        x = torch.tensor(x, dtype=torch.float) * colors[color]
        return torch_geometric.data.Data(edge_index=edge_index, x=x).to(device)

    @staticmethod
    def heuristic_state_score(state: ig.Graph, color: str):
        """Calculates a heuristic score for a state"""
        if color == 'Red':
            color = 1
        else:
            color = -1
        s = state.copy()
        for edge in s.get_edgelist():
            if s[edge[0], edge[1]] == color:
                s.delete_edges([edge])

        o_s = state.copy()
        for edge in o_s.get_edgelist():
            if o_s[edge[0], edge[1]] == color * -1:
                o_s.delete_edges([edge])

        cliques = list(s.cliques(min=3))

        num_of_cliques = len(cliques)

        avg_len_of_cliques = sum([len(c) for c in cliques]) / num_of_cliques if num_of_cliques > 0 else 0

        edge_counter = 0
        for edge in s.get_edgelist():
            s.add_edge(edge[0], edge[1])
            edge_counter += len(list(s.cliques(min=3))) - num_of_cliques
            s.delete_edges([edge])

        mixed_cliques = (set(list(state.cliques(min=3)))-set(cliques))-set(list(o_s.cliques(min=3)))
        primary_cliques = 0
        for clique in mixed_cliques:
            bias = sum([*[state[clique[node], clique[node + 1]] for node in range(len(clique) - 1)],
                           *[state[clique[0], clique[-1]]]])
            if color > 0:
                if bias > 0:
                    primary_cliques += 1
            elif color < 0:
                if bias < 0:
                    primary_cliques += 1

        score = .5 * num_of_cliques + avg_len_of_cliques + edge_counter + primary_cliques
        return score

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
            t = 0
            while not finished:
                finished = self.player.move(self.adversary)# player makes move
                if finished:
                    break
                finished = self.adversary.move(self.player)
                # adversary makes move
            self.player.epoch += 1
            self.adversary.epoch += 1
            if abs(self.player.number_of_moves - self.adversary.number_of_moves) > 1:
                print("Violation of Moves")
                print(self.player.state)
                print(self.adversary.state)
            self.player.write_info_dict()
            self.adversary.write_info_dict()

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
        """Runs ax optimization with no constraints"""
        best_parameters, values, experiment, model = ax.optimize(
            parameters=params,
            evaluation_function=self.train,
            minimize=False,
        )
        return best_parameters

    @staticmethod
    def play_against_random(player: Agent, opp: Agent, trials: int, mcts: bool = False):
        """Tests an agent against a random player"""
        p_win_count = 0
        p_loss_count = 0
        p_tie_count = 0
        o_win_count= 0
        o_loss_count = 0
        o_tie_count = 0
        print('Starting Player')
        for trial in tqdm(range(trials)):
            state = Utils.make_graph(player.number_of_nodes)
            finished = False
            turn = True
            while not finished:
                if turn:
                    if not mcts:
                        max_q, action = player.get_max_q(state)
                    else:
                        player.update_tree(state)
                        player.tree.simulate(player.hyperparameters['Trials'],player.hyperparameters['EPSILON'])
                        _ , action = player.tree.get_best_move()

                    state = Utils.transition(state, player.color, action)

                else:
                    state = Utils.transition(state, 'Red' if player.color == 'Blue' else 'Blue',
                                             choice(list(Utils.get_uncolored_edges(state))))

                turn = not turn
                

                if Utils.reward(state, player.chain_length, player.color) == 1:
                    p_win_count += 1
                    finished = True
                elif Utils.reward(state, player.chain_length, player.color) == -1:
                    p_loss_count += 1
                    finished = True
                elif not Utils.get_uncolored_edges(state):
                    p_tie_count += 1
                    finished = True
        print('Starting opp')
        for trial in range(trials):
            state = Utils.make_graph(opp.number_of_nodes)
            finished = False
            turn = False
            while not finished:
                if turn:
                    if not mcts:
                        max_q, action = opp.get_max_q(state)
                    else:
                        opp.update_tree(state)
                        opp.tree.simulate(opp.hyperparameters['Trials'],opp.hyperparameters['EPSILON'])
                        _ , action = opp.tree.get_best_move()

                    state = Utils.transition(state, opp.color, action)

                else:
                    state = Utils.transition(state, 'Red' if opp.color == 'Blue' else 'Blue',
                                             choice(list(Utils.get_uncolored_edges(state))))
                turn = not turn

                if Utils.reward(state, opp.chain_length, opp.color) == -1:
                    o_win_count += 1
                    finished = True
                elif Utils.reward(state, opp.chain_length, opp.color) == 1:
                    o_loss_count += 1
                    finished = True
                elif not Utils.get_uncolored_edges(state):
                    o_tie_count += 1
                    finished = True
        print(p_win_count,p_loss_count,o_win_count,o_loss_count)
        print(f'Player won {round(p_win_count/trials,3)}% , lost {round(p_loss_count/trials,3)}% and tied {round(p_tie_count/trials,3)}% of games')
        print(f'Opponent won {round(o_win_count/trials,3)}% , lost {round(o_loss_count/trials,3)}% and tied {round(o_tie_count/trials,3)}% of games')
        return p_win_count/trials


    @staticmethod
    def play(player: Agent, opp: Agent, goes_first: bool = True,mcts: bool = False):
        """Allows the user to play a game against an agent, agent will go first by default"""
        finished = False
        replay = True
        agent = player
        agent.reset()
        turn = True
        while replay:
            state = Utils.make_graph(agent.number_of_nodes)
            while not finished:
                if goes_first:
                    if turn:
                        if mcts:
                            opp.update_tree(state)
                            opp.tree.simulate(opp.hyperparameters['Trials'], opp.hyperparameters['EPSILON'])
                            _, action = opp.tree.get_best_move()
                        else:
                            max_q, action = agent.get_max_q(state)
                        state = Utils.transition(state, agent.color, action)
                    else:
                        Utils.display_graph(state)
                        print(f'Enter index, starting from 1, of edge to color')
                        edge = sorted(list(Utils.get_uncolored_edges(state)))[
                            int(input(f'Edges: {sorted(list(Utils.get_uncolored_edges(state)))} ')) - 1]
                        state = Utils.transition(state, 'Red' if agent.color == 'Blue' else 'Blue', edge)
                else:
                    if turn:
                        Utils.display_graph(state)
                        print(f'Enter index, starting from 1, of edge to color')
                        edge = sorted(list(Utils.get_uncolored_edges(state)))[
                            int(input(f'Edges: {sorted(list(Utils.get_uncolored_edges(state)))} ')) - 1]
                        state = Utils.transition(state, 'Red' if agent.color == 'Blue' else 'Blue', edge)
                    else:
                        if mcts:
                            opp.update_tree(state)
                            opp.tree.simulate(opp.hyperparameters['Trials'], opp.hyperparameters['EPSILON'])
                            _, action = opp.tree.get_best_move()
                        else:
                            max_q, action = agent.get_max_q(state)
                        state = Utils.transition(state, agent.color, action)

                turn = not turn

                if Utils.reward(state, agent.chain_length, agent.color) == 1:
                    Utils.display_graph(state)
                    print("Player Won!")
                    finished = True
                elif Utils.reward(state, agent.chain_length, agent.color) == -1:
                    Utils.display_graph(state)
                    print("You Won!")
                    finished = True
                elif not Utils.get_uncolored_edges(state):
                    print('Tie!')
                    finished = True
            while True:
                r = input('Would you like to play again? [y/N] ')
                if not r or r.lower() == 'n':
                    replay = False
                    finished = False
                    break
                elif r.lower() == 'y':
                    replay = True
                    finished = False
                    break
                else:
                    print('Please enter y,n or nothing')
            while True:
                r = input('Would you like to switch colors? [y/N] ')
                if not r or r.lower() == 'n':
                    goes_first = goes_first
                    break
                elif r.lower() == 'y':
                    goes_first = not goes_first
                    if goes_first:
                        agent = player
                    else:
                        agent = opp
                    break
                else:
                    print('Please enter y,n or nothing')
