# from Utils import Utils
# from DQN import DQN
# from TQL import TQL

from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from Agent import Agent
import ax
from tqdm import tqdm
import torch
from torch import nn
from itertools import permutations
import igraph as ig

colors = {'Red':2,'Blue':-1}

# noinspection PyUnresolvedReferences
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
    def display_graph(G: ig.Graph, text: bool = True):
        """Draws the graph with colored edges, If text is true it returns a modified adjacency matrix, otherwise shows it graphically and returns None"""
        # if not text:
        #     all_edges = G.edges
        #     colors = [G[u][v]['color'] for u, v in all_edges]
        #     pos = nx.circular_layout(G)
        #     nx.draw(G, pos=pos, edges=all_edges, edge_color=colors, node_color=["gray"] * nx.number_of_nodes(G),
        #             with_labels=True)
        #     plt.show()
        #     return None
        # else:
        #     return list(Utils.weighted_adj(G))

    @staticmethod
    def weighted_adj(G: ig.Graph,color:str):
        w_adj = G.get_adjacency(attribute='weight')
        return torch.tensor(list(w_adj),dtype=torch.float)*colors[color]

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
        G.es['weight'] = 1.0
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
        uncolored_edges = set(permutations([i for i in range(G.vcount())],2)) - set(G.get_edgelist())
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

HP_LIST = ["Epsilon", "Gamma", "Learning Rate", "Target Model Update Frequency", "Batch Size", "Memory Size"]


class Agent(object):

    def __init__(self, color: str, hyperparameters: dict):
        self.color = color
        self.hyperparameters = hyperparameters
        comment = " "
        for p_name, param in hyperparameters.items():
            comment += p_name + "=" + str(param) + " "
        comment = " "+comment.strip() + str(self.color)
        self.writer = SummaryWriter(comment=comment)
        self.epoch = 0
        self.wins = 0
        self.loss = 0
        self.number_of_moves = 0
        self.avg_move_time = 0

    def write_info(self):
        self.writer.add_scalar('Loss', self.loss, self.epoch)
        self.writer.add_scalar('Win Rate', self.wins/self.epoch, self.epoch)
        self.writer.add_scalar('Move Time', self.avg_move_time, self.epoch)
        self.writer.add_scalar('Number of Moves', self.number_of_moves, self.epoch)
        self.writer.flush()

    def write_network_info(self, layers: list, layer_names: list):
        for layer, name in zip(layers, layer_names):
            self.writer.add_histogram(name + " Bias", layer.bias, self.epoch)
            self.writer.add_histogram(name + " Weights", layer.weight, self.epoch)
        self.writer.flush()

    def update_writer(self,hyperparameters):
        comment = " "
        for p_name, param in hyperparameters.items():
            comment += p_name + "=" + str(param) + " "
        comment = " " + comment.strip() + str(self.color)
        self.writer = SummaryWriter(comment=comment)

from Agent import Agent
import random
from Utils import Utils
from time import time_ns
import torch
from collections import deque


class QNetwork(torch.nn.Module):
    def __init__(self, board_size, hidden_layer_size):
        super(QNetwork, self).__init__()
        self.input = torch.nn.Linear(board_size ** 2, hidden_layer_size)
        self.l2 = torch.nn.Linear(hidden_layer_size, hidden_layer_size)
        self.output = torch.nn.Linear(hidden_layer_size, board_size ** 2)

    def forward(self, x):
        y = torch.nn.functional.relu(self.input(x))
        y = torch.nn.functional.relu(self.l2(y))
        y = torch.nn.functional.relu(self.output(y))
        return y


class DQN(Agent):
    def __init__(self, color, hyperparameters, training=True, number_of_nodes: int = 6, chain_length: int = 3):
        super(DQN, self).__init__(color, hyperparameters)
        self.q_network = QNetwork(number_of_nodes,hyperparameters['HIDDEN_LAYER_SIZE'])
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=hyperparameters['LEARNING_RATE'])
        self.loss_fn = torch.nn.MSELoss(reduction='sum')
        self.q_network.apply(Utils.weight_initialization)
        self.target_network = QNetwork(number_of_nodes,hyperparameters['HIDDEN_LAYER_SIZE'])
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.state = Utils.make_graph(number_of_nodes)
        self.chain_length = chain_length
        self.action = None
        self.color = color
        self.hyperparameters = hyperparameters
        self.training = training
        self.experience_buffer = deque(maxlen=hyperparameters['BUFFER_SIZE'])
        self.update_count = 0
        self.number_of_nodes = number_of_nodes

    def move(self, opponent):
        # Update network based on the state the opponent just put the environment in
        start_time = time_ns()
        self.number_of_moves += 1
        self.hyperparameters['EPSILON'] *= self.hyperparameters['EPSILON_DECAY']
        if random.random() < self.hyperparameters["EPSILON"] and self.training:
            self.action = random.choice(list(Utils.get_uncolored_edges(self.state)))
        else:
            max_q, self.action = self.get_max_q(self.state)

        new_state = Utils.transition(self.state, self.color, self.action)
        # compute reward
        reward = Utils.reward(new_state, self.chain_length, self.color)

        # update q table
        if self.training:
            self.update_q(self.state, self.action, reward)
        opponent.opp_move(self.state, self.action, self.color)
        self.state = new_state

        # If its the end, return False, otherwise make an action
        if len(Utils.get_uncolored_edges(self.state)) < 1 or Utils.reward(self.state, self.chain_length,
                                                                          self.color) == 1:
            self.wins += 1
            self.avg_move_time = (self.avg_move_time + (time_ns() - start_time)) / 2.0
            return True
        else:
            return False

    def update_q(self, state, action, reward, color=None):
        if color is None:
            color = self.color
        self.experience_buffer.append((torch.flatten(Utils.weighted_adj(state, color)), action, reward,
                                       torch.flatten(
                                           Utils.weighted_adj(Utils.transition(state, color, action), color))))
        self.update_count += 1
        if len(self.experience_buffer) > self.hyperparameters['BATCH_SIZE']:
            sample = random.sample(self.experience_buffer, self.hyperparameters['BATCH_SIZE'])
            training_input = torch.empty(self.hyperparameters['BATCH_SIZE'], self.number_of_nodes**2,
                                         dtype=torch.float, requires_grad=True)
            training_output = torch.empty(self.hyperparameters['BATCH_SIZE'], self.number_of_nodes**2,
                                          dtype=torch.float, requires_grad=True)
            mem_count = 0
            current_states = torch.stack([exp[0] for exp in sample])
            new_states = torch.stack([exp[3] for exp in sample])
            current_qs = self.q_network.forward(current_states)
            with torch.no_grad():
                new_qs_max = torch.max(self.target_network.forward(new_states),1)
            for mem in sample:
                s, a, r, ns = mem
                max_q = new_qs_max[0][mem_count].item()
                with torch.no_grad():
                    output = current_qs[mem_count].detach().clone().requires_grad_(True)
                    output[a[0] * self.number_of_nodes + a[1]] = r + self.hyperparameters['GAMMA'] * max_q
                    training_input[mem_count] = current_qs[mem_count]
                    training_output[mem_count] = output
                mem_count += 1
            loss = self.loss_fn(training_input, training_output.detach())
            self.loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.update_count % self.hyperparameters['TARGET_MODEL_SYNC']:
                self.target_network.load_state_dict(self.q_network.state_dict())

    def get_q(self, state, action):
        q_val = self.q_network.forward(torch.flatten(Utils.weighted_adj(state, self.color)))[
            action[0] * self.number_of_nodes + action[1]].item()
        return q_val

    def get_max_q(self, state):
        # Getting max Q-value
        if len(Utils.get_uncolored_edges(state)) < 1:
            return 0, None
        max_q = None
        max_actions = []
        mqs = torch.max(self.q_network(torch.flatten(Utils.weighted_adj(state, self.color))))
        for edge in Utils.get_uncolored_edges(state):
            if max_q is None or self.get_q(state, edge) > max_q:
                max_q = self.get_q(state, edge)
                max_actions = [edge]
            elif self.get_q(state, edge) == max_q:
                max_actions.append(edge)
        action = random.choice(max_actions)
        return max_q, action

    def opp_move(self, state, action, c):
        if self.training and self.action is not None:
            reward = Utils.reward(state, self.chain_length, c)
            self.update_q(state, action, reward, color=c)
        self.state = Utils.transition(state, c, action)

    def reset(self):
        self.state = Utils.make_graph(self.number_of_nodes)
        self.action = None
        self.loss = 0
        self.number_of_moves = 0

    def hard_reset(self):
        self.reset()
        self.q_network = QNetwork(self.number_of_nodes,self.hyperparameters['HIDDEN_LAYER_SIZE'])
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.hyperparameters['LEARNING_RATE'])
        self.loss_fn = torch.nn.MSELoss(reduction='sum')
        self.q_network.apply(Utils.weight_initialization)
        self.target_network = QNetwork(self.number_of_nodes,self.hyperparameters['HIDDEN_LAYER_SIZE'])
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.epoch = 0
        self.wins = 0
        self.writer.close()


player = DQN('Red', {"GAMMA": .1, 'EPSILON': .6, 'HIDDEN_LAYER_SIZE': 20, 'BUFFER_SIZE': 1000, 'BATCH_SIZE': 150,
                     'TARGET_MODEL_SYNC': 6, 'LEARNING_RATE': 1e-2, 'EPSILON_DECAY': .9997})
opp = DQN('Blue', {"GAMMA": .1, 'EPSILON': .6, 'HIDDEN_LAYER_SIZE': 20, 'BUFFER_SIZE': 1000, 'BATCH_SIZE': 150,
                   'TARGET_MODEL_SYNC': 6, 'LEARNING_RATE': 1e-2, 'EPSILON_DECAY': .9997})
u = Utils(player, opp, 1500)
print(u.optimize_training([
    {
        'name': 'GAMMA',
        'type': 'range',
        'bounds': [.0001, .4]
    },
    {
        'name': 'EPSILON',
        'type': 'range',
        'bounds': [.4, .6]
    },
    {
        'name': 'HIDDEN_LAYER_SIZE',
        'type': 'range',
        'bounds': [20, 100]
    },
    {
        'name': 'BUFFER_SIZE',
        'type': 'range',
        'bounds': [100, 500]
    }, {
        'name': 'BATCH_SIZE',
        'type': 'range',
        'bounds': [50, 100]
    }, {
        'name': 'TARGET_MODEL_SYNC',
        'type': 'range',
        'bounds': [4, 10]
    },
    {
        'name': 'LEARNING_RATE',
        'type': 'range',
        'bounds': [1e-4, .1]
    },
    {
        'name': 'EPSILON_DECAY',
        'type': 'range',
        'bounds': [.9997, .999997]
    }

]))

# player = TQL('Red', {"GAMMA": .1, 'EPSILON': .6, 'ALPHA':.6})
# opp = TQL('Blue', {"GAMMA": .1, 'EPSILON': .6, 'ALPHA':.6})
# u = Utils(player, opp, 3000)
# u.train()
