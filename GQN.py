from Agent import Agent
import random
from Utils import Utils
from time import time_ns
import torch
from collections import deque
from torch_geometric.nn import GCNConv, EdgeConv, GATConv

ALPHA = 0.8


class GCQNetwork(torch.nn.Module):
    def __init__(self, board_size, hidden_layer_size):
        super(GCQNetwork, self).__init__()
        self.input = GCNConv(board_size - 1, board_size - 1)
        self.l2 = GCNConv(board_size - 1, board_size - 1)
        self.l3 = GCNConv(board_size - 1, board_size - 1)

        self.l4 = torch.nn.Linear(board_size * (board_size - 1), hidden_layer_size)
        self.output = torch.nn.Linear(hidden_layer_size, int((board_size * (board_size - 1)) / 2))
        self.board_size = board_size

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        g1 = self.input(x, edge_index)
        g2 = torch.nn.functional.relu(self.l2(g1, edge_index) + g1)
        g3 = torch.nn.functional.relu(self.l3(g2, edge_index) + g2)

        y = g3

        y = torch.nn.functional.relu(torch.flatten(y))
        y = torch.nn.functional.relu(self.l4(y))
        y = self.output(y)
        return y


class EdgeQNetwork(torch.nn.Module):
    def __init__(self, board_size, hidden_layer_size):
        super(EdgeQNetwork, self).__init__()
        inner_network = torch.nn.Sequential(
            torch.nn.Linear(2 * (board_size - 1), 2 * (board_size - 1)),
            torch.nn.Linear(2 * (board_size - 1), (board_size - 1)),
            torch.nn.Linear((board_size - 1), (board_size - 1)),
        )
        self.input = EdgeConv(torch.nn.Sequential(
            torch.nn.Linear(2 * (board_size - 1), 2 * (board_size - 1)),
            torch.nn.Linear(2 * (board_size - 1), (board_size - 1)),
            torch.nn.Linear((board_size - 1), (board_size - 1)),
        ))
        self.l2 = EdgeConv(torch.nn.Sequential(
            torch.nn.Linear(2 * (board_size - 1), 2 * (board_size - 1)),
            torch.nn.Linear(2 * (board_size - 1), (board_size - 1)),
            torch.nn.Linear((board_size - 1), (board_size - 1)),
        ))
        self.l3 = EdgeConv(torch.nn.Sequential(
            torch.nn.Linear(2 * (board_size - 1), 2 * (board_size - 1)),
            torch.nn.Linear(2 * (board_size - 1), (board_size - 1)),
            torch.nn.Linear((board_size - 1), (board_size - 1)),
        ))
        self.l4 = torch.nn.Linear(board_size * (board_size - 1), hidden_layer_size)
        self.output = torch.nn.Linear(hidden_layer_size, int((board_size * (board_size - 1)) / 2))
        self.board_size = board_size

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        g1 = torch.nn.functional.relu(self.input(x, edge_index))
        g2 = torch.nn.functional.relu(self.l2(g1, edge_index) + g1)
        g3 = torch.nn.functional.relu(self.l3(g2, edge_index) + g2)

        y = torch.nn.functional.relu(self.l4(torch.flatten(g3)))
        y = self.output(y)
        return y

class GATCQNetwork(torch.nn.Module):
    def __init__(self, board_size, hidden_layer_size):
        super(GATCQNetwork, self).__init__()
        self.input = GATConv(board_size - 1, board_size - 1,)
        self.l2 = GATConv(board_size - 1, board_size - 1)
        self.l3 = GATConv(board_size - 1, board_size - 1)

        self.l4 = torch.nn.Linear(board_size * (board_size - 1), hidden_layer_size)
        self.output = torch.nn.Linear(hidden_layer_size, int((board_size * (board_size - 1)) / 2))
        self.board_size = board_size

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        g1 = self.input(x,edge_index)
        g2 = (self.l2(g1, edge_index))
        g3 = (self.l3(g2, edge_index))

        y = g3

        y = torch.nn.functional.relu(torch.flatten(y))
        y = torch.nn.functional.relu(self.l4(y))
        y = self.output(y)
        return y



class GQN(Agent):
    def __init__(self, color, hyperparameters, training=True, number_of_nodes: int = 6, chain_length: int = 3):
        super(GQN, self).__init__(color, hyperparameters)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.q_network = GATCQNetwork(number_of_nodes, hyperparameters['HIDDEN_LAYER_SIZE']).to(self.device)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=hyperparameters['LEARNING_RATE'])
        self.loss_fn = torch.nn.MSELoss(reduction='mean')
        if not (type(self.q_network) == GATCQNetwork):
            self.q_network.apply(Utils.weight_initialization)
        self.target_network = GATCQNetwork(number_of_nodes, hyperparameters['HIDDEN_LAYER_SIZE']).to(self.device)
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
        start_q_time = time_ns()
        # update q table
        if self.training:
            self.update_q(self.state, self.action, reward)
        opponent.opp_move(self.state, self.action, self.color)
        self.state = new_state
        self.avg_move_time = (self.avg_move_time*(self.number_of_moves-1 if self.number_of_moves > 1 else 1) + (time_ns() - start_time)) / self.number_of_moves
        # If its the end, return False, otherwise make an action
        if (r := Utils.reward(self.state, self.chain_length, self.color)) == 1 or len(
                Utils.get_uncolored_edges(self.state)) < 1:
            self.wins += r
            if r == 0:
                Utils.display_graph(self.state)
                raise Exception
            return True
        else:
            return False

    def update_q(self, state, action, reward, color=None):
        if color is None:
            color = self.color
        self.experience_buffer.append((Utils.graph_to_data(state, color, self.device), action, reward,
                                       Utils.graph_to_data(Utils.transition(state, color, action), color, self.device)))
        self.update_count += 1
        if len(self.experience_buffer) > self.hyperparameters['BATCH_SIZE']:
            sample = random.sample(self.experience_buffer, self.hyperparameters['BATCH_SIZE'])
            training_input = torch.empty(self.hyperparameters['BATCH_SIZE'],
                                         int((self.number_of_nodes * (self.number_of_nodes - 1)) / 2),
                                         dtype=torch.float, requires_grad=True)
            training_output = torch.empty(self.hyperparameters['BATCH_SIZE'],
                                          int((self.number_of_nodes * (self.number_of_nodes - 1)) / 2),
                                          dtype=torch.float, requires_grad=True)
            mem_count = 0
            # current_states = torch.tensor([exp[0] for exp in sample])
            # new_states = torch.tensor([exp[3] for exp in sample])
            # current_qs = self.q_network.forward(current_states)
            # current_qs = torch.tensor([self.q_network.forward(exp[0]) for exp in sample]
            # with torch.no_grad():
            # new_qs_max = torch.max(self.target_network.forward(new_states), 1)
            # new_qs_max = torch.max(torch.tensor([self.target_network.forward(exp[3]) for exp in sample]), 1)
            for mem in sample:
                s, a, r, ns = mem
                current_q = self.q_network.forward(s)
                with torch.no_grad():
                    max_q = torch.max(self.target_network.forward(ns)).item()
                with torch.no_grad():
                    output = current_q.detach().clone().requires_grad_(True)
                    ind = int((a[0] * (self.number_of_nodes - 1) + a[1] - (a[0] * (a[0] + 1)) / 2) - 1)
                    output[ind] = (1 - ALPHA) * (output[ind]) + ALPHA * (r + self.hyperparameters['GAMMA'] * max_q)
                    training_input[mem_count] = current_q
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
        q_val = self.q_network.forward(Utils.graph_to_data(state, self.color, self.device))[
            int((action[0] * (self.number_of_nodes - 1) + action[1] - (action[0] * (action[0] + 1)) / 2) - 1)].item()
        return q_val

    def get_max_q(self, state):
        # Getting max Q-value
        if len(Utils.get_uncolored_edges(state)) < 1:
            return 0, None
        max_q = None
        max_actions = []
        mqs = torch.max(self.q_network(Utils.graph_to_data(state, self.color, self.device)))
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
            reward = Utils.reward(Utils.transition(state, c, action), self.chain_length, self.color)
            self.update_q(state, action, reward, color=c)
        self.state = Utils.transition(state, c, action)

    def reset(self):
        self.state = Utils.make_graph(self.number_of_nodes)
        self.action = None
        self.loss = 0
        self.number_of_moves = 0

    def hard_reset(self):
        self.reset()
        self.q_network = GATCQNetwork(self.number_of_nodes, self.hyperparameters['HIDDEN_LAYER_SIZE'])
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.hyperparameters['LEARNING_RATE'])
        self.loss_fn = torch.nn.MSELoss(reduction='mean')
        if not (type(self.q_network) == GATCQNetwork):
            self.q_network.apply(Utils.weight_initialization)
        self.target_network = GATCQNetwork(self.number_of_nodes, self.hyperparameters['HIDDEN_LAYER_SIZE'])
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.epoch = 0
        self.wins = 0
        self.writer.close()

    def store(self):
        self.save_model(self.q_network, self.target_network, self.optimizer)

    def open(self, path):
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_model_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.loss = checkpoint['loss']
