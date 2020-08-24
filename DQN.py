from Agent import Agent
import random
from Utils import Utils
from time import time_ns
import torch
from collections import deque

ALPHA = .8


class QNetwork(torch.nn.Module):
    def __init__(self, board_size, hidden_layer_size):
        super(QNetwork, self).__init__()
        self.input = torch.nn.Linear(int((board_size * (board_size - 1)) / 2), hidden_layer_size)
        self.l2 = torch.nn.Linear(hidden_layer_size, hidden_layer_size)
        self.l3 = torch.nn.Linear(hidden_layer_size, hidden_layer_size // 2)
        self.l4 = torch.nn.Linear(hidden_layer_size // 2, hidden_layer_size // 2)
        self.output = torch.nn.Linear(hidden_layer_size // 2, int((board_size * (board_size - 1)) / 2))

    def forward(self, x):
        y = torch.nn.functional.relu(self.input(x))
        y = torch.nn.functional.relu(self.l2(y))
        y = torch.nn.functional.relu(self.l3(y))
        y = torch.nn.functional.relu(self.l4(y))
        y = self.output(y)
        return y


class DQN(Agent):
    def __init__(self, color, hyperparameters, training=True, number_of_nodes: int = 6, chain_length: int = 3):
        super(DQN, self).__init__(color, hyperparameters)
        self.q_network = QNetwork(number_of_nodes, hyperparameters['HIDDEN_LAYER_SIZE'])
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=hyperparameters['LEARNING_RATE'],
                                          amsgrad=False)
        self.loss_fn = torch.nn.MSELoss(reduction='sum')
        self.q_network.apply(Utils.weight_initialization)
        self.target_network = QNetwork(number_of_nodes, hyperparameters['HIDDEN_LAYER_SIZE'])
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
        try:
            opponent.opp_move(self.state, self.action, self.color)
        except AttributeError:
            opponent.state = self.state
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
        self.experience_buffer.append((Utils.weighted_adj(state, color), action, reward,
                                       Utils.weighted_adj(Utils.transition(state, color, action), color)))
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
            current_states = torch.stack([exp[0] for exp in sample])
            new_states = torch.stack([exp[3] for exp in sample])
            current_qs = self.q_network.forward(current_states)
            m_current_qs = current_qs.detach_().clone().requires_grad_(True)
            with torch.no_grad():
                new_qs_max = torch.max(self.target_network.forward(new_states), 1)
            for mem, max_q, output in zip(sample, new_qs_max[0], m_current_qs):
                s, a, r, ns = mem
                max_q = max_q.item()
                with torch.no_grad():
                    ind = int((a[0] * (self.number_of_nodes - 1) + a[1] - (a[0] * (a[0] + 1)) / 2) - 1)
                    output[ind] = (1 - ALPHA) * (output[ind]) + ALPHA * (r + self.hyperparameters['GAMMA'] * max_q)
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
        action_index = int((action[0] * (self.number_of_nodes - 1) + action[1] - (action[0] * (action[0] + 1)) / 2) - 1)
        q_val = self.q_network.forward(Utils.weighted_adj(state, self.color))[action_index].item()
        return q_val

    def get_max_q(self, state):
        # Getting max Q-value
        u_es = Utils.get_uncolored_edges(state)
        if len(u_es) < 1:
            return 0, None
        max_q = None
        max_actions = []
        for edge in u_es:
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
        self.q_network = QNetwork(self.number_of_nodes, self.hyperparameters['HIDDEN_LAYER_SIZE'])
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.hyperparameters['LEARNING_RATE'],
                                          amsgrad=False)
        self.loss_fn = torch.nn.MSELoss(reduction='mean')
        self.q_network.apply(Utils.weight_initialization)
        self.target_network = QNetwork(self.number_of_nodes, self.hyperparameters['HIDDEN_LAYER_SIZE'])
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
