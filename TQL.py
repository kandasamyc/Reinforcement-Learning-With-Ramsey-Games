from Agent import Agent
import random
from Utils import Utils
from time import time_ns
import csv
import time
import pandas as pd
from igraph import Graph


class TQL(Agent):
    def __init__(self, color, hyperparameters, training=True, number_of_nodes: int = 6, chain_length: int = 3):
        super(TQL, self).__init__(color, hyperparameters)
        self.q_table = {}
        self.state = Utils.make_graph(number_of_nodes)
        self.chain_length = chain_length
        self.action = None
        self.color = color
        self.hyperparameters = hyperparameters
        self.training = training
        self.number_of_nodes = number_of_nodes

    def move(self, opponent:Agent):
        """ Performs a single move for a TQL Agent and return a boolean representing if the game is finished"""
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
        self.update_q(self.state, self.action, reward)
        opponent.opp_move(self.state, self.action, self.color)
        self.state = new_state

        # If its the end, return False, otherwise make an action
        if (r:=Utils.reward(self.state, self.chain_length,self.color)) == 1 or len(Utils.get_uncolored_edges(self.state)) < 1:
            if r == 1:
                self.wins += 1
            else:
                Utils.display_graph(self.state)
                raise Exception
            self.avg_move_time = (self.avg_move_time*(self.number_of_moves-1) + (time_ns() - start_time)) / self.number_of_moves
            return True
        else:
            return False

    def update_q(self, state:Graph, action:tuple, reward:int, color:str=None):
        """ Updates the q-value in the table, or adds it if it doesn't exist. Index is the adjacency of state and action"""
        if color is None:
            color = self.color
        current_q = self.get_q(state, action)
        max_q, _ = self.get_max_q(Utils.transition(state, color, action))
        new_q = (1 - self.hyperparameters['ALPHA']) * current_q + self.hyperparameters['ALPHA'] * (
                    reward + self.hyperparameters['GAMMA'] * max_q)
        self.loss = (self.loss + (new_q - current_q))
        self.q_table[str(list(state.get_adjacency(attribute="weight"))) + str(action)] = new_q

    def get_q(self, state:Graph, action):
        """Gets a single q-value from the table"""
        try:
            q_val = self.q_table[str(list(state.get_adjacency(attribute="weight"))) + str(action)]
            return q_val
        except Exception:
            self.q_table[str(list(state.get_adjacency(attribute="weight"))) + str(action)] = 0.0
            return 0.0

    def get_max_q(self, state:Graph):
        """Returns the max q-value and its corresponding action for the state"""
        # Getting max Q-value
        if len(Utils.get_uncolored_edges(state)) < 1:
            return 1, None
        max_q = None
        max_actions = []
        for edge in Utils.get_uncolored_edges(state):
            q_val = self.get_q(state, edge)
            if max_q is None or q_val > max_q:
                max_q = q_val
                max_actions = [edge]
            elif q_val == max_q:
                max_actions.append(edge)
        action = random.choice(max_actions)
        return max_q, action

    def opp_move(self, state:Graph, action:tuple, c:str):
        """Updates the agent based on the opponents move"""
        if self.training and self.action is not None:
            reward = Utils.reward(Utils.transition(state, c, action), self.chain_length, self.color)
            self.update_q(state, action, reward, color=c)
        self.state = Utils.transition(state, c, action)

    def reset(self):
        """Resets the agent for another game"""
        self.state = Utils.make_graph(self.state.vcount())
        self.action = None
        self.loss = 0
        self.number_of_moves = 0

    def hard_reset(self):
        """Fully resets the agent"""
        self.reset()
        self.q_table = {}
        self.epoch = 0
        self.wins = 0
        self.writer.close()

    def store(self):
        """Stores a q-table"""
        df = pd.DataFrame.from_dict(self.q_table,orient='index')
        df.to_pickle(f'models/{self.comment.strip()},{time.strftime("%Y %m %d-%H %M %S")}.pkl.xz')

    def load(self, path):
        """Loads a q-table"""
        self.q_table = pd.read_pickle(path).to_dict()
