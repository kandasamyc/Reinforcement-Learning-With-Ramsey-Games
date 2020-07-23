import igraph as ig
import math
from random import choice
from Utils import Utils
from Agent import Agent
from time import time_ns

class MCTSGameTree:
    def __init__(self, k, board_size, state=None, parent=None):
        if state is not None:
            self.state = ig.Graph()
            self.state.add_vertices([i for i in range(board_size)])
        else:
            self.state = state

        self.wins = 0
        self.times_visited = 0
        self.val = 0
        self.terminality = False
        self.children = []
        self.parent = parent
        self.update_terminality()
        self.c = parent.c
        self.chain_length = k
        self.color = 'Red' if parent.color is 'Blue' else 'Red'
        self.depth = parent.depth - 1
        self.value = 0
        self.board_size = board_size

    def get_best_move(self):
        max_score = None
        max_actions = []
        for child in self.children:
            child_score = child.UCTScore()
            if max_score is None or child_score > max_score:
                max_score = child_score
                max_actions = [child]
            elif child.score == max_score:
                max_actions.append(child)
        return self.children[Utils.get_uncolored_edges(self.state).index(choice(max_actions))]

    def update_terminality(self):
        if not Utils.get_uncolored_edges(self.state) or Utils.reward(self.state, self.chain_length, self.color) != 0:
            self.terminality = True
            self.value = Utils.reward(self.state, self.chain_length, self.color)
        else:
            self.children = [
                MCTSGameTree(self.k, self.board_size, Utils.transition(self.state, self.color, action), self) for action
                in Utils.get_uncolored_edges(self.state)]

    def UCTScore(self):
        return self.wins / self.times_visited + self.c * math.sqrt(
            math.log(self.parent.times_visited) / self.times_visited)

    def simulate(self, trials):
        for trial in range(trials):
            current_action = choice(self.children)
            while current_action.terminality is False:
                current_action = choice(current_action.children())
            reward = 1 if Utils.reward(current_action.state, current_action.chain_length,
                                       current_action.color) > 0 else 0
            while current_action is not self.parent:
                current_action.times_visited += 1
                current_action.wins += reward
                current_action = current_action.parent


class MCTSAgent(Agent):
    def __init__(self, hyperparameters, color, chain_length, board_size):
        super(MCTSAgent, self).__init__(color, hyperparameters)
        self.player = MCTSGameTree(chain_length, board_size)
        self.adversary = MCTSGameTree(chain_length,board_size)
        self.hyperparameters = hyperparameters

    def train(self,games):
        for game in range(games):
            self.epoch += 1
            self.number_of_moves = 0
            while True:
                start_time = time_ns()
                self.player.simulate(self.hyperparameters['Training Trials'])
                self.player = self.player.get_best_move()
                self.avg_move_time = (self.avg_move_time+start_time-time_ns())/2
                if self.player.terminality is True:
                    self

