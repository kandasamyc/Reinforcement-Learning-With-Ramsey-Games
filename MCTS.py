import igraph as ig
import math
from random import choice,random
from Utils import Utils
from Agent import Agent
from time import time_ns


class MCTSGameTree:
    def __init__(self, k, board_size, state=None, parent=None, c=None, depth=None,color=None):
        if state is None:
            self.state = Utils.make_graph(board_size)
        else:
            self.state = state

        self.wins = 0
        self.times_visited = 1
        self.val = 0
        self.terminality = False
        self.children = {}
        self.parent = parent
        if parent is not None:
            self.c = parent.c
        elif c is not None:
            self.c = c
        else:
            self.c = math.sqrt(2)
        self.chain_length = k
        if parent is not None:
            self.color = 'Red' if parent.color == 'Blue' else 'Blue'
        elif color is not None:
            self.color = color
        else:
            self.color = 'Red'
        if parent is not None:
            self.depth = parent.depth - 1
        elif depth is not None:
            self.depth = depth
        else:
            self.depth = 4
        self.value = 0
        self.board_size = board_size
        self.update_terminality()

    def get_best_move(self):
        max_score = None
        max_actions = []
        for ac,child in self.children.items():
            child_score = child.UCTScore()
            if max_score is None or child_score > max_score:
                max_score = child_score
                max_actions = [ac]
            elif child_score == max_score:
                max_actions.append(ac)
        action_choice = choice(max_actions)
        return self.children[action_choice], action_choice

    def update_terminality(self):
        if not Utils.get_uncolored_edges(self.state) or Utils.reward(self.state, self.chain_length,
                                                                     self.color) != 0 or self.depth == 0:
            self.terminality = True
            self.value = Utils.reward(self.state, self.chain_length, self.color)
        else:
            self.children = {action:
                MCTSGameTree(self.chain_length, self.board_size, state=Utils.transition(self.state, self.color, action), parent=self)
                for action
                in Utils.get_uncolored_edges(self.state)}

    def UCTScore(self):
        return self.wins / self.times_visited + self.c * math.sqrt(
            math.log(self.parent.times_visited) / self.times_visited)

    def simulate(self, trials, epsilon):
        for trial in range(trials):
            current_action = choice(list(self.children.values()))
            while current_action.terminality is False:
                if random() < epsilon:
                    current_action = choice(list(current_action.children.values()))
                else:
                    cs = list(current_action.children.values())
                    heur_vals = [Utils.heuristic_state_score(child.state,self.color) for child in cs]
                    current_action = cs[heur_vals.index(max(heur_vals))]
            reward = Utils.reward(current_action.state, current_action.chain_length,
                                       self.color)
            while current_action is not self.parent:
                current_action.times_visited += 1
                current_action.wins += reward
                current_action = current_action.parent


class MCTSAgent(Agent):
    def __init__(self, hyperparameters, color, chain_length, board_size):
        super(MCTSAgent, self).__init__(color, hyperparameters)
        self.tree = MCTSGameTree(chain_length, board_size,c=hyperparameters['C'])
        self.hyperparameters = hyperparameters
        self.state = Utils.make_graph(board_size)
        self.color = color
        self.chain_length = chain_length
        self.number_of_nodes = board_size

    def move(self, opp):
        self.tree = MCTSGameTree(self.chain_length, self.number_of_nodes, state=self.state, c=self.hyperparameters['C'],
                                 color=self.color)
        self.number_of_moves += 1
        start_time = time_ns()
        self.tree.simulate(self.hyperparameters['Trials'],self.hyperparameters['EPSILON'])
        node, action = self.tree.get_best_move()
        self.avg_move_time = (self.avg_move_time + (time_ns()-start_time))/2
        self.state = Utils.transition(self.state, self.color, action)
        if Utils.reward(self.state,self.chain_length,self.color) == 1 or not Utils.get_uncolored_edges(self.state):
            self.wins += 1
            return True
        else:
            opp.state = self.state
            return False

    def reset(self):
        self.state = Utils.make_graph(self.number_of_nodes)
        self.number_of_moves = 0

    def hard_reset(self):
        self.reset()

    def update_tree(self,state):
        self.tree = MCTSGameTree(self.chain_length, self.number_of_nodes, state=state, c=self.hyperparameters['C'],
                                 color=self.color)