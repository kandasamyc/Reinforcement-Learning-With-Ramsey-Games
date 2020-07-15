from Agent import Agent
import pandas as pd
import random
from Utils import Utils
import networkx as nx

class TQL(Agent):
    def __init__(self, color, hyperparameters,training=True,number_of_nodes:int=6,chain_length:int=3):
        self.q_table = pd.DataFrame(columns=['q'])
        self.state = Utils.make_graph(number_of_nodes)
        self.chain_length = chain_length
        self.action = None
        self.color = color
        self.hyperparameters = hyperparameters
        self.training = training
        super().__init__(color, hyperparameters)

    def move(self):
        if random.random() < self.hyperparameters and training:
            self.action = random.choice(Utils.get_uncolored_edges(self.state))
        else:
            #Getting max Q-value
            max_q = None
            max_actions = []
            for edge in Utils.get_uncolored_edges(self.state):
                new_G = Utils.transition(self.state,self.color,edge)
                if self.get_q(new_G) > max_q or max_q is None:
                    max_q = self.get_q(new_G)
                    max_actions = [edge]
                elif self.get_q == max_q:
                    max_actions.append(edge)
            self.action = random.choice(max_actions)
        
        #compute reward
        reward = Utils.reward(Utils.transition(self.state,self.color,self.action),self.chain_length,self.color)

        

    def update_q(self):
        pass

    def get_q(self):
        pass

    def opp_move():

    def reset():
        self.state = None
        self.action = None