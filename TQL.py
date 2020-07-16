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

    def move(self,state):
        #Update network based on the state the opponent just put the environment in
        if training and self.action is not None:
            self.update_q(state,Utils.reward(state,self.chain_length,self.color))
        
        #If its the end, return False, otherwise make an action
        if len(Utils.get_uncolored_edges(state)) < 1 or Utils.reward(state,self.chain_length,self.color):
            self.reset()
            return False, None
        else:
            self.state = state

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

        #update q table
        self.update_q(Utils.transition(self.state,self.color,self.action),reward)

        return True,Utils.transition(self.state,self.color,self.action)

        

    def update_q(self,new_state,reward):
        pass

    def get_q(self):
        pass

    def reset():
        self.state = None
        self.action = None