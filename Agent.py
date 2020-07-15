import networkx as nx
from torch.utils.tensorboard import SummaryWriter

HP_LIST = ["Epsilon","Gamma","Learning Rate", "Target Model Update Frequency", "Batch Size", "Memory Size"]

class Agent:
    def __init__(self,color: str,hyperparameters: list):
        self.color = color
        self.hyperparameters = hyperparameters
        comment = ""
        for param, p_name in zip(hyperparameters,HP_LIST):
            comment += p_name + "=" + str(param) + " "
        self.writer = SummaryWriter(comment=comment)
        self.epoch = 0
        self.wins = 0
        self.loss = 0

    def write_info(self,info: list):
        self.writer.add_scalar('Loss', info[0], self.epoch)
        self.writer.add_scalar('Win Rate', info[1] / info[2], self.epoch)
        self.writer.add_scalar('Average Move Time', info[3], self.epoch)
        self.writer.add_scalar('Number of Moves',info[4],self.epoch)
        self.writer.flush()

    def write_network_info(self,layers:list,layer_names: list):
        for layer, name in zip(layers,layer_names):
            self.writer.add_histogram(name + " Bias", layer.bias, self.epoch)
            self.writer.add_histogram(name + " Weights", layer.weight, self.epoch)
        self.writer.flush()








