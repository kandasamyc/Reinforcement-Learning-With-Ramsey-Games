from torch.utils.tensorboard import SummaryWriter
from torch import save
import time
import tracemalloc
import pandas

class Agent(object):

    def __init__(self, color: str, hyperparameters: dict):
        self.color = color
        self.hyperparameters = hyperparameters
        self.comment = " "
        for p_name, param in hyperparameters.items():
            self.comment += p_name + "=" + str(param) + " "
        self.comment = " "+self.comment.strip() + str(self.color)
        self.writer = SummaryWriter(comment=self.comment)
        self.epoch = 0
        self.wins = 0
        self.loss = 0
        self.number_of_moves = 0
        self.avg_move_time = 0
        self.data = {}
        tracemalloc.start()

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

    def write_MCTS_info(self):
        self.writer.add_scalar('Win Rate', self.wins/self.epoch, self.epoch)
        self.writer.add_scalar('Move Time', self.avg_move_time, self.epoch)
        self.writer.add_scalar('Number of Moves', self.number_of_moves, self.epoch)
        self.writer.flush()

    def save_model(self,q_model,target_model,optimizer):
        save({
            'epoch': self.epoch,
            'q_model_state_dict': q_model.state_dict(),
            'target_model_state_dict': target_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': self.loss,
        },
        f'models/{self.comment.strip()},{time.strftime("%Y %m %d-%H %M %S")}.pth')

    def write_info_dict(self):
        self.data[self.epoch] = [self.loss,self.wins/self.epoch,self.avg_move_time,self.number_of_moves,tracemalloc.get_traced_memory()[0]]

    def save_dict(self):
        df = pandas.DataFrame.from_dict(self.data)
        df.to_pickle(f'saved_data/{self.comment.strip()},{time.strftime("%Y %m %d-%H %M %S")}.pkl.xz')
        tracemalloc.stop()