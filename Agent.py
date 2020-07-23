from torch.utils.tensorboard import SummaryWriter



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

    def write_MCTS_info(self):
        self.writer.add_scalar('Win Rate', self.wins/self.epoch, self.epoch)
        self.writer.add_scalar('Move Time', self.avg_move_time, self.epoch)
        self.writer.add_scalar('Number of Moves', self.number_of_moves, self.epoch)
        self.writer.flush()