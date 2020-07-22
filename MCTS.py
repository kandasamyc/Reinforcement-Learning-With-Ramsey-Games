import igraph as ig


class MCTSGameTree:
    def __init__(self, k, board_size, state=None):
        if state is not None:
            self.state = ig.Graph()
            self.state.add_vertices([i for i in range(board_size)])
        else:
            self.state = state

        self.wins = 0
        self.times_visited = 0
