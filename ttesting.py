from itertools import combinations
from statsmodels.stats.weightstats import ttest_ind
import pandas as pd
import plotly.graph_objects as go
from sqlalchemy import create_engine
import numpy

engine = create_engine("mysql+pymysql://root:robotics4099@localhost/internship data")
connection = engine.connect()

count = 0

def increment():
    global count
    count = count+1

def plot_epochs(models, titles=[], legend=[],l=False):
    d = []
    for i in range(len(models)):
        d.append(go.Scatter(name=f'{legend[i]} Player', x=[i for i in range(1, len(models[i][0]) + 1)],
                            y=[i for i in models[i][0]]))
        d.append(go.Scatter(name=f'{legend[i]} Opponent', x=[i for i in range(1, len(models[i][1]) + 1)],
                            y=[i for i in models[i][1]]))

    fig = go.Figure(data=d)
    if l:
        fig.update_yaxes(type="log")
    fig.update_layout(template='none')
    fig.update_layout(
        title=titles[0],
        xaxis_title='Epoch',
        yaxis_title=titles[1],
        legend_title='Model',
    )
    fig.show()
    fig.write_image(f'./images/fig{count}.png')
    increment()


def plot_mean_vals(vals_set, title_set, x_axis_set, y_axis_set, legend_set, calculate_mean=True, ALPHA=.05):
    for vals, title, x_axis, y_axis, legend in zip(vals_set, title_set, x_axis_set, y_axis_set, legend_set):
        list_vals = vals

        if calculate_mean:
            vals = {k: [sum(v[0]) / len(v[0]), sum(v[1]) / len(v[1]), numpy.std(v[0]), numpy.std(v[1])] for k, v in vals.items()}

        sorted_vals = {k: v for k, v in sorted(vals.items(), key=lambda item: item[1][0], reverse=True)}
        fig = go.Figure(data=[
            go.Bar(name='Players', x=list(sorted_vals.keys()), y=[i[0] for i in sorted_vals.values()],error_y=dict(type='data',array=[i[2] for i in sorted_vals.values()])),
            go.Bar(name='Opponents', x=list(sorted_vals.keys()), y=[i[1] for i in sorted_vals.values()],error_y=dict(type='data',array=[i[3] for i in sorted_vals.values()]))
        ])

        fig.update_layout(barmode='group')
        fig.update_layout(template='none')
        fig.update_layout(yaxis=dict(range=[0, 1]))
        fig.update_yaxes(nticks=40)
        fig.update_layout(
            title=title,
            xaxis_title=x_axis,
            yaxis_title=y_axis,
            legend_title=legend,
        )
        fig.show()
        fig.write_image(f'./images/fig{count}.png')
        increment()

        for combo in combinations(list(list_vals.keys()), 2):
            _, p_val, _ = ttest_ind(list_vals[combo[0]][0], list_vals[combo[1]][0], usevar='unequal')
            if p_val < ALPHA / len(list_vals):
                _, high_p_val, _ = ttest_ind(list_vals[combo[0]][0], list_vals[combo[1]][0], usevar='unequal',
                                             alternative='larger')
                _, low_p_val, _ = ttest_ind(list_vals[combo[0]][0], list_vals[combo[1]][0], usevar='unequal',
                                            alternative='smaller')
                if high_p_val < ALPHA:
                    print(f'{combo[0]} Player is better than {combo[1]} Player')
                elif low_p_val < ALPHA:
                    print(f'{combo[1]} Player is better than {combo[0]} Player')
                else:
                    print(f'{combo[0]} Player and {combo[1]} Player are not equal')
            else:
                print(f'{combo[0]} Player and {combo[1]} Player cannot be proven to be not equal')

        for combo in combinations(list(list_vals.keys()), 2):
            _, p_val, _ = ttest_ind(list_vals[combo[0]][1], list_vals[combo[1]][1], usevar='unequal')
            if p_val < ALPHA / len(list_vals):
                _, high_p_val, _ = ttest_ind(list_vals[combo[0]][1], list_vals[combo[1]][1], usevar='unequal',
                                             alternative='larger')
                _, low_p_val, _ = ttest_ind(list_vals[combo[0]][1], list_vals[combo[1]][1], usevar='unequal',
                                            alternative='smaller')
                if high_p_val < ALPHA:
                    print(f'{combo[0]} Opponent is better than {combo[1]} Opponent')
                elif low_p_val < ALPHA:
                    print(f'{combo[1]} Opponent is better than {combo[0]} Opponent')
                else:
                    print(f'{combo[0]} Opponent and {combo[1]} Opponent are not equal')
            else:
                print(f'{combo[0]} Opponent and {combo[1]} Opponent cannot be proven to be not equal')

        print()


# TQL 3 Red: saved_data/GAMMA=0.3 EPSILON=0.5 ALPHA=0.38 EPSILON_DECAY=0.99997Red,2020 08 24-08 35 45.pkl.xz
TQL3R = pd.read_pickle(
    "./saved_data/GAMMA=0.3 EPSILON=0.5 ALPHA=0.38 EPSILON_DECAY=0.99997Red,2020 08 24-08 35 45.pkl.xz")
# TQL 3 Blue: saved_data/GAMMA=0.3 EPSILON=0.5 ALPHA=0.38 EPSILON_DECAY=0.99997Blue,2020 08 24-08 35 47.pkl.xz
TQL3B = pd.read_pickle(
    "./saved_data/GAMMA=0.3 EPSILON=0.5 ALPHA=0.38 EPSILON_DECAY=0.99997Blue,2020 08 24-08 35 47.pkl.xz")

# DQN 3 Blue: saved_data/HIDDEN_LAYER_SIZE=116 BUFFER_SIZE=182 BATCH_SIZE=72 TARGET_MODEL_SYNC=8 LEARNING_RATE=0.0073632 GAMMA=0.3 EPSILON=0.5 EPSILON_DECAY=0.99997Blue,2020 08 25-09 55 51.pkl.xz
DQN3B = pd.read_pickle(
    "./saved_data/HIDDEN_LAYER_SIZE=116 BUFFER_SIZE=182 BATCH_SIZE=72 TARGET_MODEL_SYNC=8 LEARNING_RATE=0.0073632 GAMMA=0.3 EPSILON=0.5 EPSILON_DECAY=0.99997Blue,2020 08 25-09 55 51.pkl.xz")

# DQN 3 Red: saved_data/HIDDEN_LAYER_SIZE=116 BUFFER_SIZE=182 BATCH_SIZE=72 TARGET_MODEL_SYNC=8 LEARNING_RATE=0.0073632 GAMMA=0.3 EPSILON=0.5 EPSILON_DECAY=0.99997Red,2020 08 25-09 55 48.pkl.xz
DQN3R = pd.read_pickle(
    "./saved_data/HIDDEN_LAYER_SIZE=116 BUFFER_SIZE=182 BATCH_SIZE=72 TARGET_MODEL_SYNC=8 LEARNING_RATE=0.0073632 GAMMA=0.3 EPSILON=0.5 EPSILON_DECAY=0.99997Red,2020 08 25-09 55 48.pkl.xz")
# GQN 3 Red: saved_data/HIDDEN_LAYER_SIZE=34 BUFFER_SIZE=186 BATCH_SIZE=98 TARGET_MODEL_SYNC=8 LEARNING_RATE=0.0052994 GAMMA=0.3 EPSILON=0.5 EPSILON_DECAY=0.99997Red,2020 08 26-09 26 39.pkl.xz
GQN3R = pd.read_pickle(
    "./saved_data/HIDDEN_LAYER_SIZE=34 BUFFER_SIZE=186 BATCH_SIZE=98 TARGET_MODEL_SYNC=8 LEARNING_RATE=0.0052994 GAMMA=0.3 EPSILON=0.5 EPSILON_DECAY=0.99997Red,2020 08 26-09 26 39.pkl.xz")
# GQN 3 Blue: saved_data/HIDDEN_LAYER_SIZE=34 BUFFER_SIZE=186 BATCH_SIZE=98 TARGET_MODEL_SYNC=8 LEARNING_RATE=0.0052994 GAMMA=0.3 EPSILON=0.5 EPSILON_DECAY=0.99997Blue,2020 08 26-09 26 39.pkl.xz
GQN3B = pd.read_pickle(
    "./saved_data/HIDDEN_LAYER_SIZE=34 BUFFER_SIZE=186 BATCH_SIZE=98 TARGET_MODEL_SYNC=8 LEARNING_RATE=0.0052994 GAMMA=0.3 EPSILON=0.5 EPSILON_DECAY=0.99997Blue,2020 08 26-09 26 39.pkl.xz")
# MCTS 3 Red: saved_data/Trials=200 C=0.01 EPSILON=0.5Red,2020 08 25-22 00 00.pkl.xz
MCTS3R = pd.read_pickle("./saved_data/Trials=200 C=0.01 EPSILON=0.5Red,2020 08 25-22 00 00.pkl.xz")
# MCTS 3 Blue: saved_data/Trials=200 C=0.01 EPSILON=0.5Blue,2020 08 25-22 00 00.pkl.xz
MCTS3B = pd.read_pickle("./saved_data/Trials=200 C=0.01 EPSILON=0.5Blue,2020 08 25-22 00 00.pkl.xz")
# GQN 3-3 Red: saved_data/HIDDEN_LAYER_SIZE=34 BUFFER_SIZE=186 BATCH_SIZE=98 TARGET_MODEL_SYNC=8 LEARNING_RATE=0.0052994 GAMMA=0.3 EPSILON=0.5 EPSILON_DECAY=0.99997Red,2020 08 30-03 20 27.pkl.xz
GQN33R = pd.read_pickle(
    "./saved_data/HIDDEN_LAYER_SIZE=34 BUFFER_SIZE=186 BATCH_SIZE=98 TARGET_MODEL_SYNC=8 LEARNING_RATE=0.0052994 GAMMA=0.3 EPSILON=0.5 EPSILON_DECAY=0.99997Red,2020 08 30-03 20 27.pkl.xz")
# GQN 3-3 Blue: saved_data/HIDDEN_LAYER_SIZE=34 BUFFER_SIZE=186 BATCH_SIZE=98 TARGET_MODEL_SYNC=8 LEARNING_RATE=0.0052994 GAMMA=0.3 EPSILON=0.5 EPSILON_DECAY=0.99997Blue,2020 08 30-03 20 28.pkl.xz
GQN33B = pd.read_pickle(
    "./saved_data/HIDDEN_LAYER_SIZE=34 BUFFER_SIZE=186 BATCH_SIZE=98 TARGET_MODEL_SYNC=8 LEARNING_RATE=0.0052994 GAMMA=0.3 EPSILON=0.5 EPSILON_DECAY=0.99997Blue,2020 08 30-03 20 28.pkl.xz")
# GQN 2-3 Red: saved_data/HIDDEN_LAYER_SIZE=34 BUFFER_SIZE=186 BATCH_SIZE=98 TARGET_MODEL_SYNC=8 LEARNING_RATE=0.0052994 GAMMA=0.3 EPSILON=0.5 EPSILON_DECAY=0.99997Red,2020 08 29-23 39 08.pkl.xz
GQN23R = pd.read_pickle(
    "./saved_data/HIDDEN_LAYER_SIZE=34 BUFFER_SIZE=186 BATCH_SIZE=98 TARGET_MODEL_SYNC=8 LEARNING_RATE=0.0052994 GAMMA=0.3 EPSILON=0.5 EPSILON_DECAY=0.99997Red,2020 08 29-23 39 08.pkl.xz")
# GQN 2-3 Blue: saved_data/HIDDEN_LAYER_SIZE=34 BUFFER_SIZE=186 BATCH_SIZE=98 TARGET_MODEL_SYNC=8 LEARNING_RATE=0.0052994 GAMMA=0.3 EPSILON=0.5 EPSILON_DECAY=0.99997Blue,2020 08 29-23 39 08.pkl.xz
GQN23B = pd.read_pickle(
    "./saved_data/HIDDEN_LAYER_SIZE=34 BUFFER_SIZE=186 BATCH_SIZE=98 TARGET_MODEL_SYNC=8 LEARNING_RATE=0.0052994 GAMMA=0.3 EPSILON=0.5 EPSILON_DECAY=0.99997Blue,2020 08 29-23 39 08.pkl.xz")
# DQNvMCTS Red: saved_data/HIDDEN_LAYER_SIZE=116 BUFFER_SIZE=182 BATCH_SIZE=72 TARGET_MODEL_SYNC=8 LEARNING_RATE=0.0073632 GAMMA=0.3 EPSILON=0.5 EPSILON_DECAY=0.99997Red,2020 08 29-02 56 39.pkl.xz
DQNM3R = pd.read_pickle(
    "saved_data/HIDDEN_LAYER_SIZE=116 BUFFER_SIZE=182 BATCH_SIZE=72 TARGET_MODEL_SYNC=8 LEARNING_RATE=0.0073632 GAMMA=0.3 EPSILON=0.5 EPSILON_DECAY=0.99997Red,2020 08 29-02 56 39.pkl.xz")
# DQNvMCTs Blue: saved_data/Trials=200 C=0.01 EPSILON=0.5Blue,2020 08 29-02 56 39.pkl.xz
DQNM3B = pd.read_pickle("./saved_data/Trials=200 C=0.01 EPSILON=0.5Blue,2020 08 29-02 56 39.pkl.xz")

# TQL 4 Red: saved_data/GAMMA=0.3 EPSILON=0.5 ALPHA=0.38 EPSILON_DECAY=0.99997Red,2020 08 29-09 25 25.pkl.xz
TQL4R = pd.read_pickle(
    "./saved_data/GAMMA=0.3 EPSILON=0.5 ALPHA=0.38 EPSILON_DECAY=0.99997Red,2020 08 29-09 25 25.pkl.xz")
# TQL 4 Blue: saved_data/GAMMA=0.3 EPSILON=0.5 ALPHA=0.38 EPSILON_DECAY=0.99997Blue,2020 08 29-09 25 43.pkl.xz
TQL4B = pd.read_pickle(
    "./saved_data/GAMMA=0.3 EPSILON=0.5 ALPHA=0.38 EPSILON_DECAY=0.99997Blue,2020 08 29-09 25 43.pkl.xz")
# DQN 4 Red: saved_data/HIDDEN_LAYER_SIZE=116 BUFFER_SIZE=182 BATCH_SIZE=72 TARGET_MODEL_SYNC=8 LEARNING_RATE=0.0073632 GAMMA=0.3 EPSILON=0.5 EPSILON_DECAY=0.99997Red,2020 08 27-17 34 27.pkl.xz
DQN4R = pd.read_pickle(
    "./saved_data/HIDDEN_LAYER_SIZE=116 BUFFER_SIZE=182 BATCH_SIZE=72 TARGET_MODEL_SYNC=8 LEARNING_RATE=0.0073632 GAMMA=0.3 EPSILON=0.5 EPSILON_DECAY=0.99997Red,2020 08 27-17 34 27.pkl.xz")
# DQN 4 Blue: saved_data/HIDDEN_LAYER_SIZE=116 BUFFER_SIZE=182 BATCH_SIZE=72 TARGET_MODEL_SYNC=8 LEARNING_RATE=0.0073632 GAMMA=0.3 EPSILON=0.5 EPSILON_DECAY=0.99997Blue,2020 08 27-17 34 28.pkl.xz
DQN4B = pd.read_pickle(
    "./saved_data/HIDDEN_LAYER_SIZE=116 BUFFER_SIZE=182 BATCH_SIZE=72 TARGET_MODEL_SYNC=8 LEARNING_RATE=0.0073632 GAMMA=0.3 EPSILON=0.5 EPSILON_DECAY=0.99997Blue,2020 08 27-17 34 28.pkl.xz")
# GQN 4 Red: saved_data/HIDDEN_LAYER_SIZE=34 BUFFER_SIZE=186 BATCH_SIZE=98 TARGET_MODEL_SYNC=8 LEARNING_RATE=0.0052994 GAMMA=0.3 EPSILON=0.5 EPSILON_DECAY=0.99997Red,2020 08 28-10 59 27.pkl.xz
GQN4R = pd.read_pickle(
    "./saved_data/HIDDEN_LAYER_SIZE=34 BUFFER_SIZE=186 BATCH_SIZE=98 TARGET_MODEL_SYNC=8 LEARNING_RATE=0.0052994 GAMMA=0.3 EPSILON=0.5 EPSILON_DECAY=0.99997Red,2020 08 28-10 59 27.pkl.xz")
# GQN 4 Blue: saved_data/HIDDEN_LAYER_SIZE=34 BUFFER_SIZE=186 BATCH_SIZE=98 TARGET_MODEL_SYNC=8 LEARNING_RATE=0.0052994 GAMMA=0.3 EPSILON=0.5 EPSILON_DECAY=0.99997Blue,2020 08 28-10 59 27.pkl.xz
GQN4B = pd.read_pickle(
    "./saved_data/HIDDEN_LAYER_SIZE=34 BUFFER_SIZE=186 BATCH_SIZE=98 TARGET_MODEL_SYNC=8 LEARNING_RATE=0.0052994 GAMMA=0.3 EPSILON=0.5 EPSILON_DECAY=0.99997Blue,2020 08 28-10 59 27.pkl.xz")
# MCTS 4 Red: saved_data/Trials=200 C=0.01 EPSILON=0.5Red,2020 08 29-01 12 45.pkl.xz
MCTS4R = pd.read_pickle("./saved_data/Trials=200 C=0.01 EPSILON=0.5Red,2020 08 29-01 12 45.pkl.xz")
# MCTS 4 blue: saved_data/Trials=200 C=0.01 EPSILON=0.5Blue,2020 08 29-01 12 45.pkl.xz
MCTS4B = pd.read_pickle("./saved_data/Trials=200 C=0.01 EPSILON=0.5Blue,2020 08 29-01 12 45.pkl.xz")

# values = [
#     {
#         'TQL 3': [[0.592, 0.624, 0.574], [0.414, 0.412, 0.376]],
#         'DQN 3': [[0.436, 0.484, 0.462], [0.396, 0.378, 0.396]],
#         'GQN-1 3': [[0.61, 0.616, 0.608], [0.414, 0.418, 0.42]],
#         'MCTS 3': [[0.998, 0.992, 0.958], [0.95, 0.948]],
#         'GQN-2 3': [[0.654, 0.626, 0.628], [0.372, 0.322, 0.336]],
#         'GQN-3 3': [[0.704, 0.712, 0.708], [0.372, 0.404, 0.374]],
#     },
#     {
#         'TQL 4': [[0.524, 0.512, 0.492], [0.454, 0.45, 0.486]],
#         'DQN 4': [[0.67, 0.608, 0.624], [0.606, 0.596, 0.588]],
#         'GQN 4': [[0.392, 0.36, 0.358], [0.408, 0.434, 0.432]],
#         'MCTS 4': [[.4, 0.4, 0.5], [.5, 0.6, 0.5]]
#     }
# ]
# plot_mean_vals(values, ['Fig. 1: Win Rate of Models against a random player on R(3)', 'Fig. 2: Win Rate of Models against a random player on R(4)'], ['R(3) Models', 'R(4) Models'],
#                ['% Win Rate', '% Win Rate'], ['Model Type', 'Model Type'])

#
# # # Win Rate
# plot_epochs([
#     [GQN3R.loc[1,:], GQN3B.loc[1,:]],
#     [MCTS3R.loc[1,:], MCTS3B.loc[1,:]],
#     [GQN23R.loc[1,:], GQN23B.loc[1,:]],
#     [GQN33R.loc[1,:], GQN33B.loc[1,:]],
# ],
#             ['Fig. 3: Win Rate of GQN and MCTS Models on R(3)', 'Win Rate'], ["GQN 1","MCTS","GQN 2", "GQN 3"])
#
# plot_epochs([
#
#     [GQN4R.loc[1,:], GQN4B.loc[1,:]],
#     [MCTS4R.loc[1,:], MCTS4B.loc[1,:]],
#
# ],
#             ['Fig. 4: Win Rate of GQN and MCTS Models on R(4)', 'Win Rate'], ["GQN 1","MCTS"])
# plot_epochs([
#     [TQL3R.loc[1,:], TQL3B.loc[1,:]],
#     [DQN3R.loc[1,:], DQN3B.loc[1,:]],
#
# ],
#             ['Fig. 5 : Win Rate of TQL and DQN Models on R(3)', 'Win Rate'], ['TQL','DQN'])
#
# plot_epochs([
#     [TQL4R.loc[1,:], TQL4B.loc[1,:]],
#     [DQN4R.loc[1,:], DQN4B.loc[1,:]],
#
#
# ],
#             ['Fig. 6: Win Rate of TQL and DQN Models on R(4)', 'Win Rate'], ['TQL','DQN'])
# # # Move Time
# plot_epochs([
#
#     [GQN3R.loc[2,:], GQN3B.loc[2,:]],
#     [MCTS3R.loc[2,:], MCTS3B.loc[2,:]],
#     [GQN23R.loc[2,:], GQN23B.loc[2,:]],
#     [GQN33R.loc[2,:], GQN33B.loc[2,:]],
# ],
#             ['Fig. 7: Move Time of GQN and MCTS Models on R(3)', 'Move Time'], ["GQN 1","MCTS","GQN 2", "GQN 3"])
#
# plot_epochs([
#
#     [GQN4R.loc[2,:], GQN4B.loc[2,:]],
#     [MCTS4R.loc[2,:], MCTS4B.loc[2,:]],
#
# ],
#             ['Fig. 8: Move Time of GQN and MCTS Models on R(4)', 'Move Time'], ["GQN 1","MCTS"])
#
# plot_epochs([
#     [TQL3R.loc[2,:], TQL3B.loc[2,:]],
#     [DQN3R.loc[2,:], DQN3B.loc[2,:]],
#
# ],
#             ['Fig. 9: Move Time of TQL and DQN Models on R(3)', 'Move Time'], ['TQL','DQN'])
#
# plot_epochs([
#     [TQL4R.loc[2,:], TQL4B.loc[2,:]],
#     [DQN4R.loc[2,:], DQN4B.loc[2,:]],
#
#
# ],
#             ['Fig. 10: Move Time of TQL and DQN Models on R(4)', 'Move Time'], ['TQL','DQN'])
#
#  #Average Number of Moves
# plot_epochs([
#     [GQN3R.loc[3, :].rolling(100,min_periods=0).mean(), GQN3B.loc[3, :].rolling(100,min_periods=0).mean()],
#     [MCTS3R.loc[3, :].rolling(100,min_periods=0).mean(), MCTS3B.loc[3, :].rolling(100,min_periods=0).mean()],
#     [GQN23R.loc[3, :].rolling(100,min_periods=0).mean(), GQN23B.loc[3, :].rolling(100,min_periods=0).mean()],
#     [GQN33R.loc[3, :].rolling(100,min_periods=0).mean(), GQN33B.loc[3, :].rolling(100,min_periods=0).mean()],
#
# ],
#     ['Fig. 11: Average Number of Moves of GQN and MCTS Models on R(3)', 'Average Number of Moves'],
#     [ "GQN 1","MCTS","GQN 2", "GQN 3"])
# #
# plot_epochs([
#     [GQN4R.loc[3, :].rolling(100,min_periods=0).mean(), GQN4B.loc[3, :].rolling(100,min_periods=0).mean()],
#     [MCTS4R.loc[3, :].rolling(100,min_periods=0).mean(), MCTS4B.loc[3, :].rolling(100,min_periods=0).mean()],
#
#     #
# ],
#     ['Fig. 12: Average Number of Moves of GQN and MCTS Models on R(4)', 'Average Number of Moves'], [ "GQN 1","MCTS"])
#
# plot_epochs([
#
#     [TQL3R.loc[3, :].rolling(100,min_periods=0).mean(), TQL3B.loc[3, :].rolling(100,min_periods=0).mean()],
#     [DQN3R.loc[3, :].rolling(100,min_periods=0).mean(), DQN3B.loc[3, :].rolling(100,min_periods=0).mean()],
#
# ],
#     ['Fig. 13: Average Number of Moves of TQL and DQN Models on R(3)', 'Average Number of Moves'],
#     [ 'TQL','DQN',])
# #
# plot_epochs([
#
#     [TQL4R.loc[3, :].rolling(100,min_periods=0).mean(), TQL4B.loc[3, :].rolling(100,min_periods=0).mean()],
#     [DQN4R.loc[3, :].rolling(100,min_periods=0).mean(), DQN4B.loc[3, :].rolling(100,min_periods=0).mean()],
#     #
# ],
#     ['Fig. 14: Average Number of Moves of TQL and DQN Models on R(4)', 'Average Number of Moves'], [ 'TQL','DQN',],True)
#
# # Memory Usage
# plot_epochs([
#     [GQN3R.loc[4, :], GQN3B.loc[4, :]],
#     [MCTS3R.loc[4, :], MCTS3B.loc[4, :]],
#     [GQN23R.loc[4, :], GQN23B.loc[4, :]],
#     [GQN3R.loc[4, :], GQN3B.loc[4, :]],
# ],
#             ['Fig. 15: Memory Usage of GQN and MCTS Models on R(3)', 'Memory Usage'], ["GQN 1", "MCTS", "GQN 2", "GQN 3"],True)
# plot_epochs([
#     [GQN4R.loc[4, :], GQN4B.loc[4, :]],
#     [MCTS4R.loc[4, :], MCTS4B.loc[4, :]],
# ],
#              ['Fig. 16: Memory Usage of GQN and MCTS Models on R(4)', 'Memory Usage'], ["GQN 1", "MCTS"])
plot_epochs([
    [TQL3R.loc[4, :], TQL3B.loc[4, :]],
    [DQN3R.loc[4, :], DQN3B.loc[4, :]],

],
            ['Fig. 17: Memory Usage of TQL and DQN Models on R(3)', 'Memory Usage'], ['TQL','DQN'])
plot_epochs([
    [TQL4R.loc[4, :], TQL4B.loc[4, :]],
    [DQN4R.loc[4, :], DQN4B.loc[4, :]],
],
             ['Fig 18: Memory Usage of TQL and DQN Models on R(4)', 'Memory Usage'], ['TQL','DQN'])

# plot_mean_vals(values, ['% Win Rate of Models on R(3)', '% Win Rate of Models on R(4)'], ['R(3) Models', 'R(4) Models'],
#                ['% Win Rate', '% Win Rate'], ['Model Type', 'Model Type'])
# names = ['TQL 3 Player','TQL 3 Opponent','DQN 3 Player','DQN 3 Opponent',"GQN-1 3 Player","GQN-1 3 Opponent","MCTS 3 Player","MCTS 3 Opponent","GQN-2 3 Player","GQN-2 3 Opponent", "GQN-3 3 Player","GQN-3 3 Opponent",'TQL 4 Player','TQL 4 Opponent', 'DQN 4 Player','DQN 4 Opponent', "GQN-1 4 Player","GQN-1 4 Opponent", "MCTS 4 Player", "MCTS 4 Opponent"]
# for i,n in zip([TQL3R, TQL3B,DQN3R, DQN3B,GQN3R, GQN3B,MCTS3R, MCTS3B,GQN23R, GQN23B,GQN33R, GQN33B,TQL4R, TQL4B, DQN4R, DQN4B,GQN4R, GQN4B,MCTS4R, MCTS4B],names):
#     i = i.transpose()
#     i.columns = ['Loss','Win Rate', 'Average Move Time', "Number of Moves", "Memory Usage"]
#     i.to_sql(n,con=connection, if_exists="replace")
#     print(i)


