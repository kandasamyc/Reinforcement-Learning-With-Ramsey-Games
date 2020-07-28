from Utils import Utils
from DQN import DQN
from TQL import TQL
from MCTS import MCTSAgent
from GQN import GQN
import math


player = GQN('Red', {"GAMMA": .17, 'EPSILON': .553, 'HIDDEN_LAYER_SIZE': 27, 'BUFFER_SIZE': 140, 'BATCH_SIZE': 70,
                     'TARGET_MODEL_SYNC': 8, 'LEARNING_RATE': .015, 'EPSILON_DECAY': .99977})
opp = GQN('Blue', {"GAMMA": .17, 'EPSILON': .553, 'HIDDEN_LAYER_SIZE': 27, 'BUFFER_SIZE': 140, 'BATCH_SIZE': 70,
                     'TARGET_MODEL_SYNC': 8, 'LEARNING_RATE': .015, 'EPSILON_DECAY': .99977})
u = Utils(player, opp, 7000)
u.train()
# player.store()
# opp.store()
# print(u.optimize_training([
#     {
#         'name': 'GAMMA',
#         'type': 'range',
#         'bounds': [.0001, .4]
#     },
#     {
#         'name': 'EPSILON',
#         'type': 'range',
#         'bounds': [.4, .6]
#     },
#     {
#         'name': 'HIDDEN_LAYER_SIZE',
#         'type': 'range',
#         'bounds': [20, 100]
#     },
#     {
#         'name': 'BUFFER_SIZE',
#         'type': 'range',
#         'bounds': [100, 500]
#     }, {
#         'name': 'BATCH_SIZE',
#         'type': 'range',
#         'bounds': [50, 100]
#     }, {
#         'name': 'TARGET_MODEL_SYNC',
#         'type': 'range',
#         'bounds': [4, 10]
#     },
#     {
#         'name': 'LEARNING_RATE',
#         'type': 'range',
#         'bounds': [1e-4, .1]
#     },
#     {
#         'name': 'EPSILON_DECAY',
#         'type': 'range',
#         'bounds': [.9997, .999997]
#     }
#
# ]))

# player = TQL('Red', {"GAMMA": .1, 'EPSILON': .6, 'ALPHA':.6})
# opp = TQL('Blue', {"GAMMA": .1, 'EPSILON': .6, 'ALPHA':.6})
# u = Utils(player, opp, 3000)
# print(u.optimize_training([
#     {
#           'name': 'GAMMA',
#           'type': 'range',
#           'bounds': [.0001, .4]
#       },
#       {
#           'name': 'EPSILON',
#           'type': 'range',
#           'bounds': [.4, .6]
#       },
#       {
#           'name': 'ALPHA',
#           'type': 'range',
#           'bounds': [.4, .8]
#       },
# ]))


# player = MCTSAgent({'Trials':50,'C':math.sqrt(2)},'Red',3,6)
# adversary = MCTSAgent({'Trials':50,'C':math.sqrt(2)},'Blue',3,6)
# u = Utils(player,adversary,50)
# u.train()