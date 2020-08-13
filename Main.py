from Utils import Utils
from DQN import DQN
from TQL import TQL
from MCTS import MCTSAgent
from GQN import GQN
import math

player = MCTSAgent({'Trials':200,'C':math.sqrt(2),'EPSILON':.5},'Red',3,6)
opp = MCTSAgent({'Trials':200,'C':math.sqrt(2),'EPSILON':.5},'Blue',3,6)
u = Utils(player,opp)

u.play_against_random(player,opp,50,mcts=True)

player = MCTSAgent({'Trials':200,'C':math.sqrt(2),'EPSILON':1},'Red',3,6)
opp = MCTSAgent({'Trials':200,'C':math.sqrt(2),'EPSILON':1},'Blue',3,6)
u = Utils(player,opp)

u.play_against_random(player,opp,50,mcts=True)

player = DQN('Red', {"GAMMA": .3, 'EPSILON': .5, 'HIDDEN_LAYER_SIZE': 100, 'BUFFER_SIZE': 140, 'BATCH_SIZE': 70,
                     'TARGET_MODEL_SYNC': 8, 'LEARNING_RATE': .0015, 'EPSILON_DECAY': .99997})
opp = DQN('Blue', {"GAMMA": .3, 'EPSILON': .5, 'HIDDEN_LAYER_SIZE': 100, 'BUFFER_SIZE': 140, 'BATCH_SIZE': 70,
                     'TARGET_MODEL_SYNC': 8, 'LEARNING_RATE': .0015, 'EPSILON_DECAY': .99997})
u = Utils(player, opp, 50000)
u.train()
player.store()
opp.store()
player.save_dict()
opp.save_dict()
player = DQN('Red', {"GAMMA": .3, 'EPSILON': .5, 'HIDDEN_LAYER_SIZE': 100, 'BUFFER_SIZE': 140, 'BATCH_SIZE': 70,
                     'TARGET_MODEL_SYNC': 8, 'LEARNING_RATE': .015, 'EPSILON_DECAY': .99997})
opp = DQN('Blue', {"GAMMA": .3, 'EPSILON': .5, 'HIDDEN_LAYER_SIZE': 100, 'BUFFER_SIZE': 140, 'BATCH_SIZE': 70,
                     'TARGET_MODEL_SYNC': 8, 'LEARNING_RATE': .015, 'EPSILON_DECAY': .99997})
u = Utils(player, opp, 50000)
u.train()
player.store()
opp.store()
player.save_dict()
opp.save_dict()
# player = TQL('Red', {"GAMMA": .3, 'EPSILON': .5, 'ALPHA': .38, 'EPSILON_DECAY': .99997},training=False)
# opp = TQL('Blue', {"GAMMA": .3, 'EPSILON': .5, 'ALPHA': .38, 'EPSILON_DECAY': .99997},training=False)
# player = GQN('Red', {"GAMMA": .3, 'EPSILON': .5, 'HIDDEN_LAYER_SIZE': 100, 'BUFFER_SIZE': 140, 'BATCH_SIZE': 70,
#                      'TARGET_MODEL_SYNC': 8, 'LEARNING_RATE': .015, 'EPSILON_DECAY': .99997},training=False)
# opp = GQN('Blue', {"GAMMA": .3, 'EPSILON': .5, 'HIDDEN_LAYER_SIZE': 100, 'BUFFER_SIZE': 140, 'BATCH_SIZE': 70,
#                      'TARGET_MODEL_SYNC': 8, 'LEARNING_RATE': .015, 'EPSILON_DECAY': .99997},training=False)
# u = Utils(player, opp, 3000)
# u.train()
# player.store()
# opp.store()
# player.save_dict()
# opp.save_dict()
# player.open('./models/GAMMA=0.17 EPSILON=0.553 HIDDEN_LAYER_SIZE=100 BUFFER_SIZE=140 BATCH_SIZE=70 TARGET_MODEL_SYNC=8 LEARNING_RATE=0.015 EPSILON_DECAY=0.99997Red,2020 08 12-10 05 59.pth')
# opp.open('./models/GAMMA=0.17 EPSILON=0.553 HIDDEN_LAYER_SIZE=100 BUFFER_SIZE=140 BATCH_SIZE=70 TARGET_MODEL_SYNC=8 LEARNING_RATE=0.015 EPSILON_DECAY=0.99997Blue,2020 08 12-10 05 59.pth')
# player = MCTSAgent({'Trials':200,'C':math.sqrt(2),'EPSILON':1},'Red',3,6)
# opp = MCTSAgent({'Trials':200,'C':math.sqrt(2),'EPSILON':1},'Blue',3,6)
# u = Utils(player,opp)
#
# u.play_against_random(player,opp,10,mcts=True)
# u.train()
# Utils.play(player,opp)
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

# player.store()
# opp.store()
# player.save_dict()
# opp.save_dict()


# player.load('models/GAMMA=0.3 EPSILON=0.5 ALPHA=0.6 EPSILON_DECAY=0.99997Red,2020 08 10-15 22 53.pkl.xz')
# opp.load('models/GAMMA=0.3 EPSILON=0.5 ALPHA=0.38 EPSILON_DECAY=0.99997Blue,2020 08 10-15 23 55.pkl.xz')
# player.load('models/GAMMA=0.3 EPSILON=0.5 ALPHA=0.6 EPSILON_DECAY=0.99997Red,2020 08 07-17 47 51.pkl.xz')
# opp.load('models/GAMMA=0.3 EPSILON=0.5 ALPHA=0.38 EPSILON_DECAY=0.99997Blue,2020 08 07-17 47 54.pkl.xz')
# u.play(player,opp)
# print(u.optimize_training([
#     {
#         'name': 'GAMMA',
#         'type': 'fixed',
#         'value': 0.3
#     },
#     {
#         'name': 'EPSILON',
#         'type': 'fixed',
#         'value': 0.5
#     },
# {
#         'name': 'EPSILON_DECAY',
#         'type': 'fixed',
#         'value': 0.99997
#     },
#     {
#         'name': 'ALPHA',
#         'type': 'range',
#         'bounds': [.2, .4]
#     },
#
# ]))
# player.store()
# opp.store()

# player = MCTSAgent({'Trials':200,'C':math.sqrt(2),'EPSILON':.5},'Red',3,6)
# adversary = MCTSAgent({'Trials':200,'C':math.sqrt(2),'EPSILON':.5},'Blue',3,6)
# u = Utils(player,adversary,50)
# Utils.play(player,adversary)
