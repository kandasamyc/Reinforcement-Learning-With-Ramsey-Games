from tqdm import tqdm

from Utils import Utils
from DQN import DQN
from TQL import TQL
from MCTS import MCTSAgent, MCTSGameTree
from GQN import GQN
import math
from ax.service.ax_client import AxClient
import igraph
from collections import Counter


#TQL

# dqn_best_params = {'HIDDEN_LAYER_SIZE': 116, 'BUFFER_SIZE': 182, 'BATCH_SIZE': 72, 'TARGET_MODEL_SYNC': 8, 'LEARNING_RATE': 0.0073632, 'GAMMA': 0.3, 'EPSILON': 0.5, 'EPSILON_DECAY': 0.99997}
# dplayer = DQN('Red', dqn_best_params,training=False)
# dopp = DQN('Blue', dqn_best_params,training=False)
# dplayer.open('./models/HIDDEN_LAYER_SIZE=116 BUFFER_SIZE=182 BATCH_SIZE=72 TARGET_MODEL_SYNC=8 LEARNING_RATE=0.0073632 GAMMA=0.3 EPSILON=0.5 EPSILON_DECAY=0.99997Red,2020 08 29-02 56 39.pth')
# dopp.open('./models/HIDDEN_LAYER_SIZE=116 BUFFER_SIZE=182 BATCH_SIZE=72 TARGET_MODEL_SYNC=8 LEARNING_RATE=0.0073632 GAMMA=0.3 EPSILON=0.5 EPSILON_DECAY=0.99997Blue,2020 08 29-03 49 52.pth')
#
# try:
#     Utils.play_against_random(dplayer,dopp,trials=500)
# except Exception:
#     print('Modified DQN Failed',flush=True)
# player = TQL('Red', {"GAMMA": .3, 'EPSILON': .5, 'ALPHA': .38, 'EPSILON_DECAY': .99997},training=False)
# opp = TQL('Blue', {"GAMMA": .3, 'EPSILON': .5, 'ALPHA': .38, 'EPSILON_DECAY': .99997},training=False)
# opp.load('./models/GAMMA=0.3 EPSILON=0.5 ALPHA=0.38 EPSILON_DECAY=0.99997Blue,2020 08 24-08 35 30.pkl.xz')
# player.load('./models/GAMMA=0.3 EPSILON=0.5 ALPHA=0.38 EPSILON_DECAY=0.99997Red,2020 08 24-08 35 19.pkl.xz')
# #
# try:
#     Utils.play_against_random(player,opp,trials=500)
# except Exception:
#     print('TQL 3 Failed',flush=True)
# #
# player = DQN('Red', dqn_best_params,training=False)
# opp = DQN('Blue', dqn_best_params,training=False)
# player.open('./models/HIDDEN_LAYER_SIZE=116 BUFFER_SIZE=182 BATCH_SIZE=72 TARGET_MODEL_SYNC=8 LEARNING_RATE=0.0073632 GAMMA=0.3 EPSILON=0.5 EPSILON_DECAY=0.99997Red,2020 08 25-09 55 42.pth')
# opp.open('./models/HIDDEN_LAYER_SIZE=116 BUFFER_SIZE=182 BATCH_SIZE=72 TARGET_MODEL_SYNC=8 LEARNING_RATE=0.0073632 GAMMA=0.3 EPSILON=0.5 EPSILON_DECAY=0.99997Blue,2020 08 25-09 55 42.pth')
# #
# try:
#     Utils.play_against_random(player,opp,trials=500)
# except Exception:
#     print('DQN 3 Failed',flush=True)
# gqn_params = {'HIDDEN_LAYER_SIZE': 34, 'BUFFER_SIZE': 186, 'BATCH_SIZE': 98, 'TARGET_MODEL_SYNC': 8, 'LEARNING_RATE': 0.0052994, 'GAMMA': 0.3, 'EPSILON': 0.5, 'EPSILON_DECAY': 0.99997}
# player = GQN('Red', gqn_params,training=False)
# opp = GQN('Blue', gqn_params,training=False)
# player.open('./models/HIDDEN_LAYER_SIZE=34 BUFFER_SIZE=186 BATCH_SIZE=98 TARGET_MODEL_SYNC=8 LEARNING_RATE=0.0052994 GAMMA=0.3 EPSILON=0.5 EPSILON_DECAY=0.99997Red,2020 08 26-09 26 38.pth')
# opp.open('./models/HIDDEN_LAYER_SIZE=34 BUFFER_SIZE=186 BATCH_SIZE=98 TARGET_MODEL_SYNC=8 LEARNING_RATE=0.0052994 GAMMA=0.3 EPSILON=0.5 EPSILON_DECAY=0.99997Blue,2020 08 26-09 26 38.pth')
# try:
#     Utils.play_against_random(player,opp,trials=500)
# except Exception:
#     print('GQN 3 Failed',flush=True)
# player = MCTSAgent({'Trials':200,'C':1.4142,'EPSILON':.5},'Red',3,6)
# opp = MCTSAgent({'Trials':200,'C':1.4142,'EPSILON':.5},'Blue',3,6)
# try:
#     Utils.play_against_random(player,opp,trials=500,mcts=True)
# except Exception:
#     print('MCTS 3 Failed',flush=True)
# player = TQL('Red', {"GAMMA": .3, 'EPSILON': .5, 'ALPHA': .38, 'EPSILON_DECAY': .99997},number_of_nodes=18,chain_length=4,training=False)
# opp = TQL('Blue', {"GAMMA": .3, 'EPSILON': .5, 'ALPHA': .38, 'EPSILON_DECAY': .99997},number_of_nodes=18,chain_length=4,training=False)
# player.load('./models/GAMMA=0.3 EPSILON=0.5 ALPHA=0.38 EPSILON_DECAY=0.99997Red,2020 08 29-08 31 32.pkl.xz')
# opp.load('./models/GAMMA=0.3 EPSILON=0.5 ALPHA=0.38 EPSILON_DECAY=0.99997Blue,2020 08 29-08 58 58.pkl.xz')
# try:
#     Utils.play_against_random(player,opp,trials=500)
# except Exception:
#     print('TQL 4 Failed',flush=True)
# dqn_best_params = {'HIDDEN_LAYER_SIZE': 116, 'BUFFER_SIZE': 182, 'BATCH_SIZE': 72, 'TARGET_MODEL_SYNC': 8, 'LEARNING_RATE': 0.0073632, 'GAMMA': 0.3, 'EPSILON': 0.5, 'EPSILON_DECAY': 0.99997}
# player = DQN('Red', dqn_best_params,number_of_nodes=18,chain_length=4,training=False)
# opp = DQN('Blue', dqn_best_params,number_of_nodes=18,chain_length=4,training=False)
# player.open('./models/HIDDEN_LAYER_SIZE=116 BUFFER_SIZE=182 BATCH_SIZE=72 TARGET_MODEL_SYNC=8 LEARNING_RATE=0.0073632 GAMMA=0.3 EPSILON=0.5 EPSILON_DECAY=0.99997Red,2020 08 27-17 34 27.pth')
# opp.open('./models/HIDDEN_LAYER_SIZE=116 BUFFER_SIZE=182 BATCH_SIZE=72 TARGET_MODEL_SYNC=8 LEARNING_RATE=0.0073632 GAMMA=0.3 EPSILON=0.5 EPSILON_DECAY=0.99997Blue,2020 08 27-17 34 27.pth')
# try:
#     Utils.play_against_random(player,opp,trials=500)
# except Exception:
#     print('DQN 4 Failed',flush=True)
# gqn_params = {'HIDDEN_LAYER_SIZE': 34, 'BUFFER_SIZE': 186, 'BATCH_SIZE': 98, 'TARGET_MODEL_SYNC': 8, 'LEARNING_RATE': 0.0052994, 'GAMMA': 0.3, 'EPSILON': 0.5, 'EPSILON_DECAY': 0.99997}
# player = GQN('Red', gqn_params,number_of_nodes=18,chain_length=4,training=False)
# opp = GQN('Blue', gqn_params,number_of_nodes=18,chain_length=4,training=False)
# player.open('./models/HIDDEN_LAYER_SIZE=34 BUFFER_SIZE=186 BATCH_SIZE=98 TARGET_MODEL_SYNC=8 LEARNING_RATE=0.0052994 GAMMA=0.3 EPSILON=0.5 EPSILON_DECAY=0.99997Red,2020 08 28-10 59 27.pth')
# opp.open('./models/HIDDEN_LAYER_SIZE=34 BUFFER_SIZE=186 BATCH_SIZE=98 TARGET_MODEL_SYNC=8 LEARNING_RATE=0.0052994 GAMMA=0.3 EPSILON=0.5 EPSILON_DECAY=0.99997Blue,2020 08 28-10 59 27.pth')
# try:
#     Utils.play_against_random(player,opp,trials=500)
# except Exception:
#     print('GQN 4 Failed',flush=True)
#
# gqn_params = {'HIDDEN_LAYER_SIZE': 34, 'BUFFER_SIZE': 186, 'BATCH_SIZE': 98, 'TARGET_MODEL_SYNC': 8, 'LEARNING_RATE': 0.0052994, 'GAMMA': 0.3, 'EPSILON': 0.5, 'EPSILON_DECAY': 0.99997}
# player = GQN('Red', gqn_params,network_id=2,training=False)
# opp = GQN('Blue', gqn_params,network_id=2,training=False)
# player.open('./models/HIDDEN_LAYER_SIZE=34 BUFFER_SIZE=186 BATCH_SIZE=98 TARGET_MODEL_SYNC=8 LEARNING_RATE=0.0052994 GAMMA=0.3 EPSILON=0.5 EPSILON_DECAY=0.99997Red,2020 08 29-23 39 07.pth')
# opp.open('./models/HIDDEN_LAYER_SIZE=34 BUFFER_SIZE=186 BATCH_SIZE=98 TARGET_MODEL_SYNC=8 LEARNING_RATE=0.0052994 GAMMA=0.3 EPSILON=0.5 EPSILON_DECAY=0.99997Blue,2020 08 29-23 39 07.pth')
#
# try:
#     Utils.play_against_random(player,opp,trials=500)
# except Exception:
#     print('GQN 3 ID 2 Failed',flush=True)
#
# gqn_params = {'HIDDEN_LAYER_SIZE': 34, 'BUFFER_SIZE': 186, 'BATCH_SIZE': 98, 'TARGET_MODEL_SYNC': 8, 'LEARNING_RATE': 0.0052994, 'GAMMA': 0.3, 'EPSILON': 0.5, 'EPSILON_DECAY': 0.99997}
# player = GQN('Red', gqn_params,network_id=3,training=False)
# opp = GQN('Blue', gqn_params,network_id=3,training=False)
# player.open('./models/HIDDEN_LAYER_SIZE=34 BUFFER_SIZE=186 BATCH_SIZE=98 TARGET_MODEL_SYNC=8 LEARNING_RATE=0.0052994 GAMMA=0.3 EPSILON=0.5 EPSILON_DECAY=0.99997Red,2020 08 30-03 20 27.pth')
# opp.open('./models/HIDDEN_LAYER_SIZE=34 BUFFER_SIZE=186 BATCH_SIZE=98 TARGET_MODEL_SYNC=8 LEARNING_RATE=0.0052994 GAMMA=0.3 EPSILON=0.5 EPSILON_DECAY=0.99997Blue,2020 08 30-03 20 27.pth')
#
# try:
#     Utils.play_against_random(player,opp,trials=500)
# except Exception:
#     print('GQN 3 ID 3 Failed',flush=True)


# player = MCTSAgent({'Trials':200,'C':1.4142,'EPSILON':.5},'Red',4,18)
# opp = MCTSAgent({'Trials':200,'C':1.4142,'EPSILON':.5},'Blue',4,18)
# try:
#     Utils.play_against_random(player,opp,trials=10,mcts=True)
# except Exception:
#     print('MCTS 4 Failed',flush=True)




# u1 = Utils(player,opp,30000)
# u1.train()
# player.store()
# opp.store()
# player.save_dict()
# opp.save_dict()

# player = TQL('Red', {"GAMMA": .3, 'EPSILON': .5, 'ALPHA': .38, 'EPSILON_DECAY': .99997},number_of_nodes=18,chain_length=4)
# opp = TQL('Blue', {"GAMMA": .3, 'EPSILON': .5, 'ALPHA': .38, 'EPSILON_DECAY': .99997},number_of_nodes=18,chain_length=4)
# u2 = Utils(player,opp,1500)
# u2.train()
# player.store()
# opp.store()
# player.save_dict()
# opp.save_dict()


# dqn_best_params = {'HIDDEN_LAYER_SIZE': 116, 'BUFFER_SIZE': 182, 'BATCH_SIZE': 72, 'TARGET_MODEL_SYNC': 8, 'LEARNING_RATE': 0.0073632, 'GAMMA': 0.3, 'EPSILON': 0.5, 'EPSILON_DECAY': 0.99997}
# player = DQN('Red', dqn_best_params)
# opp = DQN('Blue', dqn_best_params)
# u3 = Utils(player,opp,30000)
# u3.train()
# player.store()
# opp.store()
# player.save_dict()
# opp.save_dict()
# 
# dqn_best_params = {'HIDDEN_LAYER_SIZE': 116, 'BUFFER_SIZE': 182, 'BATCH_SIZE': 72, 'TARGET_MODEL_SYNC': 8, 'LEARNING_RATE': 0.0073632, 'GAMMA': 0.3, 'EPSILON': 0.5, 'EPSILON_DECAY': 0.99997}
# player = DQN('Red', dqn_best_params,number_of_nodes=18,chain_length=4)
# opp = DQN('Blue', dqn_best_params,number_of_nodes=18,chain_length=4)
# u4 = Utils(player,opp,1500)
# try:
#     u4.train()
#     player.store()
#     opp.store()
#     player.save_dict()
#     opp.save_dict()
# except Exception:
#     print(f'DQN for 4 Failed')
# player = MCTSAgent({'Trials':200,'C':.01,'EPSILON':.5},'Red',3,6)
# opp = MCTSAgent({'Trials':200,'C':.01,'EPSILON':.5},'Blue',3,6)
# u5 = Utils(player,opp,1000)
# try:
#     u5.train()
#     player.save_dict()
#     opp.save_dict()
# except Exception:
#     print('MCTS for 3 Failed')



# gqn_params = {'HIDDEN_LAYER_SIZE': 34, 'BUFFER_SIZE': 186, 'BATCH_SIZE': 98, 'TARGET_MODEL_SYNC': 8, 'LEARNING_RATE': 0.0052994, 'GAMMA': 0.3, 'EPSILON': 0.5, 'EPSILON_DECAY': 0.99997}
# player = GQN('Red', gqn_params,network_id=2)
# opp = GQN('Blue', gqn_params,network_id=2)
# u = Utils(player, opp, 3000)
# try:
#     u.train()
#     player.store()
#     opp.store()
#     player.save_dict()
#     opp.save_dict()
# except Exception:
#     print('GQN 2 for 3 Failed')

# gqn_params = {'HIDDEN_LAYER_SIZE': 34, 'BUFFER_SIZE': 186, 'BATCH_SIZE': 98, 'TARGET_MODEL_SYNC': 8, 'LEARNING_RATE': 0.0052994, 'GAMMA': 0.3, 'EPSILON': 0.5, 'EPSILON_DECAY': 0.99997}
# player = GQN('Red', gqn_params,network_id=3)
# opp = GQN('Blue', gqn_params,network_id=3)
# u1 = Utils(player, opp, 3000)
# try:
#     u1.train()
#     player.store()
#     opp.store()
#     player.save_dict()
#     opp.save_dict()
# except Exception:
#     print('GQN 3 for 3 Failed')

# player = GQN('Red', gqn_params,number_of_nodes=18,chain_length=4)
# opp = GQN('Blue', gqn_params,number_of_nodes=18,chain_length=4)
# u = Utils(player, opp, 500)
# try:
#     u.train()
#     player.store()
#     opp.store()
#     player.save_dict()
#     opp.save_dict()
# except Exception:
#     print('GQN for 4 Failed')



# player = MCTSAgent({'Trials':200,'C':.01,'EPSILON':.5},'Red',4,18)
# opp = MCTSAgent({'Trials':200,'C':.01,'EPSILON':.5},'Blue',4,18)
# u6 = Utils(player,opp,20)
# try:
#     u6.train()
#     player.save_dict()
#     opp.save_dict()
# except:
#     print('MCTS for 4 Failed')

# dqn_best_params = {'HIDDEN_LAYER_SIZE': 116, 'BUFFER_SIZE': 182, 'BATCH_SIZE': 72, 'TARGET_MODEL_SYNC': 8, 'LEARNING_RATE': 0.0073632, 'GAMMA': 0.3, 'EPSILON': 0.5, 'EPSILON_DECAY': 0.99997}
# dplayer = DQN('Red', dqn_best_params)
# dopp = DQN('Blue', dqn_best_params)
# mplayer = MCTSAgent({'Trials':200,'C':.01,'EPSILON':.5},'Red',3,6)
# mopp = MCTSAgent({'Trials':200,'C':.01,'EPSILON':.5},'Blue',3,6)
# u7 = Utils(dplayer,mopp)
# try:
#     u7.train()
#     dplayer.save_dict()
#     mopp.save_dict()
#     dplayer.store()
# except Exception:
#     print('MCTS and DQN Failed')

# u8 = Utils(mplayer,dopp)
# try:
#     u7.train()
#     mplayer.save_dict()
#     dopp.save_dict()
#     dopp.store()
# except Exception:
#     print('MCTS and DQN Failed')

# player = TQL('Red', {"GAMMA": .3, 'EPSILON': .5, 'ALPHA': .38, 'EPSILON_DECAY': .99997},number_of_nodes=18,chain_length=4)
# opp = TQL('Blue', {"GAMMA": .3, 'EPSILON': .5, 'ALPHA': .38, 'EPSILON_DECAY': .99997},number_of_nodes=18,chain_length=4)
# u2 = Utils(player,opp,1000)
# u2.train()
# player.store()
# opp.store()
# player.save_dict()
# opp.save_dict()
# ax_client = AxClient()
# ax_client.create_experiment(
    # name="GQN_Testing",
    # parameters=[
        # {
            # "name": "GAMMA",
            # "type": "fixed",
            # "value": 0.3
        # },
        # {
            # "name": "EPSILON",
            # "type": "fixed",
            # "value": 0.5
        # },
        # {
            # "name": "EPSILON_DECAY",
            # "type": "fixed",
            # "value": 0.99997
        # },
        # {
            # "name": "HIDDEN_LAYER_SIZE",
            # "type": "range",
            # "bounds": [20,200]
        # },
        # {
            # "name": "BUFFER_SIZE",
            # "type":"range",
            # "bounds": [20,200]
        # },
        # {
            # "name": "BATCH_SIZE",
            # "type":"range",
            # "bounds": [15,150]
        # },
        # {
            # "name": "TARGET_MODEL_SYNC",
            # "type":"range",
            # "bounds": [4,10]
        # },
        # {
            # "name": "LEARNING_RATE",
            # "type":"range",
            # "bounds": [1e-4,1e-2]
        # }
    # ],
    # parameter_constraints=["BATCH_SIZE <=  BUFFER_SIZE"]
# )
# player = GQN('Red', {"GAMMA": .3, 'EPSILON': .5, 'HIDDEN_LAYER_SIZE': 100, 'BUFFER_SIZE': 140, 'BATCH_SIZE': 70,
                    #  'TARGET_MODEL_SYNC': 8, 'LEARNING_RATE': .015, 'EPSILON_DECAY': .99997})
# opp = GQN('Blue', {"GAMMA": .3, 'EPSILON': .5, 'HIDDEN_LAYER_SIZE': 100, 'BUFFER_SIZE': 140, 'BATCH_SIZE': 70,
                    #  'TARGET_MODEL_SYNC': 8, 'LEARNING_RATE': .015, 'EPSILON_DECAY': .99997})
# u = Utils(player, opp, 500)
# for trial in range(8):
    # parameters, trial_index = ax_client.get_next_trial()
    # ax_client.complete_trial(trial_index,raw_data=u.train(parameters))

# 
# 
# dqn_best_params = {'HIDDEN_LAYER_SIZE': 116, 'BUFFER_SIZE': 182, 'BATCH_SIZE': 72, 'TARGET_MODEL_SYNC': 8, 'LEARNING_RATE': 0.007363222048745642, 'GAMMA': 0.3, 'EPSILON': 0.5, 'EPSILON_DECAY': 0.99997}
# player = DQN('Red', dqn_best_params,number_of_nodes=18,chain_length=4)
# opp = DQN('Blue', dqn_best_params,number_of_nodes=18,chain_length=4)
# u3 = Utils(player,opp,1000)
# u3.train()
# Utils.play_against_random(player,opp,100)
# player.store()
# opp.store()
# player.save_dict()
# opp.save_dict()
# 
# player = MCTSAgent({'Trials':200,'C':.01,'EPSILON':.5},'Red',3,6)
# opp = MCTSAgent({'Trials':200,'C':.01,'EPSILON':.5},'Blue',3,6)
# u4 = Utils(player,opp)
# u4.play_against_random(player,opp,500,mcts=True)
# player.save_dict()
# opp.save_dict()

#player = MCTSAgent({'Trials':200,'C':.01,'EPSILON':.5},'Red',4,18)
#opp = MCTSAgent({'Trials':200,'C':.01,'EPSILON':.5},'Blue',4,18)
#u5 = Utils(player,opp)
#u5.play_against_random(player,opp,10,mcts=True)
#player.save_dict()
#opp.save_dict()
#

# u2 = Utils(DQN_player, MCTS_opp, 3000)
# u2.train()
# DQN_player.store()
# DQN_player.save_dict()
# MCTS_opp.save_dict()


# Utils.play_against_random(DQN_player,DQN_opp,100)
# Utils.play_against_random(MCTS_player,MCTS_opp,50,mcts=True)

# player = DQN('Red', {"GAMMA": .3, 'EPSILON': .5, 'HIDDEN_LAYER_SIZE': 100, 'BUFFER_SIZE': 140, 'BATCH_SIZE': 70,
#                      'TARGET_MODEL_SYNC': 8, 'LEARNING_RATE': .015, 'EPSILON_DECAY': .99997})
# opp = DQN('Blue', {"GAMMA": .3, 'EPSILON': .5, 'HIDDEN_LAYER_SIZE': 100, 'BUFFER_SIZE': 140, 'BATCH_SIZE': 70,
#                      'TARGET_MODEL_SYNC': 8, 'LEARNING_RATE': .015, 'EPSILON_DECAY': .99997})
# u = Utils(player, opp, 50000)
# u.train()
# player.store()
# opp.store()
# player.save_dict()
# opp.save_dict()
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


# player.load('./models/GAMMA=0.3 EPSILON=0.5 ALPHA=0.6 EPSILON_DECAY=0.99997Red,2020 08 10-15 22 53.pkl.xz')
# opp.load('./models/GAMMA=0.3 EPSILON=0.5 ALPHA=0.38 EPSILON_DECAY=0.99997Blue,2020 08 10-15 23 55.pkl.xz')
# player.load('./models/GAMMA=0.3 EPSILON=0.5 ALPHA=0.6 EPSILON_DECAY=0.99997Red,2020 08 07-17 47 51.pkl.xz')
# opp.load('./models/GAMMA=0.3 EPSILON=0.5 ALPHA=0.38 EPSILON_DECAY=0.99997Blue,2020 08 07-17 47 54.pkl.xz')
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
# # opp.store()

g = Utils.make_graph(6)
Utils.display_graph(g)
# dqn_best_params = {'HIDDEN_LAYER_SIZE': 116, 'BUFFER_SIZE': 182, 'BATCH_SIZE': 72, 'TARGET_MODEL_SYNC': 8, 'LEARNING_RATE': 0.007363222048745642, 'GAMMA': 0.3, 'EPSILON': 0.5, 'EPSILON_DECAY': 0.99997}
# player = DQN('Red', dqn_best_params,number_of_nodes=18,chain_length=4)
# player.open('./models/HIDDEN_LAYER_SIZE=116 BUFFER_SIZE=182 BATCH_SIZE=72 TARGET_MODEL_SYNC=8 LEARNING_RATE=0.0073632 GAMMA=0.3 EPSILON=0.5 EPSILON_DECAY=0.99997Red,2020 08 27-17 34 27.pth')
# opp = DQN('Blue', dqn_best_params,number_of_nodes=18,chain_length=4)
# opp.open('./models/HIDDEN_LAYER_SIZE=116 BUFFER_SIZE=182 BATCH_SIZE=72 TARGET_MODEL_SYNC=8 LEARNING_RATE=0.0073632 GAMMA=0.3 EPSILON=0.5 EPSILON_DECAY=0.99997Blue,2020 08 27-17 34 27.pth')
# # Utils.play_against_random(player,opp,1)
# finished = False
# while not finished:
#     finished = player.move(opp)
#     Utils.display_graph(player.state)
#     if finished:
#         break
#     finished = opp.move(player)
#     Utils.display_graph(opp.state)
#
# player = MCTSAgent({'Trials':200,'C':.01,'EPSILON':.5},'Red',3,6)
# adversary = MCTSAgent({'Trials':200,'C':.01,'EPSILON':.5},'Blue',3,6)
#Utils.play_against_random(player,adversary,1,True)
# finished = False
# while not finished:
#     finished = player.move(adversary)
#     Utils.display_graph(player.state)
#     if finished:
#         break
#     finished = adversary.move(player)
#     Utils.display_graph(adversary.state)




