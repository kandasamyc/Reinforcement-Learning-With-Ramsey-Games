from Utils import Utils
from DQN import DQN
from TQL import TQL

# player = DQN('Red', {"GAMMA": .1, 'EPSILON': .6,'HIDDEN_LAYER_SIZE':20,'BUFFER_SIZE':1000,'BATCH_SIZE':150,'TARGET_MODEL_SYNC':6,'LEARNING_RATE':1e-2,'EPSILON_DECAY':.9997})
# opp = DQN('Blue', {"GAMMA": .1, 'EPSILON': .6,'HIDDEN_LAYER_SIZE':20,'BUFFER_SIZE':1000,'BATCH_SIZE':150,'TARGET_MODEL_SYNC':6,'LEARNING_RATE':1e-2,'EPSILON_DECAY':.9997})
# u = Utils(player, opp, 30)
# u.train()

player = TQL('Red', {"GAMMA": .1, 'EPSILON': .6, 'ALPHA':.6})
opp = TQL('Red', {"GAMMA": .1, 'EPSILON': .6, 'ALPHA':.6})
u = Utils(player, opp, 3000)
u.train()

