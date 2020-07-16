from Utils import Utils
from TQL import TQL

player = TQL('Red',{"ALPHA":.5,"GAMMA":.1,'EPSILON':.5})
opp = TQL('Blue',{"ALPHA":.5,"GAMMA":.1,'EPSILON':.5})
u = Utils(player,opp,500)
parameters = [
    {
        "name":"ALPHA",
        "type":"range",
        "bounds":[0.2,0.8]
    },
    {
        "name":"GAMMA",
        "type":"range",
        "bounds":[0.01,0.5]
    },
    {
        "name":"EPSILON",
        "type":"range",
        "bounds":[0.2,0.8]
    },

]
print(u.optimize_training(parameters))
