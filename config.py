import numpy
from environment import AVAILABLE_ACTIONS, GOAL, MEASUREMENT_OF_INTEREST

model_name = 'intelAct'
batch_size = 64
num_actions = len(AVAILABLE_ACTIONS)
num_measurements = len(GOAL)
bots_num = 8
p_explore = 0.25
save_freq = 1000000
available_actions = AVAILABLE_ACTIONS
measurement_of_interest = MEASUREMENT_OF_INTEREST
goal = GOAL
continue_training = True
ip_addr = 'tcp://192.168.1.6'
#ip_addr = 'tcp://127.0.0.1'

agent_num = 75
local_agent_start = 40
local_agent_end = 75