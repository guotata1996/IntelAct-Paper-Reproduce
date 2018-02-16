import numpy
from environment_basic import AVAILABLE_ACTIONS, GOAL, MEASUREMENT_OF_INTEREST

model_name = 'intelActBase'
batch_size = 64
num_actions = len(AVAILABLE_ACTIONS)
num_measurements = len(GOAL)
bots_num = 8
save_freq = 800000
p_explore = 0.5
available_actions = AVAILABLE_ACTIONS
measurement_of_interest = MEASUREMENT_OF_INTEREST
goal = GOAL
continue_training = False
ip_addr = 'tcp://192.168.1.6'
#ip_addr = 'tcp://127.0.0.1'

agent_num = 80
local_agent_start = 40
local_agent_end = 80
