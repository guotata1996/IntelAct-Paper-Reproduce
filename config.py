import numpy
from environment_basic import AVAILABLE_ACTIONS, GOAL, MEASUREMENT_OF_INTEREST

model_name = 'intelAct_basic'
batch_size = 64
num_actions = len(AVAILABLE_ACTIONS)
num_measurements = len(GOAL)
resolution = (128, 128)
bots_num = 8
p_explore = 0.1
save_freq = 1000000
available_actions = AVAILABLE_ACTIONS
measurement_of_interest = MEASUREMENT_OF_INTEREST
frame_repeat = 4
goal = GOAL
continue_training = False
#ip_addr = 'tcp://192.168.1.6'
ip_addr = 'tcp://127.0.0.1'

agent_num = 10
local_agent_start = 0
local_agent_end = 10