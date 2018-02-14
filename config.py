import numpy

model_name = 'intelAct'
batch_size = 64
agent_num = 110
local_agent_start = 75
local_agent_end = 110
num_actions = 7
num_measurements = 5
resolution = (128, 128)
bots_num = 8
p_explore = 0.1
save_freq = 1000000
available_actions = numpy.eye(9)[:7,:]
frame_repeat = 4
continue_training = True
ip_addr = 'tcp://192.168.1.6'