from base_optimizer import MasterProcess
from numpy.random import randint

class intelAct_masterprocess(MasterProcess):
    def parse_memory(self, ident, observation, predicting_result):
        if observation is None:
            #episode finished
            while self.client_memory[ident].qsize() > 1:
                self.client_memory[ident].get()
            self.network.log_performance(self.client_memory[ident].get())
            return None
        else:
            observation['real_action'] = predicting_result['action']
            self.client_memory[ident].put(observation)
            if self.client_memory[ident].qsize() > 33:
                real_measurements = []

                offsets = [0,1,2,4,8,16,32]
                for k in range(6):
                    delta_offset = offsets[k+1] - offsets[k] - 1
                    for _ in range(delta_offset):
                        self.client_memory[ident].get()
                    real_measurements.append(self.client_memory[ident].get()[bytes('measurement', encoding='utf8')])

                observation['real_measurement'] = real_measurements
                return observation
            else:
                return None