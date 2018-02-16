class Config:
    def __init__(self):
        self.kv_dict = {}
        self.update()
        env = self.environment
        exec('from {} import available_actions, goal, measurement_of_interest, frame_repeat'.format(env))
        exec("self.kv_dict['available_actions'] = available_actions")
        exec("self.kv_dict['goal'] = goal")
        exec("self.kv_dict['measurement_of_interest'] = measurement_of_interest")
        exec("self.kv_dict['frame_repeat'] = frame_repeat")
        self.kv_dict['num_measurements'] = len(self.goal)
        self.kv_dict['num_actions'] = len(self.available_actions)

    def update(self):
        with open('config.txt') as f:
            line = f.readline()
            key = line.split()[0]
            val = line.split()[2]
            self.kv_dict[key] = val

            phase = 0
            while line:
                line = f.readline()
                if len(line.split()) == 0:
                    break
                if line.startswith('-'):
                    phase += 1
                else:
                    if phase == 0:
                        key = line.split()[0]
                        val = line.split()[2]
                        self.kv_dict[key] = val
                    if phase == 1:
                        key = line.split()[0]
                        try:
                            val = int(line.split()[2])
                        except ValueError:
                            val = float(line.split()[2])
                        self.kv_dict[key] = val
                    if phase == 2:
                        key = line.split()[0]
                        val = bool(line.split()[2])
                        self.kv_dict[key] = val

    def __getattr__(self, item):
        return self.kv_dict[item]


config = Config()