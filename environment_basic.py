from vizdoom import *
import cv2
import numpy as np
import time

AVAILABLE_ACTIONS = [[1,0,0],[0,1,0],[0,0,1]]
GOAL = [1,0]
MEASUREMENT_OF_INTEREST = 1
frame_repeat = 4
resolution = (128,128)
#ultimate goal must be 1st measurement

class Environment:
    def __init__(self, rand_seed, display = False):
        self.game = DoomGame()
        self.game.set_seed(rand_seed)
        self.game.load_config("scenarios\\basic.cfg")
        self.game.add_available_game_variable(GameVariable.POSITION_Y)
        self.game.set_window_visible(display)
        self.game.init()
        self.total_rew = 0
        self.kill_count = 0

    def action(self, action):
        self.total_rew += self.game.make_action(AVAILABLE_ACTIONS[action], frame_repeat)

    def current_state(self):
        if self.game.is_episode_finished():
            self.game.new_episode()
            self.kill_count += 1
            if self.kill_count == 100:
                self.total_rew = 0
                self.kill_count = 0
            return None
        else:
            screen = self.game.get_state().screen_buffer  # 3 x h x w
            screen = screen.transpose((1, 2, 0))  # h x w x 3
            whole_screen = cv2.resize(screen, resolution)  # 60 x 108 x 3
            whole_screen = whole_screen.astype(np.float32)
            pos_y = self.game.get_game_variable(GameVariable.POSITION_Y)
            s = dict()
            s['image'] = whole_screen
            s['measurement'] = [self.total_rew / 100.0, pos_y / 100.0]
            return s


if __name__ == '__main__':
    en = Environment(3)
    while True:
        en.action(np.random.randint(3))
        st = en.current_state()
        print(st['measurement'][0])
        time.sleep(0.1)