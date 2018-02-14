from vizdoom import *
import cv2
import config
import numpy as np
import time

AVAILABLE_ACTIONS = [[1,0,0],[0,1,0],[0,0,1]]
GOAL = [1,0]
MEASUREMENT_OF_INTEREST = 1

class Environment:
    def __init__(self, rand_seed, display = False):
        self.game = DoomGame()
        self.game.set_seed(rand_seed)
        self.game.load_config("scenarios\\basic.cfg")
        self.game.add_available_game_variable(GameVariable.POSITION_Y)
        self.game.set_window_visible(display)
        self.game.init()
        self.total_rew = 0

    def action(self, action):
        self.total_rew += self.game.make_action(AVAILABLE_ACTIONS[action], config.frame_repeat)

    def current_state(self):
        if self.game.is_episode_finished():
            self.game.new_episode()
            self.total_rew = 0
            return None
        else:
            screen = self.game.get_state().screen_buffer  # 3 x h x w
            screen = screen.transpose((1, 2, 0))  # h x w x 3
            whole_screen = cv2.resize(screen, config.resolution)  # 60 x 108 x 3
            whole_screen = whole_screen.astype(np.float32)
            pos_y = self.game.get_game_variable(GameVariable.POSITION_Y)
            s = dict()
            s['image'] = whole_screen
            s['measurement'] = [self.total_rew, pos_y]
            return s


if __name__ == '__main__':
    en = Environment(2)
    while True:
        st = en.current_state()
        if st is None:
            break
        else:
            en.action(np.random.randint(3))
            print(st['measurement'])
            time.sleep(0.1)
