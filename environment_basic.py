from vizdoom import *
import cv2

AVAIABLE_ACTIONS = [[1,0,0],[0,1,0],[0,0,1]]
class Environment:
    def __init__(self, rand_seed, display = False):
        self.game = DoomGame()
        self.game.set_seed(rand_seed)
        