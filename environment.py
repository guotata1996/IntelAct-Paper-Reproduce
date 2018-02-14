from vizdoom import *
import cv2
from util import *
import numpy
import time

AVAILABLE_ACTIONS = numpy.eye(9)[:7,:].tolist()
GOAL = [1,0.5,0.5,0,0]
MEASUREMENT_OF_INTEREST = 3
frame_repeat = 4
resolution = (128,128)
bots_num = 8

class Environment:
    def __init__(self, rand_seed, display = True, HAND_MODE = False):
        self.game = DoomGame()
        self.game.set_seed(rand_seed)
        self.game.load_config("scenarios\\cig.cfg")
        self.game.set_doom_map("map01")  # Limited deathmatch.
        self.game.add_game_args("-host 1 -deathmatch +timelimit 1.0 "
                                "+sv_forcerespawn 1 +sv_noautoaim 1 +sv_respawnprotect 1 +sv_spawnfarthest 1")
        self.game.add_game_args("+name AI +colorset 0")

        self.game.set_labels_buffer_enabled(True)
        self.game.add_available_game_variable(GameVariable.HEALTH)
        self.game.add_available_game_variable(GameVariable.SELECTED_WEAPON_AMMO)
        self.game.add_available_game_variable(GameVariable.FRAGCOUNT)
        self.game.add_available_game_variable(GameVariable.ANGLE)

        self.total_frag_count = 0
        self.kill = 0
        self.death = 0
        self.opponent_location = []

        self.game.set_window_visible(display)

        if HAND_MODE:
            self.game.set_mode(Mode.SPECTATOR)
        self.game.init()
        self.game.send_game_command("removebots")
        for i in range(bots_num):
            self.game.send_game_command("addbot")

        self.hand_mode = HAND_MODE
        print ("Doomgame instance established")


    def action(self, action):
        self.game.make_action(AVAILABLE_ACTIONS[action], frame_repeat)

    def current_state(self):
        if self.game.is_episode_finished():
            self.game.new_episode()
            return None
        else:
            screen = self.game.get_state().screen_buffer  # 3 x h x w
            screen = screen.transpose((1, 2, 0))  # h x w x 3
            whole_screen = cv2.resize(screen, resolution)  # 60 x 108 x 3
            whole_screen = whole_screen.astype(np.float32)
            health = self.game.get_game_variable(GameVariable.HEALTH) / 30.0
            frag = self.game.get_game_variable(GameVariable.FRAGCOUNT)
            engage = self.game.get_game_variable(GameVariable.USER1)
            ammo = self.game.get_game_variable(GameVariable.SELECTED_WEAPON_AMMO) / 7.5
            angle = self.game.get_game_variable(GameVariable.ANGLE) / 90.0
            s = dict()
            s['image']= whole_screen
            s['measurement'] = [frag, health, ammo, angle, engage]
            if self.game.is_player_dead():
                self.game.respawn_player()
            return s


if __name__ == '__main__':
    en = Environment(13)
    while True:
        en.action(numpy.random.randint(4,7))
        time.sleep(0.1)
