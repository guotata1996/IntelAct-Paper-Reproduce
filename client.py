import multiprocessing
from util import nameClient
from environment import Environment
import zmq
from msgpack_numpy import dumps as dump
from msgpack_numpy import loads as load
from config import *

class ClientProcess(multiprocessing.Process):
	def __init__(self, index, pip_c2s=ip_addr+':8100', pip_s2c=ip_addr+':8101'):
		multiprocessing.Process.__init__(self)
		self.c2s = pip_c2s
		self.s2c = pip_s2c
		self.identity = nameClient(index)
		self.index = index
		print('client {} initialized'.format(index))

	def run(self):
		self.player = Environment(self.index * 113)
		context = zmq.Context()
		self.c2s_socket = context.socket(zmq.PUSH)
		self.c2s_socket.setsockopt(zmq.IDENTITY, self.identity)
		self.c2s_socket.connect(self.c2s)

		self.s2c_socket = context.socket(zmq.DEALER)
		self.s2c_socket.setsockopt(zmq.IDENTITY, self.identity)
		self.s2c_socket.connect(self.s2c)
		while True:
			obs = self.player.current_state()
			self.c2s_socket.send(dump((self.index, obs)), copy = False)
			if obs is not None:
				action = load(self.s2c_socket.recv(copy = False).bytes)
				self.player.action(action)