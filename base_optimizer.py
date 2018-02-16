import threading
import zmq
from msgpack_numpy import dumps as dump
from msgpack_numpy import loads as load
from tornado.concurrent import Future
from six.moves import queue
import time
import random
import os
from tqdm import tqdm
from queue import Queue

from intelActNet import Network
from config import config
from util import *

class MasterProcess(threading.Thread):
	def __init__(self, pipe_c2s=config.ip_addr+':8100', pipe_s2c=config.ip_addr+':8101'):
		super(MasterProcess, self).__init__()
		self.context = zmq.Context()
		self.c2s_socket = self.context.socket(zmq.PULL)
		self.c2s_socket.bind(pipe_c2s)
		self.s2c_socket = self.context.socket(zmq.ROUTER)
		self.s2c_socket.bind(pipe_s2c)
		self.start_time = time.time()
		self.network = Network()
		if config.continue_training:
			self.network.restore()

		self.predict_queue = queue.Queue(maxsize = config.agent_num)
		self.training_queue = queue.Queue(maxsize=config.agent_num*100)

		self.predictor = PredictThread(self.predict_queue, self.network)
		self.trainer = TrainingThread(self.training_queue, self.network)
		self.client_memory = [Queue(maxsize=35) for _ in range(config.agent_num)]

		self.predictor.start()
		self.trainer.start()

	def _put_predict_task(self, observation, callback):
		f = Future()
		f.add_done_callback(callback)
		if observation is not None:
			self.predict_queue.put([observation, f])
		else:
			f.set_result(None)
		return f

	def parse_memory(self, ident, observation, predicting_result):
		raise(NotImplementedError())

	def _on_state(self, index, obs):
		def cb(output):
			predicting_result = output.result()
			if predicting_result is not None:
				self.s2c_socket.send_multipart([nameClient(index), dump(predicting_result['action'])])
			training_data = self.parse_memory(index, obs, predicting_result)
			if training_data is not None:
				self.training_queue.put(training_data)

		self._put_predict_task(obs, cb)  # add one dimension as time axis

	def run(self):
		while True:
			for _ in tqdm(range(config.save_freq)):
				client_id, observations = load(self.c2s_socket.recv(copy = False).bytes)
				self._on_state(client_id, observations)

			self.network.save()
			config.update()

class PredictThread(threading.Thread):
	def __init__(self, predicting_queue, network):
		super(PredictThread, self).__init__()
		self.recv_queue = predicting_queue
		self.network = network

	def run(self):
		while True:
			predicting_batch = []
			future_batch = []
			observation, future = self.recv_queue.get()
			predicting_batch.append(observation)
			future_batch.append(future)
			while len(predicting_batch) < config.batch_size:
				try:
					observation, future = self.recv_queue.get_nowait()
					predicting_batch.append(observation)
					future_batch.append(future)
				except queue.Empty:
					break

			predicting_results = self.network.predict(predicting_batch)

			for c in range(predicting_results.shape[0]):
				predicting_result = predicting_results[c]
				if random.random() > config.p_explore:
					rtn_action = np.where(predicting_result == max(predicting_result))[0][0]
				else:
					rtn_action = np.random.choice(range(config.num_actions))

				final_result = dict()
				final_result['action'] = rtn_action
				future_batch[c].set_result(final_result)


class TrainingThread(threading.Thread):
	def __init__(self, training_queue, network):
		super(TrainingThread, self).__init__()
		self.recv_queue = training_queue
		self.network = network

	def run(self):
		while True:
			training_batch = []
			training_data = self.recv_queue.get()
			training_batch.append(training_data)
			while len(training_batch) < config.batch_size:
				try:
					training_data = self.recv_queue.get_nowait()
					training_batch.append(training_data)
				except queue.Empty:
					break

			self.network.train(training_batch)