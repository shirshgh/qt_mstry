from collections import namedtuple
from collections import deque

import random

transition = namedtuple('transition',
						('state', 'action', 'reward', 'next_state', 'done'))

class replay_memory():

	def __init__(self, capacity):
		self.capacity = capacity
		self.memory = []
		self.hashes = set()
		self.position = 0

	def push(self, *args):
		"""Saves a transition."""

		hash = self.calc_tuple_hash(*args)
		if (hash in self.hashes):
			return

		self.hashes.add(hash)
		if len(self.memory) < self.capacity:
			self.memory.append(None)
		self.memory[self.position] = transition(*args)

		self.position = (self.position + 1) % self.capacity

	def calc_tuple_hash(self,*args):

		state, action, reward, next_state, done = args
		hash_sum = 			  (hash(str(state)))
		hash_sum = hash_sum + (hash(action))
		hash_sum = hash_sum + (hash(reward))
		hash_sum = hash_sum + (hash(str(next_state)))
		hash_sum = hash_sum + (hash(done))
		return hash_sum

	def sample(self, batch_size):
		return random.sample(self.memory, batch_size)

	def __len__(self):
		return len(self.memory)

	def printall(self):
		for m in self.memory:
			print(m)