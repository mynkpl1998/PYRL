from collections import namedtuple
import random

Transition = namedtuple("Transition", ("state", "action", "mask", "next_state", "reward"))


class Memory(object):

	def __init__(self):
		self.memory = []

	def push(self, *args):
		self.memory.append(Transition(*args))

	def sample(self, batch_size=None):

		if batch_size is None:
			return Transition(*zip(*self.memory))
		else:
			random_batch = random.sample(self.memory, batch_size)
			return Transition(*zip(*random_batch))

	def append(self, new_memory):
		self.memory += new_memory.memory # merging two list

	def __len__(self):
		return len(self.memory)
