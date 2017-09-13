__author__ = 'rmozes1'
import sys
import numpy as np

class Status:
	INVALID_MOVE = -1.01
	SUCCESS = 0.1
	WIN = 1
	LOSE = -1

class TTTBoard:
	def __init__(self, cols=5, rows=5, seq_size=4):
		self.cols = cols
		self.rows = rows
		self.board = [[0 for i in range(rows)] for j in range(cols)]
		self.seq_size = seq_size
		self.next_player = 1

	def get(self):
		return np.reshape(np.array(self.board),(1,self.cols*self.rows))
	
	def set(self,state):
		self.board = [[state[0],state[1],state[2]],[state[3],state[4],state[5]],[state[6],state[7],state[8]]]

	def reset(self):
		  self.board = [[0 for i in range(self.rows)] for j in range(self.cols)]

	def get_new_player(self):
		self.next_player += 1
		return self.next_player - 1

	def win_row(self, player, pos_x, pos_y):
		tmp_seq_size = 0
		for i in range(max(0, pos_x - self.seq_size + 1), min(self.cols, pos_x + self.seq_size)):
			if self.board[pos_y][i] == player:
				tmp_seq_size += 1
				if tmp_seq_size == self.seq_size:
					return True
			else:
				tmp_seq_size = 0
		return False

	def win_col(self, player, pos_x, pos_y):
		tmp_seq_size = 0
		for j in range(max(0, pos_y - self.seq_size + 1), min(self.rows, pos_y + self.seq_size)):
			if self.board[j][pos_x] == player:
				tmp_seq_size += 1
				if tmp_seq_size == self.seq_size:
					return True
			else:
				tmp_seq_size = 0
		return False

	def win_diagonal(self, player, pos_x, pos_y):
		tmp_seq_size = 0
		diff = pos_x - pos_y;
		x_range = range(max(0, pos_x - self.seq_size + 1), min(self.cols, pos_x + self.seq_size))
		for i in x_range:
			j = i - diff
			if j in range(0, self.rows):
				if self.board[j][i] == player:
					tmp_seq_size += 1
					if tmp_seq_size == self.seq_size:
						return True
				else:
					tmp_seq_size = 0
		return False

	def win_reversed_diagonal(self, player, pos_x, pos_y):
		tmp_seq_size = 0
		sum = pos_x + pos_y
		x_range = range(max(0, pos_x - self.seq_size + 1), min(self.cols, pos_x + self.seq_size))
		for i in x_range:
			j = sum - i
			if j in range(0, self.rows):
				if self.board[j][i] == player:
					tmp_seq_size += 1
					if tmp_seq_size == self.seq_size:
						return True
				else:
					tmp_seq_size = 0
		return False

	def game_over(self):
		for i in range (self.rows):
			for j in range ( self.cols):
				if self.board[i][j] == 0:
					return False
		return True

	def play_random(self,player):

		pos = np.random.randint(self.cols*self.rows)
		pos_x = int(pos % self.cols)
		pos_y = int(pos / self.cols)
		while self.board[pos_y][pos_x] != 0:
			pos = np.random.randint(self.cols*self.rows)
			pos_x = int(pos % self.cols)
			pos_y = int(pos / self.cols)

		return pos,self.play(player,pos)

	def play(self, player, pos):

		pos_x = int(pos % self.cols)
		pos_y = int(pos / self.cols)
		if self.board[pos_y][pos_x] != 0:
			return Status.INVALID_MOVE

		self.board[pos_y][pos_x] = player

		if (self.win_row(player, pos_x, pos_y) or self.win_col(player, pos_x, pos_y) or
				self.win_diagonal(player, pos_x, pos_y) or self.win_reversed_diagonal(player, pos_x, pos_y)):
			return Status.WIN

		return Status.SUCCESS

	def draw(self):

		# draw top line
		sys.stdout.write(' -')
		for j in range(self.cols):
				sys.stdout.write('----')
		sys.stdout.write('\n')

		# draw board
		for i in range(self.rows):
			for j in range(self.cols):
				if self.board[i][j] == 1:
					sign = 'X'
				if self.board[i][j] == -1:
					sign = 'O'
				if self.board[i][j] == 0:
					sign = ' '
				sys.stdout.write(' | ' + sign)
			sys.stdout.write(' |')
			sys.stdout.write('\n -')
			for j in range(self.cols):
				sys.stdout.write('----')
			sys.stdout.write('\n')
