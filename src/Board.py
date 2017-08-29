__author__ = 'rmozes1'


# Tic Tac Toe

import sys


class Status:
    INVALID_MOVE = -1
    SUCCESS = 0
    WIN = 1
    GAME_OVER = 2

class TTTBoard:
    def __init__(self, cols=5, rows=5, seq_size=4):
        self.cols = cols
        self.rows = rows
        self.board = [[0 for i in range(rows)] for j in range(cols)]
        self.seq_size = seq_size
        self.next_player = 1

    def get(self):
        return self.board

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

    def play(self, player, pos_x, pos_y):

        if self.board[pos_y][pos_x] != 0:
            return Status.INVALID_MOVE

        self.board[pos_y][pos_x] = player

        if (self.win_row(player, pos_x, pos_y) or self.win_col(player, pos_x, pos_y) or
                self.win_diagonal(player, pos_x, pos_y) or self.win_reversed_diagonal(player, pos_x, pos_y)):
            return Status.WIN

        if self.game_over():
            return Status.GAME_OVER

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
                sys.stdout.write(' | ' + str(self.board[i][j]))
            sys.stdout.write(' |')
            sys.stdout.write('\n -')
            for j in range(self.cols):
                sys.stdout.write('----')
            sys.stdout.write('\n')
