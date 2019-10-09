import numpy as np
from time import time

from puzzles import PUZZLES


TEMP = 2.25
NUM_ITERATIONS = 500000


def timed(func):
	def wrapper(*args, **kwargs):
		start = time()
		func(*args, *kwargs)
		end = time()
		return end - start
	return wrapper


def fill_in_with_initial_values(puzzle):
	new_puzzle = np.zeros((9, 9))
	for i in range(len(puzzle.T)):
		col = puzzle[:, i]
		nums_to_fill = set(range(1, 10)) - set(col)
		num_generator = (n for n in nums_to_fill)
		new_col = []
		for c in col:
			if c:
				new_col.append(c)
			else:
				new_col.append(next(num_generator))
		new_puzzle[:, i] = new_col
	return new_puzzle.astype(int)


def get_score(puzzle):
	score = 0
	for row in puzzle:
		missing = set(range(1, 10)) - set(row)
		score += len(missing)
	for i in range(3):
		for j in range(3):
			row_index = i * 3
			col_index = j * 3			
			subspace = puzzle[row_index:row_index + 3, col_index:col_index + 3]
			missing = set(range(1, 10)) - set(subspace.flatten())
			score += len(missing)
	return score


def flip_cells(empty_puzzle, puzzle):
	new_puzzle = puzzle.copy()
	while True:
		col_index = np.random.randint(0, 9)
		valid_indices = np.argwhere(empty_puzzle.T[col_index] == 0).flatten()
		if len(valid_indices) >= 2:
			break
	y1, y2 = np.random.choice(valid_indices, 2, replace=False)
	new_puzzle[y2, col_index] = puzzle[y1, col_index]
	new_puzzle[y1, col_index] = puzzle[y2, col_index]
	return new_puzzle


@timed
def handle(empty_puzzle):
	temp = TEMP
	scores = []
	puzzle = fill_in_with_initial_values(empty_puzzle)
	score = get_score(puzzle)
	
	for i in range(NUM_ITERATIONS):
		scores.append(score)
		if i % 50 == 0 or score == 0:
			print('iteration {}, score {}'.format(i, score))
			if score == 0:
				break
		test_puzzle = flip_cells(empty_puzzle, puzzle)
		test_score = get_score(test_puzzle)
		delta = np.absolute(test_score - score)
		boltz = np.exp(-1 * delta / temp)
		if test_score < score or boltz - np.random.random() > 0:
			puzzle = test_puzzle
			score = test_score
		temp *= 0.999995

	print(empty_puzzle)
	print(puzzle)
	print('final score {}'.format(score))


if __name__ == '__main__':
	empty_puzzle = PUZZLES[0]
	num_seconds = handle(empty_puzzle)
	print('{} seconds'.format(num_seconds))
