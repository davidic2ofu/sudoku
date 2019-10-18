import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from time import time

from puzzles import PUZZLES, print_puzzle


MAX_NUM_ITERATIONS = 500000
TEMP = 3.0
TEMP_FACTOR = 0.999995


def timed(func):
	def wrapper(*args, **kwargs):
		start = time()
		func(*args, **kwargs)
		end = time()
		return end - start
	return wrapper


def display_plot(scores):
	plt.figure(figsize=(12,3))
	plt.scatter(range(len(scores)), scores, s=1, c=np.random.rand(len(scores)),)
	plt.title('Simulated Annealing Convergence Plot for Sudoku')
	plt.xlabel('Iterations')
	plt.ylabel('Score')
	plt.savefig('plot.png')
	os.system('open plot.png')


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
def handle(empty_puzzle, temp=TEMP, temp_factor=TEMP_FACTOR):
	start_temp = temp

	scores = []
	puzzle = fill_in_with_initial_values(empty_puzzle)
	score = get_score(puzzle)
	
	for i in range(MAX_NUM_ITERATIONS):
		scores.append(score)
		if i % 100 == 0 or score == 0:
			print('iteration {:6}, score {:3} {}'.format(i, score, '.' * score))
			if score == 0:
				break
		test_puzzle = flip_cells(empty_puzzle, puzzle)
		test_score = get_score(test_puzzle)
		delta = np.absolute(test_score - score)
		boltz = np.exp(-1 * delta / temp)
		if test_score < score or boltz - np.random.random() > 0:
			puzzle = test_puzzle
			score = test_score
		temp *= temp_factor

	display_plot(scores)

	print_puzzle(empty_puzzle)
	print_puzzle(puzzle)
	print('Starting temperature: {}'.format(start_temp))
	print('Ending temperature: {}'.format(temp)) 
	print('Temp factor: {}'.format(temp_factor))
	print('Final score {}'.format(score))


if __name__ == '__main__':
	empty_puzzle = PUZZLES['expert']
	print(sys.argv)
	if len(sys.argv) == 3:
		temp = float(sys.argv[1])
		temp_factor = float(sys.argv[2])
		num_seconds = handle(empty_puzzle, temp, temp_factor)
	else:
		num_seconds = handle(empty_puzzle)
	print('{} seconds'.format(num_seconds))
