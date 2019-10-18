import numpy as np
import sys

from puzzles import PUZZLES, print_puzzle

sum = 0


def get_box_cell_values(puzzle, y, x):
	boxrow = int(np.floor(y / 3) * 3)
	boxcol = int(np.floor(x / 3) * 3)
	box = puzzle[boxrow:boxrow + 3, boxcol:boxcol + 3].flatten()
	return box


def naive_select_cell(puzzle):
	return [c[0] for c in np.where(puzzle == 0)]


def informed_select_cell(puzzle):
	'''
	https://medium.com/my-udacity-ai-nanodegree-notes/solving-sudoku-think-constraint-satisfaction-problem-75763f0742c9
	from our lecture slides:
	agent: perceives environment through sensors and acts on environment through actuators
	rational means intelligent
	rational agent: environment, percepts, actions, rationality, agent program
	selects actions to maximize utility function
	characteristics of percepts, environment, action space dictate techniques for selecting
	rational actions.
	"map the percepts to the actions to maximize the performance measure"
	learn to compensate for partial or incorrect background knowledge (like a human)
	modeling the external world (puzzle and rules, constraints), dealing with uncertainties
	single agent, deterministic/stochastic (?), partially observable?, discrete
	'''
	zeros = {}
	y_coordinates, x_coordinates = np.where(puzzle == 0)
	coordinates = zip(y_coordinates, x_coordinates)
	for pair in coordinates:
		num_zeros_in_col = (puzzle[:, pair[1]] == 0).sum()
		num_zeros_in_row = (puzzle[pair[0], :] == 0).sum()
		num_zeros_in_box = (get_box_cell_values(puzzle, pair[0], pair[1]) == 0).sum()
		zeros[pair] = min(num_zeros_in_col, num_zeros_in_row, num_zeros_in_box)
	return min(zeros, key=zeros.get)


def solve(puzzle, select_cell):
	global sum
	sum += 1
	if not 0 in puzzle:
		print_puzzle(puzzle)
		return True
	y, x = select_cell(puzzle)
	row = puzzle[y, :]
	col = puzzle[:, x]
	box = get_box_cell_values(puzzle, y, x)
	for candidate in range(1, 10):
		if candidate not in np.concatenate((col, row, box)):
			puzzle[y, x] = candidate
			if solve(puzzle, select_cell):
				return True
			puzzle[y, x] = 0
	return False


def run(puzz):
	for diff in [naive_select_cell, informed_select_cell]:
		global sum
		sum = 0
		p = PUZZLES[puzz].copy()
		solve(p, diff)
		print(puzz, str(diff), sum)


if __name__ == '__main__':
	if len(sys.argv) == 2:
		run(sys.argv[1])
	else:
		for k in PUZZLES.keys():
			run(k)
