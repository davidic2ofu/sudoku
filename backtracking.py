import numpy as np

from puzzles import PUZZLES, print_puzzle


def get_box_cell_values(puzzle, y, x):
	boxrow = int(np.floor(y / 3) * 3)
	boxcol = int(np.floor(x / 3) * 3)
	box = puzzle[boxrow:boxrow + 3, boxcol:boxcol + 3].flatten()
	return box


def naive_select_cell(puzzle):
	return [c[0] for c in np.where(puzzle == 0)]


def informed_select_cell(puzzle):
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
