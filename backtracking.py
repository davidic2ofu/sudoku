import numpy as np

from puzzles import PUZZLES, print_puzzle


def recurse(puzzle):
	if not 0 in puzzle:
		print_puzzle(puzzle)
		return True
	y, x = [c[0] for c in np.where(puzzle == 0)]
	row = puzzle[y, :]
	col = puzzle[:, x]
	boxrow = int(np.floor(y / 3) * 3)
	boxcol = int(np.floor(x / 3) * 3)
	box = puzzle[boxrow:boxrow + 3, boxcol:boxcol + 3].flatten()
	for candidate in range(1, 10):
		if candidate not in np.concatenate((col, row, box)):
			puzzle[y, x] = candidate
			if recurse(puzzle):
				return True
			puzzle[y, x] = 0
	return False
