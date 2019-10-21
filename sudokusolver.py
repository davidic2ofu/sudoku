import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from time import time
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

PUZZLES = {
	'old': '005003140070020005002001000200030050960000028050070003000300600400050080081400500',
	'expert': '060910007002000015000004000900300000000006000403000250004000008000008704005070030',
	'easy': '009002005538064009162000030003027000054600100007015340300801906700300850091000470',
	'medium': '000000609100004000005306821004670050007000900000540000370405206000000510060020037',
	'hard': '010000030730908200005310070100860402000005006000240000050409000006080040070000090',
	'expert2': '300009801006000009800070000000400000500000060070080130010000400000203900054001000',
	'expert3': '210000400000028000000000106000507608830000007000016003042300000053000070007900000',
	'expert4': '006000004000860730040350002170400600090000080008006017200081040067043000800000300',
}

MAX_NUM_ITERATIONS = 500000
TEMP = 3.0
TEMP_FACTOR = 0.999995


def timed(func):
	def wrapper(*args, **kwargs):
		start = time()
		func(*args, **kwargs)
		end = time()
		print('Finished in {} seconds\n'.format(end - start))
	return wrapper


class SudokuSolverBase(object):

	def __init__(self, s):
		p = []
		for i in range(9):
			row = list(s[i*9:i*9+9])
			row = [int(x) for x in row]
			p.append(row)
		self.blank_puzzle = np.array(p)
		self.puzzle = None

	def __str__(self):
		blank = ['Original Puzzle'] + self.printable(self.blank_puzzle)
		if self.puzzle is not None:
			solved = ['Solved Puzzle'] + self.printable(self.puzzle)
		else:
			solved = [''] * len(blank)
		ready_to_print = list(zip(blank, solved))

		s = ''
		for a, b in ready_to_print:
			s += '\n' + '{:45}{}'.format(a, b).strip()
		return s

	def printable(self, puzzle):
		output_list = ['╔═══╤═══╤═══╦═══╤═══╤═══╦═══╤═══╤═══╗']
		for i, row in enumerate(puzzle):
			row = [x or '' for x in row]
			if i in [3, 6]:
				output_list.append('╠═══╪═══╪═══╬═══╪═══╪═══╬═══╪═══╪═══╣')
			if i in [1, 2, 4, 5, 7, 8]:
				output_list.append('╟───┼───┼───╫───┼───┼───╫───┼───┼───╢')
			output_list.append('║ {:1} │ {:1} │ {:1} ║ {:1} │ {:1} │ {:1} ║ {:1} │ {:1} │ {:1} ║'.format(*row))
		output_list.append('╚═══╧═══╧═══╩═══╧═══╧═══╩═══╧═══╧═══╝')
		return output_list


class BacktrackingSolverBase(SudokuSolverBase):

	def __init__(self, s):
		self.sum = 0
		super().__init__(s)

	def __str__(self):
		s = '\nIterations: {}'.format(self.sum) if self.sum else ''
		return super().__str__() + s

	def select_cell(self, *args):
		pass

	def get_box_cell_values(self, y, x):
		boxrow = int(np.floor(y / 3) * 3)
		boxcol = int(np.floor(x / 3) * 3)
		box = self.puzzle[boxrow:boxrow + 3, boxcol:boxcol + 3].flatten()
		return box

	def candidates(self):
		return range(1, 10)

	def _solve(self):
		self.sum += 1
		if 0 not in self.puzzle:
			return True
		y, x = self.select_cell()
		row = self.puzzle[y, :]
		col = self.puzzle[:, x]
		box = self.get_box_cell_values(y, x)
		for candidate in self.candidates():
			if candidate not in np.concatenate((col, row, box)):
				self.puzzle[y, x] = candidate
				if self._solve():
					return True
				self.puzzle[y, x] = 0
		return False

	@timed
	def solve(self):
		self.sum = 0
		self.puzzle = self.blank_puzzle.copy()
		if not self._solve():
			raise RuntimeError('Invalid Puzzle')
		print(self)


class NaiveBacktrackingSolver(BacktrackingSolverBase):

	def select_cell(self):
		return [c[0] for c in np.where(self.puzzle == 0)]


class InformedBacktrackingSolver(BacktrackingSolverBase):

	def select_cell(self):
		zeros = {}
		y_coordinates, x_coordinates = np.where(self.puzzle == 0)
		coordinates = zip(y_coordinates, x_coordinates)
		for pair in coordinates:
			num_zeros_in_col = (self.puzzle[:, pair[1]] == 0).sum()
			num_zeros_in_row = (self.puzzle[pair[0], :] == 0).sum()
			num_zeros_in_box = (self.get_box_cell_values(pair[0], pair[1]) == 0).sum()
			zeros[pair] = min(num_zeros_in_col, num_zeros_in_row, num_zeros_in_box)
		return min(zeros, key=zeros.get)


class InformedSelectiveBacktrackingSolver(InformedBacktrackingSolver):

	def candidates(self):
		unique, counts = np.unique(self.puzzle, return_counts=True)
		d = dict(zip(unique, counts))
		del d[0]
		return sorted(d.keys(), key=d.get, reverse=True)


class SimulatedAnnealingSolver(SudokuSolverBase):

	def __init__(self, s, **kwargs):
		self.iter = kwargs.get('iter', MAX_NUM_ITERATIONS)
		self.temp = kwargs.get('temp', TEMP)
		self.temp_factor = kwargs.get('temp_factor', TEMP_FACTOR)
		self.start_temp = self.temp
		self.scores = []
		super().__init__(s)

	def __str__(self):
		s = super().__str__()
		s += '\nStarting temperature: {}'.format(self.start_temp)
		s += '\nTemp factor: {}'.format(self.temp_factor)
		if self.puzzle is not None:
			s += '\nEnding temperature: {}'.format(self.temp) 
			s += '\nFinal score {}'.format(self.scores[-1])
		return s

	@timed
	def solve(self):
		self.fill_in_with_initial_values()
		score = self.get_score(self.puzzle)
		
		for i in range(self.iter):
			self.scores.append(score)
			if i % 100 == 0 or score == 0:
				print('iteration {:6}, score {:3} {}'.format(i, score, '.' * score))
				if score == 0:
					break
			test_puzzle = self.flip_cells()
			test_score = self.get_score(test_puzzle)
			delta = np.absolute(test_score - score)
			boltz = np.exp(-1 * delta / self.temp)
			if test_score < score or boltz - np.random.random() > 0:
				self.puzzle = test_puzzle
				score = test_score
			self.temp *= self.temp_factor

		self.display_plot()
		print(self)

	def fill_in_with_initial_values(self):
		new_puzzle = np.zeros((9, 9))
		for i in range(len(self.blank_puzzle.T)):
			col = self.blank_puzzle[:, i]
			nums_to_fill = set(range(1, 10)) - set(col)
			num_generator = (n for n in nums_to_fill)
			new_col = []
			for c in col:
				if c:
					new_col.append(c)
				else:
					new_col.append(next(num_generator))
			new_puzzle[:, i] = new_col
		self.puzzle = new_puzzle.astype(int)

	def get_score(self, puzzle):
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

	def flip_cells(self):
		new_puzzle = self.puzzle.copy()
		while True:
			col_index = np.random.randint(0, 9)
			valid_indices = np.argwhere(self.blank_puzzle.T[col_index] == 0).flatten()
			if len(valid_indices) >= 2:
				break
		y1, y2 = np.random.choice(valid_indices, 2, replace=False)
		new_puzzle[y2, col_index] = self.puzzle[y1, col_index]
		new_puzzle[y1, col_index] = self.puzzle[y2, col_index]
		return new_puzzle

	def display_plot(self):
		plt.figure(figsize=(12,3))
		plt.scatter(
			range(len(self.scores)),
			self.scores,s=1,
			c=np.random.rand(len(self.scores)),
		)
		plt.title('Simulated Annealing Convergence Plot for Sudoku')
		plt.xlabel('Iterations')
		plt.ylabel('Score')
		plt.savefig('plot.png')
		os.system('open plot.png')


parser = argparse.ArgumentParser()

parser.add_argument(
	'-d', '--difficulty',
	action='store',
	choices=list(PUZZLES.keys()),
	default=list(PUZZLES.keys())[-1],
	help='pick difficulty level from saved puzzles',
)

parser.add_argument(
	'-sa', '--simulated_annealing',
	action='store_true',
	help='solve using simulated annealing',
)

parser.add_argument(
	'-p', '--puzzle',
	action='store',
	help='specify your own puzzle'
)


if __name__ == '__main__':

	args = parser.parse_args()
	arg_dict = vars(args)

	if arg_dict['simulated_annealing']:
		solvers = [
			SimulatedAnnealingSolver,
		]
	else:
		solvers = [
			NaiveBacktrackingSolver,
			InformedBacktrackingSolver,
			InformedSelectiveBacktrackingSolver,
		]

	if arg_dict['puzzle']:
		p = arg_dict['puzzle']
	else:
		difficulty = arg_dict['difficulty']
		p = PUZZLES[difficulty]

	for solver in solvers:
		print('\n' + solver.__name__)
		s = solver(p)
		s.solve()
