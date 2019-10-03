"""
A threaded version of XOR using neat.threaded.

Since most python implementations use a GIL, a threaded version probably won't
run any faster than the single-threaded version.

If your evaluation function is what's taking up most of your processing time
(and you should check by using a profiler while running single-threaded) and
your python implementation does not use a GIL,
you should see a significant performance improvement by evaluating using
multiple threads.

This example is only intended to show how to do a threaded experiment
in neat-python.  You can of course roll your own threading mechanism
or inherit from ThreadedEvaluator if you need to do something more complicated.
"""

from __future__ import print_function

import os

import neat, logic, time, pickle

try:
	import visualize
except ImportError:
	visualize = None

# 2-input XOR inputs and expected outputs.

xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
xor_outputs = [(0.0,),     (1.0,),     (1.0,),     (0.0,)]


def eval_genome(genome, config):
	"""
	This function will be run in threads by ThreadedEvaluator.  It takes two
	arguments (a single genome and the genome class configuration data) and
	should return one float (that genome's fitness).
	"""

	# Initialize game
	game_matrix = logic.new_game(4)
	game_matrix = logic.add_two(game_matrix)
	game_matrix = logic.add_two(game_matrix)

	# Flatten function
	flatten = lambda l: [item for sublist in l for item in sublist]
	net = neat.nn.FeedForwardNetwork.create(genome, config)

	# Action functions
	actions = [logic.up, logic.down, logic.left, logic.right]

	while logic.game_state(game_matrix) == 'not over':
		# Flatten game matrix
		flat_matrix = flatten(game_matrix)

		# Predict moves
		output = net.activate(flat_matrix)
		# Copy list and sort predictions from lowest to highest
		sorted_output = sorted(enumerate(output), key=lambda x:x[1])
		# Get max index from output list and use assosiated function from actions
		max_index = sorted_output[-1][0]
		new_game_matrix = actions[max_index](game_matrix)
		# If move is not valid use different direction
		if not new_game_matrix[1]:
			# Get second max index from output list and use assosiated function from actions
			second_max_index = sorted_output[-2][0]
			# TODO if output has same values all directions are not checked
			new_game_matrix = actions[second_max_index](game_matrix)
		# If move is not valid use different direction
		if not new_game_matrix[1]:
			# Get third max index from output list and use assosiated function from actions
			third_max_index = sorted_output[-3][0]
			new_game_matrix = actions[third_max_index](game_matrix)
		# If move is not valid use different direction
		if not new_game_matrix[1]:
			# Get fourth max index from output list and use assosiated function from actions
			fourth_max_index = sorted_output[-4][0]
			new_game_matrix = actions[fourth_max_index](game_matrix)

		# Set game matrix to updated matrix from (game, true) tuple
		game_matrix = new_game_matrix[0]
		# Generate new tile
		if logic.game_state(game_matrix) == 'not over':
			game_matrix = logic.add_two(game_matrix)
	#print(game_matrix)
	# Fitness function is a summation of all values on game board
	return sum(flatten(game_matrix))


def run(config_file):
	"""load the config, create a population, evolve and show the result"""
	# Load configuration.
	config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
						 neat.DefaultSpeciesSet, neat.DefaultStagnation,
						 config_file)

	# Create the population, which is the top-level object for a NEAT run.
	p = neat.Population(config)
	p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-299')
	# Add a stdout reporter to show progress in the terminal.
	p.add_reporter(neat.StdOutReporter(True))
	stats = neat.StatisticsReporter()
	p.add_reporter(stats)
	p.add_reporter(neat.Checkpointer(50))

	# Run for up to 300 generations.
	pe = neat.ThreadedEvaluator(4, eval_genome)
	winner = p.run(pe.evaluate, 1)
	filehandler = open("./winner.pkl", 'wb', pickle.HIGHEST_PROTOCOL) 
	pickle.dump(winner, filehandler)
	pe.stop()

	# Display the winning genome.
	print('\nBest genome:\n{!s}'.format(winner))


if __name__ == '__main__':
	# Determine path to configuration file. This path manipulation is
	# here so that the script will run successfully regardless of the
	# current working directory.
	local_dir = os.path.dirname(__file__)
	config_path = os.path.join(local_dir, 'config-feedforward')
	run(config_path)