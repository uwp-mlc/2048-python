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

import neat, logic, time

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
		print(output)
		# Copy list and sort predictions from lowest to highest
		sorted_output = sorted(output)
		print(sorted_output)
		# Get max index from output list and use assosiated function from actions
		max_index = output.index(sorted_output[-1])
		print(max_index)
		new_game_matrix = actions[max_index](game_matrix)
		print(new_game_matrix[1])
		# If move is not valid use different direction
		if not new_game_matrix[1]:
			# Get second max index from output list and use assosiated function from actions
			second_max_index = output.index(sorted_output[-2])
			# TODO if output has same values all directions are not checked
			print(second_max_index)
			new_game_matrix = actions[second_max_index](game_matrix)
			print(new_game_matrix[1])
		# If move is not valid use different direction
		if not new_game_matrix[1]:
			# Get third max index from output list and use assosiated function from actions
			third_max_index = output.index(sorted_output[-3])
			print(third_max_index)
			new_game_matrix = actions[third_max_index](game_matrix)
			print(new_game_matrix[1])
		# If move is not valid use different direction
		if not new_game_matrix[1]:
			# Get fourth max index from output list and use assosiated function from actions
			fourth_max_index = output.index(sorted_output[-4])
			print(fourth_max_index)
			new_game_matrix = actions[fourth_max_index](game_matrix)
			print(new_game_matrix[1])

		# Set game matrix to updated matrix from (game, true) tuple
		game_matrix = new_game_matrix[0]
		# Generate new tile
		if logic.game_state(game_matrix) == 'not over':
			game_matrix = logic.add_two(game_matrix)
		print(game_matrix)
	print(game_matrix)
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

	# Add a stdout reporter to show progress in the terminal.
	p.add_reporter(neat.StdOutReporter(True))
	stats = neat.StatisticsReporter()
	p.add_reporter(stats)

	# Run for up to 300 generations.
	pe = neat.ParallelEvaluator(1, eval_genome)
	winner = p.run(pe.evaluate, 300)
	pe.stop()

	# Display the winning genome.
	print('\nBest genome:\n{!s}'.format(winner))

	# Show output of the most fit genome against training data.
	print('\nOutput:')
	winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
	for xi, xo in zip(xor_inputs, xor_outputs):
		output = winner_net.activate(xi)
		print(
			"input {!r}, expected output {!r}, got {!r}".format(xi, xo, output)
			)

	if visualize is not None:
		node_names = {-1: 'A', -2: 'B', 0: 'A XOR B'}
		visualize.draw_net(config, winner, True, node_names=node_names)
		visualize.plot_stats(stats, ylog=False, view=True)
		visualize.plot_species(stats, view=True)


if __name__ == '__main__':
	# Determine path to configuration file. This path manipulation is
	# here so that the script will run successfully regardless of the
	# current working directory.
	local_dir = os.path.dirname(__file__)
	config_path = os.path.join(local_dir, 'config-feedforward')
	run(config_path)