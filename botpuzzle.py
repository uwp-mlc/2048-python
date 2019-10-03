import pickle, neat, os
import logic
import puzzle

class BotGameGrid(puzzle.GameGrid):
    def key_down(self, event):
        with open('winner.pkl', 'rb') as filehandler:
            local_dir = os.path.dirname(__file__)
            config_path = os.path.join(local_dir, 'config-feedforward')
            config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
            winner = pickle.load(filehandler)
            net = neat.nn.FeedForwardNetwork.create(winner, config)

            # Initialize game
            game_matrix = logic.new_game(4)
            game_matrix = logic.add_two(game_matrix)
            game_matrix = logic.add_two(game_matrix)
            self.matrix = game_matrix
            self.update_grid_cells()

            # Flatten function
            flatten = lambda l: [item for sublist in l for item in sublist]

            # Action functions
            actions = [logic.up, logic.down, logic.left, logic.right]
            while logic.game_state(game_matrix) == 'not over':
                # Flatten game matrix
                flat_matrix = flatten(game_matrix)

                # Predict moves
                output = net.activate(flat_matrix)
                # Copy list and sort predictions from lowest to highest
                sorted_output = sorted(enumerate(output), key=lambda x:x[1])
                print(sorted_output)
                # Get max index from output list and use assosiated function from actions
                max_index = sorted_output[-1][0]
                new_game_matrix = actions[max_index](game_matrix)
                # If move is not valid use different direction
                if not new_game_matrix[1]:
                    # Get second max index from output list and use assosiated function from actions
                    second_max_index = sorted_output[-2][0]
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
                if logic.game_state(game_matrix) == 'not over':
                    game_matrix = logic.add_two(game_matrix)
                self.matrix = game_matrix
                self.history_matrixs.append(self.matrix)
                self.update_grid_cells()
                print(logic.game_state(game_matrix))
                # Generate new tile
                



gamegrid = BotGameGrid()