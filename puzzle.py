import random
from tkinter import Frame, Label, CENTER

import logic
import constants as c
import pickle, neat, os


class GameGrid(Frame):
    def __init__(self):
        Frame.__init__(self)

        self.grid()
        self.master.title('2048')
        self.master.bind("<Key>", self.key_down)

        # self.gamelogic = gamelogic
        self.commands = {c.KEY_UP: logic.up, c.KEY_DOWN: logic.down,
                         c.KEY_LEFT: logic.left, c.KEY_RIGHT: logic.right,
                         c.KEY_UP_ALT: logic.up, c.KEY_DOWN_ALT: logic.down,
                         c.KEY_LEFT_ALT: logic.left,
                         c.KEY_RIGHT_ALT: logic.right}

        self.grid_cells = []
        self.init_grid()
        self.init_matrix()
        self.update_grid_cells()

        self.mainloop()

    def init_grid(self):
        background = Frame(self, bg=c.BACKGROUND_COLOR_GAME,
                           width=c.SIZE, height=c.SIZE)
        background.grid()

        for i in range(c.GRID_LEN):
            grid_row = []
            for j in range(c.GRID_LEN):
                cell = Frame(background, bg=c.BACKGROUND_COLOR_CELL_EMPTY,
                             width=c.SIZE / c.GRID_LEN,
                             height=c.SIZE / c.GRID_LEN)
                cell.grid(row=i, column=j, padx=c.GRID_PADDING,
                          pady=c.GRID_PADDING)
                t = Label(master=cell, text="",
                          bg=c.BACKGROUND_COLOR_CELL_EMPTY,
                          justify=CENTER, font=c.FONT, width=5, height=2)
                t.grid()
                grid_row.append(t)

            self.grid_cells.append(grid_row)

    def gen(self):
        return random.randint(0, c.GRID_LEN - 1)

    def init_matrix(self):
        self.matrix = logic.new_game(4)
        self.history_matrixs = list()
        self.matrix = logic.add_two(self.matrix)
        self.matrix = logic.add_two(self.matrix)

    def update_grid_cells(self):
        for i in range(c.GRID_LEN):
            for j in range(c.GRID_LEN):
                new_number = self.matrix[i][j]
                if new_number == 0:
                    self.grid_cells[i][j].configure(
                        text="", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                else:
                    self.grid_cells[i][j].configure(text=str(
                        new_number), bg=c.BACKGROUND_COLOR_DICT[new_number],
                        fg=c.CELL_COLOR_DICT[new_number])
        self.update_idletasks()

    def key_down(self, event):
        key = repr(event.char)
        if key == c.KEY_BACK and len(self.history_matrixs) > 1:
            self.matrix = self.history_matrixs.pop()
            self.update_grid_cells()
            print('back on step total step:', len(self.history_matrixs))
        elif key in self.commands:
            self.matrix, done = self.commands[repr(event.char)](self.matrix)
            if done:
                self.matrix = logic.add_two(self.matrix)
                # record last move
                self.history_matrixs.append(self.matrix)
                self.update_grid_cells()
                done = False
                if logic.game_state(self.matrix) == 'win':
                    self.grid_cells[1][1].configure(
                        text="You", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                    self.grid_cells[1][2].configure(
                        text="Win!", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                if logic.game_state(self.matrix) == 'lose':
                    self.grid_cells[1][1].configure(
                        text="You", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                    self.grid_cells[1][2].configure(
                        text="Lose!", bg=c.BACKGROUND_COLOR_CELL_EMPTY)

    def generate_next(self):
        index = (self.gen(), self.gen())
        while self.matrix[index[0]][index[1]] != 0:
            index = (self.gen(), self.gen())
        self.matrix[index[0]][index[1]] = 2

class BotGameGrid(GameGrid):
    def key_down(self, event):
        print("KEY DOWN")
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