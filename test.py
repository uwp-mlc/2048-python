import logic

game_matrix = logic.new_game(4)
game_matrix = logic.add_two(game_matrix)
flat_matrix = [item for sublist in game_matrix for item in sublist]

print(flat_matrix)