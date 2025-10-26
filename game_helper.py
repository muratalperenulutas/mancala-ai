from game.game import Game
class GameHelper:
    @staticmethod
    def action_to_board_index(action, current_player):
        if current_player == 0:
            return action
        else:
            return action + 7

    @staticmethod
    def canonicalize_board(board14, player):
        b = list(board14)
        if player == 2:
            return b[7:14] + b[0:7]
        else:
            return b 

    @staticmethod
    def play_against_model(game: Game, model):
        game.reset()
        game.initialize()
        print("Game started!")
        while not game.game_finish:
            state = game.get_status()
            player = state[14]
            game.print_board()
            #print("Board state:", state)
            legal_actions = game.get_playable_pits()
            print("Playable moves:", legal_actions)
            if player == 1:
                while True:
                    try:
                        user_action = int(input("Enter your move (pit index): "))
                        if user_action in legal_actions:
                            break
                        else:
                            print("Invalid move! Playable moves are:", legal_actions)
                    except ValueError:
                        print("Please enter a valid number.")
                game.play(user_action, player)
                print(f"Your move: {user_action}")
            else:
                state_input = np.array(state).reshape(1, 22)
                q_values = model.predict(state_input, verbose=0)[0]
                best_action_index = np.argmax(q_values[legal_actions])
                action = legal_actions[best_action_index]
                game.play(action, player)
                print(f"Model's move: {action}")

        print("Game finished!")
        score_p1 = GameHelper.get_final_score(board=game.get_board(), player=1)
        score_p2 = GameHelper.get_final_score(board=game.get_board(), player=2)
        print(f"Your score: {score_p1}")
        print(f"Model's score: {score_p2}")
        if score_p1 > score_p2:
            print("You win!")
        elif score_p2 > score_p1:
            print("Model wins!")
        else:
            print("Draw!")  


    @staticmethod
    def get_move_score(game: Game):
        score = 0
        score += game.stones_earned / 2
        score += 0.3 if game.second_move else 0
        pits_range = range(0, 6) if game.current_player == 0 else range(7, 13)
        score += sum(game.node_dict[i].stone_count for i in pits_range) * 0.01
        if game.game_finish:
            if game.old_player == 0 and game.bank1.stone_count > game.bank2.stone_count:
                score += 1
            elif game.old_player == 1 and game.bank2.stone_count > game.bank1.stone_count:
                score += 1
            elif game.bank1.stone_count == game.bank2.stone_count:
                score += 0.4
            else:
                score += -1
        return score

    @staticmethod
    def get_final_score(game: Game, player: int):
        if player == 0:
            return game.bank1.stone_count
        else:
            return game.bank2.stone_count
