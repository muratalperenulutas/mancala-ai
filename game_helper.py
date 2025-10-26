from game.game import Game
from configs import Config
import numpy as np
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
    def play_against_model(game: Game, model, mode='dqn'):
        from model_helper import ModelHelper
        game.reset()
        game.initialize()
        print("Game started!")
        while not game.game_finish:
            state = game.get_board()
            player = game.current_player
            game.print_board()
            legal_actions = game.get_playable_pits()
            print("Playable moves:", legal_actions)
            if player == 0:
                while True:
                    try:
                        user_action = int(input("Enter your move (pit index): "))
                        if user_action in legal_actions:
                            break
                        else:
                            print("Invalid move! Playable moves are:", legal_actions)
                    except ValueError:
                        print("Please enter a valid number.")
                game.play(user_action)
                print(f"Your move: {user_action}")
            else:
                features = ModelHelper.build_features(game.get_board(), player, game.game_finish)
                normalized_state = ModelHelper.normalize_single_fixed(features)

                state_input = np.array(normalized_state).reshape(1, Config.INPUT_FEATURES)
                if mode == 'ppo':
                    logits, _ = model.predict(state_input, verbose=0)
                    q_values = logits[0]
                elif mode == 'dqn':
                    q_values = model.predict(state_input, verbose=0)[0]

                canonical_legal = [a - 7 for a in legal_actions if 7 <= a <= 12]

                legal_q_values = [q_values[a] for a in canonical_legal]

                best_action_index = np.argmax(legal_q_values)
                action = canonical_legal[best_action_index]
                game.play(GameHelper.action_to_board_index(action, player))
                print(f"Model's move: {action}")

        print("Game finished!")
        score_p1 = GameHelper.get_final_score(game=game, player=0)
        score_p2 = GameHelper.get_final_score(game=game, player=1)
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
