from game.game import Game
from utils.configs import *
from utils.model_helper import ModelHelper
import numpy as np

class GameHelper:
    @staticmethod
    def play_against_model(game: Game, model):
        print("Game started!")
        while not game.game_finish:
            game.display_board()
            legal_actions = game.get_playable_pits(symmetry=True)
            
            if game.current_player == 0:
                action = GameHelper._get_user_move(legal_actions)
                game.play(action)
                print(f"Your move: {action}")
            else:
                action = GameHelper._get_model_move(game, model, legal_actions)
                game.play(action, symmetry=True)
                print(f"Model's move: {action}")

        GameHelper._print_final_results(game)

    @staticmethod
    def _print_final_results(game):
        print("Game finished!")
        s1, s2 = game.board[game.bank0i], game.board[game.bank1i]
        print(f"Your score: {s1}, Model's score: {s2}")
        if s1 > s2: print("You win!")
        elif s2 > s1: print("Model wins!")
        else: print("Draw!")
    
    @staticmethod
    def _get_user_move(legal_actions):
        while True:
            try:
                move = int(input(f"Enter your move {legal_actions}: "))
                if move in legal_actions:
                    return move
                print(f"Invalid move! Playable moves: {legal_actions}")
            except ValueError:
                print("Please enter a valid number.")
    
    @staticmethod
    def _get_model_move(game, model, legal_actions):
        features = ModelHelper.build_features(game)
        state_input = np.array(ModelHelper.normalize_single_fixed(features)).reshape(1, INPUT_FEATURES)

        # PPO has 2 outputs (actor, critic)
        predictions = model(state_input, training=False)
        logits = predictions[0]    
        q_values = logits[0]

        legal_q_values = [q_values[a] for a in legal_actions]
        return legal_actions[np.argmax(legal_q_values)]            

    @staticmethod
    def play_between_models(model1, model2, num_games=100):
        wins = 0
        draws = 0
        losses = 0
        
        for i in range(num_games):
            game = Game()
            p1_starts = i % 2 == 0
            
            while not game.game_finish:
                legal_actions = game.get_playable_pits(symmetry=True)
                current_player = game.current_player
                
                if (p1_starts and current_player == 0) or (not p1_starts and current_player == 1):
                    active_model = model1
                else:
                    active_model = model2
                
                action = GameHelper._get_model_move(game, active_model, legal_actions)
                game.play(action, symmetry=True)
            
            s1 = game.board[game.bank0i]
            s2 = game.board[game.bank1i]
            
            m1_score = s1 if p1_starts else s2
            m2_score = s2 if p1_starts else s1
            
            if m1_score > m2_score:
                wins += 1
            elif m1_score < m2_score:
                losses += 1
            else:
                draws += 1
                
        return wins, draws, losses
    
    @staticmethod
    def get_final_score(game: Game, player: int):
        if player == 0:
            return game.board[game.bank0i]
        else:
            return game.board[game.bank1i]

    @staticmethod
    def get_move_score(game: Game):
        score = game.stones_earned * STONES_EARNED_WEIGHT
        score += SECOND_MOVE_WEIGHT if game.second_move else 0
        
        pits_range = range(0, 6) if game.current_player == 0 else range(7, 13)
        score += sum(game.board[i] for i in pits_range) * PIT_STONES_WEIGHT
        
        if game.game_finish:
            my_bank = game.board[game.bank0i if game.current_player == 0 else game.bank1i]
            opponent_bank = game.board[game.bank1i if game.current_player == 0 else game.bank0i]
            
            if my_bank > opponent_bank:
                score += WIN_REWARD
            elif my_bank < opponent_bank:
                score -= WIN_REWARD
            else:
                score += DRAW_REWARD
                
        return score