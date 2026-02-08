import sys
import datetime
from utils.configs import Config
import numpy as np
from utils.game_helper import GameHelper
import random
from tensorflow.keras.saving import save_model
from tensorflow.keras.models import load_model



class ModelHelper:

    @staticmethod
    def load_model(path):
        model = load_model(path)
        print(f"Model loaded from {path}.")
        return model

    @staticmethod
    def normalize_single_fixed(features, fixed_total=Config.FIXED_TOTAL):
        arr = np.array(features, dtype=np.float32).reshape(1, -1)
        board = arr[:, 0:14] / fixed_total          
        game_finish = arr[:, 14:15]                   
        store_diff = arr[:, 15:16] / fixed_total
        total_my = arr[:, 16:17] / fixed_total
        total_opp = arr[:, 17:18] / fixed_total
        empty_my = arr[:, 18:19] / 6.0
        empty_opp = arr[:, 19:20] / 6.0  
        return np.concatenate([board, game_finish, store_diff, total_my, total_opp,
                             empty_my, empty_opp], axis=1)[0]
                             
    @staticmethod
    def mask_invalid_actions(q_values, legal_actions, large_neg=-1e9):
        mask = np.zeros_like(q_values)
        mask[legal_actions] = 1
        return q_values * mask + (1 - mask) * large_neg

    @staticmethod
    def build_features(board14, player, game_finish):
        can_board = GameHelper.canonicalize_board(board14, player)
        extras = [
            can_board[7] - can_board[0],  
            sum(can_board[0:7]),         
            sum(can_board[7:14]),        
            sum(1 for i in can_board[0:7] if i == 0),
            sum(1 for i in can_board[7:14] if i == 0),
        ]

        features = np.array(can_board + [int(game_finish)] + list(extras), dtype=np.float32)
        return features
    
    @staticmethod
    def save_model(model, t=None):
        text=""
        if t ==None:
            text=datetime.now().strftime("%Y%m%d-%H%M%S")
        else:
            text=f"iter_{t+1}"

        save_model(model, f'models/mancala_model_{text}.keras',include_optimizer=True)
        print(f"Model saved {text}")


        
