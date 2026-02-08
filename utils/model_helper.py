import datetime
from game.game import Game
from utils.configs import Config
import numpy as np
from tensorflow.keras.models import load_model, save_model

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
        
        return np.concatenate([
            board, game_finish, store_diff, total_my, total_opp, empty_my, empty_opp
        ], axis=1)[0]
                             
    @staticmethod
    def mask_invalid_actions(q_values, legal_actions, large_neg=-1e9):
        mask = np.zeros_like(q_values)
        mask[legal_actions] = 1
        return q_values * mask + (1 - mask) * large_neg

    @staticmethod
    def build_features(game: Game):
        board14 = game.get_board(symmetry=True)
        
        my_side, opp_side = board14[0:7], board14[7:14]
        store_diff = board14[7] - board14[0]
        
        extras = [
            store_diff,
            sum(my_side),
            sum(opp_side),
            sum(1 for x in my_side if x == 0),
            sum(1 for x in opp_side if x == 0)
        ]

        return np.array(board14 + [int(game.game_finish)] + extras, dtype=np.float32)
    
    @staticmethod
    def save_model(model, t=None):
        if t is None:
            suffix = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        else:
            suffix = f"iter_{t}"

        save_model(model, f'models/mancala_model_{suffix}.keras', include_optimizer=True)
        print(f"Model saved: {suffix}")


        
