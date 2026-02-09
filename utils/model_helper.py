import datetime
import os
import json
import random
from game.game import Game
from utils.configs import *
import numpy as np
from tensorflow.keras.models import load_model, save_model

class ModelHelper:
    @staticmethod
    def load_model(name=None, id=None):
        model = load_model(os.path.join('models', name))
        print(f"Model loaded from {os.path.join(os.curdir, 'models', name)}.")
        return model

    @staticmethod
    def normalize_single_fixed(features, fixed_total=FIXED_TOTAL):
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
    def save_model(model, t=None, metric=None, elo=1200):
        if not os.path.exists('models'):
            os.makedirs('models')

        if t is None:
            suffix = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        else:
            suffix = f"iter_{t}"
            
        model_name = f'mancala_model_{suffix}.keras'
        model_path = os.path.join('models', model_name)
        save_model(model, model_path, include_optimizer=True)
        
        entry = {
            'id': suffix,
            'name': model_name,
            'metric': metric,
            'elo': elo,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        json_path = 'models/models.json'
        models_list = []
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    models_list = json.load(f)
            except (json.JSONDecodeError, IOError):
                models_list = []
            
        models_list.append(entry)
        with open(json_path, 'w') as f:
            json.dump(models_list, f, indent=4)

        print(f"Model saved: {suffix}")
    
    @staticmethod
    def list_saved_models():
        json_path = 'models/models.json'
        if not os.path.exists(json_path):
            return []
        try:
            with open(json_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []
    
    @staticmethod
    def get_opponent_models(percentage=0.8, top_n=10, num_models=10):
        """we will select last top_n model with priorty of 0.8 to newer models and 0.2 to older models(exclude top_n models) return num models list(list can contain duplicates)"""
        models_info = ModelHelper.list_saved_models()
        if not models_info:
            return [None] * num_models
        
        if len(models_info) <= top_n:
            sampled_info = [random.choice(models_info) for _ in range(num_models)]
        else:
            recent_models = models_info[-top_n:]
            older_models = models_info[:-top_n]
            sampled_info = []
            for _ in range(num_models):
                if random.random() < percentage:
                    sampled_info.append(random.choice(recent_models))
                else:
                    sampled_info.append(random.choice(older_models))
        
        return [ModelHelper.load_model(m['name']) for m in sampled_info]


        
