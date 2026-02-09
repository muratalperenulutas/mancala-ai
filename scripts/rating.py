from utils.model_helper import ModelHelper
from utils.game_helper import GameHelper
import numpy as np

class Rating:
    def __init__(self, k_factor=32):
        self.k_factor = k_factor
        
    def calculate_new_elo(self, old_elo, opponent_elo, score):
        expected_score = 1.0 / (1.0 + 10**((opponent_elo - old_elo) / 400.0))
        new_elo = old_elo + self.k_factor * (score - expected_score)
        return new_elo
    
    def evaluate_model(self, new_model, num_games=50):
        models_info = ModelHelper.list_saved_models()
        if not models_info:
            return 1200.0
        
        sorted_models = sorted(models_info, key=lambda x: x.get('elo', 1200), reverse=True)
        top_models_info = sorted_models[:5]
        
        current_elo = 1200.0
        if top_models_info:
            current_elo = np.mean([m.get('elo', 1200) for m in top_models_info])

        for model_info in top_models_info:
            opp_model = ModelHelper.load_model(model_info['name'])
            opp_elo = model_info.get('elo', 1200)
            
            wins, draws, losses = GameHelper.play_between_models(new_model, opp_model, num_games=num_games)
            
            total_games = wins + draws + losses
            if total_games == 0: continue
            
            actual_score = (wins * 1.0 + draws * 0.5) / total_games
            current_elo = self.calculate_new_elo(current_elo, opp_elo, actual_score)
            
        return current_elo

    def check_performance_drop(self, new_elo, threshold=200):
        models_info = ModelHelper.list_saved_models()
        if not models_info:
            return
        
        max_elo = max([m.get('elo', 1200) for m in models_info])
        
        if new_elo < max_elo - threshold:
           raise ValueError(f"Performance drop detected! New Elo: {new_elo:.2f}, Max Elo: {max_elo:.2f}")
