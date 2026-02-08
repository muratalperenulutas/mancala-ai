import os
from utils.model_helper import ModelHelper
from utils.game_helper import GameHelper
from game.game import Game

def main():
    model_path = 'models/mancala_model_iter_25.keras'
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please ensure you have a trained model in the models/ directory.")
        return

    model = ModelHelper.load_model(model_path)
    model.summary()

    game = Game()

    GameHelper.play_against_model(model=model, game=game)

if __name__ == "__main__":
    main()
