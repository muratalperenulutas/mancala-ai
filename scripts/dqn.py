import numpy as np
import random
import tensorflow as tf

from utils.replay_buffer import ReplayBuffer
from game.game import Game
from utils.model_helper import ModelHelper
from trainers.dqn_trainer import DqnTrainer, create_model, compile_model
from utils.debug_helper import debug_single_batch

def main():
    game=Game()
    
    replay = ReplayBuffer()

    model = create_model()
    compile_model(model)

    trainer = DqnTrainer(model=model)
    trainer.train_model(game=game, replay=replay, debug=True)

    test_states = []
    test_legal_masks = []

    for _ in range(5):
        features = ModelHelper.build_features(game)
        normalized_state = ModelHelper.normalize_single_fixed(features)
        legal_actions = game.get_playable_pits(symmetry=True)
        
        mask = np.zeros(6)
        for action in legal_actions:
            if 0 <= action <= 5: 
                mask[action] = 1
                
        test_states.append(normalized_state)
        test_legal_masks.append(mask)
        
        if not game.game_finish:
            action = random.choice(legal_actions)
            game.play(action)
        else:
            break
        
    if test_states:
        test_states = np.array(test_states)
        test_legal_masks = np.array(test_legal_masks)

        print("Debug output for a batch of 5 game states:")
        debug_single_batch(model, test_states, test_legal_masks)

if __name__ == "__main__":
    main()
