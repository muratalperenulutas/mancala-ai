#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import numpy as np
from utils.game_helper import GameHelper  
from utils.replay_buffer import ReplayBuffer
from utils.configs import Config
import random
import tensorflow as tf
from game.game import Game
from utils.model_helper import ModelHelper
from trainers.dqn_trainer import DqnTrainer
from trainers.dqn_trainer import create_model, compile_model

# In[2]:


version = tf.__version__
print("TensorFlow version:", version)
python_version = sys.version
print("Python version:", python_version)

# In[3]:


game = Game(verbose=False)
replay = ReplayBuffer()

# In[4]:


model=create_model()
compile_model(model)

# In[5]:


trainer = DqnTrainer(model=model)
trainer.train_model(game=game, replay=replay, debug=True)

# In[6]:


from model_debug import debug_single_batch

test_states = []
test_legal_masks = []
game.initialize()

for _ in range(5):
    features = ModelHelper.build_features(game.get_board(), game.current_player, game.game_finish)
    normalized_state = ModelHelper.normalize_single_fixed(features)
    legal_actions = game.get_playable_pits()
    
    mask = np.zeros(6)
    for action in legal_actions:
        if 0 <= action <= 5: 
            mask[action] = 1
        elif 7 <= action <= 12:
            mask[action - 7] = 1
            
    test_states.append(normalized_state)
    test_legal_masks.append(mask)
    
    action = random.choice(legal_actions)
    game.play(action)
    
test_states = np.array(test_states)
test_legal_masks = np.array(test_legal_masks)

print("Debug output for a batch of 5 game states:")
debug_single_batch(model, test_states, test_legal_masks)
