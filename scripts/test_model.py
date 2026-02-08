#!/usr/bin/env python
# coding: utf-8

# In[1]:


from utils.model_helper import ModelHelper
from utils.game_helper import GameHelper
from game.game import Game

# In[2]:


model=ModelHelper.load_model('mancala_model_iter_1876.keras')
model.summary()

# In[3]:


game=Game()
game.initialize()

# In[ ]:


GameHelper.play_against_model(model=model, game=game)
