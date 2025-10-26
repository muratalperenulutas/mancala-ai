import sys
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from game_helper import GameHelper
from model_helper import ModelHelper
from game.game import Game
from configs import Config


class Model:
    @staticmethod
    def create_model(input_shape=(Config.INPUT_FEATURES,), num_actions=Config.NUM_ACTIONS, hidden_layers=Config.HIDDEN_LAYERS):
        model = Sequential()
        model.add(Dense(hidden_layers[0], activation='relu', input_shape=input_shape))
        for units in hidden_layers[1:]:
            model.add(Dense(units, activation='relu'))
        model.add(Dense(num_actions, activation='linear'))
        return model

    @staticmethod
    def compile_model(model):
        optimizer = Adam(learning_rate=Config.LEARNING_RATE)
        model.compile(optimizer=optimizer,
                      loss='mse',
                      metrics=['accuracy'])
        model.summary()

    @staticmethod
    def apply_final_reward(episode, final_reward, K=Config.K, gamma=Config.GAMMA):
        n = len(episode)
        for i in range(n):
            s, a, r, ns, done, player, la, lna = episode[i]
            steps_from_end = n - 1 - i
            if steps_from_end < K:
                r += final_reward * (gamma ** steps_from_end)
            episode[i] = (s, a, r, ns, done, player, la, lna)
        return episode
    

    @staticmethod
    #to do : use target network for stability
    def train_step_on_batch(model, replay_buffer, batch_size=Config.BATCH_SIZE, gamma=Config.GAMMA):
        if len(replay_buffer) < batch_size:
            return
        batch = replay_buffer.sample(batch_size)
        states, actions, rewards, next_states, dones, legal_next_actions_list = zip(*batch)
        states = np.array(states)
        next_states = np.array(next_states)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)

        target_qs = model.predict(next_states, verbose=0)
        max_next_qs = np.array([
            np.max(target_qs[i][legal_next_actions_list[i]]) if legal_next_actions_list[i] else 0
            for i in range(len(batch))
        ])
        targets = rewards + (1 - dones) * gamma * max_next_qs

        q_values = model.predict(states, verbose=0)
        for i, action in enumerate(actions):
            q_values[i][action] = targets[i]

        model.train_on_batch(states, q_values)  

    @staticmethod
    #to do : use target network for stability
    def train_step_fit(model, replay_buffer, batch_size=Config.BATCH_SIZE, gamma=Config.GAMMA):
        if len(replay_buffer) < batch_size:
            return
        batch = replay_buffer.sample(batch_size)
        states, actions, rewards, next_states, dones, legal_next_actions_list = zip(*batch)
        states = np.array(states)
        next_states = np.array(next_states)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)

        target_qs = model.predict(next_states, verbose=0)
        max_next_qs = np.array([
            np.max(target_qs[i][legal_next_actions_list[i]]) if legal_next_actions_list[i] else 0
            for i in range(len(batch))
        ])
        targets = rewards + (1 - dones) * gamma * max_next_qs

        q_values = model.predict(states, verbose=0)
        for i, action in enumerate(actions):
            q_values[i][action] = targets[i]

        model.fit(states, q_values, epochs=1, verbose=0)    

    @staticmethod
    def train_model(model, game: Game, replay, game_count=Config.GAME_COUNT, input_features_size=Config.INPUT_FEATURES,
                    epsilon_start=Config.EPSILON_START, epsilon_end=Config.EPSILON_END, epsilon_decay_steps=Config.EPSILON_DECAY_STEPS,
                    gamma=Config.GAMMA, K=Config.K, debug=False):
        for t in range(game_count):
            game.initialize()
            if debug or t % 10 == 0:
                print(f"Game {t+1} started")
            episode_p0 = []
            episode_p1 = []

            while True:
                player = game.current_player

                features = ModelHelper.build_features_from_state(game.get_board(), player, game.game_finish)

                normalized_state = ModelHelper.normalize_single_fixed(features)

                if debug:
                    print("raw state:", features)
                    print("normalized:", normalized_state)

                state_input = np.array(normalized_state).reshape(1, input_features_size)

                q_values = model.predict(state_input, verbose=0)[0]

                if debug:
                    print("q_values:", q_values)

                epsilon = max(epsilon_end, epsilon_start - (epsilon_start - epsilon_end) * t / epsilon_decay_steps)
                legal_actions = game.get_playable_pits()

                if len(legal_actions) > 0 and max(legal_actions) > 6:
                    if player == 0:
                        canonical_legal = [a for a in legal_actions if 0 <= a <= 5]
                    else:
                        canonical_legal = [a - 7 for a in legal_actions if 7 <= a <= 12]
                    legal_actions = canonical_legal

                if not legal_actions:
                    break    

                if random.random() < epsilon:
                    action = random.choice(legal_actions) 

                else:
                    best_idx = int(np.argmax(q_values[legal_actions]))
                    action = legal_actions[best_idx]

                board_index = GameHelper.action_to_board_index(action, game.current_player)
                game.play(board_index)

                reward = GameHelper.get_move_score(game)

                next_features = ModelHelper.build_features_from_state(game.get_board(), player, game.game_finish)
                normalized_next_state = ModelHelper.normalize_single_fixed(next_features)

                legal_next_actions = game.get_playable_pits()
                done = game.game_finish

                transition = (normalized_state, action, reward, normalized_next_state, done, player, legal_actions, legal_next_actions)
                if player == 1:
                    episode_p0.append(transition)
                else:
                    episode_p1.append(transition)

                if done:
                    if debug:
                        print("Game finished")
                        print("final raw state:", next_features)
                    score_p0 = GameHelper.get_final_score(game=game, player=0)
                    score_p1 = GameHelper.get_final_score(game=game, player=1)
                    if score_p0 > score_p1:
                        final_reward_p0 = 1
                        final_reward_p1 = -1
                    elif score_p0 < score_p1:
                        final_reward_p0 = -1
                        final_reward_p1 = 1
                    else:
                        final_reward_p0 = 0
                        final_reward_p1 = 0

                    episode_p0 = Model.apply_final_reward(episode_p0, final_reward_p0, K=K, gamma=gamma)
                    episode_p1 = Model.apply_final_reward(episode_p1, final_reward_p1, K=K, gamma=gamma)

                    for transition in episode_p0 + episode_p1:
                        s_norm, a, r, ns_norm, d, pl, la, lna = transition
                        replay.push(s_norm, a, r, ns_norm, d, lna)

                    game.reset()
                    break
                
            if (t + 1) % Config.SAVE_FREQUENCY == 0:
                ModelHelper.save_model(model, t=t)  
      