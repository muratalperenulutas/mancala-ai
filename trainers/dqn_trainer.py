import sys
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from utils.game_helper import GameHelper
from utils.model_helper import ModelHelper
from game.game import Game
from utils.configs import Config, DQNConfig

def create_model(input_shape=(Config.INPUT_FEATURES,), num_actions=Config.NUM_ACTIONS, hidden_layers=Config.HIDDEN_LAYERS):
    model = Sequential()
    model.add(Dense(hidden_layers[0], activation='relu', input_shape=input_shape))
    for units in hidden_layers[1:]:
        model.add(Dense(units, activation='relu'))
    model.add(Dense(num_actions, activation='linear'))
    return model

def compile_model(model):
    optimizer = Adam(learning_rate=Config.LEARNING_RATE)
    model.compile(optimizer=optimizer,
                  loss='mse',
                  metrics=['accuracy'])
    model.summary()


class DqnTrainer:
    def __init__(self, model):
        self.model = model
        self.target_model = tf.keras.models.clone_model(model)
        self.target_model.set_weights(model.get_weights())
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE)

    def _get_epsilon(self, game_idx):
        return max(
            Config.EPSILON_END,
            Config.EPSILON_START - (Config.EPSILON_START - Config.EPSILON_END) * game_idx / Config.EPSILON_DECAY_STEPS
        )

    def _select_action(self, state_input, epsilon, legal_actions):
        if not legal_actions: return None
        if random.random() < epsilon:
            return random.choice(legal_actions)
        
        q_values = self.model.predict(state_input, verbose=0)[0]
        masked_q = np.full(Config.NUM_ACTIONS, -np.inf)
        masked_q[legal_actions] = q_values[legal_actions]
        return int(np.argmax(masked_q))

    def _perform_training_step(self, replay, step_count, gamma):
        if len(replay) < DQNConfig.BATCH_SIZE: return
        
        batch = replay.sample(DQNConfig.BATCH_SIZE)
        states, actions, rewards, next_states, dones, legal_next_actions = zip(*batch)

        states = np.array(states, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)
        actions = np.array(actions, dtype=np.int32)

        target_qs = self.target_model.predict(next_states, verbose=0)
        max_next_qs = np.zeros(DQNConfig.BATCH_SIZE, dtype=np.float32)
        
        for i in range(DQNConfig.BATCH_SIZE):
            if legal_next_actions[i]:
                masked = np.full(Config.NUM_ACTIONS, -np.inf)
                masked[legal_next_actions[i]] = target_qs[i][legal_next_actions[i]]
                max_next_qs[i] = np.max(masked)

        targets = rewards + (1.0 - dones) * gamma * max_next_qs

        with tf.GradientTape() as tape:
            q_pred = self.model(states, training=True)
            actions_onehot = tf.one_hot(actions, depth=Config.NUM_ACTIONS)
            q_taken = tf.reduce_sum(q_pred * actions_onehot, axis=1)
            loss = tf.reduce_mean(tf.square(targets - q_taken))

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        if step_count % 100 == 0:
            print(f"  [Train] Step {step_count}, Loss: {float(loss):.4f}")

    def _handle_episode_end(self, game, ep_p0, ep_p1, replay, K, gamma):
        score0 = GameHelper.get_final_score(game, 0)
        score1 = GameHelper.get_final_score(game, 1)
        
        res0 = 1 if score0 > score1 else (-1 if score0 < score1 else 0)
        res1 = -res0

        ep_p0 = self.apply_final_reward(ep_p0, res0, K, gamma)
        ep_p1 = self.apply_final_reward(ep_p1, res1, K, gamma)

        for transition in ep_p0 + ep_p1:
            s, a, r, ns, d, pl, la, lna = transition
            replay.push(s, a, r, ns, d, lna)

    def apply_final_reward(self, episode, final_reward, K=Config.K, gamma=Config.GAMMA):
        n = len(episode)
        for i in range(n):
            s, a, r, ns, done, player, la, lna = episode[i]
            steps_from_end = n - 1 - i
            if steps_from_end < K:
                r += final_reward * (gamma ** steps_from_end)
            episode[i] = (s, a, r, ns, done, player, la, lna)
        return episode

    def train_model(self, replay, game=Game(), game_count=Config.GAME_COUNT, gamma=Config.GAMMA, K=Config.K):
        step_count = 0
        for t in range(game_count):
            game.initialize()
            if t % 10 == 0: print(f"Game {t+1} started")
            
            episodes = {0: [], 1: []}
            epsilon = self._get_epsilon(t)

            while not game.game_finish:
                player = game.current_player
                state = ModelHelper.normalize_single_fixed(ModelHelper.build_features(game))
                legal = game.get_playable_pits(symmetry=True)
                
                action = self._select_action(np.array([state]), epsilon, legal)
                if action is None: break

                game.play(action, symmetry=True)
                step_count += 1
                
                next_state = ModelHelper.normalize_single_fixed(ModelHelper.build_features(game))
                reward = GameHelper.get_move_score(game)
                done = game.game_finish
                next_legal = game.get_playable_pits(symmetry=True)

                episodes[player].append((state, action, reward, next_state, done, player, legal, next_legal))

                if step_count % DQNConfig.TRAIN_EVERY == 0:
                    self._perform_training_step(replay, step_count, gamma)

                if step_count % DQNConfig.UPDATE_TARGET_EVERY == 0:
                    self.target_model.set_weights(self.model.get_weights())
                    print(f"  [Info] Target network synced at step {step_count}")

            self._handle_episode_end(game, episodes[0], episodes[1], replay, K, gamma)
            
            if (t + 1) % Config.SAVE_FREQUENCY == 0:
                ModelHelper.save_model(self.model, t=t)
        
        ModelHelper.save_model(self.model, t=game_count)
        print("Training finished.")
      