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
from utils.board_utils import BoardUtils
from game.game import Game
from utils.configs import Config

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

    def apply_final_reward(self,episode, final_reward, K=Config.K, gamma=Config.GAMMA):
        n = len(episode)
        for i in range(n):
            s, a, r, ns, done, player, la, lna = episode[i]
            steps_from_end = n - 1 - i
            if steps_from_end < K:
                r += final_reward * (gamma ** steps_from_end)
            episode[i] = (s, a, r, ns, done, player, la, lna)
        return episode
    
    def train_step_on_batch(self,target_model, replay_buffer, optimizer, 
                            batch_size=Config.BATCH_SIZE, gamma=Config.GAMMA):
        if len(replay_buffer) < batch_size:
            return None

        batch = replay_buffer.sample(batch_size)
        states, actions, rewards, next_states, dones, legal_next_actions_list = zip(*batch)

        states = np.array(states, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)
        actions = np.array(actions, dtype=np.int32)

        target_qs = target_model.predict(next_states, verbose=0)

        max_next_qs = np.zeros(batch_size, dtype=np.float32)
        for i in range(batch_size):
            if legal_next_actions_list[i]:
                masked = np.full(6, -np.inf)
                masked[legal_next_actions_list[i]] = target_qs[i][legal_next_actions_list[i]]
                max_next_qs[i] = np.max(masked)
            else:
                max_next_qs[i] = 0.0

        targets = rewards + (1.0 - dones) * gamma * max_next_qs

        with tf.GradientTape() as tape:
            q_pred = self.model(states, training=True)
            actions_onehot = tf.one_hot(actions, depth=6)
            q_taken = tf.reduce_sum(q_pred * actions_onehot, axis=1)
            loss = tf.reduce_mean(tf.square(targets - q_taken))

        grads = tape.gradient(loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return float(loss.numpy())

    def train_model(self, game: Game, replay, game_count=Config.GAME_COUNT, input_features_size=Config.INPUT_FEATURES,
                    epsilon_start=Config.EPSILON_START, epsilon_end=Config.EPSILON_END, epsilon_decay_steps=Config.EPSILON_DECAY_STEPS,
                    gamma=Config.GAMMA, K=Config.K, debug=False):
        
        target_model = tf.keras.models.clone_model(self.model)
        target_model.set_weights(self.model.get_weights())

        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

        BATCH_SIZE = 64
        TRAIN_EVERY = 4
        UPDATE_TARGET_EVERY = 1000
        step_count = 0


        for t in range(game_count):
            game.initialize()
            if debug or t % 10 == 0:
                print(f"Game {t+1} started")
            episode_p0 = []
            episode_p1 = []

            while True:
                player = game.current_player

                features = ModelHelper.build_features(game.get_board(), player, game.game_finish)

                normalized_state = ModelHelper.normalize_single_fixed(features)

                if debug:
                    print("raw state:", features)
                    print("normalized:", normalized_state)

                state_input = np.array(normalized_state).reshape(1, input_features_size)

                q_values = self.model.predict(state_input, verbose=0)[0]

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
                    masked_q = np.full(6, -np.inf)
                    masked_q[legal_actions] = q_values[legal_actions]
                    action = int(np.argmax(masked_q))

                board_index = BoardUtils.action_to_board_index(action, game.current_player)
                game.play(board_index)
                step_count += 1

                reward = GameHelper.get_move_score(game)

                next_player=game.current_player
                next_features = ModelHelper.build_features(game.get_board(), next_player, game.game_finish)
                normalized_next_state = ModelHelper.normalize_single_fixed(next_features)

                legal_next_actions_raw = game.get_playable_pits()
                done = game.game_finish

                if len(legal_next_actions_raw) > 0 and max(legal_next_actions_raw) > 6:
                    if next_player == 0:
                        legal_next_actions = [a for a in legal_next_actions_raw if 0 <= a <= 5]
                    else:
                        legal_next_actions = [a - 7 for a in legal_next_actions_raw if 7 <= a <= 12]
                else:
                    legal_next_actions = legal_next_actions_raw

                transition = (normalized_state, action, reward, normalized_next_state, done, player, legal_actions, legal_next_actions)
                if player == 0:
                    episode_p0.append(transition)
                else:
                    episode_p1.append(transition)

                if step_count % TRAIN_EVERY == 0 and len(replay) >= BATCH_SIZE:
                    batch = replay.sample(BATCH_SIZE)
                    states_b, actions_b, rewards_b, next_states_b, dones_b, legal_next_b = zip(*batch)

                    states_b = np.array(states_b, dtype=np.float32)
                    next_states_b = np.array(next_states_b, dtype=np.float32)
                    rewards_b = np.array(rewards_b, dtype=np.float32)
                    dones_b = np.array(dones_b, dtype=np.float32)
                    actions_b = np.array(actions_b, dtype=np.int32)

                    target_qs = target_model.predict(next_states_b, verbose=0)

                    max_next_qs = np.zeros(BATCH_SIZE, dtype=np.float32)
                    for i in range(BATCH_SIZE):
                        if legal_next_b[i]:
                            masked = np.full(6, -np.inf)
                            masked[legal_next_b[i]] = target_qs[i][legal_next_b[i]]
                            max_next_qs[i] = np.max(masked)
                        else:
                            max_next_qs[i] = 0.0

                    targets = rewards_b + (1.0 - dones_b) * gamma * max_next_qs

                    with tf.GradientTape() as tape:
                        q_pred = self.model(states_b, training=True)
                        actions_onehot = tf.one_hot(actions_b, depth=6)
                        q_taken = tf.reduce_sum(q_pred * actions_onehot, axis=1)
                        loss = tf.reduce_mean(tf.square(targets - q_taken))

                    grads = tape.gradient(loss, self.model.trainable_variables)
                    optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

                    if step_count % 100 == 0:
                        print(f"  [Train] Step {step_count}, Loss: {float(loss):.4f}, Epsilon: {epsilon:.3f}")

                if step_count % UPDATE_TARGET_EVERY == 0:
                    target_model.set_weights(self.model.get_weights())
                    print(f"  [Info] Target network synced at step {step_count}")    

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

                    episode_p0 = self.apply_final_reward(episode_p0, final_reward_p0, K=K, gamma=gamma)
                    episode_p1 = self.apply_final_reward(episode_p1, final_reward_p1, K=K, gamma=gamma)

                    for transition in episode_p0 + episode_p1:
                        s_norm, a, r, ns_norm, d, pl, la, lna = transition
                        replay.push(s_norm, a, r, ns_norm, d, lna)

                    game.reset()
                    break
                
            if (t + 1) % Config.SAVE_FREQUENCY == 0:
                ModelHelper.save_model(self.model, t=t)
      