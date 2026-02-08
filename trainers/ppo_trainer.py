import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense
from utils.configs import Config
from utils.model_helper import ModelHelper
from utils.game_helper import GameHelper


class PPOConfig:
    LR = 3e-4
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    CLIP_RATIO = 0.2
    ENTROPY_COEF = 0.01
    VALUE_COEF = 0.5
    MAX_GRAD_NORM = 0.5
    N_EPOCHS = 4
    BATCH_SIZE = 64
    N_STEPS = 2048
    LARGE_NEG = -1e9
    EPS = 1e-8


def build_actor_critic_model(input_dim=Config.INPUT_FEATURES, hidden=128):
    inp = Input(shape=(input_dim,), name="state")
    x = Dense(hidden, activation="relu", name="shared1")(inp)
    x = Dense(hidden, activation="relu", name="shared2")(x)
    logits = Dense(Config.NUM_ACTIONS, name="logits")(x)
    value = Dense(1, name="value")(x)
    return Model(inputs=inp, outputs=[logits, value])


def mask_logits_tf(logits, legal_mask):
    legal = tf.cast(legal_mask, logits.dtype)
    return logits * legal + (1.0 - legal) * PPOConfig.LARGE_NEG


def sample_action(masked_logits):
    probs = tf.nn.softmax(masked_logits, axis=-1)
    action = tf.squeeze(
        tf.random.categorical(tf.math.log(probs + PPOConfig.EPS), 1), axis=-1
    )
    return action, probs


def compute_log_probs(logits, legal_mask, actions):
    masked_logits = mask_logits_tf(logits, legal_mask)
    log_probs_all = tf.nn.log_softmax(masked_logits, axis=-1)
    actions_onehot = tf.one_hot(actions, depth=Config.NUM_ACTIONS)
    log_probs = tf.reduce_sum(actions_onehot * log_probs_all, axis=1)
    return log_probs


# GAE (Generalized Advantage Estimation)
def compute_gae(
    rewards, values, dones, last_value, gamma=PPOConfig.GAMMA, lam=PPOConfig.GAE_LAMBDA
):
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        if t == T - 1:
            next_value = last_value
        else:
            next_value = values[t + 1]
        delta = rewards[t] + gamma * next_value * (1.0 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1.0 - dones[t]) * gae
        advantages[t] = gae
    returns = advantages + values
    return advantages, returns


def normalize_advantage(adv):
    return (adv - np.mean(adv)) / (np.std(adv) + 1e-8)


class PPOTrainer:
    def __init__(self, model):
        self.model = model
        self.optimizer = tf.keras.optimizers.Adam(PPOConfig.LR)
        self.train_metrics = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "grad_norm": [],
            "clipfrac": [],
            "approx_kl": [],
        }

    @tf.function
    def ppo_update_step(
        self, states, legal_masks, actions, old_log_probs, advantages, returns
    ):
        """Single mini-batch PPO update"""
        with tf.GradientTape() as tape:
            logits, values = self.model(states, training=True)
            values = tf.squeeze(values, axis=1)

            new_log_probs = compute_log_probs(logits, legal_masks, actions)

            ratio = tf.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = (
                tf.clip_by_value(
                    ratio, 1.0 - PPOConfig.CLIP_RATIO, 1.0 + PPOConfig.CLIP_RATIO
                )
                * advantages
            )
            policy_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))

            value_loss = (
                tf.reduce_mean(tf.square(returns - values)) * PPOConfig.VALUE_COEF
            )

            masked_logits = mask_logits_tf(logits, legal_masks)
            probs = tf.nn.softmax(masked_logits, axis=-1)
            entropy = -tf.reduce_sum(
                probs * tf.math.log(probs + PPOConfig.EPS), axis=-1
            )
            entropy_loss = -PPOConfig.ENTROPY_COEF * tf.reduce_mean(entropy)

            total_loss = policy_loss + value_loss + entropy_loss

        grads = tape.gradient(total_loss, self.model.trainable_variables)
        grads, grad_norm_before_clip = tf.clip_by_global_norm(
            grads, PPOConfig.MAX_GRAD_NORM
        )
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        clipfrac = tf.reduce_mean(
            tf.cast(tf.abs(ratio - 1.0) > PPOConfig.CLIP_RATIO, tf.float32)
        )

        return {
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": tf.reduce_mean(entropy),
            "grad_norm": grad_norm_before_clip,
            "clipfrac": clipfrac,
            "approx_kl": tf.reduce_mean(old_log_probs - new_log_probs),
        }

    def collect_rollout(self, game, n_steps=PPOConfig.N_STEPS):
        states, legal_masks, actions, rewards, dones, values, log_probs = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        ep_rewards = []
        ep_lengths = []
        current_ep_reward = 0
        current_ep_length = 0

        game.reset()
        game.initialize()

        for step in range(n_steps):
            player = game.current_player
            features = ModelHelper.build_features(
                game.get_board(), player, game.game_finish
            )
            normalized_state = ModelHelper.normalize_single_fixed(features)

            legal_actions_indices = game.get_playable_pits()
            if player == 0:
                canonical_legal = [a for a in legal_actions_indices if 0 <= a <= 5]
            else:
                canonical_legal = [a - 7 for a in legal_actions_indices if 7 <= a <= 12]

            legal_mask = np.zeros(Config.NUM_ACTIONS, dtype=np.float32)
            legal_mask[canonical_legal] = 1.0

            s_np = np.expand_dims(normalized_state, axis=0).astype(np.float32)
            mask_np = np.expand_dims(legal_mask, axis=0).astype(np.float32)

            logits_np, value_np = self.model(s_np, training=False)
            logits_np = logits_np.numpy()[0]
            value_np = float(value_np.numpy()[0, 0])

            masked_logits_np = np.where(legal_mask, logits_np, PPOConfig.LARGE_NEG)
            action_tf, probs_tf = sample_action(
                tf.constant([masked_logits_np], dtype=tf.float32)
            )
            action = int(action_tf.numpy()[0])
            log_prob = float(tf.math.log(probs_tf[0, action] + PPOConfig.EPS).numpy())

            board_index = GameHelper.action_to_board_index(action, player)
            game.play(board_index)

            reward = GameHelper.get_move_score(game)
            done = game.game_finish

            states.append(normalized_state)
            legal_masks.append(legal_mask)
            actions.append(action)
            rewards.append(reward)
            dones.append(float(done))
            values.append(value_np)
            log_probs.append(log_prob)

            current_ep_reward += reward
            current_ep_length += 1

            if done:
                score_p0 = GameHelper.get_final_score(game=game, player=0)
                score_p1 = GameHelper.get_final_score(game=game, player=1)
                if score_p0 > score_p1:
                    final_reward = 1.0 if player == 0 else -1.0
                elif score_p0 < score_p1:
                    final_reward = -1.0 if player == 0 else 1.0
                else:
                    final_reward = 0.0

                K = Config.K
                gamma = PPOConfig.GAMMA
                n_episode = current_ep_length
                for i in range(max(0, len(rewards) - n_episode), len(rewards)):
                    steps_from_end = len(rewards) - 1 - i
                    if steps_from_end < K:
                        rewards[i] += final_reward * (gamma**steps_from_end)

                ep_rewards.append(current_ep_reward)
                ep_lengths.append(current_ep_length)
                current_ep_reward = 0
                current_ep_length = 0

                game.reset()
                game.initialize()

        player = game.current_player
        features = ModelHelper.build_features(
            game.get_board(), player, game.game_finish
        )
        normalized_state = ModelHelper.normalize_single_fixed(features)
        s_np = np.expand_dims(normalized_state, axis=0).astype(np.float32)
        _, last_value_np = self.model(s_np, training=False)
        last_value = float(last_value_np.numpy()[0, 0])

        values_arr = np.array(values, dtype=np.float32)
        advantages, returns = compute_gae(rewards, values_arr, dones, last_value)
        advantages = normalize_advantage(advantages)

        return {
            "states": np.array(states, dtype=np.float32),
            "legal_masks": np.array(legal_masks, dtype=np.float32),
            "actions": np.array(actions, dtype=np.int32),
            "old_log_probs": np.array(log_probs, dtype=np.float32),
            "advantages": advantages,
            "returns": returns,
            "ep_rewards": ep_rewards,
            "ep_lengths": ep_lengths,
        }

    def ppo_update(self, rollout_data):
        states = rollout_data["states"]
        legal_masks = rollout_data["legal_masks"]
        actions = rollout_data["actions"]
        old_log_probs = rollout_data["old_log_probs"]
        advantages = rollout_data["advantages"]
        returns = rollout_data["returns"]

        dataset_size = len(states)
        indices = np.arange(dataset_size)

        all_metrics = []
        for epoch in range(PPOConfig.N_EPOCHS):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, PPOConfig.BATCH_SIZE):
                end = min(start + PPOConfig.BATCH_SIZE, dataset_size)
                batch_idx = indices[start:end]

                batch_states = tf.convert_to_tensor(states[batch_idx])
                batch_masks = tf.convert_to_tensor(legal_masks[batch_idx])
                batch_actions = tf.convert_to_tensor(actions[batch_idx])
                batch_old_log_probs = tf.convert_to_tensor(old_log_probs[batch_idx])
                batch_advantages = tf.convert_to_tensor(advantages[batch_idx])
                batch_returns = tf.convert_to_tensor(returns[batch_idx])

                metrics = self.ppo_update_step(
                    batch_states,
                    batch_masks,
                    batch_actions,
                    batch_old_log_probs,
                    batch_advantages,
                    batch_returns,
                )
                all_metrics.append({k: float(v.numpy()) for k, v in metrics.items()})

        avg_metrics = {
            k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0].keys()
        }

        for k, v in avg_metrics.items():
            if k in self.train_metrics:
                self.train_metrics[k].append(v)

        return avg_metrics

    def compute_illegal_rates(self, states, legal_masks):
        logits_np = self.model(states, training=False)[0].numpy()
        arg_before = np.argmax(logits_np, axis=1)
        masked = np.where(legal_masks, logits_np, -1e9)
        arg_after = np.argmax(masked, axis=1)
        illegal_before = 1.0 - np.mean(
            [legal_masks[i, arg_before[i]] for i in range(len(arg_before))]
        )
        illegal_after = 1.0 - np.mean(
            [legal_masks[i, arg_after[i]] for i in range(len(arg_after))]
        )
        return illegal_before, illegal_after

    def train_ppo(
        self, game, total_timesteps=200_000, n_steps=PPOConfig.N_STEPS, log_interval=1
    ):
        timesteps = 0
        iteration = 0

        print(f"\nStarting PPO Training...")
        print(f"Total timesteps: {total_timesteps}")
        print(f"Rollout steps: {n_steps}")
        print(f"=" * 80)

        while timesteps < total_timesteps:
            iteration += 1

            rollout_data = self.collect_rollout(game, n_steps=n_steps)
            timesteps += len(rollout_data["states"])

            il_before, il_after = self.compute_illegal_rates(
                rollout_data["states"], rollout_data["legal_masks"]
            )

            metrics = self.ppo_update(rollout_data)

            if iteration % log_interval == 0:
                avg_ep_reward = (
                    np.mean(rollout_data["ep_rewards"])
                    if rollout_data["ep_rewards"]
                    else 0.0
                )
                avg_ep_length = (
                    np.mean(rollout_data["ep_lengths"])
                    if rollout_data["ep_lengths"]
                    else 0.0
                )

                print(f"\n{'=' * 80}")
                print(
                    f"Iteration: {iteration} | Timesteps: {timesteps}/{total_timesteps}"
                )
                print(f"{'=' * 80}")
                print(f"Episode Stats:")
                print(f"  Avg Episode Reward: {avg_ep_reward:.3f}")
                print(f"  Avg Episode Length: {avg_ep_length:.1f}")
                print(f"  Episodes Completed: {len(rollout_data['ep_rewards'])}")
                print(f"\nLoss Metrics:")
                print(f"  Policy Loss:  {metrics['policy_loss']:.4f}")
                print(f"  Value Loss:   {metrics['value_loss']:.4f}")
                print(f"  Entropy:      {metrics['entropy']:.4f}")
                print(f"\nTraining Metrics:")
                print(f"  Grad Norm:    {metrics['grad_norm']:.3f}")
                print(f"  Clip Fraction: {metrics['clipfrac']:.3f}")
                print(f"  Approx KL:    {metrics['approx_kl']:.4f}")
                print(f"\nAction Masking:")
                print(f"  Illegal Before Mask: {il_before:.3f}")
                print(f"  Illegal After Mask:  {il_after:.3f}")

            if (iteration * n_steps) % Config.SAVE_FREQUENCY == 0:
                ModelHelper.save_model(self.model, t=iteration)
                print(f"\nModel saved at iteration {iteration}")

        print(f"\n{'=' * 80}")
        print("PPO Training Complete!")
        print(f"{'=' * 80}\n")
        return self.train_metrics
