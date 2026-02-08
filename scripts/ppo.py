import os
import sys
from utils.game_helper import GameHelper
from game.game import Game
from utils.configs import Config
from utils.model_helper import ModelHelper
from trainers.ppo_trainer import build_actor_critic_model
from trainers.ppo_trainer import PPOTrainer

def main():
    game = Game()
    game.reset()

    ppo_model = build_actor_critic_model(input_dim=Config.INPUT_FEATURES, hidden=Config.PPO_HIDDEN_SIZE)
    ppo_model.summary()

    print("\nStarting PPO training...")

    trainer = PPOTrainer(ppo_model)
    metrics = trainer.train_ppo(game, total_timesteps=Config.PPO_TOTAL_TIMESTEPS, 
                                n_steps=Config.PPO_N_STEPS, log_interval=1)

    print("\nTraining completed successfully!")
    print(f"Final metrics: {metrics}")

if __name__ == "__main__":
    main()
