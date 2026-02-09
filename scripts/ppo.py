from trainers.ppo_trainer import build_actor_critic_model
from trainers.ppo_trainer import PPOTrainer
from utils.model_helper import ModelHelper

def main():
    print("\nStarting PPO training...")

    trainer = PPOTrainer()
    trainer.train_with_rating_cycle(num_cycles=10)

    print("\nTraining cycles completed successfully!")

if __name__ == "__main__":
    main()
