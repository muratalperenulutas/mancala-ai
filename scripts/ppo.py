from utils.configs import Config
from trainers.ppo_trainer import build_actor_critic_model
from trainers.ppo_trainer import PPOTrainer

def main():
    ppo_model = build_actor_critic_model(input_dim=Config.INPUT_FEATURES)
    ppo_model.summary()

    print("\nStarting PPO training...")

    trainer = PPOTrainer(ppo_model)
    metrics = trainer.train_ppo()

    print("\nTraining completed successfully!")
    print(f"Final metrics: {metrics}")

if __name__ == "__main__":
    main()
