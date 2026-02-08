# Mancala AI

A Reinforcement Learning project using DQN and PPO to play Mancala.

## Project Structure
- `game/`: Core game logic (pits, board, rules).
- `models/`: Saved model checkpoints and weights.
- `trainers/`: RL training algorithms (DQN, PPO).
- `utils/`: Common utilities, configs, and helpers.
- `tests/`: Unit tests and AI performance benchmarks.
- `scripts/`: Entry point scripts for training and debugging.
- `.devcontainer/`: Configuration for VS Code Dev Containers.

## Setup

### Using Dev Containers (Recommended)
If you have VS Code and Docker installed, simply open this folder in VS Code and click "Reopen in Container". All dependencies and tools will be automatically installed.

### Manual Setup
1. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Testing & Benchmarking
Run all tests using:
```bash
pytest
```
