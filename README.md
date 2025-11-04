
# hangman-HMM-RL

Hybrid HMM + RL Hangman solver — trains an agent from scratch on a 50,000-word corpus to learn probabilistic word structure and optimal guessing strategies.

This repository implements an experimental hangman-solving agent that combines a Hidden Markov Model (HMM) to learn probabilistic word structure with Reinforcement Learning (RL) to learn an optimal letter-guessing policy. The result is an agent that uses learned language structure and lookahead decision-making to make efficient guesses, reducing the number of mistakes required to solve words.

This README provides an overview of the project, design and algorithmic details, usage examples, and pointers for setup and experimentation. Detailed setup steps are in SETUP.md (included in this repo).

Contents
-> Project overview


-> Typical workflows (train / evaluate / play)
-> Configuration and hyperparameters
-> Dataset format
-> Expected outputs and how to read results
-> Project structure


Project overview
This project explores a hybrid approach:
-> HMM component to capture probabilistic sequential patterns in words (e.g., likely letter sequences, affinities between positions).
-> RL component (episodic, environment: hangman) to learn a policy that selects letters to guess to minimize expected mistakes / maximize solved words.

The HMM can provide a prior or belief state over candidate words or letter distributions; the RL agent uses that information (and the observed masked word plus history of guesses) to pick the next letter.

Key ideas and architecture
- Preprocessing: Build a vocabulary from a word corpus (default: 50,000 words or a curated wordlist). Normalize and filter by length as needed.
- HMM training: Learn emission/transition statistics across letter positions to model likely letter sequences or class states. The HMM can be used to produce position-wise posterior distributions over letters given partially observed word patterns (blank/known letters).
- Environment: Episode = solving one word. State = (mask pattern, guesses made, HMM-derived belief vector(s)). Action = choose an unguessed letter. Reward = positive for solving or per-step negative for mistakes (tunable).
- RL agent: Policy network (e.g., small feedforward / LSTM) that consumes the state and outputs a distribution over remaining letters. Training algorithm can be policy gradient (REINFORCE), actor-critic, or Q-learning adapted to discrete actions.
- Integration: HMM outputs are concatenated with other state features (mask, guessed/unguessed mask, letter frequencies) to form the agent input.

What’s included
- Core algorithm description and examples (high-level)
- Scripts / entry points (suggested names — see SETUP.md)
  - training script (train.py or scripts/train.sh)
  - evaluation script (evaluate.py)
  - play / interactive script to let a human play against the agent
- Example configuration and hyperparameters (configs/)
- Example dataset instructions and tokenizer
- Logging and results: metrics, per-episode logs, saved agent checkpoints, and plots (recommended output directories)
If your clone currently lacks particular scripts referenced below, use the SETUP.md guidance to create or adapt scripts to match your project layout.

Quickstart (recommended)
1. Clone the repository
   git clone https://github.com/AHaveeshKumar/hangman-HMM-RL.git
   cd hangman-HMM-RL

2. Follow the Setup instructions in SETUP.md (recommended)
   - Create a virtual environment
   - Install dependencies (pip install -r requirements.txt)
   - (Optional) Install torch with CUDA if you have a GPU

3. Prepare / download the dataset
   - Provide a wordlist (plain text, one word per line).
   - By default, the code expects a words file (e.g., data/words.txt). See the Dataset section below.

4. Train the HMM (if using a separate HMM training step)
   - Example: python scripts/train_hmm.py --words data/words.txt --out models/hmm.pkl
   - Output: saved HMM parameters and serialized model.

5. Train the RL agent
   - Example: python scripts/train_agent.py --config configs/agent.yaml --hmm models/hmm.pkl --out models/rl_agent.pt
   - Output: checkpoints, training logs, evaluation snapshots.

6. Evaluate agent performance
   - Example: python scripts/evaluate.py --agent models/rl_agent.pt --words data/test_words.txt --metrics results/eval.json

7. Play interactively
   - Example: python scripts/play.py --agent models/rl_agent.pt
   - Allows a human to select a word and watch the agent guess, or vice versa.

Typical workflows
- Experiment: change hyperparameters in configs/*.yaml, run train_agent.py, compare logs under runs/ or results/.
- Ablation: disable HMM inputs and train RL-only agent to quantify HMM contribution.
- Curriculum training: start on shorter words and progressively increase length.

Configuration and hyperparameters
- HMM
  - number_of_states: integer
  - smoothing / priors: Laplace or other smoothing options
  - training_epochs (if using EM / Baum-Welch)
- RL agent
  - algorithm: REINFORCE / A2C / PPO / DQN (algorithmic choice)
  - learning_rate
  - gamma (discount factor)
  - batch_size / episodes_per_update
  - exploration strategy (epsilon-greedy or entropy regularization)
  - reward shaping: per-step penalty for wrong guess, big reward for solving word
- Logging
  - checkpoint frequency
  - evaluation frequency
  - metrics to log: average guesses-to-solve, success rate, mistake count, cumulative reward

Dataset format
- Expectation: a plain text word-list, one word per line, normalized (lowercase, ASCII or UTF-8).
- Filtering recommendations:
  - Remove non-alphabetic characters (or adjust alphabet to include hyphen/etc.)
  - Filter by length ranges for experiments (e.g., 3–12 letters)
- Example:
  data/words.txt
  ----
  apple
  banana
  orange
  ----

Expected outputs and how to interpret them
- Checkpoints:
  - models/hmm.pkl: pickled HMM object
  - models/rl_agent.pt or .pth: saved PyTorch/TensorFlow checkpoint
- Logs:
  - training.log: per-episode scalar values of reward, loss, and metrics
  - results/eval.json: evaluation metrics summarized
- Typical metrics:
  - Solve rate: fraction of words solved within allowed mistake budget
  - Average mistakes per word
  - Average guesses-to-solve
  - Total reward
- Visualization:
  - Learning curves (reward / success rate vs episodes)
  - Confusion analyses: which word lengths or letter patterns cause errors

Project structure (recommended / example)
-dataset1/
-dataset2/
- notebook/                # trained models and checkpoints

- README.md
- SETUP.md
- requirements.txt

Notes and caveats
- This repo is experimental by nature. Hyperparameters, reward shaping, and environment design influence agent behavior substantially — treat the code as a research platform rather than a production-ready solver.
- HMMs can be trained offline and used as a fixed prior; you can also experiment with co-training where HMMs are updated online (advanced).
- If you use neural function approximators and GPUs, ensure correct device setup (CUDA available and torch installed with appropriate CUDA version).


- Please open issues for bugs, feature requests, or experiments you want to share.  (:
- 
  



