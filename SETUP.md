# Setup and Installation — hangman-HMM-RL

This document provides step-by-step instructions to set up a reproducible environment for developing, training, and evaluating the hangman-HMM-RL project. It covers both minimal (CPU) and optional GPU configurations, as well as recommended file/command conventions for running the code.

Prerequisites
- Git
- Python 3.8, 3.9, 3.10 or 3.11 (3.8+ recommended)
- At least 4 GB RAM for small experiments; more for larger batches or GPU training.
- Optional: A CUDA-capable GPU if you plan to train large neural models.

1) Clone the repository
git clone https://github.com/AHaveeshKumar/hangman-HMM-RL.git
cd hangman-HMM-RL

2) Create an isolated Python environment (recommended)
Option A — using python venv (cross-platform)
python3 -m venv .venv
# macOS 
source .venv/bin/activate

Option B — using conda
conda create -n hangman-hmm-rl python=3.10 -y
conda activate hangman-hmm-rl

3) Install required Python packages
- If the repo contains a requirements.txt use:
pip install -r requirements.txt

- If there is no requirements.txt, install a recommended minimal set:
pip install numpy scipy scikit-learn tqdm pyyaml

- For RL training and neural nets, install a deep learning framework. Two common options:
  - PyTorch (recommended for flexibility)

  - TensorFlow:
    pip install tensorflow

- Optional useful packages:
pip install matplotlib seaborn pandas tensorboard

4) Verify installation
python -c "import numpy, torch; print('numpy', numpy.__version__); print('torch', torch.__version__)"
Note: If you don't have torch, omit the torch import in the test.

5) Prepare the dataset (words list)
The project expects a line-separated wordlist. Place the file at data/words.txt (recommended).
- Example format:
  apple
  banana
  orange

If you want to build a 50,000-word corpus:
- Obtain a dictionary or word frequency list (e.g., from wordnik, word frequency datasets, or wordlists from word game communities).
- Preprocess:
  - Lowercase
  - Remove whitespace and punctuation as appropriate
  - (Optional) filter by length: keep words with length between min_len and max_len for your experiments

6) HMM training (if your pipeline separates HMM training)
- Example placeholder command:
python scripts/train_hmm.py --words data/words.txt --out models/hmm.pkl
- This script should implement learning of emission/transition probabilities and save a serialized HMM for later inference.

7) RL agent training
- Example placeholder command:
python scripts/train_agent.py --config configs/agent.yaml --hmm models/hmm.pkl --out models/rl_agent.pt
- Common config options:
  - episodes: number of training episodes
  - batch_size: episodes per policy update
  - lr: learning rate
  - gamma: discount factor
  - max_mistakes: allowed wrong guesses before episode termination

8) Evaluation
- Example placeholder command:
python scripts/evaluate.py --agent models/rl_agent.pt --words data/test_words.txt --out results/eval.json

9) Interactive play
- Example placeholder command:
python scripts/play.py --agent models/rl_agent.pt
- Play modes:
  - Agent guesses a human-selected word
  - Human guesses an agent-selected word (for evaluation or demonstrations)

10) Recording experiments and reproducibility
- Save configuration files (configs/*.yaml) with a timestamped folder under runs/.
- Save random seeds and package versions:
  - Python: python -V
  - pip freeze > runs/my-experiment/requirements-freeze.txt
  - Save the config and the commit hash:
    git rev-parse HEAD > runs/my-experiment/commit.txt

11) Troubleshooting
- Missing script errors:
  If the repo lacks any referenced scripts (train_hmm.py, train_agent.py, evaluate.py, play.py), create them using the project structure or ask for example implementations (I can generate starter scripts).
- GPU not found:
  Ensure CUDA drivers and correct torch build are installed. Verify with:
  python -c "import torch; print(torch.cuda.is_available())"
- Memory / OOM:
  Reduce batch sizes, smaller models, or use CPU training if necessary.
