This repository accompanies a paper published at the Reinforcement Learning Conference 2024:

**Towards General Negotiation Strategies with End-to-End Reinforcement Learning**\
B.M. Renting, T.M. Moerland, H.H. Hoos, C.M. Jonker


## Installation
Code was written for Python 3.10 and CUDA 12.2. To install, we recommend the usage of a Python environment:
```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install --upgrade setuptools wheel
pip install -r requirements.txt
```

## Usage
To train and test GNN based negotiation agents use the `train.py` and `evaluate.py`.

The default arguments are set to match the settings in the paper. To change/view the command line arguments run `python train.py --help`.