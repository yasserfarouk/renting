This repository accompanies a [paper](https://rlj.cs.umass.edu/2024/papers/Paper268.html) published at the Reinforcement Learning Conference 2024:

**Towards General Negotiation Strategies with End-to-End Reinforcement Learning**\
[B.M. Renting](mailto:b.m.renting@liacs.leidenuniv.nl), [T.M. Moerland](t.m.moerland@liacs.leidenuniv.nl), [H.H. Hoos](hh@aim.rwth-aachen.de), [C.M. Jonker](c.m.jonker@tudelft.nl)


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
To train and test GNN based negotiation agents use the `ppo.py` and `evaluate.py`.

The default arguments are set to match the settings in the paper. To change/view the command line arguments run `python ppo.py --help`.

The figures from the paper can be reproduced using the `paper_results.py` script. A number of tests is defined in the `TESTS` variable in the script. Each of these tests can be run as follows: `python paper_results.py --test_num $TEST_LIST_INDEX`. The trained models are included in this repository. Results will be saved in the `analysis/data` and `analysis/figures` directories.

### From NegmasRL
I assume you use the "RLC2024" branch of the repository.

Scenarios
In "ppo.py" (for training) line 51, the scenario directory to be used is set. In line 216-218, a random scenario is created and saved to that location. You might be able to change the directory of one of the scenarios in the training set to this directory so that it is used this training iteration.
In "paper_results.py" (for testing) line 25-68 a list of test cases is set up. You could delete all of them except for the first one. In line 30, a fixed scenario path is set, which is used again in line 119-121 like described before. You can do the same here by renaming one of your test scenarios to this directory between episodes.

JSON format
I do not have a description for this format, which I created myself. It is a minimal format only for discrete and linear additive scenarios. I think the example files speak for themselves here, better than I can put in writing here. The entire scenario class can be found in "environment/scenario.py" for reference.


What might also be good to know is that there is a wrapper to use GeniusWeb agents in my environment in "environment/agents/geniusweb/wrapper.py".

I hope this helps. What agent do you intend on using as a benchmark? Are they GeniusWeb or NegMAS?
Let me know if you have further questions. If it is not too much work, I might also be able to modify some code if that helps.
