# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy

from collections import defaultdict
from dataclasses import dataclass
from multiprocessing import Pool
import random

import numpy as np
import pandas as pd
import torch
import tyro
from tensordict import TensorDict

from environment.agents.geniusweb import AGENTS
from environment.agents.policy.PPO import HigaEtAl2, PureGNN2
from environment.negotiation import NegotiationEnvZoo

AGENT_MODULES = {
    "PureGNN2": PureGNN2,
    "HigaEtAl2": HigaEtAl2,
}

@dataclass
class Args:
    debug: bool = False


    deadline: int = 40
    module: str = "PureGNN2"
    use_opponent_encoding: bool = False
    opponent_sets: tuple = ("BASIC",) # ("ANL2022","ANL2023","BASIC")
    scenario: str = "environment/scenarios/fixed_utility"
    random_agent_order: bool = False
    gat_v2: bool = False
    add_self_loops: bool = True
    hidden_size: int = 256
    heads: int = 4
    gnn_layers: int = 4
    out_layers: int = 1
    episodes: int = 100

    seed: int = 1
    """seed of the experiment"""

def evaluate_agent(opponent, model_path, args):
    used_agents = [a for a in AGENTS if a.startswith(tuple(args.opponent_sets))]
    env_config = {
        "agents": [f"RL_{args.module}", opponent],
        "used_agents": used_agents,
        "scenario": args.scenario,
        "deadline": {"rounds": args.deadline, "ms": 10000},
        "random_agent_order": args.random_agent_order,
    }
    env = NegotiationEnvZoo(env_config)
    agent: PureGNN2 = AGENT_MODULES[args.module](len(used_agents), args).to("cpu")
    agent.load_state_dict(torch.load(model_path))
    agent.train(False)


    log_metrics = defaultdict(lambda: .0)
    for episode in range(1, args.episodes + 1):
        next_obs = env.reset(seed=args.seed+episode)[0]
        terminations = {f"RL_{args.module}": False}

        while not terminations[f"RL_{args.module}"]:
            next_obs = {k: torch.from_numpy(v[np.newaxis, ...]) for k, v in next_obs[f"RL_{args.module}"].items() if k != "opponent_encoding"}
            next_obs = TensorDict(next_obs, batch_size=1)
            agent.action_nvec = tuple(env.action_space(f"RL_{args.module}").nvec)
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(next_obs)

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, _, terminations, _, infos = env.step({f"RL_{args.module}": action.numpy()[0]})

        info = infos[f"RL_{args.module}"]
        log_metrics["my_utility"] += info["utility_all_agents"][f"RL_{args.module}"]
        log_metrics["opp_utility"] += info["utility_all_agents"][opponent]
        log_metrics["rounds_played"] += info["rounds_played"]
        log_metrics["self_accepted"] += info["self_accepted"]
        log_metrics["found_agreement"] += info["found_agreement"]

    log_metrics = {k: v/episode for k, v in log_metrics.items()}

    return log_metrics, opponent

def main():
    args = tyro.cli(Args)
    used_agents = [a for a in AGENTS if a.startswith(tuple(args.opponent_sets))]
    model_paths_basic_random = [
        "models/PureGNN2_24-03-06_21:49:04_e4fedf93-41c7-4376-ad74-69980d8c8f4b",
        "models/PureGNN2_24-03-06_21:49:13_bea8fb17-8e2b-4f61-a59f-0cefdc689b1a",
        "models/PureGNN2_24-03-06_21:49:18_a2db0347-4bee-48d2-88ed-847a2c7029d2",
        "models/PureGNN2_24-03-06_21:49:38_78edda4c-a100-4287-b08b-d5f5dbc30668",
        "models/PureGNN2_24-03-06_21:50:01_8ac64625-ba00-485d-9658-665e4c67d2c5",
        "models/PureGNN2_24-03-06_21:50:01_13051a93-a05e-4022-b78c-bb3e6a356d5a",
        "models/PureGNN2_24-03-06_21:50:04_383fb7d1-16e2-457c-b6c7-6a17a6766a47",
        "models/PureGNN2_24-03-06_21:50:12_015efd40-3679-4a27-9a94-edc396588e8e",
        "models/PureGNN2_24-03-06_21:50:20_89a2c237-4a22-4666-bdc2-d17b469a012d",
        "models/PureGNN2_24-03-06_21:50:25_43b1eb5f-4c51-4218-9814-b2471af11b70",
    ]
    if args.debug:
        print(evaluate_agent(used_agents[0], model_paths_basic_random[0], args))
    else:
        iterables = [list(range(len(model_paths_basic_random))), sorted(used_agents)]
        index = pd.MultiIndex.from_product(iterables, names=["model", "opponent"])
        data = pd.DataFrame(columns=["my_utility", "opp_utility","rounds_played", "self_accepted", "found_agreement"], index=index)

        for i, model_path in enumerate(model_paths_basic_random):
            results = []
            for opponent in used_agents:
                results.append(evaluate_agent(opponent, model_path, args))
            # with Pool(len(used_agents)) as pool:
            #     results = pool.starmap(evaluate_agent, [(opponent, model_path, args) for opponent in used_agents])
            for result in results:
                data.loc[(i, result[1]), result[0].keys()] = list(result[0].values())
        data.to_csv("analysis/data/GNN_basic_random_on_basic_fixed2.csv")

    

if __name__ == "__main__":
    main()