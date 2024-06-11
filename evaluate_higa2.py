# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cloudpickle
import numpy as np
import pandas as pd
import supersuit as ss
import torch
import tyro
from numpy.random import default_rng
from supersuit.vector import MakeCPUAsyncConstructor
from tensordict import TensorDict

from environment.agents.geniusweb import AGENTS
from environment.agents.policy.PPO import (
    HigaEtAl2,
    PureGNN2,
)
from environment.negotiation import NegotiationEnvZoo
from environment.scenario import Scenario

AGENT_MODULES = {
    "PureGNN2": PureGNN2,
    "HigaEtAl2": HigaEtAl2,
}

MODEL_PATHS = {
    "basic_fixed": [
        "models/HigaEtAl2_24-03-08_16:09:00_77b9728c-56b6-42e0-b6ba-d0b94d4118e4",
        "models/HigaEtAl2_24-03-08_16:09:00_a3d0da92-00e4-42e4-9454-d4edc11f9df9",
        "models/HigaEtAl2_24-03-08_16:09:01_7993bcaf-0fdd-4c1d-a311-90cb8bc75933",
        "models/HigaEtAl2_24-03-08_17:44:08_2ba8af95-904c-41fe-a983-d26d6bb26170",
        "models/HigaEtAl2_24-03-08_17:45:45_2e1ee409-07f3-4a1b-aeb3-3a6f20174f0b",
        "models/HigaEtAl2_24-03-08_17:45:52_73ee1767-2511-4e9a-b4f1-5737cffd67ea",
        "models/HigaEtAl2_24-03-08_17:47:50_73c4e58e-dbdc-48c1-a933-82978991f3f1",
        "models/HigaEtAl2_24-03-08_17:48:59_5c773c76-d861-4154-924d-20644888a532",
        "models/HigaEtAl2_24-03-08_19:12:28_db41c6fd-8b20-469b-9943-637f9f70b544",
        "models/HigaEtAl2_24-03-08_19:14:33_b1e8c2c5-1cd1-4e99-8a7c-926a71bc05fb",
    ],
    "all_fixed": [
        "models/HigaEtAl2_24-03-08_12:35:58",
        "models/HigaEtAl2_24-03-08_13:05:57",
        "models/HigaEtAl2_24-03-08_13:26:07",
        "models/HigaEtAl2_24-03-08_10:48:21",
        "models/HigaEtAl2_24-03-08_10:22:11",
        "models/HigaEtAl2_24-03-07_22:01:56",
        "models/HigaEtAl2_24-03-07_22:01:54",
        "models/HigaEtAl2_24-03-07_22:01:53",
    ],
}

TESTS = [
    {
        "name": "Higa_basic_fixed_on_basic_fixed",
        "module": "HigaEtAl2",
        "models": "basic_fixed",
        "scenario": "environment/scenarios/fixed_utility",
        "opponent_sets": ("BASIC",),
    },
    {
        "name": "Higa_all_fixed_on_all_fixed",
        "module": "HigaEtAl2",
        "models": "all_fixed",
        "scenario": "environment/scenarios/fixed_utility",
        "opponent_sets": ("ANL2022", "ANL2023", "BASIC"),
    },
]



@dataclass
class Args:
    test_num: int

    debug: bool = False

    deadline: int = 40
    use_opponent_encoding: bool = False
    opponent: str = "random"
    num_envs: int = 30 # 18 or 4
    random_agent_order: bool = False
    gat_v2: bool = False
    add_self_loops: bool = True
    hidden_size: int = 256
    heads: int = 4
    gnn_layers: int = 4
    out_layers: int = 1

    episodes_per_agent: int = 1000
    episodes_per_scenario_per_agent: int = 20

    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    cuda: bool = True


def concat_envs(env_config, num_vec_envs, num_cpus=0):
    def vec_env_args(env, num_envs):
        def env_fn():
            env_copy = cloudpickle.loads(cloudpickle.dumps(env))
            return env_copy

        return [env_fn] * num_envs, env.observation_space, env.action_space
    env = NegotiationEnvZoo(env_config)
    vec_env = ss.pettingzoo_env_to_vec_env_v1(env)
    num_cpus = min(num_cpus, num_vec_envs)
    vec_env = MakeCPUAsyncConstructor(num_cpus)(*vec_env_args(vec_env, num_vec_envs))

    vec_env.single_observation_space = vec_env.observation_space
    vec_env.single_action_space = vec_env.action_space
    vec_env.is_vector_env = True
    return vec_env

def main():
    args = tyro.cli(Args)
    test_data = TESTS[args.test_num]

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    used_agents = [a for a in AGENTS if a.startswith(tuple(test_data["opponent_sets"]))]
    args.episodes = args.episodes_per_agent * len(used_agents)
    args.episodes_per_scenario = args.episodes_per_scenario_per_agent * len(used_agents)
    env_config = {
        "agents": [f"RL_{test_data['module']}", args.opponent],
        "used_agents": used_agents,
        "scenario": test_data["scenario"],
        "deadline": {"rounds": args.deadline, "ms": 10000},
        "random_agent_order": args.random_agent_order,
    }
    envs = concat_envs(env_config, args.num_envs, num_cpus=args.num_envs)

    iterables = [list(range(len(MODEL_PATHS[test_data["models"]]))), sorted(used_agents)]
    index = pd.MultiIndex.from_product(iterables, names=["model", "opponent"])
    data = pd.DataFrame(columns=["my_utility", "opp_utility", "count", "rounds_played", "self_accepted", "found_agreement"], index=index)

    for model_index, model_path in enumerate(MODEL_PATHS[test_data["models"]]):
        agent: HigaEtAl2 = AGENT_MODULES[test_data["module"]](envs, args).to(device)
        agent.load_state_dict(torch.load(model_path))
        agent.train(False)

        episodes = 0
        log_metrics = defaultdict(lambda: defaultdict(lambda: .0))
        next_obs, _ = envs.reset(seed=args.seed)
        next_obs = TensorDict(next_obs, batch_size=(args.num_envs,), device=device)
        # TRY NOT TO MODIFY: start the game
        while episodes < args.episodes:
            print(episodes)
            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(next_obs)

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, _, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done_bool = np.logical_or(terminations, truncations)
            next_obs = TensorDict(next_obs, batch_size=(args.num_envs,), device=device)

            if next_done_bool.any():
                for info in infos:
                    if info:
                        for agent_id, utility in info["utility_all_agents"].items():
                            if agent_id != f"RL_{test_data['module']}":
                                log_metrics[agent_id]["opp_utility"] += utility
                                log_metrics[agent_id]["my_utility"] += info["utility_all_agents"][f"RL_{test_data['module']}"]
                                log_metrics[agent_id]["count"] += 1
                                log_metrics[agent_id]["rounds_played"] += info["rounds_played"]
                                log_metrics[agent_id]["self_accepted"] += info["self_accepted"]
                                log_metrics[agent_id]["found_agreement"] += info["found_agreement"]
                                episodes += 1


        for opp_id, values in log_metrics.items():
            result = {k: v / values["count"] for k, v in values.items() if k != "count"}
            result["count"] = values["count"]
            data.loc[(model_index, opp_id), result.keys()] = list(result.values())


    data.to_csv(f"analysis/data/{test_data['name']}.csv")

if __name__ == "__main__":
    main()