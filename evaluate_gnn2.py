# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

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
    "basic_random": [
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
    ],
    "all_random": [
        "models/PureGNN2_24-03-07_21:54:15_b108e910-12ca-4fc9-bc50-cd512dd88776",
        "models/PureGNN2_24-03-07_21:54:23_f5e4c325-2daa-4e83-b524-7ec83bf16ffe",
        "models/PureGNN2_24-03-07_21:54:51_5050ab79-0f9c-40b3-a427-77bac540c8d8",
        "models/PureGNN2_24-03-07_21:54:56_da6a1908-948b-44ca-a4ce-f302319c83fb",
        "models/PureGNN2_24-03-07_21:55:02_01b7c73b-396f-4ebe-8b81-a7ca552c9dbb",
        "models/PureGNN2_24-03-07_21:55:08_8168e11b-f1f0-4bea-a026-57744ebfb87b",
        "models/PureGNN2_24-03-07_21:55:15_41467e23-8bec-4930-abe6-723b28e0c363",
        "models/PureGNN2_24-03-07_21:55:22_ff95ce58-3abc-4e99-9e93-a4498da9310d",
        "models/PureGNN2_24-03-07_21:55:29_f92b1d62-d1ba-43da-afb6-5999849d4337",
        "models/PureGNN2_24-03-07_21:55:37_5723b742-a5cb-40ff-bdd2-72d85e207424",
    ],
    "basic_fixed": [
        "models/PureGNN2_24-03-08_14:02:50_656f225a-1603-4ef9-9566-38da18717f51",
        "models/PureGNN2_24-03-08_14:02:55_1670a829-d1e3-4a4f-b088-2b2fd11c8033",
        "models/PureGNN2_24-03-08_14:02:58_200d4328-b516-4fe6-9c57-9c4d8b7ebd38",
        "models/PureGNN2_24-03-08_14:03:01_9c1c24a9-3ff3-43c8-aaa3-9cebf0b5c99b",
        "models/PureGNN2_24-03-08_14:03:05_d5ed270c-016e-4458-979a-7f7e70345a64",
        "models/PureGNN2_24-03-08_14:03:08_1b67f904-16df-4405-94fa-fe490dbb14dd",
        "models/PureGNN2_24-03-08_14:03:11_47e4d66f-d10e-4a3f-b29b-1584d2bc25dc",
        "models/PureGNN2_24-03-08_14:03:14_f6df574a-05f3-4094-99df-7e5d36c4485e",
        "models/PureGNN2_24-03-08_14:13:09_6ffc8ad4-b0de-49c1-abfb-702393693585",
        "models/PureGNN2_24-03-08_14:24:51_ea76f6ec-4644-427d-bf09-4ec80361be5a",
    ]
}

TESTS = [
    {
        "name": "GNN_basic_random_on_basic_random",
        "module": "PureGNN2",
        "models": "basic_random",
        "scenario": f"environment/scenarios/random_tmp_GNN_basic_random_on_basic_random_{uuid4()}",
        "opponent_sets": ("BASIC",),
    },
    {
        "name": "GNN_all_random_on_all_random",
        "module": "PureGNN2",
        "models": "all_random",
        "scenario": f"environment/scenarios/random_tmp_GNN_all_random_on_all_random_{uuid4()}",
        "opponent_sets": ("ANL2022", "ANL2023", "BASIC"),
    },
    {
        "name": "GNN_basic_fixed_on_basic_fixed",
        "module": "PureGNN2",
        "models": "basic_fixed",
        "scenario": "environment/scenarios/fixed_utility",
        "opponent_sets": ("BASIC",),
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
    scenario_rng = default_rng(args.seed)

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
    envs = None

    iterables = [list(range(len(MODEL_PATHS[test_data["models"]]))), sorted(used_agents)]
    index = pd.MultiIndex.from_product(iterables, names=["model", "opponent"])
    data = pd.DataFrame(columns=["my_utility", "opp_utility", "count", "rounds_played", "self_accepted", "found_agreement"], index=index)

    for model_index, model_path in enumerate(MODEL_PATHS[test_data["models"]]):
        print(f"model_index: {model_index}")
        agent: PureGNN2 = AGENT_MODULES[test_data["module"]](len(used_agents), args).to(device)
        agent.load_state_dict(torch.load(model_path))
        agent.train(False)

        episodes = 0
        iteration = 0
        log_metrics = defaultdict(lambda: defaultdict(lambda: .0))
        # TRY NOT TO MODIFY: start the game
        while episodes < args.episodes:
            if test_data["scenario"].startswith("environment/scenarios/random_tmp") or iteration == 0:
                if test_data["scenario"].startswith("environment/scenarios/random_tmp"):
                    scenario = Scenario.create_random([200, 1000], scenario_rng, 5, True)
                    # scenario.calculate_specials()
                    scenario.to_directory(Path(test_data["scenario"]))
                
                if envs:
                    envs.close()
                envs = concat_envs(env_config, args.num_envs, num_cpus=args.num_envs)
                agent.action_nvec = tuple(envs.single_action_space.nvec)

                next_obs, _ = envs.reset(seed=args.seed + iteration)
                next_obs = TensorDict(next_obs, batch_size=(args.num_envs,), device=device)

            episodes_on_this_scenario = 0
            print(episodes)
            while episodes_on_this_scenario < args.episodes_per_scenario:

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
                                    episodes_on_this_scenario += 1
            iteration += 1


        for opp_id, values in log_metrics.items():
            result = {k: v / values["count"] for k, v in values.items() if k != "count"}
            result["count"] = values["count"]
            data.loc[(model_index, opp_id), result.keys()] = list(result.values())


    data.to_csv(f"analysis/data/{test_data['name']}.csv")

if __name__ == "__main__":
    main()