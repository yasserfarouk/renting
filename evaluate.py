from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import tyro
from tensordict import TensorDict

from environment.agents.geniusweb import AGENTS
from environment.agents.policy.PPO import GNN
from environment.negotiation import NegotiationEnvZoo
from ppo import Args, Policies


@dataclass
class ArgsEval(Args):
    model_paths: tuple[str, ...] | None = None
    episodes: int = 100


def evaluate_agent(opponent, model_path, args):
    agent_type = model_path.split("/")[1].split("_")[0]
    used_agents = [a for a in AGENTS if a.startswith(tuple(args.opponent_sets))]
    env_config = {
        "agents": [f"RL_{agent_type}", opponent],
        "used_agents": used_agents,
        "scenario": args.scenario,
        "deadline": {"rounds": args.deadline, "ms": 10000},
        "random_agent_order": args.random_agent_order,
    }
    env = NegotiationEnvZoo(env_config)
    env.single_action_space = env.action_space(f"RL_{agent_type}")
    env.single_observation_space = env.observation_space(f"RL_{agent_type}")
    agent: GNN = Policies[agent_type].value(env, args).to("cpu")
    agent.load_state_dict(torch.load(model_path))
    agent.train(False)


    log_metrics = defaultdict(lambda: .0)
    for episode in range(1, args.episodes + 1):
        next_obs = env.reset(seed=args.seed+episode)[0]
        terminations = {f"RL_{agent_type}": False}

        while not terminations[f"RL_{agent_type}"]:
            next_obs = {k: torch.from_numpy(v[np.newaxis, ...]) for k, v in next_obs[f"RL_{agent_type}"].items() if k != "opponent_encoding"}
            next_obs = TensorDict(next_obs, batch_size=1)
            agent.action_nvec = tuple(env.action_space(f"RL_{agent_type}").nvec)
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(next_obs)

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, _, terminations, _, infos = env.step({f"RL_{agent_type}": action.numpy()[0]})

        info = infos[f"RL_{agent_type}"]
        log_metrics["my_utility"] += info["utility_all_agents"][f"RL_{agent_type}"]
        log_metrics["opp_utility"] += info["utility_all_agents"][opponent]
        log_metrics["rounds_played"] += info["rounds_played"]
        log_metrics["self_accepted"] += info["self_accepted"]
        log_metrics["found_agreement"] += info["found_agreement"]

    log_metrics = {k: v/episode for k, v in log_metrics.items()}

    return log_metrics, opponent

def main():
    args = tyro.cli(ArgsEval)
    used_agents = [a for a in AGENTS if a.startswith(tuple(args.opponent_sets))]
    assert args.model_paths is not None
    if args.debug:
        print(evaluate_agent(used_agents[0], args.model_paths[0], args))
    else:
        iterables = [list(range(len(args.model_paths))), sorted(used_agents)]
        index = pd.MultiIndex.from_product(iterables, names=["model", "opponent"])
        data = pd.DataFrame(columns=["my_utility", "opp_utility","rounds_played", "self_accepted", "found_agreement"], index=index)

        for i, model_path in enumerate(args.model_paths):
            results = []
            for opponent in used_agents:
                results.append(evaluate_agent(opponent, model_path, args))
            for result in results:
                data.loc[(i, result[1]), result[0].keys()] = list(result[0].values())
        data.to_csv("analysis/data/evaluation.csv")

    

if __name__ == "__main__":
    main()