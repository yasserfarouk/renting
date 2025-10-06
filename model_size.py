from pathlib import Path
from rich import print
from dataclasses import dataclass
from environment.scenario import ScenarioLoader

import pandas as pd
import torch
import tyro

from environment.agents.policy.PPO import GNN
from environment.negotiation import NegotiationEnvZoo
from ppo import Args, Policies, find_opponents, MODELS_BASE, BASE


SAVE_LOC = BASE


@dataclass
class ArgsEval(Args):
    method: str = "GNN"
    exp: str = ""
    model_paths: tuple[str, ...] | None = None
    training: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.opponent_types, self.opponent_map = find_opponents(False, self.exp)
        if self.method is not None:
            base = MODELS_BASE
            if self.method:
                paths = list(base.glob(f"{self.method}_*"))
                if self.exp:
                    paths = [p for p in paths if f"_{self.exp}." in p.name]
                if paths:
                    paths = sorted(paths, reverse=True)
                print(f"Will use model {str(paths[0])}")
                if self.model_paths:
                    self.model_paths = tuple(
                        list(self.model_paths).append(str(paths[0]))
                    )
                else:
                    self.model_paths = (str(paths[0]),)


def evaluate_agent(opponent, model_path, args):
    agent_type = model_path.split("/")[-1].split("_")[0]
    exp = "_".join(model_path.split("/")[-1].split("_")[1:]).split(".")[0]
    print(f"Evaluating {agent_type} for {exp} against {opponent}")
    used_agents = [
        a for a in args.opponent_map if a.startswith(tuple(args.opponent_sets))
    ]
    env_config = {
        "agents": [f"RL_{agent_type}", opponent],
        "used_agents": used_agents,
        "scenario": args.scenario,
        "deadline": {"rounds": args.deadline, "ms": args.time_limit},
        "random_agent_order": args.random_agent_order,
        "issue_size": args.issue_size,
        "n_issues": args.n_issues,
        "testing": True,
    }
    loader = ScenarioLoader(
        Path(f"environment/scenarios/testing/{args.exp}"), random_order=False
    )
    loader.random_scenario().to_directory(Path(env_config["scenario"]))
    env = NegotiationEnvZoo(env_config)
    env.single_action_space = env.action_space(f"RL_{agent_type}")
    env.single_observation_space = env.observation_space(f"RL_{agent_type}")
    agent: GNN = Policies[agent_type].value(env, args).to("cpu")
    agent.load_state_dict(torch.load(model_path))
    agent.train(False)
    size = sum(p.numel() for p in agent.parameters())

    return size


def main():
    args = tyro.cli(ArgsEval)
    used_agents = [
        a for a in args.opponent_map if a.startswith(tuple(args.opponent_sets))
    ]
    assert args.model_paths is not None
    if args.debug:
        args.episodes = 5
        used_agents = used_agents[:1]
        args.model_paths = args.model_paths[:1]
        # print(details)
    # else:
    if 1:
        print("Calculating ...")
        iterables = [list(range(len(args.model_paths))), sorted(used_agents)]
        index = pd.MultiIndex.from_product(iterables, names=["model", "opponent"])
        data = pd.DataFrame(
            columns=[  # type: ignore
                "my_utility",
                "opp_utility",
                "rounds_played",
                "self_accepted",
                "found_agreement",
            ],
            index=index,
        )
        sizes = dict()

        for i, model_path in enumerate(args.model_paths):
            print(f"{i} of {len(args.model_paths)}: {model_path}")
            results = []
            for opponent in used_agents:
                size = evaluate_agent(opponent, model_path, args)
                sizes[model_path] = size
        print(sizes)


if __name__ == "__main__":
    main()
