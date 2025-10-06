from collections import defaultdict
from pathlib import Path
from negmas.helpers import unique_name
from rich import print
from rich.progress import track
from dataclasses import dataclass
from environment.scenario import ScenarioLoader

import numpy as np
import pandas as pd
import torch
import tyro
from tensordict import TensorDict

from environment.agents.policy.PPO import GNN
from environment.negotiation import NegotiationEnvZoo
from ppo import Args, Policies, find_opponents, MODELS_BASE, BASE
from time import perf_counter


SAVE_LOC = BASE


@dataclass
class ArgsEval(Args):
    method: str = "GNN"
    exp: str = ""
    model_paths: tuple[str, ...] | None = None
    episodes: int = 50

    # extension parameters
    issue_size: int = 0
    n_issues: int = 0

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

    log_metrics = defaultdict(list)
    detailed_metrics = []
    _strt = perf_counter()

    for episode in track(range(1, args.episodes + 1)):
        if (
            env_config["scenario"].startswith("environment/scenarios/random_tmp")
            or episode == 0
        ):
            if env_config["scenario"].startswith("environment/scenarios/random_tmp"):
                scenario = loader.next_scenario()
                # scenario = Scenario.create_random(
                #     [200, 1000], scenario_rng, 5, True
                # )
                if args.verbose:
                    print(
                        f"Testing on scenario {scenario.name} of size {scenario.size}",
                        flush=True,
                    )
                scenario.to_directory(Path(env_config["scenario"]))
        next_obs = env.reset(seed=args.seed + episode)[0]
        terminations = {f"RL_{agent_type}": False}
        step = 0

        while not terminations[f"RL_{agent_type}"]:
            next_obs = {
                k: torch.from_numpy(v[np.newaxis, ...])
                for k, v in next_obs[f"RL_{agent_type}"].items()
                if k != "opponent_encoding"
            }
            next_obs = TensorDict(next_obs, batch_size=1)
            agent.action_nvec = tuple(env.action_space(f"RL_{agent_type}").nvec)
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(next_obs)

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, _, terminations, _, infos = env.step(
                {f"RL_{agent_type}": action.numpy()[0]}
            )
            step += 1

        t_ = perf_counter() - _strt
        total_steps = step
        step = 0
        info = infos[f"RL_{agent_type}"]
        log_metrics["my_utility"].append(info["utility_all_agents"][f"RL_{agent_type}"])
        log_metrics["opp_utility"].append(info["utility_all_agents"][opponent])
        log_metrics["my_advantage"].append(
            info["advantage_all_agents"][f"RL_{agent_type}"]
        )
        log_metrics["opp_advantage"].append(info["advantage_all_agents"][opponent])
        log_metrics["rounds_played"].append(info["rounds_played"])
        log_metrics["self_accepted"].append(info["self_accepted"])
        log_metrics["found_agreement"].append(info["found_agreement"])
        log_metrics["agreement"].append(info["found_agreement"])
        log_metrics["pareto_optimality"].append(
            info.get("pareto_optimality", float("nan"))
        )
        log_metrics["nash_optimality"].append(info.get("nash_optimality", float("nan")))
        log_metrics["kalai_optimality"].append(
            info.get("kalai_optimality", float("nan"))
        )
        log_metrics["modified_kalai_optimality"].append(
            info.get("modified_kalai_optimality", float("nan"))
        )
        log_metrics["max_welfare_optimality"].append(
            info.get("max_welfare_optimality", float("nan"))
        )
        log_metrics["ks_optimality"].append(info.get("ks_optimality", float("nan")))
        log_metrics["modified_ks_optimality"].append(
            info.get("modified_ks_optimality", float("nan"))
        )

        try:
            rtime = max(t_ / args.time_limit, total_steps / args.deadline)
        except:
            rtime = None
        detailed_metrics.append(
            info
            | dict(
                learner_utility=info["utility_all_agents"][f"RL_{agent_type}"],
                opp_utility=info["utility_all_agents"][opponent],
                learner_advantage=info["advantage_all_agents"][f"RL_{agent_type}"],
                opp_advantage=info["advantage_all_agents"][opponent],
                time=t_,
                step=total_steps,
                relative_time=rtime,
                pareto_optimality=info.get("pareto_optimality", float("nan")),
                nash_optimality=info.get("nash_optimality", float("nan")),
                kalai_optimality=info.get("kalai_optimality", float("nan")),
                modified_kalai_optimality=info.get(
                    "modified_kalai_optimality", float("nan")
                ),
                max_welfare_optimality=info.get("max_welfare_optimality", float("nan")),
                ks_optimality=info.get("ks_optimality", float("nan")),
                modified_ks_optimality=info.get("modified_ks_optimality", float("nan")),
            )
        )

    mm_ = {f"{k}_mean": np.mean(v) for k, v in log_metrics.items()}
    mm_ |= {f"{k}_std": np.std(v) for k, v in log_metrics.items()}
    mm_ |= {
        "episodes": args.episodes,
        "n_negotiations": len(log_metrics),
        "model_path": model_path,
    }

    return mm_, detailed_metrics, opponent


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
        print("Saving ...")
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
        details_ = []

        save_loc = (
            SAVE_LOC
            / (args.exp if args.exp else "unknown")
            / args.method
            / unique_name("", sep="")
        )
        save_loc.mkdir(parents=True, exist_ok=True)
        for i, model_path in enumerate(args.model_paths):
            print(f"{i} of {len(args.model_paths)}: {model_path}")
            results = []
            for opponent in used_agents:
                metrics, details, opp = evaluate_agent(opponent, model_path, args)
                details_ += details
                results.append((metrics, opp))
            for result in results:
                data.loc[(i, result[1]), result[0].keys()] = list(result[0].values())
        data.to_csv(save_loc / "evaluation.csv")
        pd.DataFrame.from_records(details_).to_csv(
            save_loc / "details.csv", index=False
        )
        if args.debug:
            # metrics, details, opp = evaluate_agent(
            #     used_agents[0], args.model_paths[0], args
            # )
            print([_["scenario_src_path"] for _ in details])
            print(metrics)


if __name__ == "__main__":
    main()
