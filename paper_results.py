import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from uuid import uuid4

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import torch
import tyro
from numpy.random import default_rng
from tensordict import TensorDict

from environment.agents.geniusweb import TRAINING_AGENTS
from environment.agents.policy.PPO import GNN
from environment.scenario import ScenarioLoader
from ppo import Args, Policies, concat_envs

pio.kaleido.scope.mathjax = None

TESTS = [
    {
        "name": "scml_dynamic",
        "module": "GNN",
        "models": [
            str(i)
            for i in Path("models", "GNN", "basic_opponents_random_scenarios").iterdir()
        ],
        "scenario": f"environment/scenarios/random_tmp_scml_dynamic{uuid4()}",
        "opponent_sets": ("BASIC",),
    },
    # {
    #     "name": "GNN_basic_random_on_basic_random",
    #     "module": "GNN",
    #     "models": [
    #         str(i)
    #         for i in Path("models", "GNN", "basic_opponents_random_scenarios").iterdir()
    #     ],
    #     "scenario": f"environment/scenarios/random_tmp_GNN_basic_random_on_basic_random_{uuid4()}",
    #     "opponent_sets": ("BASIC",),
    # },
    # {
    #     "name": "GNN_all_random_on_all_random",
    #     "module": "GNN",
    #     "models": [str(i) for i in Path("models", "GNN", "all_opponents_random_scenarios").iterdir()],
    #     "scenario": f"environment/scenarios/random_tmp_GNN_all_random_on_all_random_{uuid4()}",
    #     "opponent_sets": ("ANL2022", "ANL2023", "BASIC"),
    # },
    # {
    #     "name": "GNN_basic_fixed_on_basic_fixed",
    #     "module": "GNN",
    #     "models": [str(i) for i in Path("models", "GNN", "basic_opponents_fixed_scenario").iterdir()],
    #     "scenario": "environment/scenarios/fixed_utility",
    #     "opponent_sets": ("BASIC",),
    # },
    # {
    #     "name": "GNN_all_fixed_on_all_fixed",
    #     "module": "GNN",
    #     "models": [str(i) for i in Path("models", "GNN", "all_opponents_fixed_scenario").iterdir()],
    #     "scenario": "environment/scenarios/fixed_utility",
    #     "opponent_sets": ("ANL2022", "ANL2023", "BASIC"),
    # },
    # {
    #     "name": "Higa_basic_fixed_on_basic_fixed",
    #     "module": "HigaEtAl",
    #     "models": [str(i) for i in Path("models", "HigaEtAl", "basic_opponents_fixed_scenario").iterdir()],
    #     "scenario": "environment/scenarios/fixed_utility",
    #     "opponent_sets": ("BASIC",),
    # },
    # {
    #     "name": "Higa_all_fixed_on_all_fixed",
    #     "module": "HigaEtAl",
    #     "models": [str(i) for i in Path("models", "HigaEtAl", "all_opponents_fixed_scenario").iterdir()],
    #     "scenario": "environment/scenarios/fixed_utility",
    #     "opponent_sets": ("ANL2022", "ANL2023", "BASIC"),
    # },
]


@dataclass
class ArgsEval(Args):
    test_num: int | None = None
    model_paths: tuple[str, ...] | None = None
    exp: str = "scml_dynamic"

    episodes_per_agent: int = 1000
    episodes_per_scenario_per_agent: int = 20


def main():
    args = tyro.cli(ArgsEval)
    results_dir = Path("analysis", "data")
    results_dir.mkdir(parents=True, exist_ok=True)
    assert args.test_num is not None
    test_data = TESTS[args.test_num]

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    scenario_rng = default_rng(args.seed)

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and args.cuda else "cpu"
    )

    # env setup
    used_agents = [
        a for a in TRAINING_AGENTS if a.startswith(tuple(test_data["opponent_sets"]))
    ]
    args.episodes = args.episodes_per_agent * len(used_agents)
    args.episodes_per_scenario = args.episodes_per_scenario_per_agent * len(used_agents)
    envs = None

    if args.opponent == "all":
        args.num_envs = len(used_agents)

    iterables = [list(range(len(test_data["models"]))), sorted(used_agents)]
    index = pd.MultiIndex.from_product(iterables, names=["model", "opponent"])
    data = pd.DataFrame(
        columns=[
            "my_utility",
            "opp_utility",
            "count",
            "rounds_played",
            "self_accepted",
            "found_agreement",
        ],
        index=index,
    )

    loader = ScenarioLoader(
        Path(f"environment/scenarios/{args.exp}_testing"), random_order=False
    )
    for model_index, model_path in enumerate(test_data["models"]):
        print(f"model_index: {model_index}")
        agent_type = model_path.split("/")[1].split("_")[0]

        episodes = 0
        iteration = 0
        log_metrics = defaultdict(lambda: defaultdict(lambda: 0.0))

        details_ = []
        # TRY NOT TO MODIFY: start the game
        while episodes < args.episodes:
            if (
                test_data["scenario"].startswith("environment/scenarios/random_tmp")
                or iteration == 0
            ):
                if test_data["scenario"].startswith("environment/scenarios/random_tmp"):
                    scenario = loader.next_scenario()
                    # scenario = Scenario.create_random(
                    #     [200, 1000], scenario_rng, 5, True
                    # )
                    scenario.to_directory(Path(test_data["scenario"]))

                if envs:
                    envs.close()

                env_config = {
                    "agents": [f"RL_{agent_type}", args.opponent],
                    "used_agents": used_agents,
                    "scenario": test_data["scenario"],
                    "deadline": {"rounds": args.deadline, "ms": 10000},
                    "random_agent_order": args.random_agent_order,
                }
                envs = concat_envs(env_config, args.num_envs, num_cpus=args.num_envs)
                agent: GNN = Policies[agent_type].value(envs, args).to(device)
                agent.load_state_dict(torch.load(model_path, map_location=device))
                agent.train(False)
                agent.action_nvec = tuple(envs.single_action_space.nvec)

                next_obs, _ = envs.reset(seed=args.seed + iteration)
                next_obs = TensorDict(
                    next_obs, batch_size=(args.num_envs,), device=device
                )

            episodes_on_this_scenario = 0
            print(episodes)
            while episodes_on_this_scenario < args.episodes_per_scenario:
                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, _, _, _ = agent.get_action_and_value(next_obs)

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, _, terminations, truncations, infos = envs.step(
                    action.cpu().numpy()
                )
                next_done_bool = np.logical_or(terminations, truncations)
                next_obs = TensorDict(
                    next_obs, batch_size=(args.num_envs,), device=device
                )

                if next_done_bool.any():
                    for info in infos:
                        if info:
                            for agent_id, utility in info["utility_all_agents"].items():
                                if agent_id != f"RL_{test_data['module']}":
                                    log_metrics[agent_id]["opp_utility"] += utility
                                    log_metrics[agent_id]["my_utility"] += info[
                                        "utility_all_agents"
                                    ][f"RL_{test_data['module']}"]
                                    log_metrics[agent_id]["count"] += 1
                                    log_metrics[agent_id]["rounds_played"] += info[
                                        "rounds_played"
                                    ]
                                    log_metrics[agent_id]["self_accepted"] += info[
                                        "self_accepted"
                                    ]
                                    log_metrics[agent_id]["found_agreement"] += info[
                                        "found_agreement"
                                    ]
                                    details += info | dict(
                                        learner_utility=info["utility_all_agents"][
                                            f"RL_{test_data['module']}"
                                        ],
                                        opp_utility=utility,
                                        episode=episodes,
                                        episodes_on_this_scenario=episodes_on_this_scenario,
                                    )
                                    episodes += 1
                                    episodes_on_this_scenario += 1
            iteration += 1

        for opp_id, values in log_metrics.items():
            result = {k: v / values["count"] for k, v in values.items() if k != "count"}
            result["count"] = values["count"]
            data.loc[(model_index, opp_id), result.keys()] = list(result.values())

    data.to_csv(results_dir / f"{test_data['name']}.csv")

    pd.DataFrame.from_records(details).to_csv(
        results_dir / f"details_{test_data['name']}.csv", index=False
    )
    data_plot = pd.read_csv(results_dir / f"{test_data['name']}.csv", index_col=[0, 1])
    plot_results(data_plot, test_data["name"])


def confidence_interval(data, confidence=0.99):
    nans = np.count_nonzero(np.isnan(data))
    if nans > 0:
        print(f"found {nans} NaNs")
    data = [d for d in data if not np.isnan(d)]
    dist = NormalDist.from_samples(data)
    z = NormalDist().inv_cdf((1 + confidence) / 2.0)
    h = dist.stdev * z / ((len(data) - 1) ** 0.5)
    return h


def data_to_summary(data):
    results = {}
    for opponent in data.index.unique(1):
        opp_data = data.query(f"opponent == '{opponent}'")
        results[opponent] = {
            "my_utility_mean": opp_data["my_utility"].mean(),
            "my_utility_CI_99": confidence_interval(opp_data["my_utility"], 0.99),
            "opp_utility_mean": opp_data["opp_utility"].mean(),
            "opp_utility_CI_99": confidence_interval(opp_data["opp_utility"], 0.99),
        }
    return results


def plot_results(data, name):
    figures_dir = Path("analysis", "figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    colors = px.colors.qualitative.Plotly
    summary = data_to_summary(data)
    x = []
    my_utility_mean = []
    my_utility_CI_99 = []
    opp_utility_mean = []
    opp_utility_CI_99 = []
    for opponent, values in summary.items():
        x.append(opponent.split("_")[1])
        my_utility_mean.append(values["my_utility_mean"])
        my_utility_CI_99.append(values["my_utility_CI_99"])
        opp_utility_mean.append(values["opp_utility_mean"])
        opp_utility_CI_99.append(values["opp_utility_CI_99"])

    fig = go.Figure()
    color = None
    if name.startswith("Higa"):
        color = colors[2]
    elif name == "GNN_all_random_on_all_random":
        color = colors[4]

    fig.add_trace(
        go.Bar(
            name="Ours" if name.startswith("GNN") else "Higa et al.",
            marker_color=color,
            x=x,
            y=my_utility_mean,
            error_y=dict(type="data", array=my_utility_CI_99),
            opacity=0.75,
        )
    )
    fig.add_trace(
        go.Bar(
            name="Opponent",
            x=x,
            y=opp_utility_mean,
            error_y=dict(type="data", array=opp_utility_CI_99),
            opacity=0.75,
        )
    )
    width = len(x) * 25 + 100
    fig.update_layout(
        barmode="group",
        width=width * (1 / 0.6),
        height=175 * (1 / 0.6),
        font=dict(
            family="serif",
        ),
        margin=dict(
            t=1,
            b=1,
            l=1,
            r=1,
        ),
        xaxis=dict(
            tickangle=20,
        ),
        yaxis=dict(
            title="Utility",
            range=[0, 1.1],
            dtick=0.1,
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="right", x=1),
    )
    fig.write_image(str(figures_dir / f"{name}.pdf"), scale=0.45)


if __name__ == "__main__":
    main()
