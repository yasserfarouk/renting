import argparse
from pathlib import Path

import ray
from gymnasium.spaces import MultiDiscrete
from numpy.random import default_rng
from ray.rllib.algorithms import AlgorithmConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.core.rl_module.marl_module import MultiAgentRLModuleSpec
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.env.env_context import EnvContext

import wandb
from environment.agents.geniusweb import AGENTS
from environment.negotiation import NegotiationEnv
from environment.scenario import Scenario
from policy.callbacks import InfoCallback
from policy.learner import CustomPPOTorchLearner
from policy.PPO import (FixedToFixed, FixedToFixed2, GraphToFixed,
                        GraphToGraph, GraphToGraph2,
                        GraphToGraphLargeFixedAction, HigaEtAl, PureGCN, Test)

parser = argparse.ArgumentParser()
parser.add_argument("--wandb", action="store_true")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--deadline", type=int, default=40)
parser.add_argument("--module", type=str, default="GraphToFixed", choices=["HigaEtAl", "GraphToFixed", "GraphToGraph", "PureGCN", "FixedToFixed", "FixedToFixed2", "GraphToGraph2", "GraphToGraphLargeFixedAction", "Test"])
# parser.add_argument("--rl_agent_class", type=str, default="RLAgentStackedObs", choices=["RLAgentStackedObs", "RLAgentGraphObs"])
parser.add_argument("--reward_type", type=str, default="utility", choices=["utility", "utility_and_difference", "difference"])
parser.add_argument("--entropy_coeff", type=float, default=0)
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--grad_clip", type=float, default=None)
parser.add_argument("--lambda_", type=float, default=1)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--train_batch_size", type=int, default=4000)
parser.add_argument("--sgd_minibatch_size", type=int, default=128)
parser.add_argument("--num_sgd_iter", type=int, default=30)
parser.add_argument("--pooling_op", type=str, default="max", choices=["sum", "mean", "max", "mul", "min"])
parser.add_argument("--hidden_size", type=int, default=32)
parser.add_argument("--num_gcn_layers", type=int, default=2)
parser.add_argument("--training_iterations", type=int, default=1000)
parser.add_argument("--opponent", type=str, default="random")
parser.add_argument("--use_opponent_encoding", action="store_true")
parser.add_argument("--opponent_sets", nargs="+", type=str, default=["ANL2022","ANL2023","CSE3210"])
parser.add_argument("--scenario", type=str, default="environment/scenarios/random_utility")
parser.add_argument("--random_agent_order", action="store_true")
parser.add_argument("--offer_max_first", action="store_true")

MODULES = {
    "HigaEtAl": HigaEtAl,
    "FixedToFixed": FixedToFixed,
    "FixedToFixed2": FixedToFixed2,
    "GraphToFixed": GraphToFixed,
    "GraphToGraph": GraphToGraph,
    "GraphToGraph2": GraphToGraph2,
    "PureGCN": PureGCN,
    "GraphToGraphLargeFixedAction": GraphToGraphLargeFixedAction,
    "Test": Test,
}

if __name__ == "__main__":
    args = parser.parse_args()

    used_agents = [a for a in AGENTS if a.startswith(tuple(args.opponent_sets))]

    ray.init()
    env_config = {
        "agents": [f"RL_{args.module}", "all"],
        "used_agents": used_agents,
        "scenario": args.scenario,
        "deadline": {"rounds": args.deadline, "ms": 10000},
        "random_agent_order": args.random_agent_order,
        "offer_max_first": args.offer_max_first,
    }
    if args.opponent == "all":
        if len(used_agents) > 30:
            raise ValueError("Too many agents to support training vs all agents concurrently")
        num_rollout_workers = len(used_agents)
    else:
        num_rollout_workers = 30

    rl_module_spec = SingleAgentRLModuleSpec(
        module_class=MODULES[args.module],
        model_config_dict={
            "num_used_agents": len(used_agents),
            "pooling_op": args.pooling_op,
            "hidden_size": args.hidden_size,
            "num_gcn_layers": args.num_gcn_layers,
            "use_opponent_encoding": args.use_opponent_encoding,
        },
    )

    config: PPOConfig = (
        PPOConfig()
        .environment(
            env=NegotiationEnv,
            env_config=env_config,
            # disable_env_checking=True,
        )
        .rl_module(
            rl_module_spec=MultiAgentRLModuleSpec(
                module_specs=rl_module_spec
                # module_specs=SingleAgentRLModuleSpec(module_class=PPOTorchRLModule)
            )
        )
        .experimental(
            _enable_new_api_stack=True,
            _disable_preprocessor_api=args.module not in ["HigaEtAl", "FixedToFixed", "FixedToFixed2"],
            # _disable_action_flattening=True,
            # _disable_initialize_loss_from_dummy_batch=True,
        )
        .rollouts(
            num_rollout_workers=num_rollout_workers if not args.debug else 0,
            num_envs_per_worker=1,
            batch_mode="complete_episodes",
        )
        .fault_tolerance(recreate_failed_workers=True)
        .resources(num_gpus=0, num_cpus_per_worker=1)
        .callbacks(InfoCallback)
        .training(
            grad_clip=args.grad_clip,
            lambda_=args.lambda_,
            gamma=args.gamma,
            entropy_coeff=args.entropy_coeff,
            lr=args.lr,
            train_batch_size=args.train_batch_size if not args.debug else 100,
            sgd_minibatch_size=args.sgd_minibatch_size if not args.debug else 20,
            num_sgd_iter=args.num_sgd_iter if not args.debug else 5,
            learner_class=CustomPPOTorchLearner,
        )
    )

    algo = config.build()

    if args.wandb:
        config_dict = {k: v for k, v in vars(args).items() if k not in ["wandb", "debug"]}
        obs, _ = NegotiationEnv(EnvContext(env_config, 1)).reset()
        config_dict["observation_space"] = list(next(iter(obs.values())).keys())
        config_dict["used_agents"] = used_agents
        wandb.login()
        wandb.init(
            project="RL_negotiation",
            name=f"{args.module}",
            config=config_dict,
        )

    random_generator = default_rng(0)
    timesteps_total = 0
    for i in range(args.training_iterations):
        if i > 0:
            if args.scenario == "environment/scenarios/random":
                scenario = Scenario.create_random([100, 1000], random_generator, no_utility_functions=True)
                scenario.to_directory(Path(args.scenario))
                algo = config.build()
                algo.set_weights(weights)
                # algo.get_policy().action_space = MultiDiscrete([2] + [len(v) for v in scenario.objectives.values()])
                # algo.get_policy().action_space_struct = MultiDiscrete([2] + [len(v) for v in scenario.objectives.values()])
        result = algo.train()
        # policy = algo.get_policy()
        weights = algo.get_weights()
        # config = algo.get_config()
        # clean = algo.cleanup()
        print(f"Episode reward mean: {result['episode_reward_mean']:.4f}")
        timesteps_total += result["agent_timesteps_total"]
        if args.wandb:
            wandb.log(
                {
                    "episode_reward_min": result["episode_reward_min"],
                    "episode_reward_mean": result["episode_reward_mean"],
                    "episode_reward_max": result["episode_reward_max"],
                },
                step=timesteps_total,
            )
            wandb.log(result["custom_metrics"], step=timesteps_total)

    if args.wandb:
        wandb.finish()

    ray.shutdown()
