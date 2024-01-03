import argparse

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.marl_module import MultiAgentRLModuleSpec
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.env.env_context import EnvContext

import wandb
from environment.agents.geniusweb import AGENTS
from environment.negotiation import NegotiationEnv
from policy.callbacks import InfoCallback
from policy.PPO import GraphToFixed, GraphToGraph
from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule

parser = argparse.ArgumentParser()
parser.add_argument("--wandb", action="store_true")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--deadline", type=int, default=40)
parser.add_argument("--module", type=str, default="GraphToFixed")
parser.add_argument("--entropy_coeff", type=float, default=0)
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--lambda_", type=float, default=1)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--train_batch_size", type=int, default=4000)
parser.add_argument("--sgd_minibatch_size", type=int, default=128)
parser.add_argument("--num_sgd_iter", type=int, default=30)
parser.add_argument("--pooling_op", type=str, default="max", choices=["max", "mean", "sum"])
parser.add_argument("--training_iterations", type=int, default=1000)
parser.add_argument("--opponent", type=str, default="random")
parser.add_argument("--opponent_sets", nargs="+", type=str, default=["ANL2022","ANL2023","CSE3210"])
parser.add_argument("--scenario", type=str, default="environment/scenarios/random_utility")
parser.add_argument("--random_agent_order", action="store_true")
parser.add_argument("--offer_max_first", action="store_true")

MODULES = {
    "GraphToFixed": GraphToFixed,
    "GraphToGraph": GraphToGraph,
}


if __name__ == "__main__":
    args = parser.parse_args()

    used_agents = [a for a in AGENTS if a.startswith(tuple(args.opponent_sets))]

    ray.init()
    env_config = {
        "RL_agents": ["RL_0"],
        "always_playing": ["RL_0"],
        "opponent": args.opponent,
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

    config = (
        PPOConfig()
        .environment(
            env=NegotiationEnv,
            env_config=env_config,
            # disable_env_checking=True,
        )
        .rl_module(
            rl_module_spec=MultiAgentRLModuleSpec(
                module_specs=SingleAgentRLModuleSpec(module_class=MODULES[args.module], model_config_dict={"num_used_agents": len(used_agents), "pooling_op": args.pooling_op})
                # module_specs=SingleAgentRLModuleSpec(module_class=PPOTorchRLModule)
            )
        )
        .experimental(
            _enable_new_api_stack=True,
            _disable_preprocessor_api=True,
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
            # model={"fcnet_hiddens": args.fcnet_hiddens},
            lambda_=args.lambda_,
            gamma=args.gamma,
            entropy_coeff=args.entropy_coeff,
            lr=args.lr,
            train_batch_size=args.train_batch_size if not args.debug else 100,
            sgd_minibatch_size=args.sgd_minibatch_size if not args.debug else 20,
            num_sgd_iter=args.num_sgd_iter if not args.debug else 5,
        )
    )

    if args.wandb:
        config_dict = {k: v for k, v in vars(args).items() if k not in ["wandb", "debug"]}
        obs, _ = NegotiationEnv(EnvContext(env_config, 1)).reset()
        config_dict["observation_space"] = list(next(iter(obs.values())).keys())
        config_dict["used_agents"] = used_agents
        wandb.login()
        wandb.init(
            project="RL_negotiation",
            name=f"encoding_{args.opponent}_{args.deadline}_{args.scenario}",
            config=config_dict,
        )

    algo = config.build()
    for _ in range(args.training_iterations):
        result = algo.train()
        print(f"Episode reward mean: {result['episode_reward_mean']:.4f}")
        if args.wandb:
            wandb.log(
                {
                    "episode_reward_min": result["episode_reward_min"],
                    "episode_reward_mean": result["episode_reward_mean"],
                    "episode_reward_max": result["episode_reward_max"],
                },
                step=result["timesteps_total"],
            )
            wandb.log(result["custom_metrics"], step=result["timesteps_total"])

    if args.wandb:
        wandb.finish()

    ray.shutdown()
