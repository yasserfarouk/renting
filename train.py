import argparse

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.marl_module import MultiAgentRLModuleSpec
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec

import wandb
from environment.agents.geniusweb import AGENTS
from environment.negotiation import NegotiationEnv
from policy.callbacks import InfoCallback
from policy.PPO import MyFixedPPO

parser = argparse.ArgumentParser()
parser.add_argument("--wandb", action="store_true")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--deadline", type=int, default=40)
parser.add_argument("--opponent", type=str, default="random")
parser.add_argument("--fcnet_hiddens", nargs="+", type=int, default=[32, 32])
parser.add_argument("--entropy_coeff", type=float, default=0)
parser.add_argument("--lr", type=float, default=5e-4)
parser.add_argument("--lambda_", type=float, default=0.95)
parser.add_argument("--gamma", type=float, default=1)


if __name__ == "__main__":
    args = parser.parse_args()

    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        return "policy"

    used_agents = [a for a in AGENTS if not a.startswith("BASIC")]

    ray.init(local_mode=args.debug)
    env_config = {
        "agent_configs": ["RL", args.opponent],
        "used_agents": used_agents,
        "scenario": "environment/scenarios/scenario_0001",
        "deadline": {"rounds": args.deadline, "ms": 10000},
        "random_agent_order": True,
        "offer_max_first": True,
        "wandb_config": {
            "project": "RL_negotiation",
            "name": f"{args.opponent}_{args.deadline}_{args.fcnet_hiddens}_{args.entropy_coeff}_{args.lr}_{args.lambda_}_{args.gamma}",
        },
    }
    config = (
        PPOConfig()
        .environment(NegotiationEnv, env_config=env_config)
        .rl_module(
            rl_module_spec=MultiAgentRLModuleSpec(
                module_specs=SingleAgentRLModuleSpec(module_class=MyFixedPPO)
            ),
            _enable_rl_module_api=True,
        )
        .rollouts(
            num_rollout_workers=30 if not args.debug else 0,
            num_envs_per_worker=1,
            batch_mode="complete_episodes",
            ignore_worker_failures=True,
            recreate_failed_workers=True,
        )
        .resources(num_gpus=0, num_cpus_per_worker=1)
        .callbacks(InfoCallback)
        .training(
            model={"fcnet_hiddens": args.fcnet_hiddens},
            lambda_=args.lambda_,  # default 1
            gamma=args.gamma,  # default 0.99
            entropy_coeff=args.entropy_coeff,  # default 0
            lr=args.lr,  # default 5e-5
            train_batch_size=4000 if not args.debug else 100,
            sgd_minibatch_size=128 if not args.debug else 20,
            num_sgd_iter=30 if not args.debug else 5,
        )
    )

    if args.wandb:
        wandb.login()
        wandb.init(
            project=env_config["wandb_config"]["project"],
            name=env_config["wandb_config"]["name"],
            # name="main_worker"
            # entity="RL_negotiation",
        )

    algo = config.build()
    for _ in range(1000):
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
