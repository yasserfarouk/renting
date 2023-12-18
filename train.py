import argparse
import json

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.marl_module import MultiAgentRLModuleSpec
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec

import wandb
from environment.negotiation import NegotiationEnv
from policy.callbacks import InfoCallback
from policy.PPO import MyFixedPPO

parser = argparse.ArgumentParser()
parser.add_argument("--wandb", action="store_true")
parser.add_argument("--debug", action="store_true")

if __name__ == "__main__":
    args = parser.parse_args()

    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        return "policy"

    ray.init(local_mode=args.debug)
    env_config = {
        "agent_configs": ["RL", "ANL2022.Agent007"],
        "scenario": "environment/scenarios/scenario_0001",
        "deadline": {"rounds": 40, "ms": 10000},
        "random_agent_order": True,
        "offer_max_first": True,
        "wandb_config": {
            "project": "RL_negotiation",
            "name": "test_run_agent007_40_3232_01_e5",
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
        )
        .resources(num_gpus=0, num_cpus_per_worker=1)
        .callbacks(InfoCallback)
        .training(
            model={"fcnet_hiddens": [32, 32]},
            lambda_=0.95, # default 1
            gamma=1,  # default 0.99
            entropy_coeff=0.01, # default 0
            lr=5e-5,  # default 5e-5
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
