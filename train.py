import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.marl_module import MultiAgentRLModuleSpec
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule

from environment.negotiation import NegotiationEnv


class DiscreteBCTorchModule(TorchRLModule):
    def input_specs_exploration(self):
        # Enforce that input nested dict to exploration method has a key "obs"
        return ["obs"]

    def output_specs_exploration(self):
        # Enforce that output nested dict from exploration method has a key
        # "action_dist"
        return ["action_dist"]


if __name__ == "__main__":
    ray.init(local_mode=True)
    # config = (
    #     PPOConfig()
    #     .framework("torch")
    #     .rollouts(num_rollout_workers=0)
    #     .resources(num_gpus=0)
    #     .environment(env=NegotiationEnv, env_config={"num_agents": 2})
    #     .rl_module(
    #         _enable_rl_module_api=True,
    #         rl_module_spec=MultiAgentRLModuleSpec(
    #             module_specs=SingleAgentRLModuleSpec(module_class=DiscreteBCTorchModule)
    #         ),
    #     )
    #     .training(model={"fcnet_hiddens": [32, 32]}, _enable_learner_api=True)
    # )
    config = (
        PPOConfig()
        .framework("torch")
        .rollouts(num_rollout_workers=0)
        .resources(num_gpus=0)
        .environment(env=NegotiationEnv, env_config={"num_agents": 2})
        .rl_module(_enable_rl_module_api=True)
        .training(_enable_learner_api=True)
    )
    print(config)
    algo = config.build()
    print(algo)
    print(algo.train())
    ray.shutdown()
