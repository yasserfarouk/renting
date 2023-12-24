from typing import Any, Mapping

import gymnasium as gym
import numpy as np
import torch
from ray.rllib.algorithms.ppo.ppo import PPOConfig
from ray.rllib.algorithms.ppo.ppo_rl_module import PPORLModule
from ray.rllib.core.models.base import ACTOR, CRITIC, ENCODER_OUT, STATE_OUT
from ray.rllib.core.models.configs import ActorCriticEncoderConfig, MLPHeadConfig
from ray.rllib.core.rl_module.rl_module import RLModule, SingleAgentRLModuleSpec
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.examples.env.random_env import RandomEnv
from ray.rllib.models.torch.torch_distributions import TorchCategorical
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.nested_dict import NestedDict
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class MyFixedPPO(TorchRLModule, PPORLModule):
    framework: str = "torch"

    def setup(self):
        self.encoder = torch.nn.Linear(32,2)
        self.pi = torch.nn.Linear(2,2)
        self.vf = torch.nn.Linear(2,2)


    @override(RLModule)
    def _forward_train(self, batch: NestedDict) -> Mapping[str, Any]:
        output = {}

        # Shared encoder
        encoder_outs = self.encoder(batch["obs"])
        if STATE_OUT in encoder_outs:
            output["state_out"] = encoder_outs[STATE_OUT]

        # Value head
        vf_out = self.vf(encoder_outs[ENCODER_OUT][CRITIC])
        output["vf_preds"] = vf_out.squeeze(-1)

        # Policy head
        action_logits = self.pi(encoder_outs[ENCODER_OUT][ACTOR])
        output["action_dist_inputs"] = action_logits

        # {"action_dist": torch.distributions.Categorical(logits=action_logits)}
        return output

    @override(RLModule)
    def _forward_inference(self, batch: NestedDict) -> Mapping[str, Any]:
        with torch.no_grad():
            return self._forward_train(batch)

    @override(RLModule)
    def _forward_exploration(self, batch: NestedDict) -> Mapping[str, Any]:
        with torch.no_grad():
            return self._forward_train(batch)

