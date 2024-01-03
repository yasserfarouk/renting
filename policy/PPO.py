from typing import Any, Mapping

import torch
import torch.nn.functional as F
from ray.rllib.algorithms.ppo.ppo_rl_module import PPORLModule
from ray.rllib.core.models.specs.specs_dict import SpecDict
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.models.torch.torch_distributions import (
    TorchCategorical,
    TorchMultiCategorical,
    TorchMultiDistribution,
)
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.nested_dict import NestedDict
from torch import Tensor, nn


class BaseModel(TorchRLModule, PPORLModule):
    framework: str = "torch"

    def get_initial_state(self) -> dict:
        return {}
    
    def _forward_inference(self, batch: NestedDict) -> Mapping[str, Any]:
        with torch.no_grad():
            return self._forward_train(batch)

    def _forward_exploration(self, batch: NestedDict) -> Mapping[str, Any]:
        with torch.no_grad():
            return self._forward_train(batch)

    def input_specs_inference(self) -> SpecDict:
        return self.input_specs_exploration()

    def output_specs_inference(self) -> SpecDict:
        return [SampleBatch.ACTION_DIST_INPUTS]

    def input_specs_exploration(self):
        return [SampleBatch.OBS]

    def output_specs_exploration(self) -> SpecDict:
        return [
            SampleBatch.VF_PREDS,
            SampleBatch.ACTION_DIST_INPUTS,
        ]

    def input_specs_train(self) -> SpecDict:
        return self.input_specs_exploration()

    def output_specs_train(self) -> SpecDict:
        return [
            SampleBatch.VF_PREDS,
            SampleBatch.ACTION_DIST_INPUTS,
        ]



class GraphToGraph(BaseModel):

    def setup(self):
        self.val_obj = nn.Linear(6, 6)
        self.obj_head = nn.Linear(8, 8)
        self.head_obj = nn.Linear(14, 6)  # prev + output val_obj
        self.obj_val = nn.Linear(10, 1)  # prev + value features

        # self.encoder = nn.Linear(32,2)
        # self.pi = torch.nn.Linear(2,2)
        self.vf = torch.nn.Linear(8, 1)

        self.action_dist_cls = TorchMultiDistribution

    @override(RLModule)
    def _forward_train(self, batch: NestedDict) -> Mapping[str, Any]:
        head_node: Tensor = batch["obs"]["head_node"]
        objective_nodes: Tensor = batch["obs"]["objective_nodes"]
        value_nodes: Tensor = batch["obs"]["value_nodes"]
        value_nodes_mask: Tensor = batch["obs"]["value_nodes_mask"][:1]

        max_num_values = value_nodes.shape[2]
        num_objectives = objective_nodes.shape[1]

        objective_nodes_expand = objective_nodes.unsqueeze(2).expand(
            -1, -1, max_num_values, -1
        )
        h1 = torch.cat([value_nodes, objective_nodes_expand], 3)
        h2 = self.val_obj(h1)
        h3 = F.relu(h2 * value_nodes_mask.unsqueeze(3))
        h_objective_nodes = torch.sum(h3, 2)

        head_node_expand = head_node.unsqueeze(1).expand(-1, num_objectives, -1)
        h5 = self.obj_head(torch.cat([h_objective_nodes, head_node_expand], 2))
        h_head = torch.sum(F.relu(h5), 1)
        vf_out = self.vf(h_head).squeeze(-1)

        h_head_expand = h_head.unsqueeze(1).expand(-1, num_objectives, -1)
        h6 = self.head_obj(torch.cat([h_objective_nodes, h_head_expand], 2))
        h7 = F.relu(h6)

        h8 = h7.unsqueeze(2).expand(-1, -1, max_num_values, -1)
        h9 = torch.cat([value_nodes, h8], 3)
        h10 = self.obj_val(h9)
        h11 = F.relu(h10 * value_nodes_mask.unsqueeze(3))

        # action_dist = torch.distributions.Categorical(logits=h11)
        action_dist_inputs = torch.masked_select(
            h11.squeeze(), value_nodes_mask.squeeze()
        )

        logits_per_row = int(len(action_dist_inputs) / len(batch))
        action_dist_inputs = action_dist_inputs.reshape((len(batch), logits_per_row))

        input_lens = tuple(*torch.sum(value_nodes_mask, -1).tolist())
        categoricals = [
            TorchCategorical(logits=logits)
            for logits in action_dist_inputs.split(input_lens, dim=-1)
        ]
        action_dist = TorchMultiCategorical(categoricals=categoricals)

        test = action_dist.sample()

        output = {"vf_preds": vf_out, "action_dist": action_dist}
        # test = torch.distributions.Categorical(logits=action_logits)

        # {"action_dist": torch.distributions.Categorical(logits=action_logits)}
        return output


class GraphToFixed(TorchRLModule, PPORLModule):
    framework: str = "torch"

    def setup(self):
        action_space = self.config.action_space
        logit_lens = [int(action_space["accept"].n), int(sum(action_space["outcome"].nvec))]
        pi_output_layer_dim = sum(logit_lens)
        self.num_used_agents = self.config.model_config_dict["num_used_agents"]

        match self.config.model_config_dict["pooling_op"]:
            case "max":
                self.pooling_op = lambda x, y, z: torch.max(x, y)[0]
            case "mean":
                self.pooling_op = lambda x, y, z: (torch.sum(x, y) / z)
            case "sum":
                self.pooling_op = lambda x, y, z: torch.sum(x, y)
            case _:
                raise ValueError(f"Pooling op {self.config.model_config_dict['pooling_op']} not supported")

        self.val_obj = nn.Linear(6, 30)  # value features + objective features
        self.obj_head = nn.Linear(32, 32)  # prev + objective features

        self.encoder = nn.Linear(32 + self.num_used_agents, 32)
        self.pi = nn.Linear(32, pi_output_layer_dim)
        self.vf = nn.Linear(32, 1)


        
        child_distribution_cls_struct = {
            "accept": TorchCategorical,
            "outcome": TorchMultiCategorical.get_partial_dist_cls(space=action_space["outcome"], input_lens=list(action_space["outcome"].nvec))
        }
        self.action_dist_cls = TorchMultiDistribution.get_partial_dist_cls(
            space=action_space,
            child_distribution_cls_struct=child_distribution_cls_struct,
            input_lens=logit_lens,
        )

    @override(PPORLModule)
    def get_initial_state(self) -> dict:
        return {}

    @override(RLModule)
    def _forward_train(self, batch: NestedDict) -> Mapping[str, Any]:
        head_node: Tensor = batch["obs"]["head_node"]
        objective_nodes: Tensor = batch["obs"]["objective_nodes"]
        value_nodes: Tensor = batch["obs"]["value_nodes"]
        value_nodes_mask: Tensor = batch["obs"]["value_nodes_mask"]
        opponent_encoding: Tensor = F.one_hot(batch["obs"]["opponent_encoding"], self.num_used_agents)
        accept_mask: Tensor = batch["obs"]["accept_mask"]

        max_num_values = value_nodes.shape[2]
        num_objectives = objective_nodes.shape[1]

        objective_nodes_expand = objective_nodes.unsqueeze(2).expand(
            -1, -1, max_num_values, -1
        )
        h1 = torch.cat([value_nodes, objective_nodes_expand], 3)
        h2 = self.val_obj(h1)
        h3 = F.relu(h2) * value_nodes_mask.unsqueeze(3)

        # h_objective_nodes = torch.sum(h3, 2)
        h_objective_nodes = self.pooling_op(h3, 2, objective_nodes[:, :, :1])
        # h_objective_nodes = torch.sum(h3, 2) / (objective_nodes[:, :, :1] + 1)

        head_node_expand = head_node.unsqueeze(1).expand(-1, num_objectives, -1)
        h4 = self.obj_head(torch.cat([h_objective_nodes, head_node_expand], 2))
        h5 = F.relu(h4)
        
        # h_head = torch.sum(h5, 1)
        # h_head, _ = torch.max(h5, 1)
        h_head = self.pooling_op(h5, 1, head_node[:, :1])
        # h_head = torch.sum(h5, 1) / (head_node[:, :1] + 1)
    
        h_head = torch.cat((h_head, opponent_encoding), dim=-1)
        h_head = F.relu(self.encoder(h_head))

        vf_out = self.vf(h_head).squeeze(-1)
        action_logits = self.pi(h_head)

        accept_inf_mask = torch.max(torch.log(accept_mask), torch.Tensor([torch.finfo(torch.float32).min]))
        action_logits[:, :2] += accept_inf_mask

        return {"vf_preds": vf_out, "action_dist_inputs": action_logits}

    @override(RLModule)
    def _forward_inference(self, batch: NestedDict) -> Mapping[str, Any]:
        with torch.no_grad():
            return self._forward_train(batch)

    @override(RLModule)
    def _forward_exploration(self, batch: NestedDict) -> Mapping[str, Any]:
        with torch.no_grad():
            return self._forward_train(batch)

    @override(RLModule)
    def input_specs_inference(self) -> SpecDict:
        return self.input_specs_exploration()

    @override(RLModule)
    def output_specs_inference(self) -> SpecDict:
        return [SampleBatch.ACTION_DIST_INPUTS]

    @override(RLModule)
    def input_specs_exploration(self):
        return [SampleBatch.OBS]

    @override(RLModule)
    def output_specs_exploration(self) -> SpecDict:
        return [
            SampleBatch.VF_PREDS,
            SampleBatch.ACTION_DIST_INPUTS,
        ]

    @override(RLModule)
    def input_specs_train(self) -> SpecDict:
        return self.input_specs_exploration()

    @override(RLModule)
    def output_specs_train(self) -> SpecDict:
        return [
            SampleBatch.VF_PREDS,
            SampleBatch.ACTION_DIST_INPUTS,
        ]
