from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces
from ray.rllib.algorithms.ppo.ppo_rl_module import PPORLModule
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.models.torch.torch_distributions import (
    TorchCategorical,
    TorchMultiCategorical,
    TorchMultiDistribution,
)
from ray.rllib.utils.annotations import override
from ray.rllib.utils.nested_dict import NestedDict
from torch import Tensor, nn
from torch.distributions import Categorical
from torch.distributions.distribution import Distribution
from torch.distributions.kl import kl_divergence
from torch_geometric.nn import GAT, GATv2Conv
from torch_scatter import scatter, scatter_softmax
from torch_geometric.data import Data, Batch

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


class GraphToGraph(BaseModel):

    def setup(self):
        action_space = self.config.action_space
        self.logit_lens = [int(action_space["accept"].n), int(sum(action_space["outcome"].nvec))]
        self.num_used_agents = self.config.model_config_dict["num_used_agents"]
        self.use_opponent_encoding = self.config.model_config_dict["use_opponent_encoding"]

        self.value_nodes_mask = torch.zeros((1, len(action_space["outcome"].nvec), max(action_space["outcome"].nvec)), dtype=torch.bool)
        for i, n in enumerate(action_space["outcome"].nvec):
            self.value_nodes_mask[0, i, :n] = True

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
        self.obj_head = nn.Linear(32, 64)  # prev + objective features
        if self.use_opponent_encoding:
            self.head_encoder = nn.Linear(64 + self.num_used_agents, 64)
        else:
            self.head_encoder = nn.Linear(64, 64)
        self.head_obj = nn.Linear(94, 64)  # prev + output val_obj
        self.obj_val = nn.Linear(68, 1)  # prev + value features

        self.accept_head = nn.Linear(64, 2)
        self.vf = torch.nn.Linear(64, 1)


        
        child_distribution_cls_struct = {
            "accept": TorchCategorical,
            "outcome": TorchMultiCategorical.get_partial_dist_cls(space=action_space["outcome"], input_lens=list(action_space["outcome"].nvec))
        }
        self.action_dist_cls = TorchMultiDistribution.get_partial_dist_cls(
            space=action_space,
            child_distribution_cls_struct=child_distribution_cls_struct,
            input_lens=self.logit_lens,
        )

    @override(RLModule)
    def _forward_train(self, batch: NestedDict) -> Mapping[str, Any]:
        head_node: Tensor = batch["obs"]["head_node"]
        objective_nodes: Tensor = batch["obs"]["objective_nodes"]
        value_nodes: Tensor = batch["obs"]["value_nodes"]
        # value_nodes_mask: Tensor = batch["obs"]["value_nodes_mask"]
        opponent_encoding: Tensor = F.one_hot(batch["obs"]["opponent_encoding"], self.num_used_agents)
        accept_mask: Tensor = batch["obs"]["accept_mask"]

        max_num_values = value_nodes.shape[2]
        num_objectives = objective_nodes.shape[1]

        objective_nodes_expand = objective_nodes.unsqueeze(2).expand(
            -1, -1, max_num_values, -1
        )
        h1 = torch.cat([value_nodes, objective_nodes_expand], 3)
        h2 = self.val_obj(h1)
        h3 = F.relu(h2) * self.value_nodes_mask.unsqueeze(3)

        h_objective_nodes = self.pooling_op(h3, 2, objective_nodes[:, :, :1])

        head_node_expand = head_node.unsqueeze(1).expand(-1, num_objectives, -1)
        h4 = self.obj_head(torch.cat([h_objective_nodes, head_node_expand], 2))
        h5 = F.relu(h4)

        h_head = self.pooling_op(h5, 1, head_node[:, :1])
        if self.use_opponent_encoding:
            h_head = torch.cat((h_head, opponent_encoding), dim=-1)
        h_head = F.relu(self.head_encoder(h_head))

        h_head_expand = h_head.unsqueeze(1).expand(-1, num_objectives, -1)
        h6 = self.head_obj(torch.cat([h_objective_nodes, h_head_expand], 2))
        h7 = F.relu(h6)

        h8 = h7.unsqueeze(2).expand(-1, -1, max_num_values, -1)
        h9 = torch.cat([value_nodes, h8], 3)
        h10 = self.obj_val(h9)
        h11 = F.relu(h10)

        # action_logits = torch.masked_select(h11.squeeze(), value_nodes_mask[:1])
        accept_inf_mask = torch.max(torch.log(accept_mask), torch.Tensor([torch.finfo(torch.float32).min]))
        accept_action_logits = F.relu(self.accept_head(h_head)) + accept_inf_mask
        offer_action_logits = h11.squeeze(-1)[:, self.value_nodes_mask[0]]


        action_logits = torch.cat((accept_action_logits, offer_action_logits), dim=-1)

        vf_out = self.vf(h_head).squeeze(-1)


        return {"vf_preds": vf_out, "action_dist_inputs": action_logits}


class GraphToGraph2(BaseModel):

    def setup(self):
        self.num_used_agents = self.config.model_config_dict["num_used_agents"]
        self.use_opponent_encoding = self.config.model_config_dict["use_opponent_encoding"]
        self.pooling_op = self.config.model_config_dict["pooling_op"]


        match self.pooling_op:
            case "max":
                self.head_pool = lambda x: torch.max(x, 1)[0]
            case "mean":
                self.head_pool = lambda x: torch.mean(x, 1)
            case "sum":
                self.head_pool = lambda x: torch.sum(x, 1)
            case "min":
                self.head_pool = lambda x: torch.min(x, 1)[0]
            case "mul":
                self.head_pool = lambda x: torch.prod(x, 1)
            case _:
                raise ValueError(f"Pooling op {self.pooling_op} not supported")

        self.val_obj = nn.Linear(6, 30)  # value features + objective features
        self.obj_head = nn.Linear(32, 64)  # prev + objective features
        if self.use_opponent_encoding:
            self.head_encoder = nn.Linear(64 + self.num_used_agents, 64)
        else:
            self.head_encoder = nn.Linear(64, 64)
        self.head_obj = nn.Linear(94, 64)  # prev + output val_obj
        self.obj_val = nn.Linear(68, 1)  # prev + value features

        self.accept_head = nn.Linear(64, 2)
        self.vf = torch.nn.Linear(64, 1)

        # def get_action_dist_cls(self):
        action_space = self.config.action_space
        logit_lens = [int(action_space["accept"].n), int(sum(action_space["outcome"].nvec))]
        child_distribution_cls_struct = {
            "accept": TorchCategorical,
            "outcome": TorchMultiCategorical.get_partial_dist_cls(space=action_space["outcome"], input_lens=list(action_space["outcome"].nvec))
        }
        self.action_dist_cls = TorchMultiDistribution.get_partial_dist_cls(
            space=action_space,
            child_distribution_cls_struct=child_distribution_cls_struct,
            input_lens=logit_lens,
        )
        # return action_dist_cls

    def get_action_dist_cls(self):
        return self.action_dist_cls

    def get_train_action_dist_cls(self):
        return self.get_action_dist_cls()

    def get_exploration_action_dist_cls(self):
        return self.get_action_dist_cls()

    def get_inference_action_dist_cls(self):
        return self.get_action_dist_cls()

    @override(RLModule)
    def _forward_train(self, batch: NestedDict) -> Mapping[str, Any]:
        head_node: Tensor = batch["obs"]["head_node"]
        objective_nodes: Tensor = batch["obs"]["objective_nodes"]
        value_nodes: Tensor = batch["obs"]["value_nodes"]
        value_adjacency: Tensor = batch["obs"]["value_adjacency"]
        opponent_encoding: Tensor = F.one_hot(batch["obs"]["opponent_encoding"], self.num_used_agents)
        accept_mask: Tensor = batch["obs"]["accept_mask"]

        self.logit_lens = value_adjacency[0].unique(return_counts=True)[1].tolist()

        # value nodes to objective nodes
        objective_nodes_expand = objective_nodes.gather(1, value_adjacency.unsqueeze(2).expand(-1, -1, objective_nodes.shape[2]))
        h1 = torch.cat((value_nodes, objective_nodes_expand), 2)
        h2 = F.relu(self.val_obj(h1))
        h_objective_nodes_fow = scatter(h2, value_adjacency, dim=1, reduce=self.pooling_op)

        # objective nodes to head node
        num_objectives = objective_nodes.shape[1]
        head_node_expand = head_node.unsqueeze(1).expand(-1, num_objectives, -1)
        h3 = torch.cat([h_objective_nodes_fow, head_node_expand], 2)
        h4 = F.relu(self.obj_head(h3))
        h_head = self.head_pool(h4)
        
        # head node to head node
        if self.use_opponent_encoding:
            h_head = torch.cat((h_head, opponent_encoding), dim=-1)
        h_head = F.relu(self.head_encoder(h_head))

        # head node to objective nodes
        h_head_expand = h_head.unsqueeze(1).expand(-1, num_objectives, -1)
        h5 = torch.cat([h_objective_nodes_fow, h_head_expand], 2)
        h_objective_nodes_back = F.relu(self.head_obj(h5))

        # objective nodes to value nodes
        h8 = h_objective_nodes_back.gather(1, value_adjacency.unsqueeze(2).expand(-1, -1, h_objective_nodes_back.shape[2]))
        h9 = torch.cat([value_nodes, h8], 2)
        offer_action_logits = F.relu(self.obj_val(h9)).squeeze(-1)

        # head node to accept action
        accept_inf_mask = torch.max(torch.log(accept_mask), torch.Tensor([torch.finfo(torch.float32).min]))
        accept_action_logits = F.relu(self.accept_head(h_head)) + accept_inf_mask

        # head node to value function
        vf_out = self.vf(h_head).squeeze(-1)

        # gather action logits
        action_logits = torch.cat((accept_action_logits, offer_action_logits), dim=-1)

        # return {"vf_preds": vf_out, "action_dist": {"accept": None, "outcome": None}}
        return {"vf_preds": vf_out, "action_dist_inputs": action_logits}
    

class AttentionGraphToGraph(BaseModel):

    def setup(self):
        self.num_used_agents = self.config.model_config_dict["num_used_agents"]
        self.use_opponent_encoding = self.config.model_config_dict["use_opponent_encoding"]
        self.pooling_op = self.config.model_config_dict["pooling_op"]

        match self.pooling_op:
            case "max":
                self.head_pool = lambda x: torch.max(x, 1)[0]
            case "mean":
                self.head_pool = lambda x: torch.mean(x, 1)
            case "sum":
                self.head_pool = lambda x: torch.sum(x, 1)
            case "min":
                self.head_pool = lambda x: torch.min(x, 1)[0]
            case "mul":
                self.head_pool = lambda x: torch.prod(x, 1)
            case _:
                raise ValueError(f"Pooling op {self.pooling_op} not supported")


        head_features = 2 + self.num_used_agents if self.use_opponent_encoding else 2

        num_h_obj = 32
        num_h_head = 32

        self.f_a_val_obj = nn.Linear(6, 1)
        self.f_enc_val = nn.Linear(4, 16)
        self.f_enc_val_obj = nn.Linear(16 + 2, num_h_obj)

        self.f_a_obj_head = nn.Linear(num_h_obj + head_features, 1)
        self.f_enc_obj = nn.Linear(num_h_obj, num_h_obj)
        self.f_enc_obj_head = nn.Linear(num_h_obj + head_features, num_h_head)

        self.b_a_head_obj = nn.Linear(num_h_head + num_h_obj, 1)
        self.b_enc_head = nn.Linear(num_h_head, num_h_obj)
        self.b_enc_head_obj = nn.Linear(num_h_head + num_h_obj, num_h_obj)


        self.b_a_obj_val = nn.Linear(num_h_obj + 4, 1)
        self.b_enc_obj = nn.Linear(num_h_obj, num_h_obj)
        self.b_enc_obj_val = nn.Linear(num_h_obj + 4, 1)

        self.accept_head = nn.Linear(num_h_head, 2)
        self.vf = torch.nn.Linear(num_h_head, 1)

        action_space = self.config.action_space
        logit_lens = [int(action_space["accept"].n), int(sum(action_space["outcome"].nvec))]
        child_distribution_cls_struct = {
            "accept": TorchCategorical,
            "outcome": TorchMultiCategorical.get_partial_dist_cls(space=action_space["outcome"], input_lens=list(action_space["outcome"].nvec))
        }
        self.action_dist_cls = TorchMultiDistribution.get_partial_dist_cls(
            space=action_space,
            child_distribution_cls_struct=child_distribution_cls_struct,
            input_lens=logit_lens,
        )

    @override(RLModule)
    def _forward_train(self, batch: NestedDict) -> Mapping[str, Any]:
        head_node: Tensor = batch["obs"]["head_node"]
        objective_nodes: Tensor = batch["obs"]["objective_nodes"]
        value_nodes: Tensor = batch["obs"]["value_nodes"]
        value_adjacency: Tensor = batch["obs"]["value_adjacency"]
        opponent_encoding: Tensor = F.one_hot(batch["obs"]["opponent_encoding"], self.num_used_agents)
        accept_mask: Tensor = batch["obs"]["accept_mask"]

        if self.use_opponent_encoding:
            head_node = torch.cat((head_node, opponent_encoding), dim=-1)

        # value nodes to objective nodes
        objective_nodes_expand = objective_nodes.gather(1, value_adjacency.unsqueeze(-1).expand(-1, -1, objective_nodes.shape[2]))
        h1 = torch.cat((objective_nodes_expand, value_nodes), dim=-1)

        att1 = scatter_softmax(self.f_a_val_obj(h1), value_adjacency, 1)

        h2 = att1 * F.relu(self.f_enc_val(value_nodes))
        h3 = scatter(h2, value_adjacency, dim=1, reduce="sum")

        h4 = torch.cat((objective_nodes, h3), dim=-1)
        h_objective_nodes1 = F.relu(self.f_enc_val_obj(h4))

        # objective nodes to head node
        num_objectives = objective_nodes.shape[1]
        head_node_expand = head_node.unsqueeze(1).expand(-1, num_objectives, -1)
        h5 = torch.cat([h_objective_nodes1, head_node_expand], 2)

        att2 = F.softmax(self.f_a_obj_head(h5), 1)

        h6 = att2 * F.relu(self.f_enc_obj(h_objective_nodes1))
        h7 = torch.sum(h6, 1)
        h8 = torch.cat([h7, head_node], 1)
        h_head = F.relu(self.f_enc_obj_head(h8))

        # head node to objective nodes
        h10 = F.relu(self.b_enc_head(h_head))
        h10_expand = h10.unsqueeze(1).expand(-1, num_objectives, -1)
        h11 = torch.cat([h10_expand, h_objective_nodes1], 2)
        h_objective_nodes2 = F.relu(self.b_enc_head_obj(h11))

        # objective nodes to value nodes
        h12 = F.relu(self.b_enc_obj(h_objective_nodes2))
        h12_expand = h12.gather(1, value_adjacency.unsqueeze(2).expand(-1, -1, h12.shape[2]))
        h13 = torch.cat([h12_expand, value_nodes], 2)
        offer_action_logits = F.relu(self.b_a_obj_val(h13)).squeeze(-1)

        # head node to accept action
        accept_inf_mask = torch.max(torch.log(accept_mask), torch.Tensor([torch.finfo(torch.float32).min]))
        accept_action_logits = F.relu(self.accept_head(h_head)) + accept_inf_mask

        # head node to value function
        vf_out = self.vf(h_head).squeeze(-1)

        # gather action logits
        action_logits = torch.cat((accept_action_logits, offer_action_logits), dim=-1)

        # return {"vf_preds": vf_out, "action_dist": {"accept": None, "outcome": None}}
        return {"vf_preds": vf_out, "action_dist_inputs": action_logits}
    
class AttentionGraphToGraph2(BaseModel):

    def setup(self):
        self.num_used_agents = self.config.model_config_dict["num_used_agents"]
        self.use_opponent_encoding = self.config.model_config_dict["use_opponent_encoding"]
        self.pooling_op = self.config.model_config_dict["pooling_op"]


        head_features = 2 + self.num_used_agents if self.use_opponent_encoding else 2

        self.val_obj = GATv2Conv((4, 2), 32)
        self.obj_head = GATv2Conv((32, head_features), 32)
        self.head_obj = GATv2Conv((32, 32), 32)
        self.obj_val = GATv2Conv((32, 4), 1)

        self.accept_head = nn.Linear(32, 2)
        self.vf = nn.Linear(32, 1)

        action_space = self.config.action_space
        logit_lens = [int(action_space["accept"].n), int(sum(action_space["outcome"].nvec))]
        child_distribution_cls_struct = {
            "accept": TorchCategorical,
            "outcome": TorchMultiCategorical.get_partial_dist_cls(space=action_space["outcome"], input_lens=list(action_space["outcome"].nvec))
        }
        self.action_dist_cls = TorchMultiDistribution.get_partial_dist_cls(
            space=action_space,
            child_distribution_cls_struct=child_distribution_cls_struct,
            input_lens=logit_lens,
        )

    @override(RLModule)
    def _forward_train(self, batch: NestedDict) -> Mapping[str, Any]:
        head_node: Tensor = batch["obs"]["head_node"]
        objective_nodes: Tensor = batch["obs"]["objective_nodes"]
        value_nodes: Tensor = batch["obs"]["value_nodes"]
        edge_indices_val_obj: Tensor = batch["obs"]["edge_indices_val_obj"]
        edge_indices_obj_head: Tensor = batch["obs"]["edge_indices_obj_head"]
        opponent_encoding: Tensor = F.one_hot(batch["obs"]["opponent_encoding"], self.num_used_agents)
        accept_mask: Tensor = batch["obs"]["accept_mask"]

        if self.use_opponent_encoding:
            head_node = torch.cat((head_node, opponent_encoding), dim=-1)

        offer_action_logits = []
        h_head_out = []
        for vn, ob, hn, eivo, eioh in zip(value_nodes, objective_nodes, head_node, edge_indices_val_obj, edge_indices_obj_head):
            h_obj = F.relu(self.val_obj((vn, ob), eivo))
            h_head = F.relu(self.obj_head((h_obj, hn), eioh))
            h_obj = F.relu(self.head_obj((h_head, h_obj), eioh.flip(0)))
            offer_action_logits.append(self.obj_val((h_obj, vn), eivo.flip(0)).squeeze(-1).unsqueeze(0))
            h_head_out.append(h_head)

        offer_action_logits = torch.cat(offer_action_logits, dim=0)
        h_head_out = torch.cat(h_head_out, dim=0)
        # head node to accept action
        accept_inf_mask = torch.max(torch.log(accept_mask), torch.Tensor([torch.finfo(torch.float32).min]))
        accept_action_logits = F.relu(self.accept_head(h_head_out)) + accept_inf_mask

        # head node to value function
        vf_out = self.vf(h_head_out).squeeze(-1)

        # gather action logits
        action_logits = torch.cat((accept_action_logits, offer_action_logits), dim=-1)

        # return {"vf_preds": vf_out, "action_dist": {"accept": None, "outcome": None}}
        return {"vf_preds": vf_out, "action_dist_inputs": action_logits}
    

class GraphToGraphLargeFixedAction(BaseModel):

    def setup(self):
        self.num_used_agents = self.config.model_config_dict["num_used_agents"]
        self.use_opponent_encoding = self.config.model_config_dict["use_opponent_encoding"]
        self.pooling_op = self.config.model_config_dict["pooling_op"]


        match self.pooling_op:
            case "max":
                self.head_pool = lambda x: torch.max(x, 1)[0]
            case "mean":
                self.head_pool = lambda x: torch.mean(x, 1)
            case "sum":
                self.head_pool = lambda x: torch.sum(x, 1)
            case "min":
                self.head_pool = lambda x: torch.min(x, 1)[0]
            case "mul":
                self.head_pool = lambda x: torch.prod(x, 1)
            case _:
                raise ValueError(f"Pooling op {self.pooling_op} not supported")

        self.val_obj = nn.Linear(6, 30)  # value features + objective features
        self.obj_head = nn.Linear(32, 64)  # prev + objective features
        if self.use_opponent_encoding:
            self.head_encoder = nn.Linear(64 + self.num_used_agents, 64)
        else:
            self.head_encoder = nn.Linear(64, 64)
        self.head_obj = nn.Linear(94, 64)  # prev + output val_obj
        self.obj_val = nn.Linear(68, 1)  # prev + value features

        self.accept_head = nn.Linear(64, 2)
        self.vf = torch.nn.Linear(64, 1)

        action_space = self.config.action_space
        logit_lens = [int(action_space["accept"].n), int(sum(action_space["outcome"].nvec))]
        child_distribution_cls_struct = {
            "accept": TorchCategorical,
            "outcome": TorchMultiCategorical.get_partial_dist_cls(space=action_space["outcome"], input_lens=list(action_space["outcome"].nvec))
        }
        self.action_dist_cls = TorchMultiDistribution.get_partial_dist_cls(
            space=action_space,
            child_distribution_cls_struct=child_distribution_cls_struct,
            input_lens=logit_lens,
        )

    @override(RLModule)
    def _forward_train(self, batch: NestedDict) -> Mapping[str, Any]:
        head_node: Tensor = batch["obs"]["head_node"]
        objective_nodes: Tensor = batch["obs"]["objective_nodes"]
        value_nodes: Tensor = batch["obs"]["value_nodes"]
        value_adjacency: Tensor = batch["obs"]["value_adjacency"]
        opponent_encoding: Tensor = F.one_hot(batch["obs"]["opponent_encoding"], self.num_used_agents)
        accept_mask: Tensor = batch["obs"]["accept_mask"]

        # value nodes to objective nodes
        objective_nodes_expand = objective_nodes.gather(1, value_adjacency.unsqueeze(2).expand(-1, -1, objective_nodes.shape[2]))
        h1 = torch.cat((value_nodes, objective_nodes_expand), 2)
        h2 = F.relu(self.val_obj(h1))
        h_objective_nodes_fow = scatter(h2, value_adjacency, dim=1, reduce=self.pooling_op)

        # objective nodes to head node
        num_objectives = objective_nodes.shape[1]
        head_node_expand = head_node.unsqueeze(1).expand(-1, num_objectives, -1)
        h3 = torch.cat([h_objective_nodes_fow, head_node_expand], 2)
        h4 = F.relu(self.obj_head(h3))
        h_head = self.head_pool(h4)
        
        # head node to head node
        if self.use_opponent_encoding:
            h_head = torch.cat((h_head, opponent_encoding), dim=-1)
        h_head = F.relu(self.head_encoder(h_head))

        # head node to objective nodes
        h_head_expand = h_head.unsqueeze(1).expand(-1, num_objectives, -1)
        h5 = torch.cat([h_objective_nodes_fow, h_head_expand], 2)
        h_objective_nodes_back = F.relu(self.head_obj(h5))

        # objective nodes to value nodes
        h8 = h_objective_nodes_back.gather(1, value_adjacency.unsqueeze(2).expand(-1, -1, h_objective_nodes_back.shape[2]))
        h9 = torch.cat([value_nodes, h8], 2)
        offer_action_logits = F.relu(self.obj_val(h9)).squeeze(-1)

        # head node to accept action
        accept_inf_mask = torch.max(torch.log(accept_mask), torch.Tensor([torch.finfo(torch.float32).min]))
        accept_action_logits = F.relu(self.accept_head(h_head)) + accept_inf_mask

        # head node to value function
        vf_out = self.vf(h_head).squeeze(-1)

        # gather action logits
        action_logits = torch.cat((accept_action_logits, offer_action_logits), dim=-1)

        # return {"vf_preds": vf_out, "action_dist": {"accept": None, "outcome": None}}
        return {"vf_preds": vf_out, "action_dist_inputs": action_logits}

class Test(BaseModel):

    def setup(self):
        self.num_used_agents = self.config.model_config_dict["num_used_agents"]
        self.use_opponent_encoding = self.config.model_config_dict["use_opponent_encoding"]
        self.pooling_op = self.config.model_config_dict["pooling_op"]


        match self.pooling_op:
            case "max":
                self.head_pool = lambda x: torch.max(x, 1)[0]
            case "mean":
                self.head_pool = lambda x: torch.mean(x, 1)
            case "sum":
                self.head_pool = lambda x: torch.sum(x, 1)
            case "min":
                self.head_pool = lambda x: torch.min(x, 1)[0]
            case "mul":
                self.head_pool = lambda x: torch.prod(x, 1)
            case _:
                raise ValueError(f"Pooling op {self.pooling_op} not supported")

        self.val_obj = nn.Linear(6, 30)  # value features + objective features
        self.obj_head = nn.Linear(32, 64)  # prev + objective features
        if self.use_opponent_encoding:
            self.head_encoder = nn.Linear(64 + self.num_used_agents, 64)
        else:
            self.head_encoder = nn.Linear(64, 64)
        self.head_obj = nn.Linear(94, 64)  # prev + output val_obj
        self.obj_val = nn.Linear(68, 1)  # prev + value features

        self.accept_head = nn.Linear(64, 2)
        self.vf = torch.nn.Linear(64, 1)

        self.action_dist_cls = None
        # action_space = self.config.action_space
        # logit_lens = [int(action_space["accept"].n), int(sum(action_space["outcome"].nvec))]

    def get_action_dist_cls(self):
        return TorchMultiCategorical.get_partial_dist_cls(input_lens=self.logit_lens)

    def get_train_action_dist_cls(self):
        return self.get_action_dist_cls()

    def get_exploration_action_dist_cls(self):
        return self.get_action_dist_cls()

    def get_inference_action_dist_cls(self):
        return self.get_action_dist_cls()

    @override(RLModule)
    def _forward_train(self, batch: NestedDict) -> Mapping[str, Any]:
        head_node: Tensor = batch["obs"]["head_node"]
        objective_nodes: Tensor = batch["obs"]["objective_nodes"]
        value_nodes: Tensor = batch["obs"]["value_nodes"]
        value_adjacency: Tensor = batch["obs"]["value_adjacency"]
        opponent_encoding: Tensor = F.one_hot(batch["obs"]["opponent_encoding"], self.num_used_agents)
        accept_mask: Tensor = batch["obs"]["accept_mask"]

        self.logit_lens = [2] + value_adjacency[0].unique(return_counts=True)[1].tolist()

        # value nodes to objective nodes
        objective_nodes_expand = objective_nodes.gather(1, value_adjacency.unsqueeze(2).expand(-1, -1, objective_nodes.shape[2]))
        h1 = torch.cat((value_nodes, objective_nodes_expand), 2)
        h2 = F.relu(self.val_obj(h1))
        h_objective_nodes_fow = scatter(h2, value_adjacency, dim=1, reduce=self.pooling_op)

        # objective nodes to head node
        num_objectives = objective_nodes.shape[1]
        head_node_expand = head_node.unsqueeze(1).expand(-1, num_objectives, -1)
        h3 = torch.cat([h_objective_nodes_fow, head_node_expand], 2)
        h4 = F.relu(self.obj_head(h3))
        h_head = self.head_pool(h4)
        
        # head node to head node
        if self.use_opponent_encoding:
            h_head = torch.cat((h_head, opponent_encoding), dim=-1)
        h_head = F.relu(self.head_encoder(h_head))

        # head node to objective nodes
        h_head_expand = h_head.unsqueeze(1).expand(-1, num_objectives, -1)
        h5 = torch.cat([h_objective_nodes_fow, h_head_expand], 2)
        h_objective_nodes_back = F.relu(self.head_obj(h5))

        # objective nodes to value nodes
        h8 = h_objective_nodes_back.gather(1, value_adjacency.unsqueeze(2).expand(-1, -1, h_objective_nodes_back.shape[2]))
        h9 = torch.cat([value_nodes, h8], 2)
        offer_action_logits = F.relu(self.obj_val(h9)).squeeze(-1)

        # head node to accept action
        accept_inf_mask = torch.max(torch.log(accept_mask), torch.Tensor([torch.finfo(torch.float32).min]))
        accept_action_logits = F.relu(self.accept_head(h_head)) + accept_inf_mask

        # head node to value function
        vf_out = self.vf(h_head).squeeze(-1)

        # gather action logits
        action_logits = torch.cat((accept_action_logits, offer_action_logits), dim=-1)

        # return {"vf_preds": vf_out, "action_dist": {"accept": None, "outcome": None}}
        return {"vf_preds": vf_out, "action_dist_inputs": action_logits}

class GraphToFixed(BaseModel):

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


class PureGNN(BaseModel):
    def setup(self):
        action_space = self.config.action_space
        self.logit_lens = [int(action_space["accept"].n), int(sum(action_space["outcome"].nvec))]
        self.num_used_agents = self.config.model_config_dict["num_used_agents"]
        self.use_opponent_encoding = self.config.model_config_dict["use_opponent_encoding"]

        hidden_size = self.config.model_config_dict["hidden_size"]


        if self.use_opponent_encoding:
            self.head_encoder = nn.Linear(2 + self.num_used_agents, hidden_size)
        else:
            self.head_encoder = nn.Linear(2, hidden_size)
        self.objective_encoder = nn.Linear(2, hidden_size)
        self.value_encoder = nn.Linear(4, hidden_size)

        self.gnn_layers = GAT(hidden_size, hidden_size, self.config.model_config_dict["num_gcn_layers"], hidden_size, v2=True)
        # self.skip_connects = [nn.Linear(2 * hidden_size, hidden_size) for _ in range(self.config.model_config_dict["num_gcn_layers"])]

        self.accept_head = nn.Linear(hidden_size, 2)
        self.offer_head = nn.Linear(hidden_size, 1)
        self.vf = torch.nn.Linear(hidden_size, 1)
        
        child_distribution_cls_struct = {
            "accept": TorchCategorical,
            "outcome": TorchMultiCategorical.get_partial_dist_cls(space=action_space["outcome"], input_lens=list(action_space["outcome"].nvec))
        }
        self.action_dist_cls = TorchMultiDistribution.get_partial_dist_cls(
            space=action_space,
            child_distribution_cls_struct=child_distribution_cls_struct,
            input_lens=self.logit_lens,
        )

    @override(RLModule)
    def _forward_train(self, batch: NestedDict) -> Mapping[str, Any]:
        head_node: Tensor = batch["obs"]["head_node"]
        objective_nodes: Tensor = batch["obs"]["objective_nodes"]
        value_nodes: Tensor = batch["obs"]["value_nodes"]
        edge_indices: Tensor = batch["obs"]["edge_indices"]
        opponent_encoding: Tensor = F.one_hot(batch["obs"]["opponent_encoding"], self.num_used_agents)
        accept_mask: Tensor = batch["obs"]["accept_mask"]

        if self.use_opponent_encoding:
            head_node = torch.cat((head_node, opponent_encoding), dim=-1)

        h_head_node = F.relu(self.head_encoder(head_node))
        h_objective_nodes = F.relu(self.objective_encoder(objective_nodes))
        h_value_nodes = F.relu(self.value_encoder(value_nodes))

        h_nodes = torch.cat((h_head_node.unsqueeze(1), h_objective_nodes, h_value_nodes), dim=1)

        # graph_batch = Batch([Data(x, e) for x, e in zip(h_nodes, edge_indices)])
        # for i, (gnn_layer, skip_connect) in enumerate(zip(self.gnn_layers, self.skip_connects)):
        #     h_nodes = gnn_layer(graph_batch.x, graph_batch.edge_index)
        #     h_nodes_out = torch.cat([F.relu(gnn_layer(h, e)).unsqueeze(0) for h, e in zip(h_nodes_in, edge_indices)], dim=0)
        #     h_nodes_out = torch.cat((h_nodes_in, h_nodes_out), dim=-1)
        #     h_nodes_in = F.relu(skip_connect(h_nodes_out))

        h_nodes = torch.cat([F.relu(self.gnn_layers(h, e)).unsqueeze(0) for h, e in zip(h_nodes, edge_indices)], dim=0)
        # for gcn_layer in self.gcn_layers:
        #     h_nodes_out = []
        #     for h, e in zip(h_nodes, edge_indices):
        #         out = gcn_layer(h, e)
        #         h_nodes_out.append(F.relu(out))
        #     h_nodes = torch.cat([F.relu(gcn_layer(h, e)).unsqueeze(0) for h, e in zip(h_nodes, edge_indices)], dim=0)
        #     h_nodes = torch.cat(h_nodes_out, dim=0)


        h_value_nodes_out = h_nodes[:, -value_nodes.shape[1]:, :]
        offer_action_logits = F.relu(self.offer_head(h_value_nodes_out)).squeeze(-1)
        

        accept_inf_mask = torch.max(torch.log(accept_mask), torch.Tensor([torch.finfo(torch.float32).min]))
        accept_action_logits = F.relu(self.accept_head(h_nodes[:, 0, :])) + accept_inf_mask


        action_logits = torch.cat((accept_action_logits, offer_action_logits), dim=-1)

        vf_out = self.vf(h_nodes[:, 0, :]).squeeze(-1)

        return {"vf_preds": vf_out, "action_dist_inputs": action_logits}


class HigaEtAl(BaseModel):
    def setup(self):
        action_space = self.config.action_space
        observation_space = self.config.observation_space
        logit_lens = [int(action_space["accept"].n), int(sum(action_space["outcome"].nvec))]

        self.encoder = nn.Sequential(
            nn.Linear(spaces.flatdim(observation_space), 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )

        self.vf = nn.Sequential(
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.pi = nn.Sequential(
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, sum(logit_lens))
        )


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
        H = self.encoder(batch["obs"])

        vf_out = self.vf(H).squeeze(-1)
        action_logits = self.pi(H)

        return {"vf_preds": vf_out, "action_dist_inputs": action_logits}
    

class FixedToFixed(BaseModel):
    def setup(self):
        action_space = self.config.action_space
        observation_space = self.config.observation_space
        logit_lens = [int(action_space["accept"].n), int(sum(action_space["outcome"].nvec))]

        self.encoder = nn.Sequential(
            nn.Linear(spaces.flatdim(observation_space), 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        self.vf = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.pi = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, sum(logit_lens))
        )


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
        H = self.encoder(batch["obs"])

        vf_out = self.vf(H).squeeze(-1)
        action_logits = self.pi(H)

        return {"vf_preds": vf_out, "action_dist_inputs": action_logits}


class FixedToFixed2(BaseModel):
    def setup(self):
        action_space = self.config.action_space
        observation_space = self.config.observation_space
        logit_lens = [int(action_space["accept"].n), int(sum(action_space["outcome"].nvec))]

        self.encoder = nn.Sequential(
            nn.Linear(spaces.flatdim(observation_space), 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        self.vf = nn.Linear(64, 1)
        self.pi = nn.Linear(64, sum(logit_lens))


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
        H = self.encoder(batch["obs"])

        vf_out = self.vf(H).squeeze(-1)
        action_logits = self.pi(H)

        return {"vf_preds": vf_out, "action_dist_inputs": action_logits}


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class AttentionGraphToGraph3(nn.Module):

    def __init__(self, envs, args):
        super().__init__()
        self.use_opponent_encoding = args.use_opponent_encoding
        self.num_used_agents = envs.single_observation_space["opponent_encoding"].n

        head_features = 2 + self.num_used_agents if args.use_opponent_encoding else 2

        self.val_obj = GATv2Conv((4, 2), args.hidden_size)
        self.obj_head = GATv2Conv((args.hidden_size, head_features), args.hidden_size)
        self.head_obj = GATv2Conv((args.hidden_size, args.hidden_size), args.hidden_size)
        self.obj_val = GATv2Conv((args.hidden_size, 4), 1)

        self.accept_head = layer_init(nn.Linear(args.hidden_size, 2), std=0.01)
        self.vf = layer_init(nn.Linear(args.hidden_size, 1), std=1)

        self.action_nvec = tuple(envs.single_action_space.nvec)

    def get_value(self, batch):
        head_node: Tensor = batch["head_node"]
        objective_nodes: Tensor = batch["objective_nodes"]
        value_nodes: Tensor = batch["value_nodes"]
        edge_indices_val_obj: Tensor = batch["edge_indices_val_obj"]
        edge_indices_obj_head: Tensor = batch["edge_indices_obj_head"]
        opponent_encoding: Tensor = F.one_hot(batch["opponent_encoding"], self.num_used_agents)

        if self.use_opponent_encoding:
            head_node = torch.cat((head_node, opponent_encoding), dim=-1)

        h_head_out = []
        for vn, ob, hn, eivo, eioh in zip(value_nodes, objective_nodes, head_node, edge_indices_val_obj, edge_indices_obj_head):
            h_obj = F.relu(self.val_obj((vn, ob), eivo))
            h_head = F.relu(self.obj_head((h_obj, hn), eioh))
            h_head_out.append(h_head)

        h_head_out = torch.cat(h_head_out, dim=0)
        vf_out = self.vf(h_head_out).squeeze(-1)

        return vf_out

    def get_action_and_value(self, batch, action=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        head_node: Tensor = batch["head_node"]
        objective_nodes: Tensor = batch["objective_nodes"]
        value_nodes: Tensor = batch["value_nodes"]
        edge_indices_val_obj: Tensor = batch["edge_indices_val_obj"]
        edge_indices_obj_head: Tensor = batch["edge_indices_obj_head"]
        opponent_encoding: Tensor = F.one_hot(batch["opponent_encoding"], self.num_used_agents)
        accept_mask: Tensor = batch["accept_mask"]

        if self.use_opponent_encoding:
            head_node = torch.cat((head_node, opponent_encoding), dim=-1)

        offer_action_logits = []
        h_head_out = []
        for vn, ob, hn, eivo, eioh in zip(value_nodes, objective_nodes, head_node, edge_indices_val_obj, edge_indices_obj_head):
            h_obj = F.relu(self.val_obj((vn, ob), eivo))
            h_head = F.relu(self.obj_head((h_obj, hn), eioh))
            h_obj = F.relu(self.head_obj((h_head, h_obj), eioh.flip(0)))
            offer_action_logits.append(self.obj_val((h_obj, vn), eivo.flip(0)).squeeze(-1).unsqueeze(0))
            h_head_out.append(h_head)

        offer_action_logits = torch.cat(offer_action_logits, dim=0)
        h_head_out = torch.cat(h_head_out, dim=0)
        # head node to accept action
        accept_inf_mask = torch.max(torch.log(accept_mask), torch.Tensor([torch.finfo(torch.float32).min]).to("cuda:0" if torch.cuda.is_available() else "cpu"))
        accept_action_logits = self.accept_head(h_head_out) + accept_inf_mask

        # head node to value function
        vf_out = self.vf(h_head_out).squeeze(-1)

        # gather action logits
        action_logits = torch.cat((accept_action_logits, offer_action_logits), dim=-1)

        probs = MultiCategorical(action_logits, self.action_nvec)

        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), vf_out
    

class BipartiteData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index':
            return torch.tensor([[self.x_s.size(0)], [self.x_t.size(0)]])
        return super().__inc__(key, value, *args, **kwargs)
    

class AttentionGraphToGraph4(nn.Module):

    def __init__(self, envs, args):
        super().__init__()
        self.use_opponent_encoding = args.use_opponent_encoding
        self.num_used_agents = envs.single_observation_space["opponent_encoding"].n

        head_features = 2 + self.num_used_agents if args.use_opponent_encoding else 2

        self.val_obj = GATv2Conv((4, 2), args.hidden_size)
        self.obj_head = GATv2Conv((args.hidden_size, head_features), args.hidden_size)
        self.head_obj = GATv2Conv((args.hidden_size, args.hidden_size), args.hidden_size)
        self.obj_val = GATv2Conv((args.hidden_size, 4), 1)

        self.accept_head = layer_init(nn.Linear(args.hidden_size, 2), std=0.01)
        self.vf = layer_init(nn.Linear(args.hidden_size, 1), std=1)

        self.action_nvec = tuple(envs.single_action_space.nvec)

    def bipartite_forward(self, layer: GATv2Conv, b_x_s, b_x_t, b_edge_index) -> Tensor:
        data = [BipartiteData(x_s=x_s, x_t=x_t, edge_index=edge_index) for x_s, x_t, edge_index in zip(b_x_s, b_x_t, b_edge_index)]
        batch =  Batch.from_data_list(data)
        out = layer((batch.x_s, batch.x_t), batch.edge_index)
        return out.view(b_x_t.shape[0], b_x_t.shape[1], -1)


    def get_value(self, batch):
        head_node: Tensor = batch["head_node"]
        objective_nodes: Tensor = batch["objective_nodes"]
        value_nodes: Tensor = batch["value_nodes"]
        edge_indices_val_obj: Tensor = batch["edge_indices_val_obj"]
        edge_indices_obj_head: Tensor = batch["edge_indices_obj_head"]
        opponent_encoding: Tensor = F.one_hot(batch["opponent_encoding"], self.num_used_agents)

        if self.use_opponent_encoding:
            head_node = torch.cat((head_node, opponent_encoding), dim=-1)

        h_obj = F.relu(self.bipartite_forward(self.val_obj, value_nodes, objective_nodes, edge_indices_val_obj))
        h_head = F.relu(self.bipartite_forward(self.obj_head, h_obj, head_node, edge_indices_obj_head))

        vf_out = self.vf(h_head.squeeze(1))

        return vf_out

    def get_action_and_value(self, batch, action=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        head_node: Tensor = batch["head_node"]
        objective_nodes: Tensor = batch["objective_nodes"]
        value_nodes: Tensor = batch["value_nodes"]
        edge_indices_val_obj: Tensor = batch["edge_indices_val_obj"]
        edge_indices_obj_head: Tensor = batch["edge_indices_obj_head"]
        opponent_encoding: Tensor = F.one_hot(batch["opponent_encoding"], self.num_used_agents)
        accept_mask: Tensor = batch["accept_mask"]

        if self.use_opponent_encoding:
            head_node = torch.cat((head_node, opponent_encoding), dim=-1)

        h_obj = F.relu(self.bipartite_forward(self.val_obj, value_nodes, objective_nodes, edge_indices_val_obj))
        h_head = F.relu(self.bipartite_forward(self.obj_head, h_obj, head_node, edge_indices_obj_head))
        h_obj = F.relu(self.bipartite_forward(self.head_obj, h_head, h_obj, edge_indices_obj_head.flip(1)))
        offer_action_logits = self.bipartite_forward(self.obj_val, h_obj, value_nodes, edge_indices_val_obj.flip(1)).squeeze(-1)


        # head node to accept action
        accept_inf_mask = torch.max(torch.log(accept_mask), torch.Tensor([torch.finfo(torch.float32).min]).to("cuda:0" if torch.cuda.is_available() else "cpu"))
        accept_action_logits = self.accept_head(h_head.squeeze(1)) + accept_inf_mask

        # head node to value function
        vf_out = self.vf(h_head.squeeze(1))

        # gather action logits
        action_logits = torch.cat((accept_action_logits, offer_action_logits), dim=-1)

        probs = MultiCategorical(action_logits, self.action_nvec)

        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), vf_out

class PureGNN2(nn.Module):
    def __init__(self, num_used_agents, args):
        super().__init__()
        self.use_opponent_encoding = args.use_opponent_encoding
        self.num_used_agents = num_used_agents

        head_features = 2 + self.num_used_agents if self.use_opponent_encoding else 2

        hidden_size = args.hidden_size

        self.head_encoder = layer_init(nn.Linear(head_features, hidden_size))
        self.objective_encoder = layer_init(nn.Linear(2, hidden_size))
        self.value_encoder = layer_init(nn.Linear(5, hidden_size))

        self.gnn_layers = GAT(hidden_size, hidden_size, args.gnn_layers, hidden_size, v2=args.gat_v2, heads=args.heads, add_self_loops=args.add_self_loops)
        # self.skip_connects = [nn.Linear(2 * hidden_size, hidden_size) for _ in range(self.config.model_config_dict["num_gcn_layers"])]

        if args.out_layers == 1:
            self.accept_head = layer_init(nn.Linear(hidden_size, 2), std=0.01)
            self.offer_head = layer_init(nn.Linear(hidden_size, 1), std=0.01)
            self.vf = layer_init(nn.Linear(hidden_size, 1), std=1)
        elif args.out_layers == 2:
            self.accept_head = nn.Sequential(
                layer_init(nn.Linear(hidden_size, hidden_size)),
                nn.ReLU(),
                layer_init(nn.Linear(hidden_size, 2), std=0.01),
            )
            self.offer_head = nn.Sequential(
                layer_init(nn.Linear(hidden_size, hidden_size)),
                nn.ReLU(),
                layer_init(nn.Linear(hidden_size, 1), std=0.01),
            )
            self.vf = nn.Sequential(
                layer_init(nn.Linear(hidden_size, hidden_size)),
                nn.ReLU(),
                layer_init(nn.Linear(hidden_size, 1), std=1),
            )
        else:
            raise ValueError(f"Wrong out_layers argument: {args.out_layers}")

    @property
    def action_nvec(self):
        return self._action_nvec
    
    @action_nvec.setter
    def action_nvec(self, value):
        self._action_nvec = value

    def forward_graph(self, batch):
        head_node: Tensor = batch["head_node"]
        objective_nodes: Tensor = batch["objective_nodes"]
        value_nodes: Tensor = batch["value_nodes"]
        edge_indices: Tensor = batch["edge_indices"]

        if self.use_opponent_encoding:
            opponent_encoding: Tensor = F.one_hot(batch["opponent_encoding"], self.num_used_agents)
            head_node = torch.cat((head_node, opponent_encoding), dim=-1)

        h_head_node = F.relu(self.head_encoder(head_node))
        h_objective_nodes = F.relu(self.objective_encoder(objective_nodes))
        h_value_nodes = F.relu(self.value_encoder(value_nodes))

        h_nodes = torch.cat((h_head_node.unsqueeze(1), h_objective_nodes, h_value_nodes), dim=1)


        graph_batch = Batch.from_data_list([Data(h, e).to("cuda:0" if torch.cuda.is_available() else "cpu") for h, e in zip(h_nodes, edge_indices)])
        h_nodes_out = F.relu(self.gnn_layers(graph_batch.x, graph_batch.edge_index)).reshape_as(h_nodes)
        h_value_nodes_out = h_nodes_out[:, -value_nodes.shape[1]:, :]
        h_head_node_out = h_nodes_out[:, 0, :]

        return h_head_node_out, h_value_nodes_out

    def get_value(self, batch):
        h_head_node_out, _ = self.forward_graph(batch)
        # head_node: Tensor = batch["head_node"]
        # objective_nodes: Tensor = batch["objective_nodes"]
        # value_nodes: Tensor = batch["value_nodes"]
        # edge_indices: Tensor = batch["edge_indices"]
        # opponent_encoding: Tensor = F.one_hot(batch["opponent_encoding"], self.num_used_agents)

        # if self.use_opponent_encoding:
        #     head_node = torch.cat((head_node, opponent_encoding), dim=-1)

        # h_head_node = F.relu(self.head_encoder(head_node))
        # h_objective_nodes = F.relu(self.objective_encoder(objective_nodes))
        # h_value_nodes = F.relu(self.value_encoder(value_nodes))

        # h_nodes = torch.cat((h_head_node.unsqueeze(1), h_objective_nodes, h_value_nodes), dim=1)


        # graph_batch = Batch.from_data_list([Data(h, e) for h, e in zip(h_nodes, edge_indices)])
        # h_nodes_out = F.relu(self.gnn_layers(graph_batch.x, graph_batch.edge_index)).reshape_as(h_nodes)

        # h_nodes = torch.cat([F.relu(self.gnn_layers(h, e)).unsqueeze(0) for h, e in zip(h_nodes, edge_indices)], dim=0)

        vf_out = self.vf(h_head_node_out)

        return vf_out
    
    def get_action_and_value(self, batch, action=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h_head_node_out, h_value_nodes_out = self.forward_graph(batch)
        # head_node: Tensor = batch["head_node"]
        # objective_nodes: Tensor = batch["objective_nodes"]
        # value_nodes: Tensor = batch["value_nodes"]
        # edge_indices: Tensor = batch["edge_indices"]
        # opponent_encoding: Tensor = F.one_hot(batch["opponent_encoding"], self.num_used_agents)
        accept_mask: Tensor = batch["accept_mask"]

        # if self.use_opponent_encoding:
        #     head_node = torch.cat((head_node, opponent_encoding), dim=-1)

        # h_head_node = F.relu(self.head_encoder(head_node))
        # h_objective_nodes = F.relu(self.objective_encoder(objective_nodes))
        # h_value_nodes = F.relu(self.value_encoder(value_nodes))

        # h_nodes = torch.cat((h_head_node.unsqueeze(1), h_objective_nodes, h_value_nodes), dim=1)
        # h_nodes = torch.cat([F.relu(self.gnn_layers(h, e)).unsqueeze(0) for h, e in zip(h_nodes, edge_indices)], dim=0)


        # h_value_nodes_out = h_nodes[:, -value_nodes.shape[1]:, :]
        offer_action_logits = self.offer_head(h_value_nodes_out).squeeze(-1)

        accept_inf_mask = torch.max(torch.log(accept_mask), torch.Tensor([torch.finfo(torch.float32).min]).to("cuda:0" if torch.cuda.is_available() else "cpu"))
        accept_action_logits = self.accept_head(h_head_node_out) + accept_inf_mask

        # head node to value function
        vf_out = self.vf(h_head_node_out)

        # gather action logits
        action_logits = torch.cat((accept_action_logits, offer_action_logits), dim=-1)

        probs = MultiCategorical(action_logits, self.action_nvec)

        if action is None and self.training:
            action = probs.sample()
        elif action is None and not self.training:
            action = probs.mode()
        return action, probs.log_prob(action), probs.entropy(), vf_out


class FixedToFixed3(nn.Module):
    def __init__(self, envs, args):
        super().__init__()
        assert not args.use_opponent_encoding
        self.action_nvec = tuple(envs.single_action_space.nvec)

        self.encoder = nn.Sequential(
            nn.Linear(spaces.flatdim(envs.single_observation_space), args.hidden_size),
            nn.ReLU(),
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.ReLU(),
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.ReLU()
        )

        self.vf = nn.Linear(args.hidden_size, 1)
        self.pi = nn.Linear(args.hidden_size, sum(self.action_nvec))

    def get_value(self, batch):
        self_bid: Tensor = batch["self_bid"]
        opponent_bid: Tensor = batch["opponent_bid"]
        time: Tensor = batch["time"]
        X = torch.cat((self_bid, opponent_bid, time), dim=-1)

        H = self.encoder(X)
        vf_out = self.vf(H)

        return vf_out
    
    def get_action_and_value(self, batch, action=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self_bid: Tensor = batch["self_bid"]
        opponent_bid: Tensor = batch["opponent_bid"]
        time: Tensor = batch["time"]
        X = torch.cat((self_bid, opponent_bid, time), dim=-1)

        H = self.encoder(X)
        vf_out = self.vf(H)

        # gather action logits
        action_logits = self.pi(H)

        probs = MultiCategorical(action_logits, self.action_nvec)

        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), vf_out


class HigaEtAl2(nn.Module):
    def __init__(self, envs, args):
        super().__init__()
        assert not args.use_opponent_encoding
        self.action_nvec = tuple(envs.single_action_space.nvec)

        self.encoder = nn.Sequential(
            nn.Linear(spaces.flatdim(envs.single_observation_space), 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )

        self.vf = nn.Sequential(
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.pi = nn.Sequential(
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, sum(self.action_nvec))
        )

    def get_value(self, batch):
        self_bid: Tensor = batch["self_bid"]
        opponent_bid: Tensor = batch["opponent_bid"]
        time: Tensor = batch["time"]
        X = torch.cat((self_bid, opponent_bid, time), dim=-1)

        H = self.encoder(X)
        vf_out = self.vf(H)

        return vf_out
    
    def get_action_and_value(self, batch, action=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self_bid: Tensor = batch["self_bid"]
        opponent_bid: Tensor = batch["opponent_bid"]
        time: Tensor = batch["time"]
        X = torch.cat((self_bid, opponent_bid, time), dim=-1)

        H = self.encoder(X)
        vf_out = self.vf(H)

        # gather action logits
        action_logits = self.pi(H)

        probs = MultiCategorical(action_logits, self.action_nvec)

        if action is None:
            action = probs.sample()
        elif action is None and not self.training:
            action = probs.mode()
        return action, probs.log_prob(action), probs.entropy(), vf_out
    

class MultiCategorical(Distribution):
    def __init__(self, multi_logits, nvec, validate_args=None):
        self.cats = [
            Categorical(logits=logits)
            for logits in torch.split(multi_logits, nvec, dim=-1)
        ]
        batch_shape = multi_logits.size()[:-1]
        super().__init__(batch_shape, validate_args=validate_args)

    def sample(self) -> Tensor:
        return torch.stack([cat.sample() for cat in self.cats], dim=-1)

    def mode(self) -> Tensor:
        return torch.stack([cat.mode for cat in self.cats], dim=-1)

    def log_prob(self, value: Tensor) -> Tensor:
        value = torch.unbind(value, dim=-1)
        logps = torch.stack([cat.log_prob(act) for cat, act in zip(self.cats, value)])
        return torch.sum(logps, dim=0)

    def entropy(self) -> Tensor:
        return torch.stack([cat.entropy() for cat in self.cats], dim=-1).sum(dim=-1)
    
    def kl(self, other):
        kls = torch.stack(
            [kl_divergence(cat, oth_cat) for cat, oth_cat in zip(self.cats, other.cats)],
            dim=-1,
        )
        return torch.sum(kls, dim=-1)