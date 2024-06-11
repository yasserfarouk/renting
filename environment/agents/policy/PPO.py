import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces
from torch import Tensor, nn
from torch.distributions import Categorical
from torch.distributions.distribution import Distribution
from torch.distributions.kl import kl_divergence
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GAT


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class GNN(nn.Module):
    def __init__(self, envs, args):
        super().__init__()

        hidden_size = args.hidden_size

        self.head_encoder = layer_init(nn.Linear(2, hidden_size))
        self.objective_encoder = layer_init(nn.Linear(2, hidden_size))
        self.value_encoder = layer_init(nn.Linear(5, hidden_size))

        self.gnn_layers = GAT(hidden_size, hidden_size, args.gnn_layers, hidden_size, v2=args.gat_v2, heads=args.heads, add_self_loops=args.add_self_loops)

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
        vf_out = self.vf(h_head_node_out)
        return vf_out
    
    def get_action_and_value(self, batch, action=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h_head_node_out, h_value_nodes_out = self.forward_graph(batch)
        accept_mask: Tensor = batch["accept_mask"]

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


class HigaEtAl(nn.Module):
    def __init__(self, envs, args):
        super().__init__()
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