from collections import deque

import numpy as np
from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete

from environment.deadline import Deadline
from environment.scenario import UtilityFunction


class RLAgent:
    def __init__(self, agent_id: str, utility_function: UtilityFunction, num_used_agents):
        self.agent_id = agent_id
        self.utility_function = utility_function

    @staticmethod
    def action_space(utility_function):
        values_per_objective = [len(v) for v in utility_function.value_weights.values()]
        action_space = Dict(
            {
                "outcome": MultiDiscrete(values_per_objective, dtype=np.int64),
                "accept": Discrete(2),
            }
        )
        return action_space

class GraphObs(RLAgent):
    def __init__(self, agent_id: str, utility_function: UtilityFunction, num_used_agents: int):
        super().__init__(agent_id, utility_function, num_used_agents)

        self.num_objectives = len(utility_function.objective_weights)
        values_per_objective = [len(v) for v in utility_function.value_weights.values()]
        objective_weights = [v for v in utility_function.objective_weights.values()]

        self.value_offset = np.insert(np.cumsum(values_per_objective), 0, 0)[:-1]

        self.edge_indices = []
        for i in range(self.num_objectives):
            self.edge_indices.append([0, i + 1])
            self.edge_indices.append([i + 1, 0])

        start = self.num_objectives + 1
        for i, n in enumerate(values_per_objective):
            for j in range(n):
                self.edge_indices.append([i + 1, start + j])
                self.edge_indices.append([start + j, i + 1])
            start += j + 1

        self.edge_indices = np.array(self.edge_indices, dtype=np.int64).T

        self.value_weights = np.array([v2 for v1 in utility_function.value_weights.values() for v2 in v1.values()], dtype=np.float32)
        self.counted_opp_outcomes = np.zeros_like(self.value_weights, dtype=np.float32)
        self.counted_my_outcomes = np.zeros_like(self.value_weights, dtype=np.float32)
        self.fraction_opp_outcomes = np.zeros_like(self.value_weights, dtype=np.float32)
        self.fraction_my_outcomes = np.zeros_like(self.value_weights, dtype=np.float32)

        self.objective_nodes_features =  np.array([[v, o] for v, o in zip(values_per_objective, objective_weights)], dtype=np.float32)

        self.num_opp_actions = 0
        self.num_my_actions = 0

    @staticmethod
    def observation_space(utility_function, num_used_agents):
        num_objectives = len(utility_function.objective_weights)
        values_per_objective = [len(v) for v in utility_function.value_weights.values()]

        num_edges = (num_objectives + sum(values_per_objective)) * 2
        observation_space = Dict(
            {
                "head_node": Box(np.array([0, 0]), np.array([np.inf, 1]), dtype=np.float32),
                # objectives, time (implicit num offers)
                "objective_nodes": Box(np.array([[1, 0]] * num_objectives), np.array([[np.inf, 1]] * num_objectives), shape=(num_objectives, 2), dtype=np.float32),
                # values, weight,
                "value_nodes": Box(0, 1, shape=(sum(values_per_objective), 5), dtype=np.float32),
                # weight, average offered, my outcome, opp outcome
                "edge_indices": Box(0, np.inf, shape=(2, num_edges), dtype=np.int64),
                "opponent_encoding": Discrete(num_used_agents),
                "accept_mask": Box(0, 1, shape=(2,), dtype=bool),
            }
        )
        return observation_space

    @staticmethod
    def action_space(utility_function):
        values_per_objective = [len(v) for v in utility_function.value_weights.values()]
        action_space = MultiDiscrete([2] + values_per_objective, dtype=np.int64)
        return action_space

    def get_observation(self, last_actions: deque[dict], deadline: Deadline, opponent_encoding) -> dict:
        my_outcome = np.zeros_like(self.value_weights, dtype=np.float32)
        opp_outcome = np.zeros_like(self.value_weights, dtype=np.float32)
        accept_mask = np.ones(2, dtype=bool)

        if len(last_actions) == 0:# or deadline.get_progress() <= 0.95:
            accept_mask[1] = False
        if len(last_actions) > 0:
            self.register_opp_action(last_actions[-1])
            opp_outcome[self.value_offset + last_actions[-1]["outcome"]] = 1
        if len(last_actions) > 1:
            self.register_my_action(last_actions[-2])
            my_outcome[self.value_offset + last_actions[-2]["outcome"]] = 1

        obs = {
            "head_node": np.array([self.num_objectives, deadline.get_progress()], dtype=np.float32),
            "objective_nodes": self.objective_nodes_features,
            "value_nodes": np.stack(
                [
                    self.value_weights,
                    self.fraction_my_outcomes,
                    self.fraction_opp_outcomes,
                    my_outcome,
                    opp_outcome,
                ],
                axis=-1,
                dtype=np.float32,
            ),
            "edge_indices": self.edge_indices,
            "opponent_encoding": opponent_encoding,
            "accept_mask": accept_mask,
        }
        return obs

    def register_opp_action(self, action: dict):
        self.num_opp_actions += 1
        self.counted_opp_outcomes[self.value_offset + action["outcome"]] += 1
        self.fraction_opp_outcomes = self.counted_opp_outcomes / self.num_opp_actions

    def register_my_action(self, action: dict):
        self.num_my_actions += 1
        self.counted_my_outcomes[self.value_offset + action["outcome"]] += 1
        self.fraction_my_outcomes = self.counted_my_outcomes / self.num_my_actions



class HigaEtAl(RLAgent):
    def __init__(self, agent_id: str, utility_function: UtilityFunction, num_used_agents: int):
        super().__init__(agent_id, utility_function, num_used_agents)
        self.values_per_objective = [len(v) for v in utility_function.value_weights.values()]
        self.offer_max_first = True
        self.value_offset = np.insert(np.cumsum(self.values_per_objective), 0, 0)[:-1]

    def outcome_to_one_hot(self, outcome):
        one_hot = np.zeros(sum(self.values_per_objective), dtype=np.float32)
        one_hot[self.value_offset + outcome] = 1
        return one_hot

    @staticmethod
    def observation_space(utility_function, num_used_agents):
        values_per_objective = [len(v) for v in utility_function.value_weights.values()]

        observation_space = Dict(
            {
                "self_bid": Box(0, 1, (sum(values_per_objective),), dtype=np.float32),
                "opponent_bid": Box(0, 1, (sum(values_per_objective),), dtype=np.float32),
                "time": Box(0, 1, dtype=np.float32),
            }
        )
        return observation_space

    @staticmethod
    def action_space(utility_function):
        values_per_objective = [len(v) for v in utility_function.value_weights.values()]
        action_space = MultiDiscrete([2] + values_per_objective, dtype=np.int64)
        return action_space

    def get_observation(self, last_actions: deque[dict], deadline: Deadline, opponent_encoding) -> dict:
        obs = {
            "self_bid": self.outcome_to_one_hot(last_actions[-2]["outcome"]),
            "opponent_bid": self.outcome_to_one_hot(last_actions[-1]["outcome"]),
            "time": np.array([deadline.get_progress()], dtype=np.float32),
        }
        return obs

    def get_first_action(self, last_actions: deque[dict]) -> dict:
        outcome = np.array(self.utility_function.max_utility_outcome, dtype=np.int64)
        return {"agent_id": self.agent_id, "outcome": outcome, "accept": 0}