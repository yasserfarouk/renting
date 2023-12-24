
from collections import deque

import numpy as np

from environment.deadline import Deadline
from environment.scenario import UtilityFunction


class RLAgent:
    def __init__(self, agent_id: str, utility_function: UtilityFunction, *args):
        self.agent_id = agent_id
        self.utility_function = utility_function

        self.num_objectives = len(utility_function.objective_weights)

        self.values_per_objective = [
            len(v) for v in utility_function.value_weights.values()
        ]
        self.objective_weights = [
            v for v in utility_function.objective_weights.values()
        ]

        self.value_weights = [
            v2 for v1 in utility_function.value_weights.values() for v2 in v1.values()
        ]
        self.num_opp_actions = 0
        self.action_offset = np.cumsum(np.insert(self.values_per_objective, 0, 0)[:-1])
        self.counted_opp_outcomes = np.zeros(len(self.value_weights), dtype=np.float32)

    def get_observation(self, last_actions: deque[dict], deadline: Deadline) -> dict:
        self.register_opp_action(last_actions[-1])
        my_outcome = np.zeros_like(self.value_weights)
        my_outcome[self.action_offset + last_actions[-2]["outcome"]] = 1
        opp_outcome = np.zeros_like(self.value_weights)
        opp_outcome[self.action_offset + last_actions[-1]["outcome"]] = 1
        # TODO: construct proper observation
        obs = {
            "head_node": np.array(
                [self.num_objectives, deadline.get_progress()], dtype=np.float32
            ),
            "objective_nodes": np.array(
                [self.values_per_objective, self.objective_weights], dtype=np.float32
            ).T,
            "value_nodes": np.array(
                [
                    self.value_weights,
                    self.counted_opp_outcomes / self.num_opp_actions,
                    my_outcome,
                    opp_outcome,
                ],
                dtype=np.float32,
            ).T,
        }
        return {self.agent_id: obs}

    def get_first_action(self, last_actions: deque[dict]) -> dict:
        if len(last_actions) == 1:
            self.register_opp_action(last_actions[-1])
        outcome = np.array(self.utility_function.max_utility_outcome, dtype=np.int32)
        return {"agent_id": self.agent_id, "outcome": outcome, "accept": 0}

    def register_opp_action(self, action: dict):
        self.num_opp_actions += 1
        outcome_index = self.action_offset + action["outcome"]
        self.counted_opp_outcomes[outcome_index] += 1
