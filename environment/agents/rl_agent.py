
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

        self.max_num_values = max(self.values_per_objective)
        self.value_weights = np.array([list(v.values()) + [0] * (self.max_num_values - len(v)) for v in utility_function.value_weights.values()],dtype=np.float32)
        self.counted_opp_outcomes = np.zeros_like(self.value_weights, dtype=np.float32)
        self.fraction_opp_outcomes = np.zeros_like(self.value_weights, dtype=np.float32)
        self.value_nodes_mask = np.array([[True] * len(v) + [False] * (self.max_num_values - len(v)) for v in utility_function.value_weights.values()],dtype=bool)


        self.objective_nodes_features =  np.array([[v, o] for v, o in zip(self.values_per_objective, self.objective_weights)], dtype=np.float32)

        self.num_opp_actions = 0

    def get_observation(self, last_actions: deque[dict], deadline: Deadline, opponent_encoding) -> dict:
        my_outcome = np.zeros_like(self.value_weights, dtype=np.float32)
        opp_outcome = np.zeros_like(self.value_weights, dtype=np.float32)
        accept_mask = np.ones(2, dtype=bool)

        if len(last_actions) == 0:
            accept_mask[1] = False
        if len(last_actions) > 0:
            self.register_opp_action(last_actions[-1])
            opp_outcome[np.arange(len(opp_outcome)), last_actions[-1]["outcome"]] = 1
        if len(last_actions) > 1:
            my_outcome[np.arange(len(my_outcome)), last_actions[-2]["outcome"]] = 1

        obs = {
            "head_node": np.array([self.num_objectives, deadline.get_progress()], dtype=np.float32),
            "objective_nodes": self.objective_nodes_features,
            "value_nodes": np.stack(
                [
                    self.value_weights,
                    self.fraction_opp_outcomes,
                    my_outcome,
                    opp_outcome,
                ],
                axis=-1,
                dtype=np.float32,
            ),
            "value_nodes_mask": self.value_nodes_mask,
            "opponent_encoding": opponent_encoding,
            "accept_mask": accept_mask,
        }
        return {self.agent_id: obs}

    def get_first_action(self, last_actions: deque[dict]) -> dict:
        if len(last_actions) == 1:
            self.register_opp_action(last_actions[-1])
        outcome = np.array(self.utility_function.max_utility_outcome, dtype=np.int32)
        return {"agent_id": self.agent_id, "outcome": outcome, "accept": 0}

    def register_opp_action(self, action: dict):
        self.num_opp_actions += 1
        self.counted_opp_outcomes[np.arange(len(self.counted_opp_outcomes)), action["outcome"]] += 1
        self.fraction_opp_outcomes = self.counted_opp_outcomes / self.num_opp_actions
