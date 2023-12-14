import json
import math
from itertools import product
from math import sqrt
from pathlib import Path
from shutil import rmtree
from string import ascii_lowercase, ascii_uppercase
from typing import Iterable

import numpy as np
import plotly.graph_objects as go
from numpy.random import Generator


class UtilityFunction:
    def __init__(self, objective_weights: dict, value_weights: dict[str, dict]):
        self.objective_weights = objective_weights
        self.value_weights = value_weights

    @classmethod
    def from_file(cls, file: Path):
        with open(file, "r") as f:
            weights = json.load(f)

        objective_weights = weights["objective_weights"]
        value_weights = weights["value_weights"]

        return cls(objective_weights, value_weights)

    @classmethod
    def create_random(cls, objectives: dict, np_random: Generator):
        def dirichlet_dist(names, mode, alpha=1):
            distribution = (np_random.dirichlet([alpha] * len(names)) * 100000).astype(
                int
            )
            if mode == "objectives":
                distribution[0] += 100000 - np.sum(distribution)
            if mode == "values":
                distribution = distribution - np.min(distribution)
                distribution = (distribution * 100000 / np.max(distribution)).astype(
                    int
                )
            distribution = distribution / 100000
            return {i: w for i, w in zip(names, distribution)}

        objective_weights = dirichlet_dist(list(objectives.keys()), "objectives")
        value_weights = {}
        for objective, values in objectives.items():
            value_weights[objective] = dirichlet_dist(values, "values")

        return cls(objective_weights, value_weights)

    def to_file(self, file: Path):
        weights = {
            "objective_weights": self.objective_weights,
            "value_weights": self.value_weights,
        }
        with open(file, "w") as f:
            f.write(json.dumps(weights, indent=2))

    def get_utility(self, bid: list):
        return sum(
            self.objective_weights[o] * self.value_weights[o][v] for o, v in enumerate(bid)
        )
    
    def get_max_utility_bid(self):
        bid = []
        for i in range(len(self.value_weights)):
            isseu_value_weights = self.value_weights[i]
            bid.append(min(isseu_value_weights, key=isseu_value_weights.get))
        return bid


class Scenario:
    def __init__(
        self,
        objectives: dict,
        utility_function_A: UtilityFunction,
        utility_function_B: UtilityFunction,
        SW_bid=None,
        nash_bid=None,
        kalai_bid=None,
        pareto_front=None,
        distribution=None,
        opposition=None,
        visualisation=None,
    ):
        self.objectives = objectives
        self.utility_function_A = utility_function_A
        self.utility_function_B = utility_function_B
        self.SW_bid = SW_bid
        self.nash_bid = nash_bid
        self.kalai_bid = kalai_bid
        self.pareto_front = pareto_front
        self.distribution = distribution
        self.opposition = opposition
        self.visualisation = visualisation

    @classmethod
    def create_random(cls, size, np_random: Generator):
        if isinstance(size, int):
            size = size
        elif isinstance(size, list):
            size = np_random.integers(size[0], size[1])
        else:
            raise ValueError("size must be int or list")

        while True:
            num_objectives = np_random.integers(3, 10)
            spread = np_random.dirichlet([1] * num_objectives)
            multiplier = (size / np.prod(spread)) ** (1.0 / num_objectives)
            values_per_objective = np.round(multiplier * spread).astype(np.int32)
            values_per_objective = np.clip(values_per_objective, 2, None)
            if abs(size - np.prod(values_per_objective)) < (0.1 * size):
                break

        objectives = {i: [v for v in range(vs)] for i, vs in enumerate(values_per_objective)}

        utility_function_A = UtilityFunction.create_random(objectives, np_random)
        utility_function_B = UtilityFunction.create_random(objectives, np_random)
        return cls(objectives, utility_function_A, utility_function_B)

    @classmethod
    def from_directory(cls, directory: Path):
        utility_function_A = UtilityFunction.from_file(
            directory / "utility_function_A.json"
        )
        utility_function_B = UtilityFunction.from_file(
            directory / "utility_function_B.json"
        )
        with open(directory / "objectives.json", "r") as f:
            objectives = json.load(f)

        specials_path = directory / "specials.json"
        if specials_path.exists():
            with open(specials_path, "r") as f:
                specials = json.load(f)
            return cls(
                objectives,
                utility_function_A,
                utility_function_B,
                SW_bid=specials["social_welfare"],
                nash_bid=specials["nash"],
                kalai_bid=specials["kalai"],
                pareto_front=specials["pareto_front"],
                distribution=specials["distribution"],
                opposition=specials["opposition"],
            )
        else:
            return cls(objectives, utility_function_A, utility_function_B)

    def calculate_specials(self):
        if self.nash_bid:
            return False
        self.pareto_front = self.get_pareto(list(self.iter_bids()))
        self.distribution = self.get_distribution(self.iter_bids())

        SW_utility = 0
        nash_utility = 0
        kalai_diff = 10

        for pareto_bid in self.pareto_front:
            utility_function_A, utility_B = pareto_bid["utility"][0], pareto_bid["utility"][1]

            utility_diff = abs(utility_function_A - utility_B)
            utility_prod = utility_function_A * utility_B
            utility_sum = utility_function_A + utility_B

            if utility_diff < kalai_diff:
                self.kalai_bid = pareto_bid
                kalai_diff = utility_diff
                self.opposition = sqrt((utility_function_A - 1.0) ** 2 + (utility_B - 1.0) ** 2)
            if utility_prod > nash_utility:
                self.nash_bid = pareto_bid
                nash_utility = utility_prod
            if utility_sum > SW_utility:
                self.SW_bid = pareto_bid
                SW_utility = utility_sum

        return True

    def generate_visualisation(self):
        bid_utils = [self.get_utilities(bid) for bid in self.iter_bids()]
        bid_utils = list(zip(*bid_utils))

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=bid_utils[0],
                y=bid_utils[1],
                mode="markers",
                name="bids",
                marker=dict(size=3),
            )
        )

        if self.pareto_front:
            pareto_utils = [bid["utility"] for bid in self.pareto_front]
            pareto_utils = list(zip(*pareto_utils))
            fig.add_trace(
                go.Scatter(
                    x=pareto_utils[0],
                    y=pareto_utils[1],
                    mode="lines+markers",
                    name="Pareto",
                    marker=dict(size=3),
                    line=dict(width=1.5),
                )
            )

        if self.nash_bid:
            x, y = self.nash_bid["utility"]
            fig.add_trace(
                go.Scatter(
                    x=[x],
                    y=[y],
                    mode="markers",
                    name="Nash",
                    marker=dict(size=8, line_width=2, symbol="circle-open"),
                )
            )

        if self.SW_bid:
            x, y = self.SW_bid["utility"]
            fig.add_trace(
                go.Scatter(
                    x=[x],
                    y=[y],
                    mode="markers",
                    name="SW",
                    marker=dict(size=8, line_width=2, symbol="square-open"),
                )
            )

        if self.kalai_bid:
            x, y = self.kalai_bid["utility"]
            fig.add_trace(
                go.Scatter(
                    x=[x],
                    y=[y],
                    mode="markers",
                    name="Kalai-Smorodinsky",
                    marker=dict(size=8, line_width=2, symbol="diamond-open"),
                )
            )

        fig.update_xaxes(range=[0, 1], title_text="Utility A")
        fig.update_yaxes(range=[0, 1], title_text="Utility B")

        fig.update_layout(
            title=dict(
                text=f"<sub>(size: {len(list(self.iter_bids()))}, opposition: {self.opposition:.4f}, distribution: {self.distribution:.4f})</sub>",
                x=0.5,
                xanchor="center",
            )
        )

        self.visualisation = fig

    def to_directory(self, directory: Path):
        if directory.exists():
            rmtree(directory)
        directory.mkdir(parents=True)

        with open(directory / "objectives.json", "w") as f:
            f.write(json.dumps(self.objectives, indent=2))
        self.utility_function_A.to_file(directory / "utility_function_A.json")
        self.utility_function_B.to_file(directory / "utility_function_B.json")

        if self.nash_bid:
            with open(directory / "specials.json", "w") as f:
                f.write(
                    json.dumps(
                        {
                            "size": len(list(self.iter_bids())),
                            "opposition": self.opposition,
                            "distribution": self.distribution,
                            "social_welfare": self.SW_bid,
                            "nash": self.nash_bid,
                            "kalai": self.kalai_bid,
                            "pareto_front": self.pareto_front,
                        },
                        indent=2,
                    )
                )

        if self.visualisation:
            self.visualisation.write_image(file=directory / "visualisation.pdf", scale=5)

    def iter_bids(self) -> Iterable:
        return iter(self)

    def get_utilities(self, bid):
        return self.utility_function_A.get_utility(bid), self.utility_function_B.get_utility(bid)

    def get_pareto(self, all_bids: list):
        pareto_front = []
        # dominated_bids = set()
        while True:
            candidate_bid = all_bids.pop(0)
            bid_nr = 0
            dominated = False
            while len(all_bids) != 0 and bid_nr < len(all_bids):
                bid = all_bids[bid_nr]
                if self._dominates(candidate_bid, bid):
                    # If it is dominated remove the bid from all bids
                    all_bids.pop(bid_nr)
                    # dominated_bids.add(frozenset(bid.items()))
                elif self._dominates(bid, candidate_bid):
                    dominated = True
                    # dominated_bids.add(frozenset(candidate_bid.items()))
                    bid_nr += 1
                else:
                    bid_nr += 1

            if not dominated:
                # add the non-dominated bid to the Pareto frontier
                pareto_front.append(
                    {
                        "bid": candidate_bid,
                        "utility": [
                            self.utility_function_A.get_utility(candidate_bid),
                            self.utility_function_B.get_utility(candidate_bid),
                        ],
                    }
                )

            if len(all_bids) == 0:
                break

        pareto_front = sorted(pareto_front, key=lambda d: d["utility"][0])

        return pareto_front

    def get_distribution(self, bids_iter) -> float:
        min_distance_sum = 0.0

        for i, bid in enumerate(bids_iter):
            min_distance = self.distance_to_pareto(bid)
            min_distance_sum += min_distance

        distribution = min_distance_sum / (i + 1)

        return distribution

    def _dominates(self, bid, candidate_bid):
        if self.utility_function_A.get_utility(bid) < self.utility_function_A.get_utility(candidate_bid):
            return False
        elif self.utility_function_B.get_utility(bid) < self.utility_function_B.get_utility(
            candidate_bid
        ):
            return False
        else:
            return True

    def distance_to_pareto(self, bid):
        if not self.pareto_front:
            raise ValueError("Pareto front not calculated")

        min_distance = 5.0
        for pareto_element in self.pareto_front:
            pareto_bid = pareto_element["bid"]
            distance = self.distance(pareto_bid, bid)
            if distance < min_distance:
                min_distance = distance

        return min_distance

    def distance(self, bid1, bid2=None):
        """calculate Euclidian distance in terms of utility between a bid and 0 or between two bids.

        Args:
            bid1 (dict[str, str]): bid dictionary where keys are issues and values are the values
            bid2 (dict[str, str], optional): see bid1. Defaults to None.

        Returns:
            float: Euclidian distance
        """
        if bid2 and bid1:
            a = (self.utility_function_A.get_utility(bid1) - self.utility_function_A.get_utility(bid2)) ** 2
            b = (self.utility_function_B.get_utility(bid1) - self.utility_function_B.get_utility(bid2)) ** 2
        elif bid1:
            a = self.utility_function_A.get_utility(bid1) ** 2
            b = self.utility_function_B.get_utility(bid1) ** 2
        else:
            raise ValueError("receive None bid")
        return math.sqrt(a + b)

    def __iter__(self) -> dict:
        bids_values = product(*self.objectives.values())
        for bid_values in bids_values:
            yield {i: v for i, v in zip(self.objectives.keys(), bid_values)}
