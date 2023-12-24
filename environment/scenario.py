import json
import math
from itertools import product
from math import sqrt
from pathlib import Path
from shutil import rmtree
from typing import Iterable

import numpy as np
import plotly.graph_objects as go
from numpy.random import Generator
from numpy.random import default_rng


class UtilityFunction:
    def __init__(self, objective_weights: dict, value_weights: dict[str, dict]):
        self.objective_weights = objective_weights
        self.value_weights = value_weights

    @classmethod
    def from_file(cls, file: Path):
        with open(file, "r") as f:
            #TODO: this is ugly, maybe move to a vector based solution
            weights = json.load(f)
            weights["objective_weights"] = {int(k): v for k, v in weights["objective_weights"].items()}
            weights["value_weights"] = {int(k1): {int(k2): v2 for k2, v2 in v1.items()} for k1, v1 in weights["value_weights"].items()}

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

    def get_utility(self, outcome: list | tuple):
        return sum(
            self.objective_weights[o] * self.value_weights[o][v] for o, v in enumerate(outcome)
        )
    
    @property
    def max_utility_outcome(self):
        return [max(vw, key=vw.get) for vw in self.value_weights.values()]


class Scenario:
    def __init__(
        self,
        objectives: dict,
        utility_function_A: UtilityFunction,
        utility_function_B: UtilityFunction,
        SW_outcome=None,
        nash_outcome=None,
        kalai_outcome=None,
        pareto_front=None,
        distribution=None,
        opposition=None,
        visualisation=None,
    ):
        self.objectives = objectives
        self.utility_function_A = utility_function_A
        self.utility_function_B = utility_function_B
        self.SW_outcome = SW_outcome
        self.nash_outcome = nash_outcome
        self.kalai_outcome = kalai_outcome
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
            if (abs(size - np.prod(values_per_objective)) < (0.1 * size)) and values_per_objective.sum() < 1000:
                break

        objectives = {i: [v for v in range(vs)] for i, vs in enumerate(values_per_objective)}

        utility_function_A = UtilityFunction.create_random(objectives, np_random)
        utility_function_B = UtilityFunction.create_random(objectives, np_random)
        return cls(objectives, utility_function_A, utility_function_B)

    @classmethod
    def from_directory(cls, directory: Path, np_random=default_rng()):
        with open(directory / "objectives.json", "r") as f:
            objectives = {int(k): v for k, v in json.load(f).items()}

        if (directory / "utility_function_A.json").exists():
            utility_function_A = UtilityFunction.from_file(
                directory / "utility_function_A.json"
            )
            utility_function_B = UtilityFunction.from_file(
                directory / "utility_function_B.json"
            )
            specials_path = directory / "specials.json"
            if specials_path.exists():
                with open(specials_path, "r") as f:
                    specials = json.load(f)
                return cls(
                    objectives,
                    utility_function_A,
                    utility_function_B,
                    SW_outcome=specials["social_welfare"],
                    nash_outcome=specials["nash"],
                    kalai_outcome=specials["kalai"],
                    pareto_front=specials["pareto_front"],
                    distribution=specials["distribution"],
                    opposition=specials["opposition"],
                )
        else:
            utility_function_A = UtilityFunction.create_random(objectives, np_random)
            utility_function_B = UtilityFunction.create_random(objectives, np_random)


        return cls(objectives, utility_function_A, utility_function_B)

    def calculate_specials(self):
        if self.nash_outcome:
            return False
        self.pareto_front = self.get_pareto(list(self.iter_outcomes()))
        self.distribution = self.get_distribution(self.iter_outcomes())

        SW_utility = 0
        nash_utility = 0
        kalai_diff = 10

        for pareto_outcome in self.pareto_front:
            utility_function_A, utility_B = pareto_outcome["utility"][0], pareto_outcome["utility"][1]

            utility_diff = abs(utility_function_A - utility_B)
            utility_prod = utility_function_A * utility_B
            utility_sum = utility_function_A + utility_B

            if utility_diff < kalai_diff:
                self.kalai_outcome = pareto_outcome
                kalai_diff = utility_diff
                self.opposition = sqrt((utility_function_A - 1.0) ** 2 + (utility_B - 1.0) ** 2)
            if utility_prod > nash_utility:
                self.nash_outcome = pareto_outcome
                nash_utility = utility_prod
            if utility_sum > SW_utility:
                self.SW_outcome = pareto_outcome
                SW_utility = utility_sum

        return True

    def generate_visualisation(self):
        outcome_utils = [self.get_utilities(outcome) for outcome in self.iter_outcomes()]
        outcome_utils = list(zip(*outcome_utils))

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=outcome_utils[0],
                y=outcome_utils[1],
                mode="markers",
                name="outcomes",
                marker=dict(size=3),
            )
        )

        if self.pareto_front:
            pareto_utils = [outcome["utility"] for outcome in self.pareto_front]
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

        if self.nash_outcome:
            x, y = self.nash_outcome["utility"]
            fig.add_trace(
                go.Scatter(
                    x=[x],
                    y=[y],
                    mode="markers",
                    name="Nash",
                    marker=dict(size=8, line_width=2, symbol="circle-open"),
                )
            )

        if self.SW_outcome:
            x, y = self.SW_outcome["utility"]
            fig.add_trace(
                go.Scatter(
                    x=[x],
                    y=[y],
                    mode="markers",
                    name="SW",
                    marker=dict(size=8, line_width=2, symbol="square-open"),
                )
            )

        if self.kalai_outcome:
            x, y = self.kalai_outcome["utility"]
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
                text=f"<sub>(size: {len(list(self.iter_outcomes()))}, opposition: {self.opposition:.4f}, distribution: {self.distribution:.4f})</sub>",
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

        if self.nash_outcome:
            with open(directory / "specials.json", "w") as f:
                f.write(
                    json.dumps(
                        {
                            "size": len(list(self.iter_outcomes())),
                            "opposition": self.opposition,
                            "distribution": self.distribution,
                            "social_welfare": self.SW_outcome,
                            "nash": self.nash_outcome,
                            "kalai": self.kalai_outcome,
                            "pareto_front": self.pareto_front,
                        },
                        indent=2,
                    )
                )

        if self.visualisation:
            self.visualisation.write_image(file=directory / "visualisation.pdf", scale=5)

    def iter_outcomes(self) -> Iterable:
        return iter(self)

    def get_utilities(self, outcome):
        return self.utility_function_A.get_utility(outcome), self.utility_function_B.get_utility(outcome)

    def get_pareto(self, all_outcomes: list):
        pareto_front = []
        # dominated_outcomes = set()
        while True:
            candidate_outcome = all_outcomes.pop(0)
            outcome_nr = 0
            dominated = False
            while len(all_outcomes) != 0 and outcome_nr < len(all_outcomes):
                outcome = all_outcomes[outcome_nr]
                if self._dominates(candidate_outcome, outcome):
                    # If it is dominated remove the outcome from all outcomes
                    all_outcomes.pop(outcome_nr)
                    # dominated_outcomes.add(frozenset(outcome.items()))
                elif self._dominates(outcome, candidate_outcome):
                    dominated = True
                    # dominated_outcomes.add(frozenset(candidate_outcome.items()))
                    outcome_nr += 1
                else:
                    outcome_nr += 1

            if not dominated:
                # add the non-dominated outcome to the Pareto frontier
                pareto_front.append(
                    {
                        "outcome": candidate_outcome,
                        "utility": [
                            self.utility_function_A.get_utility(candidate_outcome),
                            self.utility_function_B.get_utility(candidate_outcome),
                        ],
                    }
                )

            if len(all_outcomes) == 0:
                break

        pareto_front = sorted(pareto_front, key=lambda d: d["utility"][0])

        return pareto_front

    def get_distribution(self, outcomes_iter) -> float:
        min_distance_sum = 0.0

        for i, outcome in enumerate(outcomes_iter):
            min_distance = self.distance_to_pareto(outcome)
            min_distance_sum += min_distance

        distribution = min_distance_sum / (i + 1)

        return distribution

    def _dominates(self, outcome, candidate_outcome):
        if self.utility_function_A.get_utility(outcome) < self.utility_function_A.get_utility(candidate_outcome):
            return False
        elif self.utility_function_B.get_utility(outcome) < self.utility_function_B.get_utility(
            candidate_outcome
        ):
            return False
        else:
            return True

    def distance_to_pareto(self, outcome):
        if not self.pareto_front:
            raise ValueError("Pareto front not calculated")

        min_distance = 5.0
        for pareto_element in self.pareto_front:
            pareto_outcome = pareto_element["outcome"]
            distance = self.distance(pareto_outcome, outcome)
            if distance < min_distance:
                min_distance = distance

        return min_distance

    def distance(self, outcome1, outcome2=None):
        """calculate Euclidian distance in terms of utility between a outcome and 0 or between two outcomes.

        Args:
            outcome1 (dict[str, str]): outcome dictionary where keys are issues and values are the values
            outcome2 (dict[str, str], optional): see outcome1. Defaults to None.

        Returns:
            float: Euclidian distance
        """
        if outcome2 and outcome1:
            a = (self.utility_function_A.get_utility(outcome1) - self.utility_function_A.get_utility(outcome2)) ** 2
            b = (self.utility_function_B.get_utility(outcome1) - self.utility_function_B.get_utility(outcome2)) ** 2
        elif outcome1:
            a = self.utility_function_A.get_utility(outcome1) ** 2
            b = self.utility_function_B.get_utility(outcome1) ** 2
        else:
            raise ValueError("receive None outcome")
        return math.sqrt(a + b)

    def __iter__(self) -> tuple:
        outcomes_values = product(*self.objectives.values())
        for outcome_values in outcomes_values:
            yield outcome_values
