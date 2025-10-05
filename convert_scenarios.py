from pathlib import Path
import numpy as np
import sys
from rich import print
from rich.progress import track
from negmas.inout import LinearAdditiveUtilityFunction, Scenario as NegmasScenario
from negmas.helpers.inout import dump
from negmas.outcomes import DiscreteCartesianOutcomeSpace
import shutil


def select_one(x: list):
    return x[0] if x else None


def get_n_issues_values(base: Path, dst_base: Path) -> tuple[int, int]:
    max_n_issues = 0
    max_n_values = 0
    for d in base.iterdir():
        dst = dst_base / d.name
        if not d.is_dir():
            continue
        scenario = NegmasScenario.load(d, ignore_discount=True)
        assert scenario is not None, f"Failed to load {d}"
        assert isinstance(scenario.outcome_space, DiscreteCartesianOutcomeSpace)
        max_n_issues = max(max_n_issues, len(scenario.outcome_space.issues))
        for i, issue in enumerate(scenario.outcome_space.issues):
            max_n_values = max(max_n_values, int(issue.cardinality))
    return max_n_issues, max_n_values


def main(base: Path, extend=False):
    dst_base = base.parent.parent / base.parent.name.replace("_src", "") / base.name
    print(f"Saving to {dst_base}")
    max_n_issues, max_n_values = get_n_issues_values(base, dst_base)
    if extend:
        print(f"Will extend to {max_n_issues} issues and {max_n_values} values each")
    for d in track(base.iterdir()):
        dst = dst_base / d.name
        if not d.is_dir():
            continue
        scenario = NegmasScenario.load(d, ignore_discount=True)
        assert scenario is not None
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copytree(d, dst / "negmas")
        objectives = dict()
        assert isinstance(scenario.outcome_space, DiscreteCartesianOutcomeSpace)
        n_issues = len(scenario.outcome_space.issues)
        for i, issue in enumerate(scenario.outcome_space.issues):
            # objectives[str(i)] = list(issue.all)
            n_vals = int(issue.cardinality)
            objectives[str(i)] = [str(i) for i in range(n_vals)]
            if extend and n_vals < max_n_values:
                objectives[str(i)] += [str(j) for j in range(n_vals, max_n_values)]

        if extend and n_issues < max_n_issues:
            for i in range(n_issues, max_n_issues):
                objectives[str(i)] = [str(j) for j in range(max_n_values)]

        assert len(objectives) == (max_n_issues if extend else n_issues)

        dump(objectives, dst / "objectives.json")
        metrics = scenario.calc_stats()
        dump(
            dict(
                size=scenario.outcome_space.cardinality,
                name=d.name,
                src_path=str(d),
                load_path=None,
                opposition=metrics.opposition,
                distribution=None,
                social_welfare=dict(
                    outcome=select_one(metrics.max_welfare_outcomes),
                    utility=metrics.max_welfare_utils,
                ),
                nash=dict(
                    outcome=select_one(metrics.nash_outcomes),
                    utility=metrics.nash_utils,
                ),
                kalai=dict(
                    outcome=select_one(metrics.kalai_outcomes),
                    utility=metrics.kalai_utils,
                ),
                pareto_front=[
                    dict(outcome=o, utility=u)
                    for o, u in zip(metrics.pareto_outcomes, metrics.pareto_utils)
                ],
            ),
            dst / "specials.json",
        )
        for u, x in zip(scenario.ufuns, ["A", "B"]):
            uinfo = dict()
            assert isinstance(u, LinearAdditiveUtilityFunction), f"{type(u)}"
            weights = np.asarray(u.weights)
            weights = (
                weights / weights.sum()
                if weights.sum() > 1e-6
                else np.ones_like(weights) / len(weights)
            )
            uinfo["objective_weights"] = dict(
                zip(
                    [str(i) for i in range(n_issues)],
                    weights,
                )
            )
            if extend and n_issues < max_n_issues:
                uinfo["objective_weights"] |= dict(
                    zip(
                        [str(i) for i in range(n_issues, max_n_issues)],
                        [0.0] * (max_n_issues - n_issues),
                    )
                )
            uinfo["value_weights"] = dict()
            for i, issue in enumerate(scenario.outcome_space.issues):
                uinfo["value_weights"][str(i)] = dict()
                n_vals = int(issue.cardinality)
                for k, v in enumerate(issue.all):
                    vv = u.values[i](v)
                    # if vv < -1e-3 or vv > 1.0 + 1e-3:
                    #     print(f"{u=}, {v=}, {vv=} -> setting it to 0 or 1")
                    uinfo["value_weights"][str(i)][str(k)] = max(0.0, min(1.0, vv))
                if n_vals < max_n_values:
                    for k in range(n_vals, max_n_values):
                        uinfo["value_weights"][str(i)][str(k)] = 0.0
            for i in range(n_issues, max_n_issues):
                uinfo["value_weights"][str(i)] = dict()
                for k in range(max_n_values):
                    uinfo["value_weights"][str(i)][str(k)] = 0.0
            assert len(uinfo["value_weights"]) == (max_n_issues if extend else n_issues)
            assert len(uinfo["objective_weights"]) == (
                max_n_issues if extend else n_issues
            )
            for i, issue in enumerate(scenario.outcome_space.issues):
                assert len(uinfo["value_weights"][str(i)]) == (
                    max_n_values if extend else int(issue.cardinality)
                ), (
                    f"ufun {x}: issue {i}: {len(uinfo['value_weights'][str(i)])} not {max_n_values} ({issue.cardinality=})"
                )
            uinfo["reserved_value"] = (
                u.reserved_value
                if u.reserved_value is not None
                and not np.isinf(u.reserved_value)
                and not np.isnan(u.reserved_value)
                else 0.0
            )
            uinfo["name"] = u.name if u.name else ""
            dump(uinfo, dst / f"utility_function_{x}.json")
    print(f"{max_n_issues=}\n{max_n_values=}")


if __name__ == "__main__":
    main(
        Path(sys.argv[1]),
        extend=(len(sys.argv) > 2 and sys.argv[2].startswith("--extend")),
    )
