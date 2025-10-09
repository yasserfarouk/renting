import pandas as pd
import random
from rich import print
from pathlib import Path
import typer

not_combined = ["scml", "anac"]


def main(selector: str = "last"):
    base = Path.home() / "negmas" / "external" / "renting"
    dirs = [d for d in base.iterdir() if d.is_dir()]
    results = []
    tocombine = []
    selected_info = dict()
    for d in dirs:
        d /= "GNN"
        runs = sorted(
            [f for f in d.iterdir() if f.is_dir() and len(list(f.glob("*.csv"))) >= 2],
            reverse=True,
        )
        selected_indx = None
        if not runs:
            print(f"{d} has no runs")
            continue
        metric = selector.split("-")[-1]
        if selector == "last":
            run = runs[0]
        elif selector == "first":
            run = runs[-1]
        elif selector == "mid":
            run = runs[len(runs) // 2]
        elif selector.startswith("best"):
            _best, _run, _indx = float("-inf"), None, None
            for r in runs:
                e = pd.read_csv(r / "evaluation.csv", index_col=None)
                for i in range(len(e)):
                    if e[f"my_{metric}_mean"].iloc[i] < _best:
                        continue
                    _best = e[f"my_{metric}_mean"].iloc[i]
                    _run, _indx = r, i
            if _run is None:
                run, selected_indx = runs[0], 0
            else:
                run, selected_indx = _run, _indx
        elif selector.startswith("worst"):
            metric = selector.split("-")[-1]
            _best, _run, _indx = float("inf"), None, None
            for r in runs:
                e = pd.read_csv(r / "evaluation.csv", index_col=None)
                for i in range(len(e)):
                    if e[f"my_{metric}_mean"].iloc[i] > _best:
                        continue
                    _best = e[f"my_{metric}_mean"].iloc[i]
                    _run, _indx = r, i
            if _run is None:
                run, selected_indx = runs[0], 0
            else:
                run, selected_indx = _run, _indx
        else:
            run = random.choice(runs)
            e = pd.read_csv(run / "evaluation.csv", index_col=None)
            selected_indx = random.randint(0, len(e) - 1)

        dbname = d.parent.name
        print(f"Using {run.name} for {dbname}")
        e = pd.read_csv(run / "evaluation.csv", index_col=None)
        if selected_indx is None:
            if selector == "last":
                selected_indx = len(e) - 1
            elif selector == "first":
                selected_indx = 0
            elif selector == "mid":
                selected_indx = len(e) // 2
            else:
                selected_indx = random.randint(0, len(e) - 1)
        e = e.iloc[[selected_indx]]
        e["db"] = dbname
        if not any(dbname.startswith(_) for _ in not_combined):
            tocombine.append(e)
        else:
            results.append(e)

    combined = pd.concat(tocombine, ignore_index=True)

    mean_std_cols = [col[:-5] for col in combined.columns if col.endswith("_mean")]
    new_record = {"db": "anac1on1"}
    N = 50
    n = len(combined)
    for base in mean_std_cols:
        means = combined[f"{base}_mean"]
        stds = combined[f"{base}_std"]
        mean_val = means.mean()
        sum_means = means.sum()
        sum_means_sq = (means**2).sum()
        sum_stds_sq = (stds**2).sum()
        var_total = (
            N * sum_stds_sq + N * sum_means_sq - (N * sum_means) ** 2 / (n * N)
        ) / (n * N)
        std_val = var_total**0.5
        new_record[f"{base}_mean"] = mean_val
        new_record[f"{base}_std"] = std_val
    combined = pd.DataFrame.from_records([new_record])
    # combined = pd.concat([combined, pd.DataFrame([new_record])], ignore_index=True)
    df = pd.concat(results + [combined], ignore_index=True)
    df["time_mean"] = df["rounds_played_mean"]
    df["time_std"] = df["rounds_played_std"]
    selected = ["db"]
    for c in [
        "my_advantage",
        "max_welfare_optimality",
        "nash_optimality",
        "kalai_optimality",
        "pareto_optimality",
        "time",
    ]:
        selected += [f"{c}_mean", f"{c}_std"]
    cols = selected + [c for c in df.columns if c not in selected]
    df = df[cols]

    for c in [_ for _ in df.columns if _.endswith("_std")]:
        df[c] = df[c].round(3)
    for c in [_ for _ in df.columns if _.endswith("_mean")]:
        df[c] = df[c].round(3)
        df["_".join(c.split("_")[:-1])] = (
            df[c].astype(str) + " (" + df[c.replace("mean", "std")].astype(str) + ")"
        )
    df.drop(
        columns=[_ for _ in df.columns if _.endswith("_mean") or _.endswith("_std")],
        inplace=True,
    )
    df.rename(
        columns={
            "my_advantage": "Advantage",
            "max_welfare_optimality": "Welfare",
            "nash_optimality": "Nash",
            "kalai_optimality": "Kalai",
            "pareto_optimality": "Pareto",
            "neg_time": "Neg. Time",
        },
        inplace=True,
    )
    df = df[["db", "Advantage", "Welfare", "Nash", "Kalai", "Pareto", "Neg. Time"]]
    print(df)
    df.to_latex("renting_results.tex", index=False)
    df[["db", "Advantage", "Welfare", "Nash"]].to_latex(
        "renting_results_short.tex", index=False
    )


if __name__ == "__main__":
    typer.run(main)
