"""Estimate how passing entropy varies with concentration around passing hubs."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

CONCENTRATION = "pagerank_concentration"
ENTROPY = "conditional_receiver_entropy"
STEP = 0.01
BOOTSTRAP_DRAWS = 2000


@dataclass
class Fit:
    x: np.ndarray
    y: np.ndarray
    slope: float
    r_squared: float
    interval: tuple[float, float] | None


def _residuals(values, effects):
    if effects is None:
        return values - values.mean(axis=0)
    return values - effects @ np.linalg.lstsq(effects, values, rcond=None)[0]


def _slope(x, y):
    variance = x @ x
    return float((x @ y) / variance) if variance > 1e-12 else np.nan


def fit_regression(frame, *, controls=(), cluster="team", draws=BOOTSTRAP_DRAWS,
                   predictor=CONCENTRATION, outcome=ENTROPY):
    """Estimate change in outcome per 0.01 of the predictor, with club bootstrap intervals."""
    values = frame[[predictor, outcome]].to_numpy(dtype=float, copy=True)
    values[:, 0] /= STEP
    effects = None
    if controls:
        dummy = pd.get_dummies(frame[list(controls)].astype(str), drop_first=True, dtype=float)
        effects = np.column_stack([np.ones(len(frame)), dummy.to_numpy()])

    x, y = _residuals(values, effects).T
    slope = _slope(x, y)
    if not np.isfinite(slope):
        return None
    total = y @ y
    r_squared = float(1 - ((y - slope * x) @ (y - slope * x)) / total) if total > 0 else np.nan
    interval = None
    clubs = list(frame.groupby(cluster, sort=False).indices.values())
    if draws and len(clubs) >= 3:
        rng = np.random.default_rng(42)
        estimates = []
        for selection in rng.integers(len(clubs), size=(draws, len(clubs))):
            rows = np.concatenate([clubs[i] for i in selection])
            boot_x, boot_y = _residuals(values[rows], effects[rows] if effects is not None else None).T
            estimates.append(_slope(boot_x, boot_y))
        finite = np.asarray(estimates)
        finite = finite[np.isfinite(finite)]
        if len(finite):
            low, high = np.quantile(finite, [0.025, 0.975])
            interval = (float(low), float(high))

    if not controls:
        x, y = values.T
    return Fit(x * STEP, y, slope, r_squared, interval)


def _plot(results, output):
    figure, axes = plt.subplots(2, 2, figsize=(12, 8.5), layout="constrained")
    for ax, (record, fit, color, complete) in zip(axes.flat, results):
        if complete is None:
            ax.scatter(fit.x, fit.y, s=24, color=color, alpha=0.64)
        else:
            ax.scatter(fit.x[~complete], fit.y[~complete], s=32, facecolors="none", edgecolors=color,
                       label="Partial or prior")
            ax.scatter(fit.x[complete], fit.y[complete], s=36, color=color, label="Complete profiles")
            ax.legend(loc="best", fontsize=8, frameon=False)
        limits = np.array([fit.x.min(), fit.x.max()])
        line = fit.y.mean() + fit.slope * (limits - fit.x.mean()) / STEP
        ax.plot(limits, line, linewidth=1.7, color="#273b43", linestyle="--" if complete is not None else "-")
        adjusted = record["controls"] != "none"
        ax.set_xlabel("Adjusted PageRank concentration" if adjusted else "PageRank concentration")
        ax.set_ylabel("Adjusted passing entropy" if adjusted else "Conditional receiver entropy")
        ax.set_title(record["sample"], loc="left", weight="bold")
        unit = "club-seasons" if record["seasons"] > 1 else "teams" if complete is not None else "clubs"
        note = f"{record['observations']} {unit}  ·  slope {fit.slope:+.3f}"
        if fit.interval:
            note += f"\n95% interval [{fit.interval[0]:+.3f}, {fit.interval[1]:+.3f}]"
        if complete is not None:
            note += f"\nProvisional · {complete.sum()} complete profiles"
        ax.text(0.02, 0.04, note, transform=ax.transAxes, fontsize=8, va="bottom",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.90, "pad": 3})
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.grid(alpha=0.15)
    for ax in list(axes.flat)[len(results):]:
        ax.set_visible(False)

    figure.suptitle("Passing entropy and concentration around passing hubs", fontsize=16)
    figure.supxlabel(
        "Slopes show entropy change per +0.01 concentration. Football intervals resample clubs.",
        fontsize=9,
    )
    figure.savefig(output / "star_entropy_regression.png", dpi=165)
    plt.close(figure)


def run_regression(teams, inferred, output, *, make_plot=True):
    results = []

    def add(frame, label, color, *, controls=(), cluster="team", provisional=False, source):
        frame = frame.loc[np.isfinite(frame[[CONCENTRATION, ENTROPY]]).all(axis=1)]
        if cluster in controls:
            frame = frame.loc[frame.groupby(cluster)["season"].transform("nunique").gt(1)]
        if len(frame) < 3 or frame[CONCENTRATION].nunique() < 2:
            return
        fit = fit_regression(frame, controls=controls, cluster=cluster, draws=0 if provisional else BOOTSTRAP_DRAWS)
        if fit is None:
            return
        complete = frame["coverage_complete"].eq(True).to_numpy() if provisional else None
        record = {
            "sample": label,
            "receiver_source": source,
            "controls": " + ".join(controls) or "none",
            "observations": len(frame),
            "clubs": frame[cluster].nunique(),
            "seasons": frame["season"].nunique(),
            "first_season": frame["season"].min(),
            "last_season": frame["season"].max(),
            "complete_profiles": int(complete.sum()) if complete is not None else len(frame) if source == "recorded" else None,
            "slope_per_0_01": fit.slope,
            "ci95_low": fit.interval[0] if fit.interval else np.nan,
            "ci95_high": fit.interval[1] if fit.interval else np.nan,
            "partial_r_squared": fit.r_squared,
        }
        results.append((record, fit, color, complete))

    statsbomb = teams.loc[teams["sport"].eq("Premier League") & teams["coverage_complete"].eq(True)]
    add(statsbomb, "Premier League · StatsBomb", "#18718d", source="recorded")
    if inferred is not None:
        add(inferred, "Premier League · OpenPlay, season-adjusted", "#478765",
            controls=("season",), cluster="team_id", source="inferred")
        add(inferred, "Premier League · OpenPlay, within clubs", "#478765",
            controls=("season", "team_id"), cluster="team_id", source="inferred")
    nba = teams.loc[teams["sport"].eq("NBA")]
    add(nba, "NBA · saved exports", "#bb693c", provisional=True, source="observed tracking")
    if not results:
        raise ValueError("No team sample has enough usable rows to run the regression.")
    table = pd.DataFrame(record for record, *_ in results)
    table.to_csv(output / "star_entropy_regression.csv", index=False)
    if make_plot:
        _plot(results, output)
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "output",
                        help="Read the saved network tables and write the regression here.")
    args = parser.parse_args()
    teams = pd.read_csv(args.output / "nba_football_team_season_metrics.csv")
    inferred_path = args.output / "openplay_inferred_team_networks.csv"
    summary_path = args.output / "run_summary.json"
    use_inferred = inferred_path.exists()
    if summary_path.exists():
        with summary_path.open(encoding="utf-8") as handle:
            benchmark = json.load(handle).get("receiver_benchmark") or {}
        use_inferred = use_inferred and benchmark.get("meets_extension_criteria", False)
    inferred = pd.read_csv(inferred_path) if use_inferred else None
    table = run_regression(teams, inferred, args.output)
    print(table[["sample", "observations", "slope_per_0_01", "ci95_low", "ci95_high"]].to_string(index=False))


if __name__ == "__main__":
    main()
