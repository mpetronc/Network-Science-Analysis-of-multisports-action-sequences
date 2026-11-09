"""Check whether passing-hub concentration adds information beyond pass volumes."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_sources import DataSources, YEARS, event_file, read_json, season_name
from metrics import network_metrics
from openplay import inferred_networks, read_events, read_lineups, read_matches
from regression import CONCENTRATION, ENTROPY, fit_regression
from statsbomb import parse_match

HERE = Path(__file__).resolve().parent


class LineupShuffle:
    """Exchange receivers only between passes in the same match and lineup."""

    def __init__(self, senders, receivers, groups):
        self.senders = np.asarray(senders)
        self.receivers = np.asarray(receivers).copy()
        self.groups = np.asarray(groups)
        _, counts = np.unique(self.groups, return_counts=True)
        ends = np.cumsum(counts)
        self.left_slots = np.concatenate([
            np.arange(end - count, end - 1, 2) for end, count in zip(ends, counts)
        ]) if len(counts) else np.empty(0, dtype=int)

    def sweep(self, rng):
        order = rng.permutation(len(self.groups))
        order = order[np.argsort(self.groups[order], kind="stable")]
        a, b = order[self.left_slots], order[self.left_slots + 1]
        legal = (self.senders[a] != self.receivers[b]) & (self.senders[b] != self.receivers[a])
        a, b = a[legal], b[legal]
        self.receivers[a], self.receivers[b] = self.receivers[b], self.receivers[a]


def null_inputs(sources, baseline):
    """Use the same season roster and pass rules as the recorded StatsBomb fit."""
    rosters = defaultdict(set)
    fixtures = sources.statsbomb_matches()
    paths = sources.statsbomb_paths(fixtures)
    passes = defaultdict(list)
    for fixture, path in zip(fixtures, paths):
        match = parse_match(fixture, read_json(path))
        for club, roster in match.players.items():
            rosters[club].update(roster)
        for club, sender, receiver, options in match.passes[
            ["team_id", "passer_id", "receiver_id", "options"]
        ].itertuples(index=False, name=None):
            lineup = tuple(sorted((*options, sender))) if options else ()
            passes[club].append((sender, receiver, (match.match_id, lineup) if lineup else None))

    sizes = baseline["players"].to_numpy(int)
    ends = np.cumsum(sizes ** 2)
    starts = np.r_[0, ends[:-1]]
    rows, senders, receivers, groups = [], [], [], []
    fixed = np.zeros(ends[-1], dtype=int)
    for club, size, start in zip(baseline["team_id"], sizes, starts):
        index = {player: i for i, player in enumerate(sorted(rosters[club]))}
        if len(index) != size:
            raise ValueError(f"Saved StatsBomb roster disagrees with cached fixtures for club {club}.")
        lineup_ids = {}
        for sender, receiver, group in passes[club]:
            row = start + index[sender] * size
            if group is None:
                fixed[row + index[receiver]] += 1
                continue
            if group not in lineup_ids:
                lineup_ids[group] = len(groups)
            rows.append(row)
            senders.append(index[sender])
            receivers.append(index[receiver])
            groups.append(lineup_ids[group])

    return np.asarray(rows), np.asarray(senders), np.asarray(receivers), groups, fixed, sizes, starts, ends


def run_null(sources, baseline, output):
    rows, senders, receivers, groups, fixed, sizes, starts, ends = null_inputs(sources, baseline)

    def measure(received):
        counts = fixed + np.bincount(rows + received, minlength=len(fixed))
        values = []
        for size, start, end in zip(sizes, starts, ends):
            metrics, _ = network_metrics(counts[start:end].reshape(size, size))
            values.append((metrics[CONCENTRATION], metrics[ENTROPY]))
        return np.asarray(values)

    def slope(values):
        frame = pd.DataFrame(values, columns=[CONCENTRATION, ENTROPY]).assign(team=baseline["team"].to_numpy())
        return fit_regression(frame, draws=0).slope

    observed = measure(receivers)
    np.testing.assert_allclose(observed, baseline[[CONCENTRATION, ENTROPY]], atol=1e-10, rtol=1e-10)
    actual_slope = slope(observed)
    simulations, chain_ids = [], []
    for seed in (154, 155, 156, 157):
        shuffle = LineupShuffle(senders, receivers, groups)
        rng = np.random.default_rng(seed)
        for _ in range(20):
            shuffle.sweep(rng)
        for _ in range(100):
            for _ in range(4):
                shuffle.sweep(rng)
            simulations.append(measure(shuffle.receivers))
            chain_ids.append(seed)
        print(f"StatsBomb shuffle chain {len(chain_ids) // 100}/4 complete.", flush=True)

    values = np.array(simulations)
    slopes = np.array([slope(draw) for draw in values])
    pd.DataFrame({"chain_seed": chain_ids, "slope_per_0_01": slopes}).to_csv(
        output / "star_entropy_null_draws.csv", index=False
    )
    median_entropy = np.median(values[:, :, 1], axis=0)
    baseline[["team", "recorded_passes"]].assign(
        observed_entropy=observed[:, 1], null_median_entropy=median_entropy,
        entropy_minus_null=observed[:, 1] - median_entropy,
    ).to_csv(output / "star_entropy_null_clubs.csv", index=False)
    lo, hi = np.quantile(slopes, [0.025, 0.975])
    total = len(rows) + fixed.sum()
    return slopes, {
        "passes": int(total),
        "passes_with_known_lineup": len(rows),
        "draws": len(slopes),
        "observed_slope": actual_slope,
        "null_median_slope": float(np.median(slopes)),
        "null_central_95": [float(lo), float(hi)],
        "null_draws_at_or_below_observed_slope": int((slopes <= actual_slope).sum()),
        "mean_entropy_minus_null": float((observed[:, 1] - median_entropy).mean()),
        "chain_mean_slopes": pd.Series(slopes).groupby(chain_ids).mean().tolist(),
    }


def half_seasons(sources):
    matches = read_matches(sources)
    lineups = read_lineups(sources)
    excluded = pd.read_csv(sources.openplay("data_issues.csv"))["match_id"].dropna().astype(int)
    rows = []
    for year in YEARS:
        season = season_name(year)
        code = event_file(year).split("_")[-1][:4]
        season_matches = matches.loc[matches["season"].eq(code)]
        events = read_events(sources, year)
        if set(events["match_id"]) != set(season_matches["match_id"]):
            raise ValueError(f"{season}: OpenPlay event and match manifests disagree.")
        usable = season_matches.loc[~season_matches["match_id"].isin(excluded)].sort_values(
            ["date", "kickoff", "match_id"]
        )
        midpoint = usable.iloc[len(usable) // 2]["date"]
        early, late = [], []
        for mask, collection in [(usable["date"].lt(midpoint), early), (usable["date"].ge(midpoint), late)]:
            ids = usable.loc[mask, "match_id"]
            team_rows, _ = inferred_networks(events.loc[events["match_id"].isin(ids)], lineups, season)
            collection.extend(team_rows)
        columns = ["season", "team_id", "team", ENTROPY, CONCENTRATION, "inferred_receiver_share"]
        halves = pd.DataFrame(early)[columns].merge(
            pd.DataFrame(late)[columns].drop(columns="team"),
            on=["season", "team_id"], suffixes=("_early", "_late"), validate="one_to_one",
        )
        rows.append(halves.assign(cutoff=str(midpoint)))
        print(f"OpenPlay {season}: rebuilt early and late club networks.", flush=True)
    return pd.concat(rows, ignore_index=True)


def predict_late_entropy(halves):
    y, early, concentration = f"{ENTROPY}_late", f"{ENTROPY}_early", f"{CONCENTRATION}_early"
    results = []
    for season in sorted(halves["season"].unique())[1:]:
        train = halves.loc[halves["season"].lt(season)]
        test = halves.loc[halves["season"].eq(season)].copy()
        for label, columns in [("entropy_only", [early]), ("entropy_and_hubs", [early, concentration])]:
            design = np.column_stack([np.ones(len(train)), train[columns].to_numpy()])
            coef = np.linalg.lstsq(design, train[y].to_numpy(), rcond=None)[0]
            test[label] = np.clip(coef[0] + test[columns].to_numpy() @ coef[1:], 0, 1)
        results.append(test)
    return pd.concat(results, ignore_index=True)


def run_temporal(sources, output):
    forecasts = predict_late_entropy(half_seasons(sources))
    forecasts.to_csv(output / "star_entropy_heldout_clubs.csv", index=False)
    rows = []
    for season, frame in [*forecasts.groupby("season"), ("pooled", forecasts)]:
        actual = frame[f"{ENTROPY}_late"]
        result = {"season": season, "club_seasons": len(frame)}
        for model in ("entropy_only", "entropy_and_hubs"):
            errors = frame[model] - actual
            result[f"{model}_rmse"] = float(np.sqrt(np.mean(errors ** 2)))
            result[f"{model}_mae"] = float(np.mean(errors.abs()))
        result["rmse_improvement"] = result["entropy_only_rmse"] - result["entropy_and_hubs_rmse"]
        rows.append(result)
    scores = pd.DataFrame(rows)
    scores.to_csv(output / "star_entropy_heldout_scores.csv", index=False)
    return scores


def plot_followup(slopes, null, scores, output):
    figure, (left, right) = plt.subplots(1, 2, figsize=(12.8, 5.5), layout="constrained")
    left.hist(slopes, bins=21, color="#18718d", alpha=0.84, label="Lineup-constrained shuffles")
    left.axvline(null["observed_slope"], color="#a6502c", linewidth=2, label="Observed")
    left.set(title="What pass volumes alone reproduce", xlabel="Entropy slope per +0.01 hub concentration",
             ylabel="Shuffled seasons")
    left.legend(frameon=False, fontsize=9)

    years = scores.loc[scores["season"].ne("pooled")]
    improvements = years["rmse_improvement"]
    colors = np.where(improvements.ge(0), "#18718d", "#a6502c")
    right.bar(years["season"], improvements, color=colors)
    right.axhline(0, color="#304452", linewidth=0.8)
    right.set(title="Does early hub concentration improve the forecast?",
              ylabel="RMSE reduction after adding hub concentration")
    right.tick_params(axis="x", labelrotation=45)
    right.grid(axis="y", alpha=0.18)
    figure.suptitle("Two checks on the passing-hub regression", fontsize=16)
    figure.savefig(output / "star_entropy_followup.png", dpi=165)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=HERE / "data")
    parser.add_argument("--output", type=Path, default=HERE / "output")
    args = parser.parse_args()
    sources = DataSources(args.cache, offline=True)
    baseline = pd.read_csv(args.output / "statsbomb_team_season_metrics.csv")
    slopes, null = run_null(sources, baseline, args.output)
    scores = run_temporal(sources, args.output)
    plot_followup(slopes, null, scores, args.output)
    pooled = scores.iloc[-1].to_dict()
    pooled["seasons_with_lower_rmse"] = int(scores.iloc[:-1]["rmse_improvement"].gt(0).sum())
    summary = {"shuffle": null, "forecast": pooled}
    with (args.output / "star_entropy_followup_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
