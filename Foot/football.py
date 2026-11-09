"""Run the Premier League and NBA comparison from PyCharm or a terminal."""

import argparse
import json
from pathlib import Path

import pandas as pd

from benchmark import run_benchmark
from data_sources import DataSources, STATSBOMB_REF
from metrics import nba_exports
from openplay import read_matches, run_tactical
from plots import comparison, draw_network, openplay_prediction, possession_patterns
from regression import run_regression
from statsbomb import SEASON, predict_receivers, summarize_matches

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=HERE / "data")
    parser.add_argument("--output", type=Path, default=HERE / "output")
    parser.add_argument("--nba-season", default="2025-26")
    parser.add_argument("--nba-dir", type=Path, help="Directory containing NBA team passing_matrix.csv exports.")
    parser.add_argument("--offline", action="store_true", help="Use cached inputs only.")
    parser.add_argument("--statsbomb-only", action="store_true", help="Run the recorded-recipient study and NBA comparison.")
    parser.add_argument("--download-only", action="store_true", help="Cache the pinned public data files.")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    sources = DataSources(args.cache, offline=args.offline)
    if args.download_only:
        sources.preload()
        return
    nba_dir = args.nba_dir or HERE.parent / "NBA" / "nba_passing_output" / args.nba_season / "Regular_Season"
    args.output.mkdir(parents=True, exist_ok=True)
    nba, nba_players, nba_networks = nba_exports(nba_dir, season=args.nba_season)
    nba_players.to_csv(args.output / "nba_players.csv", index=False)
    fixtures, matches, football, networks, patterns = summarize_matches(sources, args.output)
    predict_receivers(matches, args.output)
    print(f"StatsBomb: {len(fixtures)} matches and {len(football)} club networks.", flush=True)

    teams = pd.concat([football, nba], ignore_index=True)
    teams["entropy_league_percentile"] = float("nan")
    teams["hub_league_percentile"] = float("nan")
    for _, group in teams.loc[teams["coverage_complete"]].groupby(["sport", "season"]):
        if len(group) >= 5:
            teams.loc[group.index, "entropy_league_percentile"] = group["conditional_receiver_entropy"].rank(pct=True)
            teams.loc[group.index, "hub_league_percentile"] = group["pagerank_concentration"].rank(pct=True)
    teams.to_csv(args.output / "nba_football_team_season_metrics.csv", index=False)

    scores = benchmark = inferred = None
    if not args.statsbomb_only:
        openplay_matches = read_matches(sources)
        benchmark = run_benchmark(sources, fixtures, matches, openplay_matches, args.output)
        print(f"Receiver benchmark: {benchmark['receiver_agreement_on_aligned_guesses']:.1%} agreement "
              "on aligned passes with a guess.", flush=True)
        scores = run_tactical(sources, openplay_matches, args.output,
                              infer_receivers=benchmark["meets_extension_criteria"])
        if benchmark["meets_extension_criteria"]:
            inferred = pd.read_csv(args.output / "openplay_inferred_team_networks.csv")

    if not args.no_plots:
        comparison(teams, args.output)
        possession_patterns(patterns, args.output)
        for name, (_, ids, matrix, players) in networks.items():
            draw_network(name, SEASON, ids, matrix, players, args.output / "football_networks", football=True)
        for name, (ids, matrix, players, complete) in nba_networks.items():
            draw_network(name, args.nba_season, ids, matrix, players, args.output / "nba_networks", complete=complete)
        if scores is not None:
            openplay_prediction(scores, args.output)

    run_regression(teams, inferred, args.output, make_plot=not args.no_plots)

    summary = {
        "statsbomb_ref": STATSBOMB_REF,
        "statsbomb_fixtures": len(fixtures),
        "nba_season": args.nba_season,
        "nba_team_count": len(nba),
        "nba_complete_profiles": int(nba["coverage_complete"].sum()),
        "openplay_release": "data-v0.1.0" if scores is not None else None,
        "openplay_team_seasons": len(scores) if scores is not None else None,
        "openplay_held_out_entries": int(scores["entries"].sum()) if scores is not None else None,
        "receiver_benchmark": benchmark,
    }
    with (args.output / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
    print(f"Results saved to {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
