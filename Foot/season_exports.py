"""Export Premier League networks for the six seasons that overlap with the NBA study."""

import argparse
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from data_sources import DataSources, season_name
from plots import draw_football_network
from season_reports import (
    COHORTS, MINUTES, club_stability, inferred_club_networks, inferred_seasons,
    plot_club, plot_seasons, result_summary, shot_regressions,
)

HERE = Path(__file__).resolve().parent
YEARS = range(2020, 2026)
SOURCE = "OpenPlay (Noah Bair)"
CASE_SEASON = "2021-22"
CASE_CLUB = "Man City"
CASE_LABEL = "Manchester City"


def slug(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def player_export(network, passes, appearances, shots):
    players = network.players.drop(columns="known_outgoing_profile").rename(
        columns={"minutes": "average_minutes"}
    )
    players = players.join(appearances[["appearances", "total_minutes"]], on="player_id")
    coverage = passes.groupby("player_id", observed=True)["guess"].agg(
        completed_passes="size", passes_with_inferred_receiver="count"
    )
    players = players.join(coverage, on="player_id")
    coverage_columns = ["completed_passes", "passes_with_inferred_receiver"]
    players[coverage_columns] = players[coverage_columns].fillna(0).astype(int)
    players["unresolved_receivers"] = players["completed_passes"] - players["passes_with_inferred_receiver"]
    players["non_penalty_shots"] = players["player_id"].map(shots).fillna(0).astype(int)
    players["receiver_source"] = "inferred"

    ids = players["player_id"]
    selected = passes.loc[passes["player_id"].isin(ids) & passes["guess"].isin(ids)]
    starts = selected[["player_id", "x", "y"]]
    ends = selected[["guess", "end_x", "end_y"]].set_axis(["player_id", "x", "y"], axis=1)
    locations = pd.concat([starts, ends], ignore_index=True)
    valid = locations["x"].between(0, 100) & locations["y"].between(0, 100)
    coordinates = locations.loc[valid].groupby("player_id")[["x", "y"]].mean()
    return players.join(coordinates, on="player_id")


def export_network(folder, keys, coverage, network, players, *, match_label=None):
    folder.mkdir(parents=True, exist_ok=True)
    ids = players["player_id"].to_numpy()
    names = players["player"].to_numpy()
    matrix = network.matrix.astype(np.int64)
    passing = pd.DataFrame(matrix, index=ids, columns=ids)
    passing.rename_axis("passer_id").to_csv(folder / "passing_matrix.csv")

    a, b = np.nonzero(matrix)
    edges = pd.DataFrame({
        "passer_id": ids[a], "passer_name": names[a], "receiver_id": ids[b],
        "receiver_name": names[b], "pass_count": matrix[a, b],
    }).assign(**keys, receiver_source="inferred", source=SOURCE)
    edges.to_csv(folder / "passes.csv", index=False)
    players.to_csv(folder / "players.csv", index=False)
    players[["player_id", "player", "appearances", "total_minutes", "average_minutes",
             "non_penalty_shots"]].to_csv(folder / "roster_stats.csv", index=False)

    values = network.metrics.copy()
    values["inferred_network_passes"] = values.pop("recorded_passes")
    metrics = keys | coverage | values | {"source": SOURCE}
    pd.DataFrame([metrics]).to_csv(folder / "metrics.csv", index=False)
    pd.Series(metrics).to_json(folder / "metrics.json", indent=2, force_ascii=False)
    draw_football_network(keys["team"], keys["season"], ids, matrix, players,
                          folder / "passing_network", source=SOURCE,
                          receiver_source="inferred", cohort=keys["cohort"],
                          match_label=match_label)
    return metrics, edges


def export_case(season, output, cutoff):
    games = season.matches.loc[
        season.matches["home_team"].eq(CASE_CLUB) | season.matches["away_team"].eq(CASE_CLUB)
    ].sort_values(["date", "kickoff", "match_id"])
    club_passes = season.passes.loc[season.passes["team"].eq(CASE_CLUB)]
    team_id, = club_passes["team_id"].unique()
    squad = season.roster.loc[season.roster["team_id"].eq(team_id)]
    club_shots = season.shots.loc[season.shots["team_id"].eq(team_id)]
    game_rows, player_rows = [], []
    for number, game in enumerate(games.itertuples(index=False), start=1):
        passes = club_passes.loc[club_passes["match_id"].eq(game.match_id)]
        roster = squad.loc[squad["match_id"].eq(game.match_id)]
        shots = club_shots.loc[club_shots["match_id"].eq(game.match_id), "player_id"].value_counts().to_dict()
        coverage, networks, appearances = inferred_club_networks(
            passes, roster, shots, cutoff, include_players=True
        )
        home = game.home_team == CASE_CLUB
        scored, conceded = (game.home_score, game.away_score) if home else (game.away_score, game.home_score)
        opponent = game.away_team if home else game.home_team
        keys = {"season": season.season, "team": CASE_LABEL, "team_id": int(team_id),
                "match_id": int(game.match_id), "match_number": number, "date": str(game.date),
                "opponent": opponent, "venue": "home" if home else "away",
                "goals_for": int(scored), "goals_against": int(conceded),
                "result": "W" if scored > conceded else "L" if scored < conceded else "D"}
        for cohort, network in networks.items():
            players = player_export(network, passes, appearances, shots)
            metrics, _ = export_network(output / cohort / f"match_{number:02d}",
                                        keys | {"cohort": cohort}, coverage, network, players,
                                        match_label=f"{'vs' if home else 'at'} {opponent}")
            game_rows.append(metrics)
            player_rows.append(players.assign(**keys, cohort=cohort))

    players = pd.concat(player_rows, ignore_index=True)
    games_with_stability = club_stability(pd.DataFrame(game_rows), players)
    games_with_stability.to_csv(output / "match_stability.csv", index=False)
    players.to_csv(output / "match_players.csv", index=False)
    result_summary(games_with_stability).to_csv(output / "result_summary.csv", index=False)
    plot_club(games_with_stability, CASE_LABEL, output, season=season.season,
              source="OpenPlay inferred receivers")
    return len(games)


def build_exports(sources, output, cutoff):
    rows, coverage_rows = [], []
    case_matches = 0
    for season in inferred_seasons(sources, YEARS):
        counts = {cohort: [] for cohort in COHORTS}
        edges = {cohort: [] for cohort in COHORTS}
        used_folders = set()
        clubs = season.passes.groupby(["team_id", "team"], observed=True, sort=True)
        for (team, name), passes in clubs:
            club = slug(name)
            if club in used_folders:
                raise ValueError(f"Ambiguous club directory for {name} in {season.season}.")
            used_folders.add(club)
            roster = season.roster.loc[season.roster["team_id"].eq(team)]
            shots = season.shots.loc[season.shots["team_id"].eq(team), "player_id"].value_counts().to_dict()
            coverage, networks, appearances = inferred_club_networks(
                passes, roster, shots, cutoff, include_players=True
            )
            for cohort, network in networks.items():
                players = player_export(network, passes, appearances, shots)
                keys = {"season": season.season, "team": name, "team_id": int(team), "cohort": cohort}
                folder = output / cohort / season.season
                metrics, table = export_network(folder / club, keys, coverage, network, players)
                counts[cohort].append(metrics)
                edges[cohort].append(table)

        if len(used_folders) != 20:
            raise ValueError(f"Expected 20 Premier League clubs for {season.season}, found {len(used_folders)}.")
        for cohort in COHORTS:
            folder = output / cohort / season.season
            pd.DataFrame(counts[cohort]).to_csv(folder / "team_metrics.csv", index=False)
            pd.concat(edges[cohort], ignore_index=True).to_csv(folder / "all_team_passes.csv", index=False)
            rows.extend(counts[cohort])
        season.matches.to_csv(output / f"matches_{season.season}.csv", index=False)
        coverage_rows.append(season.coverage)
        print(f"{season.season}: saved 20 club graphs and their filtered versions.", flush=True)
        if season.season == CASE_SEASON:
            case_matches = export_case(season, output / "manchester_city_2021-22", cutoff)
            print(f"{CASE_LABEL}: saved {case_matches} match networks in both versions.", flush=True)

    seasons = pd.DataFrame(rows)
    seasons.to_csv(output / "all_seasons_team_metrics.csv", index=False)
    pd.DataFrame(coverage_rows).to_csv(output / "coverage.csv", index=False)
    shot_regressions(seasons).to_csv(output / "shot_hhi_regressions.csv", index=False)
    plot_seasons(seasons, output)
    manifest = {
        "source": SOURCE,
        "receiver_source": "inferred",
        "benchmark_season": "2015-16",
        "seasons": [season_name(year) for year in YEARS],
        "unique_club_seasons": int(len(seasons) / len(COHORTS)),
        "club_season_networks": len(seasons),
        "included_fixtures": int(sum(row["used_matches"] for row in coverage_rows)),
        "minutes_cutoff": cutoff,
        "season_minutes_basis": "average minutes per appearance",
        "match_minutes_basis": "minutes in that match",
        "case_study": {"club": CASE_LABEL, "season": CASE_SEASON, "matches": case_matches},
        "edge_policy": "Weights count passes with an inferred receiver. Zero means no inferred passes for that pair.",
    }
    (output / "run_summary.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def publish(staging, output):
    if not output.exists():
        staging.replace(output)
        return
    summary = staging / "run_summary.json"
    for path in staging.rglob("*"):
        if path.is_file() and path != summary:
            destination = output / path.relative_to(staging)
            destination.parent.mkdir(parents=True, exist_ok=True)
            path.replace(destination)
    summary.replace(output / summary.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=HERE / "data")
    parser.add_argument("--output", type=Path, default=HERE / "output" / "premier_league_2020-21_to_2025-26")
    parser.add_argument("--min-minutes", type=float, default=MINUTES)
    args = parser.parse_args()
    if not 0 < args.min_minutes < 90:
        parser.error("--min-minutes must be between 0 and 90.")
    if args.output.exists() and not (args.output / "run_summary.json").is_file():
        parser.error("--output must be a new directory or an existing season export.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as folder:
        staging = Path(folder) / "export"
        staging.mkdir()
        build_exports(DataSources(args.cache, offline=True), staging, args.min_minutes)
        publish(staging, args.output)
    print(f"Premier League exports saved to {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
