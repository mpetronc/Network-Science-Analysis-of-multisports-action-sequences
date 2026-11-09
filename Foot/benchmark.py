"""Check OpenPlay receiver guesses against the overlapping StatsBomb fixtures."""

from collections import Counter, defaultdict
import json

import numpy as np
import pandas as pd

from data_sources import read_json
from metrics import count_pairs, network_metrics, pass_matrix
from openplay import read_events, read_lineups, receiver_candidates

CLUB_ALIASES = {
    "Bournemouth": "AFC Bournemouth", "Leicester": "Leicester City", "Man City": "Manchester City",
    "Man Utd": "Manchester United", "Newcastle": "Newcastle United", "Norwich": "Norwich City",
    "Stoke": "Stoke City", "Swansea": "Swansea City", "Tottenham": "Tottenham Hotspur",
    "WBA": "West Bromwich Albion", "West Ham": "West Ham United",
}


def club(name):
    return CLUB_ALIASES.get(name, name)


def match_passes(recorded, other):
    """Pair only reciprocal closest passes from the same player and half."""
    alignable = other.dropna(subset=["passer_id", "clock", "x", "y", "end_x", "end_y"])
    grouped = {key: group for key, group in alignable.groupby(["period", "passer_id"], sort=False)}
    result = []
    for key, group in recorded.groupby(["period", "passer_id"], sort=False):
        candidate = grouped.get(key)
        if candidate is None:
            continue
        dt = np.abs(group["clock"].to_numpy()[:, None] - candidate["clock"].to_numpy(dtype=float))
        start = np.hypot(group["x"].to_numpy()[:, None] - candidate["x"].to_numpy(),
                         group["y"].to_numpy()[:, None] - candidate["y"].to_numpy())
        end = np.hypot(group["end_x"].to_numpy()[:, None] - candidate["end_x"].to_numpy(),
                       group["end_y"].to_numpy()[:, None] - candidate["end_y"].to_numpy())
        cost = np.where((dt <= 3) & (start <= 12) & (end <= 18), dt * 3 + start + end, np.inf)
        row, column = np.arange(len(group)), cost.argmin(axis=1)
        keep = np.isfinite(cost[row, column]) & (cost.argmin(axis=0)[column] == row)
        result.extend(zip(group.index[row[keep]], candidate.index[column[keep]]))
    return result


def shirt_map(statsbomb_lineups, openplay_lineups):
    shirts = {}
    for lineup in statsbomb_lineups:
        team = lineup["team_name"]
        for p in lineup["lineup"]:
            key = (team, p["jersey_number"])
            if key in shirts:
                raise ValueError(f"Ambiguous shirt number {key}.")
            shirts[key] = p["player_id"]
    return {
        int(p.player_id): shirts[club(p.team), p.shirt_no]
        for p in openplay_lineups.itertuples(index=False)
        if (club(p.team), p.shirt_no) in shirts
    }


def agreement(recorded, inferred, players):
    _, actual = pass_matrix(recorded, players)
    _, guessed = pass_matrix(inferred, players)
    actual_team, actual_player = network_metrics(actual)
    guess_team, guess_player = network_metrics(guessed)
    ranks = actual_player["pagerank"].rank().corr(guess_player["pagerank"].rank())
    hubs = bool(actual_player["pagerank"].idxmax() == guess_player["pagerank"].idxmax()) if inferred and recorded else False
    return {
        "recorded_conditional_entropy": actual_team["conditional_receiver_entropy"],
        "inferred_conditional_entropy": guess_team["conditional_receiver_entropy"],
        "entropy_delta": guess_team["conditional_receiver_entropy"] - actual_team["conditional_receiver_entropy"],
        "pagerank_hhi_delta": guess_team["pagerank_hhi"] - actual_team["pagerank_hhi"],
        "hub_rank_spearman": ranks,
        "same_top_hub": hubs,
    }


def ratio(num, denominator):
    return float(num / denominator) if denominator else np.nan


def run_benchmark(sources, fixtures, statsbomb_matches, openplay_matches, output):
    lineup_paths = sources.statsbomb_paths(fixtures, kind="lineups")
    events = {mid: group for mid, group in read_events(sources, 2015).groupby("match_id", sort=False)}
    lineups = {mid: group for mid, group in read_lineups(sources, season="1516").groupby("match_id", sort=False)}
    fixture_map = {
        (club(m.home_team), club(m.away_team)): (m.match_id, m.date)
        for m in openplay_matches.loc[openplay_matches["season"].eq("1516")].itertuples(index=False)
    }

    by_team = defaultdict(Counter)
    all_actual, all_inferred, players = defaultdict(Counter), defaultdict(Counter), defaultdict(set)
    match_rows = []
    for fixture, match, lineup_path in zip(fixtures, statsbomb_matches, lineup_paths):
        home = fixture["home_team"]["home_team_name"]
        away = fixture["away_team"]["away_team_name"]
        op_id, op_date = fixture_map[(home, away)]
        if abs((pd.Timestamp(op_date) - pd.Timestamp(match.date)).days) > 1:
            raise ValueError(f"Fixture date mismatch for {home} vs {away}.")
        identities = shirt_map(read_json(lineup_path), lineups[op_id])
        guesses = receiver_candidates(events[op_id])
        guesses["passer_id"] = guesses["player_id"].map(identities)
        guesses["receiver_id"] = guesses["guess"].map(identities)
        guesses["club"] = guesses["team"].astype(str).map(club)

        for team, name in match.teams.items():
            recorded = match.passes.loc[match.passes["team_id"].eq(team)]
            other = guesses.loc[guesses["club"].eq(name)]
            pairs = match_passes(recorded, other)
            actual_ids, candidate_ids = zip(*pairs) if pairs else ([], [])
            actual = recorded.loc[list(actual_ids)]
            inferred = other.loc[list(candidate_ids), "receiver_id"].to_numpy()
            eligible = np.isfinite(inferred)
            correct = inferred == actual["receiver_id"].to_numpy()
            long = np.hypot(actual["x"] - actual["end_x"], actual["y"] - actual["end_y"]).to_numpy() >= 25
            squad = set(match.players[team])
            known = other.loc[other["passer_id"].isin(squad) & other["receiver_id"].isin(squad)]
            edges = count_pairs(recorded)
            inferred_edges = count_pairs(known)
            counts = {
                "recorded_passes": len(recorded), "openplay_passes": len(other),
                "aligned_passes": len(pairs), "aligned_with_guess": int(eligible.sum()),
                "correct_guess": int(correct.sum()), "network_passes_inferred": len(known),
                "aligned_long_with_guess": int((eligible & long).sum()),
                "correct_long_guess": int((correct & long).sum()),
            }
            by_team[name].update(counts)
            all_actual[name].update(edges)
            all_inferred[name].update(inferred_edges)
            players[name].update(squad)
            match_rows.append({
                "team": name, "date": match.date, "statsbomb_match_id": match.match_id,
                "openplay_match_id": op_id, **counts,
            })

    rows = []
    for name in sorted(by_team):
        counts = by_team[name]
        rows.append({
            "team": name, **counts,
            "openplay_alignment_coverage": ratio(counts["aligned_passes"], counts["openplay_passes"]),
            "openplay_inference_coverage": ratio(counts["network_passes_inferred"], counts["openplay_passes"]),
            "receiver_agreement": ratio(counts["correct_guess"], counts["aligned_with_guess"]),
            "long_pass_receiver_agreement": ratio(counts["correct_long_guess"], counts["aligned_long_with_guess"]),
            **agreement(all_actual[name], all_inferred[name], players[name]),
        })
    teams = pd.DataFrame(rows)
    teams.to_csv(output / "receiver_benchmark_teams.csv", index=False)
    pd.DataFrame(match_rows).to_csv(output / "receiver_benchmark_matches.csv", index=False)
    total = teams.sum(numeric_only=True)
    summary = {
        "matched_fixtures": len(fixtures),
        "openplay_passes": int(total["openplay_passes"]),
        "statsbomb_passes": int(total["recorded_passes"]),
        "aligned_passes": int(total["aligned_passes"]),
        "aligned_with_guess": int(total["aligned_with_guess"]),
        "alignment_coverage": ratio(total["aligned_passes"], total["openplay_passes"]),
        "inference_coverage": ratio(total["network_passes_inferred"], total["openplay_passes"]),
        "receiver_agreement_on_aligned_guesses": ratio(total["correct_guess"], total["aligned_with_guess"]),
        "median_club_absolute_entropy_error": float(teams["entropy_delta"].abs().median()),
        "median_club_hub_rank_agreement": float(teams["hub_rank_spearman"].median()),
    }
    summary["meets_extension_criteria"] = bool(
        summary["alignment_coverage"] >= 0.80
        and summary["inference_coverage"] >= 0.80
        and summary["receiver_agreement_on_aligned_guesses"] >= 0.95
        and summary["median_club_absolute_entropy_error"] <= 0.02
        and summary["median_club_hub_rank_agreement"] >= 0.95
    )
    with (output / "receiver_benchmark_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
    return summary
