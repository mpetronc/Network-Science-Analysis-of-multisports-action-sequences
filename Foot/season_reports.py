"""Football season and club reports built from the cached Premier League events."""

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_sources import DataSources, YEARS, event_file, read_json, season_name
from metrics import concentration, count_pairs, network_metrics, pass_matrix
from openplay import non_penalty_shots, read_events, read_matches, receiver_candidates
from regression import fit_regression
from statsbomb import SEASON, parse_match

HERE = Path(__file__).resolve().parent
MINUTES = 90 * 10 / 48
COHORTS = ("all_players", "minutes_filter")
METRICS = (
    ("receiver_entropy", "Receiver entropy"),
    ("conditional_receiver_entropy", "Conditional receiver entropy"),
    ("normalized_passing_mutual_information", "Passing mutual information (normalized)"),
    ("effective_receivers", "Effective receivers"),
)


def clock_seconds(clock):
    return sum(float(part) * 60 ** i for i, part in enumerate(reversed(clock.split(":"))))


def statsbomb_minutes(events, lineups):
    """Join lineup stints on a continuous match clock including stoppage time."""
    ends = defaultdict(float)
    for event in events:
        if event["type"]["name"] == "Half End":
            period = event["period"]
            ends[period] = max(ends[period], clock_seconds(event["timestamp"]))
    if min(ends[1], ends[2]) <= 0:
        raise ValueError("Both half-end events are required to reconstruct minutes.")
    offset = {1: 0, 2: ends[1]}

    def elapsed(clock, period):
        local = clock_seconds(clock) - (period - 1) * 45 * 60
        return offset[period] + np.clip(local, 0, ends[period])

    minutes = {}
    for team in lineups:
        for player in team["lineup"]:
            intervals = []
            for stint in player["positions"]:
                start = elapsed(stint["from"], stint["from_period"])
                end = elapsed(stint["to"], stint["to_period"]) if stint["to"] else ends[1] + ends[2]
                if end > start:
                    intervals.append((start, end))
            # A position change can overlap an adjacent stint at rounded event times.
            covered = last_end = 0.0
            for start, end in sorted(intervals):
                covered += max(0.0, end - max(start, last_end))
                last_end = max(last_end, end)
            if intervals:
                minutes[(team["team_id"], player["player_id"])] = float(covered / 60)
    return minutes


@dataclass
class CohortNetwork:
    metrics: dict
    players: pd.DataFrame
    matrix: np.ndarray


@dataclass
class InferredSeason:
    season: str
    matches: pd.DataFrame
    roster: pd.DataFrame
    passes: pd.DataFrame
    shots: pd.DataFrame
    coverage: dict


def summarize(passes, roster, shots, minutes, cutoff, *, include_players=False):
    rows = {}
    for cohort in COHORTS:
        selected = set(roster) if cohort == "all_players" else {
            player for player, played in minutes.items() if played >= cutoff and player in roster
        }
        pairs = {(a, b): n for (a, b), n in passes.items() if a in selected and b in selected}
        order, matrix = pass_matrix(pairs, selected)
        values, players = network_metrics(matrix)
        shooting = [shots.get(p, 0) for p in order]
        values.update(
            non_penalty_shots=sum(shooting),
            shot_hhi=concentration(shooting),
            all_network_passes=sum(passes.values()),
            minimum_minutes=0 if cohort == "all_players" else cutoff,
        )
        if include_players:
            players["player_id"] = order
            players["player"] = [roster[p] for p in order]
            players["minutes"] = [minutes.get(p, np.nan) for p in order]
        rows[cohort] = CohortNetwork(values, players, matrix)
    return rows


def recorded_reports(sources, cutoff, club):
    fixtures = sources.statsbomb_matches()
    events_files = sources.statsbomb_paths(fixtures)
    lineup_files = sources.statsbomb_paths(fixtures, kind="lineups")
    seasons = defaultdict(lambda: {"roster": {}, "passes": Counter(), "shots": Counter(),
                                   "minutes": Counter(), "appearances": Counter(), "matches": 0})
    match_rows, club_players = [], []
    for fixture, event_path, lineup_path in zip(fixtures, events_files, lineup_files):
        events = read_json(event_path)
        match = parse_match(fixture, events)
        all_minutes = statsbomb_minutes(events, read_json(lineup_path))
        home, away = (fixture[f"{side}_team"][f"{side}_team_id"] for side in ("home", "away"))
        goals = {home: fixture["home_score"], away: fixture["away_score"]}
        for team, name in match.teams.items():
            roster = match.players[team]
            minutes = {p: all_minutes[team, p] for p in roster if (team, p) in all_minutes}
            passes = count_pairs(match.passes.loc[match.passes["team_id"].eq(team)])
            season = seasons[team, name]
            season["roster"].update(roster)
            season["passes"].update(passes)
            season["shots"].update(match.shots[team])
            season["minutes"].update(minutes)
            season["appearances"].update(minutes.keys())
            season["matches"] += 1
            opponent = away if team == home else home
            margin = goals[team] - goals[opponent]
            keys = {"season": SEASON, "receiver_source": "recorded", "match_id": match.match_id,
                    "date": match.date, "team_id": team, "team": name, "opponent": match.teams[opponent],
                    "venue": "home" if team == home else "away", "goals_for": goals[team],
                    "goals_against": goals[opponent], "goal_difference": margin,
                    "result": "W" if margin > 0 else "L" if margin < 0 else "D",
                    "players_missing_minutes": len(roster) - len(minutes)}
            for cohort, network in summarize(
                passes, roster, match.shots[team], minutes, cutoff, include_players=name == club
            ).items():
                match_rows.append(keys | {"cohort": cohort} | network.metrics)
                if name == club:
                    club_players.append(network.players.assign(**keys, cohort=cohort))

    if not club_players:
        raise ValueError(f"Club {club!r} not found in the recorded StatsBomb season.")
    season_rows = []
    for (team, name), data in seasons.items():
        average = {p: data["minutes"][p] / games for p, games in data["appearances"].items()}
        for cohort, network in summarize(
            data["passes"], data["roster"], data["shots"], average, cutoff
        ).items():
            season_rows.append({
                "season": SEASON, "team_id": team, "team": name, "receiver_source": "recorded",
                "cohort": cohort, "matches": data["matches"],
                "players_missing_minutes": len(data["roster"]) - len(average),
                **network.metrics,
            })
    return pd.DataFrame(season_rows), pd.DataFrame(match_rows), pd.concat(club_players, ignore_index=True)


def inferred_seasons(sources, years=YEARS):
    matches = read_matches(sources)
    excluded = set(pd.read_csv(sources.openplay("data_issues.csv"))["match_id"].dropna().astype(int))
    lineups = pd.read_parquet(
        sources.openplay("lineups.parquet"),
        columns=["match_id", "team_id", "player_id", "player", "minutes_played", "played"],
        filters=[("comp", "==", "eng")],
    )
    for year in years:
        season = season_name(year)
        code = event_file(year).split("_")[-1][:4]
        fixtures = matches.loc[matches["season"].eq(code)]
        ids = set(fixtures["match_id"])
        events = read_events(sources, year)
        if set(events["match_id"]) != ids:
            raise ValueError(f"OpenPlay manifest disagrees with the events for {season}.")
        used = ids.difference(excluded)
        events = events.loc[events["match_id"].isin(used)]
        roster = lineups.loc[lineups["played"] & lineups["match_id"].isin(used)]
        guesses = receiver_candidates(events)
        shots = non_penalty_shots(events)[["match_id", "team_id", "player_id"]]
        coverage = {"season": season, "scheduled_matches": 380, "available_matches": len(ids),
                    "used_matches": len(used), "excluded_flagged_matches": len(ids & excluded)}
        yield InferredSeason(season, fixtures.loc[fixtures["match_id"].isin(used)], roster,
                             guesses, shots, coverage)


def inferred_club_networks(passes, squad, shots, cutoff, *, include_players=False):
    appearances = squad.groupby("player_id", observed=True).agg(
        player=("player", "last"), appearances=("match_id", "nunique"),
        total_minutes=("minutes_played", "sum"), average_minutes=("minutes_played", "mean"),
    )
    names = appearances["player"].to_dict()
    minutes = appearances["average_minutes"].dropna().to_dict()
    counts = passes.groupby(["player_id", "guess"], observed=True).size().to_dict()
    for p in set(passes["player_id"]) | set(passes["guess"].dropna()):
        names.setdefault(p, str(p))
    summary = {
        "receiver_source": "inferred", "matches": passes["match_id"].nunique(),
        "players_missing_minutes": len(names) - len(minutes),
        "successful_passes": len(passes), "passes_with_inferred_receiver": sum(counts.values()),
        "inferred_receiver_share": sum(counts.values()) / len(passes) if len(passes) else np.nan,
    }
    networks = summarize(counts, names, shots, minutes, cutoff, include_players=include_players)
    return summary, networks, appearances


def inferred_reports(sources, cutoff):
    rows = []
    for season in inferred_seasons(sources):
        for (team, name), passes in season.passes.groupby(["team_id", "team"], sort=False, observed=True):
            squad = season.roster.loc[season.roster["team_id"].eq(team)]
            shots = season.shots.loc[season.shots["team_id"].eq(team), "player_id"].value_counts().to_dict()
            summary, networks, _ = inferred_club_networks(passes, squad, shots, cutoff)
            for cohort, network in networks.items():
                rows.append({"season": season.season, "team_id": team, "team": name, "cohort": cohort,
                             **summary, **network.metrics})
        print(f"OpenPlay {season.season}: built both minutes cohorts.", flush=True)
    return pd.DataFrame(rows)


def js_divergence(first, second):
    """Base-2 Jensen–Shannon divergence on a common set of receivers."""
    a, b = (np.asarray(values, dtype=float) for values in (first, second))
    if a.sum() <= 0 or b.sum() <= 0:
        return np.nan
    a, b = a / a.sum(), b / b.sum()
    midpoint = (a + b) / 2
    return float(sum(0.5 * np.sum(p[p > 0] * np.log2(p[p > 0] / midpoint[p > 0])) for p in (a, b)))


def club_stability(games, players):
    rows = []
    for cohort, ordered in games.groupby("cohort", sort=False):
        distributions = {
            match: data.set_index("player_id")["passes_received"]
            for match, data in players.loc[players["cohort"].eq(cohort)].groupby("match_id")
        }
        previous = None
        for game in ordered.sort_values(["date", "match_id"]).itertuples(index=False):
            current = distributions.get(game.match_id, pd.Series(dtype=float))
            row = {"cohort": cohort, "match_id": game.match_id,
                   "previous_match_id": previous[0] if previous else None,
                   "jsd_all_players": np.nan, "jsd_returning_players": np.nan,
                   "roster_turnover_fraction": np.nan}
            if previous:
                old = previous[1]
                union = old.index.union(current.index)
                common = old.index.intersection(current.index)
                row["jsd_all_players"] = js_divergence(
                    old.reindex(union, fill_value=0), current.reindex(union, fill_value=0)
                )
                if len(common) >= 2:
                    row["jsd_returning_players"] = js_divergence(old.loc[common], current.loc[common])
                if len(union):
                    row["roster_turnover_fraction"] = 1 - len(common) / len(union)
            rows.append(row)
            previous = (game.match_id, current)
    return games.merge(pd.DataFrame(rows), on=["cohort", "match_id"], validate="one_to_one")


def result_summary(club_matches):
    return club_matches.groupby(["cohort", "result"], as_index=False).agg(
        matches=("match_id", "size"), receiver_entropy=("receiver_entropy", "mean"),
        normalized_passing_mutual_information=("normalized_passing_mutual_information", "mean"),
        shot_hhi=("shot_hhi", "mean"), mean_receiving_change=("jsd_all_players", "mean"),
        valid_stability_pairs=("jsd_all_players", "count"),
    )


def shot_regressions(seasons):
    rows = []
    for (source, cohort), sample in seasons.groupby(["receiver_source", "cohort"]):
        controls = ("season",) if source == "inferred" else ()
        for outcome in ("receiver_entropy", "normalized_passing_mutual_information"):
            usable = sample.dropna(subset=["shot_hhi", outcome])
            fit = fit_regression(usable, controls=controls, cluster="team_id",
                                 predictor="shot_hhi", outcome=outcome)
            if fit is None:
                continue
            rows.append({
                "receiver_source": source, "cohort": cohort, "outcome": outcome,
                "controls": "season" if controls else "none", "observations": len(usable),
                "clubs": usable["team_id"].nunique(), "slope_per_0_01_shot_hhi": fit.slope,
                "ci95_low": fit.interval[0] if fit.interval else np.nan,
                "ci95_high": fit.interval[1] if fit.interval else np.nan,
            })
    return pd.DataFrame(rows)


def plot_seasons(seasons, output):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.4), layout="constrained")
    styles = {"all_players": ("#18718d", "All players"), "minutes_filter": ("#ad673c", "Minutes filter")}
    for ax, (metric, label) in zip(axes.flat, METRICS):
        for cohort, (color, name) in styles.items():
            part = seasons.loc[seasons["cohort"].eq(cohort)]
            inferred = part.loc[part["receiver_source"].eq("inferred")].groupby("season")[metric].median()
            recorded = part.loc[part["receiver_source"].eq("recorded")].groupby("season")[metric].median()
            ax.plot(inferred.index, inferred, color=color, marker=".", label=name)
            ax.scatter(recorded.index, recorded, color=color, edgecolor="white", marker="D", zorder=3)
        ax.set_ylabel(label)
        ax.tick_params(axis="x", labelrotation=45, labelsize=8)
        ax.grid(alpha=0.16)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Premier League passing networks by season", fontsize=16)
    note = "Club medians · OpenPlay inferred receivers"
    if seasons["receiver_source"].eq("recorded").any():
        note += " · Diamonds show recorded StatsBomb receivers"
    fig.supxlabel(note, fontsize=9)
    fig.savefig(output / "season_trends.png", dpi=165)
    plt.close(fig)


def plot_club(games, club, output, *, season=SEASON, source="StatsBomb"):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.3), layout="constrained")
    styles = {"all_players": ("#18718d", "All players"), "minutes_filter": ("#ad673c", "Minutes filter")}
    panels = [METRICS[0], METRICS[2], ("shot_hhi", "Non-penalty shot concentration (HHI)"),
              ("jsd_all_players", "Receiving-share change from previous match (JSD)")]
    for ax, (metric, label) in zip(axes.flat, panels):
        for cohort, data in games.groupby("cohort", sort=False):
            ordered = data.sort_values(["date", "match_id"])
            color, name = styles[cohort]
            ax.plot(np.arange(1, len(ordered) + 1), ordered[metric], color=color, marker=".", label=name)
        ax.set(xlabel="Match number", ylabel=label)
        ax.grid(alpha=0.16)
    axes[0, 0].legend(frameon=False)
    fig.suptitle(f"{club} · Premier League {season} · {source}", fontsize=16)
    fig.savefig(output / "club_match_trends.png", dpi=165)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=HERE / "data")
    parser.add_argument("--output", type=Path, default=HERE / "output" / "season_reports")
    parser.add_argument("--club", default="Leicester City")
    parser.add_argument("--min-minutes", type=float, default=MINUTES)
    args = parser.parse_args()
    if not 0 < args.min_minutes < 90:
        parser.error("--min-minutes must be between 0 and 90.")
    output = args.output
    sources = DataSources(args.cache, offline=True)
    recorded, matches, players = recorded_reports(sources, args.min_minutes, args.club)
    print(f"StatsBomb: {len(matches) // 2} team-matches across both minutes cohorts.", flush=True)
    seasons = pd.concat([recorded, inferred_reports(sources, args.min_minutes)], ignore_index=True)
    club_matches = club_stability(matches.loc[matches["team"].eq(args.club)], players)
    outcomes = result_summary(club_matches)
    output.mkdir(parents=True, exist_ok=True)
    seasons.to_csv(output / "team_season_metrics.csv", index=False)
    matches.to_csv(output / "statsbomb_team_match_metrics.csv", index=False)
    club_matches.to_csv(output / "club_match_stability.csv", index=False)
    players.to_csv(output / "club_match_players.csv", index=False)
    outcomes.to_csv(output / "club_result_summary.csv", index=False)
    shot_regressions(seasons).to_csv(output / "shot_hhi_regressions.csv", index=False)
    plot_seasons(seasons, output)
    plot_club(club_matches, args.club, output)
    audit = {
        "club": args.club, "recorded_season": SEASON, "min_minutes": args.min_minutes,
        "season_threshold_unit": "average minutes per appearance",
        "match_threshold_unit": "minutes in the match",
        "recorded_team_matches_per_cohort": len(matches) // 2,
        "inferred_club_seasons_per_cohort": int(seasons["receiver_source"].eq("inferred").sum() // 2),
        "club_matches_per_cohort": len(club_matches) // 2,
        "recorded_match_players_missing_minutes": int(
            matches.loc[matches["cohort"].eq("all_players"), "players_missing_minutes"].sum()
        ),
    }
    (output / "report_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(f"Football reports saved to {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
