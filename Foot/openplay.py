"""Chronological prediction of final-third pass destinations in the Premier League."""

from collections import defaultdict

import numpy as np
import pandas as pd
from pyarrow.parquet import ParquetFile

from data_sources import YEARS, event_file, season_name
from metrics import concentration, network_metrics, pass_matrix

PERIODS = ["FirstHalf", "SecondHalf"]
CONTROLLED = {
    "Pass", "BallTouch", "BallRecovery", "TakeOn", "Goal", "SavedShot",
    "MissedShots", "ShotOnPost", "Interception", "Clearance", "KeeperPickup",
    "Claim", "Dispossessed", "Carry", "Dribble",
}
EVENT_COLUMNS = [
    "match_id", "team_id", "team", "player_id", "period", "minute", "second",
    "type", "outcome", "x", "y", "end_x", "end_y", "is_shot", "play_pattern",
]


def read_events(sources, year):
    path = sources.openplay(event_file(year))
    file = ParquetFile(path)
    columns = EVENT_COLUMNS + [name for name in ("q_OwnGoal",) if name in file.schema_arrow.names]
    events = file.read(columns=columns).to_pandas()
    events = events.loc[events["period"].isin(PERIODS)].copy()
    events["clock"] = events["minute"].astype("Int32") * 60 + events["second"].astype("Int32")
    return events


def read_matches(sources):
    return pd.read_parquet(
        sources.openplay("matches.parquet"),
        columns=["match_id", "season", "date", "kickoff", "home_team", "away_team", "home_score", "away_score"],
        filters=[("comp", "==", "eng")],
    )


def read_lineups(sources, *, season=None):
    filters = [("comp", "==", "eng")]
    if season:
        filters.append(("season", "==", season))
    return pd.read_parquet(
        sources.openplay("lineups.parquet"),
        columns=["match_id", "team_id", "team", "player_id", "player", "shirt_no", "played"],
        filters=filters,
    )


def successful_passes(events):
    return events["type"].eq("Pass") & events["outcome"].eq("Successful")


def receiver_candidates(events):
    """Use the next plausible on-ball action or abstain when the evidence is weak."""
    stops = {"End", "Foul", "OffsideGiven", "CornerAwarded", "SubstitutionOff"}
    on_ball = events["type"].isin(CONTROLLED) & events["player_id"].notna()
    action = events.loc[on_ball | events["type"].isin(stops)].copy()
    next_event = action.groupby(["match_id", "period"], sort=False, observed=True)
    next_player = next_event["player_id"].shift(-1)
    next_team = next_event["team_id"].shift(-1)
    interval = next_event["clock"].shift(-1) - action["clock"]
    distance = np.hypot(next_event["x"].shift(-1) - action["end_x"],
                        next_event["y"].shift(-1) - action["end_y"])
    plausible = (
        next_event["type"].shift(-1).isin(CONTROLLED) & next_team.eq(action["team_id"])
        & next_player.ne(action["player_id"]) & interval.between(0, 8) & (distance <= 14)
    ).fillna(False)
    action["guess"] = next_player.where(plausible)
    passes = action.loc[successful_passes(action)].copy()
    passes["period"] = passes["period"].map({"FirstHalf": 1, "SecondHalf": 2}).astype(int)
    return passes


def channel(y):
    """0, 1, 2 are right, centre, left from the attacking team's view."""
    return np.minimum((y.to_numpy(dtype=float) * 3 / 100).astype(int), 2)


def final_third_entries(events):
    touch = events.loc[events["type"].isin(CONTROLLED) & events["player_id"].notna()].copy()
    previous = touch.groupby(["match_id", "period"], sort=False, observed=True)
    prior_team = previous["team_id"].shift()
    prior_type = previous["type"].shift().astype("string")
    interval = touch["clock"] - previous["clock"].shift()
    recent = interval.between(0, 12)
    touch["previous"] = prior_type.where(prior_team.eq(touch["team_id"]), "Opposition touch")
    touch["previous"] = touch["previous"].where(recent, "No recent touch").fillna("No recent touch")

    valid = touch[["x", "y", "end_x", "end_y"]].apply(lambda values: values.between(0, 100)).all(axis=1)
    boundary = 200 / 3
    entry = touch.loc[
        successful_passes(touch) & valid & touch["x"].lt(boundary) & touch["end_x"].ge(boundary),
        ["match_id", "team_id", "team", "x", "y", "end_y", "previous"],
    ].copy()
    entry["depth"] = np.minimum((entry["x"] * 6 / 100).astype(int), 3)
    entry["origin"] = channel(entry["y"])
    entry["destination"] = channel(entry["end_y"])
    return entry


class EntryModel:
    """A frequency model that backs off to broader groups when data are sparse."""

    def __init__(self):
        self.counts = defaultdict(lambda: np.zeros(3, dtype=float))

    @staticmethod
    def keys(team, depth, origin, previous):
        return [(), (depth, origin), (team, depth, origin), (team, depth, origin, previous)]

    def fit(self, entries):
        columns = ["team_id", "depth", "origin", "previous", "destination"]
        groups = entries.groupby(columns, observed=True, sort=False).size()
        for (team, depth, origin, previous, destination), count in groups.items():
            for key in self.keys(team, depth, origin, previous):
                self.counts[key][destination] += count

    def evaluate(self, entries):
        columns = ["team_id", "team", "depth", "origin", "previous", "destination"]
        groups = entries.groupby(columns, observed=True, sort=False).size().reset_index(name="entries")
        rows = []
        for team, name, depth, origin, previous, destination, count in groups.itertuples(index=False):
            probabilities = np.full(3, 1 / 3)
            keys = self.keys(team, depth, origin, previous)
            for key in keys[:-1]:
                values = self.counts.get(key)
                if values is not None:
                    probabilities = (values + 12 * probabilities) / (values.sum() + 12)
            baseline = probabilities[destination]
            values = self.counts.get(keys[-1])
            if values is not None:
                probabilities = (values + 12 * probabilities) / (values.sum() + 12)
            rows.append((team, name, count, -count * np.log2(baseline),
                         -count * np.log2(probabilities[destination])))
        result = pd.DataFrame(rows, columns=["team_id", "team", "entries", "baseline_bits", "context_bits"])
        totals = result.groupby(["team_id", "team"], observed=True, as_index=False).sum()
        totals["baseline_bits"] /= totals["entries"]
        totals["context_bits"] /= totals["entries"]
        totals["context_gain_bits"] = totals["baseline_bits"] - totals["context_bits"]
        return totals


def non_penalty_shots(events):
    own_goals = events["q_OwnGoal"].fillna(False).astype(bool) if "q_OwnGoal" in events else pd.Series(False, index=events.index)
    return events.loc[events["is_shot"] & events["play_pattern"].ne("Penalty") & ~own_goals
                      & events["player_id"].notna()]


def shot_concentration(events, match_ids):
    shots = non_penalty_shots(events.loc[events["match_id"].isin(match_ids)])
    counts = shots.groupby(["team_id", "player_id"], observed=True).size()
    return pd.DataFrame([
        {"team_id": team, "early_shots": int(count.sum()), "shot_hhi": concentration(count)}
        for team, count in counts.groupby(level="team_id", observed=True)
    ], columns=["team_id", "early_shots", "shot_hhi"])


def inferred_networks(events, lineups, season):
    guesses = receiver_candidates(events)
    team_rows, player_rows = [], []
    squads = lineups.loc[lineups["played"] & lineups["match_id"].isin(events["match_id"])]
    for (team, name), passes in guesses.groupby(["team_id", "team"], sort=False, observed=True):
        roster = squads.loc[squads["team_id"].eq(team)].set_index("player_id")["player"].to_dict()
        roster_ids = set(roster) | set(passes["player_id"]) | set(passes["guess"].dropna())
        counts = passes.groupby(["player_id", "guess"], observed=True).size()
        order, matrix = pass_matrix(counts.to_dict(), roster_ids)
        values, players = network_metrics(matrix)
        keys = {"sport": "Premier League", "season": season, "team_id": team, "team": name}
        inferred_passes = values.pop("recorded_passes")
        team_rows.append(keys | values | {
            "matches": passes["match_id"].nunique(),
            "successful_passes": len(passes),
            "inferred_passes": inferred_passes,
            "inferred_receiver_share": inferred_passes / len(passes),
            "receiver_source": "inferred",
        })
        players["player_id"] = order
        players["player"] = [roster.get(pid, str(pid)) for pid in order]
        players = players.drop(columns="known_outgoing_profile")
        player_rows.append(players.assign(**keys, receiver_source="inferred"))
    return team_rows, player_rows


def run_tactical(sources, matches, output, *, infer_receivers=False):
    model = EntryModel()
    results, coverage, inferred_teams, inferred_players = [], [], [], []
    lineups = read_lineups(sources) if infer_receivers else None
    excluded = pd.read_csv(sources.openplay("data_issues.csv"))
    # Exclude provider-flagged match files from fitting and testing.
    excluded_ids = set(excluded["match_id"].dropna().astype(int))
    for year in YEARS:
        season = season_name(year)
        code = event_file(year).split("_")[-1][:4]
        season_matches = matches.loc[matches["season"].eq(code)]
        events = read_events(sources, year)
        match_ids = set(events["match_id"])
        if match_ids != set(season_matches["match_id"]):
            raise ValueError(f"{season}: OpenPlay event and match manifests disagree.")
        usable = season_matches.loc[~season_matches["match_id"].isin(excluded_ids)].sort_values(
            ["date", "kickoff", "match_id"]
        )
        events = events.loc[events["match_id"].isin(usable["match_id"])]
        if infer_receivers:
            team_rows, player_rows = inferred_networks(events, lineups, season)
            inferred_teams.extend(team_rows)
            inferred_players.extend(player_rows)
        midpoint = usable.iloc[len(usable) // 2]["date"]
        early = set(usable.loc[usable["date"].lt(midpoint), "match_id"])
        late = set(usable.loc[usable["date"].ge(midpoint), "match_id"])
        entries = final_third_entries(events)
        training = entries.loc[entries["match_id"].isin(early)]
        testing = entries.loc[entries["match_id"].isin(late)]
        model.fit(training)
        scores = model.evaluate(testing)
        scores = scores.merge(shot_concentration(events, early), on="team_id", how="left")
        results.append(scores.assign(season=season, test_from=str(midpoint)))
        model.fit(testing)
        coverage.append({
            "season": season, "scheduled_matches": 380, "available_matches": len(match_ids),
            "excluded_provider_flags": len(match_ids & excluded_ids), "used_matches": len(usable),
            "first_half_matches": len(early), "held_out_matches": len(late),
            "training_entries_this_season": len(training), "held_out_entries": len(testing),
            "test_from": str(midpoint),
        })
        print(f"OpenPlay {season}: {len(testing):,} held-out final-third entries.", flush=True)

    combined = pd.concat(results, ignore_index=True)
    combined.to_csv(output / "openplay_team_season_prediction.csv", index=False)
    pd.DataFrame(coverage).to_csv(output / "openplay_coverage.csv", index=False)
    if infer_receivers:
        pd.DataFrame(inferred_teams).to_csv(output / "openplay_inferred_team_networks.csv", index=False)
        pd.concat(inferred_players, ignore_index=True).to_csv(output / "openplay_inferred_players.csv", index=False)
    return combined
