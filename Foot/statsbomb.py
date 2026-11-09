"""Recorded player networks and chronological pass models for 2015–16."""

from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from data_sources import read_json
from metrics import concentration, count_pairs, network_metrics, pass_matrix

SEASON = "2015-16"
PASS_COLUMNS = [
    "team_id", "passer_id", "receiver_id", "period", "clock", "x", "y",
    "end_x", "end_y", "previous", "options",
]


@dataclass
class Match:
    match_id: int
    date: str
    teams: dict
    players: dict
    passes: pd.DataFrame
    shots: dict
    patterns: Counter


def parse_match(metadata, events):
    teams = {metadata[side][f"{side}_id"]: metadata[side][f"{side}_name"]
             for side in ("home_team", "away_team")}
    players = {team: {} for team in teams}
    active = {team: set() for team in teams}
    passed, shot_seen, actions = [], set(), defaultdict(list)
    shots = {team: Counter() for team in teams}
    patterns = Counter()

    for e in events:
        team = e.get("team", {}).get("id")
        if team not in teams:
            continue
        kind = e["type"]["name"]
        player = e.get("player", {})
        pid = player.get("id")
        if kind == "Starting XI":
            for line in e["tactics"]["lineup"]:
                p = line["player"]
                players[team][p["id"]] = p["name"]
                active[team].add(p["id"])
            continue
        if kind == "Substitution":
            replacement = e["substitution"]["replacement"]
            active[team].discard(pid)
            active[team].add(replacement["id"])
            players[team][replacement["id"]] = replacement["name"]
            continue
        if kind == "Player Off":
            active[team].discard(pid)
        if kind == "Player On" and pid in players[team]:
            active[team].add(pid)
        if kind in {"Bad Behaviour", "Foul Committed"}:
            card = e.get(kind.lower().replace(" ", "_"), {}).get("card", {}).get("name")
            if card in {"Red Card", "Second Yellow"}:
                active[team].discard(pid)
        if e["period"] not in (1, 2) or pid is None:
            continue

        possession = (e["period"], e["possession"], team)
        history = actions[possession]
        if kind == "Pass":
            detail = e["pass"]
            if "outcome" in detail or "recipient" not in detail:
                continue
            recipient = detail["recipient"]
            rid = recipient["id"]
            if rid == pid:
                continue
            players[team][pid] = player["name"]
            players[team][rid] = recipient["name"]
            x, y = e["location"]
            end_x, end_y = detail["end_location"][:2]
            # Use a common 0–100 scale with right touchline at y=0.
            options = tuple(sorted(active[team] - {pid})) if pid in active[team] and rid in active[team] else ()
            passed.append((team, pid, rid, e["period"], e["minute"] * 60 + e["second"],
                           x / 1.2, 100 - y / 0.8, end_x / 1.2, 100 - end_y / 0.8,
                           history[-1] if history else "Start", options))
        if kind == "Shot" and e["shot"].get("type", {}).get("name") != "Penalty":
            shots[team][pid] += 1
        if e.get("possession_team", {}).get("id") != team or kind not in {"Pass", "Carry", "Dribble", "Shot"}:
            continue
        history.append(kind)
        if kind == "Shot" and possession not in shot_seen:
            if len(history) >= 4:
                patterns[(team, " → ".join(history[-4:]))] += 1
            shot_seen.add(possession)

    return Match(metadata["match_id"], metadata["match_date"], teams, players,
                 pd.DataFrame(passed, columns=PASS_COLUMNS), shots, patterns)


class ReceiverModel:
    def __init__(self):
        self.by_passer = defaultdict(Counter)
        self.by_context = defaultdict(Counter)

    @staticmethod
    def inputs(passes):
        columns = ["team_id", "passer_id", "receiver_id", "x", "y", "previous", "options"]
        for team, passer, receiver, x, y, previous, options in passes[columns].itertuples(index=False, name=None):
            if options:
                yield (team, passer), (min(int(x * 3 / 100), 2), min(int(y * 3 / 100), 2), previous), receiver, options

    def fit(self, passes):
        for passer, context, receiver, _ in self.inputs(passes):
            self.by_passer[passer][receiver] += 1
            self.by_context[passer + context][receiver] += 1

    def score(self, passes):
        totals = defaultdict(lambda: np.zeros(3, dtype=float))
        for passer, context, receiver, options in self.inputs(passes):
            base_counts = self.by_passer.get(passer, {})
            context_counts = self.by_context.get(passer + context, {})
            baseline = (base_counts.get(receiver, 0) + 1) / (
                sum(base_counts.get(p, 0) for p in options) + len(options)
            )
            contextual = (context_counts.get(receiver, 0) + 12 * baseline) / (
                sum(context_counts.get(p, 0) for p in options) + 12
            )
            totals[passer[0]] += (1, -np.log2(baseline), -np.log2(contextual))
        return totals


def summarize_matches(sources, output):
    fixtures = sources.statsbomb_matches()
    paths = sources.statsbomb_paths(fixtures)
    matches = [parse_match(row, read_json(path)) for row, path in zip(fixtures, paths)]

    match_rows, team_rows, player_rows, all_passes, all_patterns = [], [], [], [], Counter()
    season_players, season_passes, season_shots, team_names = {}, defaultdict(list), defaultdict(Counter), {}
    for match in matches:
        all_patterns.update(match.patterns)
        team_names.update(match.teams)
        for team_id, name in match.teams.items():
            roster = match.players[team_id]
            team_passes = match.passes.loc[match.passes["team_id"].eq(team_id)]
            edges = count_pairs(team_passes)
            _, matrix = pass_matrix(edges, roster)
            values, _ = network_metrics(matrix)
            match_rows.append({
                "match_id": match.match_id, "date": match.date, "team_id": team_id, "team": name,
                **values, "non_penalty_shots": sum(match.shots[team_id].values()),
            })
            all_passes.extend((match.match_id, team_id, passer, receiver, count)
                              for (passer, receiver), count in edges.items())
            season_players.setdefault(team_id, {}).update(roster)
            season_passes[team_id].append(team_passes)
            season_shots[team_id].update(match.shots[team_id])

    networks = {}
    for team_id, name in sorted(team_names.items(), key=lambda entry: entry[1]):
        passes = pd.concat(season_passes[team_id], ignore_index=True)
        roster = season_players[team_id]
        order, matrix = pass_matrix(count_pairs(passes), roster)
        values, player = network_metrics(matrix)
        team_rows.append({
            "sport": "Premier League", "season": SEASON, "team": name, "team_id": team_id, **values,
            "coverage_complete": True, "matches": len(season_passes[team_id]),
            "non_penalty_shots": sum(season_shots[team_id].values()),
            "shot_hhi": concentration(list(season_shots[team_id].values())),
        })
        player["player_id"] = order
        player["player"] = [roster[pid] for pid in order]
        player["non_penalty_shots"] = [season_shots[team_id][pid] for pid in order]
        starts = passes[["passer_id", "x", "y"]].rename(columns={"passer_id": "player_id"})
        ends = passes[["receiver_id", "end_x", "end_y"]].set_axis(["player_id", "x", "y"], axis=1)
        positions = pd.concat([starts, ends]).groupby("player_id")[["x", "y"]].mean()
        player = player.join(positions, on="player_id").assign(team=name, team_id=team_id, season=SEASON)
        player_rows.append(player)
        networks[name] = (team_id, order, matrix, player)

    teams = pd.DataFrame(team_rows)
    teams.to_csv(output / "statsbomb_team_season_metrics.csv", index=False)
    pd.DataFrame(match_rows).to_csv(output / "statsbomb_team_match_metrics.csv", index=False)
    pd.DataFrame(all_passes, columns=["match_id", "team_id", "passer_id", "receiver_id", "passes"]).to_csv(
        output / "statsbomb_match_passes.csv.gz", index=False, compression="gzip"
    )
    pd.concat(player_rows, ignore_index=True).to_csv(output / "statsbomb_players.csv", index=False)
    patterns = pd.DataFrame(
        [(team_names[team], pattern, count) for (team, pattern), count in all_patterns.most_common()],
        columns=["team", "sequence", "possessions"],
    )
    patterns.to_csv(output / "statsbomb_possession_patterns.csv", index=False)
    return fixtures, matches, teams, networks, patterns


def predict_receivers(matches, output):
    midpoint = matches[len(matches) // 2].date
    model = ReceiverModel()
    for match in matches:
        if match.date < midpoint:
            model.fit(match.passes)
    totals, all_counts, names = defaultdict(lambda: np.zeros(3)), Counter(), {}
    for match in matches:
        if match.date >= midpoint:
            names.update(match.teams)
            all_counts.update(match.passes["team_id"])
            for team, values in model.score(match.passes).items():
                totals[team] += values
    rows = []
    for team, name in sorted(names.items(), key=lambda item: item[1]):
        n, base, context = totals[team]
        if n:
            rows.append({
                "team": name, "test_from": midpoint, "passes_scored": int(n),
                "eligible_share": n / all_counts[team],
                "passer_only_bits": base / n, "context_bits": context / n,
                "context_gain_bits": (base - context) / n,
            })
    pd.DataFrame(rows).to_csv(output / "statsbomb_recipient_prediction.csv", index=False)
