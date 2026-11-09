"""Measures shared by the existing NBA exports and the football pass networks."""

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


def entropy(counts):
    total = np.sum(counts)
    if total <= 0:
        return np.nan
    shares = counts[counts > 0] / total
    return float(-np.dot(shares, np.log(shares)))


def concentration(counts):
    values = np.asarray(counts, dtype=float)
    total = values.sum()
    return float(np.square(values / total).sum()) if total > 0 else np.nan


def pass_matrix(pairs, players):
    order = sorted(players)
    index = {player: i for i, player in enumerate(order)}
    matrix = np.zeros((len(order), len(order)))
    for (passer, receiver), count in pairs.items():
        matrix[index[passer], index[receiver]] = count
    return order, matrix


def network_metrics(matrix):
    """Match the NBA entropy formulas and compute weighted receiving PageRank."""
    matrix = np.asarray(matrix, dtype=float)
    n = len(matrix)
    if matrix.shape != (n, n) or np.isinf(matrix).any() or (matrix < 0).any():
        raise ValueError("Expected a square pass matrix with nonnegative counts or missing values.")
    if np.any(np.nan_to_num(matrix.diagonal()) != 0):
        raise ValueError("Self-passes are excluded.")

    observed = np.nan_to_num(matrix, nan=0.0)
    sent = observed.sum(axis=1)
    received = observed.sum(axis=0)
    total = sent.sum()
    local = np.full(n, np.nan)
    rank = np.full(n, np.nan)
    receiver_entropy = conditional_entropy = rank_hhi = normalized_hhi = np.nan
    receiver_nats = mi = normalized_mi = effective_receivers = np.nan
    if n > 1 and total > 0:
        receiver_nats = entropy(received)
        receiver_entropy = receiver_nats / np.log(n)
        for i in np.flatnonzero(sent):
            local[i] = entropy(observed[i])
        conditional_nats = float(np.nansum(local * sent) / total)
        if n > 2:
            local /= np.log(n - 1)
        conditional_entropy = conditional_nats / np.log(n - 1) if n > 2 else 0.0
        mi = max(0.0, receiver_nats - conditional_nats)
        normalized_mi = mi / np.log(n)
        effective_receivers = float(np.exp(receiver_nats))

        # Damping and uniform dangling rows make the solution unique for sparse graphs.
        transition = np.full((n, n), 1.0 / n)
        np.divide(observed, sent[:, None], out=transition, where=sent[:, None] > 0)
        damping = 0.85
        rank = np.linalg.solve(np.eye(n) - damping * transition.T, np.full(n, (1 - damping) / n))
        rank /= rank.sum()
        rank_hhi = concentration(rank)
        normalized_hhi = max(0.0, (rank_hhi - 1 / n) / (1 - 1 / n))

    team = {
        "players": n,
        "recorded_passes": int(total),
        "receiver_entropy": receiver_entropy,
        "receiver_entropy_nats": receiver_nats,
        "effective_receivers": effective_receivers,
        "passing_mutual_information": mi,
        "normalized_passing_mutual_information": normalized_mi,
        "conditional_receiver_entropy": conditional_entropy,
        "pagerank_hhi": rank_hhi,
        "pagerank_concentration": normalized_hhi,
    }
    player = pd.DataFrame({
        "passes_sent": sent.astype(int),
        "passes_received": received.astype(int),
        "known_outgoing_profile": ~np.isnan(matrix).any(axis=1),
        "passer_entropy": local,
        "pagerank": rank,
    })
    return team, player


def nba_exports(directory, *, season):
    """Read saved NBA exports without making tracking requests or changing the NBA scripts."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"No NBA season output at {directory}. Run the NBA script or set --nba-dir.")

    index_path = directory / "team_metrics.csv"
    index = pd.read_csv(index_path).set_index("team") if index_path.exists() else pd.DataFrame()
    teams, players, networks = [], [], {}
    for path in sorted(directory.glob("*/passing_matrix.csv")):
        club = path.parent.name
        matrix = pd.read_csv(path, index_col=0)
        matrix.index = matrix.index.map(str)
        matrix = matrix.loc[:, matrix.index]
        roster = pd.read_csv(path.with_name("players.csv"), dtype={"player_id": str}).set_index("player_id")
        team, player = network_metrics(matrix.to_numpy(dtype=float))
        roster = roster.reindex(matrix.index)
        known = player["known_outgoing_profile"] & roster["outgoing_profile_available"].fillna(False).to_numpy(bool)
        original = index.loc[club] if club in index.index else {}
        observed = original.get("observed_only_conditional_receiver_entropy", np.nan)
        if pd.notna(observed) and not np.isclose(observed, team["conditional_receiver_entropy"]):
            raise ValueError(f"{club}: recomputed entropy disagrees with the saved NBA export.")
        keys = {"sport": "NBA", "season": season, "team": club}
        status = original.get("status", "")
        complete = bool(known.all()) and status not in ("partial", "failed", "previous_results")
        teams.append(keys | team | {
            "coverage_complete": complete,
            "known_profiles": int(known.sum()),
            "usage_hhi": original.get("usage_hhi_starness", np.nan),
            "nba_status": status,
        })
        player["player_id"] = matrix.index
        player["player"] = roster["name"].to_numpy()
        player["known_outgoing_profile"] = known
        players.append(player.assign(**keys))
        networks[club] = (matrix.index.to_list(), matrix.to_numpy(dtype=float), player, complete)

    if not teams:
        raise ValueError(f"No team passing matrices found under {directory}.")
    return pd.DataFrame(teams), pd.concat(players, ignore_index=True), networks


def count_pairs(frame, first="passer_id", second="receiver_id"):
    return Counter(zip(frame[first], frame[second]))
