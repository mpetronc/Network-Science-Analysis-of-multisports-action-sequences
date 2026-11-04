"""Season-aggregated NBA passing networks, not possession/action-sequence analysis.

Run in PyCharm for GSW, or use --team ALL for all teams. nba_api is a
community-maintained client; data source: NBA.com (https://www.nba.com/stats).
Dependencies: python -m pip install -r requirements.txt
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
from importlib.metadata import version
import json
import logging
from pathlib import Path
import random
import re
import tempfile
import time
from typing import Callable

import numpy as np
import pandas as pd
import requests
from nba_api.stats.endpoints import leaguedashplayerstats, playerdashptpass
from nba_api.stats.library.http import NBAStatsHTTP
from nba_api.stats.static import teams as static_teams


SEASON = "2025-26"
TARGET_TEAM = "GSW"  # Any NBA abbreviation, or "ALL".
SOURCE = "NBA.com — https://www.nba.com/stats"
LOG = logging.getLogger("nba_passing")
CACHE_VERSION = 1


class DataUnavailable(RuntimeError):
    """A player/team could not be downloaded or validated."""


class DownloadsStopped(DataUnavailable):
    """Access was denied, or repeated failures require stopping downloads."""


@dataclass(frozen=True)
class Config:
    season: str = SEASON
    season_type: str = "Regular Season"
    cache_dir: Path = field(default_factory=lambda: Path.home() / "nba_passing_cache")
    refresh: bool = False
    max_cache_age_hours: float | None = None
    timeout: float = 60
    retries: int = 4
    min_delay: float = 2.0
    max_delay: float = 3.5
    backoff_factor: float = 4.0

    def __post_init__(self):
        if not re.fullmatch(r"\d{4}-\d{2}", self.season) or int(self.season[-2:]) != (int(self.season[:4]) + 1) % 100:
            raise ValueError("Season must use consecutive years, for example 2025-26.")
        if self.season_type not in ("Regular Season", "Playoffs"):
            raise ValueError("Season type must be Regular Season or Playoffs.")
        if not all(np.isfinite(value) for value in (self.timeout, self.min_delay, self.max_delay, self.backoff_factor)):
            raise ValueError("Request timing settings must be finite.")
        if self.retries < 1 or self.timeout <= 0:
            raise ValueError("Retries must be positive and timeout must exceed zero.")
        if not 0 <= self.min_delay <= self.max_delay or self.backoff_factor < 0:
            raise ValueError("Invalid request delay/backoff settings.")
        if self.max_cache_age_hours is not None and (
            not np.isfinite(self.max_cache_age_hours) or self.max_cache_age_hours < 0
        ):
            raise ValueError("Cache age must be finite and nonnegative.")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class DiskCache:
    """Atomic JSON snapshots keyed by every endpoint/query parameter.

    Old .pkl caches are intentionally not loaded. JSON preserves API payloads
    without executing serialized Python objects.
    """

    def __init__(self, config: Config):
        self.config = config

    def path_for(self, endpoint: str, parameters: dict) -> Path:
        request = {"version": CACHE_VERSION, "endpoint": endpoint, "parameters": parameters}
        digest = hashlib.sha256(
            json.dumps(request, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return Path(self.config.cache_dir).expanduser() / f"{endpoint}_{digest}.json"

    def read(self, endpoint: str, parameters: dict) -> dict | None:
        if self.config.refresh:
            return None
        path = self.path_for(endpoint, parameters)
        if not path.exists():
            return None
        try:
            snapshot = json.loads(path.read_text(encoding="utf-8"))
            if (
                snapshot["version"] != CACHE_VERSION
                or snapshot["endpoint"] != endpoint
                or snapshot["parameters"] != parameters
            ):
                raise ValueError("Cache request metadata does not match.")
            fetched = datetime.fromisoformat(snapshot["fetched_at"])
            if fetched.tzinfo is None:
                raise ValueError("Cache timestamp must include a timezone.")
            age = (datetime.now(timezone.utc) - fetched).total_seconds() / 3600
            limit = self.config.max_cache_age_hours
            if age < 0 or (limit is not None and age > limit):
                return None
            if not isinstance(snapshot["payload"], dict):
                raise ValueError("Invalid cached payload.")
            return snapshot
        except (OSError, ValueError, KeyError, TypeError) as exc:
            LOG.warning("Ignoring unreadable/invalid cache %s: %s", path.name, exc)
            return None

    def write(self, snapshot: dict) -> None:
        path = self.path_for(snapshot["endpoint"], snapshot["parameters"])
        temporary = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
            ) as handle:
                temporary = Path(handle.name)
                json.dump(snapshot, handle, allow_nan=False)
            temporary.replace(path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def extract_table(payload: dict, name: str, required: set[str]) -> pd.DataFrame:
    """Select a result set by name, never by its position in a response."""
    if not isinstance(payload, dict):
        raise ValueError("NBA response must be a JSON object.")
    tables = payload.get("resultSets", payload.get("resultSet", []))
    if isinstance(tables, dict):
        tables = [tables]
    if not isinstance(tables, list):
        raise ValueError("Invalid resultSets in NBA response.")
    table = next((t for t in tables if isinstance(t, dict) and t.get("name") == name), None)
    if table is None:
        raise ValueError(f"NBA response is missing result set {name}.")
    headers, rows = table.get("headers"), table.get("rowSet")
    if not isinstance(headers, list) or not all(isinstance(h, str) for h in headers):
        raise ValueError(f"Invalid headers in {name}.")
    if len(headers) != len(set(headers)) or not required.issubset(headers):
        raise ValueError(f"{name} needs columns {sorted(required)}; received {headers}.")
    if not isinstance(rows, list) or any(
        not isinstance(row, list) or len(row) != len(headers) for row in rows
    ):
        raise ValueError(f"Invalid rows in {name}.")
    return pd.DataFrame(rows, columns=headers)


def numeric_column(frame: pd.DataFrame, column: str, *, integer=False, positive=False):
    values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"{column} contains missing, negative, or nonfinite values.")
    if integer and (values != np.floor(values)).any():
        raise ValueError(f"{column} must contain integers.")
    if positive and (values <= 0).any():
        raise ValueError(f"{column} must be positive.")
    frame[column] = values.astype(np.int64) if integer else values


def validate_roster(payload: dict, team_id: int, team_abbr: str) -> pd.DataFrame:
    required = {"PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION", "GP", "MIN"}
    frame = extract_table(payload, "LeagueDashPlayerStats", required)
    for column in ("PLAYER_ID", "TEAM_ID", "GP", "MIN"):
        numeric_column(frame, column, integer=column != "MIN", positive=column == "PLAYER_ID")
    # Apply the team restriction even when activity fields are absent: absent
    # required fields above are an error, never a reason to return the league.
    frame = frame.loc[
        (frame["TEAM_ID"] == team_id)
        & (frame["TEAM_ABBREVIATION"] == team_abbr)
        & (frame["GP"] > 0)
        & (frame["MIN"] > 0)
    ].copy()
    if (frame["PLAYER_ID"].duplicated().any() or frame["PLAYER_NAME"].isna().any()
            or frame["PLAYER_NAME"].astype(str).str.strip().eq("").any()):
        raise ValueError("Roster contains duplicate IDs or missing/empty names.")
    return frame.sort_values("PLAYER_ID").reset_index(drop=True)


def validate_passing(payload: dict, player_id: int, team_id: int) -> pd.DataFrame:
    required = {"PLAYER_ID", "TEAM_ID", "PASS_TEAMMATE_PLAYER_ID", "PASS"}
    frame = extract_table(payload, "PassesMade", required)
    for column in required:
        numeric_column(frame, column, integer=True, positive=column != "PASS")
    if not ((frame["PLAYER_ID"] == player_id) & (frame["TEAM_ID"] == team_id)).all():
        raise ValueError("Passing response belongs to a different player/team.")
    if ((frame["PASS_TEAMMATE_PLAYER_ID"] == player_id) & (frame["PASS"] > 0)).any():
        raise ValueError("Passing response contains a positive self-pass.")
    return frame


def retry_after_seconds(response: requests.Response | None) -> float | None:
    if response is None:
        return None
    value = response.headers.get("Retry-After")
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            deadline = parsedate_to_datetime(value)
            if deadline.tzinfo is None:
                deadline = deadline.replace(tzinfo=timezone.utc)
            seconds = (deadline - datetime.now(timezone.utc)).total_seconds()
        except (TypeError, ValueError, OverflowError):
            return None
    return max(0.0, seconds) if np.isfinite(seconds) else None


class NBAClient:
    """Serial requests with bounded retries; no proxy rotation or bypass logic."""

    def __init__(self, config: Config):
        self.config = config
        self.cache = DiskCache(config)
        self.stopped_reason: str | None = None
        self.consecutive_failures = 0
        self.session = requests.Session()
        # nba_api's parser otherwise may discard HTTP status/Retry-After details.
        self.session.hooks["response"].append(self._check_status)
        self.previous_session = NBAStatsHTTP.get_session()
        NBAStatsHTTP.set_session(self.session)

    @staticmethod
    def _check_status(response, *args, **kwargs):
        try:
            response.raise_for_status()
        except requests.HTTPError:
            response.close()
            raise
        return response

    def close(self):
        NBAStatsHTTP.set_session(self.previous_session)
        self.session.close()

    def stop(self, reason: str):
        self.stopped_reason = reason
        raise DownloadsStopped(reason)

    def request(self, factory: Callable[[], dict]) -> dict:
        if self.stopped_reason:
            raise DownloadsStopped(self.stopped_reason)
        config = self.config
        for attempt in range(config.retries):
            time.sleep(random.uniform(config.min_delay, config.max_delay))
            try:
                payload = factory()
                self.consecutive_failures = 0
                return payload
            except (requests.RequestException, ValueError, KeyError) as exc:
                response = getattr(exc, "response", None)
                status = response.status_code if response is not None else None
                if status in (401, 403):
                    self.stop(f"NBA access denied (HTTP {status}); stopping downloads.")
                if status is not None and 400 <= status < 500 and status not in (408, 429):
                    self.stop(f"NBA rejected the request (HTTP {status}): {exc}")
                if attempt == config.retries - 1:
                    self.consecutive_failures += 1
                    if status == 429 or self.consecutive_failures >= 3:
                        self.stop(f"Persistent NBA request failures; stopping: {exc}")
                    raise DataUnavailable(f"Request failed after {config.retries} attempts: {exc}") from exc
                delay = min(60.0, config.backoff_factor * 2**attempt)
                retry_after = retry_after_seconds(response)
                if retry_after is not None:
                    # Stop rather than retry before a long server-directed cooldown.
                    if retry_after > 300:
                        self.stop(f"NBA requests a {retry_after:.0f}s cooldown; retry later.")
                    delay = max(delay, retry_after)
                LOG.warning("Request attempt %s failed; waiting %.1fs: %s", attempt + 1, delay, exc)
                time.sleep(delay)
        raise AssertionError("Unreachable retry state.")

    def fetch(self, endpoint_class, parameters: dict, validate: Callable) -> tuple[pd.DataFrame, dict]:
        endpoint = endpoint_class.endpoint
        query = endpoint_class(**parameters, timeout=self.config.timeout, get_request=False)
        # Include resolved API defaults as well as explicitly supplied parameters.
        resolved_parameters = dict(query.parameters)
        snapshot = self.cache.read(endpoint, resolved_parameters)
        if snapshot is not None:
            try:
                return validate(snapshot["payload"]), snapshot
            except (ValueError, KeyError, TypeError) as exc:
                LOG.warning("Cached %s response failed validation: %s", endpoint, exc)
        def download():
            query.get_request()
            return query.get_dict()

        payload = self.request(download)
        try:
            frame = validate(payload)
        except (ValueError, KeyError, TypeError) as exc:
            raise DataUnavailable(f"Invalid {endpoint} response: {exc}") from exc
        snapshot = {
            "version": CACHE_VERSION, "endpoint": endpoint, "parameters": resolved_parameters,
            "library_version": version("nba_api"),
            "fetched_at": utc_now(), "source": SOURCE, "payload": payload,
        }
        try:
            self.cache.write(snapshot)
        except OSError as exc:
            LOG.warning("Could not save cache; using downloaded data: %s", exc)
        return frame, snapshot

    def roster(self, team: dict):
        # A team-filtered historical query includes that team's season participants,
        # including traded players, instead of filtering a current league roster.
        parameters = {
            "team_id_nullable": team["id"], "season": self.config.season,
            "season_type_all_star": self.config.season_type,
            "measure_type_detailed_defense": "Base", "per_mode_detailed": "Totals",
        }
        return self.fetch(
            leaguedashplayerstats.LeagueDashPlayerStats, parameters,
            lambda payload: validate_roster(payload, team["id"], team["abbreviation"]),
        )

    def passing(self, player_id: int, team_id: int):
        parameters = {
            "player_id": player_id, "team_id": team_id, "season": self.config.season,
            "season_type_all_star": self.config.season_type, "per_mode_simple": "Totals",
        }
        return self.fetch(
            playerdashptpass.PlayerDashPtPass, parameters,
            lambda payload: validate_passing(payload, player_id, team_id),
        )


def compute_team_entropies(matrix) -> tuple[float | None, float | None]:
    """Return H(receiver)/ln(n), H(receiver|passer)/ln(n-1).

    Sender weights are empirical shares of observed passes, not an assumed
    stationary distribution. The second metric is not called an entropy rate.
    With no passes the distribution is undefined, so both results are None.
    Missing rows must be handled explicitly before calling this function.
    """
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Passing matrix must be square.")
    if not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError("Passing matrix must contain finite, nonnegative weights.")
    if np.any(np.diag(matrix) != 0):
        raise ValueError("Self-passes must be zero.")
    n = len(matrix)
    total = float(matrix.sum())
    if n <= 1 or total == 0:
        return None, None

    def entropy(probabilities):
        positive = probabilities[probabilities > 0]
        return float(-np.sum(positive * np.log(positive)))

    receiver = entropy(matrix.sum(axis=0) / total) / np.log(n)
    outgoing = matrix.sum(axis=1)
    conditional = sum(
        (outgoing[i] / total) * entropy(matrix[i] / outgoing[i])
        for i in range(n) if outgoing[i] > 0
    )
    conditional = conditional / np.log(n - 1) if n > 2 else 0.0
    return float(np.clip(receiver, 0, 1)), float(np.clip(conditional, 0, 1))


@dataclass
class TeamData:
    team: str
    season: str
    season_type: str
    player_ids: list[int]
    names: dict[int, str]
    matrix: np.ndarray
    missing: dict[int, str]
    snapshots: list[dict]
    roster_player_ids: list[int]

    def metrics(self) -> dict:
        # NaNs mean unknown outgoing profiles. Only this explicitly provisional
        # calculation replaces them with zero; exported raw matrices retain NaNs.
        observed = np.nan_to_num(self.matrix, nan=0.0)
        receiver, conditional = compute_team_entropies(observed)
        complete = not self.missing
        return {
            "team": self.team, "season": self.season, "season_type": self.season_type,
            "source": SOURCE, "created_at": utc_now(),
            "player_count": len(self.player_ids),
            "roster_player_count": len(self.roster_player_ids),
            "additional_receiver_ids": sorted(set(self.player_ids) - set(self.roster_player_ids)),
            "player_profiles_complete": complete,
            "missing_outgoing_profiles": self.missing,
            "recorded_passes_in_available_profiles": float(observed.sum()),
            "receiver_entropy": receiver if complete else None,
            "conditional_receiver_entropy": conditional if complete else None,
            "observed_only_receiver_entropy": receiver,
            "observed_only_conditional_receiver_entropy": conditional,
            "notes": [
                "Season-aggregated tracked passes; not ordered possession/action sequences.",
                "Profile coverage does not guarantee tracking exists for every game/pass.",
                "Incomplete-profile metrics are provisional and omit unknown outgoing passes.",
                "Normalization uses all included players; playing time/lineups affect interpretation.",
            ],
            "requests": [
                {key: snapshot[key] for key in ("endpoint", "parameters", "fetched_at", "library_version")}
                for snapshot in self.snapshots
            ],
        }


def build_team_data(team: dict, roster: pd.DataFrame, profiles: dict[int, pd.DataFrame],
                    failures: dict[int, str], snapshots: list[dict], config: Config) -> TeamData:
    """Retain roster nodes and all known passes, including to missing players."""
    roster_ids = [int(pid) for pid in roster["PLAYER_ID"]]
    names = {int(row.PLAYER_ID): str(row.PLAYER_NAME) for row in roster.itertuples()}
    ids = set(roster_ids)
    for frame in profiles.values():
        for row in frame.itertuples():
            receiver = int(row.PASS_TEAMMATE_PLAYER_ID)
            if receiver not in ids and row.PASS > 0:
                ids.add(receiver)
                names[receiver] = str(getattr(row, "PASS_TO", receiver))
    player_ids = sorted(ids)
    index = {pid: i for i, pid in enumerate(player_ids)}
    matrix = np.full((len(player_ids), len(player_ids)), np.nan)
    np.fill_diagonal(matrix, 0.0)
    missing = dict(failures)
    for pid in player_ids:
        frame = profiles.get(pid)
        if frame is None or frame.empty:
            missing.setdefault(pid, "No nonempty outgoing tracking profile available.")
            continue
        missing.pop(pid, None)
        matrix[index[pid], :] = 0.0
        for row in frame.itertuples():
            receiver = int(row.PASS_TEAMMATE_PLAYER_ID)
            if row.PASS > 0:
                matrix[index[pid], index[receiver]] += int(row.PASS)
    return TeamData(team["abbreviation"], config.season, config.season_type,
                    player_ids, names, matrix, missing, snapshots, roster_ids)


def run_single_team_pipeline(team: dict, client: NBAClient) -> TeamData:
    roster, roster_snapshot = client.roster(team)
    if roster.empty:
        raise DataUnavailable(f"No season participants with positive games/minutes for {team['abbreviation']}.")
    profiles, failures, snapshots = {}, {}, [roster_snapshot]
    LOG.info("%s: %s season participants", team["abbreviation"], len(roster))
    for row in roster.itertuples():
        pid = int(row.PLAYER_ID)
        if client.stopped_reason:
            failures[pid] = client.stopped_reason
            continue
        try:
            frame, snapshot = client.passing(pid, team["id"])
            snapshots.append(snapshot)
            if frame.empty:
                failures[pid] = "Empty tracking response; zero passes cannot be confirmed."
            else:
                profiles[pid] = frame
        except DataUnavailable as exc:
            failures[pid] = str(exc)
            LOG.warning("%s / %s: %s", team["abbreviation"], row.PLAYER_NAME, exc)
    return build_team_data(team, roster, profiles, failures, snapshots, client.config)


def short_labels(data: TeamData) -> dict[int, str]:
    labels = {}
    for pid in data.player_ids:
        parts = data.names.get(pid, str(pid)).split()
        labels[pid] = f"{parts[0][0]}. {parts[-1]}" if len(parts) > 1 else " ".join(parts)
    repeated = {label for label in labels.values() if list(labels.values()).count(label) > 1}
    return {pid: f"{label} ({pid})" if label in repeated else label for pid, label in labels.items()}


def build_graph(data: TeamData):
    import networkx as nx

    graph = nx.DiGraph()
    graph.add_nodes_from(data.player_ids)  # Include isolated/missing-profile players.
    for i, passer in enumerate(data.player_ids):
        for j, receiver in enumerate(data.player_ids):
            weight = data.matrix[i, j]
            if np.isfinite(weight) and weight > 0:
                graph.add_edge(passer, receiver, weight=float(weight))
    return graph


def plot_team_network(data: TeamData, output_file: Path, *, show=False):
    import matplotlib.pyplot as plt
    import networkx as nx

    graph = build_graph(data)
    fig, ax = plt.subplots(figsize=(12, 10))
    # Use undirected connectivity for placement so missing outgoing profiles do
    # not create artificial sink positions. Arrows still show pass direction.
    positions = nx.spring_layout(graph.to_undirected(), seed=42, weight=None)
    involvement = np.array([
        graph.in_degree(pid, weight="weight") + graph.out_degree(pid, weight="weight")
        for pid in graph.nodes
    ], dtype=float)
    sizes = 300 + 2700 * involvement / max(float(involvement.max(initial=0)), 1.0)
    weights = [attributes["weight"] for _, _, attributes in graph.edges(data=True)]
    widths = [0.5 + 4 * weight / max(weights) for weight in weights]
    nx.draw_networkx_nodes(
        graph, positions, ax=ax, node_size=sizes,
        node_color=["lightgray" if pid in data.missing else "gold" for pid in graph.nodes],
        edgecolors="dimgray", alpha=0.9,
    )
    nx.draw_networkx_edges(
        graph, positions, ax=ax, width=widths, edge_color="royalblue", alpha=0.6,
        arrows=True, arrowsize=14, node_size=sizes, connectionstyle="arc3,rad=0.1",
    )
    nx.draw_networkx_labels(graph, positions, labels=short_labels(data), ax=ax, font_size=9)
    profile_word = "profile" if len(data.missing) == 1 else "profiles"
    status = f" — PARTIAL: {len(data.missing)} missing {profile_word}" if data.missing else ""
    ax.set_title(f"{data.team} passing network | {data.season} {data.season_type}{status}")
    ax.axis("off")
    fig.text(0.5, 0.025,
             "Source: NBA.com | Node size: known passing involvement | Edge width: recorded passes\n"
             "Gray nodes: unknown outgoing profile. Aggregate tracking totals; not action sequences.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=160)
    if show:
        plt.show()
    plt.close(fig)
    return graph


def save_team_data(data: TeamData, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = data.metrics()
    pd.DataFrame(data.matrix, index=data.player_ids, columns=data.player_ids).rename_axis(
        "passer_id", axis=0
    ).rename_axis("receiver_id", axis=1).to_csv(output_dir / "passing_matrix.csv", na_rep="NA")
    pd.DataFrame([
        {"player_id": pid, "name": data.names[pid], "outgoing_profile_available": pid not in data.missing,
         "missing_reason": data.missing.get(pid, "")}
        for pid in data.player_ids
    ]).to_csv(output_dir / "players.csv", index=False)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8"
    )
    return metrics


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--team", default=TARGET_TEAM, help="Team abbreviation or ALL (default: GSW).")
    parser.add_argument("--season", default=SEASON)
    parser.add_argument("--season-type", default="Regular Season", choices=["Regular Season", "Playoffs"])
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / "nba_passing_cache")
    parser.add_argument("--refresh", action="store_true", help="Download fresh snapshots instead of using cache.")
    parser.add_argument("--max-cache-age-hours", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("nba_passing_output"))
    parser.add_argument("--no-show", action="store_true", help="Save plots without opening a window.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        config = Config(season=args.season, season_type=args.season_type, cache_dir=args.cache_dir,
                        refresh=args.refresh, max_cache_age_hours=args.max_cache_age_hours)
    except ValueError as exc:
        parser.error(str(exc))
    teams = sorted(static_teams.get_teams(), key=lambda team: team["abbreviation"])
    selected = args.team.strip().upper()
    if selected != "ALL":
        teams = [team for team in teams if team["abbreviation"] == selected]
    if not teams:
        parser.error(f"Unknown NBA team: {args.team}")
    output = args.output_dir.expanduser() / args.season / args.season_type.replace(" ", "_")
    LOG.info("Data source: %s. Analysis uses aggregate tracked passes.", SOURCE)
    summaries, errors = [], {}
    client = NBAClient(config)
    try:
        for team in teams:
            try:
                data = run_single_team_pipeline(team, client)
                directory = output / data.team
                metrics = save_team_data(data, directory)
                summaries.append(metrics)
                plot_team_network(data, directory / "passing_network.png",
                                  show=not args.no_show and len(teams) == 1)
                LOG.info("%s: %.0f recorded passes; %s/%s outgoing profiles available.",
                         data.team, metrics["recorded_passes_in_available_profiles"],
                         len(data.player_ids) - len(data.missing), len(data.player_ids))
                prefix = "Observed-only, provisional" if data.missing else "Available tracking network"
                LOG.info("%s entropies — receiver: %s; receiver given passer: %s", prefix,
                         metrics["observed_only_receiver_entropy"],
                         metrics["observed_only_conditional_receiver_entropy"])
            except DataUnavailable as exc:
                errors[team["abbreviation"]] = str(exc)
                LOG.error("%s: %s", team["abbreviation"], exc)
            if client.stopped_reason:
                LOG.error(client.stopped_reason)
                break
    finally:
        client.close()
    processed = {item["team"] for item in summaries} | errors.keys()
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source": SOURCE, "created_at": utc_now(), "teams": summaries, "errors": errors,
        "not_processed": [team["abbreviation"] for team in teams if team["abbreviation"] not in processed],
        "downloads_stopped": client.stopped_reason,
    }
    (output / "run_summary.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    LOG.info("Results saved in %s", output.resolve())
    return int(bool(errors or client.stopped_reason or any(not item["player_profiles_complete"] for item in summaries)))


if __name__ == "__main__":
    raise SystemExit(main())
