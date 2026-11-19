from __future__ import annotations
import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
from importlib.metadata import version
import json
import logging
import os
from pathlib import Path
import random
import re
import tempfile
import time
from typing import Callable, TextIO
import numpy as np
import pandas as pd
import requests
from nba_api.stats.endpoints import leaguedashplayerstats, playerdashptpass
from nba_api.stats.library.http import NBAStatsHTTP
from nba_api.stats.static import teams as static_teams
SEASON = "2025-26"
TARGET_TEAM = "ALL"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "nba_passing_output"
SOURCE = "NBA.com — https://www.nba.com/stats"
LOG = logging.getLogger("nba_passing")
CACHE_VERSION = 1
PASS_COLUMNS = [
    "team", "season", "season_type", "passer_id", "passer_name",
    "receiver_id", "receiver_name", "pass_count",
    "outgoing_profile_available", "source",
]
METRIC_COLUMNS = [
    "team", "season", "season_type", "status", "player_count",
    "roster_player_count", "available_profile_count",
    "missing_profile_count", "missing_player_ids",
    "player_profiles_complete",
    "recorded_passes",
    "usage_hhi_starness", "usage_effective_players",
    "passing_mutual_information", "normalized_passing_mutual_information",
    "receiver_entropy", "conditional_receiver_entropy",
    "observed_only_receiver_entropy",
    "observed_only_conditional_receiver_entropy",
    "created_at", "source", "latest_attempt_error",
]
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
    rotation_only: bool = False
    min_mpg: float = 10.0
    min_games: int = 20
    def __post_init__(self):
        if (
            not re.fullmatch(r"\d{4}-\d{2}", self.season)
            or int(self.season[-2:]) != (int(self.season[:4]) + 1) % 100
        ):
            raise ValueError("Season must use consecutive years, for example 2025-26.")
        if self.season_type not in ("Regular Season", "Playoffs"):
            raise ValueError("Season type must be Regular Season or Playoffs.")
        if not all(
            np.isfinite(value)
            for value in (
                self.timeout, self.min_delay, self.max_delay,
                self.backoff_factor
            )
        ):
            raise ValueError("Request timing settings must be finite.")
        if self.retries < 1 or self.timeout <= 0:
            raise ValueError("Retries must be positive and timeout must exceed zero.")
        if not 0 <= self.min_delay <= self.max_delay or self.backoff_factor < 0:
            raise ValueError("Invalid request delay/backoff settings.")
        if self.max_cache_age_hours is not None and (
            not np.isfinite(self.max_cache_age_hours)
            or self.max_cache_age_hours < 0
        ):
            raise ValueError("Cache age must be finite and nonnegative.")
        if not np.isfinite(self.min_mpg) or self.min_mpg < 0 or self.min_games < 0:
            raise ValueError("Rotation thresholds must be finite and nonnegative.")
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
def atomic_write(path: Path, writer: Callable[[TextIO], object]) -> None:
    """Replace a file only after the complete new content has been flushed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
def atomic_csv(frame: pd.DataFrame, path: Path, *, index=False) -> None:
    atomic_write(
        path,
        lambda handle: frame.to_csv(handle, index=index, na_rep="NA")
    )
def atomic_json(value: dict, path: Path) -> None:
    atomic_write(
        path,
        lambda handle: json.dump(value, handle, indent=2, allow_nan=False)
    )
class DiskCache:
    """Atomic JSON snapshots keyed by every endpoint/query parameter."""
    def __init__(self, config: Config):
        self.config = config
    def path_for(self, endpoint: str, parameters: dict) -> Path:
        request = {
            "version": CACHE_VERSION,
            "endpoint": endpoint,
            "parameters": parameters,
        }
        digest = hashlib.sha256(
            json.dumps(
                request,
                sort_keys=True,
                separators=(",", ":")
            ).encode()
        ).hexdigest()
        return (
            Path(self.config.cache_dir).expanduser()
            / f"{endpoint}_{digest}.json"
        )
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
            age = (
                datetime.now(timezone.utc) - fetched
            ).total_seconds() / 3600
            limit = self.config.max_cache_age_hours
            if age < 0 or (limit is not None and age > limit):
                return None
            if not isinstance(snapshot["payload"], dict):
                raise ValueError("Invalid cached payload.")
            return snapshot
        except (OSError, ValueError, KeyError, TypeError) as exc:
            LOG.warning(
                "Ignoring unreadable/invalid cache %s: %s",
                path.name,
                exc,
            )
            return None
    def write(self, snapshot: dict) -> None:
        path = self.path_for(
            snapshot["endpoint"],
            snapshot["parameters"]
        )
        temporary = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                json.dump(snapshot, handle, allow_nan=False)
            temporary.replace(path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
def extract_team_metrics(
    payload: dict,
    name: str,
    required: set[str]
) -> pd.DataFrame:
    """Select a result set by name, never by its position in a response."""
    if not isinstance(payload, dict):
        raise ValueError("NBA response must be a JSON object.")
    tables = payload.get("resultSets", payload.get("resultSet", []))
    if isinstance(tables, dict):
        tables = [tables]
    if not isinstance(tables, list):
        raise ValueError("Invalid resultSets in NBA response.")
    table = next(
        (
            t for t in tables
            if isinstance(t, dict) and t.get("name") == name
        ),
        None,
    )
    if table is None:
        raise ValueError(f"NBA response is missing result set {name}.")
    headers = table.get("headers")
    rows = table.get("rowSet")
    if not isinstance(headers, list) or not all(
        isinstance(h, str) for h in headers
    ):
        raise ValueError(f"Invalid headers in {name}.")
    if len(headers) != len(set(headers)) or not required.issubset(headers):
        raise ValueError(
            f"{name} needs columns {sorted(required)}; received {headers}."
        )
    if not isinstance(rows, list) or any(
        not isinstance(row, list) or len(row) != len(headers)
        for row in rows
    ):
        raise ValueError(f"Invalid rows in {name}.")
    return pd.DataFrame(rows, columns=headers)
def numeric_column(
    frame: pd.DataFrame,
    column: str,
    *,
    integer=False,
    positive=False
):
    values = pd.to_numeric(
        frame[column],
        errors="raise"
    ).to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(
            f"{column} contains missing, negative, or nonfinite values."
        )
    if integer and (values != np.floor(values)).any():
        raise ValueError(f"{column} must contain integers.")
    if positive and (values <= 0).any():
        raise ValueError(f"{column} must be positive.")
    frame[column] = (
        values.astype(np.int64) if integer else values
    )
def validate_roster(
    payload: dict,
    team_id: int,
    team_abbr: str
) -> pd.DataFrame:
    required = {
        "PLAYER_ID",
        "PLAYER_NAME",
        "TEAM_ID",
        "TEAM_ABBREVIATION",
        "GP",
        "MIN",
        "FGA", "FTA", "TOV",
    }
    frame = extract_team_metrics(
        payload,
        "LeagueDashPlayerStats",
        required,
    )
    for column in (
        "PLAYER_ID", "TEAM_ID", "GP", "MIN", "FGA", "FTA", "TOV"
    ):
        numeric_column(
            frame,
            column,
            integer=column != "MIN",
            positive=column == "PLAYER_ID",
        )
    frame = frame.loc[
        (frame["TEAM_ID"] == team_id)
        & (frame["TEAM_ABBREVIATION"] == team_abbr)
        & (frame["GP"] > 0)
        & (frame["MIN"] > 0)
    ].copy()
    if (
        frame["PLAYER_ID"].duplicated().any()
        or frame["PLAYER_NAME"].isna().any()
        or frame["PLAYER_NAME"].astype(str).str.strip().eq("").any()
    ):
        raise ValueError(
            "Roster contains duplicate IDs or missing/empty names."
        )
    return frame.sort_values("PLAYER_ID").reset_index(drop=True)
def validate_passing(
    payload: dict,
    player_id: int,
    team_id: int
) -> pd.DataFrame:
    required = {
        "PLAYER_ID",
        "TEAM_ID",
        "PASS_TEAMMATE_PLAYER_ID",
        "PASS",
    }
    frame = extract_team_metrics(
        payload,
        "PassesMade",
        required,
    )
    for column in required:
        numeric_column(
            frame,
            column,
            integer=True,
            positive=column != "PASS",
        )
    if not (
        (frame["PLAYER_ID"] == player_id)
        & (frame["TEAM_ID"] == team_id)
    ).all():
        raise ValueError(
            "Passing response belongs to a different player/team."
        )
    # Silently filter out self-passes.
    frame = frame[
        frame["PASS_TEAMMATE_PLAYER_ID"] != player_id
    ].copy()
    return frame
def retry_after_seconds(
    response: requests.Response | None
) -> float | None:
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
            seconds = (
                deadline - datetime.now(timezone.utc)
            ).total_seconds()
        except (TypeError, ValueError, OverflowError):
            return None
    return max(0.0, seconds) if np.isfinite(seconds) else None
class NBAClient:
    """Serial requests with bounded retries."""
    def __init__(self, config: Config):
        self.config = config
        self.cache = DiskCache(config)
        self.stopped_reason: str | None = None
        self.consecutive_failures = 0
        self.session = requests.Session()
        self.session.hooks["response"].append(self._check_status)
        self.previous_session = NBAStatsHTTP.get_session()
        NBAStatsHTTP.set_session(self.session)
    @staticmethod
    def _check_status(response, *_args, **_kwargs):
        try:
            response.raise_for_status()
        except requests.HTTPError:
            raise
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
            time.sleep(
                random.uniform(config.min_delay, config.max_delay)
            )
            try:
                payload = factory()
                self.consecutive_failures = 0
                return payload
            except (requests.RequestException, ValueError, KeyError) as exc:
                response = getattr(exc, "response", None)
                status = (
                    response.status_code
                    if response is not None
                    else None
                )
                if status in (401, 403):
                    self.stop(
                        f"NBA access denied (HTTP {status}); stopping downloads."
                    )
                if (
                    status is not None
                    and 400 <= status < 500
                    and status not in (408, 429)
                ):
                    self.stop(
                        f"NBA rejected the request (HTTP {status}): {exc}"
                    )
                if attempt == config.retries - 1:
                    self.consecutive_failures += 1
                    if (
                        status == 429
                        or self.consecutive_failures >= 3
                    ):
                        self.stop(
                            f"Persistent NBA request failures; stopping: {exc}"
                        )
                    raise DataUnavailable(
                        f"Request failed after {config.retries} attempts: {exc}"
                    ) from exc
                delay = min(
                    60.0,
                    config.backoff_factor * 2**attempt
                )
                retry_after = retry_after_seconds(response)
                if retry_after is not None:
                    if retry_after > 300:
                        self.stop(
                            f"NBA requests a {retry_after:.0f}s cooldown; retry later."
                        )
                    delay = max(delay, retry_after)
                LOG.warning(
                    "Request attempt %s failed; waiting %.1fs: %s",
                    attempt + 1,
                    delay,
                    exc,
                )
                time.sleep(delay)
        raise AssertionError("Unreachable retry state.")
    def fetch(
        self,
        endpoint_class,
        parameters: dict,
        validate: Callable
    ) -> tuple[pd.DataFrame, dict]:
        endpoint = endpoint_class.endpoint
        query = endpoint_class(
            **parameters,
            timeout=self.config.timeout,
            get_request=False,
        )
        resolved_parameters = dict(query.parameters)
        snapshot = self.cache.read(
            endpoint,
            resolved_parameters
        )
        if snapshot is not None:
            try:
                cached_frame = validate(snapshot["payload"])
                if not cached_frame.empty:
                    return cached_frame, snapshot
                LOG.info(
                    "Ignoring a cached empty %s response; requesting it again.",
                    endpoint,
                )
            except (ValueError, KeyError, TypeError) as exc:
                LOG.warning(
                    "Cached %s response failed validation: %s",
                    endpoint,
                    exc,
                )
        def download():
            query.get_request()
            return query.get_dict()
        payload = self.request(download)
        try:
            frame = validate(payload)
        except (ValueError, KeyError, TypeError) as exc:
            raise DataUnavailable(
                f"Invalid {endpoint} response: {exc}"
            ) from exc
        snapshot = {
            "version": CACHE_VERSION,
            "endpoint": endpoint,
            "parameters": resolved_parameters,
            "library_version": version("nba_api"),
            "fetched_at": utc_now(),
            "source": SOURCE,
            "payload": payload,
        }
        try:
            if not frame.empty:
                self.cache.write(snapshot)
        except OSError as exc:
            LOG.warning(
                "Could not save cache; using downloaded data: %s",
                exc,
            )
        return frame, snapshot
    def roster(self, team: dict):
        parameters = {
            "team_id_nullable": team["id"],
            "season": self.config.season,
            "season_type_all_star": self.config.season_type,
            "measure_type_detailed_defense": "Base",
            "per_mode_detailed": "Totals",
        }
        return self.fetch(
            leaguedashplayerstats.LeagueDashPlayerStats,
            parameters,
            lambda payload: validate_roster(
                payload,
                team["id"],
                team["abbreviation"],
            ),
        )
    def passing(self, player_id: int, team_id: int):
        parameters = {
            "player_id": player_id,
            "team_id": team_id,
            "season": self.config.season,
            "season_type_all_star": self.config.season_type,
            "per_mode_simple": "Totals",
        }
        return self.fetch(
            playerdashptpass.PlayerDashPtPass,
            parameters,
            lambda payload: validate_passing(
                payload,
                player_id,
                team_id,
            ),
        )
# ---------------------------------------------------------------------
# ENTROPY
# ---------------------------------------------------------------------
def compute_team_entropies(
    matrix
) -> tuple[float | None, float | None]:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Network matrix must be square.")
    if not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError(
            "Network matrix must contain finite, nonnegative weights."
        )
    if np.any(np.diag(matrix) != 0):
        raise ValueError("Self-edges must be zero.")
    n = len(matrix)
    total = float(matrix.sum())
    if n <= 1 or total == 0:
        return None, None
    def entropy(probabilities):
        positive = probabilities[probabilities > 0]
        return -float(np.sum(positive * np.log(positive)))
    # Receiver entropy is calculated from player-to-player passes only.
    receiver = entropy(
        matrix.sum(axis=0) / total
    ) / np.log(n)
    outgoing = matrix.sum(axis=1)
    conditional = sum(
        (outgoing[i] / total) * entropy(
            matrix[i] / outgoing[i]
        )
        for i in range(n)
        if outgoing[i] > 0
    )
    # There are n-1 possible receivers for each node.
    conditional = (
        conditional / np.log(n - 1)
        if n > 2
        else 0.0
    )
    return (
        float(np.clip(receiver, 0, 1)),
        float(np.clip(conditional, 0, 1)),
    )
def passing_mutual_information(matrix) -> tuple[float | None, float | None]:
    """I(P;R) in nats and I(P;R)/log(n), using observed player-only passes.

    The normalization uses log(number of player nodes) for cross-team comparison.
    Missing outgoing profiles must be interpreted as provisional results.
    """
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Passing matrix must be square")
    if not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError("Passing weights must be finite and nonnegative")
    n = matrix.shape[0]
    total = float(matrix.sum())
    if n < 2 or total <= 0:
        return None, None
    joint = matrix / total
    passer = joint.sum(axis=1)
    receiver = joint.sum(axis=0)
    expected = np.outer(passer, receiver)
    mask = joint > 0
    mi = float(np.sum(joint[mask] * np.log(joint[mask] / expected[mask])))
    mi = max(0.0, mi)  # protect against floating-point roundoff
    return mi, float(np.clip(mi / np.log(n), 0, 1))


def usage_concentration(actions: dict[int, float]) -> tuple[float | None, float | None]:
    """HHI of possession-ending action shares; effective players = 1 / HHI."""
    values = np.asarray(list(actions.values()), dtype=float)
    if values.size == 0 or not np.isfinite(values).all() or (values < 0).any():
        return None, None
    total = float(values.sum())
    if total <= 0:
        return None, None
    shares = values / total
    hhi = float(np.sum(shares ** 2))
    return hhi, float(1.0 / hhi)
# ---------------------------------------------------------------------
# TEAM DATA
# ---------------------------------------------------------------------
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
    field_goal_attempts: dict[int, int]
    usage_actions: dict[int, float] = field(default_factory=dict)
    settings: dict = field(default_factory=dict)
    def metrics(self) -> dict:
        observed = np.nan_to_num(
            self.matrix,
            nan=0.0
        )
        receiver, conditional = compute_team_entropies(observed)
        complete = not self.missing
        mi, normalized_mi = passing_mutual_information(observed)
        return {
            "team": self.team,
            "season": self.season,
            "season_type": self.season_type,
            "source": SOURCE,
            "created_at": utc_now(),
            "player_count": len(self.player_ids),
            "roster_player_count": len(self.roster_player_ids),
            "additional_receiver_ids": sorted(
                set(self.player_ids) - set(self.roster_player_ids)
            ),
            "player_profiles_complete": complete,
            "missing_outgoing_profiles": self.missing,
            "usage_hhi_starness": usage_concentration(self.usage_actions)[0],
            "usage_effective_players": usage_concentration(self.usage_actions)[1],
            "passing_mutual_information": mi if complete else None,
            "normalized_passing_mutual_information": normalized_mi if complete else None,
            "field_goal_attempts": int(
                sum(self.field_goal_attempts.values())
            ),
            "recorded_passes": float(
                observed.sum()
            ),
            "receiver_entropy": receiver if complete else None,
            "conditional_receiver_entropy": conditional if complete else None,
            "observed_only_receiver_entropy": receiver,
            "observed_only_conditional_receiver_entropy": conditional,
            "settings": self.settings,
            "notes": [
                "The passing network contains player-to-player edges only.",
                "Passing entropy excludes shooting and free-throw attempts.",
                "Season-aggregated tracked passes; not ordered possession/action sequences.",
                "Profile coverage does not guarantee tracking exists for every game/pass.",
                "Incomplete-profile metrics are provisional and omit unknown outgoing passes.",
                "Entropy normalization uses all included players.",
                "Passing mutual information is computed from unnormalized entropies, in nats; normalized MI divides by log(player count).",
                "Playing time and lineups affect interpretation.",
                "Usage HHI is computed from season possession-ending actions (FGA + 0.44 FTA + TOV), not raw USG percentages.",
            ],
            "requests": [
                {
                    key: snapshot[key]
                    for key in (
                        "endpoint",
                        "parameters",
                        "fetched_at",
                        "library_version",
                    )
                }
                for snapshot in self.snapshots
            ],
        }
def build_team_data(
    team: dict,
    roster: pd.DataFrame,
    profiles: dict[int, pd.DataFrame],
    failures: dict[int, str],
    snapshots: list[dict],
    config: Config,
    settings: dict | None = None,
) -> TeamData:
    roster_ids = [
        int(pid)
        for pid in roster["PLAYER_ID"]
    ]
    names = {
        int(player_id): str(player_name)
        for player_id, player_name in zip(
            roster["PLAYER_ID"],
            roster["PLAYER_NAME"],
        )
    }
    # FGA is retrieved from the season roster statistics.
    field_goal_attempts = {
        int(row["PLAYER_ID"]): int(row["FGA"])
        for row in roster.to_dict(orient="records")
    }
    # Season possession-ending actions approximate minutes-weighted usage.
    # Unlike raw USG%, these are additive across players.
    usage_actions = {
        int(row["PLAYER_ID"]): float(row["FGA"] + 0.44 * row["FTA"] + row["TOV"])
        for row in roster.to_dict(orient="records")
    }
    ids = set(roster_ids)
    # Only roster-qualified players may enter this induced network.
    player_ids = sorted(ids)
    index = {
        pid: i
        for i, pid in enumerate(player_ids)
    }
    matrix = np.full((len(player_ids), len(player_ids)), np.nan)
    np.fill_diagonal(matrix, 0.0)
    missing = dict(failures)
    for pid in player_ids:
        i = index[pid]
        frame = profiles.get(pid)
        if frame is None or frame.empty:
            missing.setdefault(
                pid,
                "No nonempty outgoing tracking profile available.",
            )
            continue
        missing.pop(pid, None)
        # Reset known outgoing passing edges.
        matrix[i, :len(player_ids)] = 0.0
        for row in frame.to_dict(orient="records"):
            receiver = int(row["PASS_TEAMMATE_PLAYER_ID"])
            if row["PASS"] > 0 and receiver in index:
                matrix[i, index[receiver]] += int(row["PASS"])
    # Keep qualified players in metric denominators, including unknown profiles.
    # Plotting alone omits nodes with neither incoming nor outgoing known passes.
    settings = dict(settings or {})
    settings['minimum_player_mpg'] = 10.0

    return TeamData(
        team["abbreviation"],
        config.season,
        config.season_type,
        player_ids,
        names,
        matrix,
        missing,
        snapshots,
        roster_ids,
        field_goal_attempts,
        usage_actions,
        settings,
    )
def rotation_filter(
    roster: pd.DataFrame,
    config: Config
) -> pd.DataFrame:
    keep = (
        (roster["GP"] >= config.min_games)
        & (roster["MIN"] / roster["GP"] >= config.min_mpg)
    )
    return roster.loc[keep].reset_index(drop=True)
def run_single_team_pipeline(
    team: dict,
    client: NBAClient
) -> TeamData:
    config = client.config
    abbr = team["abbreviation"]
    roster, roster_snapshot = client.roster(team)
    if roster.empty:
        raise DataUnavailable(
            f"No season participants with positive games/minutes for {abbr}."
        )
    roster_before_filter = len(roster)
    roster = roster.loc[(roster["MIN"] / roster["GP"]) >= 10.0].copy()
    if roster.empty:
        raise DataUnavailable(f"No players for {abbr} averaging at least 10 minutes")
    if config.rotation_only:
        roster = rotation_filter(roster, config)
        if roster.empty:
            raise DataUnavailable(
                f"No players for {abbr} with >= {config.min_games} games "
                f"and >= {config.min_mpg:g} mpg."
            )
        LOG.info(
            "%s: rotation filter keeps %s of %s players",
            abbr,
            len(roster),
            roster_before_filter,
        )
    profiles = {}
    failures = {}
    snapshots = [roster_snapshot]
    LOG.info(
        "%s: %s players to download",
        abbr,
        len(roster),
    )
    def download(pid: int, label: str) -> None:
        if client.stopped_reason:
            failures[pid] = client.stopped_reason
            return
        try:
            frame, snapshot = client.passing(
                pid,
                team["id"],
            )
            snapshots.append(snapshot)
            if frame.empty:
                failures[pid] = (
                    "Empty tracking response; zero passes cannot be confirmed."
                )
            else:
                profiles[pid] = frame
        except DataUnavailable as exc:
            failures[pid] = str(exc)
            LOG.warning("%s / %s: %s", abbr, label, exc)
    for row in roster.to_dict(orient="records"):
        download(
            int(row["PLAYER_ID"]),
            str(row["PLAYER_NAME"]),
        )
    extras_fetched: list[int] = []
    settings = {
        "minimum_player_mpg": 10.0,
        "rotation_only": config.rotation_only,
        "min_mpg": config.min_mpg if config.rotation_only else None,
        "min_games": config.min_games if config.rotation_only else None,
        "roster_before_filter": roster_before_filter,
        "unlisted_receivers_requested": extras_fetched,
    }
    return build_team_data(
        team,
        roster,
        profiles,
        failures,
        snapshots,
        config,
        settings,
    )
# ---------------------------------------------------------------------
# GRAPH CONSTRUCTION
# ---------------------------------------------------------------------
def short_labels(data: TeamData) -> dict[int, str]:
    labels = {}
    for pid in data.player_ids:
        parts = data.names.get(pid, str(pid)).split()
        labels[pid] = (
            f"{parts[0][0]}. {parts[-1]}"
            if len(parts) > 1
            else " ".join(parts)
        )
    repeated = {
        label
        for label in labels.values()
        if list(labels.values()).count(label) > 1
    }
    return {
        pid: f"{label} ({pid})" if label in repeated else label
        for pid, label in labels.items()
    }
def build_graph(data: TeamData):
    import networkx as nx
    graph = nx.DiGraph(
        team=data.team,
        season=data.season,
        season_type=data.season_type,
        source=SOURCE,
        player_profiles_complete=not data.missing,
        missing_profile_count=len(data.missing),
        edge_weight_description=(
            "Tracked passes between players only."
        ),
        coverage_note=(
            "Unknown outgoing passing profiles remain incomplete."
        ),
    )
    labels = short_labels(data)
    # Add all player nodes.
    for i, pid in enumerate(data.player_ids):
        incoming = float(
            np.nansum(data.matrix[:, i])
        )
        outgoing = float(
            np.nansum(data.matrix[i, :])
        )
        graph.add_node(
            pid,
            player_id=pid,
            name=data.names[pid],
            label=labels[pid],
            outgoing_profile_available=pid not in data.missing,
            missing_reason=data.missing.get(pid, ""),
            field_goal_attempts=data.field_goal_attempts.get(pid, 0),
            known_incoming_activity=incoming,
            known_outgoing_activity=outgoing,
            known_total_activity=incoming + outgoing,
        )
    # Every positive player-to-player matrix entry becomes an edge.
    node_ids = data.player_ids
    for i, passer in enumerate(node_ids):
        for j, receiver in enumerate(node_ids):
            weight = data.matrix[i, j]
            if np.isfinite(weight) and weight > 0:
                graph.add_edge(
                    passer,
                    receiver,
                    weight=float(weight),
                )
    return graph
def save_graph_data(
    data: TeamData,
    output_dir: Path
) -> None:
    import networkx as nx
    graph = build_graph(data)
    xml = "\n".join(
        nx.generate_graphml(
            graph,
            encoding="utf-8",
        )
    )
    atomic_write(
        output_dir / "passing_network.graphml",
        lambda handle: handle.write(xml),
    )
def plot_team_network(
    data: TeamData,
    output_file: Path,
    *,
    show=False
):
    import matplotlib.pyplot as plt
    import networkx as nx
    full_graph = build_graph(data)
    graph = full_graph.copy()
    # A directed isolate has neither incoming nor outgoing observed passes.
    graph.remove_nodes_from(list(nx.isolates(graph)))
    fig, ax = plt.subplots(figsize=(12, 10))
    # Layout only: combine both passing directions into one spring weight.
    # DiGraph.to_undirected() would keep just one direction's edge attributes.
    layout_graph = nx.Graph()
    layout_graph.add_nodes_from(graph.nodes)
    for passer, receiver, attributes in graph.edges(data=True):
        previous = layout_graph.get_edge_data(passer, receiver, {}).get('weight', 0.0)
        layout_graph.add_edge(passer, receiver, weight=previous + attributes['weight'])
    # Normalize weights so the force scale is comparable across team-seasons.
    largest = max((a['weight'] for _, _, a in layout_graph.edges(data=True)), default=1.0)
    for _, _, attributes in layout_graph.edges(data=True):
        attributes['weight'] /= largest
    layout_options = dict(seed=42, weight='weight', iterations=300, threshold=1e-5)
    # Older NetworkX versions implement FR without a method argument.
    from inspect import signature
    if 'method' in signature(nx.spring_layout).parameters:
        layout_options['method'] = 'force'
    positions = nx.spring_layout(layout_graph, **layout_options)
    # Node size includes incoming and outgoing activity.
    involvement = np.array(
        [
            graph.in_degree(pid, weight="weight")
            + graph.out_degree(pid, weight="weight")
            for pid in graph.nodes
        ],
        dtype=float,
    )
    max_involvement = max(
        float(involvement.max(initial=0)),
        1.0,
    )
    sizes = (
        300
        + 2700 * involvement / max_involvement
    )
    node_sizes = sizes.tolist()
    weights = [
        attributes["weight"]
        for _, _, attributes in graph.edges(data=True)
    ]
    max_weight = max(weights, default=1.0)
    widths = [
        0.5 + 4 * weight / max_weight
        for weight in weights
    ]
    node_colors = [
        "#D3D3D3" if node in data.missing else "#FFD700"
        for node in graph.nodes
    ]
    nx.draw_networkx_nodes(
        graph,
        positions,
        ax=ax,
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors="#696969",
        alpha=0.9,
    )
    nx.draw_networkx_edges(
        graph,
        positions,
        ax=ax,
        width=widths,
        edge_color="#4169E1",
        alpha=0.6,
        arrows=True,
        arrowsize=14,
        node_size=node_sizes,
        connectionstyle="arc3,rad=0.1",
    )
    labels = {pid: label for pid, label in short_labels(data).items() if pid in graph}
    nx.draw_networkx_labels(
        graph,
        positions,
        labels=labels,
        ax=ax,
        font_size=9,
    )
    profile_word = (
        "profile"
        if len(data.missing) == 1
        else "profiles"
    )
    status = (
        f" — PARTIAL: {len(data.missing)} missing {profile_word}"
        if data.missing
        else ""
    )
    ax.set_title(
        f"{data.team} Passing Network | "
        f"{data.season} {data.season_type}{status}"
    )
    if not graph:
        ax.text(0.5, 0.5, "No connected players in available passing data",
                ha="center", va="center", transform=ax.transAxes)
    ax.axis("off")
    fig.text(
        0.5,
        0.025,
        "Source: NBA.com | Node size: total incoming and outgoing activity | "
        + ("Edge width: tracked passes | No minutes cutoff\n" if data.settings.get("minimum_player_minutes")==0
         else "Edge width: tracked passes | Players: ≥10 minutes per game\n")
        + "Gray nodes: unknown outgoing passing profile",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(
        rect=(0, 0.07, 1, 1)
    )
    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    fig.savefig(
        output_file,
        dpi=160,
    )
    if show:
        plt.show()
    plt.close(fig)
    return full_graph
# ---------------------------------------------------------------------
# LEAGUE ENTROPY QUADRANT
# ---------------------------------------------------------------------
def plot_league_quadrant(
    output_dir: Path,
    season: str,
    season_type: str,
    *,
    show=False
):
    """Plots all processed teams on a 2D entropy quadrant graph."""
    import matplotlib.pyplot as plt
    metrics_file = output_dir / "team_metrics.csv"
    if not metrics_file.exists():
        LOG.info(
            "No team_metrics.csv found to plot league quadrant."
        )
        return
    df = pd.read_csv(metrics_file)
    df = df.dropna(
        subset=[
            "observed_only_receiver_entropy",
            "observed_only_conditional_receiver_entropy",
        ]
    )
    if df.empty:
        LOG.info("No valid entropy data available for quadrant plot.")
        return
    x = df[
        "observed_only_receiver_entropy"
    ].to_numpy(dtype=float)
    y = df[
        "observed_only_conditional_receiver_entropy"
    ].to_numpy(dtype=float)
    teams = df["team"].tolist()
    fig, ax = plt.subplots(figsize=(11, 10))
    ax.scatter(
        x,
        y,
        color="#1f77b4",
        s=100,
        zorder=3,
        edgecolors="#000000",
        alpha=0.85,
    )
    for i, txt in enumerate(teams):
        ax.annotate(
            txt,
            (x[i], y[i]),
            fontsize=10,
            fontweight="bold",
            xytext=(6, 6),
            textcoords="offset points",
        )
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    x_margin = max(
        (x_max - x_min) * 0.1,
        0.02,
    )
    y_margin = max(
        (y_max - y_min) * 0.1,
        0.02,
    )
    ax.set_xlim(
        x_min - x_margin,
        x_max + x_margin,
    )
    ax.set_ylim(
        y_min - y_margin,
        y_max + y_margin,
    )
    x_mid = np.median(x)
    y_mid = np.median(y)
    ax.axvline(
        x_mid,
        color="gray",
        linestyle="--",
        alpha=0.7,
        zorder=1,
    )
    ax.axhline(
        y_mid,
        color="gray",
        linestyle="--",
        alpha=0.7,
        zorder=1,
    )
    ax.set_xlabel(
        "Receiver Entropy (Players Only)",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_ylabel(
        "Conditional Entropy (Players Only)",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_title(
        f"League-Wide Passing and Shooting Entropy Quadrants | "
        f"{season} {season_type}",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.02,
        "Source: NBA.com | Dashed lines represent league medians.\n"
        "Entropy includes player-to-player passes only.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(
        rect=(0, 0.06, 1, 1)
    )
    plot_path = output_dir / "league_entropy_quadrant.png"
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    fig.savefig(
        plot_path,
        dpi=160,
    )
    LOG.info(
        "Saved league quadrant plot to %s",
        plot_path.resolve(),
    )
    if show:
        plt.show()
    plt.close(fig)
def plot_starness_quadrant(output_dir: Path, season: str, season_type: str, *, show=False, conditional=False):
    """Receiver or conditional receiver entropy (x) versus usage HHI (y)."""
    import matplotlib.pyplot as plt
    path = output_dir / "team_metrics.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    entropy_column = ("observed_only_conditional_receiver_entropy" if conditional
                      else "observed_only_receiver_entropy")
    required = [entropy_column, "usage_hhi_starness"]
    if any(col not in df.columns for col in required):
        LOG.warning("Starness metrics unavailable in team_metrics.csv.")
        return
    df = df.dropna(subset=required)
    if df.empty:
        return
    x = df[required[0]].to_numpy(dtype=float)
    y = df[required[1]].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(11, 10))
    ax.scatter(x, y, s=100, edgecolors="black", alpha=0.85, zorder=3)
    for name, xi, yi in zip(df["team"], x, y):
        ax.annotate(name, (xi, yi), xytext=(6, 6), textcoords="offset points", fontsize=10)
    ax.axvline(float(np.median(x)), color="gray", linestyle="--", alpha=0.7)
    ax.axhline(float(np.median(y)), color="gray", linestyle="--", alpha=0.7)
    ax.margins(0.15)
    ax.set_xlabel(("Conditional receiver entropy" if conditional else "Receiver entropy") + " (players only)")
    ax.set_ylabel("Usage concentration (HHI; higher = more star-centric)")
    ax.set_title(f"{'Conditional Passing Entropy' if conditional else 'Passing Entropy'} vs Usage Starness | {season} {season_type}")
    fig.text(0.5, 0.02,
             "Usage shares: FGA + 0.44 × FTA + TOV | HHI = sum(share²) | Dashed lines: team medians",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = "league_conditional_starness_quadrant.png" if conditional else "league_starness_quadrant.png"
    fig.savefig(output_dir / filename, dpi=160)
    if show:
        plt.show()
    plt.close(fig)
# ---------------------------------------------------------------------
# CSV EXPORT
# ---------------------------------------------------------------------
def passing_rows(data: TeamData) -> pd.DataFrame:
    rows = []
    for i, passer in enumerate(data.player_ids):
        # Player-to-player edges.
        for j, receiver in enumerate(data.player_ids):
            if i == j:
                continue
            rows.append({
                "team": data.team,
                "season": data.season,
                "season_type": data.season_type,
                "passer_id": passer,
                "passer_name": data.names[passer],
                "receiver_id": receiver,
                "receiver_name": data.names[receiver],
                "pass_count": data.matrix[i, j],
                "outgoing_profile_available": passer not in data.missing,
                "source": SOURCE,
            })
    return pd.DataFrame(
        rows,
        columns=PASS_COLUMNS,
    )
def team_metric_row(metrics: dict) -> dict:
    row = {
        key: metrics.get(key)
        for key in METRIC_COLUMNS
    }
    missing = metrics["missing_outgoing_profiles"]
    row.update({
        "status": (
            "complete"
            if metrics["player_profiles_complete"]
            else "partial"
        ),
        "available_profile_count": (
            metrics["player_count"] - len(missing)
        ),
        "missing_profile_count": len(missing),
        "missing_player_ids": ";".join(
            str(pid)
            for pid in sorted(missing, key=int)
        ),
        "latest_attempt_error": "",
    })
    return row
def save_team_data(
    data: TeamData,
    output_dir: Path
) -> dict:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    metrics = data.metrics()
    # Player-to-player passing matrix only.
    matrix_labels = data.player_ids
    matrix_frame = pd.DataFrame(
        data.matrix,
        index=matrix_labels,
        columns=matrix_labels,
    ).rename_axis(
        "passer_id",
        axis=0,
    ).rename_axis(
        "receiver_id",
        axis=1,
    )
    atomic_csv(
        matrix_frame,
        output_dir / "passing_matrix.csv",
        index=True,
    )
    atomic_csv(
        pd.DataFrame([
            {
                "player_id": pid,
                "name": data.names[pid],
                "field_goal_attempts": data.field_goal_attempts.get(pid, 0),
                "usage_actions": data.usage_actions.get(pid, 0.0),
                "outgoing_profile_available": pid not in data.missing,
                "missing_reason": data.missing.get(pid, ""),
            }
            for pid in data.player_ids
        ]),
        output_dir / "players.csv",
    )
    atomic_csv(
        passing_rows(data),
        output_dir / "passes.csv",
    )
    atomic_csv(
        pd.DataFrame(
            [team_metric_row(metrics)],
            columns=METRIC_COLUMNS,
        ),
        output_dir / "metrics.csv",
    )
    save_graph_data(
        data,
        output_dir,
    )
    atomic_json(
        metrics,
        output_dir / "metrics.json",
    )
    return metrics
# ---------------------------------------------------------------------
# PROGRESS / LEAGUE EXPORT
# ---------------------------------------------------------------------
def save_run_progress(
    output: Path,
    teams: list[dict],
    summaries: list[dict],
    errors: dict[str, str],
    plot_errors: dict[str, str],
    stopped_reason: str | None,
    started_at: str,
    config: Config,
) -> None:
    rows = {}
    pass_frames = []
    for path in sorted(output.glob("*/metrics.json")):
        metrics = json.loads(
            path.read_text(encoding="utf-8")
        )
        if (
            metrics["season"] != config.season
            or metrics["season_type"] != config.season_type
        ):
            continue
        rows[metrics["team"]] = team_metric_row(metrics)
        passes_path = path.with_name("passes.csv")
        if passes_path.exists():
            pass_frames.append(
                pd.read_csv(passes_path)
            )
    for team, error in errors.items():
        if team in rows:
            rows[team].update(
                status="previous_results",
                latest_attempt_error=error,
            )
        else:
            rows[team] = {
                "team": team,
                "season": config.season,
                "season_type": config.season_type,
                "status": "failed",
                "created_at": utc_now(),
                "source": SOURCE,
                "latest_attempt_error": error,
            }
    atomic_csv(
        pd.DataFrame(
            [rows[team] for team in sorted(rows)],
            columns=METRIC_COLUMNS,
        ),
        output / "team_metrics.csv",
    )
    combined_passes = (
        pd.concat(pass_frames, ignore_index=True)
        if pass_frames
        else pd.DataFrame(columns=PASS_COLUMNS)
    )
    atomic_csv(
        combined_passes,
        output / "all_team_passes.csv",
    )
    processed = (
        {item["team"] for item in summaries}
        | errors.keys()
    )
    atomic_json(
        {
            "source": SOURCE,
            "created_at": started_at,
            "updated_at": utc_now(),
            "requested_teams": [
                team["abbreviation"]
                for team in teams
            ],
            "teams": summaries,
            "errors": errors,
            "plot_errors": plot_errors,
            "not_processed": [
                team["abbreviation"]
                for team in teams
                if team["abbreviation"] not in processed
            ],
            "downloads_stopped": stopped_reason,
            "csv_files": [
                "team_metrics.csv",
                "all_team_passes.csv",
            ],
        },
        output / "run_summary.json",
    )
# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__
    )
    parser.add_argument(
        "--team",
        default=TARGET_TEAM,
        help="Team abbreviation or ALL (default: %(default)s).",
    )
    parser.add_argument(
        "--season",
        default=SEASON,
    )
    parser.add_argument(
        "--season-type",
        default="Regular Season",
        choices=["Regular Season", "Playoffs"],
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path.home() / "nba_passing_cache",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Download fresh snapshots instead of using cache.",
    )
    parser.add_argument(
        "--max-cache-age-hours",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save plots without opening a window.",
    )
    parser.add_argument(
        "--rotation-only",
        action="store_true",
        help=(
            "Only download players meeting --min-games and --min-mpg; "
            "ignore passes to others."
        ),
    )
    parser.add_argument(
        "--min-mpg",
        type=float,
        default=10.0,
        help="Rotation minutes per game (default: %(default)s).",
    )
    parser.add_argument(
        "--min-games",
        type=int,
        default=20,
        help="Rotation games played (default: %(default)s).",
    )
    parser.add_argument(
        "--min-delay",
        type=float,
        default=2.0,
        help="Minimum seconds between requests.",
    )
    parser.add_argument(
        "--max-delay",
        type=float,
        default=3.5,
        help="Maximum seconds between requests.",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    try:
        config = Config(
            season=args.season,
            season_type=args.season_type,
            cache_dir=args.cache_dir,
            refresh=args.refresh,
            max_cache_age_hours=args.max_cache_age_hours,
            min_delay=args.min_delay,
            max_delay=args.max_delay,
            rotation_only=args.rotation_only,
            min_mpg=args.min_mpg,
            min_games=args.min_games,
        )
    except ValueError as exc:
        parser.error(str(exc))
    teams = sorted(
        static_teams.get_teams(),
        key=lambda team: team["abbreviation"],
    )
    selected = args.team.strip().upper()
    if selected != "ALL":
        teams = [
            team
            for team in teams
            if team["abbreviation"] == selected
        ]
    if not teams:
        parser.error(f"Unknown NBA team: {args.team}")
    output = (
        args.output_dir.expanduser()
        / args.season
        / args.season_type.replace(" ", "_")
    )
    if config.rotation_only:
        output = output / (
            f"rotation_{config.min_mpg:g}mpg_{config.min_games}g"
        )
    LOG.info(
        "Data source: %s. Passing network uses passes only; usage starness uses shooting and turnovers.",
        SOURCE,
    )
    summaries = []
    errors = {}
    plot_errors = {}
    started_at = utc_now()
    client = NBAClient(config)
    try:
        save_run_progress(
            output,
            teams,
            summaries,
            errors,
            plot_errors,
            client.stopped_reason,
            started_at,
            config,
        )
        for team in teams:
            data = None
            try:
                data = run_single_team_pipeline(
                    team,
                    client,
                )
                directory = output / data.team
                metrics = save_team_data(
                    data,
                    directory,
                )
                summaries.append(metrics)
                LOG.info(
                    "%s: %.0f recorded passes; "
                    "%s/%s outgoing profiles available.",
                    data.team,
                    metrics["recorded_passes"],
                    len(data.player_ids) - len(data.missing),
                    len(data.player_ids),
                )
                prefix = (
                    "Observed-only, provisional"
                    if data.missing
                    else "Available tracking network"
                )
                LOG.info(
                    "%s entropies — receiver: %s; "
                    "receiver given passer: %s",
                    prefix,
                    metrics["observed_only_receiver_entropy"],
                    metrics["observed_only_conditional_receiver_entropy"],
                )
            except DataUnavailable as exc:
                errors[team["abbreviation"]] = str(exc)
                LOG.error(
                    "%s: %s",
                    team["abbreviation"],
                    exc,
                )
            save_run_progress(
                output,
                teams,
                summaries,
                errors,
                plot_errors,
                client.stopped_reason,
                started_at,
                config,
            )
            LOG.info(
                "CSV checkpoint saved after %s: %s",
                team["abbreviation"],
                output.resolve(),
            )
            if data is not None:
                try:
                    plot_team_network(
                        data,
                        output / data.team / "passing_network.png",
                        show=(
                            not args.no_show
                            and len(teams) == 1
                        ),
                    )
                except (OSError, ValueError, RuntimeError) as exc:
                    plot_errors[data.team] = str(exc)
                    LOG.warning(
                        "%s plot failed; CSV results are saved: %s",
                        data.team,
                        exc,
                    )
            if client.stopped_reason:
                LOG.error(client.stopped_reason)
                break
    finally:
        client.close()
        save_run_progress(
            output,
            teams,
            summaries,
            errors,
            plot_errors,
            client.stopped_reason,
            started_at,
            config,
        )
        try:
            plot_league_quadrant(
                output,
                config.season,
                config.season_type,
                show=not args.no_show,
            )
            plot_starness_quadrant(
                output,
                config.season,
                config.season_type,
                show=not args.no_show,
            )
            plot_starness_quadrant(
                output,
                config.season,
                config.season_type,
                show=not args.no_show,
                conditional=True,
            )
            plot_league_mutual_information(output, config.season, config.season_type, show=not args.no_show)
        except (OSError, ValueError, RuntimeError) as exc:
            LOG.warning(
                "League quadrant plot generation failed: %s",
                exc,
            )
    LOG.info(
        "Results saved in %s",
        output.resolve(),
    )
    return int(
        bool(
            errors
            or plot_errors
            or client.stopped_reason
            or any(
                not item["player_profiles_complete"]
                for item in summaries
            )
        )
    )


def enrich_saved_mutual_information(season_dir: Path, metrics: pd.DataFrame) -> pd.DataFrame:
    """Calculate MI from previously saved matrices; no NBA requests required."""
    result = metrics.copy()
    for col in ("passing_mutual_information", "normalized_passing_mutual_information"):
        if col not in result:
            result[col] = np.nan
    for idx, row in result.iterrows():
        if pd.notna(row["passing_mutual_information"]) and pd.notna(row["normalized_passing_mutual_information"]):
            continue
        team = str(row.get("team", ""))
        matrix_path = season_dir / team / "passing_matrix.csv"
        if not matrix_path.exists():
            continue
        try:
            matrix = pd.read_csv(matrix_path, index_col=0).to_numpy(dtype=float)
            mi, norm = passing_mutual_information(np.nan_to_num(matrix, nan=0.0))
            # Incomplete pass profiles are provisional, but still show observed MI.
            result.at[idx, "passing_mutual_information"] = mi
            result.at[idx, "normalized_passing_mutual_information"] = norm
        except (OSError, ValueError, TypeError) as exc:
            LOG.warning("Cannot calculate saved MI for %s: %s", matrix_path, exc)
    return result


def plot_league_mutual_information(output_dir: Path, season: str, season_type: str, *, show=False):
    """Plot MI versus usage concentration for one season."""
    import matplotlib.pyplot as plt
    path = output_dir / "team_metrics.csv"
    if not path.exists():
        return
    df = enrich_saved_mutual_information(output_dir, pd.read_csv(path))
    xcol, ycol = "normalized_passing_mutual_information", "usage_hhi_starness"
    if ycol not in df:
        return
    df = df.dropna(subset=[xcol, ycol])
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.scatter(df[xcol], df[ycol], edgecolors="black", alpha=.8)
    for row in df.itertuples():
        ax.annotate(str(row.team), (getattr(row, xcol), getattr(row, ycol)), xytext=(5,5), textcoords="offset points")
    ax.axvline(df[xcol].median(), linestyle="--", color="gray")
    ax.axhline(df[ycol].median(), linestyle="--", color="gray")
    ax.set_xlabel("Normalized passer–receiver mutual information (MI / log N)")
    ax.set_ylabel("Usage concentration (HHI)")
    ax.set_title(f"Passing Mutual Information vs Starness | {season} {season_type}")
    ax.margins(.15)
    fig.tight_layout()
    fig.savefig(output_dir / "league_mutual_information_starness_quadrant.png", dpi=170)
    if show:
        plt.show()
    plt.close(fig)


def run_decade(argv=None) -> int:
    """Run every selected 2020s NBA season, preserving each season's outputs."""
    import sys

    decade_parser = argparse.ArgumentParser(add_help=False)
    decade_parser.add_argument("--all-2020s", action="store_true")
    decade_parser.add_argument("--plot-only", action="store_true",
                               help="Regenerate all combined 2020s plots from saved season CSVs; no NBA downloads.")
    decade_parser.add_argument("--start-year", type=int, default=2020)
    decade_parser.add_argument("--end-year", type=int, default=2025)
    batch_args, remaining = decade_parser.parse_known_args(argv)

    # Running this script without flags processes the 2020s, not just the
    # single-season default (2025-26). --season explicitly opts into one season.
    explicit_season = "--season" in remaining or any(
        arg.startswith("--season=") for arg in remaining
    )
    if not batch_args.all_2020s and not batch_args.plot_only:
        if explicit_season:
            return main(remaining)
        batch_args.all_2020s = True

    start, end = batch_args.start_year, batch_args.end_year
    if not (2020 <= start <= end <= 2029):
        decade_parser.error("Year range must lie between 2020 and 2029, inclusive")
    # Avoid making requests for NBA seasons that have not started.
    today = datetime.now(timezone.utc).date()
    latest_started = min(2029, today.year if today.month >= 10 else today.year - 1)
    if end > latest_started:
        decade_parser.error(
            f"Season {end}-{str(end + 1)[-2:]} has not started; "
            f"latest available season starts in {latest_started}"
        )
    if "--season" in remaining or any(a.startswith("--season=") for a in remaining):
        decade_parser.error("Do not specify --season with --all-2020s or --plot-only")

    # Read the output location and filters using the same CLI defaults as main.
    location_parser = argparse.ArgumentParser(add_help=False)
    location_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    location_parser.add_argument("--season-type", default="Regular Season")
    location_parser.add_argument("--rotation-only", action="store_true")
    location_parser.add_argument("--min-mpg", type=float, default=10.0)
    location_parser.add_argument("--min-games", type=int, default=20)
    loc, _ = location_parser.parse_known_args(remaining)
    base = loc.output_dir.expanduser()
    mode = loc.season_type.replace(" ", "_")
    rotation = (f"rotation_{loc.min_mpg:g}mpg_{loc.min_games}g"
                if loc.rotation_only else None)

    statuses = []
    if batch_args.plot_only:
        LOG.info("Plot-only mode: reading existing season metrics without downloading NBA data")
    for year in ([] if batch_args.plot_only else range(start, end + 1)):
        season = f"{year}-{(year + 1) % 100:02d}"
        print(f"\n{'=' * 65}\nProcessing {season}\n{'=' * 65}", flush=True)
        try:
            # Continue even if one season has missing tracking profiles.
            code = main([*remaining, "--season", season, "--no-show"])
        except (Exception, SystemExit) as exc:
            # Errors in one season should not discard the other seasons.
            code = 1
            LOG.exception("Season %s failed: %s", season, exc)
        statuses.append({"season": season, "exit_code": code})

    frames = []
    missing_seasons = []
    seasons_to_collect = (
        [f"{year}-{(year + 1) % 100:02d}" for year in range(start, end + 1)]
        if batch_args.plot_only else [item["season"] for item in statuses]
    )
    for season in seasons_to_collect:
        path = base / season / mode
        if rotation:
            path = path / rotation
        csv_path = path / "team_metrics.csv"
        if csv_path.exists():
            frame = pd.read_csv(csv_path)
            frame = enrich_saved_mutual_information(path, frame)
            if not frame.empty:
                frame["season"] = season
                frames.append(frame)
                print(f"  {season}: loaded {len(frame)} team records from {csv_path}", flush=True)
            else:
                missing_seasons.append(season)
                print(f"  {season}: EMPTY team_metrics.csv", flush=True)
        else:
            missing_seasons.append(season)
            print(f"  {season}: MISSING {csv_path}", flush=True)

    decade_dir = base / "2020s_summary" / mode
    if rotation:
        decade_dir = decade_dir / rotation
    decade_dir.mkdir(parents=True, exist_ok=True)
    if not batch_args.plot_only:
        atomic_csv(pd.DataFrame(statuses), decade_dir / "season_run_status.csv")

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        counts = combined.groupby("season")["team"].nunique().sort_index()
        print("\nTeams available by season:", flush=True)
        for season, count in counts.items():
            print(f"  {season}: {count} teams", flush=True)
        if missing_seasons:
            LOG.warning("Missing season data: %s. --plot-only does not download data. "
                        "Run without --plot-only to fetch missing seasons.",
                        ", ".join(missing_seasons))
        mi_counts = combined.dropna(subset=["normalized_passing_mutual_information"]).groupby("season")["team"].nunique()
        print("Team-season points with passing MI:", dict(mi_counts), flush=True)
        if len(counts) == 1:
            LOG.warning("Only ONE season (%s) has data; combined plots will only "
                        "show that season. Check the per-season run logs and "
                        "season_run_status.csv for NBA API failures.", counts.index[0])
        combined["season_start_year"] = combined["season"].str[:4].astype(int)
        combined = combined.sort_values(["season_start_year", "team"])
        atomic_csv(combined, decade_dir / "all_seasons_team_metrics.csv")
        plot_decade_trends(combined, decade_dir)
        plot_decade_team_scatter(combined, decade_dir)
        run_regression_analysis(combined, decade_dir)
        LOG.info("Combined decade results and plots: %s", decade_dir.resolve())
        print(f"\nSaved combined 2020s plots to: {decade_dir.resolve()}")
        for name in ("all_seasons_entropy_quadrant.png",
                     "all_seasons_starness_quadrant.png",
                     "all_seasons_conditional_starness_quadrant.png",
                     "all_seasons_mutual_information_starness_quadrant.png",
                     "regression_hhi_receiver_entropy.png",
                     "regression_hhi_mutual_information.png",
                     "regression_results.csv"):
            path = decade_dir / name
            if path.exists():
                print(f"  {path.name}")
    else:
        LOG.warning("No season metrics were available for decade aggregation; run --all-2020s first")

    return int(bool(missing_seasons) or any(item["exit_code"] != 0 for item in statuses))


def plot_decade_trends(frame: pd.DataFrame, output_dir: Path) -> None:
    """Plot year-by-year league medians for the three existing measures."""
    import matplotlib.pyplot as plt

    metrics = [
        ("observed_only_receiver_entropy", "Receiver entropy"),
        ("observed_only_conditional_receiver_entropy", "Conditional receiver entropy"),
        ("usage_hhi_starness", "Usage HHI starness"),
        ("normalized_passing_mutual_information", "Normalized passing mutual information"),
    ]
    for column, title in metrics:
        if column not in frame.columns:
            continue
        valid = frame.copy()
        valid[column] = pd.to_numeric(valid[column], errors="coerce")
        valid = valid.dropna(subset=[column])
        if valid.empty:
            continue
        median = valid.groupby("season_start_year")[column].median().sort_index()
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(median.index.to_numpy(), median.to_numpy(), marker="o", linewidth=2)
        ax.set_xticks(median.index.to_numpy())
        ax.set_xticklabels([f"{year}-{(year+1)%100:02d}" for year in median.index], rotation=30)
        ax.set_title(f"League median {title} by season")
        ax.set_xlabel("NBA season")
        ax.set_ylabel(title)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / f"trend_{column}.png", dpi=160)
        plt.close(fig)



def plot_decade_team_scatter(frame: pd.DataFrame, output_dir: Path) -> None:
    """One point per team-season; a fixed color for each franchise across seasons."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    comparisons = [
        ("observed_only_receiver_entropy", "observed_only_conditional_receiver_entropy",
         "Receiver entropy", "Conditional receiver entropy", "all_seasons_entropy_quadrant.png"),
        ("observed_only_receiver_entropy", "usage_hhi_starness",
         "Receiver entropy", "Usage HHI (higher = more concentrated)", "all_seasons_starness_quadrant.png"),
        ("observed_only_conditional_receiver_entropy", "usage_hhi_starness",
         "Conditional receiver entropy", "Usage HHI (higher = more concentrated)",
         "all_seasons_conditional_starness_quadrant.png"),
        ("normalized_passing_mutual_information", "usage_hhi_starness",
         "Normalized passer–receiver mutual information (MI / log N)",
         "Usage HHI (higher = more concentrated)",
         "all_seasons_mutual_information_starness_quadrant.png"),
    ]
    # Fixed franchise colors, not reassigned between charts or seasons.
    teams = sorted(frame["team"].dropna().astype(str).unique())
    palette = plt.get_cmap("gist_ncar")
    colors = {team: palette(i / max(len(teams), 1)) for i, team in enumerate(teams)}

    for x_col, y_col, x_label, y_label, filename in comparisons:
        if x_col not in frame.columns or y_col not in frame.columns:
            LOG.warning("Skipping %s: missing metric columns", filename)
            continue
        df = frame[["team", "season", x_col, y_col]].copy()
        df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
        df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        if df.empty:
            LOG.warning("Skipping %s: no valid team-season points", filename)
            continue

        fig, ax = plt.subplots(figsize=(16, 11))
        for row in df.itertuples(index=False, name=None):
            team, season, x, y = row
            ax.scatter(x, y, color=colors[str(team)], s=80,
                       edgecolors="black", linewidths=0.45, alpha=0.85, zorder=3)
            # Label with BOTH team and season so repeated franchises are distinct.
            ax.annotate(f"{team} {season}", (x, y), xytext=(4, 4),
                        textcoords="offset points", fontsize=6.5, alpha=0.85)

        ax.axvline(float(df[x_col].median()), color="gray", linestyle="--", alpha=0.65)
        ax.axhline(float(df[y_col].median()), color="gray", linestyle="--", alpha=0.65)
        ax.set_xlabel(x_label + " (player-to-player passes only)" if "entropy" in x_col else x_label)
        ax.set_ylabel(y_label)
        ax.set_title("NBA 2020s: each point is one team in one season")
        ax.grid(alpha=0.15)
        ax.margins(0.12)
        legend = [Line2D([0], [0], marker="o", linestyle="", label=t,
                         markerfacecolor=colors[t], markeredgecolor="black", markersize=7)
                  for t in teams if t in set(df["team"].astype(str))]
        fig.legend(handles=legend, loc="center right", bbox_to_anchor=(0.995, 0.5),
                   title="Team colors", fontsize=8, ncol=1)
        fig.tight_layout(rect=(0, 0, 0.87, 1))
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / filename, dpi=170)
        plt.close(fig)



def run_regression_analysis(frame: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """OLS: predict passing metrics from usage HHI, one observation per team-season.

    Saves two scatter/regression charts and regression_results.csv, including
    pooled-all-years and separate-season estimates. Requires scipy for the
    standard two-sided t-test p-values (it is imported only when needed).
    The pooled p-values assume independent observations; recurring teams across
    seasons violate that assumption, so interpret those p-values cautiously.
    """
    import matplotlib.pyplot as plt
    from scipy.stats import linregress

    output_dir.mkdir(parents=True, exist_ok=True)
    x_col = "usage_hhi_starness"
    tests = (
        ("observed_only_receiver_entropy", "Normalized receiver entropy",
         "regression_hhi_receiver_entropy.png"),
        ("normalized_passing_mutual_information", "Normalized passer–receiver mutual information",
         "regression_hhi_mutual_information.png"),
    )
    results = []

    for y_col, y_label, image_name in tests:
        if x_col not in frame.columns or y_col not in frame.columns:
            LOG.warning("Skipping regression %s: required column unavailable", y_col)
            continue

        df = frame.copy()
        df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
        df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[x_col, y_col])
        if "team" in df.columns and "season" in df.columns:
            df = df.drop_duplicates(subset=["team", "season"], keep="last")

        def fit(subset: pd.DataFrame, season: str):
            n = len(subset)
            x = subset[x_col].to_numpy(dtype=float)
            y = subset[y_col].to_numpy(dtype=float)
            result = {"outcome": y_col, "season": season, "n": n,
                      "slope": np.nan, "intercept": np.nan, "pearson_r": np.nan,
                      "r_squared": np.nan, "p_value": np.nan, "slope_std_error": np.nan}
            # Regression is undefined if every x value is identical, or if
            # fewer than 3 points are available for a significance test.
            if n < 3 or np.ptp(x) <= 1e-14:
                return result, None
            if np.ptp(y) <= 1e-14:
                return result, None
            stats = linregress(x, y)
            result.update(slope=float(stats.slope), intercept=float(stats.intercept),
                          pearson_r=float(stats.rvalue), r_squared=float(stats.rvalue**2),
                          p_value=float(stats.pvalue), slope_std_error=float(stats.stderr))
            return result, stats

        pooled, fit_all = fit(df, "ALL")
        results.append(pooled)
        if "season" in df.columns:
            for season, subset in df.groupby("season", sort=True):
                row, _ = fit(subset, str(season))
                results.append(row)

        if df.empty:
            LOG.warning("Skipping %s: no valid paired HHI and %s observations", image_name, y_col)
            continue

        fig, ax = plt.subplots(figsize=(12, 8))
        if "season" in df.columns:
            seasons = sorted(df["season"].dropna().astype(str).unique())
            cmap = plt.get_cmap("tab10")
            for i, season in enumerate(seasons):
                part = df[df["season"].astype(str) == season]
                ax.scatter(part[x_col], part[y_col], s=55, alpha=0.75,
                           edgecolors="black", linewidths=0.3,
                           color=cmap(i % 10), label=season)
            ax.legend(title="Season", loc="best", fontsize=9)
        else:
            ax.scatter(df[x_col], df[y_col], s=55, alpha=0.75)

        if fit_all is not None:
            x_line = np.linspace(df[x_col].min(), df[x_col].max(), 200)
            ax.plot(x_line, fit_all.intercept + fit_all.slope * x_line,
                    color="black", lw=2.5, label="Pooled OLS regression")
            ax.text(0.02, 0.98,
                    f"N = {pooled['n']} team-seasons\n"
                    f"Pearson r = {pooled['pearson_r']:.3f}\n"
                    f"R² = {pooled['r_squared']:.3f}\n"
                    f"Slope = {pooled['slope']:.3f}\n"
                    f"p = {pooled['p_value']:.3g}",
                    transform=ax.transAxes, ha="left", va="top", fontsize=10,
                    bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85})
        else:
            ax.text(0.02, 0.98, "Not enough variation/data for regression",
                    transform=ax.transAxes, ha="left", va="top")
        ax.set_xlabel("Usage HHI (higher = more concentrated)")
        ax.set_ylabel(y_label)
        ax.set_title(f"NBA team-seasons: Usage HHI vs {y_label}")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(output_dir / image_name, dpi=180)
        plt.close(fig)

    columns = ["outcome", "season", "n", "slope", "intercept", "pearson_r",
               "r_squared", "p_value", "slope_std_error"]
    out = pd.DataFrame(results, columns=columns)
    atomic_csv(out, output_dir / "regression_results.csv")
    print("\nRegression results (pooled across available seasons):", flush=True)
    for _, row in out[out["season"] == "ALL"].iterrows():
        if pd.notna(row["p_value"]):
            print(f"  {row['outcome']}: n={row['n']}, r={row['pearson_r']:.3f}, "
                  f"R²={row['r_squared']:.3f}, slope={row['slope']:.4f}, "
                  f"p={row['p_value']:.4g}", flush=True)
        else:
            print(f"  {row['outcome']}: insufficient paired data/variation "
                  f"(n={row['n']})", flush=True)
    print("  Note: pooled p-values do not correct for repeated franchises across seasons.", flush=True)
    return out


# ---------------------------------------------------------------------
# SEVEN-GAME PLAYOFF CASE STUDIES (isolated from the season pipeline)
# Run: python teamgraphs.py --series-analysis
# Replot saved game data: python teamgraphs.py --series-analysis --plot-only
# Choose two different matchups: --series-spec 2022-23:BOS:MIA
#                             --series-spec 2023-24:DEN:MIN
# ---------------------------------------------------------------------
DEFAULT_SERIES_SPECS = ('2022-23:BOS:MIA', '2023-24:DEN:MIN')


def validate_series_schedule(payload, team_id, opponent_abbr):
    """Require an actual seven-game playoff matchup, never seven arbitrary games."""
    f = extract_team_metrics(payload, 'LeagueGameFinderResults',
        {'GAME_ID', 'GAME_DATE', 'TEAM_ID', 'MATCHUP', 'WL'})
    f = f.loc[pd.to_numeric(f.TEAM_ID, errors='coerce').eq(team_id)].copy()
    f = f.loc[f.MATCHUP.astype(str).str.split().str[-1].eq(opponent_abbr)]
    f['GAME_ID'] = f.GAME_ID.astype(str).str.zfill(10)
    f['GAME_DATE'] = pd.to_datetime(f.GAME_DATE, errors='raise')
    f = f.sort_values('GAME_DATE').reset_index(drop=True)
    if len(f) != 7 or f.GAME_ID.nunique() != 7 or f.GAME_DATE.nunique() != 7:
        raise ValueError('Expected exactly seven unique games against this opponent')
    if not f.GAME_ID.str.startswith('004').all():
        raise ValueError('Schedule contains non-playoff games')
    if not f.WL.isin(['W', 'L']).all() or int(f.WL.eq('W').sum()) not in (3, 4):
        raise ValueError('Not a completed 4–3 playoff series')
    return f


def validate_single_game_roster(payload, team):
    f = validate_roster(payload, team['id'], team['abbreviation'])
    if f.empty or not f.GP.eq(1).all():
        raise ValueError('Game-date roster unavailable or date filter not respected (GP must equal 1)')
    return f


def resolve_series_zero_profiles(client, team, opponent, game, ids, matrix, missing, folder):
    """Resolve empty profiles ONLY when independent tracking explicitly reports zero.

    Missing rows or unavailable tracking remain unknown. Never infer zero from
    an empty PlayerDashPtPass response alone.
    """
    if not missing:
        return []
    from nba_api.stats.endpoints import leaguedashptstats
    day = game.GAME_DATE.strftime('%m/%d/%Y')
    def validate_totals(payload):
        f = extract_team_metrics(payload, 'LeagueDashPtStats',
            {'PLAYER_ID','TEAM_ID','GP','PASSES_MADE'})
        for column in ['PLAYER_ID','TEAM_ID','GP','PASSES_MADE']:
            numeric_column(f, column, integer=True, positive=column=='PLAYER_ID')
        f = f.loc[f.TEAM_ID.eq(team['id'])].copy()
        if f.empty or not f.GP.eq(1).all() or f.PLAYER_ID.duplicated().any():
            raise ValueError('Single-game passing totals unavailable or invalid')
        return f
    try:
        totals, _ = client.fetch(leaguedashptstats.LeagueDashPtStats,
            dict(season=client.config.season, season_type_all_star='Playoffs',
                 team_id_nullable=team['id'], opponent_team_id=opponent['id'],
                 date_from_nullable=day, date_to_nullable=day,
                 player_or_team='Player', pt_measure_type='Passing', per_mode_simple='Totals'),
            validate_totals)
    except DownloadsStopped:
        raise
    except DataUnavailable as exc:
        LOG.warning('Cannot verify empty passing profiles for %s %s: %s', team['abbreviation'], day, exc)
        return []
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    atomic_csv(totals, folder/'passing_totals_verification.csv')
    zeros = set(totals.loc[totals.PASSES_MADE.eq(0),'PLAYER_ID'].astype(int))
    verified = sorted(set(missing) & zeros)
    for pid in verified:
        matrix[ids.index(pid), :] = 0.0
        del missing[pid]
    return verified


def download_series_team_game(client, team, opponent, game, folder):
    """One calendar date + opponent only: totals on this date ARE game counts.

    No season averages, cumulative series totals, or per-game season averages
    enter the matrix. Missing outgoing profiles invalidate entropy for the game.
    """
    day = game.GAME_DATE.strftime('%m/%d/%Y')
    filters = dict(date_from_nullable=day, date_to_nullable=day,
                   opponent_team_id=opponent['id'])
    roster, _ = client.fetch(leaguedashplayerstats.LeagueDashPlayerStats,
        dict(season=client.config.season, season_type_all_star='Playoffs',
             team_id_nullable=team['id'], measure_type_detailed_defense='Base',
             per_mode_detailed='Totals', **filters),
        lambda p: validate_single_game_roster(p, team))
    all_game_ids = set(roster.PLAYER_ID.astype(int))
    folder.mkdir(parents=True, exist_ok=True)
    roster.to_csv(folder/'game_players_source.csv', index=False)
    roster = roster.loc[roster.MIN >= 10.0].copy()
    if roster.empty:
        raise DataUnavailable('No players with at least 10 minutes in this game')
    ids = roster.PLAYER_ID.astype(int).tolist()
    names = dict(zip(ids, roster.PLAYER_NAME.astype(str)))
    matrix = np.zeros((len(ids), len(ids)), dtype=float)
    index = {pid: i for i, pid in enumerate(ids)}
    missing = {}
    for pid in ids:
        try:
            def validate_game_passes(payload):
                f = validate_passing(payload, pid, team['id'])
                if f.empty:
                    # Cannot distinguish an untracked profile from genuine zero passes.
                    raise ValueError('Empty game passing profile; zero passes cannot be assumed')
                if 'G' in f and not pd.to_numeric(f.G, errors='coerce').eq(1).all():
                    raise ValueError('Passing profile spans more than the requested game')
                if f.PASS_TEAMMATE_PLAYER_ID.duplicated().any():
                    raise ValueError('Duplicate receiving player rows')
                if not f.PASS_TEAMMATE_PLAYER_ID.isin(all_game_ids).all():
                    raise ValueError('Passing receiver absent from game roster')
                return f
            passes, _ = client.fetch(playerdashptpass.PlayerDashPtPass,
                dict(team_id=team['id'], player_id=pid, season=client.config.season,
                     season_type_all_star='Playoffs', per_mode_simple='Totals', **filters),
                validate_game_passes)
            for row in passes.itertuples():
                if int(row.PASS_TEAMMATE_PLAYER_ID) in index:
                    matrix[index[pid], index[int(row.PASS_TEAMMATE_PLAYER_ID)]] = float(row.PASS)
        except DownloadsStopped:
            raise
        except DataUnavailable as exc:
            matrix[index[pid], :] = np.nan
            matrix[index[pid], index[pid]] = 0
            missing[pid] = str(exc)
    verified_zeros = resolve_series_zero_profiles(client, team, opponent, game, ids, matrix, missing, folder)
    actions = dict(zip(ids, (roster.FGA + .44*roster.FTA + roster.TOV).astype(float)))
    data = TeamData(team=team['abbreviation'], season=client.config.season,
        season_type='Playoffs', player_ids=ids, names=names, matrix=matrix,
        missing=missing, snapshots=[], roster_player_ids=ids,
        field_goal_attempts=dict(zip(ids, roster.FGA.astype(int))), usage_actions=actions)
    folder.mkdir(parents=True, exist_ok=True)
    roster.to_csv(folder/'game_players.csv', index=False)
    pd.DataFrame(matrix, index=ids, columns=ids).to_csv(folder/'passing_matrix.csv')
    plot_team_network(data, folder/'passing_network_fruchterman.png')
    observed = np.nan_to_num(matrix, nan=0)
    entropy, conditional = compute_team_entropies(observed) if not missing else (None, None)
    raw_mi, normalized_mi = passing_mutual_information(observed) if not missing else (None, None)
    total = observed.sum()
    probs = observed.sum(axis=0)/total if total > 0 else np.zeros(len(ids))
    raw_entropy = -float(np.sum(probs[probs > 0]*np.log(probs[probs > 0]))) if not missing and total > 0 else None
    shares = np.array(list(actions.values()), dtype=float)
    usage_hhi = float(np.sum((shares/shares.sum())**2)) if shares.sum() > 0 else None
    # Actual individual USG_PCT is distinct from action-share HHI.
    usage_error, max_usage, max_usage_player = '', None, None
    try:
        def validate_advanced(payload):
            f = extract_team_metrics(payload, 'LeagueDashPlayerStats',
                {'PLAYER_ID', 'TEAM_ID', 'GP', 'USG_PCT'})
            f = f.loc[pd.to_numeric(f.TEAM_ID, errors='coerce').eq(team['id'])].copy()
            for col in ['GP', 'USG_PCT']:
                f[col] = pd.to_numeric(f[col], errors='raise')
            if f.empty or not f.GP.eq(1).all() or not np.isfinite(f.USG_PCT).all():
                raise ValueError('Single-game advanced usage unavailable')
            if not f.USG_PCT.between(0, 1).all():
                raise ValueError('Unexpected USG_PCT scale')
            return f
        advanced, _ = client.fetch(leaguedashplayerstats.LeagueDashPlayerStats,
            dict(season=client.config.season, season_type_all_star='Playoffs',
                 team_id_nullable=team['id'], measure_type_detailed_defense='Advanced',
                 per_mode_detailed='Totals', **filters), validate_advanced)
        advanced = advanced.loc[advanced.PLAYER_ID.isin(ids)]
        advanced.to_csv(folder/'game_advanced_usage.csv', index=False)
        top = advanced.sort_values(['USG_PCT', 'PLAYER_ID'], ascending=[False, True]).iloc[0]
        max_usage, max_usage_player = float(top.USG_PCT), names.get(int(top.PLAYER_ID), '')
    except DownloadsStopped:
        raise
    except (DataUnavailable, IndexError) as exc:
        usage_error = str(exc)
    return dict(minimum_player_minutes=10.0, receiver_entropy=entropy, receiver_entropy_nats=raw_entropy,
        receiver_entropy_fixed15=raw_entropy/np.log(15) if raw_entropy is not None else None,
        conditional_receiver_entropy=conditional, normalized_passing_mutual_information=normalized_mi,
        usage_hhi=usage_hhi, max_player_usage_rate=max_usage, max_usage_player=max_usage_player,
        player_count=len(ids), recorded_passes=float(total), complete=not missing,
        verified_zero_pass_player_ids=';'.join(map(str, verified_zeros)),
        missing_player_ids=';'.join(map(str, missing)),
        error='; '.join(missing.values()), usage_error=usage_error)


def plot_series_game_trends(frame, output_dir, series_key):
    """NaNs stay in the seven-game index so lines never bridge missing games."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 1, figsize=(11, 14), sharex=True)
    for team, part in frame.groupby('team', sort=True):
        part = part.set_index('game_number').reindex(range(1, 8))
        for ax, col, label in zip(axes,
                ['receiver_entropy_fixed15', 'receiver_entropy', 'usage_hhi', 'max_player_usage_rate'],
                ['Receiver entropy / ln(15)', 'Receiver entropy / ln(players ≥10 min)', 'Usage action-share HHI', 'Highest player USG_PCT']):
            ax.plot(range(1, 8), pd.to_numeric(part.get(col, pd.Series(np.nan, index=part.index)), errors='coerce'), marker='o', label=team)
            ax.set_ylabel(label)
            ax.grid(alpha=.2)
            ax.legend()
    axes[-1].set_xticks(range(1, 8))
    axes[-1].set_xlabel('Series game number (individual game data)')
    fig.suptitle(series_key.replace('_', ' ')+' — players with ≥10 minutes per game')
    fig.text(.5, .015, 'Only passes between players with ≥10 minutes in that game. Gaps indicate incomplete tracking.\n'
             'Usage HHI measures concentration of FGA + 0.44 FTA + TOV; individual NBA USG_PCT is saved separately.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0,.05,1,.96))
    fig.savefig(Path(output_dir)/'series_entropy_trends.png', dpi=180)
    fig.savefig(Path(output_dir)/'series_entropy_trends.pdf')
    plt.close(fig)


def run_series_analysis(argv):
    """Four-team case study; opt-in flag preserves all existing season CLI paths."""
    from nba_api.stats.endpoints import leaguegamefinder
    parser = argparse.ArgumentParser(description='Seven-game playoff entropy case studies')
    parser.add_argument('--series-analysis', action='store_true')
    parser.add_argument('--series-spec', action='append', help='SEASON:TEAM:OPPONENT; provide exactly two disjoint pairs')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--cache-dir', type=Path, default=Path.home()/'nba_passing_cache')
    parser.add_argument('--plot-only', action='store_true')
    parser.add_argument('--refresh', action='store_true')
    parser.add_argument('--timeout', type=float, default=60)
    parser.add_argument('--retries', type=int, default=4)
    parser.add_argument('--min-delay', type=float, default=2)
    parser.add_argument('--max-delay', type=float, default=3.5)
    args = parser.parse_args(argv)
    specs = args.series_spec or DEFAULT_SERIES_SPECS
    try:
        parsed = [s.split(':') for s in specs]
        if len(parsed) != 2 or any(len(s) != 3 for s in parsed):
            raise ValueError('Provide exactly two SEASON:TEAM:OPPONENT specifications')
        if len({t for _, a, b in parsed for t in (a, b)}) != 4:
            raise ValueError('The two series must contain four distinct teams')
        teams = {t['abbreviation']: t for t in static_teams.get_teams()}
        if any(t not in teams for _, a, b in parsed for t in (a,b)):
            raise ValueError('Unknown NBA team abbreviation')
    except ValueError as exc:
        parser.error(str(exc))
    root = args.output_dir.expanduser()/'seven_game_series'
    root.mkdir(parents=True, exist_ok=True)
    all_frames, failures = [], []
    for season, a, b in parsed:
        key = f'{season}_{a}_{b}'
        folder = root/key
        folder.mkdir(parents=True, exist_ok=True)
        path = folder/'game_metrics.csv'
        if args.plot_only:
            if not path.exists():
                LOG.error('No saved game data: %s', path)
                failures.append(key)
                continue
            frame = pd.read_csv(path)
        else:
            client = NBAClient(Config(season=season, season_type='Playoffs',
                cache_dir=args.cache_dir.expanduser(), refresh=args.refresh,
                timeout=args.timeout, retries=args.retries, min_delay=args.min_delay,
                max_delay=args.max_delay))
            try:
                schedule, _ = client.fetch(leaguegamefinder.LeagueGameFinder,
                    dict(player_or_team_abbreviation='T', team_id_nullable=teams[a]['id'],
                         vs_team_id_nullable=teams[b]['id'], season_nullable=season,
                         season_type_nullable='Playoffs'),
                    lambda p: validate_series_schedule(p, teams[a]['id'], b))
                atomic_csv(schedule, folder/'schedule.csv')
            except DataUnavailable as exc:
                LOG.error('%s schedule unavailable: %s', key, exc)
                failures.append(key)
                continue
            rows = []
            for number, game in enumerate(schedule.itertuples(), 1):
                for abbr, opponent in [(a,b),(b,a)]:
                    row = dict(series=key, season=season, team=abbr, opponent=opponent,
                        game_number=number, game_id=game.GAME_ID,
                        game_date=game.GAME_DATE.strftime('%Y-%m-%d'),
                        result=game.WL if abbr==a else ('L' if game.WL=='W' else 'W'))
                    print(f'{key}: {abbr} game {number}/7 ({row["game_date"]}) — downloading game data', flush=True)
                    try:
                        row.update(download_series_team_game(client, teams[abbr], teams[opponent],
                            game, folder/abbr/f'game_{number:02d}'))
                    except DataUnavailable as exc:
                        row.update(complete=False, error=str(exc))
                        LOG.warning('%s %s game %s: %s', key, abbr, number, exc)
                    print(f'  {abbr} game {number}: complete={row.get("complete", False)}, entropy={row.get("receiver_entropy")}', flush=True)
                    rows.append(row)
                    # Save progress after every team-game; cached requests are reusable.
                    atomic_csv(pd.DataFrame(rows), path)
            frame = pd.DataFrame(rows)
        for col in ['receiver_entropy_fixed15','receiver_entropy','usage_hhi','max_player_usage_rate']:
            if col not in frame:
                frame[col] = np.nan
        if frame.duplicated(['team','game_number']).any():
            raise ValueError('Duplicate team/game observations in saved series data')
        plot_series_game_trends(frame, folder, key)
        all_frames.append(frame)
    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        atomic_csv(combined, root/'all_series_game_metrics.csv')
        summary = combined.groupby(['series','team']).agg(
            valid_entropy_games=('receiver_entropy','count'),
            mean_entropy=('receiver_entropy','mean'),
            mean_fixed15_entropy=('receiver_entropy_fixed15','mean'),
            mean_usage_hhi=('usage_hhi','mean'),
            valid_usage_rate_games=('max_player_usage_rate','count'),
            mean_max_player_usage_rate=('max_player_usage_rate','mean')).reset_index()
        atomic_csv(summary, root/'four_team_comparison.csv')
        entropy_range = summary.mean_entropy.max()-summary.mean_entropy.min()
        usage_range = summary.mean_usage_hhi.max()-summary.mean_usage_hhi.min()
        atomic_json(dict(selection='Two prespecified seven-game matchups; ≥10 minutes in each game; no assumed high/low labels',
            teams=summary.team.tolist(), entropy_range=entropy_range if pd.notna(entropy_range) else None,
            usage_hhi_range=usage_range if pd.notna(usage_range) else None,
            note='Measured variation is reported, not guaranteed. Case studies are descriptive; seven games do not establish causation.'),
            root/'selection_audit.json')
        print(summary.to_string(index=False))
        print(f'Series outputs: {root}')
        if combined.receiver_entropy.count() < 28:
            LOG.warning('Fewer than 28 valid team-game entropy records; inspect gaps and error columns')
            failures.append('incomplete entropy coverage')
    return int(bool(failures) or not all_frames)


# ---------------------------------------------------------------------
# REBUILD EVERY SAVED GRAPH USING A CONSISTENT TEN-MINUTE POPULATION
# Season: >=10 MPG. Individual playoff game: >=10 minutes in that game.
# Old outputs remain available in a separate archive supplied as --source-dir.
# ---------------------------------------------------------------------
def rebuild_saved_ten_minute_analysis(argv):
    parser = argparse.ArgumentParser(description='Rebuild all saved graphs with a 10-minute player threshold; no downloads')
    parser.add_argument('--rebuild-saved-10min', action='store_true')
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--cache-dir', type=Path, default=Path.home()/'nba_passing_cache')
    args = parser.parse_args(argv)
    source, out = args.source_dir.expanduser(), args.output_dir.expanduser()
    if source.resolve() == out.resolve():
        parser.error('Use a separate source archive so full-player data are preserved')
    rosters = {}
    for path in args.cache_dir.expanduser().glob('leaguedashplayerstats_*.json'):
        try:
            snap = json.loads(path.read_text())
            q = snap['parameters']
            if q.get('MeasureType')=='Base' and q.get('PerMode')=='Totals' and not q.get('DateFrom') and not q.get('DateTo'):
                rosters[(q['Season'],q['SeasonType'],int(q['TeamID']))] = snap['payload']
        except (ValueError, KeyError):
            continue
    teams = {t['abbreviation']:t for t in static_teams.get_teams()}
    out.mkdir(parents=True, exist_ok=True)
    all_metrics = []
    seasons = sorted(source.glob('20??-??/Regular_Season'))
    for season_folder in seasons:
        season = season_folder.parent.name
        destination = out/season/'Regular_Season'
        destination.mkdir(parents=True, exist_ok=True)
        metrics_rows, pass_frames = [], []
        for matrix_path in sorted(season_folder.glob('*/passing_matrix.csv')):
            abbr = matrix_path.parent.name
            if abbr not in teams:
                continue
            payload = rosters.get((season,'Regular Season',teams[abbr]['id']))
            if payload is None:
                raise DataUnavailable(f'Minutes unavailable for {season} {abbr}; cannot include unverified players')
            roster = validate_roster(payload, teams[abbr]['id'], abbr)
            roster = roster.loc[roster.MIN/roster.GP >= 10.0].copy()
            ids = roster.PLAYER_ID.astype(int).tolist()
            saved = pd.read_csv(matrix_path, index_col=0)
            saved.index = saved.index.astype(int)
            saved.columns = saved.columns.astype(int)
            if not set(ids).issubset(saved.index):
                raise DataUnavailable(f'Qualified players absent from saved matrix: {season} {abbr}')
            matrix = saved.loc[ids,ids].to_numpy(float)
            missing = {ids[i]:'Missing outgoing tracking profile' for i in range(len(ids)) if np.isnan(matrix[i]).any()}
            names = dict(zip(ids, roster.PLAYER_NAME.astype(str)))
            actions = dict(zip(ids,(roster.FGA+.44*roster.FTA+roster.TOV).astype(float)))
            data = TeamData(team=abbr,season=season,season_type='Regular Season',
                player_ids=ids,names=names,matrix=matrix,missing=missing,snapshots=[],
                roster_player_ids=ids,field_goal_attempts=dict(zip(ids,roster.FGA.astype(int))),
                usage_actions=actions,settings={'minimum_player_mpg':10.0,'source_archive':str(source)})
            folder = destination/abbr
            metrics = save_team_data(data,folder)
            atomic_csv(roster,folder/'roster_stats.csv')
            plot_team_network(data,folder/'passing_network.png')
            metrics_rows.append(team_metric_row(metrics))
            pass_frames.append(passing_rows(data))
        frame = pd.DataFrame(metrics_rows,columns=METRIC_COLUMNS)
        atomic_csv(frame,destination/'team_metrics.csv')
        if pass_frames:
            atomic_csv(pd.concat(pass_frames,ignore_index=True),destination/'all_team_passes.csv')
        if len(frame):
            plot_league_quadrant(destination,season,'Regular Season')
            plot_starness_quadrant(destination,season,'Regular Season')
            plot_starness_quadrant(destination,season,'Regular Season',conditional=True)
            plot_league_mutual_information(destination,season,'Regular Season')
            run_regression_analysis(frame,destination)
            all_metrics.append(frame)
        print(f'{season}: rebuilt {len(frame)} season-team graphs with >=10 MPG',flush=True)
    if all_metrics:
        combined = pd.concat(all_metrics,ignore_index=True)
        combined['season_start_year'] = combined.season.str[:4].astype(int)
        combined = combined.sort_values(['season_start_year','team'])
        summary = out/'2020s_summary'/'Regular_Season'
        summary.mkdir(parents=True,exist_ok=True)
        atomic_csv(combined,summary/'all_seasons_team_metrics.csv')
        plot_decade_trends(combined,summary)
        plot_decade_team_scatter(combined,summary)
        run_regression_analysis(combined,summary)
    # Recompute every playoff outcome and usage statistic from the SAME retained IDs.
    series_source = source/'seven_game_series'
    series_destination = out/'seven_game_series'
    rebuilt_games = 0
    for metrics_path in sorted(series_source.glob('*/game_metrics.csv')):
        key = metrics_path.parent.name
        destination = series_destination/key
        destination.mkdir(parents=True,exist_ok=True)
        rows = []
        for record in pd.read_csv(metrics_path).to_dict(orient='records'):
            abbr,number = record['team'],int(record['game_number'])
            old = metrics_path.parent/abbr/f'game_{number:02d}'
            folder = destination/abbr/f'game_{number:02d}'
            folder.mkdir(parents=True,exist_ok=True)
            full_roster = pd.read_csv(old/'game_players.csv')
            roster = full_roster.loc[pd.to_numeric(full_roster.MIN,errors='raise')>=10.0].copy()
            ids = roster.PLAYER_ID.astype(int).tolist()
            saved = pd.read_csv(old/'passing_matrix.csv',index_col=0)
            saved.index = saved.index.astype(int)
            saved.columns = saved.columns.astype(int)
            matrix = saved.loc[ids,ids].to_numpy(float)
            missing = {ids[i]:'Missing outgoing profile' for i in range(len(ids)) if np.isnan(matrix[i]).any()}
            names = dict(zip(ids,roster.PLAYER_NAME.astype(str)))
            actions = dict(zip(ids,(roster.FGA+.44*roster.FTA+roster.TOV).astype(float)))
            data = TeamData(team=abbr,season=record['season'],
                season_type=f'Playoffs | Game {number} | {record["game_date"]}',
                player_ids=ids,names=names,matrix=matrix,missing=missing,snapshots=[],
                roster_player_ids=ids,field_goal_attempts=dict(zip(ids,roster.FGA.astype(int))),
                usage_actions=actions,settings={'minimum_player_minutes':10.0})
            atomic_csv(full_roster,folder/'game_players_source.csv')
            atomic_csv(roster,folder/'game_players.csv')
            atomic_csv(pd.DataFrame(matrix,index=ids,columns=ids),folder/'passing_matrix.csv',index=True)
            plot_team_network(data,folder/'passing_network_fruchterman.png')
            observed = np.nan_to_num(matrix,nan=0)
            total = observed.sum()
            entropy,conditional = compute_team_entropies(observed) if not missing else (None,None)
            _,mi = passing_mutual_information(observed) if not missing else (None,None)
            probs = observed.sum(axis=0)/total if total>0 else np.zeros(len(ids))
            raw = -float(np.sum(probs[probs>0]*np.log(probs[probs>0]))) if not missing and total>0 else None
            advanced = pd.read_csv(old/'game_advanced_usage.csv')
            advanced = advanced.loc[advanced.PLAYER_ID.isin(ids)].copy()
            atomic_csv(advanced,folder/'game_advanced_usage.csv')
            top = advanced.sort_values(['USG_PCT','PLAYER_ID'],ascending=[False,True]).iloc[0] if len(advanced) else None
            record.update(minimum_player_minutes=10.0,player_count=len(ids),
                receiver_entropy=entropy,receiver_entropy_nats=raw,
                receiver_entropy_fixed15=raw/np.log(15) if raw is not None else None,
                conditional_receiver_entropy=conditional,normalized_passing_mutual_information=mi,
                usage_hhi=usage_concentration(actions)[0],
                max_player_usage_rate=float(top.USG_PCT) if top is not None else None,
                max_usage_player=names.get(int(top.PLAYER_ID),'') if top is not None else None,
                complete=not missing,recorded_passes=float(total),
                missing_player_ids=';'.join(map(str,missing)),error='; '.join(missing.values()))
            rows.append(record)
            rebuilt_games += 1
        atomic_csv(pd.DataFrame(rows),destination/'game_metrics.csv')
        schedule = metrics_path.parent/'schedule.csv'
        if schedule.exists():
            atomic_csv(pd.read_csv(schedule,dtype={'GAME_ID':str}),destination/'schedule.csv')
    if rebuilt_games:
        run_series_analysis(['--series-analysis','--plot-only','--output-dir',str(out)])
    atomic_json({'season_minimum_mpg':10,'game_minimum_minutes':10,
        'source_archive':str(source),'season_teams':sum(map(len,all_metrics)),
        'playoff_team_games':rebuilt_games,
        'population_policy':'Only passes between eligible players; recompute all entropy, MI and usage HHI from the same IDs. Isolates omitted from drawings only.',
        'rebuilt_at':utc_now()},out/'ten_minute_filter_audit.json')
    print(f'Finished: {sum(map(len,all_metrics))} season-team networks and {rebuilt_games} playoff networks',flush=True)
    return 0




# ---------------------------------------------------------------------
# GOLDEN STATE 2021-22: ALL 82 REGULAR-SEASON GAMES
# No minutes cutoff. User policy: missing passing profiles become zero rows.
# --warriors-82 resumes saved games; --warriors-82 --plot-only redraws charts.
# ---------------------------------------------------------------------
def validate_warriors_82_schedule(payload):
    f = extract_team_metrics(payload, 'LeagueGameFinderResults',
        {'GAME_ID','GAME_DATE','TEAM_ID','MATCHUP','WL','PTS','PLUS_MINUS'})
    f = f.loc[pd.to_numeric(f.TEAM_ID,errors='coerce').eq(1610612744)].copy()
    f['GAME_ID'] = f.GAME_ID.astype(str).str.zfill(10)
    f['GAME_DATE'] = pd.to_datetime(f.GAME_DATE,errors='raise')
    f = f.sort_values('GAME_DATE').reset_index(drop=True)
    if len(f)!=82 or f.GAME_ID.nunique()!=82 or f.GAME_DATE.nunique()!=82:
        raise ValueError('Expected exactly 82 distinct Golden State regular-season games')
    if not f.GAME_ID.str.startswith('002').all() or not f.WL.isin(['W','L']).all():
        raise ValueError('Invalid regular-season schedule or game results')
    return f


def download_warriors_regular_game(client, team, opponent, game, folder, workers=2):
    """Actual single-date counts, including all players with positive minutes.

    Missing outgoing profiles are filled with zero per user instruction. Their
    IDs and error messages remain explicit; computed entropy is zero-imputed.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    day = game.GAME_DATE.strftime('%m/%d/%Y')
    filters = dict(date_from_nullable=day,date_to_nullable=day,opponent_team_id=opponent['id'])
    common = dict(season='2021-22',season_type_all_star='Regular Season',**filters)
    roster,_ = client.fetch(leaguedashplayerstats.LeagueDashPlayerStats,
        dict(team_id_nullable=team['id'],measure_type_detailed_defense='Base',per_mode_detailed='Totals',**common),
        lambda p:validate_single_game_roster(p,team))
    ids = roster.PLAYER_ID.astype(int).tolist()
    names = dict(zip(ids,roster.PLAYER_NAME.astype(str)))
    index = {pid:i for i,pid in enumerate(ids)}
    matrix = np.zeros((len(ids),len(ids)),dtype=float)
    failures = {}
    def fetch_player(pid):
        def validate(payload):
            f = validate_passing(payload,pid,team['id'])
            if 'G' in f and not pd.to_numeric(f.G,errors='coerce').eq(1).all():
                raise ValueError('Passing profile does not refer to one game')
            if f.PASS_TEAMMATE_PLAYER_ID.duplicated().any() or not f.PASS_TEAMMATE_PLAYER_ID.isin(ids).all():
                raise ValueError('Invalid receiving players in game passing profile')
            return f
        f,_ = client.fetch(playerdashptpass.PlayerDashPtPass,
            dict(team_id=team['id'],player_id=pid,per_mode_simple='Totals',**common),validate)
        return f
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_player,pid):pid for pid in ids}
        for future in as_completed(futures):
            pid = futures[future]
            try:
                passes = future.result()
                if passes.empty:
                    failures[pid] = 'Empty passing profile filled with zero'
                for row in passes.itertuples():
                    matrix[index[pid],index[int(row.PASS_TEAMMATE_PLAYER_ID)]] = float(row.PASS)
            except DownloadsStopped:
                for pending in futures:pending.cancel()
                raise
            except DataUnavailable as exc:
                failures[pid] = str(exc)
    actions = dict(zip(ids,(roster.FGA+.44*roster.FTA+roster.TOV).astype(float)))
    folder.mkdir(parents=True,exist_ok=True)
    atomic_csv(roster,folder/'game_players.csv')
    atomic_csv(pd.DataFrame(matrix,index=ids,columns=ids),folder/'passing_matrix.csv',index=True)
    total = float(matrix.sum())
    entropy,conditional = compute_team_entropies(matrix) if total>0 else (0.0,0.0)
    _,mi = passing_mutual_information(matrix) if total>0 else (0.0,0.0)
    probs = matrix.sum(axis=0)/total if total>0 else np.zeros(len(ids))
    raw = -float(np.sum(probs[probs>0]*np.log(probs[probs>0])))
    # Individual usage is saved as in the playoff analysis, without a minute filter.
    advanced_error,max_usage,max_name = '',None,None
    try:
        def validate_advanced(payload):
            f = extract_team_metrics(payload,'LeagueDashPlayerStats',{'PLAYER_ID','TEAM_ID','GP','USG_PCT'})
            f = f.loc[pd.to_numeric(f.TEAM_ID,errors='coerce').eq(team['id'])].copy()
            f['GP'] = pd.to_numeric(f.GP,errors='raise')
            f['USG_PCT'] = pd.to_numeric(f.USG_PCT,errors='raise')
            if f.empty or not f.GP.eq(1).all() or not f.USG_PCT.between(0,1).all():
                raise ValueError('Invalid single-game usage data')
            return f
        advanced,_ = client.fetch(leaguedashplayerstats.LeagueDashPlayerStats,
            dict(team_id_nullable=team['id'],measure_type_detailed_defense='Advanced',per_mode_detailed='Totals',**common),validate_advanced)
        advanced = advanced.loc[advanced.PLAYER_ID.isin(ids)]
        atomic_csv(advanced,folder/'game_advanced_usage.csv')
        if len(advanced):
            top = advanced.sort_values(['USG_PCT','PLAYER_ID'],ascending=[False,True]).iloc[0]
            max_usage,max_name = float(top.USG_PCT),names.get(int(top.PLAYER_ID),'')
    except DownloadsStopped:
        raise
    except DataUnavailable as exc:
        advanced_error = str(exc)
    data = TeamData(team='GSW',season='2021-22',season_type=f'Regular Season | {day}',
        player_ids=ids,names=names,matrix=matrix,missing=failures,snapshots=[],roster_player_ids=ids,
        field_goal_attempts=dict(zip(ids,roster.FGA.astype(int))),usage_actions=actions,
        settings={'minimum_player_minutes':0,'missing_pass_policy':'zero'})
    # Missing-profile gray nodes still identify imputation; isolates are hidden.
    plot_team_network(data,folder/'passing_network_fruchterman.png')
    atomic_json({'missing_pass_policy':'zero','zero_filled_player_errors':{str(k):v for k,v in failures.items()},
        'zero_total_passes':total==0,'minimum_player_minutes':0},folder/'coverage_audit.json')
    return dict(analysis_version='gsw82_v1_zero_no_minutes',minimum_player_minutes=0,
        receiver_entropy=entropy,receiver_entropy_nats=raw,receiver_entropy_fixed15=raw/np.log(15),
        effective_receivers=float(np.exp(raw)) if total>0 else 0.0,
        conditional_receiver_entropy=conditional,normalized_passing_mutual_information=mi,
        usage_hhi=usage_concentration(actions)[0],max_player_usage_rate=max_usage,max_usage_player=max_name,
        player_count=len(ids),recorded_passes=total,passing_zero_filled=bool(failures),
        zero_filled_profile_count=len(failures),zero_filled_player_ids=';'.join(map(str,sorted(failures))),
        zero_total_passes=total==0,passing_errors='; '.join(failures.values()),usage_error=advanced_error)


def download_warriors_receiver_game(client, team, opponent, game, folder, workers=1):
    """Receiver entropy requires player receiving totals, not every pairwise edge.

    LeagueDashPtStats(Passing, Totals, single date) supplies those marginals
    directly. All players with positive game minutes are retained. Missing
    player receiving records become zero under the user's chosen policy.
    Pairwise MI/conditional entropy are not inferred from marginal totals.
    """
    from nba_api.stats.endpoints import leaguedashptstats
    day = game.GAME_DATE.strftime('%m/%d/%Y')
    common = dict(season='2021-22',season_type_all_star='Regular Season',
        date_from_nullable=day,date_to_nullable=day,opponent_team_id=opponent['id'])
    roster,_ = client.fetch(leaguedashplayerstats.LeagueDashPlayerStats,
        dict(team_id_nullable=team['id'],measure_type_detailed_defense='Base',per_mode_detailed='Totals',**common),
        lambda p:validate_single_game_roster(p,team))
    ids = roster.PLAYER_ID.astype(int).tolist()
    names = dict(zip(ids,roster.PLAYER_NAME.astype(str)))
    folder.mkdir(parents=True,exist_ok=True)
    atomic_csv(roster,folder/'game_players.csv')
    error = ''
    def validate_receiving(payload):
        f = extract_team_metrics(payload,'LeagueDashPtStats',{'PLAYER_ID','TEAM_ID','GP','PASSES_RECEIVED'})
        for col in ['PLAYER_ID','TEAM_ID','GP','PASSES_RECEIVED']:
            numeric_column(f,col,integer=True,positive=col=='PLAYER_ID')
        f = f.loc[f.TEAM_ID.eq(team['id'])].copy()
        if not f.GP.eq(1).all() or f.PLAYER_ID.duplicated().any():
            raise ValueError('Receiving totals do not identify one game per player')
        return f
    try:
        tracking,_ = client.fetch(leaguedashptstats.LeagueDashPtStats,
            dict(team_id_nullable=team['id'],player_or_team='Player',pt_measure_type='Passing',per_mode_simple='Totals',**common),
            validate_receiving)
        atomic_csv(tracking,folder/'receiving_tracking.csv')
        counts = tracking.set_index('PLAYER_ID').PASSES_RECEIVED.reindex(ids)
    except DownloadsStopped:
        raise
    except DataUnavailable as exc:
        counts = pd.Series(np.nan,index=ids)
        error = str(exc)
    missing = counts[counts.isna()].index.astype(int).tolist()
    counts = counts.fillna(0).astype(float)
    total = float(counts.sum())
    probabilities = counts.to_numpy()/total if total>0 else np.zeros(len(ids))
    raw = -float(np.sum(probabilities[probabilities>0]*np.log(probabilities[probabilities>0])))
    entropy = raw/np.log(len(ids)) if len(ids)>1 and total>0 else 0.0
    receiver_table = pd.DataFrame({'player_id':ids,'name':[names[pid] for pid in ids],
        'minutes':roster.MIN.to_numpy(),'passes_received':counts.to_numpy(),
        'zero_filled':[pid in missing for pid in ids]})
    atomic_csv(receiver_table,folder/'receiver_counts.csv')
    actions = dict(zip(ids,(roster.FGA+.44*roster.FTA+roster.TOV).astype(float)))
    advanced_error,max_usage,max_name = '',None,None
    try:
        def validate_advanced(payload):
            f = extract_team_metrics(payload,'LeagueDashPlayerStats',{'PLAYER_ID','TEAM_ID','GP','USG_PCT'})
            f = f.loc[pd.to_numeric(f.TEAM_ID,errors='coerce').eq(team['id'])].copy()
            f['GP'] = pd.to_numeric(f.GP,errors='raise')
            f['USG_PCT'] = pd.to_numeric(f.USG_PCT,errors='raise')
            if f.empty or not f.GP.eq(1).all() or not f.USG_PCT.between(0,1).all():
                raise ValueError('Invalid single-game usage data')
            return f
        advanced,_ = client.fetch(leaguedashplayerstats.LeagueDashPlayerStats,
            dict(team_id_nullable=team['id'],measure_type_detailed_defense='Advanced',per_mode_detailed='Totals',**common),validate_advanced)
        advanced = advanced.loc[advanced.PLAYER_ID.isin(ids)]
        atomic_csv(advanced,folder/'game_advanced_usage.csv')
        if len(advanced):
            top = advanced.sort_values(['USG_PCT','PLAYER_ID'],ascending=[False,True]).iloc[0]
            max_usage,max_name = float(top.USG_PCT),names.get(int(top.PLAYER_ID),'')
    except DownloadsStopped:
        raise
    except DataUnavailable as exc:
        advanced_error = str(exc)
    atomic_json({'source':'LeagueDashPtStats Passing: PASSES_RECEIVED, single game',
        'missing_pass_policy':'zero','zero_filled_player_ids':missing,'tracking_error':error,
        'minimum_player_minutes':0,'zero_total_passes':total==0},folder/'coverage_audit.json')
    return dict(analysis_version='gsw82_v2_receiving_totals_zero_no_minutes',minimum_player_minutes=0,
        receiver_entropy=entropy,receiver_entropy_nats=raw,receiver_entropy_fixed15=raw/np.log(15),
        effective_receivers=float(np.exp(raw)) if total>0 else 0.,usage_hhi=usage_concentration(actions)[0],
        max_player_usage_rate=max_usage,max_usage_player=max_name,player_count=len(ids),recorded_passes=total,
        passing_zero_filled=bool(missing),zero_filled_profile_count=len(missing),
        zero_filled_player_ids=';'.join(map(str,missing)),zero_total_passes=total==0,
        passing_errors=error or ('Missing receiving records set to zero' if missing else ''),usage_error=advanced_error,
        passing_data_source='single-game PASSES_RECEIVED totals')


def plot_warriors_82_trends(frame,output_dir):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    out = Path(output_dir)
    d = frame.set_index('game_number').reindex(range(1,83))
    for columns,stem,title in [
        (['receiver_entropy_fixed15','receiver_entropy','usage_hhi','max_player_usage_rate'],
         'warriors_82_game_trends','Golden State Warriors 2021–22 — 82-game regular season'),
        (['receiver_entropy','effective_receivers'],
         'warriors_entropy_wins_losses','Golden State Warriors — passing entropy and effective receivers')]:
        fig,axes = plt.subplots(len(columns),1,figsize=(17,3.0*len(columns)),sharex=True,squeeze=False)
        labels = {'receiver_entropy_fixed15':'Receiver entropy\n÷ ln(15)',
            'receiver_entropy':'Normalized\nreceiver entropy',
            'usage_hhi':'Usage HHI','max_player_usage_rate':'Highest player\nusage rate',
            'effective_receivers':'Effective receivers\nexp(raw entropy)'}
        for ax,col in zip(axes[:,0],columns):
            y = pd.to_numeric(d.get(col,pd.Series(np.nan,index=d.index)),errors='coerce')
            ax.plot(d.index,y,color='#b8bdc5',lw=1,alpha=.75,zorder=1)
            ax.plot(d.index,y.rolling(7,min_periods=4).mean().where(y.notna()),color='#244b82',lw=2,alpha=.85,zorder=2)
            for result,color in [('W','#168348'),('L','#d23b3b')]:
                mask = d.result.eq(result)&y.notna()
                ax.scatter(d.index[mask],y[mask],s=35,color=color,edgecolor='white',linewidth=.35,zorder=3)
            if 'passing_zero_filled' in d and col not in ['usage_hhi','max_player_usage_rate']:
                mask = d.passing_zero_filled.fillna(False).astype(str).str.lower().eq('true')&y.notna()
                ax.scatter(d.index[mask],y[mask],s=65,facecolors='none',edgecolors='#333333',linewidth=.8,zorder=4)
            ax.set_ylabel(labels[col]);ax.grid(alpha=.17);ax.spines[['top','right']].set_visible(False)
        axes[-1,0].set_xticks([1,*range(10,81,10),82]);axes[-1,0].set_xlim(0,83)
        axes[-1,0].set_xlabel('Regular-season game number, in date order')
        legend=[Line2D([0],[0],marker='o',color='none',markerfacecolor='#168348',markeredgecolor='none',label='Win'),
                Line2D([0],[0],marker='o',color='none',markerfacecolor='#d23b3b',markeredgecolor='none',label='Loss'),
                Line2D([0],[0],color='#244b82',lw=2,label='Trailing 7-game average'),
                Line2D([0],[0],marker='o',color='none',markerfacecolor='none',markeredgecolor='#333333',label='Includes unavailable receiving records set to zero')]
        fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(.5,.965),ncol=4,frameon=False,fontsize=9)
        fig.suptitle(title,fontsize=16,y=.995)
        fig.text(.5,.012,'All players who appeared; no minutes cutoff. Unavailable receiving records are zero-filled by request.\n'
            'Zero-filled entropy is an imputed estimate; exp(H) is set to 0 when no passes are observed. Fixed-15 entropy is not bounded at 1.',ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.055,1,.925))
        fig.savefig(out/(stem+'.png'),dpi=170)
        fig.savefig(out/(stem+'.pdf'))
        plt.close(fig)


def run_warriors_82(argv):
    from nba_api.stats.endpoints import leaguegamefinder
    parser = argparse.ArgumentParser(description='GSW 2021-22: 82 individual regular-season games; no minute cutoff; missing passes=0')
    parser.add_argument('--warriors-82',action='store_true')
    parser.add_argument('--plot-only',action='store_true')
    parser.add_argument('--refresh',action='store_true')
    parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--cache-dir',type=Path,default=None)
    parser.add_argument('--timeout',type=float,default=25)
    parser.add_argument('--retries',type=int,default=2)
    parser.add_argument('--min-delay',type=float,default=.5)
    parser.add_argument('--max-delay',type=float,default=1.)
    parser.add_argument('--workers',type=int,choices=[1,2],default=2)
    args = parser.parse_args(argv)
    out = args.output_dir.expanduser()/'GSW_2021-22_82_games'
    out.mkdir(parents=True,exist_ok=True)
    path = out/'game_metrics.csv'
    if args.plot_only:
        f = pd.read_csv(path)
    else:
        teams = {t['abbreviation']:t for t in static_teams.get_teams()}
        client = NBAClient(Config(season='2021-22',season_type='Regular Season',
            cache_dir=args.cache_dir.expanduser() if args.cache_dir else out/'cache',refresh=args.refresh,
            timeout=args.timeout,retries=args.retries,min_delay=args.min_delay,max_delay=args.max_delay))
        rows=[]
        if path.exists() and not args.refresh:
            saved = pd.read_csv(path,dtype={'game_id':str})
            if 'analysis_version' in saved and saved.analysis_version.eq('gsw82_v2_receiving_totals_zero_no_minutes').all():
                rows = saved.to_dict(orient='records')
        done = {str(r['game_id']).zfill(10) for r in rows}
        try:
            schedule,_ = client.fetch(leaguegamefinder.LeagueGameFinder,
                dict(player_or_team_abbreviation='T',team_id_nullable=1610612744,season_nullable='2021-22',season_type_nullable='Regular Season'),validate_warriors_82_schedule)
            atomic_csv(schedule,out/'schedule.csv')
            print(f'Validated 82 games: {schedule.WL.eq("W").sum()} wins, {schedule.WL.eq("L").sum()} losses. Resuming {len(done)} saved games.',flush=True)
            for number,game in enumerate(schedule.itertuples(),1):
                if game.GAME_ID in done:continue
                opponent = game.MATCHUP.split()[-1]
                row=dict(game_number=number,game_id=game.GAME_ID,game_date=game.GAME_DATE.strftime('%Y-%m-%d'),
                    team='GSW',season='2021-22',opponent=opponent,result=game.WL,
                    points=float(game.PTS),point_margin=float(game.PLUS_MINUS))
                print(f'GSW game {number}/82 | {row["game_date"]} vs {opponent} | {game.WL}',flush=True)
                try:
                    row.update(download_warriors_receiver_game(client,teams['GSW'],teams[opponent],game,out/f'game_{number:02d}',args.workers))
                except DownloadsStopped:
                    raise
                except DataUnavailable as exc:
                    # A game with an unavailable roster cannot identify player participation.
                    # Keep it explicit, rather than inventing a full player population.
                    row.update(analysis_version='gsw82_v2_receiving_totals_zero_no_minutes',minimum_player_minutes=0,
                        receiver_entropy=0.,receiver_entropy_nats=0.,receiver_entropy_fixed15=0.,effective_receivers=0.,
                        conditional_receiver_entropy=0.,normalized_passing_mutual_information=0.,
                        usage_hhi=None,max_player_usage_rate=None,player_count=0,recorded_passes=0.,
                        passing_zero_filled=True,zero_filled_profile_count=0,zero_filled_player_ids='',zero_total_passes=True,
                        passing_errors='Roster unavailable; game assigned zero entropy per missing=0 policy: '+str(exc))
                rows.append(row)
                f = pd.DataFrame(rows).sort_values('game_number')
                atomic_csv(f,path)
                print(f'  entropy={row["receiver_entropy"]:.4f}; zero-filled profiles={row["zero_filled_profile_count"]}',flush=True)
                if number%10==0:plot_warriors_82_trends(f,out)
        finally:
            client.close()
        f = pd.DataFrame(rows).sort_values('game_number')
    if f.duplicated('game_number').any() or f.duplicated('game_id').any():
        raise ValueError('Duplicate games in Warriors results')
    # Explicit, source-verified score corrections preserve the original schedule.
    corrections_path = out/'verified_score_corrections.csv'
    if corrections_path.exists():
        corrections = pd.read_csv(corrections_path,dtype={'game_id':str})
        for correction in corrections.itertuples():
            mask = f.game_id.astype(str).str.zfill(10).eq(str(correction.game_id).zfill(10))
            f.loc[mask,'point_margin'] = float(correction.corrected_point_margin)
        atomic_csv(f,path)
    plot_warriors_82_trends(f,out)
    numeric=['receiver_entropy','receiver_entropy_fixed15','effective_receivers','usage_hhi','max_player_usage_rate']
    atomic_csv(f.groupby('result')[numeric].agg(['mean','median','count']),out/'wins_losses_summary.csv',index=True)
    filled=f.passing_zero_filled.astype(str).str.lower().eq('true')
    atomic_json(dict(team='GSW',season='2021-22',expected_games=82,processed_games=len(f),
        wins=int(f.result.eq('W').sum()),losses=int(f.result.eq('L').sum()),minimum_player_minutes=0,
        missing_passing_policy='zero',games_with_zero_filled_profiles=int(filled.sum()),
        zero_total_pass_games=int(f.zero_total_passes.astype(str).str.lower().eq('true').sum()),
        distinction='Games are processed; imputed zero profiles do not establish complete observed tracking.'),out/'analysis_audit.json')
    print(f'Finished {len(f)}/82 games. Games containing zero-filled profiles: {filled.sum()}. Output: {out}',flush=True)
    return int(len(f)!=82)




# ---------------------------------------------------------------------
# GAME-TO-GAME RECEIVING-SHARE STABILITY (saved Warriors games; no downloads)
# Run: python teamgraphs.py --warriors-stability
# Jensen-Shannon DIVERGENCE in bits (not its square-root distance): 0..1.
# ---------------------------------------------------------------------
def receiving_share_js_divergence(first, second):
    """Base-2 Jensen-Shannon divergence; zeros are valid, zero totals are not."""
    p,q = np.asarray(first,dtype=float),np.asarray(second,dtype=float)
    if p.ndim!=1 or p.shape!=q.shape or not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError('Receiving distributions must be equal-length finite vectors')
    if (p<0).any() or (q<0).any() or p.sum()<=0 or q.sum()<=0:
        raise ValueError('Receiving distributions need nonnegative counts and positive totals')
    p,q = p/p.sum(),q/q.sum()
    middle = (p+q)/2
    positive_p,positive_q = p>0,q>0
    value = .5*np.sum(p[positive_p]*np.log2(p[positive_p]/middle[positive_p]))
    value += .5*np.sum(q[positive_q]*np.log2(q[positive_q]/middle[positive_q]))
    return float(np.clip(value,0,1))


def warriors_receiving_records(game_folder):
    """ID-indexed counts for every player who appeared, including observed zeros."""
    f = pd.read_csv(Path(game_folder)/'receiver_counts.csv')
    required = {'player_id','passes_received','zero_filled'}
    if not required.issubset(f):
        raise ValueError('Saved receiving records lack IDs, counts, or coverage flags')
    f['player_id'] = pd.to_numeric(f.player_id,errors='raise').astype(int)
    f['passes_received'] = pd.to_numeric(f.passes_received,errors='raise')
    if f.player_id.duplicated().any() or not np.isfinite(f.passes_received).all() or f.passes_received.lt(0).any():
        raise ValueError('Duplicate players or invalid receiving counts')
    if f.zero_filled.astype(str).str.lower().isin(['true','1']).any():
        raise ValueError('Receiving records include unavailable values filled with zero')
    if f.passes_received.sum()<=0:
        raise ValueError('No positive observed receiving total')
    return f.set_index('player_id')


def calculate_warriors_stability(frame, game_root):
    """Consecutive games only; any zero-filled game invalidates both adjacent pairs.

    Full-roster JSD aligns the union of player IDs, with absent players assigned
    zero shares. Returning-player JSD retains the intersection of the two game
    rosters and renormalizes each game's counts over that identical player set.
    Neither variant is a measure of passing order or possession sequences.
    """
    d = frame.sort_values('game_number').copy()
    if d.game_number.duplicated().any() or d.game_id.duplicated().any():
        raise ValueError('Duplicate game numbers or IDs in Warriors metrics')
    games = d.to_dict(orient='records')
    rows,contributions = [],[]
    for i,current in enumerate(games):
        row = {k:current[k] for k in ['game_number','game_id','game_date','opponent','result','point_margin']}
        row.update(previous_game_number=None,previous_result=None,comparison_available=False,
            js_divergence_all_players=np.nan,js_divergence_returning_players=np.nan,
            entropy_change_nats=np.nan,absolute_entropy_change_nats=np.nan,
            normalized_entropy_change=np.nan,player_turnover_fraction=np.nan,
            common_player_count=np.nan,previous_common_receiver_share=np.nan,
            current_common_receiver_share=np.nan,reason='')
        if i==0:
            row['reason']='First game: no previous game'
            rows.append(row)
            continue
        previous = games[i-1]
        row['previous_game_number']=previous['game_number']
        row['previous_result']=previous['result']
        if int(current['game_number'])!=int(previous['game_number'])+1:
            row['reason']='Nonconsecutive saved game numbers; comparison not bridged'
            rows.append(row)
            continue
        try:
            for record in [previous,current]:
                if str(record.get('passing_zero_filled',False)).lower() in ('true','1'):
                    raise ValueError(f'Game {record["game_number"]} contains zero-filled unavailable receiving data')
            a = warriors_receiving_records(Path(game_root)/f'game_{int(previous["game_number"]):02d}')
            b = warriors_receiving_records(Path(game_root)/f'game_{int(current["game_number"]):02d}')
            union = sorted(set(a.index)|set(b.index))
            common = sorted(set(a.index)&set(b.index))
            p = a.passes_received.reindex(union,fill_value=0).to_numpy(float)
            q = b.passes_received.reindex(union,fill_value=0).to_numpy(float)
            row['js_divergence_all_players']=receiving_share_js_divergence(p,q)
            row['player_turnover_fraction']=1-len(common)/len(union)
            row['common_player_count']=len(common)
            a_common,b_common = a.passes_received.reindex(common),b.passes_received.reindex(common)
            row['previous_common_receiver_share']=float(a_common.sum()/a.passes_received.sum())
            row['current_common_receiver_share']=float(b_common.sum()/b.passes_received.sum())
            if len(common)>=2 and a_common.sum()>0 and b_common.sum()>0:
                row['js_divergence_returning_players']=receiving_share_js_divergence(a_common,b_common)
            # Recalculate raw entropy directly from the observed shares for consistency.
            p,q = p/p.sum(),q/q.sum()
            h_previous=-float(np.sum(p[p>0]*np.log(p[p>0])))
            h_current=-float(np.sum(q[q>0]*np.log(q[q>0])))
            row['entropy_change_nats']=h_current-h_previous
            row['absolute_entropy_change_nats']=abs(h_current-h_previous)
            row['normalized_entropy_change']=float(current['receiver_entropy']-previous['receiver_entropy'])
            row['comparison_available']=True
            # Attribute all-player JSD to changes in each player's receiving share.
            midpoint=(p+q)/2
            pieces=np.zeros(len(union))
            mask=p>0;pieces[mask]+=.5*p[mask]*np.log2(p[mask]/midpoint[mask])
            mask=q>0;pieces[mask]+=.5*q[mask]*np.log2(q[mask]/midpoint[mask])
            for j,pid in enumerate(union):
                name = b.loc[pid,'name'] if pid in b.index and 'name' in b else (
                    a.loc[pid,'name'] if pid in a.index and 'name' in a else str(pid))
                contributions.append(dict(game_number=current['game_number'],previous_game_number=previous['game_number'],
                    player_id=pid,name=name,previous_receiver_share=p[j],current_receiver_share=q[j],
                    js_divergence_contribution=pieces[j],appeared_in_both=pid in common))
        except (OSError,ValueError,KeyError,TypeError) as exc:
            row['reason']=str(exc)
        rows.append(row)
    return pd.DataFrame(rows),pd.DataFrame(contributions)


def plot_warriors_stability(frame, output_dir):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    d = frame.set_index('game_number').reindex(range(2,83))
    fig,axes = plt.subplots(4,1,figsize=(17,12),sharex=True)
    panels=[('js_divergence_all_players','Receiving-share change\nAll players (JSD)'),
            ('js_divergence_returning_players','Receiving-share change\nReturning players (JSD)'),
            ('entropy_change_nats','Change in raw entropy\nCurrent − previous (nats)'),
            ('player_turnover_fraction','Player turnover\n1 − intersection / union')]
    for ax,(col,label) in zip(axes,panels):
        y = pd.to_numeric(d[col],errors='coerce')
        ax.plot(d.index,y,color='#b8bdc5',lw=1,alpha=.8,zorder=1)
        for outcome,color in [('W','#168348'),('L','#d23b3b')]:
            mask=d.result.eq(outcome)&y.notna()
            ax.scatter(d.index[mask],y[mask],color=color,s=34,edgecolors='white',linewidth=.3,zorder=2)
        for number in d.index[d.comparison_available.ne(True)]:
            ax.axvspan(number-.35,number+.35,color='#999999',alpha=.13,zorder=0)
        ax.set_ylabel(label);ax.grid(alpha=.17);ax.spines[['top','right']].set_visible(False)
        if col in ['js_divergence_all_players','js_divergence_returning_players','player_turnover_fraction']:
            ax.set_ylim(bottom=0)
        else:ax.axhline(0,color='#777777',lw=.8)
    axes[-1].set_xticks([2,*range(10,81,10),82]);axes[-1].set_xlim(1,83)
    axes[-1].set_xlabel('Current regular-season game number; each point compares with the immediately preceding game')
    legend=[Line2D([0],[0],marker='o',color='none',markerfacecolor='#168348',markeredgecolor='none',label='Current game: win'),
        Line2D([0],[0],marker='o',color='none',markerfacecolor='#d23b3b',markeredgecolor='none',label='Current game: loss'),
        Line2D([0],[0],color='#999999',lw=7,alpha=.3,label='Unavailable comparison')]
    fig.suptitle('Golden State Warriors 2021–22 — game-to-game receiving-share stability',fontsize=16,y=.995)
    fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(.5,.97),ncol=3,frameon=False)
    fig.text(.5,.012,'Lower JSD = more similar receiving shares (base-2 divergence, theoretical range 0–1). No minutes cutoff.\n'
        'Returning-player shares are renormalized over players appearing in both games. Zero-filled missing games invalidate adjacent comparisons.',
        ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.055,1,.93))
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    fig.savefig(out/'warriors_game_to_game_stability.png',dpi=170)
    fig.savefig(out/'warriors_game_to_game_stability.pdf')
    plt.close(fig)


def run_warriors_stability(argv):
    from scipy.stats import pearsonr,pointbiserialr
    parser=argparse.ArgumentParser(description='Warriors 2021-22 stability from saved per-game receiving shares; no downloads')
    parser.add_argument('--warriors-stability',action='store_true')
    parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT_DIR)
    args=parser.parse_args(argv)
    game_root=args.output_dir.expanduser()/'GSW_2021-22_82_games'
    out=game_root/'stability';out.mkdir(parents=True,exist_ok=True)
    frame=pd.read_csv(game_root/'game_metrics.csv',dtype={'game_id':str})
    result,contributions=calculate_warriors_stability(frame,game_root)
    atomic_csv(result,out/'game_to_game_stability.csv')
    atomic_csv(contributions,out/'player_share_change_contributions.csv')
    valid=result.loc[result.comparison_available.eq(True)].copy()
    metrics=['js_divergence_all_players','js_divergence_returning_players','absolute_entropy_change_nats','player_turnover_fraction']
    for grouping,filename in [('result','current_game_wins_losses.csv'),('previous_result','changes_after_wins_losses.csv')]:
        summary=valid.groupby(grouping)[metrics].agg(['mean','median','count'])
        summary.columns=['_'.join(col) for col in summary.columns]
        atomic_csv(summary.reset_index(),out/filename)
    stats=[]
    for metric in metrics:
        sample=valid.dropna(subset=[metric,'point_margin'])
        row=dict(metric=metric,n=len(sample),current_win_correlation=np.nan,current_win_p_value=np.nan,
            point_margin_correlation=np.nan,point_margin_p_value=np.nan)
        if len(sample)>=3 and sample[metric].nunique()>1:
            if sample.result.nunique()>1:
                r=pointbiserialr(sample.result.eq('W').astype(int),sample[metric])
                row.update(current_win_correlation=float(r.statistic),current_win_p_value=float(r.pvalue))
            if sample.point_margin.nunique()>1:
                r=pearsonr(sample.point_margin,sample[metric])
                row.update(point_margin_correlation=float(r.statistic),point_margin_p_value=float(r.pvalue))
        stats.append(row)
    atomic_csv(pd.DataFrame(stats),out/'exploratory_correlations.csv')
    atomic_csv(valid.sort_values('js_divergence_all_players'),out/'comparisons_ranked_by_stability.csv')
    plot_warriors_stability(result,out)
    atomic_json(dict(games=len(frame),possible_consecutive_comparisons=max(len(frame)-1,0),
        valid_comparisons=len(valid),unavailable_current_game_numbers=result.loc[(result.game_number>1)&~result.comparison_available,'game_number'].astype(int).tolist(),
        definition='Base-2 Jensen-Shannon divergence, NOT square-root distance; 0 identical, 1 disjoint',
        all_players='Union of appeared-player IDs; absent players have zero share',
        returning_players='Intersection of appeared-player IDs; renormalize both distributions over that set',
        minutes_cutoff=0,missing_policy='Original zero-filled season graph preserved; exclude transitions touching an unavailable game',
        inference='Exploratory unadjusted correlations; adjacent comparisons share games and are not independent; no causal or predictive claim',
        classification='Continuous divergence; no arbitrary stable/unstable threshold'),out/'stability_audit.json')
    print(f'Computed {len(valid)} valid comparisons from {len(frame)} games. Outputs: {out}',flush=True)
    return 0




# ---------------------------------------------------------------------
# WARRIORS TEAM PERFORMANCE VS EIGENVECTOR PASSING CONCENTRATION
# python teamgraphs.py --warriors-eigenvector-performance
# --plot-only uses saved networks; no API requests.
# Primary network: undirected weight A_ij + A_ji, combining passing and receiving.
# The concentration index below is a custom score, not Freeman centralization.
# ---------------------------------------------------------------------
def eigenvector_passing_concentration(matrix, player_ids):
    """Return primary EC concentration and the exact analyzed player population.

    Eigenvectors are L1-normalized to shares p_i. Concentration is
    (sum(p_i**2) - 1/N)/(1 - 1/N). Zero is equal centrality; higher values
    mean greater inequality. One is a theoretical share-distribution bound,
    not a claim that an undirected star graph attains it. Isolates are omitted.
    Missing weights are zero by user policy; no-edge/disconnected networks do
    not get a fabricated balance score. Incoming/outgoing direction is not
    distinguished by this primary combined-involvement network.
    """
    import networkx as nx
    a=np.asarray(matrix,dtype=float)
    ids=list(map(int,player_ids))
    if a.shape!=(len(ids),len(ids)) or len(set(ids))!=len(ids):
        raise ValueError('Matrix/ID mismatch or duplicate players')
    if np.isinf(a).any() or np.nanmin(a,initial=0)<0 or np.any(np.nan_to_num(np.diag(a))!=0):
        raise ValueError('Invalid passing weights or self-passes')
    a=np.nan_to_num(a,nan=0.0)
    active=np.flatnonzero((a.sum(axis=0)+a.sum(axis=1))>0)
    selected=[ids[i] for i in active]
    a=a[np.ix_(active,active)]
    result=dict(network_valid=False,eigenvector_concentration=np.nan,eigenvector_balance=np.nan,
        eigenvector_hhi=np.nan,network_player_count=len(selected),isolated_player_count=len(ids)-len(selected),
        top_eigenvector_player_id=None,top_eigenvector_share=np.nan,network_error='')
    if len(selected)<2 or a.sum()<=0:
        result['network_error']='No observed passing network; eigenvector balance undefined'
        return result,{},selected,a
    graph=nx.from_numpy_array(a+a.T)
    if not nx.is_connected(graph):
        result['network_error']='Disconnected active network: no single comparable eigenvector balance score'
        return result,{},selected,a
    largest=max(v['weight'] for _,_,v in graph.edges(data=True))
    for _,_,v in graph.edges(data=True):v['weight']/=largest
    try:
        centrality=nx.eigenvector_centrality(graph,weight='weight',max_iter=5000,tol=1e-10)
    except nx.PowerIterationFailedConvergence:
        result['network_error']='Eigenvector iteration did not converge'
        return result,{},selected,a
    values=np.array([centrality[i] for i in range(len(selected))],float)
    shares=values/values.sum();hhi=float(shares@shares);n=len(shares)
    concentration=float(np.clip((hhi-1/n)/(1-1/n),0,1))
    scores=dict(zip(selected,shares.tolist()))
    top=min(selected,key=lambda pid:(-scores[pid],pid))
    result.update(network_valid=True,eigenvector_concentration=concentration,
        eigenvector_balance=1-concentration,eigenvector_hhi=hhi,
        top_eigenvector_player_id=top,top_eigenvector_share=scores[top])
    return result,scores,selected,a


def download_warriors_passing_matrix(client,game_root,game_record,workers=2):
    """Reconstruct actual directed pair counts; receiving totals alone are insufficient."""
    from concurrent.futures import ThreadPoolExecutor,as_completed
    number=int(game_record['game_number']);folder=Path(game_root)/f'game_{number:02d}'
    roster=pd.read_csv(folder/'game_players.csv')
    ids=roster.PLAYER_ID.astype(int).tolist();index={pid:i for i,pid in enumerate(ids)}
    matrix=np.zeros((len(ids),len(ids)),float);failures={};mismatches={}
    tracking_path=folder/'receiving_tracking.csv'
    tracking=pd.read_csv(tracking_path) if tracking_path.exists() else pd.DataFrame()
    totals=(dict(zip(tracking.PLAYER_ID.astype(int),pd.to_numeric(tracking.PASSES_MADE,errors='coerce')))
        if {'PLAYER_ID','PASSES_MADE'}.issubset(tracking) else {})
    no_tracking=(str(game_record.get('zero_total_passes',False)).lower() in ['true','1']
        and str(game_record.get('passing_zero_filled',False)).lower() in ['true','1'])
    if no_tracking:
        return ids,matrix,{pid:'Game tracking unavailable; weight set to zero' for pid in ids},mismatches
    day=pd.Timestamp(game_record['game_date']).strftime('%m/%d/%Y')
    teams={t['abbreviation']:t for t in static_teams.get_teams()}
    def request_player(pid):
        def validate(payload):
            f=validate_passing(payload,pid,1610612744)
            if 'G' in f and not pd.to_numeric(f.G,errors='coerce').eq(1).all():
                raise ValueError('Passing profile is not single-game data')
            if f.PASS_TEAMMATE_PLAYER_ID.duplicated().any() or not f.PASS_TEAMMATE_PLAYER_ID.isin(ids).all():
                raise ValueError('Invalid or duplicate receivers in passing profile')
            return f
        return client.fetch(playerdashptpass.PlayerDashPtPass,
            dict(team_id=1610612744,player_id=pid,season='2021-22',season_type_all_star='Regular Season',
                 date_from_nullable=day,date_to_nullable=day,opponent_team_id=teams[game_record['opponent']]['id'],
                 per_mode_simple='Totals'),validate)[0]
    # Explicitly observed zero passing totals require no per-player download.
    requested=[pid for pid in ids if not (pid in totals and np.isfinite(totals[pid]) and totals[pid]==0)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(request_player,pid):pid for pid in requested}
        for future in as_completed(futures):
            pid=futures[future]
            try:
                f=future.result()
                if f.empty:failures[pid]='Empty passing profile; weights set to zero'
                for row in f.itertuples():matrix[index[pid],index[int(row.PASS_TEAMMATE_PLAYER_ID)]]=float(row.PASS)
                if pid in totals and np.isfinite(totals[pid]) and matrix[index[pid]].sum()!=totals[pid]:
                    mismatches[pid]=dict(matrix_passes=float(matrix[index[pid]].sum()),tracking_passes=float(totals[pid]))
            except DownloadsStopped:
                for pending in futures:pending.cancel()
                raise
            except DataUnavailable as exc:failures[pid]=str(exc)
    return ids,matrix,failures,mismatches


def plot_warriors_eigenvector_network(matrix,ids,scores,names,path,number,date):
    import matplotlib.pyplot as plt
    import networkx as nx
    from inspect import signature
    d=nx.from_numpy_array(matrix,create_using=nx.DiGraph)
    d=nx.relabel_nodes(d,dict(enumerate(ids)))
    layout=nx.Graph();layout.add_nodes_from(ids)
    for a,b,v in d.edges(data=True):
        previous=layout.get_edge_data(a,b,{}).get('weight',0.)
        layout.add_edge(a,b,weight=previous+v['weight'])
    fig,ax=plt.subplots(figsize=(12,10))
    if len(ids) and scores:
        options=dict(seed=42,weight='weight',iterations=300)
        if 'method' in signature(nx.spring_layout).parameters:options['method']='force'
        largest=max(v['weight'] for _,_,v in layout.edges(data=True))
        for _,_,v in layout.edges(data=True):v['weight']/=largest
        pos=nx.spring_layout(layout,**options)
        sizes=[300+3300*scores[pid]/max(scores.values()) for pid in ids]
        maximum=max((v['weight'] for _,_,v in d.edges(data=True)),default=1)
        nx.draw_networkx_nodes(d,pos,ax=ax,node_size=sizes,node_color='#FFD700',edgecolors='#696969')
        nx.draw_networkx_edges(d,pos,ax=ax,node_size=sizes,width=[.4+3*v['weight']/maximum for _,_,v in d.edges(data=True)],
            arrows=True,arrowsize=12,alpha=.35,edge_color='#4169E1',connectionstyle='arc3,rad=.1')
        labels={pid:names.get(pid,str(pid)) for pid in ids}
        nx.draw_networkx_labels(d,pos,labels=labels,ax=ax,font_size=8)
    else:ax.text(.5,.5,'Eigenvector balance unavailable for this game',ha='center',transform=ax.transAxes)
    ax.set_title(f'GSW 2021–22 | Game {number} | {date}');ax.axis('off')
    fig.text(.5,.025,'Node size: eigenvector centrality of combined passing/receiving network | Edges: directed pass counts\n'
        'Fruchterman–Reingold layout; no minutes cutoff; isolated players omitted.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.07,1,1));fig.savefig(path,dpi=160);plt.close(fig)


def fit_eigenvector_performance_model(sample,outcome,predictors,label):
    """OLS with Bartlett Newey-West HAC(7) covariance, in game-number order."""
    from scipy.stats import t,linregress
    d=sample.sort_values('game_number').replace([np.inf,-np.inf],np.nan).dropna(subset=[outcome,*predictors])
    x=d[predictors].to_numpy(float);y=d[outcome].to_numpy(float)
    X=np.column_stack([np.ones(len(d)),x]);n,k=X.shape
    if n<=k or np.linalg.matrix_rank(X)<k or np.ptp(y)<=1e-14:return []
    inv=np.linalg.inv(X.T@X);beta=inv@X.T@y;residual=y-X@beta
    scores=X*residual[:,None];meat=scores.T@scores
    # Match lags by actual game number; do not bridge excluded games as adjacent.
    positions={int(g):i for i,g in enumerate(d.game_number)}
    for lag in range(1,min(7,int(d.game_number.max()-d.game_number.min()))+1):
        cross=np.zeros((k,k))
        for number,i in positions.items():
            if number-lag in positions:cross+=np.outer(scores[i],scores[positions[number-lag]])
        meat+=(1-lag/8)*(cross+cross.T)
    covariance=n/(n-k)*inv@meat@inv
    se=np.sqrt(np.maximum(np.diag(covariance),0));pvalues=np.full(k,np.nan)
    positive=se>0;pvalues[positive]=2*t.sf(abs(beta[positive]/se[positive]),n-k)
    r2=1-float(residual@residual)/float(np.sum((y-y.mean())**2))
    pearson,p_ols=np.nan,np.nan
    if len(predictors)==1:
        simple=linregress(x[:,0],y);pearson=float(simple.rvalue);p_ols=float(simple.pvalue)
    return [dict(model=label,outcome=outcome,term=term,coefficient=float(beta[j]),
        standard_error_hac7=float(se[j]),p_value_hac7=float(pvalues[j]),n=n,r_squared=r2,
        pearson_r=pearson,p_value_ols=p_ols if j==1 else np.nan,
        condition_number=float(np.linalg.cond(X))) for j,term in enumerate(['intercept',*predictors])]


def classify_warriors_passing_styles(frame):
    """Relative, outcome-independent game labels from the Warriors EC median.

    This is a visualization split, not an absolute NBA star/balanced threshold.
    Invalid networks are unclassified. Ties at the median go to more-balanced.
    """
    d=frame.copy()
    valid=d.network_valid.astype(str).str.lower().eq('true')
    concentration=pd.to_numeric(d.eigenvector_concentration,errors='coerce')
    valid &= concentration.notna() & np.isfinite(concentration)
    if not valid.any():raise ValueError('No valid eigenvector concentrations for style classification')
    cutoff=float(concentration.loc[valid].median())
    d['passing_style']='unavailable'
    d.loc[valid & concentration.le(cutoff),'passing_style']='more_balanced'
    d.loc[valid & concentration.gt(cutoff),'passing_style']='more_star_centered'
    d['passing_style_cutoff']=cutoff
    d['star_centered_indicator']=np.where(valid,d.passing_style.eq('more_star_centered').astype(float),np.nan)
    mean_entropy=float(pd.to_numeric(d.loc[valid,'network_conditional_entropy'],errors='coerce').mean())
    d['entropy_centered']=pd.to_numeric(d.network_conditional_entropy,errors='coerce')-mean_entropy
    d['entropy_x_star_style']=d.entropy_centered*d.star_centered_indicator
    return d,cutoff


def plot_eigenvector_performance(frame,results,out):
    """Conditional receiver entropy is X; team performance is Y. Eigenvector scores ONLY set style."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter
    from scipy.stats import linregress
    d,cutoff=classify_warriors_passing_styles(frame)
    d=d.loc[~d.passing_style.eq('unavailable')]
    styles=[('more_balanced','More balanced','o','-'),('more_star_centered','More star-centered','^','--')]
    out=Path(out)
    for outcome,ylabel,stem in [('team_ast_to_ratio','Team assist-to-turnover ratio','conditional_entropy_team_ast_to_by_passing_style'),
            ('team_efg_pct','Team effective field-goal percentage','conditional_entropy_team_efg_by_passing_style')]:
        fig,ax=plt.subplots(figsize=(12,8))
        notes=[];style_legend=[]
        for key,label,marker,line_style in styles:
            group=d.loc[d.passing_style.eq(key)].dropna(subset=['network_conditional_entropy',outcome])
            for result,color in [('W','#168348'),('L','#d23b3b')]:
                part=group.loc[group.result.eq(result)]
                ax.scatter(part.network_conditional_entropy,part[outcome],marker=marker,color=color,
                    s=65 if marker=='o' else 85,alpha=.8,edgecolors='white',linewidths=.6,zorder=3)
            discrepancy=group.loc[~group.full_observed_network.astype(str).str.lower().eq('true')]
            if len(discrepancy):
                ax.scatter(discrepancy.network_conditional_entropy,discrepancy[outcome],marker=marker,
                    facecolors='none',edgecolors='#333333',s=105 if marker=='o' else 135,linewidths=1,zorder=4)
            if len(group)>=3 and group.network_conditional_entropy.nunique()>1 and group[outcome].nunique()>1:
                fit=linregress(group.network_conditional_entropy,group[outcome])
                xx=np.linspace(group.network_conditional_entropy.min(),group.network_conditional_entropy.max(),100)
                ax.plot(xx,fit.intercept+fit.slope*xx,color='#333333',ls=line_style,lw=2,zorder=2)
                notes.append(f'{label}: N={len(group)}, r={fit.rvalue:.3f}, R²={fit.rvalue**2:.3f}')
            else:notes.append(f'{label}: N={len(group)}, insufficient variation for fit')
            style_legend.append(Line2D([0],[0],color='#333333',ls=line_style,marker=marker,
                markerfacecolor='white',markersize=8,lw=2,label=label))
        outcome_legend=[Line2D([0],[0],color='none',marker='o',markerfacecolor='#168348',markeredgecolor='none',markersize=8,label='Win'),
            Line2D([0],[0],color='none',marker='o',markerfacecolor='#d23b3b',markeredgecolor='none',markersize=8,label='Loss')]
        fig.legend(handles=style_legend+outcome_legend,loc='upper center',bbox_to_anchor=(.5,.94),ncol=4,frameon=False)
        fig.text(.5,.865,'\n'.join(notes),ha='center',va='top',fontsize=10,
            bbox=dict(facecolor='white',edgecolor='#dddddd',alpha=.92))
        ax.set_xlabel('Normalized conditional receiver entropy H(R | P)',labelpad=10)
        ax.set_ylabel(ylabel,labelpad=10)
        if outcome=='team_efg_pct':ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.grid(alpha=.16);ax.spines[['top','right']].set_visible(False)
        fig.suptitle('Warriors 2021–22: conditional entropy and '+('team eFG%' if outcome=='team_efg_pct' else 'team assist-to-turnover ratio'),fontsize=16,y=.985)
        fig.text(.5,.022,f'Marker shape and line style indicate passing structure: eigenvector concentration ≤ {cutoff:.5f} = more balanced; above = more star-centered.\n'
            'Conditional entropy: receiver uncertainty given the passer, normalized by ln(N−1). No minutes cutoff. Dark outlines mark profile discrepancies.',
            ha='center',fontsize=8.5)
        fig.tight_layout(rect=(0,.075,1,.81))
        fig.savefig(out/(stem+'.png'),dpi=180);fig.savefig(out/(stem+'.pdf'))
        legacy_stem='entropy_team_efg_by_passing_style' if outcome=='team_efg_pct' else 'entropy_team_ast_to_by_passing_style'
        fig.savefig(out/(legacy_stem+'.png'),dpi=180);fig.savefig(out/(legacy_stem+'.pdf'))
        if outcome=='team_ast_to_ratio':
            # Preserve the existing user-facing link with the corrected axes.
            fig.savefig(out/'eigenvector_team_performance.png',dpi=180)
            fig.savefig(out/'eigenvector_team_performance.pdf')
        plt.close(fig)
    atomic_json(dict(method='Warriors-season median eigenvector concentration; outcome-independent relative split',
        cutoff=cutoff,more_balanced_games=int(d.passing_style.eq('more_balanced').sum()),
        more_star_centered_games=int(d.passing_style.eq('more_star_centered').sum()),
        x_axis='Normalized conditional receiver entropy H(R | P)',primary_y_axis='Official team assist-to-turnover ratio',
        eigenvector_role='Passing-style classification only; not a regression axis or continuous performance predictor',
        limitation='Labels apply to game-level structures of one team, not a comparison of different franchises'),
        out/'passing_style_classification.json')



def run_warriors_eigenvector_performance(argv):
    parser=argparse.ArgumentParser(description='Warriors conditional entropy vs team AST/TO, grouped by eigenvector passing style; no minutes cutoff')
    parser.add_argument('--warriors-eigenvector-performance',action='store_true')
    parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--plot-only',action='store_true')
    parser.add_argument('--refresh',action='store_true')
    parser.add_argument('--workers',type=int,choices=[1,2],default=2)
    parser.add_argument('--timeout',type=float,default=25)
    parser.add_argument('--retries',type=int,default=2)
    parser.add_argument('--min-delay',type=float,default=.5)
    parser.add_argument('--max-delay',type=float,default=1.)
    args=parser.parse_args(argv)
    game_root=args.output_dir.expanduser()/'GSW_2021-22_82_games'
    out=game_root/'eigenvector_team_performance';out.mkdir(parents=True,exist_ok=True)
    games=pd.read_csv(game_root/'game_metrics.csv',dtype={'game_id':str})
    schedule=pd.read_csv(game_root/'schedule.csv',dtype={'GAME_ID':str})
    schedule['GAME_ID']=schedule.GAME_ID.str.zfill(10)
    stats=schedule.set_index('GAME_ID')
    if len(games)!=82 or games.game_id.str.zfill(10).nunique()!=82 or stats.index.duplicated().any():
        raise ValueError('Expected 82 unique Warriors games and a unique matching schedule')
    path=out/'team_game_metrics.csv';rows=[]
    if path.exists() and not args.refresh:
        rows=pd.read_csv(path,dtype={'game_id':str}).to_dict(orient='records')
    done={int(r['game_number']) for r in rows}
    client=None
    if not args.plot_only:
        client=NBAClient(Config(season='2021-22',season_type='Regular Season',cache_dir=game_root/'cache',
            timeout=args.timeout,retries=args.retries,min_delay=args.min_delay,max_delay=args.max_delay,refresh=args.refresh))
    try:
        for record in games.sort_values('game_number').to_dict(orient='records'):
            number=int(record['game_number']);folder=out/f'game_{number:02d}'
            if number in done:continue
            if args.plot_only:continue
            folder.mkdir(parents=True,exist_ok=True)
            print(f'Eigenvector network: GSW game {number}/82 ({record["game_date"]})',flush=True)
            ids,matrix,failures,mismatches=download_warriors_passing_matrix(client,game_root,record,args.workers)
            atomic_csv(pd.DataFrame(matrix,index=ids,columns=ids),folder/'passing_matrix.csv',index=True)
            ec,scores,selected,active=eigenvector_passing_concentration(matrix,ids)
            roster=pd.read_csv(game_root/f'game_{number:02d}'/'game_players.csv')
            names=dict(zip(roster.PLAYER_ID.astype(int),roster.PLAYER_NAME.astype(str)))
            atomic_csv(pd.DataFrame([dict(player_id=pid,name=names.get(pid,str(pid)),eigenvector_share=scores.get(pid,np.nan)) for pid in selected],columns=['player_id','name','eigenvector_share']),
                folder/'player_eigenvector_centrality.csv')
            plot_warriors_eigenvector_network(active,selected,scores,names,folder/'passing_network_eigenvector.png',number,record['game_date'])
            official=stats.loc[str(record['game_id']).zfill(10)]
            for col in ['FGM','FGA','FG3M','AST','TOV']:
                if not np.isfinite(official[col]) or official[col]<0:raise ValueError('Invalid official team box-score totals')
            if official.FGM>official.FGA or official.FG3M>official.FGM:raise ValueError('Invalid team shooting totals')
            h,conditional=compute_team_entropies(active) if active.sum()>0 else (np.nan,np.nan)
            row=dict(game_number=number,game_id=str(record['game_id']).zfill(10),game_date=record['game_date'],
                result=record['result'],opponent=record['opponent'],point_margin=record['point_margin'],
                home_game=int('vs.' in str(official.MATCHUP)),roster_player_count=len(ids),**ec,
                network_receiver_entropy=h,network_conditional_entropy=conditional,
                saved_receiver_entropy=record['receiver_entropy'],minimum_player_minutes=0,
                team_efg_pct=float((official.FGM+.5*official.FG3M)/official.FGA) if official.FGA>0 else np.nan,
                team_ast_to_ratio=float(official.AST/official.TOV) if official.TOV>0 else np.nan,
                team_fgm=float(official.FGM),team_fga=float(official.FGA),team_fg3m=float(official.FG3M),
                team_assists=float(official.AST),team_turnovers=float(official.TOV),
                top_eigenvector_player_name=names.get(ec['top_eigenvector_player_id'],''),
                zero_filled_profile_count=len(failures),profile_total_mismatch_count=len(mismatches),
                full_observed_network=not failures and not mismatches)
            rows.append(row);atomic_csv(pd.DataFrame(rows).sort_values('game_number'),path)
            atomic_json({'zero_filled_profiles':{str(k):v for k,v in failures.items()},
                'profile_total_mismatches':{str(k):v for k,v in mismatches.items()},'missing_policy':'zero',
                'network_error':ec['network_error']},folder/'network_coverage_audit.json')
            print(f'  valid={row["network_valid"]}; EC concentration={row["eigenvector_concentration"]:.5f}; team eFG%={row["team_efg_pct"]:.3f}',flush=True)
    finally:
        if client is not None:client.close()
    frame=pd.DataFrame(rows).sort_values('game_number')
    if frame.empty:raise ValueError('No saved network metrics; run normally before --plot-only')
    if frame.duplicated('game_number').any():raise ValueError('Duplicate network observations')
    frame,style_cutoff=classify_warriors_passing_styles(frame)
    frame['regression_entropy_measure']='conditional receiver entropy / ln(N-1)'
    atomic_csv(frame,path)
    eligible=frame.loc[~frame.passing_style.eq('unavailable')].copy()
    model_rows=[]
    for outcome in ['team_efg_pct','team_ast_to_ratio']:
        model_rows.extend(fit_eigenvector_performance_model(eligible,outcome,['network_conditional_entropy'],'pooled_conditional_entropy'))
        for style in ['more_balanced','more_star_centered']:
            group=eligible.loc[eligible.passing_style.eq(style)]
            model_rows.extend(fit_eigenvector_performance_model(group,outcome,['network_conditional_entropy'],'conditional_entropy_'+style))
        complete=eligible.loc[eligible.full_observed_network.astype(str).str.lower().eq('true')]
        model_rows.extend(fit_eigenvector_performance_model(complete,outcome,['network_conditional_entropy'],'complete_profiles_conditional_entropy'))
        model_rows.extend(fit_eigenvector_performance_model(eligible,outcome,
            ['entropy_centered','star_centered_indicator','entropy_x_star_style','network_player_count','home_game','game_number'],
            'conditional_entropy_by_style_adjusted_interaction'))
    results=pd.DataFrame(model_rows);atomic_csv(results,out/'regression_results.csv')
    plot_eigenvector_performance(frame,results,out)
    atomic_json(dict(processed_games=len(frame),valid_networks=len(eligible),
        invalid_games=frame.loc[~frame.network_valid.astype(str).str.lower().eq('true'),'game_number'].astype(int).tolist(),
        matrix='Actual PlayerDashPtPass per-game pair counts; not inferred from receiving totals',
        eigenvector_network='Undirected combined involvement: A_ij + A_ji; positive-degree players only',
        concentration='Custom normalized HHI of L1 eigenvector shares: (sum(p_i^2)-1/N)/(1-1/N); not Freeman centralization',
        meaning='Higher = more unequal passing-hub importance; not necessarily a scoring superstar',
        eigenvector_role='Style classification only; not a continuous performance predictor or regression axis',
        style_cutoff=style_cutoff,style_classification='Within-season median split of valid Warriors games',
        primary_regression='Team assist-to-turnover ratio as outcome, normalized conditional receiver entropy as predictor; eigenvector-based styles distinguish groups',
        team_performance='Official complete-team LeagueGameFinder totals; eFG=(FGM+0.5*FG3M)/FGA; AST/TO=AST/TOV',
        no_minutes_cutoff=True,missing_weights_policy='zero per user; no-edge or disconnected active graphs excluded from EC regressions',
        inference='Exploratory conditional-entropy/performance OLS by passing style; HAC(7) with actual game-number lags; interaction model controls player count, home court and game number',
        imputation_sensitivity='Complete-profile regressions exclude zero-filled or tracking-total-mismatch networks'),out/'analysis_audit.json')
    print(f'Finished {len(frame)}/82 games; {len(eligible)} valid networks. Outputs: {out}',flush=True)
    return int(len(frame)!=82)




# ---------------------------------------------------------------------
# CONDITIONAL PASSING ENTROPY VS OPEN THREE-POINT SHOT CREATION
# python teamgraphs.py --warriors-open-shots
# Uses the SAME fixed Warriors eigenvector-style labels, not shooting outcomes.
# Open = 4-6 Feet - Open plus 6+ Feet - Wide Open, by closest defender.
# ---------------------------------------------------------------------
OPEN_SHOT_BUCKETS=[('very_tight','0-2 Feet - Very Tight'),('tight','2-4 Feet - Tight'),
    ('open','4-6 Feet - Open'),('wide_open','6+ Feet - Wide Open')]


def validate_warriors_shot_split(payload):
    f=extract_team_metrics(payload,'LeagueDashPTShots',{'TEAM_ID','GP','G','FGA','FG3A','FG3M'})
    for col in ['TEAM_ID','GP','G','FGA','FG3A','FG3M']:
        numeric_column(f,col,integer=True,positive=col=='TEAM_ID')
    if not f.TEAM_ID.eq(1610612744).all() or f.TEAM_ID.duplicated().any():
        raise ValueError('Wrong or duplicate team in shot split')
    if len(f) and (not f.GP.eq(1).all() or not f.G.eq(1).all()):
        raise ValueError('Shot split is not single-game data; date filter may not be respected')
    if (f.FG3M>f.FG3A).any() or (f.FG3A>f.FGA).any():
        raise ValueError('Impossible three-point attempt/make counts')
    return f


def summarize_open_shot_splits(splits,official_three_attempts):
    """Empty categories are confirmed zero only when bucket totals exhaust ALL.

    Missing totals never create a fabricated zero open-shot percentage. The
    denominator is tracked 3PA, and coverage versus official 3PA is reported.
    """
    total=splits['all'];tracked=int(total['fg3a'])
    counts={key:int(splits[key]['fg3a']) for key,_ in OPEN_SHOT_BUCKETS}
    bucket_sum=sum(counts.values());numerator=counts['open']+counts['wide_open']
    unavailable=[key for key,_ in OPEN_SHOT_BUCKETS if not splits[key]['row_present']]
    errors=[]
    if not total['row_present'] or tracked<=0:errors.append('No positive tracked three-point total')
    if bucket_sum!=tracked:errors.append('Defender-distance buckets do not sum to tracked three-point attempts')
    if tracked>official_three_attempts:errors.append('Tracked three-point total exceeds official attempts')
    if numerator>tracked:errors.append('Open attempts exceed tracked attempts')
    valid=not errors
    inferred_zero=unavailable if valid else []
    return dict(shot_split_valid=valid,tracked_three_attempts=tracked,
        official_three_attempts=int(official_three_attempts),classified_three_attempts=bucket_sum,
        very_tight_three_attempts=counts['very_tight'],tight_three_attempts=counts['tight'],
        open_three_attempts=counts['open'],wide_open_three_attempts=counts['wide_open'],
        open_or_wide_three_attempts=numerator,
        open_three_share=float(numerator/tracked) if valid else np.nan,
        wide_open_three_share=float(counts['wide_open']/tracked) if valid else np.nan,
        tracked_three_coverage=float(tracked/official_three_attempts) if official_three_attempts>0 else np.nan,
        full_official_shot_coverage=bool(valid and tracked==official_three_attempts),
        zero_bucket_inferred_count=len(inferred_zero),zero_bucket_inferred_keys=';'.join(inferred_zero),
        shot_data_error='; '.join(errors))


def download_warriors_open_shot_game(client,record,folder,official_three_attempts,workers=2):
    from concurrent.futures import ThreadPoolExecutor,as_completed
    from nba_api.stats.endpoints import leaguedashteamptshot
    teams={t['abbreviation']:t for t in static_teams.get_teams()}
    day=pd.Timestamp(record['game_date']).strftime('%m/%d/%Y')
    common=dict(season='2021-22',season_type_all_star='Regular Season',per_mode_simple='Totals',
        team_id_nullable=1610612744,opponent_team_id_nullable=teams[record['opponent']]['id'],
        date_from_nullable=day,date_to_nullable=day)
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    specifications=[('all',''),*OPEN_SHOT_BUCKETS];splits={}
    def fetch_split(key,bucket):
        frame,_=client.fetch(leaguedashteamptshot.LeagueDashTeamPtShot,
            dict(close_def_dist_range_nullable=bucket,**common),validate_warriors_shot_split)
        atomic_csv(frame,folder/(key+'_shots.csv'))
        if frame.empty:return dict(fg3a=0,fg3m=0,fga=0,row_present=False,request_error='Empty category response')
        row=frame.iloc[0]
        return dict(fg3a=int(row.FG3A),fg3m=int(row.FG3M),fga=int(row.FGA),row_present=True,request_error='')
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(fetch_split,key,bucket):key for key,bucket in specifications}
        for future in as_completed(futures):
            key=futures[future]
            try:splits[key]=future.result()
            except DownloadsStopped:
                for pending in futures:pending.cancel()
                raise
            except DataUnavailable as exc:
                splits[key]=dict(fg3a=0,fg3m=0,fga=0,row_present=False,request_error=str(exc))
    result=summarize_open_shot_splits(splits,int(official_three_attempts))
    result['shot_request_errors']='; '.join(key+': '+s['request_error'] for key,s in splits.items() if s['request_error'])
    json_checks={key:(None if isinstance(value,(float,np.floating)) and not np.isfinite(value) else value) for key,value in result.items()}
    atomic_json({'date':record['game_date'],'splits':splits,'checks':json_checks,
        'measure':'FG3A in 4-6 Feet - Open and 6+ Feet - Wide Open / all tracked FG3A',
        'empty_bucket_rule':'Zero only if other bucket counts exhaust an independently observed tracked total'},folder/'shot_coverage_audit.json')
    return result


def plot_warriors_open_shot_creation(frame,out):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter
    from scipy.stats import linregress
    valid=frame.comparison_eligible.astype(str).str.lower().eq('true')
    d=frame.loc[valid].copy();fig,ax=plt.subplots(figsize=(12,8));notes=[];legend=[]
    for style,label,marker,line_style in [('more_balanced','More balanced','o','-'),
            ('more_star_centered','More star-centered','^','--')]:
        group=d.loc[d.passing_style.eq(style)]
        for result,color in [('W','#168348'),('L','#d23b3b')]:
            p=group.loc[group.result.eq(result)]
            ax.scatter(p.network_conditional_entropy,p.open_three_share,marker=marker,color=color,
                s=65 if marker=='o' else 85,alpha=.8,edgecolor='white',linewidth=.6,zorder=3)
        flagged=group.loc[~group.full_observed_network.astype(str).str.lower().eq('true')
            | ~group.full_official_shot_coverage.astype(str).str.lower().eq('true')
            | group.zero_bucket_inferred_count.gt(0)]
        if len(flagged):ax.scatter(flagged.network_conditional_entropy,flagged.open_three_share,
            marker=marker,facecolors='none',edgecolors='#333333',s=105 if marker=='o' else 135,linewidth=1,zorder=4)
        if len(group)>=3 and group.network_conditional_entropy.nunique()>1 and group.open_three_share.nunique()>1:
            fit=linregress(group.network_conditional_entropy,group.open_three_share)
            xx=np.linspace(group.network_conditional_entropy.min(),group.network_conditional_entropy.max(),100)
            ax.plot(xx,fit.intercept+fit.slope*xx,color='#333333',ls=line_style,lw=2,zorder=2)
            notes.append(f'{label}: N={len(group)}, r={fit.rvalue:.3f}, R²={fit.rvalue**2:.3f}')
        else:notes.append(f'{label}: N={len(group)}, insufficient data/variation')
        legend.append(Line2D([0],[0],color='#333333',ls=line_style,marker=marker,markerfacecolor='white',markersize=8,label=label))
    legend.extend([Line2D([0],[0],color='none',marker='o',markerfacecolor='#168348',markeredgecolor='none',markersize=8,label='Win'),
        Line2D([0],[0],color='none',marker='o',markerfacecolor='#d23b3b',markeredgecolor='none',markersize=8,label='Loss')])
    fig.suptitle('Warriors 2021–22: conditional passing entropy and open three-point creation',fontsize=15,y=.985)
    fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(.5,.94),ncol=4,frameon=False)
    fig.text(.5,.865,'\n'.join(notes),ha='center',va='top',fontsize=10,
        bbox=dict(facecolor='white',edgecolor='#dddddd',alpha=.92))
    ax.set_xlabel('Normalized conditional receiver entropy H(R | P)',labelpad=10)
    ax.set_ylabel('Open or wide-open 3PA / all tracked 3PA',labelpad=10)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0));ax.grid(alpha=.16);ax.spines[['top','right']].set_visible(False)
    fig.text(.5,.022,'Open = closest defender 4–6 feet; wide open = 6+ feet. Attempts are counted regardless of whether the shot was made.\n'
        'Passing styles use the fixed Warriors-season eigenvector split. No minutes cutoff. Dark outlines flag coverage/profile issues or inferred empty buckets.',
        ha='center',fontsize=8.5)
    fig.tight_layout(rect=(0,.075,1,.81));fig.savefig(Path(out)/'conditional_entropy_open_three_share.png',dpi=180)
    fig.savefig(Path(out)/'conditional_entropy_open_three_share.pdf');plt.close(fig)


def run_warriors_open_shots(argv):
    parser=argparse.ArgumentParser(description='Warriors conditional passing entropy vs open-three attempt share, grouped by eigenvector style')
    parser.add_argument('--warriors-open-shots',action='store_true')
    parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--plot-only',action='store_true')
    parser.add_argument('--refresh',action='store_true')
    parser.add_argument('--workers',type=int,choices=[1,2],default=2)
    parser.add_argument('--timeout',type=float,default=25)
    parser.add_argument('--retries',type=int,default=2)
    parser.add_argument('--min-delay',type=float,default=.5)
    parser.add_argument('--max-delay',type=float,default=1.)
    args=parser.parse_args(argv)
    game_root=args.output_dir.expanduser()/'GSW_2021-22_82_games'
    out=game_root/'open_shot_creation';out.mkdir(parents=True,exist_ok=True)
    networks=pd.read_csv(game_root/'eigenvector_team_performance/team_game_metrics.csv',dtype={'game_id':str})
    networks,cutoff=classify_warriors_passing_styles(networks)
    if len(networks)!=82 or networks.game_number.nunique()!=82:raise ValueError('Expected all 82 saved Warriors network records')
    schedule=pd.read_csv(game_root/'schedule.csv',dtype={'GAME_ID':str});schedule['GAME_ID']=schedule.GAME_ID.str.zfill(10)
    if schedule.GAME_ID.duplicated().any():raise ValueError('Duplicate schedule game IDs')
    official=schedule.set_index('GAME_ID')
    path=out/'open_shot_game_metrics.csv';rows=[]
    if path.exists() and not args.refresh:rows=pd.read_csv(path,dtype={'game_id':str}).to_dict(orient='records')
    done={int(r['game_number']) for r in rows};client=None
    if not args.plot_only:
        client=NBAClient(Config(season='2021-22',season_type='Regular Season',cache_dir=game_root/'cache',
            timeout=args.timeout,retries=args.retries,min_delay=args.min_delay,max_delay=args.max_delay,refresh=args.refresh))
    try:
        for record in networks.sort_values('game_number').to_dict(orient='records'):
            number=int(record['game_number'])
            if number in done or args.plot_only:continue
            print(f'Closest-defender shots: GSW game {number}/82 ({record["game_date"]})',flush=True)
            attempts=int(official.loc[str(record['game_id']).zfill(10),'FG3A'])
            shots=download_warriors_open_shot_game(client,record,out/f'game_{number:02d}',attempts,args.workers)
            row=dict(record,**shots)
            rows.append(row);atomic_csv(pd.DataFrame(rows).sort_values('game_number'),path)
            print(f'  valid splits={shots["shot_split_valid"]}; open 3PA={shots["open_or_wide_three_attempts"]}/{shots["tracked_three_attempts"]}; share={shots["open_three_share"]:.3f}',flush=True)
    finally:
        if client is not None:client.close()
    if not rows:raise ValueError('No saved shot data; run without --plot-only first')
    frame=pd.DataFrame(rows).sort_values('game_number')
    if frame.game_number.duplicated().any():raise ValueError('Duplicate shot-game records')
    # Preserve the existing classification even if missing shot data reduce the sample.
    styles=networks.set_index('game_number')
    for col in ['passing_style','passing_style_cutoff','star_centered_indicator','entropy_centered','entropy_x_star_style']:
        frame[col]=frame.game_number.map(styles[col])
    frame['comparison_eligible']=(frame.network_valid.astype(str).str.lower().eq('true')
        & frame.shot_split_valid.astype(str).str.lower().eq('true')
        & pd.to_numeric(frame.network_conditional_entropy,errors='coerce').notna()
        & pd.to_numeric(frame.open_three_share,errors='coerce').between(0,1))
    atomic_csv(frame,path)
    sample=frame.loc[frame.comparison_eligible].copy();model_rows=[]
    for outcome in ['open_three_share','wide_open_three_share']:
        model_rows.extend(fit_eigenvector_performance_model(sample,outcome,['network_conditional_entropy'],'pooled_conditional_entropy'))
        for style in ['more_balanced','more_star_centered']:
            model_rows.extend(fit_eigenvector_performance_model(sample.loc[sample.passing_style.eq(style)],outcome,
                ['network_conditional_entropy'],'conditional_entropy_'+style))
        complete=sample.loc[sample.full_observed_network.astype(str).str.lower().eq('true')
            & sample.full_official_shot_coverage.astype(str).str.lower().eq('true')
            & sample.zero_bucket_inferred_count.eq(0)]
        model_rows.extend(fit_eigenvector_performance_model(complete,outcome,['network_conditional_entropy'],'complete_coverage_conditional_entropy'))
        model_rows.extend(fit_eigenvector_performance_model(sample,outcome,
            ['entropy_centered','star_centered_indicator','entropy_x_star_style','network_player_count','home_game','game_number'],
            'conditional_entropy_by_style_adjusted_interaction'))
    results=pd.DataFrame(model_rows);atomic_csv(results,out/'regression_results.csv')
    plot_warriors_open_shot_creation(frame,out)
    if len(sample):
        summary=sample.groupby('passing_style')[['open_three_share','wide_open_three_share','tracked_three_attempts','open_or_wide_three_attempts','tracked_three_coverage']].agg(['mean','median','count'])
        summary.columns=['_'.join(x) for x in summary.columns];atomic_csv(summary.reset_index(),out/'passing_style_summary.csv')
    atomic_json(dict(processed_games=len(frame),eligible_comparisons=len(sample),
        excluded_game_numbers=frame.loc[~frame.comparison_eligible,'game_number'].astype(int).tolist(),
        outcome='FG3A(open 4-6 ft + wide open 6+ ft) / all tracked FG3A',
        predictor='Conditional receiver entropy H(R|P)/ln(N-1), computed from directed pairwise passes',
        style_cutoff=cutoff,style_rule='Same predetermined median eigenvector concentration labels as the existing Warriors analysis; not recalculated from shooting outcomes',
        validation='Single-game GP=G=1; four distance buckets sum to tracked FG3A; tracked FG3A cannot exceed official FG3A',
        empty_buckets='Infer zero only when counts exhaust independently observed all-tracked 3PA; unavailable totals do not become 0% open-shot share',
        coverage='Main outcome refers to tracked attempts; official-attempt coverage reported and checked in sensitivity models',
        no_minutes_cutoff=True,inference='Exploratory OLS; HAC(7) with actual game-number lags; interaction model controls player count, home court and game number',
        limitations='Closest-defender distance is a proxy for defensive pressure; one team-season and relative passing-style labels do not establish causality'),out/'analysis_audit.json')
    print(f'Finished {len(frame)}/82 games; {len(sample)} eligible entropy/open-shot comparisons. Outputs: {out}',flush=True)
    return int(len(frame)!=82)




# ---------------------------------------------------------------------
# FOUR CROSS-TEAM REGRESSIONS: 2025-26 REGULAR-SEASON TEAM TOTALS
# python teamgraphs.py --team-season-topics
# ---------------------------------------------------------------------
def season_topics_frame(payload, name, required):
    return extract_team_metrics(payload, name, set(required))


def season_topics_daily_passing(payload, expected_teams, actual_player_teams=None):
    f=season_topics_frame(payload,'LeagueDashPtStats',
        ['PLAYER_ID','TEAM_ID','GP','PASSES_MADE','PASSES_RECEIVED'])
    for col in ['PLAYER_ID','TEAM_ID','GP','PASSES_MADE','PASSES_RECEIVED']:
        numeric_column(f,col,integer=True,positive=col in ['PLAYER_ID','TEAM_ID'])
    if f.duplicated(['TEAM_ID','PLAYER_ID']).any():raise ValueError('Duplicate daily player-team passing rows')
    if not f.empty and not f.GP.eq(1).all():raise ValueError('Date-filtered passing includes aggregated games')
    if actual_player_teams is not None:
        # NBA tracking may attach the player's later team even for an earlier date.
        # Official game participation determines the team that day; never use current team labels.
        unknown=~f.PLAYER_ID.isin(actual_player_teams)
        if (f.loc[unknown,['PASSES_MADE','PASSES_RECEIVED']].sum(axis=1)>0).any():
            raise ValueError('Positive passing counts for a player absent from official games that date')
        f=f.loc[~unknown].copy();f['REPORTED_TEAM_ID']=f.TEAM_ID
        f['TEAM_ID']=f.PLAYER_ID.map(actual_player_teams).astype(int)
    if not set(f.TEAM_ID).issubset(expected_teams):raise ValueError('Unexpected teams after historical roster alignment')
    return f


def season_topics_network(client, team, folder, workers, roster):
    """Full season participants, no minutes cutoff; unavailable outgoing rows=0."""
    from concurrent.futures import ThreadPoolExecutor,as_completed
    path=folder/'network_metrics.json'
    if path.exists() and not client.config.refresh:
        saved=json.loads(path.read_text())
        if saved.get('participant_ids')==sorted(roster.PLAYER_ID.astype(int)):return saved
    folder.mkdir(parents=True,exist_ok=True)
    if roster.empty:raise DataUnavailable('No season roster for '+team['abbreviation'])
    atomic_csv(roster,folder/'roster_stats.csv')
    ids=sorted(roster.PLAYER_ID.astype(int));index={p:i for i,p in enumerate(ids)}
    matrix=np.zeros((len(ids),len(ids)));missing={};outside=0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(client.passing,p,team['id']):p for p in ids}
        for future in as_completed(futures):
            pid=futures[future]
            try:
                profile,_=future.result()
                if profile.empty:missing[str(pid)]='Empty outgoing profile; weights zero'
                for r in profile.to_dict('records'):
                    receiver=int(r['PASS_TEAMMATE_PLAYER_ID']);count=int(r['PASS'])
                    if receiver in index:matrix[index[pid],index[receiver]]+=count
                    else:outside+=count
            except DownloadsStopped:raise
            except DataUnavailable as exc:missing[str(pid)]=str(exc)
    ec,scores,active_ids,active=eigenvector_passing_concentration(matrix,ids)
    receiver,conditional=compute_team_entropies(active)
    atomic_csv(pd.DataFrame(matrix,index=ids,columns=ids).rename_axis('passer_id'),folder/'passing_matrix.csv',index=True)
    names=roster.set_index('PLAYER_ID').PLAYER_NAME.to_dict()
    atomic_csv(pd.DataFrame([dict(player_id=p,name=names.get(p,''),eigenvector_share=scores.get(p,0),
        outgoing_profile_available=str(p) not in missing) for p in ids]),folder/'player_centrality.csv')
    result=dict(team=team['abbreviation'],team_id=team['id'],season='2025-26',
        conditional_receiver_entropy=conditional,receiver_entropy=receiver,
        **ec,roster_player_count=len(ids),missing_profile_count=len(missing),full_network_profiles=not missing,
        recorded_passes=float(matrix.sum()),outside_roster_passes=outside,
        missing_profiles=missing,minimum_minutes=0,participant_ids=ids,roster_source='Official player-game participants including traded players')
    safe={k:(None if isinstance(v,(float,np.floating)) and not np.isfinite(v) else v) for k,v in result.items()}
    atomic_json(safe,path)
    return safe


def season_topics_stability(games,players,daily_dir,out):
    """Average consecutive-game receiving-share JSD; never bridge missing games.

    Align player IDs on the union. Missing tracking is saved as zero per user,
    but pairs involving unavailable values are excluded from primary JSD.
    Full-roster JSD includes lineup changes. Returning-player JSD is sensitivity.
    """
    rows=[];pairs=[];cache={}
    for team_id,g in games.groupby('TEAM_ID'):
        g=g.sort_values(['GAME_DATE','GAME_ID']).reset_index(drop=True)
        previous=None
        for i,game in g.iterrows():
            date=str(game.GAME_DATE)[:10]
            if date not in cache:
                path=daily_dir/(date+'.csv')
                cache[date]=pd.read_csv(path) if path.exists() else pd.DataFrame(columns=['TEAM_ID','PLAYER_ID','PASSES_RECEIVED'])
            tracking=cache[date];tracking=tracking.loc[tracking.TEAM_ID.eq(team_id)].copy()
            roster=players.loc[players.GAME_ID.astype(str).str.zfill(10).eq(str(game.GAME_ID).zfill(10)) & players.TEAM_ID.eq(team_id)]
            appeared=set(roster.PLAYER_ID.astype(int));observed=set(tracking.PLAYER_ID.astype(int))
            # Official player participation also captures short appearances rounded to 0 minutes.
            ids=sorted(appeared|observed)
            counts=tracking.set_index('PLAYER_ID').PASSES_RECEIVED.reindex(ids,fill_value=0).astype(float)
            missing=appeared-observed
            coverage_ok=bool(appeared and not missing and not observed-appeared and counts.sum()>0)
            current=dict(team_id=int(team_id),team=str(game.TEAM_ABBREVIATION),game_number=i+1,
                game_id=str(game.GAME_ID).zfill(10),game_date=date,
                missing_player_count=len(missing),missing_player_ids=';'.join(map(str,sorted(missing))),
                full_receiving_coverage=coverage_ok,tracked_passes_received=float(counts.sum()))
            rows.append(current)
            if previous is not None:
                prev,prev_counts=previous
                union=sorted(set(prev_counts.index)|set(counts.index));common=sorted(set(prev_counts.index)&set(counts.index))
                pair=dict(team=current['team'],team_id=int(team_id),game_number=i+1,
                    previous_game_id=prev['game_id'],game_id=current['game_id'],
                    comparison_available=coverage_ok and prev['full_receiving_coverage'],
                    js_divergence=np.nan,returning_player_js_divergence=np.nan,
                    player_turnover_fraction=1-len(common)/len(union) if union else np.nan)
                if pair['comparison_available']:
                    pair['js_divergence']=receiving_share_js_divergence(prev_counts.reindex(union,fill_value=0),counts.reindex(union,fill_value=0))
                    if len(common)>=2 and prev_counts.reindex(common).sum()>0 and counts.reindex(common).sum()>0:
                        pair['returning_player_js_divergence']=receiving_share_js_divergence(prev_counts.reindex(common),counts.reindex(common))
                pairs.append(pair)
            previous=(current,counts)
    daily=pd.DataFrame(rows);p=pd.DataFrame(pairs)
    atomic_csv(daily,out/'game_receiving_coverage.csv');atomic_csv(p,out/'consecutive_game_stability.csv')
    summary=[]
    for team,g in p.groupby('team'):
        v=g.loc[g.comparison_available]
        summary.append(dict(team=team,mean_receiving_jsd=v.js_divergence.mean(),
            mean_returning_player_jsd=v.returning_player_js_divergence.mean(),
            mean_lineup_change=v.player_turnover_fraction.mean(),valid_stability_pairs=len(v),
            possible_stability_pairs=len(g),stability_pair_coverage=len(v)/len(g),
            stability_eligible=bool(len(v)>=np.ceil(.75*len(g)))))
    return pd.DataFrame(summary)


def season_topics_fit(frame,x,y,model,topic,predictors=None):
    """Independent team-season OLS, ordinary Pearson p and HC3 robust inference."""
    from scipy.stats import linregress,t
    predictors=predictors or [x]
    d=frame.replace([np.inf,-np.inf],np.nan).dropna(subset=[y,*predictors])
    X=np.column_stack([np.ones(len(d)),d[predictors].to_numpy(float)]);v=d[y].to_numpy(float)
    n,k=X.shape
    if n<=k or np.linalg.matrix_rank(X)<k or np.ptp(v)==0:return []
    inv=np.linalg.inv(X.T@X);b=np.linalg.lstsq(X,v,rcond=None)[0];e=v-X@b
    h=np.einsum('ij,jk,ik->i',X,inv,X)
    cov=inv@(X.T@((e/(1-h))[:,None]**2*X))@inv
    se=np.sqrt(np.maximum(np.diag(cov),0));p=2*t.sf(np.abs(np.divide(b,se,out=np.full_like(b,np.nan),where=se>0)),n-k)
    r2=1-float(e@e)/float(np.sum((v-v.mean())**2));simple=linregress(d[predictors[0]],v) if len(predictors)==1 else None
    return [dict(topic=topic,model=model,x=x,y=y,term=term,n=n,coefficient=float(b[j]),
        standard_error_hc3=float(se[j]),p_value_hc3=float(p[j]),r_squared=r2,
        pearson_r=float(simple.rvalue) if simple else np.nan,
        p_value_ols=float(simple.pvalue) if simple and j==1 else np.nan,
        slope_ci95_low=float(b[j]-t.ppf(.975,n-k)*se[j]),
        slope_ci95_high=float(b[j]+t.ppf(.975,n-k)*se[j])) for j,term in enumerate(['intercept',*predictors])]


SEASON_TOPICS=[
    ('assisted_baskets','conditional_receiver_entropy','assisted_basket_pct','Passing diversity and assisted baskets',
        'Normalized conditional receiver entropy H(R | P)','Assisted field goals (%)'),
    ('catch_shoot','conditional_receiver_entropy','catch_shoot_fga_per100','Passing diversity and catch-and-shoot opportunities',
        'Normalized conditional receiver entropy H(R | P)','Catch-and-shoot FGA per 100 team possessions'),
    ('ball_holding','eigenvector_concentration','seconds_per_touch','Passing concentration and ball holding',
        'Eigenvector concentration (higher = more star-centered)','Average seconds per touch'),
    ('stability_turnovers','mean_receiving_jsd','turnovers_per100','Passing instability and turnovers',
        'Mean consecutive-game receiving-share JSD (higher = less stable)','Turnovers per 100 team possessions')]


def season_topics_spread_labels(fig):
    """Place short team labels near points while reducing label collisions."""
    from matplotlib.text import Annotation
    from matplotlib.transforms import Bbox
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    offsets=[(4,4),(4,-10),(-23,4),(-23,-10),(4,12),(-23,12),(12,-3),(-30,-3),
             (4,-18),(-23,-18),(13,13),(-31,13),(13,-18),(-31,-18),(0,21),(0,-27)]
    for ax in fig.axes:
        labels=[x for x in ax.texts if isinstance(x,Annotation)]
        marker_boxes=[]
        for label in labels:
            px,py=ax.transData.transform(label.xy)
            marker_boxes.append(Bbox.from_bounds(px-3,py-3,6,6))
        placed=[]
        for label in labels:
            choices=[]
            for offset in offsets:
                label.set_position(offset);box=label.get_window_extent(renderer).expanded(1.13,1.15)
                collision=sum(box.overlaps(other) for other in placed)
                markers=sum(box.overlaps(other) for other in marker_boxes)
                outside=not ax.bbox.contains(box.x0,box.y0) or not ax.bbox.contains(box.x1,box.y1)
                score=collision*10000+markers*100+outside*100000+.1*(offset[0]**2+offset[1]**2)
                choices.append((score,offset,box))
            _,offset,box=min(choices,key=lambda t:t[0]);label.set_position(offset);placed.append(box)


def plot_season_topics(frame,results,out):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from scipy.stats import linregress
    def draw(ax,spec):
        key,x,y,title,xlabel,ylabel=spec
        d=frame.loc[frame.stability_eligible] if key=='stability_turnovers' else frame
        d=d.dropna(subset=[x,y])
        for style,color,marker in [('more_balanced','#2473b8','o'),('more_star_centered','#e18a24','^')]:
            g=d.loc[d.passing_style.eq(style)]
            ax.scatter(g[x],g[y],c=color,marker=marker,s=53,alpha=.9,edgecolors='white',linewidths=.5,zorder=3)
            for _,r in g.iterrows():ax.annotate(r.team,(r[x],r[y]),xytext=(4,4),textcoords='offset points',fontsize=7)
        if len(d)>=3 and d[x].nunique()>1:
            fit=linregress(d[x],d[y]);xx=np.linspace(d[x].min(),d[x].max(),100)
            ax.plot(xx,fit.intercept+fit.slope*xx,c='#333333',lw=1.8,label='Pooled trend')
        r=results.loc[(results.topic==key)&results.model.eq('primary_pooled')&results.term.eq(x)]
        if len(r):
            r=r.iloc[0];ax.set_title(title+'\n'+f'N={r.n:.0f} | r={r.pearson_r:.3f} | R²={r.r_squared:.3f} | HC3 p={r.p_value_hc3:.3g}',fontsize=11,pad=12)
        else:ax.set_title(title+'\nInsufficient validated data',fontsize=11)
        ax.set_xlabel(xlabel,fontsize=9);ax.set_ylabel(ylabel,fontsize=9)
        ax.grid(alpha=.16);ax.spines[['top','right']].set_visible(False);ax.margins(.12)
    legend=[Line2D([0],[0],marker='o',ls='',color='#2473b8',label='More balanced'),
        Line2D([0],[0],marker='^',ls='',color='#e18a24',label='More star-centered'),
        Line2D([0],[0],color='#333333',label='Pooled trend')]
    fig,axes=plt.subplots(2,2,figsize=(16,12))
    for ax,spec in zip(axes.flat,SEASON_TOPICS):draw(ax,spec)
    fig.suptitle('NBA 2025–26 regular season: four team-season comparisons',fontsize=18,y=.985)
    fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(.5,.955),ncol=3,frameon=False)
    fig.text(.5,.015,'One point per team-season. No minutes cutoff. Styles: league-median eigenvector concentration.\n'
        'Stability: season mean of consecutive-game receiving-share JSD; unavailable pairs excluded; at least 75% pair coverage required.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.91));season_topics_spread_labels(fig);fig.savefig(out/'four_team_season_regressions.png',dpi=180)
    fig.savefig(out/'four_team_season_regressions.pdf');plt.close(fig)
    for spec in SEASON_TOPICS:
        fig,ax=plt.subplots(figsize=(11,8));draw(ax,spec)
        fig.suptitle('NBA 2025–26 regular season · one point per team',fontsize=13,y=.985)
        fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(.5,.95),ncol=3,frameon=False)
        fig.tight_layout(rect=(0,.025,1,.88));season_topics_spread_labels(fig);fig.savefig(out/(spec[0]+'.png'),dpi=180)
        fig.savefig(out/(spec[0]+'.pdf'));plt.close(fig)


def run_team_season_topics(argv):
    from concurrent.futures import ThreadPoolExecutor,as_completed
    from nba_api.stats.endpoints import leaguedashteamstats,leaguedashptstats,leaguegamelog
    parser=argparse.ArgumentParser(description='Four 2025-26 cross-team season regressions, no minutes cutoff')
    parser.add_argument('--team-season-topics',action='store_true')
    parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--plot-only',action='store_true');parser.add_argument('--refresh',action='store_true')
    parser.add_argument('--workers',type=int,choices=[1,2],default=2)
    parser.add_argument('--timeout',type=float,default=30);parser.add_argument('--retries',type=int,default=2)
    parser.add_argument('--min-delay',type=float,default=.5);parser.add_argument('--max-delay',type=float,default=1)
    args=parser.parse_args(argv);out=args.output_dir.expanduser()/'2025-26/team_season_topics';out.mkdir(parents=True,exist_ok=True)
    metrics_path=out/'team_season_metrics.csv'
    if args.plot_only:
        frame=pd.read_csv(metrics_path)
    else:
        c=NBAClient(Config(season='2025-26',cache_dir=out/'cache',refresh=args.refresh,
            timeout=args.timeout,retries=args.retries,min_delay=args.min_delay,max_delay=args.max_delay))
        def fetch(cls,params,name,required,filename):
            f,_=c.fetch(cls,params,lambda p:season_topics_frame(p,name,required));atomic_csv(f,out/filename);return f
        try:
            common=dict(season='2025-26',season_type_all_star='Regular Season')
            base=fetch(leaguedashteamstats.LeagueDashTeamStats,dict(per_mode_detailed='Totals',**common),
                'LeagueDashTeamStats',['TEAM_ID','GP','AST','FGM','TOV'],'official_team_totals.csv')
            advanced=fetch(leaguedashteamstats.LeagueDashTeamStats,dict(per_mode_detailed='Totals',measure_type_detailed_defense='Advanced',**common),
                'LeagueDashTeamStats',['TEAM_ID','GP','POSS'],'official_team_advanced.csv')
            touches=fetch(leaguedashptstats.LeagueDashPtStats,dict(player_or_team='Team',per_mode_simple='Totals',pt_measure_type='Possessions',**common),
                'LeagueDashPtStats',['TEAM_ID','GP','TOUCHES','TIME_OF_POSS','AVG_SEC_PER_TOUCH'],'team_touches.csv')
            shots=fetch(leaguedashptstats.LeagueDashPtStats,dict(player_or_team='Team',per_mode_simple='Totals',pt_measure_type='CatchShoot',**common),
                'LeagueDashPtStats',['TEAM_ID','GP','CATCH_SHOOT_FGA'],'team_catch_shoot.csv')
            games=fetch(leaguegamelog.LeagueGameLog,dict(season='2025-26'),
                'LeagueGameLog',['TEAM_ID','TEAM_ABBREVIATION','GAME_ID','GAME_DATE','AST','FGM','TOV'],'team_games.csv')
            players=fetch(leaguegamelog.LeagueGameLog,dict(season='2025-26',player_or_team_abbreviation='P'),
                'LeagueGameLog',['TEAM_ID','PLAYER_ID','GAME_ID','GAME_DATE'],'player_games.csv')
            teams=static_teams.get_teams();expected={t['id'] for t in teams}
            for f in [base,advanced,touches,shots]:
                if len(f)!=30 or f.TEAM_ID.duplicated().any() or set(f.TEAM_ID)!=expected or not f.GP.eq(82).all():
                    raise ValueError('Expected one full 82-game 2025-26 record for each of 30 teams')
            if len(games)!=2460 or games.duplicated(['TEAM_ID','GAME_ID']).any() or not games.groupby('TEAM_ID').size().eq(82).all():
                raise ValueError('Expected 2460 regular-season team-game records')
            if players.duplicated(['TEAM_ID','GAME_ID','PLAYER_ID']).any():raise ValueError('Duplicate official player-game records')
            check=games.groupby('TEAM_ID')[['AST','FGM','TOV']].sum().sort_index()
            np.testing.assert_allclose(check,base.set_index('TEAM_ID').loc[check.index,['AST','FGM','TOV']])
            nets=[]
            for i,team in enumerate(sorted(teams,key=lambda t:t['abbreviation'])):
                print(f'Season network {i+1}/30: {team["abbreviation"]}',flush=True)
                participant_games=players.loc[players.TEAM_ID.eq(team['id'])]
                roster=participant_games.groupby(['PLAYER_ID','PLAYER_NAME'],as_index=False).agg(GP=('GAME_ID','size'))
                nets.append(season_topics_network(c,team,out/'networks'/team['abbreviation'],args.workers,roster))
            daily_dir=out/'daily_passing';daily_dir.mkdir(exist_ok=True)
            dates=sorted(games.GAME_DATE.astype(str).str[:10].unique())
            def daily(date):
                target=daily_dir/(date+'.csv')
                if target.exists() and not args.refresh:
                    # Empty responses have their own saved audit and remain visibly unavailable.
                    saved=pd.read_csv(target)
                    if 'REPORTED_TEAM_ID' in saved:return date,len(saved)
                day=pd.Timestamp(date).strftime('%m/%d/%Y');expected_today=set(games.loc[games.GAME_DATE.astype(str).str[:10].eq(date),'TEAM_ID'])
                participation=players.loc[players.GAME_DATE.astype(str).str[:10].eq(date)]
                if participation.PLAYER_ID.duplicated().any():raise ValueError('A player appears for multiple teams on one date')
                actual_teams=participation.set_index('PLAYER_ID').TEAM_ID.to_dict()
                try:
                    f,_=c.fetch(leaguedashptstats.LeagueDashPtStats,dict(player_or_team='Player',per_mode_simple='Totals',pt_measure_type='Passing',
                        date_from_nullable=day,date_to_nullable=day,**common),lambda p:season_topics_daily_passing(p,expected_today,actual_teams))
                    error='Empty tracking response' if f.empty else ''
                except DownloadsStopped:raise
                except DataUnavailable as exc:
                    f=pd.DataFrame(columns=['TEAM_ID','PLAYER_ID','PASSES_RECEIVED']);error=str(exc)
                atomic_csv(f,target)
                atomic_json(dict(date=date,player_rows=len(f),error=error),daily_dir/(date+'_audit.json'))
                return date,len(f)
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures={pool.submit(daily,date):date for date in dates}
                for i,future in enumerate(as_completed(futures),1):
                    date,count=future.result()
                    if i%10==0 or i==len(dates):print(f'Daily receiving tracking {i}/{len(dates)} dates complete (latest {date}: {count} players)',flush=True)
            stability=season_topics_stability(games,players,daily_dir,out)
            frame=pd.DataFrame(nets).drop(columns=['missing_profiles','participant_ids'])
            frame=frame.merge(stability,on='team',validate='one_to_one')
            b=base.set_index('TEAM_ID');a=advanced.set_index('TEAM_ID');t=touches.set_index('TEAM_ID');s=shots.set_index('TEAM_ID')
            for col in ['FGM','AST','TOV']:frame[col.lower()]=frame.team_id.map(b[col])
            frame['games']=frame.team_id.map(b.GP);frame['possessions']=frame.team_id.map(a.POSS)
            frame['touches']=frame.team_id.map(t.TOUCHES);frame['time_of_possession_minutes']=frame.team_id.map(t.TIME_OF_POSS)
            frame['reported_seconds_per_touch']=frame.team_id.map(t.AVG_SEC_PER_TOUCH)
            frame['seconds_per_touch']=60*frame.time_of_possession_minutes/frame.touches
            if not (abs(frame.seconds_per_touch-frame.reported_seconds_per_touch)<.06).all():
                raise ValueError('Time-of-possession totals do not reconcile with reported seconds per touch')
            frame['catch_shoot_fga']=frame.team_id.map(s.CATCH_SHOOT_FGA)
            for col in ['possessions','touches','fgm']:
                if not np.isfinite(frame[col]).all() or frame[col].le(0).any():raise ValueError('Invalid season denominator '+col)
            frame['assisted_basket_pct']=100*frame.ast/frame.fgm
            frame['catch_shoot_fga_per100']=100*frame.catch_shoot_fga/frame.possessions
            frame['turnovers_per100']=100*frame.tov/frame.possessions
            cutoff=float(frame.loc[frame.network_valid,'eigenvector_concentration'].median())
            frame['passing_style']=np.where(frame.eigenvector_concentration.le(cutoff),'more_balanced','more_star_centered')
            frame.loc[~frame.network_valid,'passing_style']='unavailable';frame['style_cutoff']=cutoff
            atomic_csv(frame,metrics_path)
        finally:c.close()
    model_rows=[]
    for key,x,y,*_ in SEASON_TOPICS:
        d=frame.loc[frame.network_valid.astype(str).str.lower().eq('true')].copy()
        if key=='stability_turnovers':d=d.loc[d.stability_eligible.astype(str).str.lower().eq('true')]
        model_rows.extend(season_topics_fit(d,x,y,'primary_pooled',key))
        model_rows.extend(season_topics_fit(d.loc[d.full_network_profiles.astype(str).str.lower().eq('true')],x,y,'complete_profiles_sensitivity',key))
        if key!='stability_turnovers':
            model_rows.extend(season_topics_fit(d,x,y,'player_count_adjusted',key,[x,'network_player_count']))
        else:
            model_rows.extend(season_topics_fit(d,'mean_returning_player_jsd',y,'returning_players_sensitivity',key))
        if key!='ball_holding':
            for style in ['more_balanced','more_star_centered']:
                model_rows.extend(season_topics_fit(d.loc[d.passing_style.eq(style)],x,y,style,key))
            d['entropy_or_jsd_centered']=d[x]-d[x].mean();d['star_indicator']=d.passing_style.eq('more_star_centered').astype(int)
            d['style_interaction']=d.entropy_or_jsd_centered*d.star_indicator
            model_rows.extend(season_topics_fit(d,x,y,'style_interaction',key,['entropy_or_jsd_centered','star_indicator','style_interaction']))
    results=pd.DataFrame(model_rows);results['primary_p_holm_hc3']=np.nan
    primary=results.index[results.model.eq('primary_pooled')&results.term.ne('intercept')]
    order=results.loc[primary,'p_value_hc3'].sort_values().index;running=0
    for rank,idx in enumerate(order):
        running=max(running,min(1,float(results.loc[idx,'p_value_hc3'])*(len(order)-rank)))
        results.loc[idx,'primary_p_holm_hc3']=running
    atomic_csv(results,out/'regression_results.csv');plot_season_topics(frame,results,out)
    summary=results.loc[primary];atomic_csv(summary,out/'four_primary_regressions.csv')
    atomic_json(dict(season='2025-26',season_type='Regular Season',observations='One team-season per point, 30 teams',
        no_minutes_cutoff=True,passing_missing_policy='Unavailable outgoing passing weights=0; profile counts reported; complete-profile sensitivity included',
        entropy='H(receiver | passer) / ln(N-1), weighted directed season passes, isolates omitted',
        centrality='Undirected A+A.T weighted eigenvector, L1 shares; normalized share HHI concentration',
        style_rule='Median eigenvector concentration across this season, fixed for all outcomes',
        outcomes=dict(assisted_baskets='100*team AST/FGM',catch_shoot='100*catch-and-shoot FGA/official team POSS',
            ball_holding='60*total TIME_OF_POSS minutes / total TOUCHES, checked against NBA AVG_SEC_PER_TOUCH',
            stability='Mean adjacent-game base-2 receiving-share JSD; no bridging missing games; >=75% of 81 pairs required',
            turnovers='100*team TOV/official team POSS'),
        stability_teams=int(frame.stability_eligible.astype(str).str.lower().eq('true').sum()),
        inference='Cross-sectional OLS with HC3 robust t inference; Holm correction across four primary slopes; style tests exploratory',
        limitations=['Associations do not establish causality','Only one season; 30 observations and roughly 15 per style',
            'Season networks combine different lineups and trades; node count adjusted sensitivity provided',
            'Receiving-share stability measures player involvement, not ordered passes or tactical execution',
            'Full profile availability does not guarantee all real-world passes were tracked',
            'Catch-and-shoot attempts measure realized shots, not every opportunity']),out/'analysis_audit.json')
    print(summary[['topic','n','pearson_r','r_squared','p_value_hc3','primary_p_holm_hc3']].to_string(index=False),flush=True)
    print(f'Finished four 2025-26 team-season topics: {out}',flush=True)
    return 0


# ---------------------------------------------------------------------
# FIVE ADDITIONAL 2025-26 TEAM-SEASON HYPOTHESES
# python teamgraphs.py --team-season-five-topics
# ---------------------------------------------------------------------
FIVE_SEASON_TOPICS=[
 ('assist_distribution','conditional_receiver_entropy','assist_concentration','Passing diversity and assist concentration',
  'Normalized conditional receiver entropy H(R | P)','Assist concentration (higher = less evenly shared)'),
 ('scoring_star','eigenvector_concentration','top_scorer_share_pct','Passing concentration and scoring dependence',
  'Eigenvector concentration (higher = more star-centered)',"Top scorer’s share of team points (%)"),
 ('potential_assists','passes_per100','potential_assists_per100','Passing volume and potential assists',
  'Passes made per 100 team possessions','Potential assists per 100 team possessions'),
 ('secondary_assists','conditional_receiver_entropy','secondary_assists_per100','Passing diversity and secondary assists',
  'Normalized conditional receiver entropy H(R | P)','Secondary assists per 100 team possessions'),
 ('roster_disruption','mean_minutes_jsd','mean_receiving_jsd','Playing-time changes and passing instability',
  'Mean consecutive-game minutes-share JSD','Mean consecutive-game receiving-share JSD')]


def normalized_share_concentration(counts):
    """Normalized HHI over all supplied participants, including zero contributors."""
    v=np.asarray(counts,dtype=float)
    if v.ndim!=1 or not np.isfinite(v).all() or (v<0).any():raise ValueError('Invalid concentration counts')
    if len(v)<2 or v.sum()<=0:return np.nan,np.nan
    shares=v/v.sum();hhi=float(shares@shares)
    return float((hhi-1/len(v))/(1-1/len(v))),hhi


def season_minutes_disruption(players,pairs):
    """Match the exact consecutive pairs retained for receiving stability."""
    players=players.copy();players['GAME_ID']=players.GAME_ID.astype(str).str.zfill(10)
    if players.duplicated(['TEAM_ID','GAME_ID','PLAYER_ID']).any():raise ValueError('Duplicate player-games')
    lookup={(int(team),str(game)):g.set_index('PLAYER_ID').MIN.astype(float)
        for (team,game),g in players.groupby(['TEAM_ID','GAME_ID'])}
    rows=[]
    for r in pairs.to_dict('records'):
        row=dict(r,minutes_jsd=np.nan,returning_minutes_jsd=np.nan)
        if str(r['comparison_available']).lower()=='true':
            team=int(r['team_id']);a=lookup[(team,str(r['previous_game_id']).zfill(10))];b=lookup[(team,str(r['game_id']).zfill(10))]
            union=sorted(set(a.index)|set(b.index));common=sorted(set(a.index)&set(b.index))
            row['minutes_jsd']=receiving_share_js_divergence(a.reindex(union,fill_value=0),b.reindex(union,fill_value=0))
            if len(common)>=2 and a.reindex(common).sum()>0 and b.reindex(common).sum()>0:
                row['returning_minutes_jsd']=receiving_share_js_divergence(a.reindex(common),b.reindex(common))
        rows.append(row)
    detail=pd.DataFrame(rows);summary=[]
    for team,g in detail.groupby('team'):
        v=g.dropna(subset=['minutes_jsd','js_divergence'])
        returning=v.dropna(subset=['returning_minutes_jsd','returning_player_js_divergence'])
        summary.append(dict(team=team,mean_minutes_jsd=v.minutes_jsd.mean(),
            matched_mean_receiving_jsd=v.js_divergence.mean(),matched_pairs=len(v),
            matched_mean_returning_minutes_jsd=returning.returning_minutes_jsd.mean(),
            matched_mean_returning_receiving_jsd=returning.returning_player_js_divergence.mean(),returning_pairs=len(returning)))
    return detail,pd.DataFrame(summary)


def plot_five_season_topics(frame,results,out):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from scipy.stats import linregress
    legends=[Line2D([0],[0],ls='',marker='o',color='#2473b8',label='More balanced'),
        Line2D([0],[0],ls='',marker='^',color='#e18a24',label='More star-centered'),
        Line2D([0],[0],color='#333333',label='Pooled trend')]
    def draw(ax,spec):
        key,x,y,title,xlabel,ylabel=spec;d=frame.dropna(subset=[x,y])
        if key=='roster_disruption':d=d.loc[d.stability_eligible.astype(str).str.lower().eq('true')]
        for style,color,marker in [('more_balanced','#2473b8','o'),('more_star_centered','#e18a24','^')]:
            g=d.loc[d.passing_style.eq(style)]
            ax.scatter(g[x],g[y],color=color,marker=marker,s=55,edgecolors='white',linewidths=.5,zorder=3)
            for _,r in g.iterrows():ax.annotate(r.team,(r[x],r[y]),xytext=(4,4),textcoords='offset points',fontsize=7)
        if len(d)>=3:
            fit=linregress(d[x],d[y]);xx=np.linspace(d[x].min(),d[x].max(),100)
            ax.plot(xx,fit.intercept+fit.slope*xx,color='#333333',lw=1.8)
        r=results.loc[results.topic.eq(key)&results.model.eq('primary_pooled')&results.term.eq(x)].iloc[0]
        ax.set_title(title+'\n'+f'N={r.n:.0f} | r={r.pearson_r:.3f} | R²={r.r_squared:.3f} | HC3 p={r.p_value_hc3:.3g}',fontsize=11,pad=11)
        ax.set_xlabel(xlabel,fontsize=9);ax.set_ylabel(ylabel,fontsize=9);ax.grid(alpha=.16)
        ax.spines[['top','right']].set_visible(False);ax.margins(.12)
    fig,axes=plt.subplots(3,2,figsize=(16,17))
    for ax,spec in zip(axes.flat,FIVE_SEASON_TOPICS):draw(ax,spec)
    axes.flat[-1].axis('off')
    axes.flat[-1].text(.02,.9,'How to read these graphs',fontsize=14,weight='bold',transform=axes.flat[-1].transAxes)
    axes.flat[-1].text(.02,.8,'• One point per team: 2025–26 regular season.\n\n'
        '• The same 15 balanced / 15 star-centered labels\n  are retained from the previous analysis.\n\n'
        '• Assist concentration uses normalized HHI;\n  higher values mean fewer players supply assists.\n\n'
        '• JSD measures changes in player shares;\n  higher values mean greater game-to-game change.\n\n'
        '• Correlations are exploratory associations.\n  Robust inference and sensitivity tests are saved.',
        fontsize=11,va='top',transform=axes.flat[-1].transAxes,linespacing=1.4)
    fig.suptitle('NBA 2025–26: five additional team-season hypotheses',fontsize=18,y=.985)
    fig.legend(handles=legends,loc='upper center',bbox_to_anchor=(.5,.96),ncol=3,frameon=False)
    fig.tight_layout(rect=(0,.02,1,.93));season_topics_spread_labels(fig)
    fig.savefig(out/'five_team_season_regressions.png',dpi=180);fig.savefig(out/'five_team_season_regressions.pdf');plt.close(fig)
    for spec in FIVE_SEASON_TOPICS:
        fig,ax=plt.subplots(figsize=(11,8));draw(ax,spec)
        fig.suptitle('NBA 2025–26 regular season · one point per team',fontsize=13,y=.985)
        fig.legend(handles=legends,loc='upper center',bbox_to_anchor=(.5,.95),ncol=3,frameon=False)
        fig.tight_layout(rect=(0,.025,1,.88));season_topics_spread_labels(fig)
        fig.savefig(out/(spec[0]+'.png'),dpi=180);fig.savefig(out/(spec[0]+'.pdf'));plt.close(fig)


def run_five_team_season_topics(argv):
    from nba_api.stats.endpoints import leaguedashptstats
    parser=argparse.ArgumentParser(description='Five additional 2025-26 cross-team season hypotheses')
    parser.add_argument('--team-season-five-topics',action='store_true')
    parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--plot-only',action='store_true');parser.add_argument('--refresh',action='store_true')
    args=parser.parse_args(argv)
    parent=args.output_dir.expanduser()/'2025-26/team_season_topics';out=parent/'five_additional_topics';out.mkdir(parents=True,exist_ok=True)
    if args.plot_only:frame=pd.read_csv(out/'team_season_metrics.csv')
    else:
        frame=pd.read_csv(parent/'team_season_metrics.csv')
        if len(frame)!=30 or frame.team.duplicated().any() or not frame.games.eq(82).all():raise ValueError('Expected 30 complete 2025-26 team-seasons')
        players=pd.read_csv(parent/'player_games.csv',dtype={'GAME_ID':str})
        official=pd.read_csv(parent/'official_team_totals.csv').set_index('TEAM_ID')
        if players.duplicated(['TEAM_ID','GAME_ID','PLAYER_ID']).any():raise ValueError('Duplicate player participation')
        for column in ['AST','PTS','MIN']:
            v=pd.to_numeric(players[column],errors='raise')
            if not np.isfinite(v).all() or v.lt(0).any():raise ValueError('Invalid '+column)
        rows=[];player_shares=[]
        for _,r in frame.iterrows():
            g=players.loc[players.TEAM_ID.eq(r.team_id)]
            totals=g.groupby(['PLAYER_ID','PLAYER_NAME'],as_index=False).agg(assists=('AST','sum'),points=('PTS','sum'),games=('GAME_ID','size'))
            if len(totals)!=r.roster_player_count:raise ValueError('Season participant mismatch')
            np.testing.assert_allclose(totals.assists.sum(),official.loc[r.team_id,'AST'])
            # Official team points may include team-only heaves; use the official denominator.
            team_points=float(official.loc[r.team_id,'PTS'])
            if totals.points.sum()>team_points:raise ValueError('Player points exceed team points')
            concentration,hhi=normalized_share_concentration(totals.assists)
            scorer=totals.sort_values(['points','PLAYER_ID'],ascending=[False,True]).iloc[0]
            rows.append(dict(team=r.team,assist_concentration=concentration,assist_hhi=hhi,
                top_scorer_share_pct=100*scorer.points/team_points,top_scorer_id=int(scorer.PLAYER_ID),
                top_scorer_name=scorer.PLAYER_NAME,team_points=team_points,player_points_total=float(totals.points.sum())))
            totals['team']=r.team;totals['assist_share']=totals.assists/totals.assists.sum()
            totals['scoring_share']=totals.points/team_points;player_shares.append(totals)
        atomic_csv(pd.concat(player_shares,ignore_index=True),out/'player_season_shares.csv')
        frame=frame.merge(pd.DataFrame(rows),on='team',validate='one_to_one')
        c=NBAClient(Config(season='2025-26',cache_dir=parent/'cache',refresh=args.refresh,timeout=30,retries=2,min_delay=.5,max_delay=1))
        try:
            tracking,_=c.fetch(leaguedashptstats.LeagueDashPtStats,
                dict(season='2025-26',season_type_all_star='Regular Season',player_or_team='Team',pt_measure_type='Passing',per_mode_simple='Totals'),
                lambda p:season_topics_frame(p,'LeagueDashPtStats',['TEAM_ID','GP','PASSES_MADE','POTENTIAL_AST','SECONDARY_AST']))
        finally:c.close()
        if len(tracking)!=30 or tracking.TEAM_ID.duplicated().any() or not tracking.GP.eq(82).all() or set(tracking.TEAM_ID)!=set(frame.team_id):
            raise ValueError('Passing tracking does not contain all 30 full team-seasons')
        atomic_csv(tracking,out/'official_team_passing.csv');tracking=tracking.set_index('TEAM_ID')
        for source,dest in [('PASSES_MADE','passes'),('POTENTIAL_AST','potential_assists'),('SECONDARY_AST','secondary_assists')]:
            frame[dest]=frame.team_id.map(tracking[source])
            if not np.isfinite(frame[dest]).all() or frame[dest].lt(0).any():raise ValueError('Invalid tracking total '+dest)
            frame[dest+'_per100']=100*frame[dest]/frame.possessions
            frame[dest+'_pergame']=frame[dest]/frame.games
        pairs=pd.read_csv(parent/'consecutive_game_stability.csv',dtype={'game_id':str,'previous_game_id':str})
        details,disruption=season_minutes_disruption(players,pairs)
        atomic_csv(details,out/'minutes_and_receiving_game_pairs.csv')
        frame=frame.merge(disruption,on='team',validate='one_to_one')
        np.testing.assert_allclose(frame.mean_receiving_jsd,frame.matched_mean_receiving_jsd)
        if not frame.matched_pairs.eq(frame.valid_stability_pairs).all():raise ValueError('Unmatched stability pair counts')
        atomic_csv(frame,out/'team_season_metrics.csv')
    results=[]
    for topic,x,y,*_ in FIVE_SEASON_TOPICS:
        d=frame.loc[frame.network_valid.astype(str).str.lower().eq('true')].copy()
        if topic=='roster_disruption':d=d.loc[d.stability_eligible.astype(str).str.lower().eq('true')]
        results.extend(season_topics_fit(d,x,y,'primary_pooled',topic))
        if topic in ['assist_distribution','scoring_star','secondary_assists']:
            results.extend(season_topics_fit(d,x,y,'player_count_adjusted',topic,[x,'network_player_count']))
        if topic=='assist_distribution':results.extend(season_topics_fit(d,x,'assist_hhi','raw_hhi_sensitivity',topic))
        if topic=='potential_assists':results.extend(season_topics_fit(d,'passes_pergame','potential_assists_pergame','per_game_sensitivity',topic))
        if topic=='roster_disruption':
            results.extend(season_topics_fit(d,'matched_mean_returning_minutes_jsd','matched_mean_returning_receiving_jsd','returning_players_sensitivity',topic))
            results.extend(season_topics_fit(d,x,y,'lineup_change_adjusted',topic,[x,'mean_lineup_change']))
        if topic!='scoring_star':
            for style in ['more_balanced','more_star_centered']:
                results.extend(season_topics_fit(d.loc[d.passing_style.eq(style)],x,y,style,topic))
            d['centered_predictor']=d[x]-d[x].mean();d['star_indicator']=d.passing_style.eq('more_star_centered').astype(int)
            d['style_interaction']=d.centered_predictor*d.star_indicator
            results.extend(season_topics_fit(d,x,y,'style_interaction',topic,['centered_predictor','star_indicator','style_interaction']))
    r=pd.DataFrame(results);r['p_holm_five_hc3']=np.nan
    primary=r.index[r.model.eq('primary_pooled')&r.term.ne('intercept')];order=r.loc[primary,'p_value_hc3'].sort_values().index;running=0
    for rank,idx in enumerate(order):
        running=max(running,min(1,float(r.loc[idx,'p_value_hc3'])*(len(order)-rank)));r.loc[idx,'p_holm_five_hc3']=running
    atomic_csv(r,out/'regression_results.csv');atomic_csv(r.loc[primary],out/'five_primary_regressions.csv')
    plot_five_season_topics(frame,r,out)
    atomic_json(dict(season='2025-26',observation='30 full regular-season team totals; not individual games',
        passing_style='Unchanged league-median eigenvector labels from the previous four-topic analysis',no_minutes_cutoff=True,
        assist_concentration='(sum(player assist shares squared)-1/N)/(1-1/N), N=all team-season participants including zero-assist players',
        scoring_share='Highest individual team-stint points / official team points; traded players count only points for that team',
        tracking_rates='100*season total/official team possessions; GP=82 validated for every team',
        roster_changes='Average adjacent-game base-2 JSD of player minutes shares, matched to exactly the valid receiving-share pairs',
        inference='OLS HC3 robust t inference; Holm adjustment across five primary slopes; group and sensitivity models exploratory',
        limitations=['Raw season potential-assist totals contain failed daily validations; use --passing-volume-by-style for matched valid-game rates',
            'These are exploratory follow-ups to earlier weak results, not a preregistered confirmatory study',
            'Shared participation and playing time can drive associations between player distributions',
            'Passing volume and potential assists are mechanically related opportunity counts; a correlation does not establish passing effectiveness',
            'Minutes are the rounded values supplied in official player-game logs',
            'Season totals obscure within-game tactics and lineup overlap; 30 teams only',
            'Secondary assists depend on the eventual basket being made; not a complete count of multi-player combinations']),out/'analysis_audit.json')
    print(r.loc[primary,['topic','n','pearson_r','r_squared','p_value_hc3','p_holm_five_hc3']].to_string(index=False),flush=True)
    print(f'Finished five additional topics: {out}',flush=True)
    return 0


def download_hub_absence_profiles(parent,out):
    import sys
    from concurrent.futures import ThreadPoolExecutor,as_completed
    from nba_api.stats.endpoints import playerdashptpass
    m=sys.modules[__name__]
    cache=out/'cache';cache.mkdir(parents=True,exist_ok=True)
    g=m.pd.read_csv(parent/'team_games.csv',dtype={'GAME_ID':str});ps=m.pd.read_csv(parent/'player_games.csv',dtype={'GAME_ID':str});n=m.pd.read_csv(parent/'team_season_metrics.csv');daily=m.pd.concat([m.pd.read_csv(f).assign(date=f.stem) for f in (parent/'daily_passing').glob('*.csv')],ignore_index=True)
    plan=[];elig=[]
    for _,r in n.sort_values('team').iterrows():
     t=g[g.TEAM_ID==r.team_id].sort_values(['GAME_DATE','GAME_ID']).copy();hub=int(r.top_eigenvector_player_id);hubgames=ps[(ps.TEAM_ID==r.team_id)&(ps.PLAYER_ID==hub)];present=set(hubgames.GAME_ID);t['present']=t.GAME_ID.isin(present);eligible=len(present)>=20 and (~t.present).sum()>=10 and ps.loc[ps.PLAYER_ID==hub,'TEAM_ID'].nunique()==1
     elig.append(dict(team=r.team,team_id=int(r.team_id),hub_id=hub,hub_name=hubgames.PLAYER_NAME.iloc[0],present_games=len(present),absent_games=int((~t.present).sum()),eligible=eligible))
     if not eligible:continue
     t['block']=t.present.ne(t.present.shift()).cumsum()
     for _,b in t[~t.present].groupby('block'):
      start=str(b.GAME_DATE.min())[:10];end=str(b.GAME_DATE.max())[:10]
      for pid,pg in ps[(ps.TEAM_ID==r.team_id)&ps.GAME_ID.isin(b.GAME_ID)].groupby('PLAYER_ID'):
       observed=daily[(daily.TEAM_ID==r.team_id)&(daily.PLAYER_ID==pid)&daily.date.between(start,end)]
       plan.append(dict(team=r.team,team_id=int(r.team_id),hub_id=hub,start=start,end=end,player_id=int(pid),expected_games=len(pg),
        observed_daily_games=len(observed),daily_passes_made=int(observed.PASSES_MADE.sum())))
    plan=m.pd.DataFrame(plan);m.atomic_csv(plan,out/'download_plan.csv');m.atomic_csv(m.pd.DataFrame(elig),out/'eligibility.csv')
    c=m.NBAClient(m.Config(season='2025-26',cache_dir=cache,timeout=25,retries=2,min_delay=.5,max_delay=1))
    def one(r):
     folder=out/'absent_blocks'/r['team']/(r['start']+'_'+r['end']);folder.mkdir(parents=True,exist_ok=True);target=folder/(str(r['player_id'])+'.csv');audit=target.with_suffix('.json')
     if target.exists() and audit.exists():return json.loads(audit.read_text())
     params=dict(player_id=r['player_id'],team_id=r['team_id'],season='2025-26',season_type_all_star='Regular Season',per_mode_simple='Totals',
     date_from_nullable=m.pd.Timestamp(r['start']).strftime('%m/%d/%Y'),date_to_nullable=m.pd.Timestamp(r['end']).strftime('%m/%d/%Y'))
     def validate(payload):
      f=m.validate_passing(payload,r['player_id'],r['team_id'])
      if len(f):
       m.numeric_column(f,'G',integer=True)
       if f.G.gt(r['expected_games']).any():raise ValueError('Passing date filter contains more games than official participation')
      return f
     try:
      f,_=c.fetch(playerdashptpass.PlayerDashPtPass,params,validate);error=''
      observed_zero=(r['observed_daily_games']==r['expected_games'] and r['daily_passes_made']==0)
      complete=not f.empty or observed_zero
      if not complete:error='Empty profile with no independently confirmed zero passing total'
     except m.DownloadsStopped:raise
     except m.DataUnavailable as e:f=m.pd.DataFrame(columns=['PLAYER_ID','TEAM_ID','PASS_TEAMMATE_PLAYER_ID','PASS']);complete=False;error=str(e)
     m.atomic_csv(f,target);result=dict(r,complete=complete,error=error,profile_passes=float(f.PASS.sum()),
      daily_difference=float(f.PASS.sum()-r['daily_passes_made']))
     m.atomic_json(result,audit);return result
    try:
     print(f'Fetching {len(plan)} absence-block player profiles for {sum(e["eligible"] for e in elig)} teams',flush=True)
     results=[]
     with ThreadPoolExecutor(max_workers=2) as pool:
      futures={pool.submit(one,r):r for r in plan.to_dict('records')}
      for i,f in enumerate(as_completed(futures),1):
       results.append(f.result())
       if i%25==0 or i==len(futures):
        m.atomic_csv(m.pd.DataFrame(results),out/'download_progress.csv');print(f'Absence profiles {i}/{len(plan)}, unavailable={sum(not v["complete"] for v in results)}',flush=True)
     print('Absence profile download complete',flush=True)
    finally:c.close()


# ---------------------------------------------------------------------
# HUB ABSENCE AND OFFENSIVE RESILIENCE: 2025-26 TEAM-SEASONS
# python teamgraphs.py --hub-absence-resilience
# ---------------------------------------------------------------------
def build_hub_absence_data(parent,out):
    """Reconstruct hub-present networks and adjusted with/without comparisons."""
    from nba_api.stats.endpoints import teamgamelogs
    from scipy.stats import t
    full=pd.read_csv(parent/'team_season_metrics.csv')
    games=pd.read_csv(parent/'team_games.csv',dtype={'GAME_ID':str})
    players=pd.read_csv(parent/'player_games.csv',dtype={'GAME_ID':str})
    plan=pd.read_csv(out/'download_plan.csv');elig=pd.read_csv(out/'eligibility.csv')
    client=NBAClient(Config(season='2025-26',cache_dir=out/'cache',timeout=30,retries=2,min_delay=.5,max_delay=1))
    try:
        advanced,_=client.fetch(teamgamelogs.TeamGameLogs,
            dict(season_nullable='2025-26',season_type_nullable='Regular Season',measure_type_player_game_logs_nullable='Advanced',per_mode_simple_nullable='Totals'),
            lambda p:season_topics_frame(p,'TeamGameLogs',['TEAM_ID','GAME_ID','POSS','OFF_RATING']))
    finally:client.close()
    advanced['GAME_ID']=advanced.GAME_ID.astype(str).str.zfill(10)
    if len(advanced)!=2460 or advanced.duplicated(['TEAM_ID','GAME_ID']).any():raise ValueError('Expected 2460 unique official advanced team-games')
    atomic_csv(advanced,out/'official_game_advanced.csv')
    games=games.merge(advanced[['TEAM_ID','GAME_ID','POSS','OFF_RATING']],on=['TEAM_ID','GAME_ID'],validate='one_to_one')
    if games.POSS.isna().any() or games.POSS.le(0).any():raise ValueError('Invalid possession totals')
    games['game_off_rating']=100*games.PTS/games.POSS
    if not np.allclose(games.game_off_rating,games.OFF_RATING,atol=.051,rtol=0):raise ValueError('Game ratings fail points/possessions validation')
    # Opponent defensive strength excludes the focal game to avoid using its outcome in its own control.
    opponent=games[['TEAM_ID','GAME_ID','PTS','POSS']].rename(columns={'TEAM_ID':'opponent_id','PTS':'opponent_points','POSS':'opponent_possessions'})
    games=games.merge(opponent,on='GAME_ID',validate='many_to_many');games=games.loc[games.TEAM_ID.ne(games.opponent_id)].copy()
    if len(games)!=2460 or games.duplicated(['TEAM_ID','GAME_ID']).any():raise ValueError('Opponent matching failed')
    allowed=games.groupby('TEAM_ID').opponent_points.sum();possessions=games.groupby('TEAM_ID').opponent_possessions.sum()
    games['opponent_def_rating_excluding_game']=100*(games.opponent_id.map(allowed)-games.PTS)/(games.opponent_id.map(possessions)-games.POSS)
    all_rows=[];team_rows=[];within_models=[];audit_rows=[]
    candidates=elig.loc[elig.eligible.astype(str).str.lower().eq('true')]
    for er in candidates.to_dict('records'):
        team=er['team'];team_id=int(er['team_id']);hub=int(er['hub_id'])
        season=full.loc[full.team.eq(team)].iloc[0]
        matrix_frame=pd.read_csv(parent/'networks'/team/'passing_matrix.csv',index_col=0)
        ids=matrix_frame.index.astype(int).tolist();idx={pid:i for i,pid in enumerate(ids)}
        full_matrix=matrix_frame.to_numpy(float);absent_matrix=np.zeros_like(full_matrix);missing=[];outside=[]
        for task in plan.loc[plan.team.eq(team)].to_dict('records'):
            folder=out/'absent_blocks'/team/(task['start']+'_'+task['end']);path=folder/(str(int(task['player_id']))+'.csv')
            audit=json.loads(path.with_suffix('.json').read_text());profile=pd.read_csv(path)
            if not audit['complete']:missing.append(audit)
            for row in profile.to_dict('records'):
                receiver=int(row['PASS_TEAMMATE_PLAYER_ID']);count=float(row['PASS'])
                if receiver in idx:absent_matrix[idx[int(task['player_id'])],idx[receiver]]+=count
                elif count>0:outside.append(dict(player_id=int(task['player_id']),receiver_id=receiver,passes=count))
        present_matrix=full_matrix-absent_matrix
        negative=np.argwhere(present_matrix<0)
        errors=[]
        if len(negative):errors.append('Absence-period edge counts exceed full-season counts; incompatible snapshots')
        if outside:errors.append('Positive passes to players absent from the official season roster')
        if missing:errors.append('Unavailable absence-period profiles prevent exact subtraction')
        coverage_complete=not errors
        # Unknown profile weights remain zero in saved subtraction; provisional networks excluded from primary models.
        if len(negative):present_matrix=np.maximum(present_matrix,0)
        ec,scores,active_ids,active=eigenvector_passing_concentration(present_matrix,ids)
        receiver,conditional=compute_team_entropies(active)
        folder=out/'present_networks'/team;folder.mkdir(parents=True,exist_ok=True)
        atomic_csv(pd.DataFrame(present_matrix,index=ids,columns=ids).rename_axis('passer_id'),folder/'passing_matrix.csv',index=True)
        atomic_csv(pd.DataFrame([dict(player_id=pid,eigenvector_share=scores.get(pid,0)) for pid in ids]),folder/'centrality.csv')
        atomic_json(dict(team=team,hub_id=hub,source='Full-season pair counts minus disjoint absence-period pair counts',
            absence_blocks=plan.loc[plan.team.eq(team),['start','end']].drop_duplicates().to_dict('records'),
            full_passes=float(full_matrix.sum()),absent_passes=float(absent_matrix.sum()),present_passes=float(present_matrix.sum()),
            missing_profiles=missing,outside_receivers=outside,negative_edge_count=len(negative),
            coverage_complete=coverage_complete,errors=errors),folder/'network_audit.json')
        tg=games.loc[games.TEAM_ID.eq(team_id)].sort_values(['GAME_DATE','GAME_ID']).copy().reset_index(drop=True)
        own_players=players.loc[players.TEAM_ID.eq(team_id)]
        hub_game_ids=set(own_players.loc[own_players.PLAYER_ID.eq(hub),'GAME_ID'])
        tg['hub_present']=tg.GAME_ID.isin(hub_game_ids).astype(int);tg['game_number']=np.arange(1,len(tg)+1)
        tg['home_game']=tg.MATCHUP.str.contains('vs.',regex=False).astype(int)
        tg['back_to_back']=pd.to_datetime(tg.GAME_DATE).diff().dt.days.eq(1).astype(int)
        stable_ids=set(players.groupby('PLAYER_ID').TEAM_ID.nunique().loc[lambda s:s.eq(1)].index)
        key=own_players.loc[own_players.PLAYER_ID.ne(hub)&own_players.PLAYER_ID.isin(stable_ids)].groupby('PLAYER_ID').agg(minutes=('MIN','sum'),appearances=('GAME_ID','size'))
        key=key.loc[key.appearances.ge(20)].sort_values(['minutes'],ascending=False).head(3)
        key_weights=key.minutes/key.minutes.sum() if len(key) else pd.Series(dtype=float)
        appeared=own_players.groupby('GAME_ID').PLAYER_ID.agg(set).to_dict()
        tg['other_key_absence']=tg.GAME_ID.map(lambda game:sum(float(w) for pid,w in key_weights.items() if pid not in appeared.get(game,set())))
        tg['team']=team;tg['hub_id']=hub;tg['hub_name']=er['hub_name']
        controls=['home_game','opponent_def_rating_excluding_game','other_key_absence','game_number','back_to_back']
        controls=[c for c in controls if tg[c].nunique()>1]
        fits=fit_eigenvector_performance_model(tg,'game_off_rating',['hub_present',*controls],'within_team_adjusted_presence')
        for fit in fits:fit.update(team=team,hub_name=er['hub_name'])
        within_models.extend(fits)
        coef=next((fit for fit in fits if fit['term']=='hub_present'),None)
        with_hub=tg.loc[tg.hub_present.eq(1)];without_hub=tg.loc[tg.hub_present.eq(0)]
        with_rating=100*with_hub.PTS.sum()/with_hub.POSS.sum();without_rating=100*without_hub.PTS.sum()/without_hub.POSS.sum()
        adjusted=coef['coefficient'] if coef else np.nan;se=coef['standard_error_hac7'] if coef else np.nan
        critical=t.ppf(.975,len(tg)-len(controls)-2)
        row=dict(team=team,team_id=team_id,hub_id=hub,hub_name=er['hub_name'],present_games=len(with_hub),absent_games=len(without_hub),
            hub_present_concentration=ec['eigenvector_concentration'],hub_present_conditional_entropy=conditional,
            hub_present_player_count=len(active_ids),hub_present_network_valid=ec['network_valid'],network_complete=coverage_complete,
            original_season_concentration=season.eigenvector_concentration,original_passing_style=season.passing_style,
            off_rating_with=with_rating,off_rating_without=without_rating,raw_absence_penalty=with_rating-without_rating,
            adjusted_absence_penalty=adjusted,penalty_se_hac7=se,penalty_ci95_low=adjusted-critical*se,penalty_ci95_high=adjusted+critical*se,
            within_model_p=coef['p_value_hac7'] if coef else np.nan,
            other_key_player_ids=';'.join(map(str,key.index)),network_errors='; '.join(errors),
            primary_eligible=bool(coverage_complete and ec['network_valid'] and np.isfinite(adjusted)))
        team_rows.append(row);all_rows.append(tg)
        audit_rows.append(dict(team=team,controls=controls,other_key_weights={str(pid):float(w) for pid,w in key_weights.items()},
            presence_definition='Official player-game appearance, including brief appearances; no inference about injury or absence cause'))
        print(f'{team}: {len(with_hub)} played / {len(without_hub)} missed; network complete={coverage_complete}; adjusted penalty={adjusted:.2f}',flush=True)
    metrics=pd.DataFrame(team_rows);game_frame=pd.concat(all_rows,ignore_index=True)
    atomic_csv(metrics,out/'team_season_metrics.csv');atomic_csv(game_frame,out/'game_analysis_data.csv')
    atomic_csv(pd.DataFrame(within_models),out/'within_team_models.csv');atomic_json(dict(teams=audit_rows),out/'control_definitions.json')
    return metrics,game_frame


def fit_hub_resilience_panel(games,teams,predictor,label):
    """One-stage team fixed effects with team-clustered finite-sample inference."""
    from scipy.stats import t
    valid=teams.loc[teams.primary_eligible.astype(str).str.lower().eq('true')].dropna(subset=[predictor])
    d=games.loc[games.team.isin(valid.team)].copy()
    d['centered_structure']=d.team.map(valid.set_index('team')[predictor])-valid[predictor].mean()
    d['presence_x_structure']=d.hub_present*d.centered_structure
    columns=['hub_present','presence_x_structure','home_game','opponent_def_rating_excluding_game','other_key_absence','game_number','back_to_back']
    columns=[c for c in columns if d[c].nunique()>1]
    fe=pd.get_dummies(d.team,drop_first=True,dtype=float)
    X=np.column_stack([np.ones(len(d)),d[columns].to_numpy(float),fe.to_numpy(float)])
    y=d.game_off_rating.to_numpy(float);n,k=X.shape;groups=d.team.unique();G=len(groups)
    if G<5 or np.linalg.matrix_rank(X)<k:return []
    inv=np.linalg.inv(X.T@X);b=np.linalg.lstsq(X,y,rcond=None)[0];e=y-X@b;meat=np.zeros((k,k))
    for group in groups:
        mask=d.team.eq(group).to_numpy();score=X[mask].T@e[mask];meat+=np.outer(score,score)
    cov=(G/(G-1))*((n-1)/(n-k))*inv@meat@inv;se=np.sqrt(np.maximum(np.diag(cov),0))
    critical=t.ppf(.975,G-1);p=2*t.sf(np.abs(np.divide(b,se,out=np.full_like(b,np.nan),where=se>0)),G-1)
    terms=['intercept',*columns,*['team_'+str(c) for c in fe.columns]]
    return [dict(model=label,structure=predictor,term=term,coefficient=float(b[j]),standard_error_cluster=float(se[j]),
        p_value_cluster=float(p[j]),ci95_low=float(b[j]-critical*se[j]),ci95_high=float(b[j]+critical*se[j]),
        team_clusters=G,game_observations=n) for j,term in enumerate(terms) if term in ['hub_present','presence_x_structure']]


def plot_hub_absence_resilience(frame,results,out):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from scipy.stats import linregress
    d=frame.loc[frame.primary_eligible.astype(str).str.lower().eq('true')].copy()
    specifications=[('concentration','hub_present_concentration','Passing concentration and hub-absence penalty',
        'Eigenvector concentration in hub-present games'),
        ('entropy','hub_present_conditional_entropy','Passing entropy and hub-absence penalty',
        'Conditional passing entropy in hub-present games')]
    legend=[Line2D([0],[0],ls='',marker='o',color='#2473b8',label='More balanced (original season label)'),
        Line2D([0],[0],ls='',marker='^',color='#e18a24',label='More star-centered (original season label)')]
    for key,x,title,xlabel in specifications:
        fig,ax=plt.subplots(figsize=(12,8))
        for style,color,marker in [('more_balanced','#2473b8','o'),('more_star_centered','#e18a24','^')]:
            g=d.loc[d.original_passing_style.eq(style)]
            ax.errorbar(g[x],g.adjusted_absence_penalty,yerr=[g.adjusted_absence_penalty-g.penalty_ci95_low,g.penalty_ci95_high-g.adjusted_absence_penalty],
                fmt='none',ecolor=color,alpha=.18,linewidth=1,zorder=1)
            ax.scatter(g[x],g.adjusted_absence_penalty,color=color,marker=marker,s=65,edgecolors='white',linewidths=.5,zorder=3)
            for _,r in g.iterrows():ax.annotate(r.team,(r[x],r.adjusted_absence_penalty),xytext=(4,4),textcoords='offset points',fontsize=8)
        if len(d)>=3:
            fit=linregress(d[x],d.adjusted_absence_penalty);xx=np.linspace(d[x].min(),d[x].max(),100)
            ax.plot(xx,fit.intercept+fit.slope*xx,c='#333333',lw=1.8)
        row=results.loc[results.model.eq('adjusted_penalty')&results.x.eq(x)&results.term.eq(x)]
        subtitle='Insufficient valid networks'
        if len(row):
            r=row.iloc[0];subtitle=f'N={r.n:.0f} teams | r={r.pearson_r:.3f} | R²={r.r_squared:.3f} | HC3 p={r.p_value_hc3:.3g}'
        fig.suptitle('NBA 2025–26: '+title,fontsize=15,y=.985)
        fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(.5,.94),ncol=2,frameon=False)
        ax.set_title(subtitle,fontsize=11,pad=14);ax.axhline(0,color='#777777',lw=1,ls='--')
        ax.set_xlabel(xlabel);ax.set_ylabel('Adjusted offensive-rating loss when hub misses game\n(points per 100 possessions; positive = worse without hub)')
        ax.grid(alpha=.16);ax.spines[['top','right']].set_visible(False)
        fig.text(.5,.018,'At least 20 appearances and 10 missed games. Controls: opponent defense, home court, other key absences, season progress and back-to-backs.\n'
            'Error bars: within-team 95% HAC(7) intervals. Absence means did not appear; this is an observational comparison.',ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.07,1,.88));season_topics_spread_labels(fig)
        fig.savefig(out/(key+'_absence_penalty.png'),dpi=180);fig.savefig(out/(key+'_absence_penalty.pdf'));plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,9));g=d.sort_values('adjusted_absence_penalty');yy=np.arange(len(g))
    ax.scatter(g.raw_absence_penalty,yy,color='#999999',marker='x',label='Unadjusted difference')
    ax.errorbar(g.adjusted_absence_penalty,yy,xerr=[g.adjusted_absence_penalty-g.penalty_ci95_low,g.penalty_ci95_high-g.adjusted_absence_penalty],
        fmt='o',color='#2473b8',capsize=3,label='Adjusted difference with 95% interval')
    ax.set_yticks(yy,[str(r.team)+' — '+str(r.hub_name) for _,r in g.iterrows()]);ax.axvline(0,color='#555555',ls='--',lw=1)
    ax.set_xlabel('Offensive-rating loss without hub (points per 100 possessions)');ax.grid(axis='x',alpha=.16)
    ax.set_title('2025–26 hub-absence comparisons: uncertainty for each team',pad=35);ax.legend(loc='lower center',bbox_to_anchor=(.5,1.01),ncol=2,frameon=False)
    fig.tight_layout();fig.savefig(out/'team_absence_penalties.png',dpi=180);fig.savefig(out/'team_absence_penalties.pdf');plt.close(fig)


def run_hub_absence_resilience(argv):
    parser=argparse.ArgumentParser(description='2025-26 passing-hub absence and team offensive resilience')
    parser.add_argument('--hub-absence-resilience',action='store_true');parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--plot-only',action='store_true');args=parser.parse_args(argv)
    parent=args.output_dir.expanduser()/'2025-26/team_season_topics';out=parent/'hub_absence_resilience';out.mkdir(parents=True,exist_ok=True)
    if args.plot_only:
        frame=pd.read_csv(out/'team_season_metrics.csv');games=pd.read_csv(out/'game_analysis_data.csv',dtype={'GAME_ID':str})
    else:
        download_hub_absence_profiles(parent,out)
        frame,games=build_hub_absence_data(parent,out)
    eligible=frame.loc[frame.primary_eligible.astype(str).str.lower().eq('true')].copy();rows=[]
    for topic,x in [('concentration','hub_present_concentration'),('entropy','hub_present_conditional_entropy')]:
        for y,label in [('adjusted_absence_penalty','adjusted_penalty'),('raw_absence_penalty','raw_penalty_sensitivity')]:
            rows.extend(season_topics_fit(eligible,x,y,label,topic))
        rows.extend(season_topics_fit(eligible.loc[eligible.absent_games.ge(15)],x,'adjusted_absence_penalty','at_least_15_absences_sensitivity',topic))
        rows.extend(season_topics_fit(eligible,x,'adjusted_absence_penalty','player_count_adjusted',topic,[x,'hub_present_player_count']))
    rows.extend(season_topics_fit(eligible,'hub_present_conditional_entropy','adjusted_absence_penalty','entropy_controlling_concentration','entropy',
        ['hub_present_conditional_entropy','hub_present_concentration']))
    results=pd.DataFrame(rows);results['p_holm_two_hc3']=np.nan
    primary=results.index[results.model.eq('adjusted_penalty')&results.term.ne('intercept')];order=results.loc[primary,'p_value_hc3'].sort_values().index;running=0
    for rank,idx in enumerate(order):
        running=max(running,min(1,float(results.loc[idx,'p_value_hc3'])*(len(order)-rank)));results.loc[idx,'p_holm_two_hc3']=running
    atomic_csv(results,out/'regression_results.csv')
    panel=[]
    for x,label in [('hub_present_concentration','concentration_panel_interaction'),('hub_present_conditional_entropy','entropy_panel_interaction')]:
        panel.extend(fit_hub_resilience_panel(games,frame,x,label))
    atomic_csv(pd.DataFrame(panel),out/'panel_clustered_results.csv')
    plot_hub_absence_resilience(frame,results,out)
    atomic_json(dict(season='2025-26',primary_research_question='Does passing concentration predict larger adjusted offensive-rating loss when the passing hub does not appear?',
        supporting_question='Does greater conditional passing entropy accompany smaller losses?',
        eligible_teams=len(eligible),candidate_teams=len(frame),eligibility='Hub highest full-season eigenvector share; >=20 appearances, >=10 missed games; hub not traded between NBA teams that season',
        predictor='Actual pairwise network restricted to hub-present games, reconstructed as full season minus disjoint hub-absence date intervals',
        positive_penalty='Team offense is worse without hub; negative means offense is better without hub',
        outcome='Adjusted within-team coefficient of hub appearance in points per 100 possessions; HAC(7) game-number covariance',
        controls='Home court, opponent defense excluding focal game, weighted absence of three other stable-team high-minute players, game number, back-to-back',
        inference='Cross-team OLS HC3; Holm across concentration and entropy slopes. One-stage team fixed-effects presence-by-structure interactions use team-clustered covariance as sensitivity.',
        no_minutes_cutoff=True,limitations=['Exploratory follow-up after testing other hypotheses','Observational absences, not randomized injuries; absence reasons not inferred',
            'Primary hub identified using season totals, so selection is retrospective','No appearance is different from on-court/off-court possessions',
            'Presence includes brief appearances and mid-game injuries','Other player availability and role changes are only partly controlled',
            'Networks and absence penalties are estimates; one-stage clustered model checks two-stage uncertainty',
            'Small eligible sample, especially for long absences; no guarantee of a strong relationship']),out/'analysis_audit.json')
    print(results.loc[primary,['topic','n','pearson_r','r_squared','p_value_hc3','p_holm_two_hc3']].to_string(index=False),flush=True)
    print('One-stage clustered interactions:',flush=True)
    print(pd.DataFrame(panel).loc[lambda d:d.term.eq('presence_x_structure')].to_string(index=False),flush=True)
    print(f'Finished hub-absence resilience analysis: {out}',flush=True)
    return 0


# ---------------------------------------------------------------------
# OPPONENT DEFENSE AND RECEIVING-DISTRIBUTION DISRUPTION, 2025-26
# python teamgraphs.py --defense-passing-disruption
# ---------------------------------------------------------------------
def prepare_defense_disruption(parent,out):
    """Leave-one-game-out receiving baselines, with matched minutes controls."""
    from nba_api.stats.endpoints import teamgamelogs
    official=pd.read_csv(parent/'team_games.csv',dtype={'GAME_ID':str})
    players=pd.read_csv(parent/'player_games.csv',dtype={'GAME_ID':str})
    season=pd.read_csv(parent/'team_season_metrics.csv')
    advanced_path=parent/'hub_absence_resilience/official_game_advanced.csv'
    if advanced_path.exists():advanced=pd.read_csv(advanced_path,dtype={'GAME_ID':str})
    else:
        c=NBAClient(Config(season='2025-26',cache_dir=out/'cache',timeout=30,retries=2,min_delay=.5,max_delay=1))
        try:
            advanced,_=c.fetch(teamgamelogs.TeamGameLogs,dict(season_nullable='2025-26',season_type_nullable='Regular Season',
                measure_type_player_game_logs_nullable='Advanced',per_mode_simple_nullable='Totals'),
                lambda p:season_topics_frame(p,'TeamGameLogs',['TEAM_ID','GAME_ID','POSS','OFF_RATING']))
        finally:c.close()
    advanced['GAME_ID']=advanced.GAME_ID.astype(str).str.zfill(10)
    if len(official)!=2460 or len(advanced)!=2460 or official.duplicated(['TEAM_ID','GAME_ID']).any() or advanced.duplicated(['TEAM_ID','GAME_ID']).any():
        raise ValueError('Expected all 2460 unique team-games')
    if players.duplicated(['TEAM_ID','GAME_ID','PLAYER_ID']).any():raise ValueError('Duplicate official player-game rows')
    games=official.merge(advanced[['TEAM_ID','GAME_ID','POSS']],on=['TEAM_ID','GAME_ID'],validate='one_to_one')
    opp=games[['TEAM_ID','GAME_ID','PTS','POSS']].rename(columns={'TEAM_ID':'opponent_id','PTS':'opponent_points','POSS':'opponent_possessions'})
    games=games.merge(opp,on='GAME_ID');games=games.loc[games.TEAM_ID.ne(games.opponent_id)].copy()
    if len(games)!=2460 or games.duplicated(['TEAM_ID','GAME_ID']).any():raise ValueError('Opponent matching failed')
    allowed=games.groupby('TEAM_ID').opponent_points.sum();defposs=games.groupby('TEAM_ID').opponent_possessions.sum()
    league_rating=100*games.PTS.sum()/games.POSS.sum()
    games['opponent_def_rating_loo']=100*(games.opponent_id.map(allowed)-games.PTS)/(games.opponent_id.map(defposs)-games.POSS)
    games['opponent_strength']=league_rating-games.opponent_def_rating_loo
    # Pregame sensitivity uses only earlier dates and requires 15 prior opponent games.
    history={}
    for date,day in games.groupby('GAME_DATE',sort=True):
        before=games.loc[games.GAME_DATE.lt(date)];lg=100*before.PTS.sum()/before.POSS.sum() if len(before) else np.nan
        prior=before.groupby('TEAM_ID').agg(allowed=('opponent_points','sum'),poss=('opponent_possessions','sum'),gp=('GAME_ID','size'))
        for _,row in day.iterrows():
            pid=row.opponent_id
            if pid in prior.index:
                v=prior.loc[pid];history[(int(row.TEAM_ID),row.GAME_ID)]=(lg-100*v.allowed/v.poss,int(v.gp))
            else:history[(int(row.TEAM_ID),row.GAME_ID)]=(np.nan,0)
    games['pregame_opponent_strength']=[history[(int(r.TEAM_ID),r.GAME_ID)][0] for _,r in games.iterrows()]
    games['opponent_prior_games']=[history[(int(r.TEAM_ID),r.GAME_ID)][1] for _,r in games.iterrows()]
    records=[];references=[];daily_cache={}
    for team_id,tg in games.groupby('TEAM_ID'):
        tg=tg.sort_values(['GAME_DATE','GAME_ID']).reset_index(drop=True)
        team=str(tg.TEAM_ABBREVIATION.iloc[0]);tp=players.loc[players.TEAM_ID.eq(team_id)]
        ids=sorted(tp.PLAYER_ID.astype(int).unique());index={pid:i for i,pid in enumerate(ids)}
        n=len(ids);R=np.zeros((len(tg),n));M=np.zeros_like(R);A=np.zeros_like(R,dtype=bool);valid=np.zeros(len(tg),dtype=bool);missing=[];samples=[]
        for i,row in tg.iterrows():
            date=str(row.GAME_DATE)[:10]
            if date not in daily_cache:daily_cache[date]=pd.read_csv(parent/'daily_passing'/(date+'.csv'))
            tracking=daily_cache[date].loc[daily_cache[date].TEAM_ID.eq(team_id)]
            roster=tp.loc[tp.GAME_ID.eq(row.GAME_ID)]
            known=set(tracking.PLAYER_ID.astype(int));appeared=set(roster.PLAYER_ID.astype(int));unknown=appeared-known
            if tracking.PLAYER_ID.duplicated().any():raise ValueError('Duplicate tracked player within game')
            if known-appeared:raise ValueError('Receiving records contain a nonparticipant after historical team alignment')
            for _,v in roster.iterrows():
                M[i,index[int(v.PLAYER_ID)]]=float(v.MIN);A[i,index[int(v.PLAYER_ID)]]=True
            for _,v in tracking.iterrows():R[i,index[int(v.PLAYER_ID)]]=float(v.PASSES_RECEIVED)
            if not np.isfinite(R[i]).all() or (R[i]<0).any() or not np.isfinite(M[i]).all() or (M[i]<0).any():raise ValueError('Invalid receiving/minutes values')
            valid[i]=bool(not unknown and R[i].sum()>0 and M[i].sum()>0)
            missing.append(';'.join(map(str,sorted(unknown))))
        sumR=R[valid].sum(axis=0);sumM=M[valid].sum(axis=0)
        for j,pid in enumerate(ids):references.append(dict(team=team,player_id=pid,valid_game_passes_received=sumR[j],valid_game_minutes=sumM[j]))
        for i,row in tg.iterrows():
            data=row.to_dict();data.update(team=team,game_number=i+1,home_game=int('vs.' in str(row.MATCHUP)),
                back_to_back=int(i>0 and (pd.Timestamp(row.GAME_DATE)-pd.Timestamp(tg.iloc[i-1].GAME_DATE)).days==1),
                receiving_coverage=bool(valid[i]),missing_player_ids=missing[i],receiving_disruption=np.nan,minutes_disruption=np.nan,
                usual_minutes_absent_share=np.nan,minutes_expected_receiving_jsd=np.nan,history_receiving_coverage=np.nan,
                zero_minute_positive_receiver=False,tracked_passes_received=float(R[i].sum()),reference_games=int(valid.sum()-int(valid[i])))
            if valid[i]:
                baseR=sumR-R[i];baseM=sumM-M[i]
                data['receiving_disruption']=receiving_share_js_divergence(R[i],baseR)
                data['minutes_disruption']=receiving_share_js_divergence(M[i],baseM)
                data['usual_minutes_absent_share']=float(baseM[~A[i]].sum()/baseM.sum())
                known_history=(baseM>0)&(M[i]>0)
                share=float(R[i,known_history].sum()/R[i].sum());data['history_receiving_coverage']=share
                data['zero_minute_positive_receiver']=bool(((M[i]==0)&(R[i]>0)).any())
                expected=np.zeros(n);expected[known_history]=M[i,known_history]*baseR[known_history]/baseM[known_history]
                if share>=.9 and known_history.sum()>=5 and R[i,known_history].sum()>0 and expected.sum()>0 and not data['zero_minute_positive_receiver']:
                    data['minutes_expected_receiving_jsd']=receiving_share_js_divergence(R[i,known_history],expected[known_history])
            records.append(data)
    frame=pd.DataFrame(records)
    classification=season.set_index('team')
    frame['passing_style']=frame.team.map(classification.passing_style)
    frame['eigenvector_concentration']=frame.team.map(classification.eigenvector_concentration)
    frame['star_indicator']=frame.passing_style.eq('more_star_centered').astype(int)
    frame['strength_x_star']=frame.opponent_strength*frame.star_indicator
    frame['pregame_strength_x_star']=frame.pregame_opponent_strength*frame.star_indicator
    frame['log_receiving_volume']=np.log(frame.tracked_passes_received.where(frame.tracked_passes_received.gt(0)))
    # Existing adjacent-game records are joined only for exact consecutive pairs, never bridged.
    pair_path=parent/'five_additional_topics/minutes_and_receiving_game_pairs.csv'
    if pair_path.exists():
        pairs=pd.read_csv(pair_path,dtype={'game_id':str,'previous_game_id':str})
        pairs['game_id']=pairs.game_id.str.zfill(10);pairs['previous_game_id']=pairs.previous_game_id.str.zfill(10)
        keep=pairs[['team','game_id','previous_game_id','comparison_available','js_divergence','minutes_jsd','player_turnover_fraction']]
        frame=frame.merge(keep,left_on=['team','GAME_ID'],right_on=['team','game_id'],how='left',validate='one_to_one')
        prior_strength=frame.set_index(['team','GAME_ID']).opponent_strength.to_dict()
        frame['previous_opponent_strength']=[prior_strength.get((r.team,str(r.previous_game_id)),np.nan) for _,r in frame.iterrows()]
    atomic_csv(frame,out/'game_metrics.csv');atomic_csv(pd.DataFrame(references),out/'season_reference_counts.csv')
    return frame


def fit_defense_disruption(frame,predictors,label,outcome='receiving_disruption'):
    """Team fixed effects; two-way team and game clustering, t df=min(G)-1."""
    from scipy.stats import t
    d=frame.replace([np.inf,-np.inf],np.nan).dropna(subset=[outcome,*predictors]).copy()
    predictors=[c for c in predictors if d[c].nunique()>1]
    fe=pd.get_dummies(d.team,drop_first=True,dtype=float)
    X=np.column_stack([np.ones(len(d)),d[predictors].to_numpy(float),fe.to_numpy(float)])
    y=d[outcome].to_numpy(float);n,k=X.shape
    if n<=k or np.linalg.matrix_rank(X)<k:raise ValueError('Insufficient or collinear defense-disruption model '+label)
    inv=np.linalg.inv(X.T@X);b=np.linalg.lstsq(X,y,rcond=None)[0];e=y-X@b
    def cluster_meat(groups):
        meat=np.zeros((k,k));unique=pd.unique(groups);G=len(unique)
        for group in unique:
            mask=np.asarray(groups==group);score=X[mask].T@e[mask];meat+=np.outer(score,score)
        return G/(G-1)*(n-1)/(n-k)*meat,G
    a,G=cluster_meat(d.team.to_numpy());c,H=cluster_meat(d.GAME_ID.to_numpy())
    individual=X*e[:,None];subtract=n/(n-k)*(individual.T@individual)
    covariance=inv@(a+c-subtract)@inv;diag=np.diag(covariance)
    if (diag[:len(predictors)+1]<-1e-12).any():raise ValueError('Negative two-way clustered coefficient variance')
    se=np.sqrt(np.maximum(diag,0));df=min(G,H)-1;critical=t.ppf(.975,df)
    p=2*t.sf(np.abs(np.divide(b,se,out=np.full_like(b,np.nan),where=se>0)),df)
    total_r2=1-float(e@e)/float(np.sum((y-y.mean())**2));rows=[]
    for j,term in enumerate(['intercept',*predictors]):
        reduced=np.delete(X,j,axis=1);yr=y-reduced@np.linalg.lstsq(reduced,y,rcond=None)[0]
        xr=X[:,j]-reduced@np.linalg.lstsq(reduced,X[:,j],rcond=None)[0]
        corr=float(np.corrcoef(xr,yr)[0,1]) if j>0 and np.std(xr)>0 else np.nan
        rows.append(dict(model=label,outcome=outcome,term=term,coefficient=float(b[j]),standard_error_twoway=float(se[j]),
            p_value_twoway=float(p[j]),ci95_low=float(b[j]-critical*se[j]),ci95_high=float(b[j]+critical*se[j]),
            partial_r=corr,partial_r_squared=corr**2,full_model_r_squared=total_r2,n=n,teams=G,games=H,df=df,
            condition_number=float(np.linalg.cond(X))))
    combinations=[]
    if 'opponent_strength' in predictors and 'strength_x_star' in predictors:
        j=predictors.index('opponent_strength')+1;l=predictors.index('strength_x_star')+1
        for name,weights in [('balanced_slope',{j:1}),('star_centered_slope',{j:1,l:1}),('slope_difference',{l:1})]:
            w=np.zeros(k)
            for pos,value in weights.items():w[pos]=value
            estimate=float(w@b);error=float(np.sqrt(max(0,w@covariance@w)))
            combinations.append(dict(model=label,comparison=name,coefficient=estimate,standard_error_twoway=error,
                p_value_twoway=float(2*t.sf(abs(estimate/error),df)) if error>0 else np.nan,
                ci95_low=estimate-critical*error,ci95_high=estimate+critical*error,n=n,teams=G,df=df))
    return rows,combinations


DEFENSE_CONTROLS=['minutes_disruption','usual_minutes_absent_share','home_game','back_to_back','game_number']


def plot_defense_disruption(frame,results,groups,out):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from scipy.stats import linregress
    d=frame.loc[frame.receiving_coverage.astype(str).str.lower().eq('true')].dropna(subset=['receiving_disruption',*DEFENSE_CONTROLS,'opponent_strength']).copy()
    # Frisch-Waugh-Lovell residuals show exactly the adjusted pooled coefficient.
    fe=pd.get_dummies(d.team,drop_first=True,dtype=float)
    Z=np.column_stack([np.ones(len(d)),d[DEFENSE_CONTROLS].to_numpy(float),fe.to_numpy(float)])
    x=d.opponent_strength.to_numpy();y=d.receiving_disruption.to_numpy()
    d['adjusted_strength']=x-Z@np.linalg.lstsq(Z,x,rcond=None)[0]
    d['adjusted_disruption']=y-Z@np.linalg.lstsq(Z,y,rcond=None)[0]
    atomic_csv(d[['team','GAME_ID','passing_style','adjusted_strength','adjusted_disruption']],out/'adjusted_plot_data.csv')
    fig,axes=plt.subplots(1,2,figsize=(16,7.5))
    legend=[]
    for ax,xc,yc,title in [(axes[0],'opponent_strength','receiving_disruption','Before adjusting for minutes and team context'),
        (axes[1],'adjusted_strength','adjusted_disruption','After adjusting for minutes and team context')]:
        for style,color,marker,label in [('more_balanced','#2473b8','o','More balanced'),('more_star_centered','#e18a24','^','More star-centered')]:
            s=d.loc[d.passing_style.eq(style)];ax.scatter(s[xc],s[yc],c=color,marker=marker,s=12,alpha=.17,edgecolors='none',rasterized=True)
        fit=linregress(d[xc],d[yc]);xx=np.linspace(d[xc].min(),d[xc].max(),100);ax.plot(xx,fit.intercept+fit.slope*xx,color='#333333',lw=2)
        bins=pd.qcut(d[xc],10,duplicates='drop');means=d.groupby(bins,observed=True)[[xc,yc]].mean()
        ax.scatter(means[xc],means[yc],color='#111111',s=28,zorder=4)
        ax.set_title(title+'\n'+f'r={fit.rvalue:.3f} | R²={fit.rvalue**2:.4f}',fontsize=11)
        ax.set_xlabel('Opponent defensive strength (higher = stronger)' if xc=='opponent_strength' else 'Opponent strength after removing controls')
        ax.set_ylabel('Receiving-share JSD from usual pattern' if yc=='receiving_disruption' else 'Receiving disruption above/below expected')
        ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False)
    primary=results.loc[results.model.eq('primary_pooled')&results.term.eq('opponent_strength')].iloc[0]
    fig.suptitle('NBA 2025–26: do stronger defenses disrupt receiving patterns?',fontsize=16,y=.99)
    legend=[Line2D([0],[0],marker='o',color='#2473b8',ls='',label='More balanced'),Line2D([0],[0],marker='^',color='#e18a24',ls='',label='More star-centered'),
        Line2D([0],[0],marker='o',color='#111111',ls='',label='Means across opponent-strength bins')]
    fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(.5,.94),ncol=3,frameon=False)
    fig.text(.5,.045,f'N={len(d):,} team-games, 30 teams. Adjusted slope p={primary.p_value_twoway:.3g}; errors clustered by team and game.\n'
        'Reference: other complete games in the same team-season. Controls include minutes redistribution, absent-player share, home court, back-to-backs and season progress.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.11,1,.86));fig.savefig(out/'defense_receiving_disruption.png',dpi=180)
    fig.savefig(out/'defense_receiving_disruption.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,6))
    labels={'balanced_slope':'More balanced','star_centered_slope':'More star-centered','slope_difference':'Difference: star-centered minus balanced'}
    g=groups.loc[groups.model.eq('primary_style_interaction')].copy();yy=np.arange(len(g))
    ax.errorbar(g.coefficient,yy,xerr=[g.coefficient-g.ci95_low,g.ci95_high-g.coefficient],fmt='o',capsize=4,color='#2473b8')
    ax.set_yticks(yy,[labels[v] for v in g.comparison]);ax.axvline(0,color='#555555',ls='--');ax.grid(axis='x',alpha=.15)
    ax.set_xlabel('Change in receiving-share JSD per point of stronger opponent defense')
    ax.set_title('2025–26: adjusted relationships by passing structure\n95% intervals with team-and-game clustered errors',pad=16)
    fig.tight_layout();fig.savefig(out/'passing_style_slope_comparison.png',dpi=180);fig.savefig(out/'passing_style_slope_comparison.pdf');plt.close(fig)


def run_defense_disruption(argv):
    parser=argparse.ArgumentParser(description='Opponent defense vs receiving-pattern disruption, all 30 NBA teams in 2025-26')
    parser.add_argument('--defense-passing-disruption',action='store_true');parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--plot-only',action='store_true');args=parser.parse_args(argv)
    parent=args.output_dir.expanduser()/'2025-26/team_season_topics';out=parent/'defense_passing_disruption';out.mkdir(parents=True,exist_ok=True)
    frame=pd.read_csv(out/'game_metrics.csv',dtype={'GAME_ID':str,'previous_game_id':str}) if args.plot_only else prepare_defense_disruption(parent,out)
    sample=frame.loc[frame.receiving_coverage.astype(str).str.lower().eq('true')].copy();rows=[];comparisons=[]
    specifications=[(sample,['opponent_strength',*DEFENSE_CONTROLS],'primary_pooled','receiving_disruption'),
        (sample,['opponent_strength','strength_x_star',*DEFENSE_CONTROLS],'primary_style_interaction','receiving_disruption'),
        (sample,['opponent_strength',*DEFENSE_CONTROLS,'log_receiving_volume'],'receiving_volume_sensitivity','receiving_disruption'),
        (sample.loc[sample.opponent_prior_games.ge(15)],['pregame_opponent_strength',*DEFENSE_CONTROLS],'pregame_strength_sensitivity','receiving_disruption'),
        (sample,['opponent_strength',*DEFENSE_CONTROLS],'minutes_expected_pattern_sensitivity','minutes_expected_receiving_jsd')]
    if 'js_divergence' in frame:
        consecutive=sample.loc[sample.comparison_available.astype(str).str.lower().eq('true')].copy()
        specifications.append((consecutive,['opponent_strength','previous_opponent_strength','minutes_jsd','player_turnover_fraction','home_game','back_to_back','game_number'],
            'previous_game_pattern_sensitivity','js_divergence'))
    for data,predictors,label,outcome in specifications:
        result,group=fit_defense_disruption(data,predictors,label,outcome);rows.extend(result);comparisons.extend(group)
    results=pd.DataFrame(rows);results['p_holm_two']=np.nan
    primary=[results.index[results.model.eq('primary_pooled')&results.term.eq('opponent_strength')][0],
        results.index[results.model.eq('primary_style_interaction')&results.term.eq('strength_x_star')][0]]
    running=0
    for rank,idx in enumerate(results.loc[primary,'p_value_twoway'].sort_values().index):
        running=max(running,min(1,float(results.loc[idx,'p_value_twoway'])*(2-rank)));results.loc[idx,'p_holm_two']=running
    groups=pd.DataFrame(comparisons);atomic_csv(results,out/'regression_results.csv');atomic_csv(groups,out/'style_comparisons.csv')
    summary=sample.groupby(['team','passing_style'],as_index=False).agg(valid_games=('GAME_ID','size'),mean_disruption=('receiving_disruption','mean'),
        mean_minutes_disruption=('minutes_disruption','mean'),mean_opponent_strength=('opponent_strength','mean'))
    atomic_csv(summary,out/'team_summary.csv');plot_defense_disruption(frame,results,groups,out)
    atomic_json(dict(season='2025-26',teams=frame.team.nunique(),team_games=len(frame),valid_games=len(sample),
        excluded_games=frame.loc[~frame.receiving_coverage.astype(str).str.lower().eq('true'),['team','GAME_ID','missing_player_ids']].to_dict('records'),
        primary_outcome='Base-2 JSD between game receiving shares and team-season receiving shares excluding that game; only complete games enter references',
        primary_predictor='League scoring rate minus opponent points allowed per 100 defensive possessions, excluding focal game; higher means stronger defense',
        controls=DEFENSE_CONTROLS,fixed_effects='Offensive team intercepts',
        uncertainty='Two-way team and match clustering with finite-sample corrections and t(df=29); Holm across primary opponent slope and style interaction',
        style='Fixed 15/15 league-median eigenvector concentration labels from season networks',
        sensitivities=['Pregame opponent defense with >=15 prior games','Receiving volume control','Minutes-adjusted expected receiving pattern',
            'Consecutive-game reference with previous opponent strength and no bridging unavailable games'],
        no_minutes_cutoff=True,limitations=['Observational exploratory study; not a randomized test of defensive pressure',
            'Receiving shares describe player involvement, not pass order, spatial pressure or complete pairwise passing networks',
            'Season reference and style labels use retrospective data; pregame-strength sensitivity addresses predictor look-ahead only',
            'Season-level defense can reflect schedule and personnel, not the exact defensive scheme faced that night',
            'Minutes are rounded official game-log values','Potentially important injuries, lineups and tactical changes remain unmeasured']),out/'analysis_audit.json')
    print(results.loc[primary,['model','term','n','teams','coefficient','partial_r','partial_r_squared','p_value_twoway','p_holm_two']].to_string(index=False),flush=True)
    print(groups.to_string(index=False),flush=True);print(f'Finished defense disruption analysis: {out}',flush=True)
    return 0


# ---------------------------------------------------------------------
# HUB-ABSENCE REDISTRIBUTION AND WITHIN-SEASON FAMILIARITY, 2025-26
# python teamgraphs.py --roles-and-familiarity
# ---------------------------------------------------------------------
def build_roles_familiarity_data(parent,out):
    context=pd.read_csv(parent/'defense_passing_disruption/game_metrics.csv',dtype={'GAME_ID':str})
    players=pd.read_csv(parent/'player_games.csv',dtype={'GAME_ID':str})
    hub_metrics=pd.read_csv(parent/'hub_absence_resilience/team_season_metrics.csv')
    hub_context=pd.read_csv(parent/'hub_absence_resilience/game_analysis_data.csv',dtype={'GAME_ID':str})
    daily=pd.concat([pd.read_csv(f).assign(game_date=f.stem) for f in (parent/'daily_passing').glob('*.csv')],ignore_index=True)
    for c in ['PASSES_MADE','PASSES_RECEIVED','POTENTIAL_AST']:
        if not np.isfinite(daily[c]).all() or daily[c].lt(0).any():raise ValueError('Unavailable or negative passing totals '+c)
    lookup={(int(t),str(d)):g.set_index('PLAYER_ID') for (t,d),g in daily.groupby(['TEAM_ID','game_date'])}
    player_lookup={(int(t),str(g)):v.set_index('PLAYER_ID') for (t,g),v in players.groupby(['TEAM_ID','GAME_ID'])}
    familiarity=[];core_games=[];core_definitions=[]
    for team_id,tg in context.groupby('TEAM_ID'):
        tg=tg.sort_values(['GAME_DATE','GAME_ID']).copy().reset_index(drop=True)
        team=str(tg.team.iloc[0]);roster_ids=sorted(players.loc[players.TEAM_ID.eq(team_id),'PLAYER_ID'].astype(int).unique())
        index={pid:i for i,pid in enumerate(roster_ids)};n=len(roster_ids)
        shared=np.zeros((n,n),dtype=int)
        for i,row in tg.iterrows():
            pg=player_lookup[(int(team_id),row.GAME_ID)];ids=pg.index.astype(int).tolist();ix=[index[p] for p in ids]
            minutes=pg.MIN.to_numpy(float);pair_weights=np.outer(minutes,minutes);upper=np.triu_indices(len(ids),1)
            counts=shared[np.ix_(ix,ix)][upper];weights=pair_weights[upper]
            weighted=float(np.average(counts,weights=weights)) if len(counts) and weights.sum()>0 else np.nan
            equal=float(np.mean(counts)) if len(counts) else np.nan
            data=row.to_dict();tracking=lookup[(int(team_id),str(row.GAME_DATE)[:10])]
            complete=bool(row.receiving_coverage and set(ids).issubset(set(tracking.index)))
            passes=float(tracking.PASSES_MADE.sum());potential=float(tracking.POTENTIAL_AST.sum())
            errors=[]
            if not complete:errors.append('Incomplete player receiving coverage')
            if passes<=0:errors.append('No positive passing denominator')
            if potential>passes or tracking.POTENTIAL_AST.gt(tracking.PASSES_MADE).any():errors.append('Potential assists exceed passes made')
            complete=bool(complete and not errors)
            data.update(prior_team_games=i,prior_shared_games=weighted,prior_shared_games_equal_weight=equal,
                prior_shared_game_fraction=weighted/i if i>0 else np.nan,prior_pair_count=len(counts),
                largest_prior_pair_count=int(max(counts)) if len(counts) else 0,
                passes_made=passes,potential_assists=potential,potential_assists_per100passes=100*potential/passes if complete and passes>0 else np.nan,
                passing_complete=complete,passing_data_error='; '.join(errors),current_player_count=len(ids))
            data['familiarity_x_star']=weighted*int(row.star_indicator) if np.isfinite(weighted) else np.nan
            familiarity.append(data)
            # Update only AFTER measuring the current game's predictor: no current/future appearances enter history.
            shared[np.ix_(ix,ix)]+=1
        selected=hub_metrics.loc[hub_metrics.team.eq(team)]
        if selected.empty:continue
        h=selected.iloc[0];hub=int(h.hub_id)
        present_games=set(players.loc[players.TEAM_ID.eq(team_id)&players.PLAYER_ID.eq(hub),'GAME_ID'])
        good=tg.loc[tg.receiving_coverage.astype(str).str.lower().eq('true')]
        with_ids=set(players.loc[players.TEAM_ID.eq(team_id)&players.GAME_ID.isin(good.loc[good.GAME_ID.isin(present_games),'GAME_ID']),'PLAYER_ID'])
        without_ids=set(players.loc[players.TEAM_ID.eq(team_id)&players.GAME_ID.isin(good.loc[~good.GAME_ID.isin(present_games),'GAME_ID']),'PLAYER_ID'])
        core=sorted((with_ids&without_ids)-{hub})
        if len(core)<5:raise ValueError('Fewer than five common non-hub players for '+team)
        core_definitions.append(dict(team=team,hub_id=hub,hub_name=h.hub_name,core_player_count=len(core),core_player_ids=';'.join(map(str,core)),
            definition='Same non-hub player set in both conditions; each appeared in at least one complete game with and without hub'))
        usual_minutes=players.loc[players.TEAM_ID.eq(team_id)&players.GAME_ID.isin(present_games)&players.PLAYER_ID.isin(core)].groupby('PLAYER_ID').MIN.sum().reindex(core,fill_value=0)
        hc=hub_context.loc[hub_context.team.eq(team)].set_index('GAME_ID')
        for _,row in good.iterrows():
            pg=player_lookup[(int(team_id),row.GAME_ID)];tracking=lookup[(int(team_id),str(row.GAME_DATE)[:10])]
            rec=tracking.PASSES_RECEIVED.reindex(core,fill_value=0).to_numpy(float)
            mins=pg.MIN.reindex(core,fill_value=0).to_numpy(float)
            receiver_hhi,raw_hhi=normalized_share_concentration(rec);minutes_hhi,_=normalized_share_concentration(mins)
            if not np.isfinite(receiver_hhi) or not np.isfinite(minutes_hhi):continue
            shares=rec/rec.sum();positive=shares>0;entropy=float(-np.sum(shares[positive]*np.log(shares[positive]))/np.log(len(core)))
            nonhub_total=float(tracking.loc[tracking.index!=hub,'PASSES_RECEIVED'].sum())
            core_active=sum(pid in pg.index for pid in core)
            data=row.to_dict();data.update(hub_id=hub,hub_name=h.hub_name,hub_absent=int(row.GAME_ID not in present_games),
                core_player_count=len(core),core_active_count=core_active,core_receiving_hhi=receiver_hhi,core_raw_hhi=raw_hhi,
                core_receiving_entropy=entropy,core_minutes_hhi=minutes_hhi,
                core_nonhub_receiving_coverage=float(rec.sum()/nonhub_total) if nonhub_total>0 else np.nan,
                core_usual_minutes_absent_share=float(usual_minutes.loc[~usual_minutes.index.isin(pg.index)].sum()/usual_minutes.sum()),
                other_key_absence=float(hc.loc[row.GAME_ID,'other_key_absence']),
                hub_present_concentration=float(h.hub_present_concentration))
            data['absence_x_concentration']=data['hub_absent']*data['hub_present_concentration']
            data['absence_x_star']=data['hub_absent']*int(row.star_indicator)
            core_games.append(data)
    fam=pd.DataFrame(familiarity);core=pd.DataFrame(core_games)
    atomic_csv(fam,out/'familiarity_game_metrics.csv');atomic_csv(core,out/'redistribution_game_metrics.csv')
    atomic_csv(pd.DataFrame(core_definitions),out/'common_player_panels.csv')
    return fam,core


ROLE_CONTROLS=['core_minutes_hhi','core_usual_minutes_absent_share','other_key_absence','home_game','opponent_strength','game_number','back_to_back']
FAMILIARITY_CONTROLS=['opponent_strength','minutes_disruption','usual_minutes_absent_share','home_game','back_to_back','game_number','current_player_count']


def summarize_hub_redistribution(core,out):
    from scipy.stats import t
    rows=[];within=[]
    for team,g in core.groupby('team'):
        present=g.loc[g.hub_absent.eq(0)];absent=g.loc[g.hub_absent.eq(1)]
        controls=[c for c in ROLE_CONTROLS if g[c].nunique()>1]
        fits=fit_eigenvector_performance_model(g,'core_receiving_hhi',['hub_absent',*controls],'remaining_player_concentration')
        for r in fits:r['team']=team
        within.extend(fits);coef=next((r for r in fits if r['term']=='hub_absent'),None)
        critical=t.ppf(.975,len(g)-len(controls)-2)
        first=g.iloc[0];adjusted=coef['coefficient'] if coef else np.nan;se=coef['standard_error_hac7'] if coef else np.nan
        rows.append(dict(team=team,hub_name=first.hub_name,hub_id=int(first.hub_id),
            hub_present_concentration=first.hub_present_concentration,passing_style=first.passing_style,star_indicator=int(first.star_indicator),
            core_player_count=int(first.core_player_count),complete_with_hub=len(present),complete_without_hub=len(absent),
            mean_hhi_with=present.core_receiving_hhi.mean(),mean_hhi_without=absent.core_receiving_hhi.mean(),
            raw_hhi_change=absent.core_receiving_hhi.mean()-present.core_receiving_hhi.mean(),
            adjusted_hhi_change=adjusted,change_se_hac7=se,ci95_low=adjusted-critical*se,ci95_high=adjusted+critical*se,
            mean_common_panel_coverage=g.core_nonhub_receiving_coverage.mean(),
            primary_eligible=bool(len(present)>=20 and len(absent)>=10 and coef is not None)))
    summary=pd.DataFrame(rows);atomic_csv(summary,out/'redistribution_team_metrics.csv');atomic_csv(pd.DataFrame(within),out/'redistribution_within_team_models.csv')
    return summary


def plot_roles_familiarity(summary,fam,cross_results,fam_results,out):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from scipy.stats import linregress
    fig,axes=plt.subplots(1,2,figsize=(16,7.5));d=summary.loc[summary.primary_eligible].copy()
    for style,color,marker in [('more_balanced','#2473b8','o'),('more_star_centered','#e18a24','^')]:
        g=d.loc[d.passing_style.eq(style)]
        axes[0].errorbar(g.hub_present_concentration,g.adjusted_hhi_change,yerr=[g.adjusted_hhi_change-g.ci95_low,g.ci95_high-g.adjusted_hhi_change],fmt='none',ecolor=color,alpha=.2)
        axes[0].scatter(g.hub_present_concentration,g.adjusted_hhi_change,color=color,marker=marker,s=55,zorder=3)
        for _,r in g.iterrows():axes[0].annotate(r.team,(r.hub_present_concentration,r.adjusted_hhi_change),xytext=(4,4),textcoords='offset points',fontsize=8)
    fit=linregress(d.hub_present_concentration,d.adjusted_hhi_change);xx=np.linspace(d.hub_present_concentration.min(),d.hub_present_concentration.max(),100)
    axes[0].plot(xx,fit.intercept+fit.slope*xx,color='#333333');axes[0].axhline(0,color='#888888',ls='--',lw=1)
    row=cross_results.loc[cross_results.model.eq('redistribution_primary')&cross_results.term.eq('hub_present_concentration')].iloc[0]
    axes[0].set_title('2. Hub absence and redistribution among remaining players\n'+f'N={len(d)} teams | r={row.pearson_r:.3f} | R²={row.r_squared:.3f} | HC3 p={row.p_value_hc3:.3g}',fontsize=11)
    axes[0].set_xlabel('Passing concentration in hub-present games');axes[0].set_ylabel('Adjusted change in remaining-player receiving concentration\n(positive = less evenly shared without hub)')
    s=fam.loc[fam.passing_complete.astype(str).str.lower().eq('true')&fam.prior_team_games.ge(10)].dropna(subset=['prior_shared_games','potential_assists_per100passes',*FAMILIARITY_CONTROLS]).copy()
    fe=pd.get_dummies(s.team,drop_first=True,dtype=float);Z=np.column_stack([np.ones(len(s)),s[FAMILIARITY_CONTROLS],fe])
    x=s.prior_shared_games.to_numpy(float);y=s.potential_assists_per100passes.to_numpy(float)
    s['adjusted_familiarity']=x-Z@np.linalg.lstsq(Z,x,rcond=None)[0];s['adjusted_productivity']=y-Z@np.linalg.lstsq(Z,y,rcond=None)[0]
    atomic_csv(s[['team','GAME_ID','passing_style','adjusted_familiarity','adjusted_productivity']],out/'familiarity_adjusted_plot_data.csv')
    for style,color,marker in [('more_balanced','#2473b8','o'),('more_star_centered','#e18a24','^')]:
        g=s.loc[s.passing_style.eq(style)];axes[1].scatter(g.adjusted_familiarity,g.adjusted_productivity,color=color,marker=marker,s=13,alpha=.17,edgecolors='none',rasterized=True)
    fit=linregress(s.adjusted_familiarity,s.adjusted_productivity);xx=np.linspace(s.adjusted_familiarity.min(),s.adjusted_familiarity.max(),100)
    axes[1].plot(xx,fit.intercept+fit.slope*xx,color='#333333')
    bins=pd.qcut(s.adjusted_familiarity,10,duplicates='drop');means=s.groupby(bins,observed=True)[['adjusted_familiarity','adjusted_productivity']].mean()
    axes[1].scatter(means.adjusted_familiarity,means.adjusted_productivity,color='#111111',s=25)
    r=fam_results.loc[fam_results.model.eq('familiarity_primary')&fam_results.term.eq('prior_shared_games')].iloc[0]
    axes[1].set_title('3. Prior shared appearances and passing productivity\n'+f'N={len(s):,} team-games | partial r={r.partial_r:.3f} | partial R²={r.partial_r_squared:.4f} | p={r.p_value_twoway:.3g}',fontsize=11)
    axes[1].set_xlabel('Prior shared team-game appearances after controls');axes[1].set_ylabel('Potential assists per 100 passes above/below expected')
    for ax in axes:ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('NBA 2025–26: hub-absence redistribution and within-season familiarity',fontsize=16,y=.99)
    fig.legend(handles=[Line2D([0],[0],ls='',marker='o',color='#2473b8',label='More balanced'),Line2D([0],[0],ls='',marker='^',color='#e18a24',label='More star-centered')],loc='upper center',bbox_to_anchor=(.5,.94),ncol=2,frameon=False)
    fig.text(.5,.025,'Topic 2 uses a fixed set of non-hub teammates who appeared in both conditions; intervals are within-team HAC(7).\n'
        'Topic 3 uses prior same-game appearances, not shared court minutes; controls include minutes changes and season progress; errors cluster by team and match.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.10,1,.86));season_topics_spread_labels(fig)
    fig.savefig(out/'roles_and_familiarity.png',dpi=180);fig.savefig(out/'roles_and_familiarity.pdf');plt.close(fig)


def run_roles_familiarity(argv):
    parser=argparse.ArgumentParser(description='2025-26 hub-absence redistribution and prior-shared-appearance familiarity')
    parser.add_argument('--roles-and-familiarity',action='store_true');parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--plot-only',action='store_true');args=parser.parse_args(argv)
    parent=args.output_dir.expanduser()/'2025-26/team_season_topics';out=parent/'roles_and_familiarity';out.mkdir(parents=True,exist_ok=True)
    if args.plot_only:
        fam=pd.read_csv(out/'familiarity_game_metrics.csv',dtype={'GAME_ID':str});core=pd.read_csv(out/'redistribution_game_metrics.csv',dtype={'GAME_ID':str})
    else:fam,core=build_roles_familiarity_data(parent,out)
    summary=summarize_hub_redistribution(core,out);d=summary.loc[summary.primary_eligible].copy();cross=[]
    for x,y,label,predictors in [('hub_present_concentration','adjusted_hhi_change','redistribution_primary',None),
        ('star_indicator','adjusted_hhi_change','balanced_vs_star',None),
        ('hub_present_concentration','raw_hhi_change','raw_change_sensitivity',None),
        ('hub_present_concentration','adjusted_hhi_change','baseline_and_panel_size_sensitivity',['hub_present_concentration','mean_hhi_with','core_player_count'])]:
        cross.extend(season_topics_fit(d,x,y,label,'hub_redistribution',predictors))
    cross=pd.DataFrame(cross)
    eligible_core=core.loc[core.team.isin(d.team)].copy()
    panel,unused=fit_defense_disruption(eligible_core,['hub_absent','absence_x_concentration',*ROLE_CONTROLS],'redistribution_one_stage','core_receiving_hhi')
    atomic_csv(pd.DataFrame(panel),out/'redistribution_one_stage_results.csv')
    s=fam.loc[fam.passing_complete.astype(str).str.lower().eq('true')&fam.prior_team_games.ge(10)].copy();fits=[];groups=[]
    specs=[(s,['prior_shared_games',*FAMILIARITY_CONTROLS],'familiarity_primary'),
        (s,['prior_shared_games','familiarity_x_star',*FAMILIARITY_CONTROLS],'familiarity_style_interaction'),
        (s,['prior_shared_game_fraction',*FAMILIARITY_CONTROLS],'prior_fraction_sensitivity'),
        (s,['prior_shared_games_equal_weight',*FAMILIARITY_CONTROLS],'equal_pair_weight_sensitivity'),
        (fam.loc[fam.passing_complete.astype(str).str.lower().eq('true')&fam.prior_team_games.ge(20)],['prior_shared_games',*FAMILIARITY_CONTROLS],'at_least_20_prior_games_sensitivity')]
    for data,predictors,label in specs:
        result,group=fit_defense_disruption(data,predictors,label,'potential_assists_per100passes');fits.extend(result);groups.extend(group)
    fits=pd.DataFrame(fits)
    # Explicit correction across the two requested primary hypotheses.
    a=cross.index[cross.model.eq('redistribution_primary')&cross.term.eq('hub_present_concentration')][0]
    b=fits.index[fits.model.eq('familiarity_primary')&fits.term.eq('prior_shared_games')][0]
    pair=[('redistribution',float(cross.loc[a,'p_value_hc3'])),('familiarity',float(fits.loc[b,'p_value_twoway']))]
    adjusted={};running=0
    for rank,(name,value) in enumerate(sorted(pair,key=lambda v:v[1])):running=max(running,min(1,value*(2-rank)));adjusted[name]=running
    cross['primary_p_holm_two']=np.nan;fits['primary_p_holm_two']=np.nan
    cross.loc[a,'primary_p_holm_two']=adjusted['redistribution'];fits.loc[b,'primary_p_holm_two']=adjusted['familiarity']
    atomic_csv(cross,out/'redistribution_regressions.csv');atomic_csv(fits,out/'familiarity_regressions.csv')
    team=s.groupby(['team','passing_style'],as_index=False).agg(valid_games=('GAME_ID','size'),mean_prior_shared_fraction=('prior_shared_game_fraction','mean'),
        mean_prior_shared_games=('prior_shared_games','mean'),potential_assists=('potential_assists','sum'),passes_made=('passes_made','sum'))
    team['potential_assists_per100passes']=100*team.potential_assists/team.passes_made
    atomic_csv(team,out/'familiarity_team_summary.csv')
    atomic_csv(pd.DataFrame(season_topics_fit(team,'mean_prior_shared_fraction','potential_assists_per100passes','team_season_sensitivity','familiarity')),out/'familiarity_team_season_regression.csv')
    plot_roles_familiarity(summary,fam,cross,fits,out)
    atomic_json(dict(season='2025-26',redistribution_teams=len(d),redistribution_games=len(eligible_core),familiarity_teams=s.team.nunique(),familiarity_games=len(s),
        excluded_familiarity_records=fam.loc[~fam.passing_complete.astype(str).str.lower().eq('true'),['team','GAME_ID','passing_data_error']].to_dict('records'),
        topic2='Does hub-present passing concentration predict a larger increase in receiving concentration among common non-hub teammates when hub is absent?',
        topic2_panel='Fixed common player set: appeared at least once in both complete with-hub and without-hub games; hub always excluded; normalized receiving HHI with fixed N',
        topic2_controls=ROLE_CONTROLS,topic2_eligibility='At least 20 complete games with hub and 10 without; >=5 common teammates',
        topic3='Does a more familiar rotation generate more team potential assists per pass?',
        familiarity='Minutes-product-weighted mean prior same-team same-game co-appearance count over current player pairs; current game excluded; within-season history only',
        topic3_controls=FAMILIARITY_CONTROLS,topic3_eligibility='At least 10 prior team games; complete current passing records; no minutes cutoff',
        uncertainty='Topic 2: within-team HAC(7), then cross-team HC3 with one-stage team/match-clustered sensitivity. Topic 3: team fixed effects and two-way team/match clustering. Holm across two primary tests.',
        limitations=['Exploratory observational associations','Receiving balance is different from offensive effectiveness',
            'Common-player panel excludes teammates who never appeared in both hub conditions; coverage reported',
            'Shared appearances do not imply shared court time, and do not capture experience before 2025-26',
            'Familiarity and productivity are analyzed at team-game level; pair-specific potential assists are not available in saved data',
            'Current minutes weights are retrospective opportunity proxies; rotations, injuries and tactics can still confound findings']),out/'analysis_audit.json')
    print('TOPIC 2:',cross.loc[a,['n','pearson_r','r_squared','p_value_hc3','primary_p_holm_two']].to_dict(),flush=True)
    print('TOPIC 3:',fits.loc[b,['n','teams','coefficient','partial_r','partial_r_squared','p_value_twoway','primary_p_holm_two']].to_dict(),flush=True)
    print(f'Finished redistribution and familiarity analysis: {out}',flush=True);return 0


def download_six_playoffs(parent,out):
    import sys
    from concurrent.futures import ThreadPoolExecutor,as_completed
    from nba_api.stats.endpoints import leaguegamelog,leaguedashptstats
    m=sys.modules[__name__]
    c=m.NBAClient(m.Config(season='2025-26',season_type='Playoffs',cache_dir=out/'playoff_cache',timeout=25,retries=2,min_delay=.5,max_delay=1))
    try:
     common=dict(season='2025-26',season_type_all_star='Playoffs')
     def fetch(cls,p,name,required,file):
      f,_=c.fetch(cls,p,lambda x:m.season_topics_frame(x,name,required));m.atomic_csv(f,out/file);return f
     games=fetch(leaguegamelog.LeagueGameLog,common,'LeagueGameLog',['TEAM_ID','GAME_ID','GAME_DATE'],'playoff_team_games.csv')
     players=fetch(leaguegamelog.LeagueGameLog,dict(player_or_team_abbreviation='P',**common),'LeagueGameLog',['TEAM_ID','PLAYER_ID','GAME_ID','MIN'],'playoff_player_games.csv')
     tracking=fetch(leaguedashptstats.LeagueDashPtStats,dict(player_or_team='Player',per_mode_simple='Totals',pt_measure_type='Passing',**common),'LeagueDashPtStats',['PLAYER_ID','GP','PASSES_MADE'],'playoff_player_tracking.csv')
     if games.TEAM_ID.nunique()!=16 or games.duplicated(['TEAM_ID','GAME_ID']).any():raise ValueError('Expected 16 playoff teams and unique games')
     plan=players.groupby(['TEAM_ID','PLAYER_ID','PLAYER_NAME'],as_index=False).agg(games=('GAME_ID','size'),minutes=('MIN','sum'))
     teams={t['id']:t['abbreviation'] for t in m.static_teams.get_teams()};plan['team']=plan.TEAM_ID.map(teams)
     m.atomic_csv(plan,out/'playoff_download_plan.csv');print(f'Playoffs: {games.TEAM_ID.nunique()} teams, {games.GAME_ID.nunique()} games, {len(plan)} player profiles',flush=True)
     def one(r):
      folder=out/'playoff_profiles'/r['team'];folder.mkdir(parents=True,exist_ok=True);path=folder/(str(int(r['PLAYER_ID']))+'.csv');audit=path.with_suffix('.json')
      if path.exists() and audit.exists():return json.loads(audit.read_text())
      try:
       f,_=c.passing(int(r['PLAYER_ID']),int(r['TEAM_ID']));tr=tracking.loc[tracking.PLAYER_ID.eq(r['PLAYER_ID'])]
       known_zero=bool(len(tr)==1 and tr.GP.iloc[0]==r['games'] and tr.PASSES_MADE.iloc[0]==0)
       complete=bool(not f.empty or known_zero);error='' if complete else 'Empty outgoing profile without independent zero confirmation'
       if not f.empty and f.G.gt(r['games']).any():raise ValueError('Profile covers too many playoff games')
      except m.DownloadsStopped:raise
      except m.DataUnavailable as exc:
       f=m.pd.DataFrame(columns=['PLAYER_ID','TEAM_ID','PASS_TEAMMATE_PLAYER_ID','PASS']);complete=False;error=str(exc)
      m.atomic_csv(f,path);result=dict(r,complete=complete,error=error);m.atomic_json(result,audit);return result
     results=[]
     with ThreadPoolExecutor(max_workers=2) as pool:
      fs=[pool.submit(one,r) for r in plan.to_dict('records')]
      for i,f in enumerate(as_completed(fs),1):
       results.append(f.result())
       if i%20==0 or i==len(fs):
        m.atomic_csv(m.pd.DataFrame(results),out/'playoff_download_progress.csv');print(f'Playoff profiles {i}/{len(fs)}, unavailable={sum(not r["complete"] for r in results)}',flush=True)
     print('Playoff downloads complete',flush=True)
    finally:c.close()


# ---------------------------------------------------------------------
# SIX PLAYER-ROLE / NETWORK QUESTIONS, 2025-26 (+ 2026 PLAYOFF COMPARISON)
# python teamgraphs.py --six-network-topics
# ---------------------------------------------------------------------
def six_player_cluster_fit(frame,x,y,controls,label,weight=None):
    """Team fixed effects; covariance clustered by team and player identity."""
    from scipy.stats import t
    required=[x,y,*controls]+([weight] if weight else [])
    d=frame.replace([np.inf,-np.inf],np.nan).dropna(subset=required).copy()
    if weight:d=d.loc[d[weight].gt(0)].copy()
    controls=[c for c in controls if d[c].nunique()>1 and c!=x]
    fe=pd.get_dummies(d.team,drop_first=True,dtype=float)
    Z=np.column_stack([np.ones(len(d)),d[controls].to_numpy(float),fe.to_numpy(float)])
    X=np.column_stack([Z[:,0],d[x].to_numpy(float),Z[:,1:]])
    response=d[y].to_numpy(float);w=d[weight].to_numpy(float) if weight else np.ones(len(d));w=w/w.mean();sw=np.sqrt(w)
    Xw=X*sw[:,None];n,k=X.shape
    if np.linalg.matrix_rank(Xw)<k or n<=k:raise ValueError('Collinear player model '+label)
    inv=np.linalg.inv(Xw.T@Xw);b=np.linalg.lstsq(Xw,response*sw,rcond=None)[0];e=response-X@b;scores=X*(w*e)[:,None]
    meat=np.zeros((k,k));counts=[]
    for key in ['team','player_id']:
        codes=pd.get_dummies(d[key],dtype=float).to_numpy();cluster_score=codes.T@scores;G=codes.shape[1];counts.append(G)
        meat+=G/(G-1)*(n-1)/(n-k)*(cluster_score.T@cluster_score)
    if d.duplicated(['team','player_id']).any():raise ValueError('Player model needs one row per team/player')
    meat-=n/(n-k)*(scores.T@scores);cov=inv@meat@inv;df=min(counts)-1
    if cov[1,1]<0:raise ValueError('Negative clustered primary variance')
    se=float(np.sqrt(cov[1,1]));p=float(2*t.sf(abs(b[1]/se),df));crit=t.ppf(.975,df)
    rx=d[x].to_numpy(float)-Z@np.linalg.lstsq(Z*sw[:,None],d[x].to_numpy(float)*sw,rcond=None)[0]
    ry=response-Z@np.linalg.lstsq(Z*sw[:,None],response*sw,rcond=None)[0]
    partial=float(np.sum(w*rx*ry)/np.sqrt(np.sum(w*rx*rx)*np.sum(w*ry*ry)))
    total=float(np.sum(w*(response-np.average(response,weights=w))**2));r2=1-float(np.sum(w*e*e))/total
    row=dict(model=label,x=x,y=y,coefficient=float(b[1]),ci95_low=float(b[1]-crit*se),ci95_high=float(b[1]+crit*se),
        standard_error=se,p_value=p,partial_r=partial,partial_r_squared=partial**2,full_model_r_squared=r2,
        n=n,teams=counts[0],unique_players=counts[1],df=df,weights=weight or 'none',inference='Two-way team/player clustered t')
    plot=d[['team','player_id','passing_style']].copy();plot['x_residual']=rx;plot['y_residual']=ry;plot['analysis_weight']=w
    return row,plot


def six_team_fit(frame,x,y,controls,label):
    result=season_topics_fit(frame,x,y,label,label,[x,*controls]);r=next(v for v in result if v['term']==x)
    d=frame.dropna(subset=[x,y,*controls]);Z=np.column_stack([np.ones(len(d)),d[controls].to_numpy(float)])
    rx=d[x].to_numpy(float)-Z@np.linalg.lstsq(Z,d[x].to_numpy(float),rcond=None)[0]
    ry=d[y].to_numpy(float)-Z@np.linalg.lstsq(Z,d[y].to_numpy(float),rcond=None)[0]
    partial=float(np.corrcoef(rx,ry)[0,1])
    row=dict(model=label,x=x,y=y,coefficient=r['coefficient'],ci95_low=r['slope_ci95_low'],ci95_high=r['slope_ci95_high'],
        standard_error=r['standard_error_hc3'],p_value=r['p_value_hc3'],partial_r=partial,partial_r_squared=partial**2,
        full_model_r_squared=r['r_squared'],n=r['n'],teams=len(d),inference='HC3 robust t')
    plot=d[['team','passing_style']].copy();plot['x_residual']=rx if controls else d[x].to_numpy(float);plot['y_residual']=ry if controls else d[y].to_numpy(float);plot['analysis_weight']=1
    return row,plot


def prepare_six_regular_data(parent,out):
    import networkx as nx
    ps=pd.read_csv(parent/'player_games.csv',dtype={'GAME_ID':str})
    grouped=ps.groupby(['TEAM_ID','PLAYER_ID'],as_index=False).agg(PLAYER_NAME=('PLAYER_NAME','first'),minutes=('MIN','sum'),appearances=('GAME_ID','size'),
        fga=('FGA','sum'),fgm=('FGM','sum'),fg3m=('FG3M','sum'),assists=('AST','sum'))
    base=pd.read_csv(parent/'team_season_metrics.csv');official=pd.read_csv(parent/'official_team_totals.csv').set_index('TEAM_ID')
    nodes=[];teams=[];edges=[]
    for _,r in base.iterrows():
        mf=pd.read_csv(parent/'networks'/r.team/'passing_matrix.csv',index_col=0);ids=mf.index.astype(int).tolist();a=mf.to_numpy(float)
        ec,scores,active_ids,active=eigenvector_passing_concentration(a,ids)
        if not ec['network_valid']:raise ValueError('Invalid regular-season network '+r.team)
        stats=grouped.loc[grouped.TEAM_ID.eq(r.team_id)].set_index('PLAYER_ID')
        if set(stats.index)!=set(ids):raise ValueError('Official roster/network mismatch')
        graph=nx.from_numpy_array(active+active.T)
        for i,j,v in graph.edges(data=True):v['distance']=1/v['weight']
        between=nx.betweenness_centrality(graph,weight='distance',normalized=True)
        communities=nx.community.louvain_communities(graph,weight='weight',seed=2026)
        membership={node:k for k,group in enumerate(communities) for node in group}
        incoming=a.sum(axis=0);total=a.sum();minutes_total=stats.minutes.sum()
        for i,pid in enumerate(active_ids):
            v=stats.loc[pid];strength=float(graph.degree(i,weight='weight'))
            shares=np.array([sum(graph[i][j]['weight'] for j in graph.neighbors(i) if membership[j]==k)/strength for k in range(len(communities))])
            original=ids.index(pid)
            nodes.append(dict(team=r.team,team_id=int(r.team_id),player_id=int(pid),name=v.PLAYER_NAME,passing_style=r.passing_style,
                ec_share=scores[pid],betweenness=between[i],community_participation=float(1-shares@shares),community=membership[i],
                degree_share=graph.degree(i)/(len(active_ids)-1),pass_strength=strength,log_pass_strength=np.log1p(strength),
                incoming_passes=incoming[original],incoming_share_pct=100*incoming[original]/total,
                minutes=v.minutes,log_minutes=np.log1p(v.minutes),minutes_share=v.minutes/minutes_total,appearances=v.appearances,
                fga=v.fga,fgm=v.fgm,fg3m=v.fg3m,efg_pct=100*(v.fgm+.5*v.fg3m)/v.fga if v.fga>0 else np.nan))
        pairs=[]
        for i in range(len(ids)):
            for j in range(i+1,len(ids)):
                count=float(a[i,j]+a[j,i])
                pairs.append((count,ids[i],ids[j]));edges.append(dict(team=r.team,player_1=ids[i],player_2=ids[j],combined_passes=count,share=count/total))
        pairs.sort(reverse=True);top=pairs[0];pair_counts=[v[0] for v in pairs]
        fga_hhi,raw_fga=normalized_share_concentration(stats.fga.to_numpy(float));min_hhi,_=normalized_share_concentration(stats.minutes.to_numpy(float))
        pair_hhi,_=normalized_share_concentration(pair_counts)
        teams.append(dict(team=r.team,team_id=int(r.team_id),passing_style=r.passing_style,network_player_count=len(active_ids),roster_player_count=len(ids),
            conditional_entropy=r.conditional_receiver_entropy,ec_concentration=r.eigenvector_concentration,
            top_pair_share_pct=100*top[0]/total,top_three_pairs_share_pct=100*sum(v[0] for v in pairs[:3])/total,
            top_pair_includes_hub=bool(ec['top_eigenvector_player_id'] in top[1:]),top_pair_player_1=stats.loc[top[1],'PLAYER_NAME'],top_pair_player_2=stats.loc[top[2],'PLAYER_NAME'],pair_concentration=pair_hhi,
            shooting_concentration=fga_hhi,raw_shooting_hhi=raw_fga,minutes_concentration=min_hhi,
            player_fga_total=float(stats.fga.sum()),official_team_fga=float(official.loc[r.team_id,'FGA']),
            unassigned_team_fga=float(official.loc[r.team_id,'FGA']-stats.fga.sum())))
    node=pd.DataFrame(nodes);team=pd.DataFrame(teams)
    for x in ['ec_share','betweenness']:
        node[x+'_z']=node.groupby('team')[x].transform(lambda s:(s-s.mean())/s.std(ddof=0))
    if node[['ec_share_z','betweenness_z']].isna().any().any():raise ValueError('Constant within-team node metric')
    atomic_csv(node,out/'regular_player_metrics.csv');atomic_csv(team,out/'regular_team_metrics.csv');atomic_csv(pd.DataFrame(edges),out/'pair_shares.csv')
    return node,team,ps


def prepare_playoff_comparison(parent,out,regular_players,regular_teams):
    from scipy.stats import ttest_rel,wilcoxon,t
    pg=pd.read_csv(out/'playoff_player_games.csv',dtype={'GAME_ID':str});games=pd.read_csv(out/'playoff_team_games.csv',dtype={'GAME_ID':str})
    rows=[];audits=[]
    for team_id,g in pg.groupby('TEAM_ID'):
        team=regular_teams.loc[regular_teams.team_id.eq(team_id),'team'].iloc[0]
        ids=sorted(map(int,g.PLAYER_ID.unique()));index={p:i for i,p in enumerate(ids)};a=np.zeros((len(ids),len(ids)));missing=[];outside=[]
        for pid in ids:
            path=out/'playoff_profiles'/team/(str(pid)+'.csv');audit=json.loads(path.with_suffix('.json').read_text());profile=pd.read_csv(path)
            if not audit['complete']:missing.append(pid)
            for v in profile.to_dict('records'):
                other=int(v['PASS_TEAMMATE_PLAYER_ID'])
                if other in index:a[index[pid],index[other]]+=v['PASS']
                elif v['PASS']>0:outside.append(dict(passer_id=int(pid),receiver_id=other,passes=int(v['PASS'])))
        full=pd.read_csv(parent/'networks'/team/'passing_matrix.csv',index_col=0);reg_ids=full.index.astype(int).tolist();reg_index={p:i for i,p in enumerate(reg_ids)}
        # Match players active in both periods; iterative pruning keeps the same set in both graphs.
        common=sorted(set(ids)&set(reg_ids))
        while True:
            reg=full.to_numpy(float)[np.ix_([reg_index[p] for p in common],[reg_index[p] for p in common])]
            post=a[np.ix_([index[p] for p in common],[index[p] for p in common])]
            good=((reg.sum(axis=0)+reg.sum(axis=1))>0)&((post.sum(axis=0)+post.sum(axis=1))>0)
            if good.all():break
            common=[p for p,keep in zip(common,good) if keep]
        if len(common)<3:raise ValueError('Insufficient matched playoff participants')
        reg_h,reg_c=compute_team_entropies(reg);post_h,post_c=compute_team_entropies(post)
        # Finite-sample sensitivity: estimate regular entropy at the playoff pass count.
        sample_count=int(post.sum());total_count=int(reg.sum());rng=np.random.default_rng(2026+int(team_id))
        if sample_count>total_count:raise ValueError('Playoff pass total exceeds available matched regular passes')
        draws=rng.multivariate_hypergeometric(reg.astype(int).ravel(),sample_count,size=1000).reshape(1000,len(common),len(common))
        row_totals=draws.sum(axis=2,keepdims=True)
        ratios=np.divide(draws,row_totals,out=np.ones_like(draws,dtype=float),where=draws>0)
        sampled_entropy=-np.sum((draws/sample_count)*np.log(ratios),axis=(1,2))/np.log(len(common)-1)
        sample_mean=float(sampled_entropy.mean())
        regular_minutes=regular_players.loc[regular_players.TEAM_ID.eq(team_id)&regular_players.PLAYER_ID.isin(common)].groupby('PLAYER_ID').MIN.sum().reindex(common,fill_value=0)
        playoff_minutes=g.loc[g.PLAYER_ID.isin(common)].groupby('PLAYER_ID').MIN.sum().reindex(common,fill_value=0)
        mr,_=normalized_share_concentration(regular_minutes);mp,_=normalized_share_concentration(playoff_minutes)
        rows.append(dict(team=team,team_id=int(team_id),passing_style=regular_teams.loc[regular_teams.team.eq(team),'passing_style'].iloc[0],
            regular_games=82,playoff_games=games.loc[games.TEAM_ID.eq(team_id),'GAME_ID'].nunique(),matched_players=len(common),
            regular_conditional_entropy=reg_c,playoff_conditional_entropy=post_c,entropy_change=post_c-reg_c,
            regular_minutes_concentration=mr,playoff_minutes_concentration=mp,minutes_concentration_change=mp-mr,
            excluded_non_teammate_passes=sum(v['passes'] for v in outside),
            matched_pass_count=sample_count,regular_matched_pass_entropy=sample_mean,
            playoff_minus_matched_pass_entropy=post_c-sample_mean,matched_pass_draws=1000,
            complete=not missing))
        folder=out/'matched_playoff_networks'/team;folder.mkdir(parents=True,exist_ok=True)
        atomic_csv(pd.DataFrame(reg,index=common,columns=common),folder/'regular_matrix.csv',index=True)
        atomic_csv(pd.DataFrame(post,index=common,columns=common),folder/'playoff_matrix.csv',index=True)
        audits.append(dict(team=team,matched_player_ids=common,missing_profiles=missing,outside_receiver_ids=outside))
    frame=pd.DataFrame(rows);v=frame.loc[frame.complete];diff=v.entropy_change.to_numpy();n=len(diff);sd=float(diff.std(ddof=1));mean=float(diff.mean());se=sd/np.sqrt(n);critical=t.ppf(.975,n-1)
    test=ttest_rel(v.playoff_conditional_entropy,v.regular_conditional_entropy)
    ranks=wilcoxon(diff,alternative='two-sided')
    signs=2*((np.arange(2**n)[:,None]>>np.arange(n))&1)-1
    sign_p=float(np.mean(np.abs(signs@diff/n)>=abs(mean)-1e-14))
    strict=v.loc[v.excluded_non_teammate_passes.eq(0)]
    strict_test=ttest_rel(strict.playoff_conditional_entropy,strict.regular_conditional_entropy)
    matched=ttest_rel(v.playoff_conditional_entropy,v.regular_matched_pass_entropy)
    result=dict(model='playoff_primary',effect=mean,ci95_low=mean-critical*se,ci95_high=mean+critical*se,
        p_value=float(test.pvalue),paired_t=float(test.statistic),wilcoxon_p=float(ranks.pvalue),sign_flip_p=sign_p,
        standardized_paired_effect=mean/sd,n=n,teams_decreased=int((diff<0).sum()),
        strict_no_invalid_edge_teams=len(strict),strict_mean_change=float(strict.entropy_change.mean()),strict_p=float(strict_test.pvalue),
        matched_pass_mean_change=float(v.playoff_minus_matched_pass_entropy.mean()),matched_pass_p=float(matched.pvalue),
        excluded_non_teammate_passes=int(v.excluded_non_teammate_passes.sum()),inference='Paired team t; Wilcoxon and exact sign-flip sensitivities')
    atomic_csv(frame,out/'playoff_comparison.csv');atomic_json(dict(teams=audits),out/'playoff_network_audit.json')
    return frame,result


def prepare_hub_role_changes(parent,out,regular_players,regular_teams):
    rows=[];audits=[]
    hm=pd.read_csv(parent/'hub_absence_resilience/team_season_metrics.csv')
    for _,r in hm.loc[hm.primary_eligible].iterrows():
        full=pd.read_csv(parent/'networks'/r.team/'passing_matrix.csv',index_col=0);ids=full.index.astype(int).tolist();index={p:i for i,p in enumerate(ids)}
        present=pd.read_csv(parent/'hub_absence_resilience/present_networks'/r.team/'passing_matrix.csv',index_col=0).to_numpy(float)
        absent=full.to_numpy(float)-present;hub=int(r.hub_id)
        common=[p for p in ids if p!=hub]
        while True:
            take=[index[p] for p in common];a=present[np.ix_(take,take)];b=absent[np.ix_(take,take)]
            active=((a.sum(0)+a.sum(1))>0)&((b.sum(0)+b.sum(1))>0)
            if active.all():break
            common=[p for p,keep in zip(common,active) if keep]
        ec1,s1,_,_=eigenvector_passing_concentration(a,common);ec2,s2,_,_=eigenvector_passing_concentration(b,common)
        if not ec1['network_valid'] or not ec2['network_valid']:
            audits.append(dict(team=r.team,error='Disconnected matched non-hub network'));continue
        own=regular_players.loc[regular_players.TEAM_ID.eq(r.team_id)]
        played=set(own.loc[own.PLAYER_ID.eq(hub),'GAME_ID'])
        pstats=own.loc[own.GAME_ID.isin(played)&own.PLAYER_ID.isin(common)].groupby('PLAYER_ID').agg(minutes=('MIN','sum'),assists=('AST','sum'),games=('GAME_ID','size')).reindex(common,fill_value=0)
        astats=own.loc[~own.GAME_ID.isin(played)&own.PLAYER_ID.isin(common)].groupby('PLAYER_ID').agg(minutes=('MIN','sum'),games=('GAME_ID','size')).reindex(common,fill_value=0)
        names=own.drop_duplicates('PLAYER_ID').set_index('PLAYER_ID').PLAYER_NAME.to_dict()
        for pid in common:
            with_share=pstats.loc[pid,'minutes']/pstats.minutes.sum();without_share=astats.loc[pid,'minutes']/astats.minutes.sum()
            rows.append(dict(team=r.team,player_id=int(pid),name=names[pid],hub_name=r.hub_name,
                passing_style=regular_teams.loc[regular_teams.team.eq(r.team),'passing_style'].iloc[0],
                present_ec=s1[pid],absent_ec=s2[pid],ec_change=s2[pid]-s1[pid],
                baseline_assist_share=100*pstats.loc[pid,'assists']/pstats.assists.sum(),
                minutes_share_change=without_share-with_share,present_minutes_share=with_share,absent_minutes_share=without_share,
                with_hub_games=int(pstats.loc[pid,'games']),without_hub_games=int(astats.loc[pid,'games'])))
        audits.append(dict(team=r.team,hub_id=hub,matched_nonhub_ids=common,error=''))
    frame=pd.DataFrame(rows);atomic_csv(frame,out/'hub_role_player_metrics.csv');atomic_json(dict(teams=audits),out/'hub_role_audit.json')
    top=frame.sort_values('ec_change',ascending=False).groupby('team').head(3).sort_values(['team','ec_change'],ascending=[True,False])
    atomic_csv(top,out/'largest_role_gains.csv')
    from scipy.stats import spearmanr
    role_teams=[]
    for team,g in frame.groupby('team'):
        before=g.sort_values(['present_ec','player_id'],ascending=[False,True]).iloc[0]
        after=g.sort_values(['absent_ec','player_id'],ascending=[False,True]).iloc[0]
        role_teams.append(dict(team=team,hub_name=g.hub_name.iloc[0],matched_nonhub_players=len(g),
            top_with_hub=before['name'],top_without_hub=after['name'],top_player_changed=bool(before.player_id!=after.player_id),
            centrality_total_variation=float(.5*g.ec_change.abs().sum()),rank_correlation=float(spearmanr(g.present_ec,g.absent_ec).statistic)))
    atomic_csv(pd.DataFrame(role_teams),out/'hub_role_team_summary.csv');return frame


def plot_six_network_topics(primary,plot_data,playoffs,out):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    titles={1:'1. Passing hubs and network bridges',2:'2. Shooter efficiency and incoming passes',
        3:'3. Regular season versus playoffs',4:'4. Do normal creators inherit hub importance?',
        5:'5. Individual hubs and strongest partnerships',6:'6. Passing diversity and shooting concentration'}
    xlabels={1:'Within-team eigenvector centrality after minutes control',2:'Shooting eFG% after minutes and availability controls',
        4:'Normal assist share after baseline-role/minutes controls',5:'Eigenvector passing concentration',6:'Normalized conditional passing entropy H(R | P)'}
    ylabels={1:'Network betweenness after minutes control',2:'Incoming-pass share above/below expected (percentage points)',
        4:'Hub-absent centrality above/below expected',5:'Top player-pair share of passes (%)',6:'Shooting concentration (normalized player-FGA HHI)'}
    def draw(ax,topic,show_team_labels=True):
        r=primary.loc[primary.topic.eq(topic)].iloc[0]
        if topic==3:
            d=playoffs.loc[playoffs.complete]
            for _,row in d.iterrows():
                color='#2473b8' if row.passing_style=='more_balanced' else '#e18a24'
                ax.plot([0,1],[row.regular_conditional_entropy,row.playoff_conditional_entropy],marker='o',color=color,alpha=.75,lw=1)
                if show_team_labels:ax.annotate(row.team,(1,row.playoff_conditional_entropy),xytext=(4,4),textcoords='offset points',fontsize=7)
            ax.set_xticks([0,1],['Regular season','Playoffs']);ax.set_xlim(-.15,1.35)
            ax.set_ylabel('Conditional entropy on matched player set')
            ax.set_title(titles[topic]+'\n'+f'{int(r.n)} paired teams | mean change={r.effect:.3f} | paired p={r.p_value:.3g}',fontsize=10.5,pad=12)
        else:
            d=plot_data[topic]
            for style,color,marker in [('more_balanced','#2473b8','o'),('more_star_centered','#e18a24','^')]:
                g=d.loc[d.passing_style.eq(style)]
                ax.scatter(g.x_residual,g.y_residual,color=color,marker=marker,s=32 if len(d)<=40 else 12,alpha=.85 if len(d)<=40 else .28,edgecolors='none',rasterized=len(d)>100)
                if len(d)<=40:
                    for _,v in g.iterrows():ax.annotate(v.team,(v.x_residual,v.y_residual),xytext=(4,4),textcoords='offset points',fontsize=7)
            x=d.x_residual.to_numpy(float);y=d.y_residual.to_numpy(float);w=d.analysis_weight.to_numpy(float)
            X=np.column_stack([np.ones(len(d)),x]);b=np.linalg.lstsq(X*np.sqrt(w)[:,None],y*np.sqrt(w),rcond=None)[0]
            if topic==6:
                descriptions=[]
                for style,color,name in [('more_balanced','#2473b8','Balanced'),('more_star_centered','#e18a24','Star-centered')]:
                    g=d.loc[d.passing_style.eq(style)]
                    fit=season_topics_fit(g,'x_residual','y_residual',style,6)
                    slope=next(v for v in fit if v['term']=='x_residual')
                    intercept=next(v for v in fit if v['term']=='intercept')['coefficient']
                    xx=np.linspace(g.x_residual.min(),g.x_residual.max(),100)
                    ax.plot(xx,intercept+slope['coefficient']*xx,color=color,lw=2)
                    descriptions.append(f"{name}: n={len(g)}, r={slope['pearson_r']:.3f}, R²={slope['r_squared']:.3f}, p={slope['p_value_hc3']:.3g}")
            else:
                xx=np.linspace(x.min(),x.max(),100);ax.plot(xx,b[0]+b[1]*xx,color='#333333',lw=1.8)
            ax.set_xlabel(xlabels[topic],fontsize=8.5);ax.set_ylabel(ylabels[topic],fontsize=8.5)
            label='r' if topic in [5,6] else 'partial r'
            subtitle='\n'.join(descriptions) if topic==6 else f'N={int(r.n)} | {label}={r.partial_r:.3f} | R²={r.partial_r_squared:.3f} | p={r.p_value:.3g}'
            ax.set_title(titles[topic]+'\n'+subtitle,fontsize=10.5,pad=12)
            ax.margins(.12)
        ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False)
    legend=[Line2D([0],[0],ls='',marker='o',color='#2473b8',label='More balanced (fixed season label)'),
        Line2D([0],[0],ls='',marker='^',color='#e18a24',label='More star-centered (fixed season label)')]
    fig,axes=plt.subplots(3,2,figsize=(18,16))
    for topic,ax in enumerate(axes.flat,1):draw(ax,topic,show_team_labels=topic!=3)
    fig.suptitle('NBA 2025–26: six passing-network and player-role questions',fontsize=18,y=.99)
    fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(.5,.96),ncol=2,frameon=False)
    fig.text(.5,.013,'Player comparisons use team/player clustered errors; team comparisons use HC3. Shooting-efficiency model is weighted by FGA.\n'
        'Topics 1 and 5 compare measures derived from the same network. Topic 3 is a paired postseason comparison; the other topics use regular-season data.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.93));season_topics_spread_labels(fig)
    fig.savefig(out/'six_network_topics.png',dpi=180);fig.savefig(out/'six_network_topics.pdf');plt.close(fig)
    for topic in range(1,7):
        fig,ax=plt.subplots(figsize=(11,8));draw(ax,topic)
        fig.suptitle('NBA 2025–26 · '+titles[topic],fontsize=13,y=.985)
        fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(.5,.94),ncol=2,frameon=False)
        fig.tight_layout(rect=(0,.025,1,.88));season_topics_spread_labels(fig)
        fig.savefig(out/f'topic_{topic}.png',dpi=180);fig.savefig(out/f'topic_{topic}.pdf');plt.close(fig)


def run_six_network_topics(argv):
    parser=argparse.ArgumentParser(description='Six 2025-26 player-role/network questions, including matched 2026 playoffs')
    parser.add_argument('--six-network-topics',action='store_true');parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--use-saved-playoffs',action='store_true',help='Skip API downloads and use the already saved playoff profiles')
    args=parser.parse_args(argv);parent=args.output_dir.expanduser()/'2025-26/team_season_topics';out=parent/'six_network_topics';out.mkdir(parents=True,exist_ok=True)
    if not args.use_saved_playoffs:download_six_playoffs(parent,out)
    node,team,players=prepare_six_regular_data(parent,out)
    post,paired=prepare_playoff_comparison(parent,out,players,team)
    roles=prepare_hub_role_changes(parent,out,players,team)
    primary=[];all_models=[];plot_data={}
    def add_player(topic,data,x,y,controls,label,weight=None,main=False):
        row,plot=six_player_cluster_fit(data,x,y,controls,label,weight);row['topic']=topic;all_models.append(row)
        if main:primary.append(dict(row,effect=row['coefficient']));plot_data[topic]=plot
    def add_team(topic,x,y,controls,label,main=False,data=None):
        row,plot=six_team_fit(team if data is None else data,x,y,controls,label);row['topic']=topic;all_models.append(row)
        if main:primary.append(dict(row,effect=row['coefficient']));plot_data[topic]=plot
    add_player(1,node,'ec_share_z','betweenness_z',['log_minutes'],'hub_bridge_primary',main=True)
    add_player(1,node,'ec_share_z','betweenness_z',['log_minutes','log_pass_strength','degree_share'],'hub_bridge_strength_degree_sensitivity')
    add_player(1,node,'ec_share_z','community_participation',['log_minutes'],'community_participation_sensitivity')
    add_player(2,node,'efg_pct','incoming_share_pct',['minutes_share','appearances'],'shooter_targeting_primary','fga',True)
    add_player(2,node,'efg_pct','incoming_share_pct',['minutes_share','appearances'],'shooter_targeting_unweighted_sensitivity')
    add_player(2,node.loc[node.fga.ge(100)],'efg_pct','incoming_share_pct',['minutes_share','appearances'],'shooter_targeting_100fga_sensitivity')
    primary.append(dict(paired,topic=3,partial_r=np.nan,partial_r_squared=np.nan))
    # At zero change in minutes concentration, intercept estimates the remaining playoff shift.
    minute_models=season_topics_fit(post.loc[post.complete],'minutes_concentration_change','entropy_change','playoff_minutes_sensitivity','playoffs')
    atomic_csv(pd.DataFrame(minute_models),out/'playoff_minutes_sensitivity.csv')
    add_player(4,roles,'baseline_assist_share','absent_ec',['present_ec','minutes_share_change'],'hub_role_gain_primary',main=True)
    reliable=roles.loc[roles.with_hub_games.ge(5)&roles.without_hub_games.ge(5)]
    add_player(4,reliable,'baseline_assist_share','absent_ec',['present_ec','minutes_share_change'],'hub_role_gain_5game_sensitivity')
    add_team(5,'ec_concentration','top_pair_share_pct',[],'hub_partnership_primary',True)
    add_team(5,'ec_concentration','top_pair_share_pct',['network_player_count','minutes_concentration'],'hub_partnership_adjusted_sensitivity')
    add_team(5,'ec_concentration','top_three_pairs_share_pct',[],'three_pair_sensitivity')
    add_team(6,'conditional_entropy','shooting_concentration',[],'passing_shooting_primary',True)
    add_team(6,'conditional_entropy','shooting_concentration',[],'passing_shooting_unadjusted_sensitivity')
    add_team(6,'conditional_entropy','raw_shooting_hhi',[],'passing_shooting_raw_hhi_sensitivity')
    result=pd.DataFrame(primary).sort_values('topic');result['p_holm_six']=np.nan;running=0
    for rank,idx in enumerate(result.p_value.sort_values().index):
        running=max(running,min(1,float(result.loc[idx,'p_value'])*(6-rank)));result.loc[idx,'p_holm_six']=running
    atomic_csv(result,out/'six_primary_results.csv');atomic_csv(pd.DataFrame(all_models),out/'regression_sensitivities.csv')
    for topic,plot in plot_data.items():atomic_csv(plot,out/f'topic_{topic}_plot_data.csv')
    # Exploratory subgroup fits retain the fixed season eigenvector style labels.
    style_rows=[]
    for style,group in team.groupby('passing_style'):
        fit=season_topics_fit(group,'conditional_entropy','shooting_concentration',style,6)
        style_rows.extend(dict(row,passing_style=style,controls='none') for row in fit)
    atomic_csv(pd.DataFrame(style_rows),out/'topic_6_style_regressions.csv')
    interaction=team.copy()
    interaction['star_indicator']=interaction.passing_style.eq('more_star_centered').astype(float)
    interaction['entropy_x_star']=interaction.conditional_entropy*interaction.star_indicator
    fit=season_topics_fit(interaction,'conditional_entropy','shooting_concentration','style_slope_difference',6,
        ['conditional_entropy','star_indicator','entropy_x_star'])
    atomic_csv(pd.DataFrame(fit),out/'topic_6_style_interaction.csv')
    plot_six_network_topics(result,plot_data,post,out)
    atomic_json(dict(season='2025-26',regular_teams=len(team),regular_player_stints=len(node),playoff_teams=int(paired['n']),
        role_change_teams=roles.team.nunique(),role_change_player_stints=len(roles),
        definitions=dict(bridging='Weighted betweenness with inverse-pass-count edge distance; secondary Louvain community participation',
            targeting='Incoming pass share vs player eFG%, controlling minutes share and appearances; weighted by player FGA',
            playoffs='Within-team conditional entropy difference, playoffs minus regular season, on identical matched active player sets',
            roles='Hub always removed from both graphs; same active non-hub participants. Absent centrality regressed on normal assist share, controlling present centrality and minutes-share change',
            partnerships='Largest unordered player-pair combined pass count divided by all team passes',
            shooting='Normalized HHI of assigned player FGA vs conditional passing entropy; no regression controls'),
        hypotheses='Original six questions selected before outcomes; topic 6 subsequently revised at user request to use no regression controls; updated Holm correction across six',
        no_minutes_cutoff=True,playoff_minimum_games='No winner-based cutoff; all 16 playoff teams included if profiles complete',
        inference='Player models: offensive-team fixed effects and two-way team/player clustering; team models HC3; playoff contrast paired t with Wilcoxon/exact sign-flip sensitivities',
        limitations=['Exploratory follow-ups after prior analyses','Centrality, betweenness and partnership metrics reuse the same passes and are structurally related',
            'Shared court minutes and player positions are not controlled in shooter targeting; reverse causality and role differences remain',
            'Playoff entropy differences also reflect opponents, minutes allocation and sample length; not an isolated causal playoff effect',
            'Paired playoff teams are part of one tournament rather than fully independent draws',
            'Postseason count sensitivity resamples aggregated passes, not whole games; it does not model temporal dependence',
            'Non-teammate tracking edges are excluded by official roster membership, with a strict-team exclusion sensitivity',
            'Role gains are connectivity changes, not proof of increased skill or improved offense',
            'Normalized concentration indices depend partly on the selected player population; panel sizes and sensitivities are saved',
            'Unassigned team-only shot attempts are reported and excluded from player-share denominators']),out/'analysis_audit.json')
    print(result[['topic','n','partial_r','partial_r_squared','effect','p_value','p_holm_six']].to_string(index=False),flush=True)
    print('Playoff contrast:',paired,flush=True);print(f'Finished six network topics: {out}',flush=True);return 0


def download_eight_context(parent,out):
    import sys
    from concurrent.futures import ThreadPoolExecutor,as_completed
    from nba_api.stats.endpoints import leaguegamelog,leaguedashplayerclutch,leaguedashteamclutch,playerdashptpass
    m=sys.modules[__name__]
    cache=out/'cache';cache.mkdir(parents=True,exist_ok=True)
    c=m.NBAClient(m.Config(season='2025-26',cache_dir=cache,timeout=25,retries=2,min_delay=.5,max_delay=1))
    try:
     prior,_=c.fetch(leaguegamelog.LeagueGameLog,dict(season='2024-25',season_type_all_star='Regular Season',player_or_team_abbreviation='P'),
      lambda p:m.season_topics_frame(p,'LeagueGameLog',['PLAYER_ID','TEAM_ID','GAME_ID','MIN']))
     m.atomic_csv(prior,out/'prior_player_games.csv')
     team_clutch,_=c.fetch(leaguedashteamclutch.LeagueDashTeamClutch,dict(season='2025-26',season_type_all_star='Regular Season',per_mode_detailed='Totals'),
      lambda p:m.season_topics_frame(p,'LeagueDashTeamClutch',['TEAM_ID','AST','GP']))
     m.atomic_csv(team_clutch,out/'official_team_clutch.csv')
     games=m.pd.read_csv(parent/'team_games.csv',dtype={'GAME_ID':str});players=m.pd.read_csv(parent/'player_games.csv',dtype={'GAME_ID':str})
     cutoff=str(games.GAME_DATE.sort_values().iloc[len(games)//2-1])[:10]
     m.atomic_json(dict(cutoff=cutoff,rule='League-wide median regular-season game date; first period <= cutoff, second > cutoff'),out/'calendar_split.json')
     daily=m.pd.concat([m.pd.read_csv(f).assign(date=f.stem) for f in (parent/'daily_passing').glob('*.csv')],ignore_index=True)
     teams={t['id']:t['abbreviation'] for t in m.static_teams.get_teams()};plan=[]
     for team_id,tg in games.groupby('TEAM_ID'):
      team=teams[int(team_id)];own=players.loc[players.TEAM_ID.eq(team_id)]
      home=set(tg.loc[tg.MATCHUP.str.contains('vs.',regex=False),'GAME_ID']);early=set(tg.loc[tg.GAME_DATE.le(cutoff),'GAME_ID'])
      full=m.pd.read_csv(parent/'networks'/team/'passing_matrix.csv',index_col=0);totals=full.sum(axis=1).to_dict()
      for split,gameids in [('home',home),('early',early)]:
       for pid,g in own.loc[own.GAME_ID.isin(gameids)].groupby('PLAYER_ID'):
        d=daily.loc[daily.TEAM_ID.eq(team_id)&daily.PLAYER_ID.eq(pid)&daily.date.isin(set(g.GAME_DATE.astype(str).str[:10]))]
        plan.append(dict(kind=split,team=team,team_id=int(team_id),player_id=int(pid),season='2025-26',expected_games=len(g),cutoff=cutoff,
         zero_confirmed=bool(totals.get(int(pid),0)==0 or (len(d)==len(g) and d.PASSES_MADE.sum()==0))))
      previous=prior.loc[prior.TEAM_ID.eq(team_id)]
      retained=set(previous.PLAYER_ID)&set(own.PLAYER_ID)
      for pid in sorted(retained):
       g=previous.loc[previous.PLAYER_ID.eq(pid)]
       plan.append(dict(kind='prior',team=team,team_id=int(team_id),player_id=int(pid),season='2024-25',expected_games=len(g),cutoff='',zero_confirmed=False))
     plan=m.pd.DataFrame(plan);m.atomic_csv(plan,out/'profile_download_plan.csv')
     print(f'Cutoff {cutoff}; {len(plan)} profiles (home {sum(plan.kind=="home")}, early {sum(plan.kind=="early")}, retained prior {sum(plan.kind=="prior")})',flush=True)
     def profile(r):
      folder=out/'profiles'/r['kind']/r['team'];folder.mkdir(parents=True,exist_ok=True);path=folder/(str(r['player_id'])+'.csv');audit=path.with_suffix('.json')
      if path.exists() and audit.exists():return json.loads(audit.read_text())
      params=dict(player_id=r['player_id'],team_id=r['team_id'],season=r['season'],season_type_all_star='Regular Season',per_mode_simple='Totals')
      if r['kind']=='home':params['location_nullable']='Home'
      if r['kind']=='early':params['date_to_nullable']=m.pd.Timestamp(r['cutoff']).strftime('%m/%d/%Y')
      def validate(p):
       f=m.validate_passing(p,r['player_id'],r['team_id'])
       if not f.empty and f.G.gt(r['expected_games']).any():raise ValueError('Split filter includes more games than official participation')
       return f
      try:
       f,_=c.fetch(playerdashptpass.PlayerDashPtPass,params,validate);complete=not f.empty or r['zero_confirmed'];error='' if complete else 'Empty profile without independently confirmed zero'
      except m.DownloadsStopped:raise
      except m.DataUnavailable as e:f=m.pd.DataFrame(columns=['PLAYER_ID','TEAM_ID','PASS_TEAMMATE_PLAYER_ID','PASS']);complete=False;error=str(e)
      m.atomic_csv(f,path);result=dict(r,complete=bool(complete),error=error);m.atomic_json(result,audit);return result
     results=[]
     with ThreadPoolExecutor(max_workers=2) as pool:
      futures=[pool.submit(profile,r) for r in plan.to_dict('records')]
      for i,f in enumerate(as_completed(futures),1):
       results.append(f.result())
       if i%40==0 or i==len(futures):m.atomic_csv(m.pd.DataFrame(results),out/'profile_download_progress.csv');print(f'Context profiles {i}/{len(plan)}, unavailable={sum(not r["complete"] for r in results)}',flush=True)
     def clutch(team_id):
      team=teams[team_id];path=out/'clutch_players'/f'{team}.csv';path.parent.mkdir(exist_ok=True)
      if path.exists():return team
      f,_=c.fetch(leaguedashplayerclutch.LeagueDashPlayerClutch,dict(season='2025-26',season_type_all_star='Regular Season',per_mode_detailed='Totals',team_id_nullable=team_id,clutch_time='Last 5 Minutes',point_diff='5',ahead_behind='Ahead or Behind'),
        lambda p:m.season_topics_frame(p,'LeagueDashPlayerClutch',['PLAYER_ID','AST','GP','MIN']))
      f['QUERY_TEAM_ID']=team_id;m.atomic_csv(f,path);return team
     with ThreadPoolExecutor(max_workers=2) as pool:
      for name in pool.map(clutch,sorted(teams)):print('Clutch assists saved',name,flush=True)
     print('Eight-topic downloads complete',flush=True)
    finally:c.close()


# ---------------------------------------------------------------------
# EIGHT CONTEXT / CONTINUITY / ROLE STUDIES (2025-26; persistence adds 2024-25)
# python teamgraphs.py --eight-context-topics
# ---------------------------------------------------------------------
def paired_context_test(frame,left,right,label):
    from scipy.stats import t,ttest_rel,wilcoxon
    d=frame.dropna(subset=[left,right]);diff=d[right]-d[left];n=len(diff)
    if n<5:raise ValueError('Insufficient paired observations '+label)
    se=float(diff.std(ddof=1)/np.sqrt(n));critical=t.ppf(.975,n-1)
    return dict(model=label,n=n,effect=float(diff.mean()),ci95_low=float(diff.mean()-critical*se),
        ci95_high=float(diff.mean()+critical*se),p_value=float(ttest_rel(d[right],d[left]).pvalue),
        wilcoxon_p=float(wilcoxon(diff).pvalue),partial_r=np.nan,partial_r_squared=np.nan,inference='Paired team t; Wilcoxon sensitivity')


def context_fingerprint(matrix,ids):
    import networkx as nx
    ec,_,active_ids,a=eigenvector_passing_concentration(matrix,ids)
    if not ec['network_valid']:raise ValueError('Invalid fingerprint graph: '+ec['network_error'])
    receiver,conditional=compute_team_entropies(a);mi,nmi=passing_mutual_information(a);total=a.sum();n=len(a)
    incoming=a.sum(0);outgoing=a.sum(1);hi,_=normalized_share_concentration(incoming);ho,_=normalized_share_concentration(outgoing)
    graph=nx.from_numpy_array(a+a.T);pairs=sorted([a[i,j]+a[j,i] for i in range(n) for j in range(i+1,n)],reverse=True)
    strength=np.sort(incoming+outgoing);gini=float((2*np.sum(np.arange(1,n+1)*strength)/(n*strength.sum()))-(n+1)/n)
    return dict(receiver_entropy=receiver,conditional_entropy=conditional,normalized_mi=nmi,
        ec_concentration=ec['eigenvector_concentration'],incoming_hhi=hi,outgoing_hhi=ho,
        reciprocity=float(np.minimum(a,a.T).sum()/total),top_three_pair_share=float(sum(pairs[:3])/total),
        weighted_clustering=float(nx.average_clustering(graph,weight='weight')),strength_gini=gini)


CONTEXT_FEATURES=['receiver_entropy','conditional_entropy','normalized_mi','ec_concentration','incoming_hhi',
    'outgoing_hhi','reciprocity','top_three_pair_share','weighted_clustering','strength_gini']


def build_context_split_networks(parent,out):
    season=pd.read_csv(parent/'team_season_metrics.csv');plan=pd.read_csv(out/'profile_download_plan.csv')
    rows=[];fingerprints=[];audits=[]
    ps=pd.read_csv(parent/'player_games.csv',dtype={'GAME_ID':str});context=pd.read_csv(parent/'defense_passing_disruption/game_metrics.csv',dtype={'GAME_ID':str})
    for _,team in season.iterrows():
        full=pd.read_csv(parent/'networks'/team.team/'passing_matrix.csv',index_col=0);ids=full.index.astype(int).tolist();index={p:i for i,p in enumerate(ids)}
        for kind,complement in [('home','away'),('early','late')]:
            a=np.zeros(full.shape);missing=[];outsiders=[]
            for task in plan.loc[plan.team.eq(team.team)&plan.kind.eq(kind)].to_dict('records'):
                path=out/'profiles'/kind/team.team/(str(int(task['player_id']))+'.csv');audit=json.loads(path.with_suffix('.json').read_text());profile=pd.read_csv(path)
                if not audit['complete']:missing.append(int(task['player_id']))
                for v in profile.to_dict('records'):
                    receiver=int(v['PASS_TEAMMATE_PLAYER_ID'])
                    if receiver in index:a[index[int(task['player_id'])],index[receiver]]+=v['PASS']
                    elif v['PASS']>0:outsiders.append(dict(passer_id=int(task['player_id']),receiver_id=receiver,passes=float(v['PASS'])))
            b=full.to_numpy(float)-a;negative=np.argwhere(b<0);complete=bool(not missing and not len(negative))
            if len(negative):
                audits.append(dict(team=team.team,split=kind,missing_profiles=missing,negative_edges=len(negative),outside_edges=outsiders,complete=False));continue
            folder=out/'split_networks'/team.team;folder.mkdir(parents=True,exist_ok=True)
            for label,matrix in [(kind,a),(complement,b)]:
                atomic_csv(pd.DataFrame(matrix,index=ids,columns=ids),folder/(label+'.csv'),index=True)
                features=context_fingerprint(matrix,ids)
                fingerprints.append(dict(team=team.team,period=label,passing_style=team.passing_style,complete=complete,**features))
            audits.append(dict(team=team.team,split=kind,missing_profiles=missing,negative_edges=0,outside_edges=outsiders,complete=complete))
            if kind=='home':
                common=ids.copy()
                while True:
                    ix=[index[p] for p in common];x=a[np.ix_(ix,ix)];y=b[np.ix_(ix,ix)]
                    keep=((x.sum(0)+x.sum(1))>0)&((y.sum(0)+y.sum(1))>0)
                    if keep.all():break
                    common=[p for p,k in zip(common,keep) if k]
                _,home=compute_team_entropies(x);_,away=compute_team_entropies(y)
                home_ec,_,_,_=eigenvector_passing_concentration(x,common);away_ec,_,_,_=eigenvector_passing_concentration(y,common)
                def strongest(z):
                    values=[(float(z[i,j]+z[j,i]),common[i],common[j]) for i in range(len(common)) for j in range(i+1,len(common))]
                    value,p1,p2=max(values);return f'{p1}:{p2}',value/z.sum()
                home_pair,home_pair_share=strongest(x);away_pair,away_pair_share=strongest(y)
                tg=context.loc[context.team.eq(team.team)];hg=set(tg.loc[tg.home_game.eq(1),'GAME_ID']);ag=set(tg.loc[tg.home_game.eq(0),'GAME_ID'])
                own=ps.loc[ps.TEAM_ID.eq(team.team_id)&ps.PLAYER_ID.isin(common)]
                mh,_=normalized_share_concentration(own.loc[own.GAME_ID.isin(hg)].groupby('PLAYER_ID').MIN.sum().reindex(common,fill_value=0))
                ma,_=normalized_share_concentration(own.loc[own.GAME_ID.isin(ag)].groupby('PLAYER_ID').MIN.sum().reindex(common,fill_value=0))
                rows.append(dict(team=team.team,passing_style=team.passing_style,matched_players=len(common),home_entropy=home,away_entropy=away,
                    home_ec_concentration=home_ec['eigenvector_concentration'],away_ec_concentration=away_ec['eigenvector_concentration'],
                    home_top_pair=home_pair,away_top_pair=away_pair,same_top_pair=home_pair==away_pair,
                    home_top_pair_share=home_pair_share,away_top_pair_share=away_pair_share,
                    minutes_concentration_change=mh-ma,opponent_strength_change=tg.loc[tg.home_game.eq(1)].opponent_strength.mean()-tg.loc[tg.home_game.eq(0)].opponent_strength.mean(),
                    home_minus_away=home-away,complete=complete,home_passes=x.sum(),away_passes=y.sum()))
    comparison=pd.DataFrame(rows);fingerprint=pd.DataFrame(fingerprints)
    atomic_csv(comparison,out/'home_away_comparison.csv');atomic_csv(fingerprint,out/'network_fingerprints.csv');atomic_json(dict(teams=audits),out/'split_network_audit.json')
    return comparison,fingerprint


def build_context_game_data(parent,out):
    frame=pd.read_csv(parent/'defense_passing_disruption/game_metrics.csv',dtype={'GAME_ID':str}).copy()
    players=pd.read_csv(parent/'player_games.csv',dtype={'GAME_ID':str});season=pd.read_csv(parent/'team_season_metrics.csv')
    hubs=season.set_index('team').top_eigenvector_player_id.to_dict()
    daily=pd.concat([pd.read_csv(f).assign(date=f.stem) for f in (parent/'daily_passing').glob('*.csv')],ignore_index=True)
    tracking={(int(t),str(d)):g.set_index('PLAYER_ID') for (t,d),g in daily.groupby(['TEAM_ID','date'])}
    roster={(int(t),str(g)):v.set_index('PLAYER_ID') for (t,g),v in players.groupby(['TEAM_ID','GAME_ID'])}
    rows=[];stable_ids=set(players.groupby('PLAYER_ID').TEAM_ID.nunique().loc[lambda s:s.eq(1)].index)
    for team_id,g in frame.groupby('TEAM_ID'):
        g=g.sort_values('GAME_DATE');team=g.team.iloc[0];hub=int(hubs[team]);own=players.loc[players.TEAM_ID.eq(team_id)]
        key=own.loc[own.PLAYER_ID.ne(hub)&own.PLAYER_ID.isin(stable_ids)].groupby('PLAYER_ID').agg(minutes=('MIN','sum'),gp=('GAME_ID','size'))
        key=key.loc[key.gp.ge(20)].sort_values('minutes',ascending=False).head(3);weights=key.minutes/key.minutes.sum()
        previous_date=None
        for _,r in g.iterrows():
            pg=roster[(int(team_id),r.GAME_ID)];tr=tracking[(int(team_id),str(r.GAME_DATE)[:10])];data=r.to_dict();now=pd.Timestamp(r.GAME_DATE)
            data['days_rest']=max(0,(now-previous_date).days-1) if previous_date is not None else np.nan;previous_date=now
            data['hub_id']=hub;data['hub_present']=int(hub in pg.index);data['hub_absent']=1-data['hub_present']
            data['hub_minutes_share']=float(pg.loc[hub,'MIN']/pg.MIN.sum()) if hub in pg.index else 0
            data['other_key_absence']=sum(float(w) for p,w in weights.items() if p not in pg.index)
            data['receiving_entropy_game']=np.nan;data['hub_involvement_share']=np.nan;data['minutes_entropy_game']=np.nan
            if bool(r.receiving_coverage):
                active=tr.loc[(tr.PASSES_MADE+tr.PASSES_RECEIVED).gt(0)];counts=active.PASSES_RECEIVED.to_numpy(float);p=counts/counts.sum();positive=p>0;n=len(p)
                data['receiving_entropy_game']=float(-np.sum(p[positive]*np.log(p[positive]))/np.log(n))
                m=pg.MIN.reindex(active.index,fill_value=0).to_numpy(float);m=m/m.sum();positive=m>0
                data['minutes_entropy_game']=float(-np.sum(m[positive]*np.log(m[positive]))/np.log(n))
                total=float(tr.PASSES_MADE.sum()+tr.PASSES_RECEIVED.sum())
                data['hub_involvement_share']=float((tr.loc[hub,'PASSES_MADE']+tr.loc[hub,'PASSES_RECEIVED'])/total) if hub in tr.index else 0
            rows.append(data)
    result=pd.DataFrame(rows);atomic_csv(result,out/'context_game_metrics.csv');return result,players,tracking,roster


def analyze_context_clutch(parent,out,players):
    season=pd.read_csv(parent/'team_season_metrics.csv');official=pd.read_csv(out/'official_team_clutch.csv').set_index('TEAM_ID');rows=[];details=[]
    for _,r in season.iterrows():
        totals=players.loc[players.TEAM_ID.eq(r.team_id)].groupby('PLAYER_ID').agg(assists=('AST','sum'),minutes=('MIN','sum'))
        clutch=pd.read_csv(out/'clutch_players'/(r.team+'.csv'))
        if clutch.PLAYER_ID.duplicated().any():raise ValueError('Duplicate clutch player '+r.team)
        clutch=clutch.set_index('PLAYER_ID');extra=set(clutch.index)-set(totals.index)
        if extra and clutch.loc[list(extra),'AST'].sum()>0:raise ValueError('Positive clutch assists for nonparticipant '+r.team)
        c=clutch.AST.reindex(totals.index,fill_value=0);cm=clutch.MIN.reindex(totals.index,fill_value=0);normal=totals.assists-c;nm=totals.minutes-cm
        if not np.isfinite(c).all() or c.lt(0).any() or (normal<0).any() or c.sum()!=official.loc[r.team_id,'AST']:raise ValueError('Clutch assists do not reconcile '+r.team)
        hub=int(r.top_eigenvector_player_id);clutch_share=float(c.get(hub,0)/c.sum());normal_share=float(normal.get(hub,0)/normal.sum())
        def unbiased(v):
            v=np.asarray(v,float);N=len(v);total=v.sum();h=float(np.sum(v*(v-1))/(total*(total-1)))
            return (h-1/N)/(1-1/N)
        ch=unbiased(c);nh=unbiased(normal)
        rows.append(dict(team=r.team,passing_style=r.passing_style,clutch_assists=float(c.sum()),nonclutch_assists=float(normal.sum()),
            clutch_hub_assist_share=clutch_share,nonclutch_hub_assist_share=normal_share,hub_share_change=clutch_share-normal_share,
            hub_minutes_share_change=float(cm.get(hub,0)/cm.sum()-nm.get(hub,0)/nm.sum()),
            clutch_assist_hhi_unbiased=ch,nonclutch_assist_hhi_unbiased=nh))
        for pid in totals.index:details.append(dict(team=r.team,player_id=int(pid),clutch_assists=float(c.loc[pid]),nonclutch_assists=float(normal.loc[pid]),clutch_minutes=float(cm.loc[pid])))
    frame=pd.DataFrame(rows);atomic_csv(frame,out/'clutch_assist_proxy.csv');atomic_csv(pd.DataFrame(details),out/'clutch_player_counts.csv')
    result=paired_context_test(frame,'nonclutch_hub_assist_share','clutch_hub_assist_share','clutch_assist_proxy_primary')
    result['proxy']='Assist share, not all-pass concentration';return frame,result


def analyze_context_volatility(games,season,out):
    rows=[];residuals=[]
    for team,g in games.groupby('team'):
        controls=['opponent_strength','home_game','back_to_back','game_number','hub_absent','other_key_absence']
        controls=[c for c in controls if g[c].nunique()>1]
        X=np.column_stack([np.ones(len(g)),g[controls].to_numpy(float)]);y=(100*g.PTS/g.POSS).to_numpy(float)
        if np.linalg.matrix_rank(X)<X.shape[1]:raise ValueError('Collinear volatility controls '+team)
        b=np.linalg.lstsq(X,y,rcond=None)[0];e=y-X@b
        rows.append(dict(team=team,adjusted_offense_volatility=float(np.sqrt(e@e/(len(g)-len(b)))),raw_offense_volatility=float(y.std(ddof=1)),
            mean_off_rating=float(y.mean()),game_count=len(g),controls=len(b)))
        for game,error in zip(g.GAME_ID,e):residuals.append(dict(team=team,GAME_ID=game,residual_off_rating=float(error)))
    frame=pd.DataFrame(rows).merge(season[['team','passing_style','eigenvector_concentration','network_player_count']],on='team',validate='one_to_one')
    atomic_csv(frame,out/'offensive_volatility.csv');atomic_csv(pd.DataFrame(residuals),out/'offensive_residuals.csv');return frame


def analyze_secondary_creator(parent,out,games,players,tracking,roster,season):
    stable=set(players.groupby('PLAYER_ID').TEAM_ID.nunique().loc[lambda s:s.eq(1)].index);elig=[];all_games=[];summaries=[]
    from scipy.stats import t
    for _,s in season.iterrows():
        team=s.team;tid=int(s.team_id);hub=int(s.top_eigenvector_player_id);own=players.loc[players.TEAM_ID.eq(tid)]
        candidates=own.loc[own.PLAYER_ID.ne(hub)&own.PLAYER_ID.isin(stable)].groupby('PLAYER_ID').agg(assists=('AST','sum'),gp=('GAME_ID','size'),name=('PLAYER_NAME','first'))
        candidates=candidates.loc[candidates.gp.ge(20)].sort_values(['assists'],ascending=False)
        if candidates.empty:continue
        secondary=int(candidates.index[0]);g=games.loc[games.team.eq(team)&games.hub_present.eq(1)&games.receiving_coverage].copy()
        appeared=set(own.loc[own.PLAYER_ID.eq(secondary),'GAME_ID']);g['secondary_absent']=(~g.GAME_ID.isin(appeared)).astype(int)
        yes=g.loc[g.secondary_absent.eq(0)];no=g.loc[g.secondary_absent.eq(1)];eligible=len(yes)>=20 and len(no)>=10
        elig.append(dict(team=team,hub_id=hub,secondary_id=secondary,secondary_name=candidates.iloc[0]['name'],both_present=len(yes),secondary_absent_hub_present=len(no),eligible=eligible))
        if not eligible:continue
        participants_yes=set(own.loc[own.GAME_ID.isin(yes.GAME_ID),'PLAYER_ID']);participants_no=set(own.loc[own.GAME_ID.isin(no.GAME_ID),'PLAYER_ID'])
        core=sorted((participants_yes&participants_no)-{secondary});baseR=np.zeros(len(core));baseM=np.zeros(len(core))
        for _,r in yes.iterrows():
            baseR+=tracking[(tid,str(r.GAME_DATE)[:10])].PASSES_RECEIVED.reindex(core,fill_value=0).to_numpy(float)
            baseM+=roster[(tid,r.GAME_ID)].MIN.reindex(core,fill_value=0).to_numpy(float)
        records=[]
        for _,r in g.iterrows():
            count=tracking[(tid,str(r.GAME_DATE)[:10])].PASSES_RECEIVED.reindex(core,fill_value=0).to_numpy(float)
            minute=roster[(tid,r.GAME_ID)].MIN.reindex(core,fill_value=0).to_numpy(float)
            refR=baseR-count if r.secondary_absent==0 else baseR;refM=baseM-minute if r.secondary_absent==0 else baseM
            row=r.to_dict();row.update(secondary_id=secondary,secondary_name=candidates.iloc[0]['name'],
                remaining_pattern_jsd=receiving_share_js_divergence(count,refR),remaining_minutes_jsd=receiving_share_js_divergence(minute,refM))
            records.append(row)
        data=pd.DataFrame(records);controls=['remaining_minutes_jsd','hub_minutes_share','home_game','opponent_strength','game_number','back_to_back']
        controls=[c for c in controls if data[c].nunique()>1]
        fit=fit_eigenvector_performance_model(data,'remaining_pattern_jsd',['secondary_absent',*controls],'secondary_creator_absence')
        effect=next(v for v in fit if v['term']=='secondary_absent');se=effect['standard_error_hac7'];critical=t.ppf(.975,len(data)-len(controls)-2)
        summaries.append(dict(team=team,passing_style=s.passing_style,eigenvector_concentration=s.eigenvector_concentration,
            secondary_name=candidates.iloc[0]['name'],both_present=len(yes),secondary_absent_hub_present=len(no),
            adjusted_disruption_increase=effect['coefficient'],ci95_low=effect['coefficient']-critical*se,ci95_high=effect['coefficient']+critical*se))
        all_games.append(data)
    frame=pd.DataFrame(summaries);atomic_csv(frame,out/'secondary_creator_team_metrics.csv');atomic_csv(pd.DataFrame(elig),out/'secondary_creator_eligibility.csv')
    if all_games:atomic_csv(pd.concat(all_games,ignore_index=True),out/'secondary_creator_game_metrics.csv')
    return frame


def analyze_pair_persistence(parent,out,players,season):
    prior=pd.read_csv(out/'prior_player_games.csv',dtype={'GAME_ID':str});rows=[];audits=[]
    plan=pd.read_csv(out/'profile_download_plan.csv');plan=plan.loc[plan.kind.eq('prior')]
    for _,s in season.iterrows():
        team=s.team;ids=sorted(map(int,plan.loc[plan.team.eq(team),'player_id']));index={p:i for i,p in enumerate(ids)};a=np.zeros((len(ids),len(ids)));complete={}
        for pid in ids:
            path=out/'profiles/prior'/team/(str(pid)+'.csv');meta=json.loads(path.with_suffix('.json').read_text());complete[pid]=bool(meta['complete']);f=pd.read_csv(path)
            for v in f.to_dict('records'):
                receiver=int(v['PASS_TEAMMATE_PLAYER_ID'])
                if receiver in index:a[index[pid],index[receiver]]+=v['PASS']
        current=pd.read_csv(parent/'networks'/team/'passing_matrix.csv',index_col=0);current.index=current.index.astype(int);current.columns=current.columns.astype(int)
        matrices=[]
        for source in [prior,players]:
            own=source.loc[source.TEAM_ID.eq(s.team_id)&source.PLAYER_ID.isin(ids)]
            minutes=own.pivot(index='GAME_ID',columns='PLAYER_ID',values='MIN').reindex(columns=ids,fill_value=0).fillna(0)
            appeared=own.assign(appeared=1).pivot(index='GAME_ID',columns='PLAYER_ID',values='appeared').reindex(columns=ids,fill_value=0).fillna(0)
            M=minutes.to_numpy(float);A=appeared.to_numpy(float);matrices.append((A.T@A,M.T@M))
        for i,pid in enumerate(ids):
            for j in range(i+1,len(ids)):
                other=ids[j];g0=matrices[0][0][i,j];g1=matrices[1][0][i,j]
                if not complete[pid] or not complete[other] or g0<=0 or g1<=0:continue
                old=a[i,j]+a[j,i];new=float(current.loc[pid,other]+current.loc[other,pid])
                rows.append(dict(team=team,player_id=f'{pid}:{other}',player_1=pid,player_2=other,passing_style=s.passing_style,
                    prior_pair_passes=old,current_pair_passes=new,prior_shared_games=int(g0),current_shared_games=int(g1),
                    prior_passes_per_shared_game=old/g0,current_passes_per_shared_game=new/g1,
                    prior_minutes_opportunity=float(np.sqrt(matrices[0][1][i,j]/g0)),current_minutes_opportunity=float(np.sqrt(matrices[1][1][i,j]/g1)),
                    reliability_weight=min(g0,g1)))
        audits.append(dict(team=team,retained_players=len(ids),unavailable_prior_profiles=[p for p in ids if not complete[p]]))
    frame=pd.DataFrame(rows);atomic_csv(frame,out/'pair_persistence_metrics.csv');atomic_json(dict(teams=audits),out/'pair_persistence_audit.json');return frame


def analyze_anonymous_identity(fingerprints,out):
    early=fingerprints.loc[fingerprints.period.eq('early')&fingerprints.complete].set_index('team')
    late=fingerprints.loc[fingerprints.period.eq('late')&fingerprints.complete].set_index('team')
    teams=sorted(set(early.index)&set(late.index));train=early.loc[teams,CONTEXT_FEATURES].to_numpy(float);test=late.loc[teams,CONTEXT_FEATURES].to_numpy(float)
    mean=train.mean(0);sd=train.std(0);keep=sd>1e-12
    X=(train[:,keep]-mean[keep])/sd[keep];Y=(test[:,keep]-mean[keep])/sd[keep]
    distance=np.sum((Y[:,None,:]-X[None,:,:])**2,axis=2);ranking=np.argsort(distance,axis=1);actual=np.arange(len(teams));pred=ranking[:,0]
    accuracy=float(np.mean(pred==actual));top3=float(np.mean(np.any(ranking[:,:3]==actual[:,None],axis=1)))
    rng=np.random.default_rng(2026);permuted=np.array([np.mean(rng.permutation(len(teams))[pred]==actual) for _ in range(10000)])
    p=float((1+np.sum(permuted>=accuracy-1e-12))/(1+len(permuted)))
    rows=[dict(team=teams[i],prediction=teams[pred[i]],correct=bool(pred[i]==i),true_team_rank=int(np.flatnonzero(ranking[i]==i)[0]+1)) for i in actual]
    atomic_csv(pd.DataFrame(rows),out/'anonymous_team_predictions.csv')
    atomic_csv(pd.DataFrame(distance,index=teams,columns=teams),out/'anonymous_team_distances.csv',index=True)
    result=dict(model='anonymous_identity_primary',n=len(teams),effect=accuracy,p_value=p,top1_accuracy=accuracy,top3_accuracy=top3,
        chance_top1=1/len(teams),chance_top3=3/len(teams),features=[x for x,k in zip(CONTEXT_FEATURES,keep) if k],
        training='Early period only; feature scaling fit to early networks',testing='Later-period networks; one prediction per team',
        inference='10,000 seeded label permutations; no player identities or hyperparameter tuning')
    atomic_json(result,out/'anonymous_identity_results.json');return result


def analyze_role_separation(players,season,out):
    rows=[]
    for _,s in season.iterrows():
        g=players.loc[players.TEAM_ID.eq(s.team_id)].groupby('PLAYER_ID').agg(assists=('AST','sum'),points=('PTS','sum'),fga=('FGA','sum'),minutes=('MIN','sum'),name=('PLAYER_NAME','first'))
        value=receiving_share_js_divergence(g.assists.to_numpy(float),g.points.to_numpy(float));minute_hhi,_=normalized_share_concentration(g.minutes)
        top_ast=g.assists.idxmax();top_pts=g.points.idxmax();hub=int(s.top_eigenvector_player_id)
        rows.append(dict(team=s.team,passing_style=s.passing_style,eigenvector_concentration=s.eigenvector_concentration,
            role_separation_jsd=value,minutes_concentration=minute_hhi,player_count=len(g),
            top_assister=g.loc[top_ast,'name'],top_scorer=g.loc[top_pts,'name'],same_assist_scoring_leader=bool(top_ast==top_pts),
            passing_hub_is_top_scorer=bool(hub==top_pts)))
    frame=pd.DataFrame(rows);atomic_csv(frame,out/'playmaking_scoring_roles.csv');return frame


def plot_eight_context(primary,home,clutch,restplot,volatility,secondary,pairplot,identity,roleplot,out):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    titles=['1. Home versus away passing entropy','2. Clutch reliance: assist-share proxy','3. Back-to-backs and receiving entropy',
        '4. Passing concentration and offensive volatility','5. Secondary-creator absence and disruption',
        '6. Partnership persistence across seasons','7. Anonymous team identification','8. Separation of playmaking and scoring roles']
    def scatter(ax,data,x,y,labels=False,weighted=False):
        for style,color,marker in [('more_balanced','#2473b8','o'),('more_star_centered','#e18a24','^')]:
            g=data.loc[data.passing_style.eq(style)];ax.scatter(g[x],g[y],color=color,marker=marker,s=42 if len(data)<40 else 10,alpha=.9 if len(data)<40 else .23,edgecolors='none',rasterized=len(data)>100)
            if labels:
                for _,v in g.iterrows():ax.annotate(v.team,(v[x],v[y]),xytext=(4,4),textcoords='offset points',fontsize=7)
        X=np.column_stack([np.ones(len(data)),data[x]]);w=data.analysis_weight.to_numpy() if weighted else np.ones(len(data));b=np.linalg.lstsq(X*np.sqrt(w)[:,None],data[y].to_numpy()*np.sqrt(w),rcond=None)[0]
        xx=np.linspace(data[x].min(),data[x].max(),100);ax.plot(xx,b[0]+b[1]*xx,color='#333333',lw=1.7)
    def draw(ax,topic):
        r=primary.loc[primary.topic.eq(topic)].iloc[0]
        if topic in [1,2]:
            d=home.loc[home.complete] if topic==1 else clutch
            cols=['away_entropy','home_entropy'] if topic==1 else ['nonclutch_hub_assist_share','clutch_hub_assist_share'];scale=1 if topic==1 else 100
            for _,v in d.iterrows():
                color='#2473b8' if v.passing_style=='more_balanced' else '#e18a24'
                ax.plot([0,1],[v[cols[0]]*scale,v[cols[1]]*scale],marker='o',lw=.8,alpha=.65,color=color,markersize=3)
            ax.set_xticks([0,1],['Away','Home'] if topic==1 else ['Non-clutch','Clutch']);ax.set_xlim(-.2,1.2)
            ax.set_ylabel('Conditional entropy on matched player sets' if topic==1 else "Passing hub’s share of team assists (%)")
        elif topic==3:
            values=[restplot.loc[restplot.back_to_back.eq(v),'adjusted_entropy'] for v in [0,1]]
            ax.boxplot(values,tick_labels=['At least one day off','Back-to-back'],showfliers=False)
            ax.set_ylabel('Receiving entropy after context controls')
        elif topic==4:
            scatter(ax,volatility,'eigenvector_concentration','adjusted_offense_volatility',True)
            ax.set_xlabel('Eigenvector passing concentration');ax.set_ylabel('Adjusted game offensive-rating standard deviation')
        elif topic==5:
            scatter(ax,secondary,'eigenvector_concentration','adjusted_disruption_increase',True);ax.axhline(0,color='#888888',ls='--',lw=1)
            ax.set_xlabel('Season passing concentration');ax.set_ylabel('Adjusted pattern disruption when secondary creator is absent')
        elif topic==6:
            scatter(ax,pairplot,'x_residual','y_residual',weighted=True)
            ax.set_xlabel('2024–25 pair pass rate after minutes controls');ax.set_ylabel('2025–26 pair pass rate after minutes controls')
        elif topic==7:
            xx=np.array([0,1]);ax.bar(xx-.16,[identity['top1_accuracy']*100,identity['top3_accuracy']*100],.32,label='Observed',color='#2473b8')
            ax.bar(xx+.16,[identity['chance_top1']*100,identity['chance_top3']*100],.32,label='Chance',color='#bbbbbb')
            ax.set_xticks(xx,['Top-1 match','Top-3 match']);ax.set_ylabel('Held-out later-period accuracy (%)');ax.set_ylim(0,100);ax.legend(frameon=False)
        elif topic==8:
            scatter(ax,roleplot,'x_residual','y_residual',True)
            ax.set_xlabel('Passing concentration after minutes/player-count controls');ax.set_ylabel('Assist-versus-scoring role separation above/below expected')
        stat=f'N={int(r.n)} | p={r.p_value:.3g}'
        if topic in [1,2]:stat+=f' | mean change={r.effect:.4f}'
        elif topic==7:stat+=f' | top-1 accuracy={identity["top1_accuracy"]:.1%}'
        elif np.isfinite(r.get('partial_r',np.nan)):stat+=f' | r={r.partial_r:.3f}'
        ax.set_title(titles[topic-1]+'\n'+stat,fontsize=10.5,pad=11);ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False)
    fig,axes=plt.subplots(4,2,figsize=(18,21))
    for topic,ax in enumerate(axes.flat,1):draw(ax,topic)
    fig.suptitle('Eight NBA context and continuity studies: 2025–26',fontsize=18,y=.99)
    fig.legend(handles=[Line2D([0],[0],ls='',marker='o',color='#2473b8',label='More balanced where shown'),Line2D([0],[0],ls='',marker='^',color='#e18a24',label='More star-centered where shown')],loc='upper center',bbox_to_anchor=(.5,.967),ncol=2,frameon=False)
    fig.text(.5,.013,'Clutch is an ASSIST-SHARE proxy, not full passing data. Rest uses game receiving entropy. Partnership persistence adds 2024–25.\n'
        'Anonymous identification trains on early-period network features and tests later-period networks without player identities.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.045,1,.94));season_topics_spread_labels(fig)
    fig.savefig(out/'eight_context_topics.png',dpi=180);fig.savefig(out/'eight_context_topics.pdf');plt.close(fig)
    for topic in range(1,9):
        fig,ax=plt.subplots(figsize=(11,8));draw(ax,topic);fig.tight_layout();season_topics_spread_labels(fig)
        fig.savefig(out/f'topic_{topic}.png',dpi=180);fig.savefig(out/f'topic_{topic}.pdf');plt.close(fig)


def run_eight_context_topics(argv):
    parser=argparse.ArgumentParser(description='Eight context/continuity questions: 2025-26, plus 2024-25 for persistence')
    parser.add_argument('--eight-context-topics',action='store_true');parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--use-saved-context',action='store_true');args=parser.parse_args(argv)
    parent=args.output_dir.expanduser()/'2025-26/team_season_topics';out=parent/'eight_context_topics';out.mkdir(parents=True,exist_ok=True)
    if not args.use_saved_context:download_eight_context(parent,out)
    season=pd.read_csv(parent/'team_season_metrics.csv');home,fingerprints=build_context_split_networks(parent,out)
    games,players,tracking,roster=build_context_game_data(parent,out);clutch,clutch_result=analyze_context_clutch(parent,out,players)
    volatility=analyze_context_volatility(games,season,out)
    secondary=analyze_secondary_creator(parent,out,games,players,tracking,roster,season)
    pairs=analyze_pair_persistence(parent,out,players,season);identity=analyze_anonymous_identity(fingerprints,out);roles=analyze_role_separation(players,season,out)
    primary=[];sensitivities=[]
    primary.append(dict(paired_context_test(home.loc[home.complete],'away_entropy','home_entropy','home_away_primary'),topic=1))
    primary.append(dict(clutch_result,topic=2))
    home_adjust=season_topics_fit(home.loc[home.complete],'minutes_concentration_change','home_minus_away',
        'home_context_sensitivity','home',['minutes_concentration_change','opponent_strength_change'])
    atomic_csv(pd.DataFrame(home_adjust),out/'home_context_sensitivity.csv')
    home_other=[paired_context_test(home.loc[home.complete],'away_ec_concentration','home_ec_concentration','home_centrality_secondary'),
        paired_context_test(home.loc[home.complete],'away_top_pair_share','home_top_pair_share','home_partnership_secondary')]
    atomic_csv(pd.DataFrame(home_other),out/'home_secondary_tests.csv')
    cm=season_topics_fit(clutch,'hub_minutes_share_change','hub_share_change','clutch_minutes_sensitivity','clutch')
    atomic_csv(pd.DataFrame(cm),out/'clutch_minutes_sensitivity.csv')
    rest=games.loc[games.receiving_coverage&games.days_rest.notna()].copy()
    rest_controls=['minutes_entropy_game','usual_minutes_absent_share','hub_absent','home_game','opponent_strength','game_number']
    rr,_=fit_defense_disruption(rest,['back_to_back',*rest_controls],'rest_primary','receiving_entropy_game')
    primary_rest=next(v for v in rr if v['term']=='back_to_back');primary.append(dict(topic=3,n=primary_rest['n'],effect=primary_rest['coefficient'],
        ci95_low=primary_rest['ci95_low'],ci95_high=primary_rest['ci95_high'],p_value=primary_rest['p_value_twoway'],
        partial_r=primary_rest['partial_r'],partial_r_squared=primary_rest['partial_r_squared'],model='rest_primary'))
    for r in rr:r.update(topic=3)
    sensitivities.extend(rr)
    hub=rest.loc[rest.hub_present.eq(1)]
    hr,_=fit_defense_disruption(hub,['back_to_back','hub_minutes_share',*rest_controls],'rest_hub_involvement_sensitivity','hub_involvement_share');sensitivities.extend(hr)
    fe=pd.get_dummies(rest.team,drop_first=True,dtype=float);Z=np.column_stack([np.ones(len(rest)),rest[rest_controls],fe]);X=np.column_stack([rest.back_to_back,Z]);beta=np.linalg.lstsq(X,rest.receiving_entropy_game,rcond=None)[0]
    restplot=rest[['team','GAME_ID','back_to_back']].copy();restplot['adjusted_entropy']=rest.receiving_entropy_game.to_numpy()-Z@beta[1:];atomic_csv(restplot,out/'rest_adjusted_plot_data.csv')
    v,vplot=six_team_fit(volatility,'eigenvector_concentration','adjusted_offense_volatility',[],'volatility_primary');primary.append(dict(v,topic=4,effect=v['coefficient']))
    sv,sp=six_team_fit(secondary,'eigenvector_concentration','adjusted_disruption_increase',[],'secondary_creator_primary');primary.append(dict(sv,topic=5,effect=sv['coefficient']))
    p,pplot=six_player_cluster_fit(pairs,'prior_passes_per_shared_game','current_passes_per_shared_game',['prior_minutes_opportunity','current_minutes_opportunity'],'partnership_persistence_primary','reliability_weight')
    p['inference']='Two-way team/player-pair clustered t';p['unique_pairs']=p.pop('unique_players');primary.append(dict(p,topic=6,effect=p['coefficient']))
    atomic_csv(pplot,out/'persistence_adjusted_plot_data.csv')
    persistence_checks=[]
    for subset,label in [(pairs.loc[pairs.prior_pair_passes.ge(20)],'baseline_20_pass_sensitivity'),
        (pairs.loc[pairs.prior_shared_games.ge(10)&pairs.current_shared_games.ge(10)],'ten_joint_games_sensitivity')]:
        check,_=six_player_cluster_fit(subset,'prior_passes_per_shared_game','current_passes_per_shared_game',
            ['prior_minutes_opportunity','current_minutes_opportunity'],label,'reliability_weight')
        check['inference']='Two-way team/pair clustered t';persistence_checks.append(check)
    atomic_csv(pd.DataFrame(persistence_checks),out/'persistence_sensitivities.csv')
    primary.append(dict(identity,topic=7))
    role,roleplot=six_team_fit(roles,'eigenvector_concentration','role_separation_jsd',['minutes_concentration','player_count'],'role_separation_primary');primary.append(dict(role,topic=8,effect=role['coefficient']))
    atomic_csv(roleplot,out/'role_separation_adjusted_plot_data.csv')
    result=pd.DataFrame(primary).sort_values('topic');result['p_holm_eight']=np.nan;running=0
    for rank,idx in enumerate(result.p_value.sort_values().index):
        running=max(running,min(1,float(result.loc[idx,'p_value'])*(8-rank)));result.loc[idx,'p_holm_eight']=running
    atomic_csv(result,out/'eight_primary_results.csv');atomic_csv(pd.DataFrame(sensitivities),out/'game_model_results.csv')
    plot_eight_context(result,home,clutch,restplot,volatility,secondary,pplot,identity,roleplot,out)
    atomic_json(dict(season='2025-26',persistence_seasons=['2024-25','2025-26'],home_away_teams=int(home.complete.sum()),
        clutch_definition='Last five minutes of fourth quarter or overtime with score margin at most five, using NBA clutch endpoint',
        clutch_proxy='Selected passing hub share of assists, clutch vs non-clutch; full all-pass clutch networks unavailable in supported endpoint',
        rest_metric='Actual game receiver entropy, not conditional pairwise entropy; hub involvement tested separately',
        secondary_creator='Highest-assist stable-team teammate excluding passing hub; may be the actual assist leader if network hub is a receiver',
        anonymous_features=CONTEXT_FEATURES,anonymous_validation='No player identities; scaling fit only on early period; later period never used for feature selection or tuning',
        partner_measure='Passes per shared game appearance, with minutes-opportunity controls and joint-appearance reliability weights; not actual shared court minutes',
        role_separation='Base-2 JSD of player assist and point-share distributions; controls minutes concentration and participant count',
        multiple_testing='Holm correction across eight exploratory primary tests',no_minutes_cutoff=True,
        limitations=['Observational follow-ups after prior topic exploration','Assist proxy does not measure every clutch pass',
            'Receiving entropy does not measure conditional receiver choices or pass order','Schedule/rest associations do not isolate physiological fatigue',
            'One held-out snapshot per team gives a small identification test','Partner persistence covers only retained pairs with shared appearances in both seasons',
            'Secondary absence reasons are unknown; common-player panel and minutes controls do not remove all confounding']),out/'analysis_audit.json')
    print(result[['topic','n','effect','p_value','p_holm_eight']].to_string(index=False),flush=True)
    print(f'Finished eight context studies: {out}',flush=True);return 0


# ---------------------------------------------------------------------
# SEPARATE 2025-26 PASSING-VOLUME REGRESSIONS BY FIXED EIGENVECTOR STYLE
# python teamgraphs.py --passing-volume-by-style
# ---------------------------------------------------------------------
def run_passing_volume_by_style(argv):
    from scipy.stats import t
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    parser=argparse.ArgumentParser(description='2025-26 passing volume vs potential assists, separate regressions by fixed passing style')
    parser.add_argument('--passing-volume-by-style',action='store_true');parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT_DIR)
    args=parser.parse_args(argv);parent=args.output_dir.expanduser()/'2025-26/team_season_topics'
    out=parent/'passing_volume_by_style';out.mkdir(parents=True,exist_ok=True)
    frame=pd.read_csv(parent/'five_additional_topics/team_season_metrics.csv')
    source=pd.read_csv(parent/'five_additional_topics/official_team_passing.csv').set_index('TEAM_ID')
    reference=pd.read_csv(parent/'team_season_metrics.csv').set_index('team')
    if len(frame)!=30 or frame.team.duplicated().any() or not frame.games.eq(82).all():raise ValueError('Expected 30 full team-seasons')
    if frame.passing_style.value_counts().to_dict()!={'more_balanced':15,'more_star_centered':15}:raise ValueError('Expected fixed 15/15 style split')
    if not frame.passing_style.eq(frame.team.map(reference.passing_style)).all():raise ValueError('Style labels differ from original analysis')
    np.testing.assert_allclose(frame.passes_per100,100*frame.team_id.map(source.PASSES_MADE)/frame.possessions)
    np.testing.assert_allclose(frame.potential_assists_per100,100*frame.team_id.map(source.POTENTIAL_AST)/frame.possessions)
    original=frame.copy()
    daily=pd.read_csv(parent/'roles_and_familiarity/familiarity_game_metrics.csv',dtype={'GAME_ID':str})
    if len(daily)!=2460 or daily.duplicated(['team','GAME_ID']).any():raise ValueError('Expected unique complete season team-game index')
    valid=daily.loc[daily.passing_complete.astype(str).str.lower().eq('true')].copy()
    if valid.POSS.le(0).any() or not np.isfinite(valid[['passes_made','potential_assists','POSS']]).all().all():raise ValueError('Invalid matched game totals')
    totals=valid.groupby('team').agg(valid_games=('GAME_ID','size'),valid_passes=('passes_made','sum'),valid_potential=('potential_assists','sum'),valid_possessions=('POSS','sum'))
    if len(totals)!=30:raise ValueError('Missing team in valid game aggregation')
    frame['original_passes_per100']=frame.passes_per100;frame['original_potential_assists_per100']=frame.potential_assists_per100
    frame['valid_tracking_games']=frame.team.map(totals.valid_games)
    frame['valid_tracking_passes']=frame.team.map(totals.valid_passes);frame['valid_tracking_potential_assists']=frame.team.map(totals.valid_potential)
    frame['valid_tracking_possessions']=frame.team.map(totals.valid_possessions)
    frame['passes_per100']=100*frame.valid_tracking_passes/frame.valid_tracking_possessions
    frame['potential_assists_per100']=100*frame.valid_tracking_potential_assists/frame.valid_tracking_possessions
    atomic_csv(daily.loc[~daily.passing_complete.astype(str).str.lower().eq('true'),['team','GAME_ID','GAME_DATE','passing_data_error']],out/'excluded_game_records.csv')
    original_rows=[]
    for style in ['more_balanced','more_star_centered']:
        original_rows.extend(season_topics_fit(original.loc[original.passing_style.eq(style)],'passes_per100','potential_assists_per100','original_uncorrected_'+style,'original_season_totals'))
    atomic_csv(pd.DataFrame(original_rows),out/'original_total_comparison.csv')
    x='passes_per100';y='potential_assists_per100';rows=[];summary=[];covariances={}
    for style in ['more_balanced','more_star_centered']:
        d=frame.loc[frame.passing_style.eq(style)].copy()
        fits=season_topics_fit(d,x,y,style,'passing_volume_by_style');rows.extend(fits)
        slope=next(r for r in fits if r['term']==x);intercept=next(r for r in fits if r['term']=='intercept')
        X=np.column_stack([np.ones(len(d)),d[x]]);v=d[y].to_numpy(float);b=np.linalg.lstsq(X,v,rcond=None)[0]
        inv=np.linalg.inv(X.T@X);res=v-X@b;leverage=np.einsum('ij,jk,ik->i',X,inv,X)
        score=X*(res/(1-leverage))[:,None];cov=inv@(score.T@score)@inv;covariances[style]=cov
        summary.append(dict(passing_style=style,n=len(d),intercept=intercept['coefficient'],slope=slope['coefficient'],
            slope_per10passes=10*slope['coefficient'],pearson_r=slope['pearson_r'],r_squared=slope['r_squared'],
            p_slope_hc3=slope['p_value_hc3'],p_pearson_ols=slope['p_value_ols'],
            slope_ci95_low=slope['slope_ci95_low'],slope_ci95_high=slope['slope_ci95_high'],
            min_passes_per100=d[x].min(),max_passes_per100=d[x].max(),
            mean_passes_per100=d[x].mean(),mean_potential_assists_per100=d[y].mean()))
    frame['centered_passes']=frame[x]-frame[x].mean();frame['star_indicator']=frame.passing_style.eq('more_star_centered').astype(int)
    frame['passes_x_star']=frame.centered_passes*frame.star_indicator
    rows.extend(season_topics_fit(frame,x,y,'clean_pooled','passing_volume_by_style'))
    interaction=season_topics_fit(frame,x,y,'style_interaction','passing_volume_by_style',['centered_passes','star_indicator','passes_x_star']);rows.extend(interaction)
    atomic_csv(pd.DataFrame(rows),out/'regression_results.csv');s=pd.DataFrame(summary);atomic_csv(s,out/'group_summary.csv')
    for col in ['games','passes','potential_assists','possessions']:frame['original_full_'+col]=frame.team.map(original.set_index('team')[col])
    export_columns=['team','team_id','season','passing_style','style_cutoff','eigenvector_concentration',
        'valid_tracking_games','valid_tracking_passes','valid_tracking_potential_assists','valid_tracking_possessions',
        'passes_per100','potential_assists_per100','original_full_games','original_full_passes','original_full_potential_assists',
        'original_full_possessions','original_passes_per100','original_potential_assists_per100','centered_passes','star_indicator','passes_x_star']
    atomic_csv(frame[export_columns],out/'team_metrics.csv')
    fig,ax=plt.subplots(figsize=(12,8));legend=[];notes=[]
    for style,color,marker,linestyle,label in [('more_balanced','#2473b8','o','-','More balanced'),('more_star_centered','#e18a24','^','--','More star-centered')]:
        d=frame.loc[frame.passing_style.eq(style)];r=s.loc[s.passing_style.eq(style)].iloc[0]
        ax.scatter(d[x],d[y],color=color,marker=marker,s=65,edgecolors='white',linewidths=.6,zorder=3)
        for _,v in d.iterrows():ax.annotate(v.team,(v[x],v[y]),xytext=(4,4),textcoords='offset points',fontsize=8)
        xx=np.linspace(d[x].min(),d[x].max(),150);line=r.intercept+r.slope*xx;design=np.column_stack([np.ones(len(xx)),xx])
        se=np.sqrt(np.maximum(np.einsum('ij,jk,ik->i',design,covariances[style],design),0));critical=t.ppf(.975,len(d)-2)
        ax.plot(xx,line,color=color,lw=2,ls=linestyle);ax.fill_between(xx,line-critical*se,line+critical*se,color=color,alpha=.10)
        legend.append(Line2D([0],[0],color=color,marker=marker,ls=linestyle,label=label))
        notes.append(f'{label}: N={int(r.n)}, r={r.pearson_r:.3f}, R²={r.r_squared:.3f}, HC3 slope p={r.p_slope_hc3:.3g}')
    difference=next(r for r in interaction if r['term']=='passes_x_star')
    fig.suptitle('NBA 2025–26: passing volume and potential assists by passing structure',fontsize=15,y=.985)
    fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(.5,.94),ncol=2,frameon=False)
    fig.text(.5,.86,'\n'.join(notes)+f'\nDifference between slopes: HC3 p={difference["p_value_hc3"]:.3g}',ha='center',va='top',fontsize=10)
    ax.set_xlabel('Passes made per 100 team possessions');ax.set_ylabel('Potential assists per 100 team possessions')
    ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False)
    fig.text(.5,.02,'One point per team-season; rates use matching valid-game possessions (23 teams: 82 games; 7 teams: 81).\n'
        'Separate OLS lines; shaded bands are pointwise 95% HC3 mean-response intervals.\n'
        'Fixed season eigenvector labels classify structure; centrality is not an axis. The outcome measures opportunity volume, not potential assists per pass.',ha='center',fontsize=8.5)
    fig.tight_layout(rect=(0,.08,1,.77));season_topics_spread_labels(fig)
    fig.savefig(out/'passing_volume_by_style.png',dpi=180);fig.savefig(out/'passing_volume_by_style.pdf');plt.close(fig)
    atomic_json(dict(season='2025-26',season_type='Regular Season',team_count=30,groups={'more_balanced':15,'more_star_centered':15},
        axes={'x':x,'y':y},classification='Original 30-team median eigenvector concentration split, unchanged',
        inference='Separate OLS regressions with HC3 robust t inference; slopes compared using a centered-passes by style interaction',
        source='Daily official NBA tracking aggregated over valid complete games with matching official possessions; original season totals retained separately',
        valid_team_games=len(valid),excluded_team_games=len(daily)-len(valid),
        coverage={str(int(k)):int(v) for k,v in frame.valid_tracking_games.value_counts().items()},
        data_quality='Original BOS/MIA season potential-assist totals include impossible April 1 daily counts; rebuilt from valid games; incomplete receiving records also excluded',
        no_minutes_cutoff=True,limits=['Exploratory subgroup comparison; only 15 teams per group',
            'Different correlations do not imply different slopes','More passes create more chances to register potential assists',
            'Rates share a possessions denominator; association does not establish causality or greater efficiency per pass']),out/'analysis_audit.json')
    print(s.to_string(index=False),flush=True);print('Slope difference:',difference,flush=True)
    print(f'Finished separate style regressions: {out}',flush=True);return 0


_original_run_decade_with_series = run_decade


def run_four_tactical_questions(argv):
    """Saved-data feasibility audit and explicitly labeled tactical follow-ups."""
    from scipy.stats import ttest_rel,ttest_ind
    import matplotlib.pyplot as plt
    parser=argparse.ArgumentParser(description='Four tactical questions: supported tests and explicit data limitations')
    parser.add_argument('--four-tactical-questions',action='store_true')
    parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT_DIR)
    args=parser.parse_args(argv)
    parent=args.output_dir.expanduser()/'2025-26/team_season_topics'
    out=parent/'four_tactical_questions';out.mkdir(parents=True,exist_ok=True)
    games=pd.read_csv(parent/'eight_context_topics/context_game_metrics.csv',dtype={'GAME_ID':str})
    clutch=pd.read_csv(parent/'eight_context_topics/clutch_assist_proxy.csv')
    hubs=pd.read_csv(parent/'hub_absence_resilience/team_season_metrics.csv')
    season=pd.read_csv(parent/'team_season_metrics.csv')
    styles=season.set_index('team').passing_style.to_dict()
    status=[dict(question=1,status='Unavailable exact test',reason='Passing aggregates have no event timestamps or score-margin filter. Final wins/losses cannot substitute for leading/trailing possessions.'),
        dict(question=2,status='Receiver-entropy substitute only',reason='Daily receiver totals available, but full daily directed networks are not saved. Receiver entropy is not H(receiver | passer).'),
        dict(question=3,status='Assist-share substitute only',reason='Clutch assist counts available, but all-pass clutch networks and their eigenvector centrality are unavailable.'),
        dict(question=4,status='Supported',reason='Full season and hub-present directed networks allow subtraction; matched nonhub panel removes hub mechanically in both conditions. Offensive ratings are saved.')]
    atomic_csv(pd.DataFrame(status),out/'availability.csv')
    results=[];comparisons=[];plotrows=[]
    valid=games.loc[games.receiving_coverage.eq(True)].dropna(subset=['receiving_entropy_game','opponent_strength']).copy()
    for style,d in [('pooled',valid),*list(valid.groupby('passing_style'))]:
        fits,_=fit_defense_disruption(d,['opponent_strength'],'defense_entropy_'+style,'receiving_entropy_game')
        results.extend(dict(r,question=2,passing_style=style) for r in fits)
        # Plot within-team centered values; no minutes or roster covariates.
        d=d.copy();d['x']=d.opponent_strength-d.groupby('team').opponent_strength.transform('mean')
        d['y']=d.receiving_entropy_game-d.groupby('team').receiving_entropy_game.transform('mean')
        if style!='pooled':plotrows.append(d[['team','GAME_ID','passing_style','x','y']])
    defensive_results=pd.DataFrame(results);atomic_csv(defensive_results,out/'defense_receiver_entropy_regressions.csv')
    atomic_csv(valid,out/'defense_receiver_entropy_games.csv')
    for style,d in [('pooled',clutch),*list(clutch.groupby('passing_style'))]:
        test=ttest_rel(d.clutch_hub_assist_share,d.nonclutch_hub_assist_share)
        comparisons.append(dict(question=3,comparison=style,outcome='clutch_minus_nonclutch_hub_assist_share',n=len(d),effect=d.hub_share_change.mean(),p_value=test.pvalue))
    test=ttest_ind(clutch.loc[clutch.passing_style.eq('more_star_centered'),'hub_share_change'],
        clutch.loc[clutch.passing_style.eq('more_balanced'),'hub_share_change'],equal_var=False)
    comparisons.append(dict(question=3,comparison='star_minus_balanced',outcome='clutch_hub_share_change',n=len(clutch),
        effect=clutch.loc[clutch.passing_style.eq('more_star_centered'),'hub_share_change'].mean()-clutch.loc[clutch.passing_style.eq('more_balanced'),'hub_share_change'].mean(),p_value=test.pvalue))
    absent_rows=[]
    for _,r in hubs.loc[hubs.primary_eligible.eq(True)].iterrows():
        full=pd.read_csv(parent/'networks'/r.team/'passing_matrix.csv',index_col=0)
        present=pd.read_csv(parent/'hub_absence_resilience/present_networks'/r.team/'passing_matrix.csv',index_col=0)
        assert full.index.astype(str).tolist()==present.index.astype(str).tolist()
        assert full.columns.astype(str).tolist()==present.columns.astype(str).tolist()
        ids=full.index.astype(int).tolist();p=present.to_numpy(float);a=full.to_numpy(float)-p
        if a.min() < -1e-8:raise ValueError('Negative absent counts '+r.team)
        a=np.maximum(a,0);hub=ids.index(int(r.hub_id))
        if a[hub,:].sum()+a[:,hub].sum()>0:raise ValueError('Absent hub has passes '+r.team)
        take=[i for i in range(len(ids)) if i!=hub]
        while True:
            pp=p[np.ix_(take,take)];aa=a[np.ix_(take,take)]
            active=(pp.sum(0)+pp.sum(1)>0)&(aa.sum(0)+aa.sum(1)>0)
            if active.all():break
            take=[i for i,keep in zip(take,active) if keep]
        if len(take)<3:raise ValueError('Too few matched nonhub players')
        hp=compute_team_entropies(pp)[1];ha=compute_team_entropies(aa)[1]
        absent_rows.append(dict(team=r.team,passing_style=styles[r.team],hub_name=r.hub_name,matched_nonhub_players=len(take),
            hub_present_games=r.present_games,hub_absent_games=r.absent_games,conditional_entropy_present=hp,conditional_entropy_absent=ha,
            entropy_loss=hp-ha,raw_offensive_penalty=r.raw_absence_penalty,adjusted_offensive_penalty=r.adjusted_absence_penalty,
            baseline_concentration=r.hub_present_concentration,original_season_concentration=r.original_season_concentration))
    absence=pd.DataFrame(absent_rows);atomic_csv(absence,out/'hub_absence_matched_metrics.csv')
    modelrows=[]
    for y in ['entropy_loss','raw_offensive_penalty','adjusted_offensive_penalty']:
        for style,d in [('pooled',absence),*list(absence.groupby('passing_style'))]:
            fit=season_topics_fit(d,'baseline_concentration',y,'absence_'+style,4)
            modelrows.extend(dict(r,passing_style=style) for r in fit)
        b=absence.loc[absence.passing_style.eq('more_balanced'),y];s=absence.loc[absence.passing_style.eq('more_star_centered'),y]
        test=ttest_ind(s,b,equal_var=False)
        comparisons.append(dict(question=4,comparison='star_minus_balanced',outcome=y,n=len(absence),effect=s.mean()-b.mean(),p_value=test.pvalue,
            balanced_n=len(b),star_n=len(s),balanced_mean=b.mean(),star_mean=s.mean()))
    interactions=[]
    for y in ['entropy_loss','raw_offensive_penalty','adjusted_offensive_penalty']:
        d=absence.copy();d['star']=d.passing_style.eq('more_star_centered').astype(float)
        d['concentration_x_star']=d.baseline_concentration*d.star
        interactions.extend(season_topics_fit(d,'baseline_concentration',y,'absence_style_interaction',4,
            ['baseline_concentration','star','concentration_x_star']))
    atomic_csv(pd.DataFrame(interactions),out/'hub_absence_style_interactions.csv')
    atomic_csv(pd.DataFrame(modelrows),out/'hub_absence_regressions.csv')
    comp=pd.DataFrame(comparisons);atomic_csv(comp,out/'paired_and_group_comparisons.csv')
    plots=pd.concat(plotrows,ignore_index=True);atomic_csv(plots,out/'defense_within_team_plot_data.csv')
    palette=[('more_balanced','#2473b8','o','Balanced'),('more_star_centered','#e18a24','^','Star-centered')]
    def draw(ax,q):
        if q==1:
            ax.axis('off');ax.text(.5,.75,'1. Leading versus trailing',ha='center',fontsize=14,transform=ax.transAxes)
            ax.text(.5,.45,'Exact test unavailable\n\nPassing data has no event times or score-state splits.\nWins versus losses would answer a different question.',ha='center',va='center',fontsize=11,transform=ax.transAxes);return
        if q==2:
            for style,color,marker,label in palette:
                d=plots.loc[plots.passing_style.eq(style)];r=defensive_results.loc[defensive_results.passing_style.eq(style)&defensive_results.term.eq('opponent_strength')].iloc[0]
                ax.scatter(d.x,d.y,s=7,color=color,alpha=.10,rasterized=True)
                x=np.linspace(d.x.min(),d.x.max(),100);ax.plot(x,r.coefficient*x,color=color,lw=2,label=f'{label}: within-team r={r.partial_r:.3f}, p={r.p_value_twoway:.3g}')
            ax.set(xlabel='Opponent defensive strength relative to team schedule mean',ylabel='Receiver entropy relative to own team mean',title='2. Defense and receiving diversity (substitute)');ax.legend(fontsize=8)
        elif q==3:
            for style,color,marker,label in palette:
                d=clutch.loc[clutch.passing_style.eq(style)]
                for _,r in d.iterrows():ax.plot([0,1],100*np.array([r.nonclutch_hub_assist_share,r.clutch_hub_assist_share]),color=color,alpha=.5,marker=marker,lw=.9)
                ax.plot([],[],color=color,label=label)
            ax.set_xticks([0,1],['Non-clutch','Clutch']);ax.set_xlim(-.15,1.15)
            ax.set(ylabel='Season passing hub share of team assists (%)',title='3. Pressure and hub assist share (substitute)');ax.legend(fontsize=8)
        else:
            y='entropy_loss' if q==4 else 'raw_offensive_penalty'
            for style,color,marker,label in palette:
                d=absence.loc[absence.passing_style.eq(style)]
                ax.scatter(d.baseline_concentration,d[y],color=color,marker=marker,label=label)
                fit=season_topics_fit(d,'baseline_concentration',y,style,4)
                slope=next(r['coefficient'] for r in fit if r['term']=='baseline_concentration');intercept=next(r['coefficient'] for r in fit if r['term']=='intercept')
                x=np.linspace(d.baseline_concentration.min(),d.baseline_concentration.max(),100);ax.plot(x,intercept+slope*x,color=color)
                for _,r in d.iterrows():ax.annotate(r.team,(r.baseline_concentration,r[y]),xytext=(3,3),textcoords='offset points',fontsize=7)
            ax.axhline(0,color='gray',lw=.7);ax.set(xlabel='Hub-present eigenvector passing concentration',ylabel='Conditional entropy: present minus absent' if q==4 else 'Offensive rating: present minus absent',title='4a. Hub absence: matched nonhub passing' if q==4 else '4b. Hub absence: offensive efficiency');ax.legend(fontsize=8)
        ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False)
    fig,axes=plt.subplots(3,2,figsize=(17,18))
    for q,ax in zip(range(1,6),axes.flat):draw(ax,q)
    axes.flat[5].axis('off');axes.flat[5].text(.05,.85,'2025–26 regular season\n\n2: team fixed effects; team/match clustered inference.\n3: paired team comparisons; assist share is not eigenvector centrality.\n4: 21 eligible teams; at least 20 present and 10 absent games.\nMatched entropy removes hub from BOTH networks.\nNo minutes or roster controls in the primary plots.\nExploratory tests; p-values are not multiplicity-adjusted.\nNo causal conclusions.',va='top',fontsize=11,transform=axes.flat[5].transAxes)
    fig.tight_layout();season_topics_spread_labels(fig);fig.savefig(out/'four_tactical_questions.png',dpi=150);fig.savefig(out/'four_tactical_questions.pdf');plt.close(fig)
    for q,name in [(2,'defense_receiver_entropy'),(3,'clutch_assist_proxy'),(4,'absence_entropy'),(5,'absence_offensive_rating')]:
        fig,ax=plt.subplots(figsize=(11,8));draw(ax,q);fig.tight_layout();season_topics_spread_labels(fig);fig.savefig(out/(name+'.pdf'));fig.savefig(out/(name+'.png'),dpi=150);plt.close(fig)
    report=['# Four tactical questions — 2025–26 NBA regular season','',pd.DataFrame(status).to_string(index=False),'','## Opponent defense / receiver entropy (not conditional entropy)',defensive_results.loc[defensive_results.term.eq('opponent_strength')].to_string(index=False),'','## Paired clutch assist proxy and absence group comparisons',comp.to_string(index=False),'','## Hub-absence regressions',pd.DataFrame(modelrows).loc[lambda d:d.term.eq('baseline_concentration')].to_string(index=False),'',
        'Score-state and clutch eigenvector questions remain untested. Assist share omits unassisted scoring and non-assist passes. Receiver entropy omits passer-conditioned choices. Clutch is final five minutes of regulation/overtime, margin at most five. Increased hub assist share can reflect increased hub playing time. Hub presence means appearance, not injury status. Nonhub matched panels avoid mechanically deleting the hub only in one condition. Aggregated entropy has unequal pass sample sizes and changes in teammate availability/minutes; no pass-count or minutes adjustment is applied here. Absent networks are full season minus hub-present networks. Primary group comparisons use fixed season classifications; concentration regression uses hub-present networks. Raw offense differences confound opponent and teammate availability; adjusted penalties from the existing saved analysis are included as a sensitivity, not called uncontrolled. Eligible absence sample is 21 teams, not all 30. Soccer data are not present, so no soccer results are claimed. All tests exploratory and unadjusted for multiple comparisons.']
    (out/'README.md').write_text('\n\n'.join(report))
    print(defensive_results.loc[defensive_results.term.eq('opponent_strength')].to_string(index=False),flush=True)
    print(comp.to_string(index=False),flush=True)
    print(pd.DataFrame(modelrows).loc[lambda d:d.term.eq('baseline_concentration')].to_string(index=False),flush=True)
    print('Saved four tactical questions to '+str(out),flush=True)
    return 0



def run_hub_successor_analysis(argv):
    """Distinguish replacement passing hubs from broader nonhub redistribution."""
    from scipy.stats import ttest_rel,binomtest
    import matplotlib.pyplot as plt
    parser=argparse.ArgumentParser(description='Hub absence: replacement hub versus balanced redistribution')
    parser.add_argument('--hub-successor-analysis',action='store_true');parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT_DIR)
    args=parser.parse_args(argv);parent=args.output_dir.expanduser()/'2025-26/team_season_topics';out=parent/'hub_successor_analysis';out.mkdir(parents=True,exist_ok=True)
    eligible=pd.read_csv(parent/'hub_absence_resilience/team_season_metrics.csv');season=pd.read_csv(parent/'team_season_metrics.csv').set_index('team')
    players=pd.read_csv(parent/'player_games.csv',dtype={'GAME_ID':str});rows=[];details=[]
    for _,r in eligible.loc[eligible.primary_eligible.eq(True)].iterrows():
        full=pd.read_csv(parent/'networks'/r.team/'passing_matrix.csv',index_col=0)
        present=pd.read_csv(parent/'hub_absence_resilience/present_networks'/r.team/'passing_matrix.csv',index_col=0)
        if full.index.astype(str).tolist()!=present.index.astype(str).tolist() or full.columns.astype(str).tolist()!=present.columns.astype(str).tolist():raise ValueError('Panel mismatch')
        ids=full.index.astype(int).tolist();p=present.to_numpy(float);a=full.to_numpy(float)-p;hub=int(r.hub_id)
        if a.min() < -1e-8:raise ValueError('Negative absent pass counts')
        a=np.maximum(a,0);take=[i for i,pid in enumerate(ids) if pid!=hub]
        while True:
            pp=p[np.ix_(take,take)];aa=a[np.ix_(take,take)];keep=(pp.sum(0)+pp.sum(1)>0)&(aa.sum(0)+aa.sum(1)>0)
            if keep.all():break
            take=[i for i,k in zip(take,keep) if k]
        panel=[ids[i] for i in take];before,s1,_,_=eigenvector_passing_concentration(pp,panel);after,s2,_,_=eigenvector_passing_concentration(aa,panel)
        if not before['network_valid'] or not after['network_valid']:raise ValueError('Invalid matched network '+r.team)
        own=players.loc[players.TEAM_ID.eq(r.team_id)];names=own.drop_duplicates('PLAYER_ID').set_index('PLAYER_ID').PLAYER_NAME.to_dict()
        played=set(own.loc[own.PLAYER_ID.eq(hub),'GAME_ID']);absstats=own.loc[~own.GAME_ID.isin(played)].groupby('PLAYER_ID').agg(games=('GAME_ID','size'),minutes=('MIN','sum'))
        top1=int(before['top_eigenvector_player_id']);top2=int(after['top_eigenvector_player_id'])
        c1=before['eigenvector_concentration'];c2=after['eigenvector_concentration'];t1=before['top_eigenvector_share'];t2=after['top_eigenvector_share']
        dc=c2-c1;dt=t2-t1
        kind='More concentrated around a replacement hub' if dc>0 and dt>0 else ('More balanced' if dc<0 and dt<0 else 'Mixed indicators')
        row=dict(team=r.team,passing_style=season.loc[r.team,'passing_style'],original_hub=r.hub_name,matched_nonhub_players=len(panel),present_games=int(r.present_games),absent_games=int(r.absent_games),
            concentration_present=c1,concentration_absent=c2,concentration_change=dc,top_share_present=t1,top_share_absent=t2,top_share_change=dt,
            leading_nonhub_before=names.get(top1,str(top1)),leading_nonhub_absent=names.get(top2,str(top2)),leader_changed=top1!=top2,
            absent_leader_id=top2,absent_leader_games=int(absstats.loc[top2,'games']),absent_leader_minutes=float(absstats.loc[top2,'minutes']),
            absent_leader_mpg=float(absstats.loc[top2,'minutes']/absstats.loc[top2,'games']),redistribution=kind,
            entropy_change=compute_team_entropies(aa)[1]-compute_team_entropies(pp)[1],adjusted_offensive_loss=r.adjusted_absence_penalty,
            centrality_total_variation=.5*sum(abs(s2[pid]-s1[pid]) for pid in panel))
        rows.append(row)
        for pid in panel:details.append(dict(team=r.team,player_id=pid,name=names.get(pid,str(pid)),centrality_present=s1[pid],centrality_absent=s2[pid],centrality_change=s2[pid]-s1[pid]))
    frame=pd.DataFrame(rows);atomic_csv(frame,out/'team_successors.csv');atomic_csv(pd.DataFrame(details),out/'player_centrality_changes.csv')
    tests=[]
    for label,g in [('all',frame),*list(frame.groupby('passing_style'))]:
        for x,y in [('concentration_present','concentration_absent'),('top_share_present','top_share_absent')]:
            test=ttest_rel(g[y],g[x]);tests.append(dict(group=label,outcome=y,n=len(g),before_mean=g[x].mean(),after_mean=g[y].mean(),mean_change=(g[y]-g[x]).mean(),p_paired_t=test.pvalue))
    atomic_csv(pd.DataFrame(tests),out/'paired_results.csv')
    fig,axes=plt.subplots(1,2,figsize=(16,7))
    for style,color,marker,label in [('more_balanced','#2473b8','o','Originally balanced'),('more_star_centered','#e18a24','^','Originally star-centered')]:
        g=frame.loc[frame.passing_style.eq(style)]
        axes[0].scatter(g.concentration_change,100*g.top_share_change,color=color,marker=marker,label=label)
        for _,r in g.iterrows():axes[0].annotate(r.team,(r.concentration_change,100*r.top_share_change),xytext=(4,4),textcoords='offset points',fontsize=8)
    axes[0].axhline(0,color='gray',lw=.8);axes[0].axvline(0,color='gray',lw=.8)
    axes[0].set(xlabel='Change in nonhub eigenvector concentration (absent − present)',ylabel='Change in largest nonhub centrality share (percentage points)',title='Does one replacement dominate, or does influence spread?')
    axes[0].legend(fontsize=8);axes[0].grid(alpha=.15)
    for kind,color in [('More balanced','#2473b8'),('More concentrated around a replacement hub','#c65c28'),('Mixed indicators','#999999')]:
        g=frame.loc[frame.redistribution.eq(kind)]
        axes[1].scatter(g.concentration_change,g.adjusted_offensive_loss,color=color,label=kind)
        for _,r in g.iterrows():axes[1].annotate(r.team,(r.concentration_change,r.adjusted_offensive_loss),xytext=(4,4),textcoords='offset points',fontsize=8)
    axes[1].axhline(0,color='gray',lw=.8);axes[1].axvline(0,color='gray',lw=.8);axes[1].set(xlabel='Change in nonhub eigenvector concentration',ylabel='Adjusted offensive rating loss during hub absence',title='Redistribution pattern versus offensive performance');axes[1].legend(fontsize=7);axes[1].grid(alpha=.15)
    fig.suptitle('NBA 2025–26: replacement passing hubs versus balanced redistribution · 21 eligible teams',fontsize=14)
    fig.text(.5,.015,'Same nonhub player panel in both conditions. Original hub excluded from BOTH networks. No minutes or roster covariate controls.\nDirectional descriptions have no minimum-effect threshold; pooled absence networks do not establish a consistent game-by-game successor.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.065,1,.95));season_topics_spread_labels(fig);fig.savefig(out/'hub_successors.pdf');fig.savefig(out/'hub_successors.png',dpi=150);plt.close(fig)
    report=['# Replacement hubs or balanced redistribution?','',frame.to_string(index=False),'',pd.DataFrame(tests).to_string(index=False),'',
        'Eigenvector shares use the weighted undirected network A+A.T, L1 normalized. Compare the identical active nonhub panel in both conditions; the original hub is removed from BOTH graphs. More balanced means BOTH overall concentration and maximum centrality share fall; replacement concentration means BOTH rise; other combinations are mixed. These are descriptive directional categories without a minimum-effect threshold, not statistically established team-level changes. A new highest-ranked remaining player does not itself mean a new dominant star. These are passing hubs, not shooting stars or skill estimates. Aggregating all absence games cannot distinguish one stable replacement from different game-specific replacements. Players with fewer minutes, trades and co-absences may change rankings. Per-game directed networks are needed to test persistence of successors. Exposure and availability are not controlled in the primary analysis. No-edge players are omitted. Twenty-one teams meet the existing 20 present/10 absent game criterion. Original team styles are retained. Adjusted offensive losses are reused from the prior saved within-team analysis. Exploratory paired tests are unadjusted for multiplicity.']
    (out/'README.md').write_text('\n\n'.join(report))
    print(frame[['team','original_hub','leading_nonhub_absent','concentration_change','top_share_change','leader_changed','redistribution']].to_string(index=False),flush=True)
    print(frame.redistribution.value_counts().to_string(),flush=True);print(pd.DataFrame(tests).to_string(index=False),flush=True)
    print('Saved hub successor analysis: '+str(out),flush=True);return 0



def centrality_suite_scores(matrix,ids):
    """Common active panel; directed receiving metrics and combined involvement metrics."""
    import networkx as nx
    a=np.asarray(matrix,float);ids=np.asarray(ids,int);active=(a.sum(0)+a.sum(1))>0;a=a[np.ix_(active,active)];ids=ids[active];n=len(ids)
    if n<3 or (a<0).any() or not np.isfinite(a).all():raise ValueError('Invalid centrality panel')
    b=a+a.T;graph=nx.from_numpy_array(b)
    if not nx.is_connected(graph):raise ValueError('Disconnected centrality network')
    # Normalizing total edge weight makes closeness invariant to total passing exposure.
    for i,j,v in graph.edges(data=True):v['distance']=b.sum()/v['weight']
    vals,vecs=np.linalg.eigh(b);ec=np.abs(vecs[:,-1]);ec/=ec.sum()
    bc=nx.betweenness_centrality(graph,weight='distance',normalized=True)
    close=nx.closeness_centrality(graph,distance='distance')
    dg=nx.from_numpy_array(a,create_using=nx.DiGraph);pr=nx.pagerank(dg,alpha=.85,weight='weight',tol=1e-12,max_iter=1000)
    katz=np.linalg.solve(np.eye(n)-.85*b/vals[-1],np.ones(n));katz/=katz.sum()
    u,s,vh=np.linalg.svd(a,full_matrices=False);hub=np.abs(u[:,0]);auth=np.abs(vh[0]);hub/=hub.sum();auth/=auth.sum()
    result=pd.DataFrame(dict(player_id=ids,degree=[graph.degree(i)/(n-1) for i in range(n)],
        in_strength=a.sum(0)/a.sum(),out_strength=a.sum(1)/a.sum(),strength=b.sum(1)/b.sum(),eigenvector=ec,
        betweenness=[bc[i] for i in range(n)],closeness=[close[i] for i in range(n)],pagerank=[pr[i] for i in range(n)],katz=katz,hits_hub=hub,hits_authority=auth))
    return result,a


def run_centrality_idea_suite(argv):
    from scipy.stats import spearmanr,ttest_rel
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    parser=argparse.ArgumentParser(description='Six centrality research ideas and nine metric-specific exploratory tests')
    parser.add_argument('--centrality-idea-suite',action='store_true');parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT_DIR)
    args=parser.parse_args(argv);parent=args.output_dir.expanduser()/'2025-26/team_season_topics';out=parent/'centrality_idea_suite';out.mkdir(parents=True,exist_ok=True)
    season=pd.read_csv(parent/'team_season_metrics.csv');players=pd.read_csv(parent/'player_games.csv',dtype={'GAME_ID':str});advanced=pd.read_csv(parent/'official_team_advanced.csv').set_index('TEAM_ID')
    context=pd.read_csv(parent/'eight_context_topics/context_game_metrics.csv',dtype={'GAME_ID':str});hubmetrics=pd.read_csv(parent/'hub_absence_resilience/team_season_metrics.csv')
    metrics=['degree','in_strength','out_strength','strength','eigenvector','betweenness','closeness','pagerank','katz','hits_hub','hits_authority']
    stable=set(players.groupby('PLAYER_ID').TEAM_ID.nunique().loc[lambda s:s.eq(1)].index)
    nodes=[];teamrows=[];leaderrows=[];absence_rows=[];succession=[];successorplayers=[];ties=[]
    def concentration(values):
        v=np.asarray(values,float)
        if v.sum()<=0:return np.nan
        q=v/v.sum();return float((q@q-1/len(q))/(1-1/len(q)))
    def leader(g,key):
        top=g.loc[np.isclose(g[key],g[key].max(),rtol=1e-10,atol=1e-14)].sort_values('player_id')
        return int(top.iloc[0].player_id),len(top)
    for _,r in season.iterrows():
        raw=pd.read_csv(parent/'networks'/r.team/'passing_matrix.csv',index_col=0);ids=raw.index.astype(int).tolist();node,a=centrality_suite_scores(raw.to_numpy(),ids)
        own=players.loc[players.TEAM_ID.eq(r.team_id)];stats=own.groupby('PLAYER_ID').agg(name=('PLAYER_NAME','first'),minutes=('MIN','sum'),games=('GAME_ID','size'),fga=('FGA','sum'),fgm=('FGM','sum'),fg3m=('FG3M','sum'))
        node=node.join(stats,on='player_id',validate='one_to_one');node['team']=r.team;node['passing_style']=r.passing_style
        node['shot_share']=node.fga/node.fga.sum();node['efg_pct']=100*(node.fgm+.5*node.fg3m)/node.fga.replace(0,np.nan)
        nodes.append(node)
        leaders={}
        for key in metrics:
            pid,tie=leader(node,key);leaders[key]=pid
            leaderrows.append(dict(team=r.team,metric=key,player_id=pid,name=stats.loc[pid,'name'],tied_leaders=tie))
        roleids=[leaders['hits_hub'],leaders['hits_authority'],leaders['betweenness']]
        ranks=[spearmanr(node[x],node[y]).statistic for x,y in [('strength','eigenvector'),('strength','pagerank'),('eigenvector','pagerank')]]
        teamrow=dict(team=r.team,team_id=r.team_id,passing_style=r.passing_style,player_count=len(node),conditional_entropy=r.conditional_receiver_entropy,off_rating=advanced.loc[r.team_id,'OFF_RATING'],
            shooting_concentration=concentration(node.fga),turnovers_per100passes=100*r.tov/r.recorded_passes,mean_degree=node.degree.mean(),mean_closeness=node.closeness.mean(),
            role_breadth=len(set(roleids)),role_overlap=3-len(set(roleids)),rank_agreement=np.mean(ranks),
            receiving_shooting_alignment=spearmanr(node.pagerank,node.shot_share).statistic,
            strength_shooting_alignment=spearmanr(node.in_strength,node.shot_share).statistic,
            mean_receiving_disruption=context.loc[context.team.eq(r.team)&context.receiving_coverage.eq(True),'receiving_disruption'].mean(),
            role_leader_ties=max(leader(node,key)[1] for key in ['hits_hub','hits_authority','betweenness']))
        for key in metrics:teamrow[key+'_concentration']=concentration(node[key])
        teamrows.append(teamrow)
        # Idea 1: offense and receiver-entropy changes during absences of different leaders.
        tg=context.loc[context.team.eq(r.team)].copy();tg['game_off_rating']=100*tg.PTS/tg.POSS
        for key in ['strength','eigenvector','betweenness','pagerank','hits_hub','hits_authority']:
            pid=leaders[key];played=set(own.loc[own.PLAYER_ID.eq(pid),'GAME_ID']);present=tg.GAME_ID.isin(played);eligible=pid in stable and present.sum()>=20 and (~present).sum()>=10 and leader(node,key)[1]==1
            rr=dict(team=r.team,metric=key,player_id=pid,name=stats.loc[pid,'name'],present_games=int(present.sum()),absent_games=int((~present).sum()),eligible=eligible,
                reason='eligible' if eligible else 'needs unique leader, stable team membership, >=20 appearances and >=10 absences')
            if eligible:
                rr['raw_offensive_loss']=100*tg.loc[present,'PTS'].sum()/tg.loc[present,'POSS'].sum()-100*tg.loc[~present,'PTS'].sum()/tg.loc[~present,'POSS'].sum()
                rr['receiver_entropy_loss']=tg.loc[present,'receiving_entropy_game'].mean()-tg.loc[~present,'receiving_entropy_game'].mean()
                tg['leader_present']=present.astype(int)
                model=fit_eigenvector_performance_model(tg,'game_off_rating',['leader_present','home_game','opponent_strength','back_to_back','game_number'],'leader_absence_'+key)
                coef=next(v for v in model if v['term']=='leader_present');rr['context_adjusted_loss']=coef['coefficient'];rr['p_hac7']=coef['p_value_hac7']
            absence_rows.append(rr)
    node=pd.concat(nodes,ignore_index=True);teams=pd.DataFrame(teamrows);absences=pd.DataFrame(absence_rows)
    # Idea 3: compare roles on exactly the same nonhub player population in both contexts.
    for _,r in hubmetrics.loc[hubmetrics.primary_eligible.eq(True)].iterrows():
        f=pd.read_csv(parent/'networks'/r.team/'passing_matrix.csv',index_col=0);p=pd.read_csv(parent/'hub_absence_resilience/present_networks'/r.team/'passing_matrix.csv',index_col=0)
        ids=f.index.astype(int).tolist();a=f.to_numpy()-p.to_numpy();take=[i for i,pid in enumerate(ids) if pid!=int(r.hub_id)]
        while True:
            before=p.to_numpy()[np.ix_(take,take)];after=a[np.ix_(take,take)];active=(before.sum(0)+before.sum(1)>0)&(after.sum(0)+after.sum(1)>0)
            if active.all():break
            take=[i for i,keep in zip(take,active) if keep]
        panel=[ids[i] for i in take];s1,_=centrality_suite_scores(before,panel);s2,_=centrality_suite_scores(after,panel)
        names=node.loc[node.team.eq(r.team)].set_index('player_id').name.to_dict();record=dict(team=r.team,hub=r.hub_name)
        leadersafter=[]
        for key in ['strength','eigenvector','betweenness','pagerank','hits_hub','hits_authority']:
            pid,tie=leader(s2,key);old,oldtie=leader(s1,key);leadersafter.append(pid)
            record[key+'_successor']=names[pid];record[key+'_successor_id']=pid;record[key+'_ties']=tie;record[key+'_leader_changed']=pid!=old
            for _,v in s2.iterrows():
                baseline=float(s1.loc[s1.player_id.eq(v.player_id),key].iloc[0]);successorplayers.append(dict(team=r.team,player_id=int(v.player_id),name=names[int(v.player_id)],metric=key,before=baseline,after=v[key],change=v[key]-baseline))
        record['distinct_successors']=len(set(leadersafter));record['all_roles_same_successor']=len(set(leadersafter))==1;succession.append(record)
    atomic_csv(node,out/'player_centralities.csv');atomic_csv(teams,out/'team_metrics.csv');atomic_csv(pd.DataFrame(leaderrows),out/'metric_leaders.csv')
    atomic_csv(absences,out/'absence_by_centrality.csv');atomic_csv(pd.DataFrame(succession),out/'successor_roles.csv');atomic_csv(pd.DataFrame(successorplayers),out/'successor_player_changes.csv')
    # Six ideas: five inferential primary tests, one descriptive role-replacement analysis.
    primary=[];pair=absences.loc[absences.eligible.eq(True)].pivot(index='team',columns='metric',values='context_adjusted_loss')
    paired=pair[['betweenness','strength']].dropna();test=ttest_rel(paired.betweenness,paired.strength)
    primary.append(dict(idea=1,test='Bridge vs busiest passer: context-adjusted offensive absence loss',n=len(paired),effect=(paired.betweenness-paired.strength).mean(),p_value=test.pvalue))
    atomic_csv(paired.reset_index(),out/'bridge_vs_strength_paired_teams.csv')
    tests=[(2,'role_breadth','off_rating','Role specialization and offense'),(4,'receiving_shooting_alignment','off_rating','Receiving/shooting alignment and offense'),
        (5,'betweenness_concentration','mean_receiving_disruption','Bridge concentration and receiving disruption'),(6,'rank_agreement','off_rating','Centrality agreement and offense')]
    plot_specs=[];allfits=[]
    def add_fit(idea,x,y,title):
        fits=season_topics_fit(teams,x,y,title,idea);allfits.extend(fits);row=next(v for v in fits if v['term']==x)
        primary.append(dict(idea=idea,test=title,n=row['n'],effect=row['coefficient'],r=row['pearson_r'],r_squared=row['r_squared'],p_value=row['p_value_hc3']))
        plot_specs.append((str(idea),x,y,title))
    for spec in tests:add_fit(*spec)
    result=pd.DataFrame(primary);result['p_holm_five']=np.nan;running=0
    for rank,idx in enumerate(result.p_value.sort_values().index):running=max(running,min(1,result.loc[idx,'p_value']*(5-rank)));result.loc[idx,'p_holm_five']=running
    atomic_csv(result,out/'six_ideas_primary_results.csv')
    metric_tests=[('degree','mean_degree','conditional_entropy','Connection breadth and passing entropy'),
        ('in_strength','in_strength_concentration','shooting_concentration','Receiving concentration and shooting concentration'),
        ('out_strength','out_strength_concentration','turnovers_per100passes','Distributor concentration and turnovers'),
        ('eigenvector','eigenvector_concentration','off_rating','Passing-core concentration and offense'),
        ('betweenness','betweenness_concentration','mean_receiving_disruption','Bridge concentration and disruption'),
        ('closeness','mean_closeness','off_rating','Passing-route closeness and offense'),
        ('pagerank','pagerank_concentration','shooting_concentration','Receiving PageRank concentration and shooting'),
        ('katz','katz_concentration','off_rating','Indirect influence concentration and offense'),
        ('hits','hits_hub_concentration','hits_authority_concentration','Distributor and receiver specialization')]
    metric_results=[]
    for name,x,y,title in metric_tests:
        fits=season_topics_fit(teams,x,y,name,'centrality_metrics');allfits.extend(fits);row=next(v for v in fits if v['term']==x)
        metric_results.append(dict(metric=name,test=title,n=row['n'],r=row['pearson_r'],r_squared=row['r_squared'],slope=row['coefficient'],p_value=row['p_value_hc3']))
        plot_specs.append((name,x,y,title))
    mr=pd.DataFrame(metric_results);mr['p_holm_nine']=np.nan;running=0
    for rank,idx in enumerate(mr.p_value.sort_values().index):running=max(running,min(1,mr.loc[idx,'p_value']*(9-rank)));mr.loc[idx,'p_holm_nine']=running
    atomic_csv(mr,out/'nine_metric_tests.csv')
    # Rank association with shot share is checked across teams; these are not out-of-sample predictions.
    alignment=teams[['team','receiving_shooting_alignment','strength_shooting_alignment']].copy();atomic_csv(alignment,out/'receiving_shooting_alignment.csv')
    # A more targeted dependence-on-bridge test: strongest BC teams vs own bridge absence penalty.
    bridge=absences.loc[absences.metric.eq('betweenness')&absences.eligible.eq(True)].merge(teams[['team','betweenness_concentration']],on='team',validate='one_to_one')
    if len(bridge)>=5:
        allfits.extend(season_topics_fit(bridge,'betweenness_concentration','context_adjusted_loss','bridge_absence_vulnerability_sensitivity',5))
    atomic_csv(bridge,out/'bridge_absence_vulnerability.csv');atomic_csv(pd.DataFrame(allfits),out/'all_regressions.csv')
    stylefits=[]
    with PdfPages(out/'all_centrality_ideas.pdf') as pdf:
        fig,ax=plt.subplots(figsize=(11,8))
        ax.scatter(paired.strength,paired.betweenness,color='#2473b8')
        for team,r in paired.iterrows():ax.annotate(team,(r.strength,r.betweenness),xytext=(4,4),textcoords='offset points',fontsize=8)
        limits=[min(paired.min().min(),0)-1,max(paired.max().max(),0)+1];ax.plot(limits,limits,color='gray',ls='--')
        ax.set(xlabel='Adjusted offensive loss: busiest passer absent',ylabel='Adjusted offensive loss: highest-betweenness player absent',title='1. Which passing role is harder to replace?\n17 matched teams; 12 have the same player in both roles')
        ax.grid(alpha=.15);fig.tight_layout();season_topics_spread_labels(fig);pdf.savefig(fig);fig.savefig(out/'idea_1_absence_comparison.pdf');plt.close(fig)
        fig,ax=plt.subplots(figsize=(11,8));counts=pd.DataFrame(succession).distinct_successors.value_counts().reindex(range(1,7),fill_value=0)
        ax.bar(counts.index,counts.values,color='#2473b8');ax.set(xticks=list(range(1,7)),xlabel='Distinct players leading six centrality measures during hub absence',ylabel='Teams',title='3. One replacement or several role specialists?\n21 matched nonhub networks; six measures are not independent roles')
        for k,v in counts.items():ax.text(k,v+.1,str(v),ha='center')
        fig.tight_layout();pdf.savefig(fig);fig.savefig(out/'idea_3_successor_roles.pdf');plt.close(fig)
        for label,x,y,title in plot_specs:
            fig,ax=plt.subplots(figsize=(11,8))
            for style,color,marker,name in [('more_balanced','#2473b8','o','Balanced'),('more_star_centered','#e18a24','^','Star-centered')]:
                g=teams.loc[teams.passing_style.eq(style)];ax.scatter(g[x],g[y],color=color,marker=marker,label=name)
                fit=season_topics_fit(g,x,y,label+'_'+style,label);stylefits.extend(dict(v,passing_style=style) for v in fit)
                if fit:
                    slope=next(v for v in fit if v['term']==x);intercept=next(v['coefficient'] for v in fit if v['term']=='intercept');xx=np.linspace(g[x].min(),g[x].max(),100);ax.plot(xx,intercept+slope['coefficient']*xx,color=color)
                for _,v in g.iterrows():ax.annotate(v.team,(v[x],v[y]),xytext=(4,4),textcoords='offset points',fontsize=7)
            pooled=next(v for v in season_topics_fit(teams,x,y,label,label) if v['term']==x)
            ax.set(xlabel=x.replace('_',' '),ylabel=y.replace('_',' '),title='NBA 2025–26 · '+title+'\n'+f"Pooled r={pooled['pearson_r']:.3f} | R²={pooled['r_squared']:.3f} | HC3 p={pooled['p_value_hc3']:.3g}");ax.legend();ax.grid(alpha=.15)
            fig.text(.5,.015,'30 teams · fixed original style labels · no minutes or player-count covariates · exploratory associations',ha='center',fontsize=9)
            fig.tight_layout(rect=(0,.04,1,1));season_topics_spread_labels(fig);pdf.savefig(fig);fig.savefig(out/('test_'+label+'.png'),dpi=130);fig.savefig(out/('test_'+label+'.pdf'));plt.close(fig)
    atomic_csv(pd.DataFrame(stylefits),out/'style_regressions.csv')
    extra_results=[]
    # Distinct-role absence comparison: identical leaders cannot distinguish role types.
    pidtable=absences.loc[absences.eligible.eq(True)].pivot(index='team',columns='metric',values='player_id')
    distinct=paired.loc[pidtable.loc[paired.index,'betweenness'].ne(pidtable.loc[paired.index,'strength'])]
    if len(distinct)>1:
        t=ttest_rel(distinct.betweenness,distinct.strength)
        extra_results.append(dict(model='bridge_vs_strength_distinct_players',n=len(distinct),effect=(distinct.betweenness-distinct.strength).mean(),p_value=t.pvalue))
    al=ttest_rel(teams.receiving_shooting_alignment,teams.strength_shooting_alignment)
    extra_results.append(dict(model='PageRank_vs_instrength_shot_alignment',n=len(teams),effect=(teams.receiving_shooting_alignment-teams.strength_shooting_alignment).mean(),p_value=al.pvalue))
    interactions=[];sensitivity=[];loo=[]
    minutes=players.groupby(['TEAM_ID','PLAYER_ID']).MIN.sum().reset_index()
    mh={teamid:concentration(g.MIN) for teamid,g in minutes.groupby('TEAM_ID')}
    teams['minutes_concentration']=teams.team_id.map(mh)
    for x in ['eigenvector_concentration','katz_concentration']:
        for controls,label in [(['player_count'],'player_count'),(['minutes_concentration'],'minutes'),(['player_count','minutes_concentration'],'both')]:
            row,_=six_team_fit(teams,x,'off_rating',controls,x+'_'+label);sensitivity.append(dict(row,controls=label))
        for team in teams.team:
            fit=season_topics_fit(teams.loc[teams.team.ne(team)],x,'off_rating',x+'_leave_out_'+team,'sensitivity')
            row=next(v for v in fit if v['term']==x);loo.append(dict(row,omitted_team=team))
    for label,x,y,title in plot_specs:
        z=teams.copy();z['star_indicator']=z.passing_style.eq('more_star_centered').astype(float);z['metric_x_star']=z[x]*z.star_indicator
        interactions.extend(season_topics_fit(z,x,y,label+'_style_interaction',label,[x,'star_indicator','metric_x_star']))
    atomic_csv(pd.DataFrame(extra_results),out/'additional_comparisons.csv')
    atomic_csv(pd.DataFrame(sensitivity),out/'offense_covariate_sensitivities.csv')
    atomic_csv(pd.DataFrame(loo),out/'offense_leave_one_team_out.csv')
    atomic_csv(pd.DataFrame(interactions),out/'style_slope_interactions.csv')

    methodology=['# All centrality ideas: NBA 2025–26','', 'Six ideas implemented: 1 comparative leader-absence offense study (receiver-entropy substitute also saved; no full arbitrary-leader absence passing networks); 2 role breadth vs offensive rating; 3 multi-centrality replacement roles (descriptive); 4 receiving/shooting alignment vs offense; 5 bridge concentration vs receiving-pattern disruption (surrogate), plus direct bridge-absence offense sensitivity; 6 centrality rank agreement vs offense. Nine additional centrality-specific links implemented separately.',
        'All active players retained; isolates excluded. Degree is combined unique-neighbor fraction. In/out strength are directed pass shares. Eigenvector and Katz use A+A.T. Katz attenuation = 0.85 / spectral radius. PageRank uses sender-to-receiver edges with alpha 0.85. HITS hub/authority use principal left/right singular vectors of directed A. Betweenness and closeness use distance=sum(A+A.T)/edge_weight, so closeness is independent of total pass exposure. Paths are possible network routes, not observed sequences. Different distance conventions may change the findings.',
        'All metric concentrations are normalized HHI of L1 score shares. BC concentration is undefined if all BC scores are zero; unavailable tests cannot be treated as null findings. Role breadth = distinct leaders of HITS hub, HITS authority and betweenness (1–3). Leader ties are recorded, with smallest player ID used only for deterministic descriptive listing; absence comparisons exclude tied leaders. Rank agreement = mean Spearman across combined strength, eigenvector, and directed PageRank. Receiving/shooting alignment = within-team Spearman of PageRank vs FGA share. This is association, not predictive validation.',
        'Absence study requires stable-team players, unique metric leader, >=20 games present and >=10 absent; no minutes cutoff. Presence = any appearance, not known injury status. Context-adjusted within-team game offensive rating uses home, opponent defensive strength, back-to-back and game index, with HAC7 errors. Co-absences and playing time are not controlled in this exploratory suite. Comparison uses matched teams across leader types; identical leaders yield zero paired differences and count as such. Full-season leaders are chosen retrospectively, not pre-injury; absences may influence selection. All absence exposure and eligible/excluded reasons are saved.',
        'Succession uses the same nonhub panel with original hub removed in both contexts. Six metrics are not six independent roles: collinear rankings may identify the same player. Aggregated absence matrices cannot establish stable game-by-game replacement leaders. No claim that a passing hub is a scoring star.',
        'Team fits use HC3; six-idea family has five inferential tests plus one descriptive analysis, corrected with Holm-five. Nine metric tests use a separate Holm-nine correction. These two families do not correct all prior topic exploration; subgroup lines and additional tests are unadjusted exploratory follow-ups. Team metrics on the same passes may be structurally coupled. No minutes/roster covariates in cross-team fits. Fixed original team-style split retained. All performance and passing data are saved official-source inputs; no new download or soccer data. Small n=30, or smaller eligible absence subsets; observational, not causal.',
        '',pd.DataFrame(extra_results).to_string(index=False),'',pd.DataFrame(sensitivity).to_string(index=False),'',result.to_string(index=False),'',mr.to_string(index=False),'',pd.DataFrame(succession).to_string(index=False)]
    (out/'README.md').write_text('\n\n'.join(methodology))
    print(result.to_string(index=False),flush=True);print(mr.to_string(index=False),flush=True)
    print('Successor role counts:',pd.DataFrame(succession).distinct_successors.value_counts().sort_index().to_dict(),flush=True)
    print('Saved all centrality ideas: '+str(out),flush=True);return 0



def run_assist_network_ideas(argv):
    """Ten fixed exploratory assist-network questions, with coverage and multiplicity audits."""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    parser=argparse.ArgumentParser(description='Ten 2025-26 directed assist-network questions using saved player passing profiles')
    parser.add_argument('--assist-network-ideas',action='store_true');parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT_DIR)
    args=parser.parse_args(argv);parent=args.output_dir.expanduser()/'2025-26/team_season_topics';out=parent/'assist_network_ideas';out.mkdir(parents=True,exist_ok=True)
    import json
    lookup={}
    for file in (parent/'cache').glob('playerdashptpass*.json'):
        d=json.loads(file.read_text());p=d['parameters']
        if p.get('Season')!='2025-26' or p.get('SeasonType')!='Regular Season' or p.get('PerMode')!='Totals' or any(p.get(k) not in ['',None,0,'0'] for k in ['DateFrom','DateTo','Location','Outcome','OpponentTeamID','LastNGames','Month','SeasonSegment','VsConference','VsDivision']):continue
        key=(int(p['TeamID']),int(p['PlayerID']))
        if key not in lookup or (d.get('fetched_at',''),str(file))>(lookup[key][1].get('fetched_at',''),str(lookup[key][0])):
            lookup[key]=(file,d)
    season=pd.read_csv(parent/'team_season_metrics.csv');official=pd.read_csv(parent/'official_team_totals.csv').set_index('TEAM_ID');advanced=pd.read_csv(parent/'official_team_advanced.csv').set_index('TEAM_ID')
    players=pd.read_csv(parent/'player_games.csv',dtype={'GAME_ID':str});rows=[];edges=[];audit=[];node_rows=[];sources=[]
    def hhi(values):
        q=np.asarray(values,float);q=q/q.sum();n=len(q);return float((q@q-1/n)/(1-1/n))
    for _,r in season.iterrows():
        saved=pd.read_csv(parent/'networks'/r.team/'passing_matrix.csv',index_col=0);ids=saved.index.astype(int).tolist();index={pid:i for i,pid in enumerate(ids)};n=len(ids)
        assist=np.zeros((n,n));passes=np.zeros((n,n));outside=0
        own=players.loc[players.TEAM_ID.eq(r.team_id)].groupby('PLAYER_ID').agg(name=('PLAYER_NAME','first'),fga=('FGA','sum'),assists=('AST','sum'),minutes=('MIN','sum'))
        for pid in ids:
            key=(int(r.team_id),pid)
            if key not in lookup:raise ValueError('Missing full profile '+str(key))
            path,data=lookup[key];tables=data['payload']['resultSets'];table=next(v for v in tables if v['name']=='PassesMade');g=pd.DataFrame(table['rowSet'],columns=table['headers'])
            if not g.empty:
                for col in ['PASS','AST','FGM']:
                    values=pd.to_numeric(g[col],errors='raise')
                    if not np.isfinite(values).all() or values.lt(0).any() or not np.allclose(values,np.round(values)):raise ValueError('Invalid counts')
                if not g.PLAYER_ID.eq(pid).all() or not g.TEAM_ID.eq(r.team_id).all():raise ValueError('Wrong profile identity')
                if g.AST.gt(g.PASS).any() or g.AST.gt(g.FGM).any():raise ValueError('Impossible assist counts')
                for _,v in g.iterrows():
                    receiver=int(v.PASS_TEAMMATE_PLAYER_ID)
                    if receiver not in index:outside+=float(v.AST);continue
                    if receiver==pid:
                        if v.AST>0:raise ValueError('Impossible self-assist')
                        continue  # Match existing passing code: exclude self-pass tracking artifacts.
                    assist[index[pid],index[receiver]]+=v.AST;passes[index[pid],index[receiver]]+=v.PASS
            sources.append(dict(team=r.team,player_id=pid,cache_file=str(path),fetched_at=data.get('fetched_at','')))
        if not np.array_equal(passes,saved.to_numpy(float)):raise ValueError('Rebuilt passing network differs '+r.team)
        total=float(assist.sum());official_ast=float(official.loc[r.team_id,'AST']);coverage=total/official_ast
        if total<=0 or coverage>1.000001 or coverage<.99:raise ValueError('Unacceptable assist coverage '+r.team)
        active=(assist.sum(0)+assist.sum(1))>0;activeids=np.array(ids)[active];a=assist[np.ix_(active,active)];p=passes[np.ix_(active,active)];N=len(a)
        provider=a.sum(1);recipient=a.sum(0);stats=own.reindex(activeids);_,ah=compute_team_entropies(a);_,ph=compute_team_entropies(p)
        maxedge=np.unravel_index(np.argmax(a),a.shape);creator=int(activeids[np.argmax(provider)]);scorer=int(activeids[np.argmax(recipient)])
        row=dict(team=r.team,team_id=int(r.team_id),passing_style=r.passing_style,assist_players=N,roster_players=n,omitted_assist_isolates=n-N,
            network_assists=total,official_assists=official_ast,missing_assist_count=official_ast-total,assist_coverage=coverage,exact_assist_total=total==official_ast,
            assist_conditional_entropy=ah,raw_assist_conditional_entropy=ah*np.log(N-1),passing_conditional_entropy_matched=ph,entropy_gap=ph-ah,
            creator_concentration=hhi(provider),scorer_concentration=hhi(recipient),shooting_concentration=hhi(stats.fga),minutes_concentration=hhi(stats.minutes),
            top_assist_pair_share=float(a.max()/total),top_assist_pair=own.loc[activeids[maxedge[0]],'name']+' → '+own.loc[activeids[maxedge[1]],'name'],
            creator_scorer_role_separation=receiving_share_js_divergence(provider,recipient),passing_assist_edge_divergence=receiving_share_js_divergence(p.ravel(),a.ravel()),
            off_rating=advanced.loc[r.team_id,'OFF_RATING'],efg_pct=100*advanced.loc[r.team_id,'EFG_PCT'],assist_turnover_ratio=official_ast/official.loc[r.team_id,'TOV'],
            turnovers_per100poss=100*official.loc[r.team_id,'TOV']/advanced.loc[r.team_id,'POSS'],
            main_assister=own.loc[creator,'name'],main_assisted_scorer=own.loc[scorer,'name'],same_creator_scorer=creator==scorer)
        rows.append(row);audit.append(dict(team=r.team,missing_profiles=0,outside_roster_assists=outside,network_assists=total,official_assists=official_ast,missing=official_ast-total,coverage=coverage))
        folder=out/'networks'/r.team;folder.mkdir(parents=True,exist_ok=True);atomic_csv(pd.DataFrame(a,index=activeids,columns=activeids),folder/'assist_matrix.csv',index=True)
        for i,pid in enumerate(activeids):
            node_rows.append(dict(team=r.team,player_id=int(pid),name=own.loc[pid,'name'],assists_given=provider[i],assisted_baskets_scored=recipient[i],creator_share=provider[i]/total,recipient_share=recipient[i]/total,
                official_assists=own.loc[pid,'assists'],assist_difference=own.loc[pid,'assists']-provider[i],fga=stats.loc[pid,'fga'],minutes=stats.loc[pid,'minutes']))
            for j,qid in enumerate(activeids):
                if a[i,j]>0:edges.append(dict(team=r.team,assister_id=int(pid),scorer_id=int(qid),assister=own.loc[pid,'name'],scorer=own.loc[qid,'name'],assists=a[i,j],passes=p[i,j]))
    teams=pd.DataFrame(rows);atomic_csv(teams,out/'team_metrics.csv');atomic_csv(pd.DataFrame(edges),out/'assist_edges.csv');atomic_csv(pd.DataFrame(node_rows),out/'player_assist_roles.csv');atomic_csv(pd.DataFrame(audit),out/'coverage_audit.csv');atomic_csv(pd.DataFrame(sources),out/'source_profiles.csv')
    # Specify all ten hypotheses before fitting. All fits and null results are retained.
    specs=[(1,'assist_conditional_entropy','off_rating','Assist diversity and offensive efficiency','Conditional assist entropy H(scorer | assister)','Offensive rating (points / 100 possessions)'),
        (2,'assist_conditional_entropy','efg_pct','Assist diversity and shooting efficiency','Conditional assist entropy H(scorer | assister)','Team effective field-goal percentage'),
        (3,'assist_conditional_entropy','assist_turnover_ratio','Assist diversity and assist-to-turnover ratio','Conditional assist entropy H(scorer | assister)','Official team assists / turnovers'),
        (4,'creator_concentration','off_rating','Creator concentration and offensive efficiency','Concentration of assists supplied (normalized HHI)','Offensive rating'),
        (5,'scorer_concentration','shooting_concentration','Assisted-scorer concentration and shooting concentration','Concentration of assisted baskets scored (normalized HHI)','Concentration of player shot attempts (normalized HHI)'),
        (6,'top_assist_pair_share','off_rating','Reliance on one assist partnership and offense','Largest directed assist pair share','Offensive rating'),
        (7,'creator_scorer_role_separation','off_rating','Creator–finisher role separation and offense','JS divergence: assist-provider vs assisted-scorer shares','Offensive rating'),
        (8,'entropy_gap','off_rating','Passing-to-assist diversity gap and offense','Matched-panel passing entropy minus assist entropy','Offensive rating'),
        (9,'creator_concentration','turnovers_per100poss','Creator concentration and ball security','Concentration of assists supplied (normalized HHI)','Turnovers / 100 possessions'),
        (10,'passing_assist_edge_divergence','off_rating','Passing-to-assist route differences and offense','JS divergence: passing edges vs assist edges','Offensive rating')]
    primary=[];allfits=[];sensitivities=[];styles=[];interactions=[]
    for k,x,y,title,xlabel,ylabel in specs:
        fit=season_topics_fit(teams,x,y,title,k);allfits.extend(fit);r=next(v for v in fit if v['term']==x)
        primary.append(dict(idea=k,question=title,x=x,y=y,n=r['n'],r=r['pearson_r'],r_squared=r['r_squared'],slope=r['coefficient'],p_hc3=r['p_value_hc3']))
        strict=teams.loc[teams.exact_assist_total.eq(True)]
        sensitivities.extend(season_topics_fit(strict,x,y,'exact_coverage_'+str(k),k))
        row,_=six_team_fit(teams,x,y,['minutes_concentration','assist_players'],'minutes_player_count_'+str(k));sensitivities.append(dict(row,topic=k,term=x))
        z=teams.copy();z['star_indicator']=z.passing_style.eq('more_star_centered').astype(float);z['metric_x_star']=z[x]*z.star_indicator
        interactions.extend(season_topics_fit(z,x,y,'style_interaction_'+str(k),k,[x,'star_indicator','metric_x_star']))
        for style,g in teams.groupby('passing_style'):styles.extend(dict(v,passing_style=style) for v in season_topics_fit(g,x,y,'style_'+str(k),k))
    result=pd.DataFrame(primary);result['p_holm_ten']=np.nan;running=0
    for rank,idx in enumerate(result.p_hc3.sort_values().index):running=max(running,min(1,result.loc[idx,'p_hc3']*(10-rank)));result.loc[idx,'p_holm_ten']=running
    leave_one_out=[]
    for idea,x,y in [(4,'creator_concentration','off_rating'),(6,'top_assist_pair_share','off_rating')]:
        for omitted in teams.team:
            fits=season_topics_fit(teams.loc[teams.team.ne(omitted)],x,y,'leave_one_out_'+str(idea),idea)
            leave_one_out.extend(dict(v,omitted_team=omitted) for v in fits)
    atomic_csv(pd.DataFrame(leave_one_out),out/'leave_one_team_out.csv')
    atomic_csv(result,out/'ten_primary_results.csv');atomic_csv(pd.DataFrame(allfits),out/'regressions.csv');atomic_csv(pd.DataFrame(sensitivities),out/'sensitivity_regressions.csv');atomic_csv(pd.DataFrame(styles),out/'style_regressions.csv');atomic_csv(pd.DataFrame(interactions),out/'style_slope_interactions.csv')
    with PdfPages(out/'ten_assist_network_ideas.pdf') as pdf:
        for k,x,y,title,xlabel,ylabel in specs:
            fig,ax=plt.subplots(figsize=(11,8));r=result.loc[result.idea.eq(k)].iloc[0]
            for style,col,marker,name in [('more_balanced','#2473b8','o','Balanced'),('more_star_centered','#e18a24','^','Star-centered')]:
                g=teams.loc[teams.passing_style.eq(style)];fits=season_topics_fit(g,x,y,'plot',k);slope=next(v for v in fits if v['term']==x);intercept=next(v['coefficient'] for v in fits if v['term']=='intercept')
                ax.scatter(g[x],g[y],color=col,marker=marker,label=f"{name}: r={slope['pearson_r']:.3f}, p={slope['p_value_hc3']:.3g}")
                xx=np.linspace(g[x].min(),g[x].max(),100);ax.plot(xx,intercept+slope['coefficient']*xx,color=col)
                for _,v in g.iterrows():ax.annotate(v.team,(v[x],v[y]),xytext=(4,4),textcoords='offset points',fontsize=7)
            ax.set(xlabel=xlabel,ylabel=ylabel,title=f'{k}. {title}\n'+f'Pooled r={r.r:.3f} | R²={r.r_squared:.3f} | Holm-ten p={r.p_holm_ten:.3g}')
            ax.legend(fontsize=8);ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False);ax.margins(.12)
            fig.text(.5,.015,'2025–26 NBA · 30 teams · original passing-network style labels · no primary minutes/player-count controls\nAssist-edge coverage 99.55–100%; exact-match 20-team sensitivity saved. Subgroup p-values are exploratory and unadjusted.',ha='center',fontsize=9)
            fig.tight_layout(rect=(0,.055,1,1));season_topics_spread_labels(fig);pdf.savefig(fig);fig.savefig(out/f'idea_{k}.pdf');fig.savefig(out/f'idea_{k}.png',dpi=150);plt.close(fig)
    (out/'README.md').write_text('\n\n'.join(['# Ten assist-network ideas — 2025–26 NBA regular season',
        'Source: latest timestamped saved full-season Totals snapshot for each team/player (duplicate cache snapshots resolved deterministically). Self-pass artifacts excluded as in the existing project; self-assists rejected. Saved NBA PlayerDashPtPass PassesMade.AST by passer/scorer, distinct from FGM or FGA. Directed edge A[i,j] = credited assists by i to j. All 30 teams have full saved profile coverage; no downloads. Source schema: https://github.com/swar/nba_api/blob/master/docs/nba_api/stats/endpoints/playerdashptpass.md',
        'Twenty teams exactly match official AST. Ten differ by 1–9 assists (network totals below official totals); 99.55–100% credit coverage. Known unallocated credit is not invented or assigned to edges. All deficits and per-player discrepancies saved. Exact-match 20-team regressions saved separately; exact-match exclusion may introduce sample selection. No impossible AST>PASS or AST>FGM rows accepted. Rebuilt PASS matrices must exactly reproduce saved full-season passing matrices.',
        'Assist isolates omitted. Same active assist-node panel used for provider, recipient, shooting and minutes concentrations. Nodes with zero assists supplied but positive incoming assists remain. Nodes with no incoming or outgoing assist edges are omitted. H(scorer|assister) normalized by log(N−1), passer weighted by assists supplied. Passing entropy comparison uses the same assist-active player set. Raw conditional entropy in nats is also saved. Concentration is normalized HHI. Role separation = base-2 JS(provider shares,recipient shares), not proof of distinct players under every possession. Top pair is directed (assister → scorer). Edge divergence compares normalized all-pass and assist counts on identical panels; no estimate of the probability an individual pass causes a basket.',
        'All ten hypotheses fit and retained; HC3 robust team inference; Holm correction across these ten primary tests. Style groups remain the original 15/15 season passing eigenvector classes, not assist-derived relabeling. Each plot has separate lines; slope-interaction tests saved. Subgroup/interaction and sensitivity tests are exploratory, unadjusted. The Holm-ten correction does not cover all prior analyses in this project. Primary models have no minutes/roster covariates; additional minutes-concentration/player-count controls and exact-credit-coverage sensitivity are saved.',
        'Assist networks include made assisted baskets only; they omit missed opportunities, unassisted scores and earlier creation. FGM/FGA fields are not used as potential-assist counts. Scorer concentration and shot concentration overlap by construction; AST/TO includes assist count. Structure-derived correlations are not independent evidence of better offense. Full-season aggregate networks hide lineups and game-to-game variation. Observational sample, not causal. No minutes cutoff; role concentration can reflect unequal playing time. No soccer data tested.',
        result.to_string(index=False),pd.DataFrame(audit).to_string(index=False)]))
    print(result.to_string(index=False),flush=True);print('Exact total teams:',int(teams.exact_assist_total.sum()),'minimum coverage:',teams.assist_coverage.min(),flush=True)
    print('Saved ten assist-network ideas: '+str(out),flush=True);return 0



def run_assist_creator_followups(argv):
    """Creator concentration x main-creator entropy, recipient succession and within-team changes."""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import json
    parser=argparse.ArgumentParser(description='Three follow-ups to the assist-network study: interaction, succession and changes')
    parser.add_argument('--assist-creator-followups',action='store_true');parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT_DIR)
    args=parser.parse_args(argv);parent=args.output_dir.expanduser()/'2025-26/team_season_topics';out=parent/'assist_creator_followups';out.mkdir(parents=True,exist_ok=True)
    base=pd.read_csv(parent/'assist_network_ideas/team_metrics.csv');players=pd.read_csv(parent/'player_games.csv',dtype={'GAME_ID':str});games=pd.read_csv(parent/'team_games.csv',dtype={'GAME_ID':str})
    advanced=pd.read_csv(parent/'hub_absence_resilience/official_game_advanced.csv',dtype={'GAME_ID':str})
    poss=advanced.set_index(['TEAM_ID','GAME_ID']).POSS.to_dict();games['possessions']=[poss[(r.TEAM_ID,r.GAME_ID)] for _,r in games.iterrows()]
    cutoff=json.loads((parent/'eight_context_topics/calendar_split.json').read_text())['cutoff']
    season=pd.read_csv(parent/'team_season_metrics.csv').set_index('team');fulls={};assists={};records=[];periods=[];coverage=[]
    def conc(v):
        v=np.asarray(v,float);q=v/v.sum();return float((q@q-1/len(q))/(1-1/len(q)))
    def ent(v):
        v=np.asarray(v,float);p=v[v>0]/v.sum();return float(-(p*np.log(p)).sum())
    def load_edges(path,ids):
        if not path.exists():raise ValueError('Missing saved assist split '+str(path))
        f=pd.read_csv(path);required={'PLAYER_ID','PASS_TEAMMATE_PLAYER_ID','AST','PASS'}
        if not required.issubset(f.columns):raise ValueError('Split has no pairwise assist fields')
        pos={pid:i for i,pid in enumerate(ids)};a=np.zeros((len(ids),len(ids)));p=np.zeros_like(a);outside=0
        for _,v in f.iterrows():
            source=int(v.PLAYER_ID);target=int(v.PASS_TEAMMATE_PLAYER_ID)
            if not np.isfinite(v.AST) or v.AST<0 or v.AST>v.PASS:raise ValueError('Invalid split assist count')
            if source==target:
                if v.AST>0:raise ValueError('Self assist')
                continue
            if source not in pos or target not in pos:outside+=v.AST;continue
            a[pos[source],pos[target]]+=v.AST;p[pos[source],pos[target]]+=v.PASS
        return a,p,outside
    # Full matrices on official season roster, including zero placeholders only for panel alignment.
    edges=pd.read_csv(parent/'assist_network_ideas/assist_edges.csv')
    for _,r in base.iterrows():
        f=pd.read_csv(parent/'networks'/r.team/'passing_matrix.csv',index_col=0);ids=f.index.astype(int).tolist();ix={pid:i for i,pid in enumerate(ids)};a=np.zeros(f.shape)
        for _,v in edges.loc[edges.team.eq(r.team)].iterrows():a[ix[int(v.assister_id)],ix[int(v.scorer_id)]]=v.assists
        fulls[r.team]=(ids,f.to_numpy(float));assists[r.team]=a
        active=(a.sum(0)+a.sum(1))>0;N=int(active.sum());creator=int(np.argmax(a.sum(1)));main_entropy=ent(a[creator])/np.log(N-1)
        records.append(dict(r.to_dict(),main_creator_id=ids[creator],main_creator_recipient_entropy=main_entropy,main_creator_recipient_count=int((a[creator]>0).sum())))
        early=np.zeros_like(a);earlypasses=np.zeros_like(a);outside=0
        folder=parent/'eight_context_topics/profiles/early'/r.team
        for file in folder.glob('*.csv'):
            part,p,extra=load_edges(file,ids);early+=part;earlypasses+=p;outside+=extra
        expected=pd.read_csv(parent/'eight_context_topics/split_networks'/r.team/'early.csv',index_col=0)
        if not np.array_equal(earlypasses,expected.to_numpy(float)):raise ValueError('Early pass reconstruction mismatch '+r.team)
        late=a-early
        if late.min() < -1e-8:raise ValueError('Negative late assist count '+r.team)
        late=np.maximum(late,0);common=((early.sum(0)+early.sum(1))>0)&((late.sum(0)+late.sum(1))>0)
        # Iterate until every selected node has at least one edge in both induced networks.
        take=np.flatnonzero(common).tolist()
        while True:
            e=early[np.ix_(take,take)];l=late[np.ix_(take,take)];keep=(e.sum(0)+e.sum(1)>0)&(l.sum(0)+l.sum(1)>0)
            if keep.all():break
            take=[i for i,k in zip(take,keep) if k]
        tg=games.loc[games.TEAM_ID.eq(r.team_id)];first=tg.GAME_DATE.astype(str).str[:10].le(cutoff);ep=tg.loc[first];lp=tg.loc[~first]
        stats=players.loc[players.TEAM_ID.eq(r.team_id)].copy();ef=stats.GAME_DATE.astype(str).str[:10].le(cutoff)
        mins=[]
        for mask in [ef,~ef]:mins.append(stats.loc[mask].groupby('PLAYER_ID').MIN.sum().reindex([ids[i] for i in take],fill_value=0).to_numpy(float))
        ec=conc(e.sum(1));lc=conc(l.sum(1));eoff=100*ep.PTS.sum()/ep.possessions.sum();loff=100*lp.PTS.sum()/lp.possessions.sum()
        period=dict(team=r.team,passing_style=r.passing_style,early_games=len(ep),late_games=len(lp),matched_players=len(take),early_concentration=ec,late_concentration=lc,
            concentration_change=lc-ec,early_off_rating=eoff,late_off_rating=loff,off_rating_change=loff-eoff,
            all_roster_raw_hhi_change=float(np.sum((late.sum(1)/late.sum())**2)-np.sum((early.sum(1)/early.sum())**2)),
            minutes_concentration_change=conc(mins[1])-conc(mins[0]),early_network_assists=early.sum(),late_network_assists=late.sum(),
            early_official_assists=ep.AST.sum(),late_official_assists=lp.AST.sum(),early_outside_roster_assists=outside)
        period['exact_both_periods']=period['early_network_assists']==period['early_official_assists'] and period['late_network_assists']==period['late_official_assists']
        periods.append(period)
    interaction=pd.DataFrame(records);changes=pd.DataFrame(periods)
    interaction['creator_c']=interaction.creator_concentration-interaction.creator_concentration.mean();interaction['entropy_c']=interaction.main_creator_recipient_entropy-interaction.main_creator_recipient_entropy.mean();interaction['creator_x_entropy']=interaction.creator_c*interaction.entropy_c
    fits=season_topics_fit(interaction,'creator_x_entropy','off_rating','primary_creator_entropy_interaction','interaction',['creator_c','entropy_c','creator_x_entropy'])
    for controls,label,data in [(['creator_c','entropy_c','creator_x_entropy','minutes_concentration','assist_players'],'composition_sensitivity',interaction),
        (['creator_c','entropy_c','creator_x_entropy'],'exact_credit_coverage',interaction.loc[interaction.exact_assist_total.eq(True)])]:
        fits.extend(season_topics_fit(data,'creator_x_entropy','off_rating',label,'interaction',controls))
    changefits=season_topics_fit(changes,'concentration_change','off_rating_change','primary_period_changes','changes')
    changefits.extend(season_topics_fit(changes,'all_roster_raw_hhi_change','off_rating_change','raw_hhi_all_roster_sensitivity','changes'))
    changefits.extend(season_topics_fit(changes,'concentration_change','off_rating_change','minutes_change_sensitivity','changes',['concentration_change','minutes_concentration_change']))
    strict=changes.loc[changes.exact_both_periods.eq(True)]
    if len(strict)>4:changefits.extend(season_topics_fit(strict,'concentration_change','off_rating_change','exact_both_periods_sensitivity','changes'))
    for style,g in changes.groupby('passing_style'):changefits.extend(season_topics_fit(g,'concentration_change','off_rating_change','style_'+style,'changes'))
    primary_interaction=next(r for r in fits if r['model']=='primary_creator_entropy_interaction' and r['term']=='creator_x_entropy')
    primary_changes=next(r for r in changefits if r['model']=='primary_period_changes' and r['term']=='concentration_change')
    primary=pd.DataFrame([dict(question='Concentration x main creator entropy',n=primary_interaction['n'],effect=primary_interaction['coefficient'],p_value=primary_interaction['p_value_hc3']),
        dict(question='Within-team concentration change vs offense change',n=primary_changes['n'],effect=primary_changes['coefficient'],r=primary_changes['pearson_r'],r_squared=primary_changes['r_squared'],p_value=primary_changes['p_value_hc3'])])
    primary['p_holm_two']=np.nan;run=0
    for rank,idx in enumerate(primary.p_value.sort_values().index):run=max(run,min(1,primary.loc[idx,'p_value']*(2-rank)));primary.loc[idx,'p_holm_two']=run
    # Supply to the main creator's three preferred recipients. Compare rates per team-game.
    hubs=pd.read_csv(parent/'hub_absence_resilience/team_season_metrics.csv');plan=pd.read_csv(parent/'hub_absence_resilience/download_plan.csv');succ=[];detail=[];eligibility=[];split_audit=[]
    for _,r in interaction.iterrows():
        match=hubs.loc[hubs.team.eq(r.team)];eligible=len(match)==1 and int(match.iloc[0].hub_id)==int(r.main_creator_id)
        eligibility.append(dict(team=r.team,main_creator=r.main_assister,eligible=eligible,reason='saved main-creator absence blocks available' if eligible else 'no saved absence blocks for this main creator; not substituted with another player'))
        if not eligible:continue
        hr=match.iloc[0];ids,fullpass=fulls[r.team];ix={pid:i for i,pid in enumerate(ids)};a=assists[r.team];absent=np.zeros_like(a);abspass=np.zeros_like(a)
        tasks=plan.loc[plan.team.eq(r.team)];split_error=''
        for _,task in tasks.iterrows():
            path=parent/'hub_absence_resilience/absent_blocks'/r.team/(task.start+'_'+task.end)/(str(int(task.player_id))+'.csv')
            try:v,p,_=load_edges(path,ids)
            except ValueError as error:
                split_error=str(error)+' in '+str(path);break
            absent+=v;abspass+=p
        if split_error:
            eligibility[-1]['eligible']=False;eligibility[-1]['reason']=split_error
            continue
        savedpresent=pd.read_csv(parent/'hub_absence_resilience/present_networks'/r.team/'passing_matrix.csv',index_col=0)
        if not np.array_equal(abspass,fullpass-savedpresent.to_numpy(float)):raise ValueError('Absence passing split mismatch '+r.team)
        present=a-absent
        if present.min() < -1e-8:
            eligibility[-1]['eligible']=False;eligibility[-1]['reason']='Negative present assist counts after subtraction';continue
        present=np.maximum(present,0);hub=ix[int(r.main_creator_id)]
        if absent[hub].sum()>0 or absent[:,hub].sum()>0:raise ValueError('Absent creator has assist edges')
        targets=np.argsort(-a[hub],kind='stable')[:3];targets=[j for j in targets if a[hub,j]>0]
        if len(targets)!=3:raise ValueError('Main creator has fewer than three recipients')
        own=players.loc[players.TEAM_ID.eq(r.team_id)];names=own.drop_duplicates('PLAYER_ID').set_index('PLAYER_ID').PLAYER_NAME.to_dict()
        before=present[:,targets].sum(1)/hr.present_games;after=absent[:,targets].sum(1)/hr.absent_games;before[hub]=0;after[hub]=0
        gain=np.maximum(after-before,0);gain[hub]=0;total=gain.sum();best=int(np.argmax(gain));share=gain[best]/total if total>0 else np.nan
        tg=games.loc[games.TEAM_ID.eq(r.team_id)];played=set(own.loc[own.PLAYER_ID.eq(int(r.main_creator_id)),'GAME_ID']);mask=tg.GAME_ID.isin(played)
        split_audit.append(dict(team=r.team,present_network_assists=present.sum(),present_official_assists=tg.loc[mask,'AST'].sum(),absent_network_assists=absent.sum(),absent_official_assists=tg.loc[~mask,'AST'].sum()))
        targetnames='; '.join(names[ids[j]] for j in targets)
        succ.append(dict(team=r.team,main_creator=r.main_assister,original_top_three_recipients=targetnames,
            main_creator_assists_to_targets_per_present_game=a[hub,targets].sum()/hr.present_games,
            largest_gainer=names[ids[best]] if total>0 else '',largest_gainer_additional_assists_per_game=gain[best],largest_gainer_share_of_positive_gains=share,
            positive_gainers=int((gain>1e-12).sum()),effective_gainers=1/np.sum((gain/total)**2) if total>0 else np.nan,
            nonhub_supply_present_per_game=before.sum(),nonhub_supply_absent_per_game=after.sum(),
            majority_gainer=bool(share>=.5) if total>0 else False,present_games=int(hr.present_games),absent_games=int(hr.absent_games)))
        for i,pid in enumerate(ids):
            if i!=hub:detail.append(dict(team=r.team,player_id=pid,name=names[pid],assists_to_targets_per_game_present=before[i],assists_to_targets_per_game_absent=after[i],change=after[i]-before[i],positive_gain=gain[i]))
    succession=pd.DataFrame(succ)
    for data,name in [(interaction,'interaction_team_metrics'),(changes,'period_changes'),(pd.DataFrame(fits),'interaction_regressions'),(pd.DataFrame(changefits),'change_regressions'),(primary,'primary_results'),(succession,'recipient_succession'),(pd.DataFrame(detail),'recipient_supplier_changes'),(pd.DataFrame(eligibility),'succession_eligibility'),(pd.DataFrame(split_audit),'absence_assist_coverage')]:atomic_csv(data,out/(name+'.csv'))
    with PdfPages(out/'assist_creator_followups.pdf') as pdf:
        fig,ax=plt.subplots(figsize=(11,8))
        for style,col,marker,name in [('more_balanced','#2473b8','o','Balanced'),('more_star_centered','#e18a24','^','Star-centered')]:
            g=interaction.loc[interaction.passing_style.eq(style)];ax.scatter(g.creator_concentration,g.off_rating,color=col,marker=marker,label=name)
        b={r['term']:r['coefficient'] for r in fits if r['model']=='primary_creator_entropy_interaction'};xx=np.linspace(interaction.creator_concentration.min(),interaction.creator_concentration.max(),100)
        for quantile,color,ls in [(.25,'#555555','--'),(.75,'#111111','-')]:
            e=interaction.main_creator_recipient_entropy.quantile(quantile)-interaction.main_creator_recipient_entropy.mean();x=xx-interaction.creator_concentration.mean();yy=b['intercept']+b['creator_c']*x+b['entropy_c']*e+b['creator_x_entropy']*x*e
            ax.plot(xx,yy,color=color,ls=ls,label=('Lower' if quantile==.25 else 'Higher')+' creator recipient entropy (25th/75th percentile)')
        ax.set(xlabel='Creator concentration (normalized assist-provider HHI)',ylabel='Offensive rating',title=f"Does diverse targeting change the concentration–offense relationship?\nInteraction HC3 p={primary_interaction['p_value_hc3']:.3g}");ax.legend(fontsize=8);ax.grid(alpha=.15);fig.tight_layout();pdf.savefig(fig);fig.savefig(out/'interaction.png',dpi=150);fig.savefig(out/'interaction.pdf');plt.close(fig)
        fig,ax=plt.subplots(figsize=(11,8))
        for style,col,marker,name in [('more_balanced','#2473b8','o','Balanced'),('more_star_centered','#e18a24','^','Star-centered')]:
            g=changes.loc[changes.passing_style.eq(style)];ax.scatter(g.concentration_change,g.off_rating_change,color=col,marker=marker,label=name)
            for _,v in g.iterrows():ax.annotate(v.team,(v.concentration_change,v.off_rating_change),xytext=(4,4),textcoords='offset points',fontsize=8)
        slope=primary_changes['coefficient'];intercept=next(r['coefficient'] for r in changefits if r['model']=='primary_period_changes' and r['term']=='intercept');xx=np.linspace(changes.concentration_change.min(),changes.concentration_change.max(),100);ax.plot(xx,intercept+slope*xx,color='#333333')
        ax.axhline(0,color='gray',lw=.7);ax.axvline(0,color='gray',lw=.7);ax.set(xlabel='Late minus early creator concentration (matched-player network)',ylabel='Late minus early offensive rating',title=f"Within-team changes; calendar split {cutoff}\nr={primary_changes['pearson_r']:.3f}; R²={primary_changes['r_squared']:.3f}; HC3 p={primary_changes['p_value_hc3']:.3g}");ax.legend();ax.grid(alpha=.15);fig.tight_layout();season_topics_spread_labels(fig);pdf.savefig(fig);fig.savefig(out/'period_changes.png',dpi=150);fig.savefig(out/'period_changes.pdf');plt.close(fig)
        fig,ax=plt.subplots(figsize=(12,8));z=succession.sort_values('largest_gainer_share_of_positive_gains');labels=z.team+' — '+z.largest_gainer
        ax.barh(labels,100*z.largest_gainer_share_of_positive_gains,color='#2473b8');ax.axvline(50,color='gray',ls='--');ax.set(xlim=(0,100),xlabel='Largest supplier’s share of positive assist-rate gains to the creator’s top three recipients (%)',title='Who supplies an absent creator’s usual recipients?\nMajority threshold is descriptive; rates are per team-game');fig.tight_layout();pdf.savefig(fig);fig.savefig(out/'recipient_succession.png',dpi=150);fig.savefig(out/'recipient_succession.pdf');plt.close(fig)
    report=['# Three creator follow-ups — 2025–26 NBA',primary.to_string(index=False),
        'Interaction: creator concentration x main creator recipient entropy, centered before multiplication, with both main effects retained. Conditional on successful assisted baskets; no causal claim. Low/high lines are model predictions at entropy quartiles, not a split-sample regression. Main creator chosen by full-season assists supplied. Main entropy denominator log(active assist N−1). HC3 errors, 30 teams; exact-total 20-team and minutes/player-count sensitivities saved.',
        'Changes: early <= '+cutoff+', late > cutoff; common active assist participants in both periods, iteratively matched induced panel. Calendar periods have unequal game counts; offensive rating uses official summed PTS/POSS. Early AST from saved early profiles, late = full minus early. PASS reconstruction must exactly match saved early network, AST differences must be nonnegative. Outcomes cover full team, while matched network intentionally omits players participating in only one period. All-roster raw-HHI sensitivity includes those changes. Per-period assist totals compared with official logs; strict both-period exact subset saved. No claim that early-period concentration predicts future outcomes; this is a change/change association.',
        'Succession: initially 19 candidate teams where original EC-hub absence blocks correspond to main assister. TOR/UTA have different main assisters; nine others lack required saved absence samples, so are not silently substituted. Invalid saved assist splits are excluded and named in succession_eligibility.csv; LAL has impossible AST>PASS rows. Original creator’s top three recipients fixed using full-season credited assist row. Positive gain = max(absent assists/team-game to those targets minus other creators’ present rate,0), original hub excluded. Largest positive-gain share is descriptive; >=50% denotes one majority gainer, not statistical evidence of a new star. Positive clipping does not measure how much of the lost creator’s production is replaced. Source rates reflect player appearances, trades, target availability and co-absences; no playing-time or pair-court-time controls. Aggregate absence blocks cannot prove the same replacement leads each game. Split assist coverage saved. A favorite recipient missing simultaneously can change apparent inheritance.',
        'Two inferential primary questions corrected together with Holm-two; succession is descriptive. This correction does not cover all prior exploration. Additional controls and subgroup checks are exploratory unadjusted sensitivities. No new downloads. Directed assists, not all chances or potential assists. Full prior CLI preserved.',
        pd.DataFrame(fits).to_string(index=False),pd.DataFrame(changefits).to_string(index=False),succession.to_string(index=False)]
    (out/'README.md').write_text('\n\n'.join(report));print(primary.to_string(index=False),flush=True);print('Exact period totals',int(changes.exact_both_periods.sum()),flush=True);print(succession[['team','main_creator','largest_gainer','largest_gainer_share_of_positive_gains','positive_gainers','majority_gainer']].to_string(index=False),flush=True);print('Saved creator follow-ups: '+str(out),flush=True);return 0



def run_passing_hub_followups(argv):
    """Repeat the three assist follow-ups on all-pass networks, using original EC hub definition."""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import json
    parser=argparse.ArgumentParser(description='Passing-network hub diversity, recipient succession and period changes')
    parser.add_argument('--passing-hub-followups',action='store_true');parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT_DIR)
    args=parser.parse_args(argv);parent=args.output_dir.expanduser()/'2025-26/team_season_topics';out=parent/'passing_hub_followups';out.mkdir(parents=True,exist_ok=True)
    season=pd.read_csv(parent/'team_season_metrics.csv');players=pd.read_csv(parent/'player_games.csv',dtype={'GAME_ID':str});games=pd.read_csv(parent/'team_games.csv',dtype={'GAME_ID':str});adv=pd.read_csv(parent/'official_team_advanced.csv').set_index('TEAM_ID')
    gameadv=pd.read_csv(parent/'hub_absence_resilience/official_game_advanced.csv',dtype={'GAME_ID':str});poss=gameadv.set_index(['TEAM_ID','GAME_ID']).POSS.to_dict();games['possessions']=[poss[(r.TEAM_ID,r.GAME_ID)] for _,r in games.iterrows()]
    cutoff=json.loads((parent/'eight_context_topics/calendar_split.json').read_text())['cutoff']
    def conc(v):
        q=np.asarray(v,float);q=q/q.sum();return float((q@q-1/len(q))/(1-1/len(q)))
    def row_entropy(v,N):
        p=v[v>0]/v.sum();return float(-np.sum(p*np.log(p))/np.log(N-1))
    rows=[];periodrows=[];full={}
    for _,r in season.iterrows():
        f=pd.read_csv(parent/'networks'/r.team/'passing_matrix.csv',index_col=0);ids=f.index.astype(int).tolist();a=f.to_numpy(float);full[r.team]=(ids,a)
        ec,scores,activeids,active=eigenvector_passing_concentration(a,ids)
        if not ec['network_valid']:raise ValueError('Invalid season EC '+r.team)
        hub=int(ec['top_eigenvector_player_id']);index=ids.index(hub);own=players.loc[players.TEAM_ID.eq(r.team_id)];names=own.drop_duplicates('PLAYER_ID').set_index('PLAYER_ID').PLAYER_NAME.to_dict()
        rows.append(dict(team=r.team,team_id=int(r.team_id),passing_style=r.passing_style,hub_id=hub,hub_name=names[hub],eigenvector_concentration=ec['eigenvector_concentration'],provider_concentration=conc(active.sum(1)),
            hub_recipient_entropy=row_entropy(a[index],len(activeids)),team_conditional_entropy=compute_team_entropies(active)[1],off_rating=adv.loc[r.team_id,'OFF_RATING'],
            player_count=len(activeids),minutes_concentration=conc(own.groupby('PLAYER_ID').MIN.sum().reindex(activeids,fill_value=0))))
        ep=pd.read_csv(parent/'eight_context_topics/split_networks'/r.team/'early.csv',index_col=0);lp=pd.read_csv(parent/'eight_context_topics/split_networks'/r.team/'late.csv',index_col=0)
        if ep.index.astype(int).tolist()!=ids or lp.index.astype(int).tolist()!=ids:raise ValueError('Split index mismatch')
        e=ep.to_numpy(float);l=lp.to_numpy(float)
        if not np.array_equal(e+l,a):raise ValueError('Period pass matrices do not sum to full season')
        take=np.flatnonzero(((e.sum(0)+e.sum(1))>0)&((l.sum(0)+l.sum(1))>0)).tolist()
        while True:
            ee=e[np.ix_(take,take)];ll=l[np.ix_(take,take)];keep=(ee.sum(0)+ee.sum(1)>0)&(ll.sum(0)+ll.sum(1)>0)
            if keep.all():break
            take=[i for i,k in zip(take,keep) if k]
        panel=[ids[i] for i in take];ce,_,_,_=eigenvector_passing_concentration(ee,panel);cl,_,_,_=eigenvector_passing_concentration(ll,panel)
        if not ce['network_valid'] or not cl['network_valid']:raise ValueError('Invalid matched period EC')
        tg=games.loc[games.TEAM_ID.eq(r.team_id)];first=tg.GAME_DATE.astype(str).str[:10].le(cutoff);early=tg.loc[first];late=tg.loc[~first]
        old=100*early.PTS.sum()/early.possessions.sum();new=100*late.PTS.sum()/late.possessions.sum();mins=[]
        ef=own.GAME_DATE.astype(str).str[:10].le(cutoff)
        for mask in [ef,~ef]:mins.append(own.loc[mask].groupby('PLAYER_ID').MIN.sum().reindex(panel,fill_value=0).to_numpy(float))
        periodrows.append(dict(team=r.team,passing_style=r.passing_style,matched_players=len(panel),early_games=len(early),late_games=len(late),
            early_concentration=ce['eigenvector_concentration'],late_concentration=cl['eigenvector_concentration'],concentration_change=cl['eigenvector_concentration']-ce['eigenvector_concentration'],
            provider_concentration_change=conc(ll.sum(1))-conc(ee.sum(1)),raw_all_roster_provider_hhi_change=np.sum((l.sum(1)/l.sum())**2)-np.sum((e.sum(1)/e.sum())**2),
            early_off_rating=old,late_off_rating=new,off_rating_change=new-old,minutes_concentration_change=conc(mins[1])-conc(mins[0]),
            early_passes=e.sum(),late_passes=l.sum()))
    teams=pd.DataFrame(rows);changes=pd.DataFrame(periodrows)
    teams['concentration_c']=teams.eigenvector_concentration-teams.eigenvector_concentration.mean();teams['entropy_c']=teams.hub_recipient_entropy-teams.hub_recipient_entropy.mean();teams['interaction']=teams.concentration_c*teams.entropy_c
    fits=season_topics_fit(teams,'interaction','off_rating','primary_ec_x_hub_entropy','interaction',['concentration_c','entropy_c','interaction'])
    fits.extend(season_topics_fit(teams,'interaction','off_rating','composition_sensitivity','interaction',['concentration_c','entropy_c','interaction','minutes_concentration','player_count']))
    z=teams.copy();z['provider_c']=z.provider_concentration-z.provider_concentration.mean();z['provider_x_entropy']=z.provider_c*z.entropy_c
    fits.extend(season_topics_fit(z,'provider_x_entropy','off_rating','outgoing_pass_share_sensitivity','interaction',['provider_c','entropy_c','provider_x_entropy']))
    cf=season_topics_fit(changes,'concentration_change','off_rating_change','primary_period_changes','changes')
    cf.extend(season_topics_fit(changes,'concentration_change','off_rating_change','minutes_change_sensitivity','changes',['concentration_change','minutes_concentration_change']))
    cf.extend(season_topics_fit(changes,'provider_concentration_change','off_rating_change','outgoing_pass_share_sensitivity','changes'))
    cf.extend(season_topics_fit(changes,'raw_all_roster_provider_hhi_change','off_rating_change','all_roster_raw_provider_hhi_sensitivity','changes'))
    for style,g in changes.groupby('passing_style'):cf.extend(season_topics_fit(g,'concentration_change','off_rating_change','style_'+style,'changes'))
    ir=next(v for v in fits if v['model']=='primary_ec_x_hub_entropy' and v['term']=='interaction');cr=next(v for v in cf if v['model']=='primary_period_changes' and v['term']=='concentration_change')
    primary=pd.DataFrame([dict(question='EC concentration x hub recipient entropy',n=ir['n'],coefficient=ir['coefficient'],p_value=ir['p_value_hc3']),dict(question='Within-team passing concentration change vs offense change',n=cr['n'],coefficient=cr['coefficient'],r=cr['pearson_r'],r_squared=cr['r_squared'],p_value=cr['p_value_hc3'])]);primary['p_holm_two']=np.nan;running=0
    for rank,idx in enumerate(primary.p_value.sort_values().index):running=max(running,min(1,primary.loc[idx,'p_value']*(2-rank)));primary.loc[idx,'p_holm_two']=running
    # Unlike assist fields, saved passing splits are validated for all 21 eligible original EC hubs.
    hubmetrics=pd.read_csv(parent/'hub_absence_resilience/team_season_metrics.csv');succ=[];details=[]
    for _,r in hubmetrics.loc[hubmetrics.primary_eligible.eq(True)].iterrows():
        ids,a=full[r.team];p=pd.read_csv(parent/'hub_absence_resilience/present_networks'/r.team/'passing_matrix.csv',index_col=0)
        if p.index.astype(int).tolist()!=ids:raise ValueError('Present panel mismatch')
        beforemat=p.to_numpy(float);after=a-beforemat;hub=ids.index(int(r.hub_id))
        if after.min()<0 or after[hub].sum()+after[:,hub].sum()>0:raise ValueError('Invalid absent passing matrix')
        targets=np.argsort(-a[hub],kind='stable')[:3]
        if not (a[hub,targets]>0).all():raise ValueError('Too few preferred receivers')
        names=players.loc[players.TEAM_ID.eq(r.team_id)].drop_duplicates('PLAYER_ID').set_index('PLAYER_ID').PLAYER_NAME.to_dict()
        before=beforemat[:,targets].sum(1)/r.present_games;afterrate=after[:,targets].sum(1)/r.absent_games;before[hub]=0;afterrate[hub]=0;gain=np.maximum(afterrate-before,0);total=gain.sum();best=int(np.argmax(gain));share=gain[best]/total if total>0 else np.nan
        succ.append(dict(team=r.team,hub_name=r.hub_name,preferred_receivers='; '.join(names[ids[j]] for j in targets),largest_gainer=names[ids[best]] if total>0 else '',
            top_gainer_share=share,positive_gainers=int((gain>1e-12).sum()),effective_gainers=1/np.sum((gain/total)**2) if total>0 else np.nan,majority_gainer=bool(share>=.5) if total>0 else False,
            original_hub_passes_to_targets_per_present_game=a[hub,targets].sum()/r.present_games,nonhub_supply_present_per_game=before.sum(),nonhub_supply_absent_per_game=afterrate.sum(),
            largest_gain_passes_per_game=gain[best],present_games=int(r.present_games),absent_games=int(r.absent_games)))
        for i,pid in enumerate(ids):
            if i!=hub:details.append(dict(team=r.team,player_id=pid,name=names[pid],passes_to_targets_per_game_present=before[i],passes_to_targets_per_game_absent=afterrate[i],change=afterrate[i]-before[i],positive_gain=gain[i]))
    succession=pd.DataFrame(succ)
    for d,name in [(teams,'interaction_team_metrics'),(changes,'period_changes'),(pd.DataFrame(fits),'interaction_regressions'),(pd.DataFrame(cf),'change_regressions'),(primary,'primary_results'),(succession,'receiver_succession'),(pd.DataFrame(details),'receiver_supplier_changes')]:atomic_csv(d,out/(name+'.csv'))
    with PdfPages(out/'passing_hub_followups.pdf') as pdf:
        fig,ax=plt.subplots(figsize=(11,8))
        for style,col,marker,name in [('more_balanced','#2473b8','o','Balanced'),('more_star_centered','#e18a24','^','Star-centered')]:
            g=teams.loc[teams.passing_style.eq(style)];ax.scatter(g.eigenvector_concentration,g.off_rating,color=col,marker=marker,label=name)
        b={v['term']:v['coefficient'] for v in fits if v['model']=='primary_ec_x_hub_entropy'};xx=np.linspace(teams.eigenvector_concentration.min(),teams.eigenvector_concentration.max(),100)
        for q,color,ls in [(.25,'#666666','--'),(.75,'#111111','-')]:
            e=teams.hub_recipient_entropy.quantile(q)-teams.hub_recipient_entropy.mean();x=xx-teams.eigenvector_concentration.mean();yy=b['intercept']+b['concentration_c']*x+b['entropy_c']*e+b['interaction']*x*e;ax.plot(xx,yy,color=color,ls=ls,label=('Lower' if q==.25 else 'Higher')+' hub receiver entropy (25th/75th percentile)')
        ax.set(xlabel='Eigenvector passing concentration',ylabel='Offensive rating',title=f"Does the hub’s passing diversity change the concentration–offense relationship?\nInteraction HC3 p={ir['p_value_hc3']:.3g}");ax.legend(fontsize=8);ax.grid(alpha=.15);fig.tight_layout();pdf.savefig(fig);fig.savefig(out/'interaction.pdf');fig.savefig(out/'interaction.png',dpi=150);plt.close(fig)
        fig,ax=plt.subplots(figsize=(11,8))
        for style,col,marker,name in [('more_balanced','#2473b8','o','Balanced'),('more_star_centered','#e18a24','^','Star-centered')]:
            g=changes.loc[changes.passing_style.eq(style)];ax.scatter(g.concentration_change,g.off_rating_change,color=col,marker=marker,label=name)
            for _,v in g.iterrows():ax.annotate(v.team,(v.concentration_change,v.off_rating_change),xytext=(4,4),textcoords='offset points',fontsize=8)
        xx=np.linspace(changes.concentration_change.min(),changes.concentration_change.max(),100);intercept=next(v['coefficient'] for v in cf if v['model']=='primary_period_changes' and v['term']=='intercept');ax.plot(xx,intercept+cr['coefficient']*xx,color='#333333');ax.axhline(0,color='gray',lw=.7);ax.axvline(0,color='gray',lw=.7)
        ax.set(xlabel='Late minus early eigenvector concentration (matched-player network)',ylabel='Late minus early offensive rating',title=f"Within-team passing changes; calendar split {cutoff}\nr={cr['pearson_r']:.3f}; R²={cr['r_squared']:.3f}; HC3 p={cr['p_value_hc3']:.3g}");ax.legend();ax.grid(alpha=.15);fig.tight_layout();season_topics_spread_labels(fig);pdf.savefig(fig);fig.savefig(out/'period_changes.pdf');fig.savefig(out/'period_changes.png',dpi=150);plt.close(fig)
        fig,ax=plt.subplots(figsize=(12,8));z=succession.sort_values('top_gainer_share');ax.barh(z.team+' — '+z.largest_gainer,100*z.top_gainer_share,color='#2473b8');ax.axvline(50,color='gray',ls='--');ax.set(xlim=(0,100),xlabel='Largest supplier’s share of positive pass-rate gains to the hub’s top three receivers (%)',title='Who supplies the absent passing hub’s usual receivers?\n21 eligible teams; majority threshold is descriptive');fig.tight_layout();pdf.savefig(fig);fig.savefig(out/'receiver_succession.pdf');fig.savefig(out/'receiver_succession.png',dpi=150);plt.close(fig)
    report=['# All-pass network follow-ups — NBA 2025–26',primary.to_string(index=False),
        'Primary hub definition preserved: highest L1 eigenvector share in weighted A+A.T, with normalized HHI of shares measuring passing concentration. This differs intentionally from the assist study’s assist-provider HHI. Outgoing-pass-share HHI is included as the direct provider-concentration sensitivity. Hub receiver entropy = entropy of the original hub’s outgoing recipient proportions / log(active N−1), not whole-team conditional entropy. Same original 15/15 passing style labels; no primary minutes or player-count controls.',
        'Interaction: both centered main effects and their product included; quartile lines are predictions, not separate low/high samples. HC3 inference; optional minutes/player-count sensitivity. Thirty teams, observational; concentration and hub entropy share network inputs.',
        'Period comparison: early <= '+cutoff+', late > cutoff. Saved early+late passing matrices must exactly equal full-season matrices. Same iteratively matched active players used in both induced networks; isolates excluded. Official team PTS/POSS yields period offensive ratings; full-team outcome includes roster changes omitted from matched passing panel. Outgoing concentration and all-roster raw outgoing HHI changes saved as sensitivity checks. Period game counts differ; pass counts are season-period totals. No claim of future prediction or a causal effect of changing tactics.',
        'Receiver succession: all 21 eligible original EC-hub networks, >=20 present and >=10 absent games. Includes LAL because this uses valid PASS matrices, not its invalid AST split. For original hub’s three most-passed-to teammates, calculate each remaining supplier’s absent minus present passes per team-game. Positive gains clipped at zero; largest positive-gain share >=50% means one majority gainer descriptively, not statistical proof of a replacement star. Positive gains do not measure fraction of the absent hub’s production replaced. Recipient availability, minutes, roster moves and co-absences are not controlled. Same person need not lead every game. No additional downloads; no claim that soccer was tested.',
        'Two primary inferential tests corrected using Holm-two. Succession descriptive. This correction does not cover all prior exploration; sensitivities and subgroup comparisons unadjusted.',pd.DataFrame(fits).to_string(index=False),pd.DataFrame(cf).to_string(index=False),succession.to_string(index=False)]
    (out/'README.md').write_text('\n\n'.join(report));print(primary.to_string(index=False),flush=True);print(succession[['team','hub_name','largest_gainer','top_gainer_share','positive_gainers','majority_gainer']].to_string(index=False),flush=True);print('Saved all-pass follow-ups: '+str(out),flush=True);return 0



def run_passing_lineup_balance(argv):
    """Passing volume versus playing-time balance, with original eigenvector styles."""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    parser=argparse.ArgumentParser(description='Does greater passing volume accompany balanced playing time and passing influence?')
    parser.add_argument('--passing-lineup-balance',action='store_true');parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT_DIR)
    args=parser.parse_args(argv);parent=args.output_dir.expanduser()/'2025-26/team_season_topics';out=parent/'passing_lineup_balance';out.mkdir(parents=True,exist_ok=True)
    season=pd.read_csv(parent/'team_season_metrics.csv');players=pd.read_csv(parent/'player_games.csv',dtype={'GAME_ID':str});rows=[]
    for _,r in season.iterrows():
        f=pd.read_csv(parent/'networks'/r.team/'passing_matrix.csv',index_col=0);a=f.to_numpy(float);ids=f.index.astype(int).tolist();ec,_,_,_=eigenvector_passing_concentration(a,ids)
        if not ec['network_valid'] or not np.isclose(a.sum(),r.recorded_passes):raise ValueError('Network audit failed '+r.team)
        m=players.loc[players.TEAM_ID.eq(r.team_id)].groupby('PLAYER_ID').MIN.sum();m=m.loc[m.gt(0)];q=m/m.sum();N=len(q);raw=float(q@q);concentration=(raw-1/N)/(1-1/N)
        rows.append(dict(team=r.team,team_id=int(r.team_id),passing_style=r.passing_style,passes=a.sum(),possessions=r.possessions,games=r.games,
            passes_per100=100*a.sum()/r.possessions,passes_per_game=a.sum()/r.games,eigenvector_concentration=ec['eigenvector_concentration'],
            minutes_concentration=concentration,raw_minutes_hhi=raw,effective_minutes_players=1/raw,player_count=N))
    d=pd.DataFrame(rows);specs=[(1,'passes_per100','minutes_concentration','Passing volume and playing-time balance','Player-to-player passes per 100 possessions','Playing-time concentration (normalized minutes HHI; lower = more balanced)'),
        (2,'passes_per100','eigenvector_concentration','Passing volume and passing-network balance','Player-to-player passes per 100 possessions','Eigenvector passing concentration (lower = more balanced)'),
        (3,'eigenvector_concentration','minutes_concentration','Passing influence and playing-time balance','Eigenvector passing concentration','Playing-time concentration (normalized minutes HHI)')]
    results=[];fits=[];styles=[];interactions=[];sensitivity=[]
    for k,x,y,title,xlabel,ylabel in specs:
        f=season_topics_fit(d,x,y,title,k);fits.extend(f);r=next(v for v in f if v['term']==x)
        results.append(dict(question=k,title=title,x=x,y=y,n=r['n'],r=r['pearson_r'],r_squared=r['r_squared'],slope=r['coefficient'],p_value=r['p_value_hc3']))
        for style,g in d.groupby('passing_style'):styles.extend(dict(v,passing_style=style) for v in season_topics_fit(g,x,y,'style_'+str(k),k))
        z=d.copy();z['star_indicator']=z.passing_style.eq('more_star_centered').astype(float);z['metric_x_star']=z[x]*z.star_indicator
        interactions.extend(season_topics_fit(z,x,y,'style_interaction_'+str(k),k,[x,'star_indicator','metric_x_star']))
    result=pd.DataFrame(results);result['p_holm_three']=np.nan;running=0
    for rank,idx in enumerate(result.p_value.sort_values().index):running=max(running,min(1,result.loc[idx,'p_value']*(3-rank)));result.loc[idx,'p_holm_three']=running
    sensitivity.extend(season_topics_fit(d,'passes_per100','raw_minutes_hhi','raw_minutes_hhi','sensitivity'))
    sensitivity.extend(season_topics_fit(d,'passes_per_game','minutes_concentration','per_game_volume','sensitivity'))
    sensitivity.extend(season_topics_fit(d,'passes_per100','minutes_concentration','player_count_sensitivity','sensitivity',['passes_per100','player_count']))
    robustness=[];leave_out=[]
    for x,y,controls,label in [('passes_per100','raw_minutes_hhi',[],'raw_minutes_interaction'),('passes_per_game','minutes_concentration',[],'per_game_interaction'),('passes_per100','minutes_concentration',['player_count'],'player_count_interaction')]:
        z=d.copy();z['star_indicator']=z.passing_style.eq('more_star_centered').astype(float);z['volume_x_star']=z[x]*z.star_indicator
        robustness.extend(season_topics_fit(z,x,y,label,'robustness',[x,'star_indicator','volume_x_star',*controls]))
    for team in d.team:
        z=d.loc[d.team.ne(team)].copy();z['star_indicator']=z.passing_style.eq('more_star_centered').astype(float);z['volume_x_star']=z.passes_per100*z.star_indicator
        leave_out.extend(dict(v,omitted_team=team) for v in season_topics_fit(z,'passes_per100','minutes_concentration','leave_one_team_out','robustness',['passes_per100','star_indicator','volume_x_star']))
    atomic_csv(pd.DataFrame(robustness),out/'interaction_robustness.csv');atomic_csv(pd.DataFrame(leave_out),out/'interaction_leave_one_out.csv')
    for frame,name in [(d,'team_metrics'),(result,'primary_results'),(pd.DataFrame(fits),'regressions'),(pd.DataFrame(styles),'style_regressions'),(pd.DataFrame(interactions),'style_interactions'),(pd.DataFrame(sensitivity),'sensitivities')]:atomic_csv(frame,out/(name+'.csv'))
    with PdfPages(out/'passing_lineup_balance.pdf') as pdf:
        for k,x,y,title,xlabel,ylabel in specs:
            fig,ax=plt.subplots(figsize=(11,8));r=result.loc[result.question.eq(k)].iloc[0]
            for style,col,marker,name in [('more_balanced','#2473b8','o','Balanced passing structure'),('more_star_centered','#e18a24','^','Star-centered passing structure')]:
                g=d.loc[d.passing_style.eq(style)];fit=season_topics_fit(g,x,y,'plot',k);s=next(v for v in fit if v['term']==x);intercept=next(v['coefficient'] for v in fit if v['term']=='intercept')
                ax.scatter(g[x],g[y],color=col,marker=marker,label=name+(f"; r={s['pearson_r']:.3f}" if k==1 else ''))
                # Style-split EC relationships truncate the predictor/outcome; pooled line is primary.
                if k==1:
                    xx=np.linspace(g[x].min(),g[x].max(),100);ax.plot(xx,intercept+s['coefficient']*xx,color=col)
                for _,v in g.iterrows():ax.annotate(v.team,(v[x],v[y]),xytext=(4,4),textcoords='offset points',fontsize=8)
            xx=np.linspace(d[x].min(),d[x].max(),100);intercept=next(v['coefficient'] for v in fits if v['topic']==k and v['term']=='intercept');ax.plot(xx,intercept+r.slope*xx,color='#333333',ls='--',label='All teams')
            ax.set(xlabel=xlabel,ylabel=ylabel,title=title+'\n'+f'NBA 2025–26 | r={r.r:.3f} | R²={r.r_squared:.3f} | Holm-three p={r.p_holm_three:.3g}');ax.legend(fontsize=8);ax.grid(alpha=.15);ax.margins(.12)
            fig.text(.5,.015,'30 teams · no primary minutes or player-count controls · original eigenvector styles retained\nPlaying-time balance describes season minutes distribution, not the composition of individual five-player lineups.',ha='center',fontsize=9)
            fig.tight_layout(rect=(0,.055,1,1));season_topics_spread_labels(fig);pdf.savefig(fig);fig.savefig(out/f'question_{k}.pdf');fig.savefig(out/f'question_{k}.png',dpi=150);plt.close(fig)
    (out/'README.md').write_text('\n\n'.join(['# Passing volume, playing-time balance and eigenvector centrality',
        'Interpretation of user question: combined passing means total completed recorded player-to-player passes, not assists. Count each directed pass once; symmetrizing A+A.T for eigenvector analysis must not double the volume numerator. Main X is passes per 100 official team possessions, with per-game sensitivity. All 30 2025–26 regular-season teams, all participants, no minimum minutes cutoff.',
        'Balanced lineups operationalized as evenly distributed season player minutes (normalized HHI; lower = more even). This is playing-time/rotation balance, not an analysis of five-player lineup compositions, lineup quality, skills or positions. Injuries, trades and season-long participation affect minutes shares. Raw minutes HHI and effective number of minutes-share players saved. Normalization uses all positive-minute official participants; graph isolates omitted from EC calculation.',
        'Eigenvector concentration uses original weighted A+A.T definition and normalized HHI of L1 centrality shares. Original season-median style groups retained, 15 each. Question 1 separates regression lines by style; direct slope interaction saved. Questions 2 and 3 classify on the same EC variable being plotted; subgroup correlations are descriptive restricted-range results, not independent evidence of moderation. Pooled results are primary.',
        'Primary team regressions use HC3 robust t inference, no covariates. Holm-three correction across these three questions; does not adjust all previous topic exploration. Player-count, raw-HHI and per-game sensitivities are exploratory unadjusted. Observational, not causal. Higher passing volume does not necessarily mean higher conditional entropy or better teamwork.',result.to_string(index=False)]))
    print(result.to_string(index=False),flush=True);print(pd.DataFrame(styles).loc[lambda z:z.term.eq(z.x)&z.topic.eq(1)].to_string(index=False),flush=True);print('Saved passing/playing-time balance analysis: '+str(out),flush=True);return 0



def warriors_role_hac(data,outcome,predictors,label,joint_terms=None):
    """HAC7 t tests and optional joint F test; lags use actual season game numbers."""
    from scipy.stats import t,f
    d=data.sort_values('game_number').replace([np.inf,-np.inf],np.nan).dropna(subset=[outcome,*predictors]).copy()
    X=np.column_stack([np.ones(len(d)),d[predictors].to_numpy(float)]);y=d[outcome].to_numpy(float);n,k=X.shape
    if n<=k or np.linalg.matrix_rank(X)<k:raise ValueError('Insufficient role-model sample '+label)
    inv=np.linalg.inv(X.T@X);b=np.linalg.lstsq(X,y,rcond=None)[0];e=y-X@b;score=X*e[:,None];meat=score.T@score;positions={int(g):i for i,g in enumerate(d.game_number)}
    for lag in range(1,8):
        cross=np.zeros((k,k))
        for number,i in positions.items():
            if number-lag in positions:cross+=np.outer(score[i],score[positions[number-lag]])
        meat+=(1-lag/8)*(cross+cross.T)
    cov=n/(n-k)*inv@meat@inv;se=np.sqrt(np.maximum(np.diag(cov),0));terms=['intercept',*predictors]
    rows=[dict(model=label,outcome=outcome,term=term,n=n,coefficient=b[j],standard_error_hac7=se[j],p_value_hac7=2*t.sf(abs(b[j]/se[j]),n-k) if se[j]>0 else np.nan,
        ci95_low=b[j]-t.ppf(.975,n-k)*se[j],ci95_high=b[j]+t.ppf(.975,n-k)*se[j],r_squared=1-(e@e)/np.sum((y-y.mean())**2)) for j,term in enumerate(terms)]
    joint=None
    if joint_terms:
        ix=[terms.index(v) for v in joint_terms];v=b[ix];C=cov[np.ix_(ix,ix)];stat=float(v@np.linalg.solve(C,v)/len(ix));joint=dict(model=label,n=n,F=stat,df_num=len(ix),df_den=n-k,p_value=float(f.sf(stat,len(ix),n-k)))
    return rows,joint


def run_warriors_role_checks(argv):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from scipy.stats import linregress
    parser=argparse.ArgumentParser(description='Warriors 2021-22 rotation balance, hub identity and Curry role without Green')
    parser.add_argument('--warriors-role-checks',action='store_true');parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT_DIR)
    args=parser.parse_args(argv);root=args.output_dir.expanduser()/'GSW_2021-22_82_games';out=root/'role_checks';out.mkdir(parents=True,exist_ok=True)
    base=pd.read_csv(root/'eigenvector_team_performance/team_game_metrics.csv',dtype={'game_id':str});rows=[]
    def conc(v):
        v=np.asarray(v,float);q=v/v.sum();return float((q@q-1/len(q))/(1-1/len(q)))
    for _,r in base.loc[base.network_valid.eq(True)].iterrows():
        number=int(r.game_number);folder=root/f'game_{number:02d}';pg=pd.read_csv(folder/'game_players.csv');pg=pg.loc[pg.MIN.gt(0)].set_index('PLAYER_ID')
        f=pd.read_csv(root/'eigenvector_team_performance'/f'game_{number:02d}'/'passing_matrix.csv',index_col=0);ids=f.index.astype(int).tolist();a=f.to_numpy(float)
        ec,scores,activeids,active=eigenvector_passing_concentration(a,ids)
        if not ec['network_valid'] or not np.isclose(ec['eigenvector_concentration'],r.eigenvector_concentration):raise ValueError('Network mismatch '+str(number))
        row=r.to_dict();row.update(minutes_concentration=conc(pg.MIN),raw_minutes_hhi=float(np.sum((pg.MIN/pg.MIN.sum())**2)),
            curry_present=201939 in pg.index,green_present=203110 in pg.index,klay_present=202691 in pg.index,
            efg_percent=100*r.team_efg_pct,green_hub=int(r.top_eigenvector_player_id==203110),poole_hub=int(r.top_eigenvector_player_id==1629673))
        if 201939 in pg.index:
            i=ids.index(201939);v=pg.loc[201939];q=a[i];q=q[q>0]/q.sum()
            row.update(curry_ec_share=scores.get(201939,np.nan),curry_minutes=v.MIN,curry_minutes_share=v.MIN/pg.MIN.sum(),
                curry_reception_share=a[:,i].sum()/a.sum(),curry_outgoing_share=a[i].sum()/a.sum(),curry_shot_share=v.FGA/r.team_fga,
                curry_recipient_entropy=float(-np.sum(q*np.log(q))/np.log(len(activeids)-1)),curry_efg_percent=100*(v.FGM+.5*v.FG3M)/v.FGA if v.FGA>0 else np.nan,
                green_absent=int(203110 not in pg.index))
            take=[j for j,pid in enumerate(ids) if pid!=203110];sm=a[np.ix_(take,take)];panel=[ids[j] for j in take];check,cs,_,_=eigenvector_passing_concentration(sm,panel)
            row['curry_ec_green_removed_both']=cs.get(201939,np.nan) if check['network_valid'] else np.nan
        rows.append(row)
    data=pd.DataFrame(rows).sort_values('game_number');curry=data.loc[data.curry_present.eq(True)].copy();identity=data.loc[data.top_eigenvector_player_id.isin([201939,203110,1629673])].copy();fits=[];joints=[];primary=[]
    f,_=warriors_role_hac(data,'eigenvector_concentration',['minutes_concentration'],'rotation_primary');fits.extend(f);r=next(v for v in f if v['term']=='minutes_concentration');corr=linregress(data.minutes_concentration,data.eigenvector_concentration).rvalue
    primary.append(dict(question=1,test='Rotation concentration vs passing eigenvector concentration',n=len(data),effect=r['coefficient'],r=corr,r_squared=corr**2,p_value=r['p_value_hac7']))
    f,joint=warriors_role_hac(identity,'efg_percent',['green_hub','poole_hub'],'hub_identity_primary',['green_hub','poole_hub']);fits.extend(f);joints.append(joint)
    primary.append(dict(question=2,test='Curry/Green/Poole hub identity vs team eFG%',n=len(identity),p_value=joint['p_value']))
    f,_=warriors_role_hac(curry,'curry_ec_share',['green_absent'],'curry_without_green_primary');fits.extend(f);r=next(v for v in f if v['term']=='green_absent');primary.append(dict(question=3,test='Curry centrality: Green absent vs present, Curry playing',n=len(curry),effect=r['coefficient'],p_value=r['p_value_hac7']))
    # All secondary outcomes retained, including weak findings.
    for y in ['network_conditional_entropy','team_ast_to_ratio','team_turnovers','point_margin','minutes_concentration']:
        f,j=warriors_role_hac(identity,y,['green_hub','poole_hub'],'hub_identity_'+y,['green_hub','poole_hub']);fits.extend(f);joints.append(j)
    for y in ['curry_recipient_entropy','curry_reception_share','curry_outgoing_share','curry_shot_share','curry_minutes','curry_efg_percent','efg_percent','team_turnovers','team_ast_to_ratio','curry_ec_green_removed_both']:
        f,_=warriors_role_hac(curry,y,['green_absent'],'green_absence_'+y);fits.extend(f)
    f,_=warriors_role_hac(curry,'curry_ec_share',['green_absent','curry_minutes_share','network_player_count','home_game','klay_present','game_number'],'curry_composition_calendar_sensitivity');fits.extend(f)
    for y in ['curry_outgoing_share','curry_shot_share','team_turnovers']:
        f,_=warriors_role_hac(curry,y,['green_absent','curry_minutes_share','network_player_count','home_game','klay_present','game_number'],'curry_context_'+y);fits.extend(f)
    f,_=warriors_role_hac(data,'eigenvector_hhi',['raw_minutes_hhi'],'rotation_both_raw_hhi_sensitivity');fits.extend(f)
    f,_=warriors_role_hac(data,'eigenvector_concentration',['minutes_concentration','network_player_count'],'rotation_player_count_sensitivity');fits.extend(f)
    f,_=warriors_role_hac(data,'eigenvector_concentration',['raw_minutes_hhi'],'rotation_raw_hhi_sensitivity');fits.extend(f)
    complete=data.loc[data.full_observed_network.eq(True)]
    for d,y,x,label,jt in [(complete,'eigenvector_concentration',['minutes_concentration'],'rotation_complete',None),
        (complete.loc[complete.top_eigenvector_player_id.isin([201939,203110,1629673])],'efg_percent',['green_hub','poole_hub'],'identity_complete',['green_hub','poole_hub']),
        (complete.loc[complete.curry_present.eq(True)],'curry_ec_share',['green_absent'],'curry_complete',None)]:
        f,j=warriors_role_hac(d,y,x,label,jt);fits.extend(f)
        if j:joints.append(j)
    result=pd.DataFrame(primary);result['p_holm_three']=np.nan;running=0
    for rank,idx in enumerate(result.p_value.sort_values().index):running=max(running,min(1,result.loc[idx,'p_value']*(3-rank)));result.loc[idx,'p_holm_three']=running
    group=identity.groupby('top_eigenvector_player_name').agg(games=('game_number','size'),efg_mean=('efg_percent','mean'),entropy_mean=('network_conditional_entropy','mean'),ast_to_mean=('team_ast_to_ratio','mean'),turnovers_mean=('team_turnovers','mean'),margin_mean=('point_margin','mean')).reset_index()
    green_summary=curry.groupby('green_absent').agg(games=('game_number','size'),curry_ec_share=('curry_ec_share','mean'),curry_entropy=('curry_recipient_entropy','mean'),curry_reception_share=('curry_reception_share','mean'),curry_shot_share=('curry_shot_share','mean'),curry_minutes=('curry_minutes','mean'),team_efg=('efg_percent','mean'),turnovers=('team_turnovers','mean')).reset_index()
    for d,name in [(data,'game_metrics'),(result,'primary_results'),(pd.DataFrame(fits),'regressions'),(pd.DataFrame(joints),'joint_tests'),(group,'hub_identity_summary'),(green_summary,'green_availability_summary')]:atomic_csv(d,out/(name+'.csv'))
    rng=np.random.default_rng(20261111)
    with PdfPages(out/'warriors_role_checks.pdf') as pdf:
        fig,ax=plt.subplots(figsize=(11,8));r=result.loc[result.question.eq(1)].iloc[0]
        for label,color in [('W','#168348'),('L','#d23b3b')]:
            g=data.loc[data.result.eq(label)];ax.scatter(g.minutes_concentration,g.eigenvector_concentration,color=color,s=35,label='Win' if label=='W' else 'Loss')
        f=linregress(data.minutes_concentration,data.eigenvector_concentration);xx=np.linspace(data.minutes_concentration.min(),data.minutes_concentration.max(),100);ax.plot(xx,f.intercept+f.slope*xx,color='#333333')
        ax.set(xlabel='Playing-time concentration (normalized minutes HHI)',ylabel='Eigenvector passing concentration',title=f"Warriors 2021–22: rotation balance and passing balance\nn=81; r={r.r:.3f}; R²={r.r_squared:.3f}; HAC7 p={r.p_value:.3g}");ax.legend();ax.grid(alpha=.15);fig.tight_layout();pdf.savefig(fig);fig.savefig(out/'rotation_balance.png',dpi=150);fig.savefig(out/'rotation_balance.pdf');plt.close(fig)
        fig,ax=plt.subplots(figsize=(11,8));names=['Stephen Curry','Draymond Green','Jordan Poole']
        for index,name in enumerate(names):
            g=identity.loc[identity.top_eigenvector_player_name.eq(name)];x=index+rng.uniform(-.15,.15,len(g))
            for label,color in [('W','#168348'),('L','#d23b3b')]:
                take=g.result.eq(label).to_numpy();ax.scatter(x[take],g.loc[take,'efg_percent'],color=color,alpha=.8,s=30)
            ax.plot([index-.2,index+.2],[g.efg_percent.mean()]*2,color='black',lw=2)
        ax.set(xticks=range(3),xticklabels=['Curry (36)','Green (24)','Poole (20)'],ylabel='Team eFG%',title=f"Shooting efficiency by game’s passing hub\nHAC7 joint p={result.loc[result.question.eq(2),'p_value'].iloc[0]:.3g}; horizontal bars are means")
        ax.grid(axis='y',alpha=.15);fig.tight_layout();pdf.savefig(fig);fig.savefig(out/'hub_identity.png',dpi=150);fig.savefig(out/'hub_identity.pdf');plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(14,7))
        for ax,y,ylabel in [(axes[0],'curry_ec_share','Curry eigenvector centrality share'),(axes[1],'curry_ec_green_removed_both','Curry centrality share, Green removed from both networks')]:
            for status in [0,1]:
                g=curry.loc[curry.green_absent.eq(status)];x=status+rng.uniform(-.12,.12,len(g))
                for label,color in [('W','#168348'),('L','#d23b3b')]:
                    take=g.result.eq(label).to_numpy();ax.scatter(x[take],g.loc[take,y],color=color,alpha=.8,s=30)
                ax.plot([status-.2,status+.2],[g[y].mean()]*2,color='black',lw=2)
            ax.set(xticks=[0,1],xticklabels=['Green present (36)','Green absent (28)'],ylabel=ylabel);ax.grid(axis='y',alpha=.15)
        fig.suptitle('Curry’s passing role when Green is absent · Curry plays in all 64 games');fig.tight_layout();pdf.savefig(fig);fig.savefig(out/'curry_without_green.png',dpi=150);fig.savefig(out/'curry_without_green.pdf');plt.close(fig)
    notes=['# Warriors 2021–22: three role questions',result.to_string(index=False),group.to_string(index=False),green_summary.to_string(index=False),
        'Saved 82-game season, 81 valid passing networks; game 38 excluded, not imputed as a zero network. Seventy-six marked fully observed; complete-network sensitivity saved. All positive-minute players included; no 10-minute cutoff. Valid-game full-matrix data only; missing tracking weights follow prior project convention but partial games are flagged. Eigenvector uses weighted A+A.T, L1 shares. Minutes concentration uses every official positive-minute participant, independent of passing connectivity.',
        'Primary 1: same-game minutes concentration vs EC concentration. Primary 2: joint difference in team eFG% across Curry/Green/Poole hub identity; 80 games, one Chiozza-led game excluded rather than folded into a group. Reference Curry; hub identity is observed retrospectively, not assigned or causal. Primary 3: Green absence coefficient for Curry EC share in 64 Curry appearances (36 Green present,28 absent). Green absence means no positive-minute appearance, not known injury cause. Presence of Green changes the normalization population; remove Green from BOTH networks sensitivity estimates beyond that mechanical comparison. Other player availability, opponent and seasonal changes can still confound.',
        'All models OLS with HAC7 t inference using actual season game-number gaps; excluded games are not bridged. Hub identity uses HAC7 joint F approximation with df 2,n−3. Holm-three on three primary questions, not all previous project exploration. Other outcomes, composition controls, calendar trend and full-observed sensitivities are exploratory unadjusted. Curry minutes share, network player count, home, Klay appearance and linear season game number are included only in named sensitivity. No new NBA downloads. Green/red = wins/losses. Do not interpret same-game hub identity as an intervention or star quality. Rotation concentration reflects minutes opportunity, so a strong EC correlation can arise without tactical changes.']
    (out/'README.md').write_text('\n\n'.join(notes));print(result.to_string(index=False),flush=True);print(group.to_string(index=False),flush=True);print(green_summary.to_string(index=False),flush=True);print('Saved Warriors role checks: '+str(out),flush=True);return 0



def run_warriors_entropy_ideas(argv):
    """Ten game-level entropy ideas, with eight inferential tests and descriptive decompositions."""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from scipy.stats import linregress
    parser=argparse.ArgumentParser(description='Ten Warriors 2021-22 entropy, stability and redistribution ideas')
    parser.add_argument('--warriors-entropy-ideas',action='store_true');parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT_DIR)
    args=parser.parse_args(argv);root=args.output_dir.expanduser()/'GSW_2021-22_82_games';out=root/'entropy_ideas';out.mkdir(parents=True,exist_ok=True)
    base=pd.read_csv(root/'role_checks/game_metrics.csv',dtype={'game_id':str});seasonids=set();networks={};rosters={};playerrows=[];rows=[]
    def entropy(v):
        v=np.asarray(v,float);q=v[v>0]/v.sum();return float(-np.sum(q*np.log(q))) if v.sum()>0 else np.nan
    def components(a):
        a=np.asarray(a,float);counts=a.sum(1);w=counts/counts.sum();h=np.array([entropy(v) if v.sum()>0 else 0 for v in a]);return w,h
    for _,r in base.iterrows():
        g=int(r.game_number);f=pd.read_csv(root/'eigenvector_team_performance'/f'game_{g:02d}'/'passing_matrix.csv',index_col=0);ids=f.index.astype(int).tolist();a=f.to_numpy(float);active=(a.sum(0)+a.sum(1))>0;ids=np.array(ids)[active].tolist();a=a[np.ix_(active,active)]
        pg=pd.read_csv(root/f'game_{g:02d}'/'game_players.csv').set_index('PLAYER_ID');networks[g]=(ids,a);rosters[g]=pg;seasonids.update(ids);N=len(ids);w,h=components(a);raw=float(w@h);norm=raw/np.log(N-1)
        if not np.isclose(norm,r.network_conditional_entropy):raise ValueError('Conditional entropy mismatch')
        shots=pg.FGA.reindex(ids,fill_value=0).to_numpy(float);shotraw=entropy(shots);normh=h/np.log(N-1);spread=float(np.sqrt(np.sum(w*(normh-norm)**2)))
        row=r.to_dict();row.update(conditional_entropy=norm,raw_conditional_entropy=raw,shooting_entropy=shotraw/np.log(N),raw_shooting_entropy=shotraw,
            top_pair_share=max((a[i,j]+a[j,i]) for i in range(N) for j in range(i+1,N))/a.sum(),recipient_entropy_spread=spread,
            passes=a.sum(),active_players=N)
        rows.append(row)
        for i,pid in enumerate(ids):
            playerrows.append(dict(game_number=g,game_date=r.game_date,player_id=pid,name=pg.loc[pid,'PLAYER_NAME'],outgoing_passes=a[i].sum(),outgoing_share=w[i],recipient_entropy=normh[i],
                raw_recipient_entropy=h[i],entropy_contribution_nats=w[i]*h[i],minutes=pg.loc[pid,'MIN'],turnovers=pg.loc[pid,'TOV'],fga=pg.loc[pid,'FGA'],
                full_observed_network=r.full_observed_network,home_game=r.home_game,game_result=r.result))
    data=pd.DataFrame(rows).sort_values('game_number');players=pd.DataFrame(playerrows);allids=sorted(seasonids);pos={p:i for i,p in enumerate(allids)}
    def align(ids,a):
        output=np.zeros((len(allids),len(allids)));ix=[pos[p] for p in ids];output[np.ix_(ix,ix)]=a;return output
    pairs=[]
    for _,r in data.iterrows():
        g=int(r.game_number)
        if g-1 not in networks:continue
        beforeids,before=networks[g-1];ids,a=networks[g];prior=data.loc[data.game_number.eq(g-1)].iloc[0];B=align(beforeids,before);A=align(ids,a)
        common=sorted(set(ids)&set(beforeids));ib=[beforeids.index(p) for p in common];ia=[ids.index(p) for p in common];bb=before[np.ix_(ib,ib)];aa=a[np.ix_(ia,ia)]
        common_js=receiving_share_js_divergence(bb.ravel(),aa.ravel()) if bb.sum()>0 and aa.sum()>0 else np.nan
        pairs.append(dict(game_number=g,previous_game_number=g-1,game_date=r.game_date,opponent=r.opponent,result=r.result,previous_loss=int(prior.result=='L'),
            edge_js_divergence=receiving_share_js_divergence(B.ravel(),A.ravel()),common_player_edge_js=common_js,raw_entropy_change=r.raw_conditional_entropy-prior.raw_conditional_entropy,
            absolute_raw_entropy_change=abs(r.raw_conditional_entropy-prior.raw_conditional_entropy),normalized_entropy_change=r.conditional_entropy-prior.conditional_entropy,
            full_pair=bool(r.full_observed_network and prior.full_observed_network),player_set_change=1-len(common)/len(set(ids)|set(beforeids))))
    pairdata=pd.DataFrame(pairs);data['meeting_number']=data.groupby('opponent').cumcount()+1;data['repeat_meeting']=data.meeting_number.gt(1).astype(int)
    contributions=players.groupby(['player_id','name']).agg(games=('game_number','size'),total_entropy_contribution=('entropy_contribution_nats','sum'),mean_recipient_entropy=('recipient_entropy','mean'),mean_raw_recipient_entropy=('raw_recipient_entropy','mean'),passes=('outgoing_passes','sum'),mean_minutes=('minutes','mean')).reset_index()
    contributions['share_of_game_averaged_entropy']=contributions.total_entropy_contribution/contributions.total_entropy_contribution.sum()
    contributions=contributions.sort_values('share_of_game_averaged_entropy',ascending=False)
    for pid,g in players.groupby('player_id'):
        i=contributions.index[contributions.player_id.eq(pid)][0];contributions.loc[i,'pass_weighted_recipient_entropy']=np.average(g.recipient_entropy,weights=g.outgoing_passes) if g.outgoing_passes.sum()>0 else np.nan
    fits=[];primary=[];plot_specs=[]
    def test(k,label,d,x,y,controls=None):
        predictors=[x,*(controls or [])];f,_=warriors_role_hac(d,y,predictors,label);fits.extend(dict(v,idea=k) for v in f);row=next(v for v in f if v['term']==x)
        simple=linregress(d[x],d[y]) if not controls else None
        primary.append(dict(idea=k,test=label,n=row['n'],effect=row['coefficient'],r=simple.rvalue if simple else np.nan,r_squared=simple.rvalue**2 if simple else np.nan,p_value=row['p_value_hac7']))
        return row
    test(1,'Passing diversity vs shooting diversity',data,'conditional_entropy','shooting_entropy');plot_specs.append((1,data,'conditional_entropy','shooting_entropy','Passing diversity and shooting diversity'))
    test(2,'Passing entropy vs strongest partnership share',data,'conditional_entropy','top_pair_share');plot_specs.append((2,data,'conditional_entropy','top_pair_share','Passing entropy and pair dependence'))
    test(5,'Next-game entropy change after losses vs wins',pairdata,'previous_loss','raw_entropy_change');plot_specs.append((5,pairdata,'previous_loss','raw_entropy_change','Does entropy change after a loss?'))
    test(6,'Passing-pattern divergence vs absolute entropy change',pairdata,'edge_js_divergence','absolute_raw_entropy_change');plot_specs.append((6,pairdata,'edge_js_divergence','absolute_raw_entropy_change','Pattern changes versus unpredictability changes'))
    opponent_fe=pd.get_dummies(data.opponent,prefix='opponent',drop_first=True,dtype=float);opdata=pd.concat([data.reset_index(drop=True),opponent_fe.reset_index(drop=True)],axis=1)
    test(7,'Repeat-opponent meeting vs passing entropy (opponent FE)',opdata,'repeat_meeting','conditional_entropy',opponent_fe.columns.tolist());plot_specs.append((7,data,'meeting_number','conditional_entropy','Entropy across repeated opponent meetings (plot unadjusted)'))
    for pid,name in [(201939,'Curry'),(203110,'Green')]:
        p=players.loc[players.player_id.eq(pid)&players.outgoing_passes.gt(0)].copy();p['log_passes']=np.log(p.outgoing_passes)
        test(8,name+' recipient entropy vs own turnovers (passing exposure adjusted)',p,'recipient_entropy','turnovers',['log_passes']);plot_specs.append(('8_'+name,p,'recipient_entropy','turnovers',name+': passing diversity and turnovers (plot unadjusted)'))
    test(9,'Eigenvector concentration vs spread in individual entropy',data,'eigenvector_concentration','recipient_entropy_spread');plot_specs.append((9,data,'eigenvector_concentration','recipient_entropy_spread','Passing centralization and unequal passing options'))
    result=pd.DataFrame(primary);result['p_holm_eight']=np.nan;run=0
    for rank,idx in enumerate(result.p_value.sort_values().index):run=max(run,min(1,result.loc[idx,'p_value']*(8-rank)));result.loc[idx,'p_holm_eight']=run
    sensitivities=[]
    for k,d,x,y,label in [(1,data,'raw_conditional_entropy','raw_shooting_entropy','raw_entropy'),(6,pairdata,'common_player_edge_js','absolute_raw_entropy_change','returning_players'),
        (1,data.loc[data.full_observed_network.eq(True)],'conditional_entropy','shooting_entropy','complete_networks'),(2,data.loc[data.full_observed_network.eq(True)],'conditional_entropy','top_pair_share','complete_networks'),
        (5,pairdata.loc[pairdata.full_pair.eq(True)],'previous_loss','raw_entropy_change','complete_pairs'),(6,pairdata.loc[pairdata.full_pair.eq(True)],'edge_js_divergence','absolute_raw_entropy_change','complete_pairs'),
        (9,data.loc[data.full_observed_network.eq(True)],'eigenvector_concentration','recipient_entropy_spread','complete_networks')]:
        f,_=warriors_role_hac(d,y,[x],label);sensitivities.extend(dict(v,idea=k) for v in f)
    f,_=warriors_role_hac(data,'shooting_entropy',['conditional_entropy','minutes_concentration','active_players'],'minutes_player_count');sensitivities.extend(dict(v,idea=1) for v in f)
    # Idea 10: exact midpoint decomposition on a shared nonhub player panel.
    decompositions=[];decomp_players=[];decomp_inputs=[]
    for hub,name,flag in [(201939,'Curry','curry_present'),(203110,'Green','green_present')]:
        presentgames=set(data.loc[data[flag].eq(True),'game_number']);absentgames=set(data.loc[data[flag].eq(False),'game_number'])
        P=sum((align(*networks[int(g)]) for g in presentgames),np.zeros((len(allids),len(allids))));A=sum((align(*networks[int(g)]) for g in absentgames),np.zeros_like(P))
        take=[i for i,pid in enumerate(allids) if pid!=hub]
        while True:
            p=P[np.ix_(take,take)];a=A[np.ix_(take,take)];keep=(p.sum(1)>0)&(a.sum(1)>0)
            if keep.all():break
            take=[j for j,ok in zip(take,keep) if ok]
        decomp_inputs.append((name,p.copy(),a.copy()))
        wp,hp=components(p);wa,ha=components(a);diversity=.5*(wp+wa)*(ha-hp);weights=.5*(hp+ha)*(wa-wp);delta=wa@ha-wp@hp
        if not np.isclose(diversity.sum()+weights.sum(),delta):raise ValueError('Entropy decomposition does not reconcile')
        decompositions.append(dict(absent_player=name,present_games=len(presentgames),absent_games=len(absentgames),matched_nonhub_players=len(take),present_raw_entropy=wp@hp,absent_raw_entropy=wa@ha,
            raw_entropy_change=delta,within_player_diversity_component=diversity.sum(),passing_weight_component=weights.sum(),present_passes=p.sum(),absent_passes=a.sum()))
        names=players.drop_duplicates('player_id').set_index('player_id').name.to_dict()
        for j,index in enumerate(take):decomp_players.append(dict(absent_player=name,player_id=allids[index],name=names[allids[index]],present_pass_share=wp[j],absent_pass_share=wa[j],present_entropy=hp[j],absent_entropy=ha[j],diversity_component=diversity[j],weight_component=weights[j]))
    decomposition_sensitivity=[];rng=np.random.default_rng(20261111)
    def entropy_draws(matrices):
        counts=matrices.sum(2);total=counts.sum(1);w=counts/total[:,None]
        q=np.divide(matrices,counts[:,:,None],out=np.zeros_like(matrices,dtype=float),where=counts[:,:,None]>0)
        logs=np.zeros_like(q);np.log(q,out=logs,where=q>0);h=-(q*logs).sum(2)
        return w,h
    for name,p,a in decomp_inputs:
        sample=int(min(p.sum(),a.sum()));draws=[]
        for mat in [p,a]:
            if mat.sum()==sample:draws.append(np.broadcast_to(mat,(1000,*mat.shape)))
            else:draws.append(rng.multivariate_hypergeometric(mat.astype(np.int64).ravel(),sample,size=1000).reshape(1000,*mat.shape))
        wp,hp=entropy_draws(draws[0]);wa,ha=entropy_draws(draws[1]);v=(.5*(wp+wa)*(ha-hp)).sum(1);w=(.5*(hp+ha)*(wa-wp)).sum(1)
        decomposition_sensitivity.append(dict(absent_player=name,variant='equal_pass_count_1000_draw_mean',matched_pass_count=sample,raw_entropy_change=float((v+w).mean()),diversity_component=float(v.mean()),passing_weight_component=float(w.mean())))
    for hub,name,flag in [(201939,'Curry','curry_present'),(203110,'Green','green_present')]:
        complete=data.loc[data.full_observed_network.eq(True)];present=set(complete.loc[complete[flag].eq(True),'game_number']);absent=set(complete.loc[complete[flag].eq(False),'game_number'])
        P=sum((align(*networks[int(g)]) for g in present),np.zeros((len(allids),len(allids))));A=sum((align(*networks[int(g)]) for g in absent),np.zeros_like(P));take=[i for i,pid in enumerate(allids) if pid!=hub]
        while True:
            p=P[np.ix_(take,take)];a=A[np.ix_(take,take)];keep=(p.sum(1)>0)&(a.sum(1)>0)
            if keep.all():break
            take=[j for j,ok in zip(take,keep) if ok]
        wp,hp=components(p);wa,ha=components(a);v=float(np.sum(.5*(wp+wa)*(ha-hp)));w=float(np.sum(.5*(hp+ha)*(wa-wp)))
        decomposition_sensitivity.append(dict(absent_player=name,variant='complete_networks_only',present_games=len(present),absent_games=len(absent),raw_entropy_change=v+w,diversity_component=v,passing_weight_component=w))
    atomic_csv(pd.DataFrame(decomposition_sensitivity),out/'absence_decomposition_sensitivities.csv')
    atomic_csv(result,out/'primary_results.csv');atomic_csv(pd.DataFrame(fits),out/'regressions.csv');atomic_csv(pd.DataFrame(sensitivities),out/'sensitivities.csv')
    for d,name in [(data,'game_metrics'),(players,'player_game_entropies'),(pairdata,'consecutive_game_changes'),(contributions,'player_entropy_roles'),(pd.DataFrame(decompositions),'absence_entropy_decomposition'),(pd.DataFrame(decomp_players),'absence_player_components')]:atomic_csv(d,out/(name+'.csv'))
    # Shared summaries for ten questions, including descriptive analyses 3,4,10.
    with PdfPages(out/'warriors_entropy_ideas.pdf') as pdf:
        for k,d,x,y,title in plot_specs:
            fig,ax=plt.subplots(figsize=(11,8));outcome='result' if 'result' in d else 'game_result'
            for label,col in [('W','#168348'),('L','#d23b3b')]:
                g=d.loc[d[outcome].eq(label)];ax.scatter(g[x],g[y],color=col,s=30,alpha=.8,label='Win' if label=='W' else 'Loss')
            raw=linregress(d[x],d[y]);xx=np.linspace(d[x].min(),d[x].max(),100);ax.plot(xx,raw.intercept+raw.slope*xx,color='#333333')
            key=8 if str(k).startswith('8_') else k;pr=result.loc[result.idea.eq(key)]
            if key==8:pr=pr.loc[pr.test.str.startswith(str(k).split('_')[1])]
            pv=pr.iloc[0].p_holm_eight
            ax.set(xlabel=x.replace('_',' '),ylabel=y.replace('_',' '),title=f'{k}. {title}\nPrimary Holm-eight p={pv:.3g}; n={len(d)}');ax.legend();ax.grid(alpha=.15)
            fig.text(.5,.015,'Warriors 2021–22 · green wins / red losses · HAC7 inference with actual game gaps\nPlots 7 and 8 show unadjusted points; primary tests adjust opponent identity or passing exposure, respectively.',ha='center',fontsize=8)
            fig.tight_layout(rect=(0,.05,1,1));pdf.savefig(fig);fig.savefig(out/f'idea_{k}.png',dpi=130);fig.savefig(out/f'idea_{k}.pdf');plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(15,7));top=contributions.head(12).sort_values('share_of_game_averaged_entropy')
        axes[0].barh(top.name,100*top.share_of_game_averaged_entropy,color='#2473b8');axes[0].set(xlabel='Share of total game-level conditional entropy contribution (%)',title='3. Who contributes to team passing entropy?')
        frequent=contributions.loc[contributions.games.ge(15)].sort_values('pass_weighted_recipient_entropy');axes[1].barh(frequent.name,frequent.pass_weighted_recipient_entropy,color='#2473b8');axes[1].set(xlabel='Pass-weighted normalized recipient entropy',title='4. Individual passing-role diversity\nDisplay: players with >=15 observed network games')
        fig.tight_layout();pdf.savefig(fig);fig.savefig(out/'ideas_3_4_roles.pdf');fig.savefig(out/'ideas_3_4_roles.png',dpi=130);plt.close(fig)
        fig,ax=plt.subplots(figsize=(11,8));dec=pd.DataFrame(decompositions);x=np.arange(2);ax.bar(x-.15,dec.within_player_diversity_component,width=.3,label='Changes in individual recipient diversity');ax.bar(x+.15,dec.passing_weight_component,width=.3,label='Changes in distribution of passing volume');ax.axhline(0,color='gray',lw=.8);ax.set(xticks=x,xticklabels=['Curry absent','Green absent'],ylabel='Contribution to conditional entropy change (nats)',title='10. Why does nonhub passing entropy change?\nSame nonhub player panel in both conditions');ax.legend();fig.tight_layout();pdf.savefig(fig);fig.savefig(out/'idea_10_decomposition.pdf');fig.savefig(out/'idea_10_decomposition.png',dpi=130);plt.close(fig)
    notes=['# Warriors: ten entropy ideas',result.to_string(index=False),pd.DataFrame(decompositions).to_string(index=False),
        '81 valid game networks (game 38 excluded), 76 fully observed. Consecutive games only; missing game is never bridged. Same Warriors 2021–22 saved matrices and official player statistics; no NBA downloads. Primary style is passing-based; no minutes cutoff. First game has no prior-game comparison.',
        'Eight inferential tests cover ideas 1,2,5,6,7,8(Curry),8(Green),9 with Holm-eight correction; ideas 3,4,10 descriptive. HAC7 with actual game-number gaps for all inferential models. Correction applies only to this suite, not all prior topic exploration. Sensitivities unadjusted exploratory. Strong structural correlations reuse the same passes and need not indicate better offense.',
        'Idea 1: conditional passing entropy H(receiver|passer)/log(N−1) vs entropy of FGA shares/log(N), both on game active network IDs. Raw entropy sensitivity avoids shared normalization. Idea 2: strongest unordered pair share of recorded passes. Idea 3: each player contribution = outgoing share x own recipient entropy in nats; contribution shares summed across games, with games weighted equally, not all passes pooled. It is not importance, skill or efficiency. Idea 4: individual recipient entropy means and pass-weighted means, all players saved; figure >=15-game threshold is display only, not a minutes cutoff. Small pass counts downward-bias empirical entropy; player rankings remain descriptive.',
        'Idea 5: raw H change from previous to current game vs previous loss indicator, not a causal coaching response or a win prediction. Idea 6: base-2 JS of normalized joint passer/receiver edge distributions on union IDs vs absolute raw H change. Common-player edge sensitivity saved, but the entropy outcome remains whole-network. Entropy scalar can hide changes in player identities/connections. Idea 7: repeat meeting vs first meeting with opponent fixed effects; roster/availability/date may confound. Figure is unadjusted for opponent effects. Idea 8: own turnover count vs own recipient entropy adjusted log passes made, separately Curry and Green; turnovers include non-pass errors, not passing accuracy. Figure raw, inference exposure adjusted. Idea 9: pass-weighted SD of normalized individual recipient entropies vs eigenvector concentration; bounded-entropy mean/variance dependence can induce relationships.',
        'Idea 10: aggregate present/absent networks over valid games only; remove the absent star from BOTH; iteratively retain common positive-outgoing nonhub panel. H=sum(w*h). Exact symmetric midpoint decomposition delta H = sum(mean(w)*delta h) + sum(mean(h)*delta w). Conditional on selected panel; no causal separation of tactics from availability/minutes, and unequal pass sample counts can bias plug-in entropy. Equal-count sensitivity uses 1000 seeded hypergeometric aggregate-pass thinnings; this is not a game bootstrap or a confidence interval. A complete-network decomposition is also saved. Decomposition is accounting, not regression or statistical proof. Curry/Green absences overlap other absences and season stages. All-network and complete sensitivities distinguish coverage where inferential tests run.']
    (out/'README.md').write_text('\n\n'.join(notes));print(result.to_string(index=False),flush=True);print('Consecutive pairs:',len(pairdata),flush=True);print(contributions.head(6).to_string(index=False),flush=True);print(pd.DataFrame(decompositions).to_string(index=False),flush=True);print('Saved Warriors entropy ideas: '+str(out),flush=True);return 0



def warriors_pure_passing_markov(a,ids):
    """Pure observed directed-pass chain, no restarts or invented edges."""
    import networkx as nx
    a=np.asarray(a,float);ids=np.asarray(ids,int);active=(a.sum(0)+a.sum(1))>0;a=a[np.ix_(active,active)];ids=ids[active]
    graph=nx.from_numpy_array(a,create_using=nx.DiGraph)
    audit=dict(active_players=len(ids),zero_outgoing_nodes=int((a.sum(1)==0).sum()),strongly_connected=nx.is_strongly_connected(graph),aperiodic=nx.is_aperiodic(graph) if nx.is_strongly_connected(graph) else False)
    if not audit['strongly_connected']:return audit,None,None,None,None
    P=a/a.sum(1)[:,None];n=len(ids);B=P.T-np.eye(n);B[-1]=1;rhs=np.zeros(n);rhs[-1]=1;pi=np.linalg.solve(B,rhs)
    if (pi<-1e-10).any() or not np.allclose(pi@P,pi,atol=1e-10) or not np.isclose(pi.sum(),1):raise ValueError('Invalid stationary solution')
    Z=np.linalg.inv(np.eye(n)-P+np.outer(np.ones(n),pi));M=(np.diag(Z)[None,:]-Z)/pi[None,:];np.fill_diagonal(M,0)
    if np.min(M)<-1e-7:raise ValueError('Negative first-passage time')
    audit.update(stationary_concentration=float((pi@pi-1/n)/(1-1/n)),modeled_two_step_return=float(pi@np.diag(P@P)),
        kemeny_steps=float(np.trace(Z)-1),stationarity_max_error=float(np.max(np.abs(pi@P-pi))))
    if 201939 in ids:
        j=list(ids).index(201939);take=[i for i in range(n) if i!=j];Q=P[np.ix_(take,take)];h=np.linalg.solve(np.eye(n-1)-Q,np.ones(n-1))
        if not np.allclose(h,M[take,j],atol=1e-7):raise ValueError('MFPT methods disagree')
        audit.update(curry_stationary_share=float(pi[j]),expected_passes_to_curry_uniform=float(h.mean()),expected_passes_to_curry_stationary_start=float(np.average(h,weights=pi[take])))
    return audit,ids,P,pi,M


def run_warriors_markov_checks(argv):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from scipy.stats import linregress
    parser=argparse.ArgumentParser(description='Warriors pure-pass Markov model: routing, access and efficiency')
    parser.add_argument('--warriors-markov-checks',action='store_true');parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT_DIR)
    args=parser.parse_args(argv);root=args.output_dir.expanduser()/'GSW_2021-22_82_games';out=root/'markov_checks';out.mkdir(parents=True,exist_ok=True)
    base=pd.read_csv(root/'role_checks/game_metrics.csv',dtype={'game_id':str});rows=[];nodes=[]
    for _,r in base.iterrows():
        g=int(r.game_number);f=pd.read_csv(root/'eigenvector_team_performance'/f'game_{g:02d}'/'passing_matrix.csv',index_col=0);ids=f.index.astype(int).tolist();audit,active,P,pi,M=warriors_pure_passing_markov(f.to_numpy(),ids)
        row=dict(r.to_dict(),**audit);row['primary_eligible']=bool(audit['strongly_connected'] and r.full_observed_network)
        row['exclusion_reason']='' if row['primary_eligible'] else ('not strongly connected: no single irreducible whole-network chain' if not audit['strongly_connected'] else 'incomplete recorded network')
        rows.append(row)
        if P is not None:
            folder=out/'chains'/f'game_{g:02d}';folder.mkdir(parents=True,exist_ok=True);atomic_csv(pd.DataFrame(P,index=active,columns=active),folder/'transition_matrix.csv',index=True);atomic_csv(pd.DataFrame(M,index=active,columns=active),folder/'first_passage_steps.csv',index=True)
            names=pd.read_csv(root/f'game_{g:02d}'/'game_players.csv').set_index('PLAYER_ID').PLAYER_NAME.to_dict()
            for pid,value in zip(active,pi):nodes.append(dict(game_number=g,player_id=int(pid),name=names[int(pid)],stationary_share=value))
    allgames=pd.DataFrame(rows);data=allgames.loc[allgames.primary_eligible.eq(True)].copy();curry=data.dropna(subset=['expected_passes_to_curry_uniform']).copy()
    specs=[(1,data,'stationary_concentration','efg_percent','Modeled circulation concentration and shooting efficiency','Stationary-distribution concentration (normalized HHI)','Team eFG%'),
        (2,curry,'expected_passes_to_curry_uniform','efg_percent','Modeled access to Curry and shooting efficiency','Expected modeled passes to reach Curry (uniform other-player start)','Team eFG%'),
        (3,data,'modeled_two_step_return','team_ast_to_ratio','Modeled two-step returns and ball security','Stationary-weighted two-step return probability','Team assists / turnovers')]
    primary=[];fits=[];sensitivity=[]
    for k,d,x,y,title,xlabel,ylabel in specs:
        f,_=warriors_role_hac(d,y,[x],title);fits.extend(dict(v,idea=k) for v in f);r=next(v for v in f if v['term']==x);simple=linregress(d[x],d[y]);primary.append(dict(idea=k,test=title,n=len(d),r=simple.rvalue,r_squared=simple.rvalue**2,slope=r['coefficient'],p_value=r['p_value_hac7']))
        controls=['minutes_concentration','network_player_count','home_game','game_number']
        f,_=warriors_role_hac(d,y,[x,*controls],title+'_composition_calendar');sensitivity.extend(dict(v,idea=k) for v in f)
        expanded=allgames.loc[allgames.strongly_connected.eq(True)].dropna(subset=[x,y])
        f,_=warriors_role_hac(expanded,y,[x],title+'_all_irreducible');sensitivity.extend(dict(v,idea=k) for v in f)
    f,_=warriors_role_hac(curry,'efg_percent',['expected_passes_to_curry_stationary_start'],'alternative_start_distribution');sensitivity.extend(dict(v,idea=2) for v in f)
    result=pd.DataFrame(primary);result['p_holm_three']=np.nan;running=0
    for rank,idx in enumerate(result.p_value.sort_values().index):running=max(running,min(1,result.loc[idx,'p_value']*(3-rank)));result.loc[idx,'p_holm_three']=running
    atomic_csv(allgames,out/'game_metrics_and_eligibility.csv');atomic_csv(pd.DataFrame(nodes),out/'player_stationary_shares.csv');atomic_csv(result,out/'primary_results.csv');atomic_csv(pd.DataFrame(fits),out/'regressions.csv');atomic_csv(pd.DataFrame(sensitivity),out/'sensitivities.csv')
    with PdfPages(out/'warriors_markov_checks.pdf') as pdf:
        for k,d,x,y,title,xlabel,ylabel in specs:
            fig,ax=plt.subplots(figsize=(11,8));r=result.loc[result.idea.eq(k)].iloc[0]
            for label,col in [('W','#168348'),('L','#d23b3b')]:
                g=d.loc[d.result.eq(label)];ax.scatter(g[x],g[y],color=col,s=35,alpha=.85,label='Win' if label=='W' else 'Loss')
            fit=linregress(d[x],d[y]);xx=np.linspace(d[x].min(),d[x].max(),100);ax.plot(xx,fit.intercept+fit.slope*xx,color='#333333')
            ax.set(xlabel=xlabel,ylabel=ylabel,title=f'{k}. {title}\nn={r.n}; r={r.r:.3f}; R²={r.r_squared:.3f}; Holm-three p={r.p_holm_three:.3g}');ax.legend();ax.grid(alpha=.15)
            fig.text(.5,.015,'Warriors 2021–22 · aggregate directed-pass model, not observed possession sequences\nPrimary: fully observed, strongly connected networks. No invented restarts, shots or turnover transitions.',ha='center',fontsize=9)
            fig.tight_layout(rect=(0,.055,1,1));pdf.savefig(fig);fig.savefig(out/f'question_{k}.png',dpi=150);fig.savefig(out/f'question_{k}.pdf');plt.close(fig)
    definitions=['# Warriors Markov passing model',result.to_string(index=False),
        'The research focus shifts from entropy to modeled ball routing and hub dependence. Player states, directed transitions P[i,j]=passes(i,j)/sum_j passes(i,j). Each step is a modeled pass; no duration, shot, turnover or possession-end states. Aggregate pair counts do not demonstrate the Markov property or observed multi-pass sequences. Conditional transitions are assumed time-homogeneous and memoryless within each game, even though players substitute and lineups differ. Player-level connectivity is not evidence all players shared the court.',
        'Primary: 66 fully observed strongly connected networks, with Curry first-passage analysis in 54. Original 82-game season has 81 valid networks; game 38 already excluded. Fourteen of 81 valid networks not strongly connected; five marked incomplete; overlap means 15 exclusions from primary. Twelve valid networks have at least one zero-outgoing active node. Exclusions audited, not imputed with invented passes; sample selection may affect generalizability. All 67 irreducible networks sensitivity saved. Isolates omitted. No teleportation or arbitrary damping; the stationary distribution is not default PageRank. Stationary exists uniquely for each included irreducible finite chain; aperiodicity recorded. No mixing-time outcome is claimed.',
        'Stationary concentration = normalized HHI of pi, where pi P=pi, sum pi=1. It describes long-run modeled pass routing, not actual possession time, touches, importance or shot creation. Mean first-passage-to-Curry: solve (I−Q)h=1, starting uniformly over other active players. Fundamental-matrix calculation independently cross-checked. Alternative stationary-weighted start saved. These expected passes are model units, not observed or required passes before a Curry shot. Two-step return = sum_i pi_i*(P²)[i,i], not counted give-and-go possessions. Kemeny steps saved descriptively, not one of the selected performance hypotheses.',
        'Three selected primary tests use gap-aware HAC7 and Holm-three correction; this does not cover prior project exploration. Optional minutes/player count/home/game index controls unadjusted exploratory. Same-game team eFG and AST/TO are outcomes; no causality, forecasting or validation against actual possessions established. This is a feasibility study, not proof that Markov is better than entropy. Source mathematical definitions: https://personal.math.ubc.ca/~holmescerfon/teaching/asa22/handout-Lecture2_2022.pdf ; https://mro.massey.ac.nz/items/afe4cfe0-1be0-44d6-a08d-b8e07c892076 . Saved real NBA networks only; no new data downloads.']
    (out/'README.md').write_text('\n\n'.join(definitions));print(result.to_string(index=False),flush=True);print('Primary games:',len(data),'Curry:',len(curry),'all irreducible:',int(allgames.strongly_connected.sum()),flush=True);print('Saved Markov checks: '+str(out),flush=True);return 0



def warriors_probability_matrix(a):
    """Zero storage rows denote undefined distributions, never invented passes."""
    a=np.asarray(a,float);counts=a.sum(1)
    return np.divide(a,counts[:,None],out=np.zeros_like(a),where=counts[:,None]>0),counts>0


def probability_shift_opponent_test(vectors,labels):
    """Exploratory exact circular-shift reference, retaining serial profile order."""
    V=np.sqrt(np.asarray(vectors,float));labels=np.asarray(labels);n=len(labels);groups=pd.unique(labels)
    def score(values):
        overall=values.mean(0);total=float(np.sum((values-overall)**2))
        between=sum(int(np.sum(labels==g))*float(np.sum((values[labels==g].mean(0)-overall)**2)) for g in groups)
        return between/total if total>0 else np.nan
    observed=score(V);null=np.array([score(np.roll(V,k,axis=0)) for k in range(n)])
    return dict(n=n,opponents=len(groups),association_r_squared=observed,adjusted_r_squared=1-(1-observed)*(n-1)/(n-len(groups)),
        circular_shift_p=float(np.mean(null>=observed-1e-12)),null_median_r_squared=float(np.median(null)),shift_reference_size=n)


def run_warriors_probability_ideas(argv):
    import networkx as nx
    from scipy.stats import spearmanr
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    parser=argparse.ArgumentParser(description='Eight directed probability-weighted Warriors passing-network ideas')
    parser.add_argument('--warriors-probability-ideas',action='store_true');parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT_DIR)
    args=parser.parse_args(argv);root=args.output_dir.expanduser()/'GSW_2021-22_82_games';out=root/'probability_network_ideas';out.mkdir(parents=True,exist_ok=True)
    base=pd.read_csv(root/'role_checks/game_metrics.csv',dtype={'game_id':str});rawnets={};rosters={};names={};seasonids=set()
    for _,r in base.iterrows():
        g=int(r.game_number);f=pd.read_csv(root/'eigenvector_team_performance'/f'game_{g:02d}'/'passing_matrix.csv',index_col=0);a=f.to_numpy(float);ids=f.index.astype(int).to_numpy();active=a.sum(0)+a.sum(1)>0;a=a[np.ix_(active,active)];ids=ids[active].tolist()
        pg=pd.read_csv(root/f'game_{g:02d}'/'game_players.csv');pg=pg.loc[pg.MIN.gt(0)].set_index('PLAYER_ID');names.update(pg.PLAYER_NAME.to_dict());seasonids.update(ids);rawnets[g]=(ids,a);rosters[g]=pg
    allids=sorted(seasonids);idx={p:i for i,p in enumerate(allids)};K=len(allids)
    def align(ids,a):
        B=np.zeros((K,K));take=[idx[p] for p in ids];B[np.ix_(take,take)]=a;return B
    nets={g:align(ids,a) for g,(ids,a) in rawnets.items()};meta=base.set_index('game_number');complete=set(base.loc[base.full_observed_network.eq(True),'game_number'].astype(int));Pall={};rowvalid={}
    node_rows=[];game_rows=[];pair_rows=[];first_rows=[];rank_rows=[];rank_sensitivity=[]
    for _,r in base.iterrows():
        g=int(r.game_number);ids,a=rawnets[g];P,valid=warriors_probability_matrix(a);Pglobal,vg=warriors_probability_matrix(nets[g]);Pall[g]=Pglobal;rowvalid[g]=vg;S=int(valid.sum());incoming=P.sum(0);pref=incoming/S;rawrecv=a.sum(0);rawshare=rawrecv/rawrecv.sum()
        votes=np.zeros(len(ids))
        for i in np.flatnonzero(valid):
            top=np.flatnonzero(np.isclose(P[i],P[i].max(),atol=1e-12,rtol=1e-10));votes[top]+=1/len(top)
        rawlead=int(ids[int(np.argmax(rawrecv))]);problead=int(ids[int(np.argmax(pref))]);rawties=int(np.isclose(rawrecv,rawrecv.max()).sum());probties=int(np.isclose(pref,pref.max()).sum())
        G=nx.from_numpy_array(a,create_using=nx.DiGraph);Gp=nx.from_numpy_array(P,create_using=nx.DiGraph)
        pra=np.array(list(nx.pagerank(G,alpha=.85,tol=1e-12,max_iter=2000).values()));prp=np.array(list(nx.pagerank(Gp,alpha=.85,tol=1e-12,max_iter=2000).values()));prdiff=float(np.max(np.abs(pra-prp)))
        if prdiff>1e-9:raise ValueError('PageRank row-rescaling invariance failed')
        strong=nx.is_strongly_connected(G);ec_raw=np.full(len(ids),np.nan);ec_prob=np.full(len(ids),np.nan)
        if strong:
            value,vec=np.linalg.eig(a.T);j=int(np.argmax(value.real));v=np.real(vec[:,j]);ec_raw=np.abs(v)/np.abs(v).sum()
            B=P.T-np.eye(len(ids));B[-1]=1;rhs=np.zeros(len(ids));rhs[-1]=1;ec_prob=np.linalg.solve(B,rhs)
            if not np.allclose(ec_prob@P,ec_prob):raise ValueError('Directed probability EC failed')
        rank_rows.append(dict(game_number=g,full_observed_network=r.full_observed_network,raw_receiver=names[rawlead],probability_receiver=names[problead],leader_changed=rawlead!=problead,raw_leader_ties=rawties,probability_leader_ties=probties,
            incoming_rank_spearman=spearmanr(rawshare,pref).statistic,pagerank_max_difference=prdiff,directed_eigenvector_available=strong,
            directed_eigenvector_rank_spearman=spearmanr(ec_raw,ec_prob).statistic if strong else np.nan,
            directed_eigenvector_leader_changed=bool(np.argmax(ec_raw)!=np.argmax(ec_prob)) if strong else np.nan))
        for threshold in [5,10,20]:
            donors=a.sum(1)>=threshold
            if not donors.any():continue
            votes2=P[donors].sum(0)/donors.sum();raw2=a[donors].sum(0);rank_sensitivity.append(dict(game_number=g,threshold=threshold,donors=int(donors.sum()),probability_receiver=names[ids[int(np.argmax(votes2))]],
                same_preference_leader_as_all_donors=int(np.argmax(votes2))==int(np.argmax(pref)),incoming_rank_spearman=spearmanr(raw2,votes2).statistic,full_observed_network=r.full_observed_network))
        for i,pid in enumerate(ids):
            positive=np.sort(P[i])[::-1];maximum=positive[0] if valid[i] else np.nan;second=positive[1] if valid[i] else np.nan
            node_rows.append(dict(game_number=g,game_date=r.game_date,player_id=pid,name=names[pid],passes_made=a[i].sum(),passes_received=rawrecv[i],incoming_preference_strength=incoming[i],incoming_preference_share=pref[i],first_choice_vote_share=votes[i]/S,
                raw_receiving_share=rawshare[i],defined_outgoing=bool(valid[i]),outgoing_probability_sum=P[i].sum() if valid[i] else np.nan,maximum_probability=maximum,first_second_gap=maximum-second if valid[i] else np.nan,
                raw_pagerank=pra[i],probability_pagerank=prp[i],raw_directed_eigenvector=ec_raw[i],probability_directed_eigenvector=ec_prob[i],full_observed_network=r.full_observed_network))
            if valid[i]:
                top=np.flatnonzero(np.isclose(P[i],maximum));first_rows.append(dict(game_number=g,player_id=pid,name=names[pid],first_choice='; '.join(names[ids[j]] for j in top),first_choice_ties=len(top),maximum_probability=maximum,first_second_gap=maximum-second,passes_made=a[i].sum(),full_observed_network=r.full_observed_network))
            for j in range(i+1,len(ids)):
                if valid[i] and valid[j]:pair_rows.append(dict(game_number=g,player_1=min(pid,ids[j]),player_2=max(pid,ids[j]),mutual_preference=P[i,j]*P[j,i],minimum_source_passes=min(a[i].sum(),a[j].sum()),full_observed_network=r.full_observed_network))
        game_rows.append(dict(game_number=g,game_date=r.game_date,opponent=r.opponent,result=r.result,full_observed_network=r.full_observed_network,active_players=len(ids),defined_sources=S,undefined_sources=len(ids)-S,low_count_sources_under10=int(((a.sum(1)>0)&(a.sum(1)<10)).sum()),
            preference_concentration=float((pref@pref-1/len(ids))/(1-1/len(ids))),raw_receiving_concentration=float((rawshare@rawshare-1/len(ids))/(1-1/len(ids)))))
        folder=out/'networks'/f'game_{g:02d}';folder.mkdir(parents=True,exist_ok=True);export=P.copy();export[~valid,:]=np.nan;atomic_csv(pd.DataFrame(export,index=ids,columns=ids),folder/'passing_probabilities.csv',index=True)
    nodes=pd.DataFrame(node_rows);ranks=pd.DataFrame(rank_rows);pairdata=pd.DataFrame(pair_rows);first=pd.DataFrame(first_rows);games=pd.DataFrame(game_rows);primarynodes=nodes.loc[nodes.full_observed_network.eq(True)];primaryranks=ranks.loc[ranks.full_observed_network.eq(True)]
    preferences=primarynodes.groupby(['player_id','name']).agg(observed_games=('game_number','size'),mean_preference_share=('incoming_preference_share','mean'),mean_raw_receiving_share=('raw_receiving_share','mean'),mean_first_choice_vote_share=('first_choice_vote_share','mean'),season_preference_share_sum=('incoming_preference_share','sum')).reset_index()
    preferences['season_game_averaged_preference_share']=preferences.season_preference_share_sum/len(complete)
    preferences=preferences.sort_values('season_game_averaged_preference_share',ascending=False)
    leadercounts=primaryranks.groupby('probability_receiver').size().sort_values(ascending=False).rename('games_leading_probability_strength').reset_index()
    mutual=pairdata.loc[pairdata.full_observed_network.eq(True)].groupby(['player_1','player_2']).agg(eligible_games=('game_number','size'),mean_mutual_preference=('mutual_preference','mean'),median_mutual_preference=('mutual_preference','median')).reset_index()
    mutual['pair']=[names[a]+' ↔ '+names[b] for a,b in zip(mutual.player_1,mutual.player_2)];mutual=mutual.sort_values('mean_mutual_preference',ascending=False)
    mutual10=pairdata.loc[pairdata.full_observed_network.eq(True)&pairdata.minimum_source_passes.ge(10)].groupby(['player_1','player_2']).agg(eligible_games_min10=('game_number','size'),mean_mutual_preference_min10=('mutual_preference','mean')).reset_index();mutual=mutual.merge(mutual10,on=['player_1','player_2'],how='left')
    # 3: Curry supply changes with Green absent; remove/recondition Green in BOTH conditions.
    green_edges=[]
    for g in sorted(complete):
        pg=rosters[g]
        if 201939 not in pg.index:continue
        a=nets[g].copy();a[:,idx[203110]]=0;a[idx[203110],:]=0;P,v=warriors_probability_matrix(a);green_absent=int(203110 not in pg.index)
        for other in allids:
            if other in [201939,203110] or other not in pg.index:continue
            for direction,source,target in [('Curry to teammate',201939,other),('Teammate to Curry',other,201939)]:
                i,j=idx[source],idx[target]
                if not v[i]:continue
                green_edges.append(dict(game_number=g,direction=direction,partner_id=other,partner=names[other],green_absent=green_absent,probability=P[i,j],source_passes_to_nonGreen=a[i].sum()))
    green_data=pd.DataFrame(green_edges);green_tests=[]
    for (direction,pid),d in green_data.groupby(['direction','partner_id']):
        n0=int(d.green_absent.eq(0).sum());n1=int(d.green_absent.eq(1).sum())
        if min(n0,n1)<5:continue
        fits,_=warriors_role_hac(d,'probability',['green_absent'],'preference_change');r=next(v for v in fits if v['term']=='green_absent')
        green_tests.append(dict(direction=direction,partner_id=pid,partner=names[pid],green_present_games=n0,green_absent_games=n1,probability_with_green=d.loc[d.green_absent.eq(0),'probability'].mean(),probability_without_green=d.loc[d.green_absent.eq(1),'probability'].mean(),change=r['coefficient'],ci95_low=r['ci95_low'],ci95_high=r['ci95_high'],p_value=r['p_value_hac7']))
    green=pd.DataFrame(green_tests);green['p_holm_edges']=np.nan;running=0
    for rank,i in enumerate(green.p_value.sort_values().index):running=max(running,min(1,green.loc[i,'p_value']*(len(green)-rank)));green.loc[i,'p_holm_edges']=running
    # 4: probability preference stability, separating changes in receiver availability.
    stability=[]
    for g in sorted(nets):
        if g-1 not in nets:continue
        A=nets[g];B=nets[g-1];P=Pall[g];Q=Pall[g-1];common_receivers=set(rosters[g].index)&set(rosters[g-1].index)
        for pid in allids:
            i=idx[pid]
            if not rowvalid[g][i] or not rowvalid[g-1][i]:continue
            take=[idx[p] for p in common_receivers if p in idx and p!=pid];ca=A[i,take];cb=B[i,take]
            common_js=receiving_share_js_divergence(ca,cb) if ca.sum()>0 and cb.sum()>0 else np.nan
            stability.append(dict(game_number=g,previous_game_number=g-1,player_id=pid,name=names[pid],preference_js=receiving_share_js_divergence(P[i],Q[i]),common_receiver_js=common_js,
                minimum_source_passes=min(A[i].sum(),B[i].sum()),full_pair=g in complete and g-1 in complete))
    stable=pd.DataFrame(stability);stable_primary=stable.loc[stable.full_pair.eq(True)]
    stability_summary=stable_primary.groupby(['player_id','name']).agg(consecutive_pairs=('game_number','size'),mean_js=('preference_js','mean'),median_js=('preference_js','median'),mean_common_receiver_js=('common_receiver_js','mean')).reset_index()
    ss10=stable_primary.loc[stable_primary.minimum_source_passes.ge(10)].groupby('player_id').agg(pairs_min10=('game_number','size'),mean_js_min10=('preference_js','mean'),mean_common_js_min10=('common_receiver_js','mean')).reset_index();stability_summary=stability_summary.merge(ss10,on='player_id',how='left')
    choice_summary=first.loc[first.full_observed_network.eq(True)].groupby(['player_id','name']).agg(games=('game_number','size'),mean_first_choice_probability=('maximum_probability','mean'),median_first_choice_probability=('maximum_probability','median'),mean_first_second_gap=('first_second_gap','mean'),mean_passes=('passes_made','mean')).reset_index()
    cs10=first.loc[first.full_observed_network.eq(True)&first.passes_made.ge(10)].groupby('player_id').agg(games_min10=('game_number','size'),mean_first_choice_min10=('maximum_probability','mean'),mean_gap_min10=('first_second_gap','mean')).reset_index();choice_summary=choice_summary.merge(cs10,on='player_id',how='left')
    # 6: compare targets after excluding the two compared passers from BOTH rows.
    shared=[];core=[201939,203110,1629673];pair_names={(201939,203110):'Curry–Green',(201939,1629673):'Curry–Poole',(203110,1629673):'Green–Poole'}
    for g in sorted(nets):
        A=nets[g]
        for (p,q),label in pair_names.items():
            if not rowvalid[g][idx[p]] or not rowvalid[g][idx[q]]:continue
            take=[j for j in range(K) if j not in [idx[p],idx[q]]];x=A[idx[p],take];y=A[idx[q],take]
            if x.sum()<=0 or y.sum()<=0:continue
            shared.append(dict(game_number=g,pair=label,shared_target_js=receiving_share_js_divergence(x,y),raw_row_js=receiving_share_js_divergence(Pall[g][idx[p]],Pall[g][idx[q]]),minimum_source_passes=min(A[idx[p]].sum(),A[idx[q]].sum()),complete=g in complete))
    similarities=pd.DataFrame(shared);wide=similarities.loc[similarities.complete.eq(True)].pivot(index='game_number',columns='pair',values='shared_target_js').dropna();shared_tests=[]
    for left,right in [('Curry–Green','Curry–Poole'),('Curry–Green','Green–Poole'),('Curry–Poole','Green–Poole')]:
        d=(wide[left]-wide[right]).rename('difference').reset_index();fits,_=warriors_role_hac(d,'difference',[],'target_similarity');r=fits[0];shared_tests.append(dict(comparison=left+' minus '+right,n=len(d),difference=r['coefficient'],p_value=r['p_value_hac7']))
    shared_results=pd.DataFrame(shared_tests);shared_results['p_holm_three']=np.nan;running=0
    for rank,i in enumerate(shared_results.p_value.sort_values().index):running=max(running,min(1,shared_results.loc[i,'p_value']*(3-rank)));shared_results.loc[i,'p_holm_three']=running
    shared_summary=similarities.loc[similarities.complete.eq(True)].groupby('pair').agg(games=('game_number','size'),mean_shared_target_js=('shared_target_js','mean'),mean_raw_row_js=('raw_row_js','mean')).reset_index()
    shared_summary['matched_trio_games']=len(wide);shared_summary['matched_trio_mean_js']=[wide[p].mean() for p in shared_summary.pair]
    # 8: opponent association; circular shifts preserve order of profiles, not a causal experiment.
    opponent_tests=[];opponent_distances=[]
    for pid in core:
        candidate=[g for g in sorted(complete) if rowvalid[g][idx[pid]]];counts=pd.Series([meta.loc[g,'opponent'] for g in candidate]).value_counts();candidate=[g for g in candidate if counts[meta.loc[g,'opponent']]>=2]
        for variant in ['all_receivers','fixed_core_receivers']:
            targetids=[p for p in allids if p!=pid]
            chosen=candidate.copy()
            if variant=='fixed_core_receivers':
                targetids=[p for p in targetids if np.mean([p in rosters[g].index for g in candidate])>=.8]
                chosen=[g for g in candidate if all(p in rosters[g].index for p in targetids)]
            take=[idx[p] for p in targetids];chosen=[g for g in chosen if nets[g][idx[pid],take].sum()>0]
            cc=pd.Series([meta.loc[g,'opponent'] for g in chosen]).value_counts();chosen=[g for g in chosen if cc[meta.loc[g,'opponent']]>=2]
            if len(chosen)<12 or len(targetids)<2:continue
            X=np.array([nets[g][idx[pid],take]/nets[g][idx[pid],take].sum() for g in chosen]);labels=np.array([meta.loc[g,'opponent'] for g in chosen]);test=probability_shift_opponent_test(X,labels)
            opponent_tests.append(dict(player_id=pid,name=names[pid],variant=variant,targets='; '.join(names[p] for p in targetids),**test))
            same=[];different=[]
            for i in range(len(chosen)):
                for j in range(i+1,len(chosen)):
                    value=receiving_share_js_divergence(X[i],X[j]);(same if labels[i]==labels[j] else different).append(value)
            opponent_distances.append(dict(name=names[pid],variant=variant,within_opponent_mean_js=np.mean(same),between_opponent_mean_js=np.mean(different),within_pairs=len(same),between_pairs=len(different)))
    opponent=pd.DataFrame(opponent_tests);opponent['p_holm_three']=np.nan
    primary_op=opponent.loc[opponent.variant.eq('all_receivers')];running=0
    for rank,i in enumerate(primary_op.circular_shift_p.sort_values().index):running=max(running,min(1,opponent.loc[i,'circular_shift_p']*(len(primary_op)-rank)));opponent.loc[i,'p_holm_three']=running
    for d,name in [(nodes,'player_game_preferences'),(games,'game_quality_metrics'),(preferences,'receiving_preference_summary'),(leadercounts,'receiving_leader_counts'),(pairdata,'mutual_partnership_games'),(mutual,'mutual_partnerships'),(green_data,'green_absence_edge_data'),(green,'green_absence_preference_tests'),(stable,'consecutive_preference_changes'),(stability_summary,'preference_stability_summary'),(choice_summary,'first_choice_summary'),(first,'first_choices_by_game'),(similarities,'shared_target_similarity_games'),(shared_summary,'shared_target_similarity_summary'),(shared_results,'shared_target_paired_tests'),(ranks,'raw_probability_rank_comparison'),(pd.DataFrame(rank_sensitivity),'donor_count_sensitivity'),(opponent,'opponent_preference_tests'),(pd.DataFrame(opponent_distances),'opponent_similarity_summary')]:atomic_csv(d,out/(name+'.csv'))
    # Availability-matched descriptions for pair rankings and individual stability.
    same_games=set(wide.index.astype(int))
    mutual_matched=pairdata.loc[pairdata.full_observed_network.eq(True)&pairdata.game_number.isin(same_games)&pairdata.player_1.isin(core)&pairdata.player_2.isin(core)].groupby(['player_1','player_2']).agg(games=('game_number','size'),mean_mutual_preference=('mutual_preference','mean')).reset_index()
    mutual_matched['pair']=[names[a]+' ↔ '+names[b] for a,b in zip(mutual_matched.player_1,mutual_matched.player_2)]
    stability_wide=stable_primary.loc[stable_primary.player_id.isin(core)].pivot(index='game_number',columns='player_id',values='preference_js').dropna()
    common_wide=stable_primary.loc[stable_primary.player_id.isin(core)].pivot(index='game_number',columns='player_id',values='common_receiver_js').reindex(stability_wide.index)
    stability_matched=pd.DataFrame([dict(player_id=pid,name=names[pid],matched_consecutive_pairs=len(stability_wide),mean_js=stability_wide[pid].mean(),mean_common_receiver_js=common_wide[pid].mean()) for pid in core])
    atomic_csv(mutual_matched,out/'mutual_core_pairs_matched_games.csv');atomic_csv(stability_matched,out/'stability_core_matched_pairs.csv')
    green10=[]
    for (direction,pid),d in green_data.loc[green_data.source_passes_to_nonGreen.ge(10)].groupby(['direction','partner_id']):
        if min(int(d.green_absent.eq(0).sum()),int(d.green_absent.eq(1).sum()))<5 or d.probability.nunique()<2:continue
        fit,_=warriors_role_hac(d,'probability',['green_absent'],'min10_nonGreen_passes');r=next(v for v in fit if v['term']=='green_absent');green10.append(dict(direction=direction,partner_id=pid,partner=names[pid],n=len(d),change=r['coefficient'],p_value=r['p_value_hac7']))
    atomic_csv(pd.DataFrame(green10),out/'green_absence_min10_sensitivity.csv')
    example_games=[]
    for condition in [('Curry and Green present',True),('Curry present, Green absent',False)]:
        candidate=base.loc[base.full_observed_network.eq(True)&base.curry_present.eq(True)&base.green_present.eq(condition[1])]
        if len(candidate):example_games.append((int(candidate.game_number.min()),condition[0]))
    with PdfPages(out/'eight_probability_network_ideas.pdf') as pdf:
        def save(fig,name):
            fig.tight_layout();pdf.savefig(fig);fig.savefig(out/(name+'.pdf'));fig.savefig(out/(name+'.png'),dpi=140);plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(15,8));shown=preferences.head(12).sort_values('season_game_averaged_preference_share');axes[0].barh(shown.name,100*shown.season_game_averaged_preference_share,color='#2473b8');axes[0].set(xlabel='Equal-game averaged receiving preference share (%)',title='1. Broadly preferred receivers\nAbsences contribute zero to this season-level total')
        axes[1].barh(leadercounts.probability_receiver.iloc[::-1],leadercounts.games_leading_probability_strength.iloc[::-1],color='#2473b8');axes[1].set(xlabel='Fully observed games leading incoming probability strength',title='Who leads in individual games?');save(fig,'idea_1_receiving_hubs')
        fig,ax=plt.subplots(figsize=(12,8));shown=mutual.loc[mutual.eligible_games.ge(15)].head(12).iloc[::-1];ax.barh(shown.pair,shown.mean_mutual_preference,color='#2473b8');ax.set(xlabel='Mean P(i→j) × P(j→i), including zero-affinity eligible games',title='2. Mutual passing preferences\nDisplay: at least 15 games with both outgoing distributions defined');save(fig,'idea_2_mutual_partnerships')
        fig,axes=plt.subplots(1,2,figsize=(16,9))
        for ax,direction in zip(axes,['Curry to teammate','Teammate to Curry']):
            g=green.loc[green.direction.eq(direction)].sort_values('change');y=np.arange(len(g));ax.errorbar(100*g.change,y,xerr=np.vstack([100*(g.change-g.ci95_low),100*(g.ci95_high-g.change)]),fmt='o',color='#2473b8',capsize=2)
            ax.set_yticks(y,g.partner);ax.axvline(0,color='gray',ls='--');ax.set(xlabel='Probability change without Green (percentage points)',title=direction+'\nGreen excluded from outgoing target set in BOTH conditions')
        fig.suptitle('3. Preference changes in Curry appearances · unadjusted 95% HAC7 intervals');save(fig,'idea_3_green_absence')
        fig,ax=plt.subplots(figsize=(12,8));shown=stability_summary.loc[stability_summary.consecutive_pairs.ge(10)].sort_values('mean_js');y=np.arange(len(shown));ax.barh(y-.15,shown.mean_js,height=.3,label='Full receiver set');ax.barh(y+.15,shown.mean_common_receiver_js,height=.3,label='Only receivers available in both games');ax.set_yticks(y,shown.name);ax.set(xlabel='Mean consecutive-game JS divergence (0 = same preferences)',title='4. Passing-preference stability\nDisplay: at least 10 complete consecutive comparisons');ax.legend();save(fig,'idea_4_stability')
        fig,axes=plt.subplots(1,2,figsize=(15,8));shown=choice_summary.loc[choice_summary.games.ge(15)].sort_values('mean_first_choice_probability');axes[0].barh(shown.name,100*shown.mean_first_choice_probability,color='#2473b8');axes[0].set(xlabel='Mean largest outgoing probability (%)',title='5. Reliance on the first-choice receiver');axes[1].barh(shown.name,100*shown.mean_first_second_gap,color='#e18a24');axes[1].set(xlabel='Mean first-choice minus second-choice gap (percentage points)',title='How far ahead is the preferred option?');save(fig,'idea_5_first_choice')
        fig,ax=plt.subplots(figsize=(11,8));shown=shared_summary;ax.bar(shown.pair,shown.matched_trio_mean_js,color=['#2473b8','#e18a24','#168348']);ax.set(ylabel='Mean shared-target JS divergence (lower = more similar)',title=f'6. Do the core passers target the same other teammates?\nSame {len(wide)} games; compared players excluded from both target sets');save(fig,'idea_6_shared_targets')
        fig,axes=plt.subplots(1,2,figsize=(15,7));axes[0].hist(primaryranks.incoming_rank_spearman,bins=12,color='#2473b8');axes[0].set(xlabel='Raw receiving vs probability-strength rank correlation',ylabel='Games',title='7. Volume hubs versus preference hubs')
        axes[1].hist(primaryranks.loc[primaryranks.directed_eigenvector_available.eq(True),'directed_eigenvector_rank_spearman'],bins=12,color='#e18a24');axes[1].set(xlabel='Raw vs probability directed-left-eigenvector rank correlation',ylabel='Games',title='PageRank unchanged by outgoing row normalization\nDirected eigenvector comparison uses irreducible games');save(fig,'idea_7_rank_comparison')
        fig,ax=plt.subplots(figsize=(12,8));os=pd.DataFrame(opponent_distances).loc[lambda z:z.variant.eq('all_receivers')];x=np.arange(len(os));ax.bar(x-.18,os.within_opponent_mean_js,width=.36,label='Same opponent');ax.bar(x+.18,os.between_opponent_mean_js,width=.36,label='Different opponents');ax.set(xticks=x,xticklabels=os.name,ylabel='Mean pairwise JS divergence',title='8. Opponent-associated passing preferences\nDescriptive distances; circular-shift test accounts for group sizes and profile order');ax.legend();save(fig,'idea_8_opponents')
        for g,title in example_games:
            ids,A=rawnets[g];P,valid=warriors_probability_matrix(A);G=nx.from_numpy_array(A,create_using=nx.DiGraph);poslayout=nx.spring_layout(G.to_undirected(),seed=2026,weight='weight',iterations=150);fig,axes=plt.subplots(1,2,figsize=(18,8))
            for ax,weights,label in [(axes[0],A,'Raw pass counts'),(axes[1],P,'Outgoing passing probabilities')]:
                displayed=[]
                for i in range(len(ids)):
                    positive=np.flatnonzero(weights[i]>0);best=sorted(positive,key=lambda j:(-weights[i,j],int(ids[j])))[:2];displayed.extend((i,int(j)) for j in best)
                # Retain a real incoming edge for any receiver isolated by display pruning.
                for receiver in range(len(ids)):
                    if not any(receiver in edge for edge in displayed):
                        sources=np.flatnonzero(weights[:,receiver]>0)
                        if len(sources):displayed.append((int(sources[np.argmax(weights[sources,receiver])]),receiver))
                values=np.array([weights[i,j] for i,j in displayed]);widths=.6+4*values/max(values.max(),1);nodecolors=['#2473b8' if pid==201939 else '#168348' if pid==203110 else '#e18a24' if pid==1629673 else '#c8cdd3' for pid in ids]
                nx.draw_networkx_nodes(G,poslayout,node_color=nodecolors,node_size=700,ax=ax);nx.draw_networkx_edges(G,poslayout,edgelist=displayed,width=widths,alpha=.45,arrows=True,arrowsize=13,connectionstyle='arc3,rad=.12',ax=ax)
                nx.draw_networkx_labels(G,poslayout,labels={i:names[pid].replace('Stephen ','').replace('Draymond ','').replace('Jordan ','') for i,pid in enumerate(ids)},font_size=8,ax=ax)
                edge_labels={}
                for i,pid in enumerate(ids):
                    if pid in core and weights[i].sum()>0:
                        j=int(np.argmax(weights[i]));edge_labels[(i,j)]=str(int(weights[i,j])) if label=='Raw pass counts' else f'{100*weights[i,j]:.1f}%'
                nx.draw_networkx_edge_labels(G,poslayout,edge_labels=edge_labels,font_size=7,label_pos=.4,rotate=False,bbox=dict(facecolor='white',edgecolor='none',alpha=.8),ax=ax)
                ax.set_title(label);ax.axis('off')
            fig.suptitle(f'Game {g}: {title}\nSame Fruchterman–Reingold positions; Strongest two outgoing edges plus links retaining active receivers; core first-choice weights labeled');save(fig,f'network_example_game_{g:02d}')
    # Detailed reusable report: all eight ideas, definitions, results and limitations.
    changed=int(primaryranks.leader_changed.sum());unique=primaryranks.loc[primaryranks.raw_leader_ties.eq(1)&primaryranks.probability_leader_ties.eq(1)];ecrows=primaryranks.loc[primaryranks.directed_eigenvector_available.eq(True)]
    summary=[dict(idea=1,result='Preferred receiver game leaders',detail=leadercounts.to_string(index=False)),dict(idea=2,result='Mutual pair rankings',detail=mutual.loc[mutual.eligible_games.ge(15)].head(10).to_string(index=False)),
        dict(idea=3,result='Curry preference shifts without Green',detail=str(int(green.p_holm_edges.lt(.05).sum()))+' of '+str(len(green))+' tested directional edges have Holm p<.05'),
        dict(idea=4,result='Consecutive preference comparisons',detail=str(len(stable_primary))+' player-game comparisons; compare common-receiver and min10 sensitivities'),
        dict(idea=5,result='First-choice dependence',detail=choice_summary.loc[choice_summary.player_id.isin(core)].to_string(index=False)),
        dict(idea=6,result='Core passer target similarity',detail=shared_summary.to_string(index=False)),
        dict(idea=7,result='Count-vs-probability hubs',detail=f'{changed}/{len(primaryranks)} incoming leaders differ; mean rank correlation {primaryranks.incoming_rank_spearman.mean():.4f}; PageRank maximum difference {ranks.pagerank_max_difference.max():.3g}'),
        dict(idea=8,result='Opponent association',detail=opponent.to_string(index=False))]
    atomic_csv(pd.DataFrame(summary),out/'eight_idea_summary.csv')
    report=['# Warriors 2021–22: eight probability-weighted passing-network ideas',
        '## Scope and definitions',
        f'82-game saved season; 81 valid networks, game 38 omitted. Primary descriptions use {len(complete)} fully observed games; all 81 are exported and available in CSV. No minimum minutes cutoff. A zero-pass source has no defined outgoing probability distribution; its CSV row is blank (NaN), not assigned fictitious probabilities. Receiving-only nodes remain valid recipients. Isolates omitted. No new NBA downloads.',
        'P[i,j]=A[i,j]/sum_j A[i,j] for each defined outgoing row. Each source has one probability unit, independent of pass count. Incoming preference strength sum_i P[i,j] can exceed 1; divide by number of defined sources to obtain a receiving preference share summing to 1. Only one share of an equal-source vote, not actual possession or minutes share. First-choice votes split equally across ties. All metrics keep edge direction; no symmetrization.',
        '## 1. Is there a broadly preferred receiver?',leadercounts.to_string(index=False),preferences.to_string(index=False),
        'Season-game-averaged shares treat absence as zero contribution; mean when active is also saved. A leader is not necessarily chosen by every teammate. Different source/receiver availability changes the voting set. donor_count_sensitivity.csv repeats incoming preference rankings with sources making >=5,10,20 passes. These are pass-count reliability checks, not a minutes cutoff.',
        '## 2. Mutual preferred partnerships',mutual.loc[mutual.eligible_games.ge(15)].head(15).to_string(index=False),
        mutual_matched.to_string(index=False),'Mutual score P(i→j)*P(j→i); same-trio-game comparison is also saved to avoid ranking different availability samples as directly comparable. Both sources must have defined rows. Mean includes zero-pair scores in eligible co-observed games. This is mutual preference, not an observed give-and-go sequence. Exposure and >=10-pass sensitivity saved. Fifteen-game display threshold avoids ranking one-game pairs above recurring partnerships; full table retains all.',
        '## 3. Curry preferences when Green is absent',green.to_string(index=False),
        'Curry appearances in fully observed games only. Both endpoints must officially play. Green excluded as a target from all outgoing rows and probabilities reconditioned in BOTH availability conditions; this avoids mechanically redistributing Green’s probability mass in only one condition. Each edge requires >=5 coappearance games in each condition. HAC7 inference uses actual game gaps; Holm correction across all tested incoming/outgoing edge contrasts. Other lineup changes, roles, date and minutes are not controlled, so changes are not causal. Min10 nonGreen-source-pass sensitivity saved; it has fewer observations and unadjusted p-values.',
        '## 4. Are preferences stable?',stability_summary.to_string(index=False),
        stability_matched.to_string(index=False),'JS divergence, base2, lies between 0 (identical) and 1 (disjoint). Consecutive season games only; game 38 is not bridged. Primary pairs require both complete networks. Full receiver-set divergence includes availability changes. Common-receiver divergence restricts and reconditions to teammates present in both games. Min10-source-passes check reduces sampling noise. Large divergence can reflect limited samples and availability, not deliberate tactical change.',
        '## 5. One preferred receiver or several?',choice_summary.to_string(index=False),
        'Largest outgoing probability measures first-option dependence; first-minus-second gap measures how far ahead it is. Both are sensitive to small pass totals; all observations and >=10-pass counterparts saved. No outgoing-strength centrality is calculated as a discriminator because it equals 1 by construction for every defined row.',
        '## 6. Do Curry, Green and Poole target the same teammates?',shared_summary.to_string(index=False),shared_results.to_string(index=False),
        'Remove BOTH compared players from both target sets before renormalizing, so self-pass impossibility and reciprocal passes do not drive differences. Main paired comparisons use the same games with all three pair measures available, HAC7, Holm-three. Unrestricted available-game and raw-row results are also saved. Divergence reflects role and lineup context, not player quality. Compared players need not always share court minutes; game-level aggregate data does not identify simultaneous lineups.',
        '## 7. Does probability weighting identify different hubs?',
        f'Incoming raw-vs-probability leader differs in {changed}/{len(primaryranks)} complete games. Among games with unique leaders in both measures: {int(unique.leader_changed.sum())}/{len(unique)}. Mean Spearman incoming rank agreement: {primaryranks.incoming_rank_spearman.mean():.4f}. Directed eigenvectors available in {len(ecrows)} complete irreducible games; mean rank agreement {ecrows.directed_eigenvector_rank_spearman.mean():.4f}.',
        f'Maximum PageRank difference over all valid games: {ranks.pagerank_max_difference.max():.3g}. Standard weighted PageRank normalizes outgoing edge weights internally; row-scaling A into P leaves it unchanged. It is not independent evidence that count and probability networks have the same meaning. PageRank here uses alpha=.85 and standard uniform dangling redistribution, an algorithmic convention, not fictitious observed passes. Directed eigenvector means LEFT eigenvector: A.T v=rho v or P.T pi=pi. A row-stochastic P has a constant RIGHT eigenvector, so right eigenvector is not a useful importance ranking. Pure directed eigenvectors are not assigned to reducible games because uniqueness/comparability is not assured. Incoming strength is available without that assumption.',
        '## 8. Are preferences opponent-specific?',opponent.to_string(index=False),pd.DataFrame(opponent_distances).to_string(index=False),
        'Separate Curry/Green/Poole analyses; only opponents faced >=2 times in included games. Association R² from between-opponent variation in square-root probability profiles; high R² can arise from many opponent groups, so adjusted R² and null median reported. Exact circular shifts of profile order provide an exploratory reference preserving serial profile order and opponent counts. This requires approximate temporal exchangeability/stationarity, which injuries/season trends can violate; it is not a causal or predictive test. Holm-three across the three main all-receiver tests. Fixed-core sensitivity uses receivers present in >=80% of the player’s eligible games, retains games with all those receivers present, and reconditions probabilities to that core. It reduces availability differences but may have small samples. Fixed-core p-values unadjusted. No claim of game-plan adaptation from an association alone.',
        '## Interpretation and verification',
        'Algorithm references: https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.link_analysis.pagerank_alg.pagerank.html ; https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.eigenvector_centrality.html .',
        'All eight ideas retained, not selected for strong regressions. This is preference-network analysis, not a simulation of actual possession sequences. No shots, turnovers or win effects inferred from the weights alone. Sample restrictions and separate within-question correction families do not adjust all previous project exploration. Top receiver/pair ties and observation counts are exported. Example graphs use common Fruchterman–Reingold positions and display strongest two outgoing edges per source plus observed incoming edges needed to avoid visually isolated receivers; core first-choice weights labeled; all edges enter the analysis. Full updated script and CLI --warriors-probability-ideas preserve previous analyses.'
    ]
    def mdtable(frame):
        def value(v):
            if pd.isna(v):return '—'
            if isinstance(v,(float,np.floating)):return f'{v:.4g}'
            return str(v).replace('|','/').replace('\n',' ')
        lines=['| '+' | '.join(map(str,frame.columns))+' |','| '+' | '.join(['---']*len(frame.columns))+' |']
        lines.extend('| '+' | '.join(value(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None));return '\n'.join(lines)
    core_pref=preferences.loc[preferences.player_id.isin(core)].copy();core_pref['Mean preference share when active (%)']=100*core_pref.mean_preference_share
    core_choice=choice_summary.loc[choice_summary.player_id.isin(core)].copy();core_choice['First choice (%)']=100*core_choice.mean_first_choice_probability;core_choice['First–second gap (pp)']=100*core_choice.mean_first_second_gap
    matched_stability=stability_matched[['name','matched_consecutive_pairs','mean_js','mean_common_receiver_js']].rename(columns={'name':'Player','matched_consecutive_pairs':'Same consecutive pairs','mean_js':'Full-target divergence','mean_common_receiver_js':'Common-receiver divergence'})
    readable=['# Warriors probability-weighted passing networks: results explained',
        f'Completed all eight ideas. Primary sample: {len(complete)} fully observed games from the saved 2021–22 season. All 81 valid games are exported; game 38 is unavailable. P(i,j) is the share of player i’s recorded passes directed to player j. Each defined outgoing row sums to one, so players have equal total outgoing probability regardless of passing volume. This measures preferences, not actual possession sequences, player quality, or offensive success.',
        '## 1. Who is a broadly preferred receiver?',mdtable(leadercounts.rename(columns={'probability_receiver':'Receiver','games_leading_probability_strength':'Games leading'})),mdtable(core_pref[['name','observed_games','Mean preference share when active (%)']].rename(columns={'name':'Player','observed_games':'Observed games'})),
        'Curry leads the most individual game networks. Poole’s larger season-total preference contribution reflects more observed games, so it should not be interpreted as a higher preference share when both players are available. Incoming preference strength sums the probability weight supplied by each teammate; the reported share divides by the number of sources. It is not the share of all actual passes. A receiving-only player remains eligible even when their own outgoing distribution is undefined.',
        '## 2. Which partnerships are mutually preferred?',mdtable(mutual_matched[['pair','games','mean_mutual_preference']].rename(columns={'pair':'Pair','games':'Same games','mean_mutual_preference':'Mean mutual-preference product'})),
        'Among these three core pairs in the same games, Curry–Green has the largest mutual-preference product, followed by Green–Poole. Using every available coappearance instead produces different rankings because the game sets differ. The product P(i,j)P(j,i) describes reciprocal targeting; it is not the proportion of actual possessions containing a give-and-go. Frequent pair rankings and >=10-source-pass sensitivity are saved separately.',
        '## 3. Do Curry’s preferences change without Green?',
        f'{len(green)} directional Curry-related edges met the five-coappearance-game requirement in both conditions. None passes the within-question Holm correction. The smallest corrected p is {green.p_holm_edges.min():.4g}. Therefore the dataset does not provide clear evidence for a specific replacement target or supplier after the multiple comparisons are considered.',
        'Green was excluded from all receiver choices and distributions were reconditioned in both conditions. Otherwise a probability increase for other receivers could occur automatically merely because Green disappears. Both source and target must officially participate in the compared games. Other availability and seasonal differences remain possible explanations. Full estimates, intervals and low-pass-count checks are in the CSVs.',
        '## 4. Are passing preferences stable?',mdtable(matched_stability),
        'Zero divergence means identical preferences; larger values mean more different distributions. On all available consecutive games, Poole appears more variable than Curry. On the same 21 consecutive comparisons, their averages are nearly equal. This is a useful warning about comparing different availability samples. Green has a lower descriptive average on the matched sample. Restricting receivers to teammates present in both games reduces all three divergences, showing that availability accounts for part of the apparent preference change. These averages alone do not establish statistically different tactical stability.',
        '## 5. One preferred option or several?',mdtable(core_choice[['name','games','First choice (%)','First–second gap (pp)']].rename(columns={'name':'Player','games':'Games'})),
        'Green’s largest recipient probability is roughly 41% on average, compared with about 24–27% for Curry and Poole. Green also has a larger lead of his first choice over his second. Thus Green’s passing is more focused on a first option, while Curry and Poole distribute across several options. This does not rank their effectiveness; players have different roles. Every core-player row in this table also meets the >=10-source-pass check, so that particular threshold does not change these core averages.',
        '## 6. Do the three players target the same other teammates?',mdtable(shared_summary[['pair','matched_trio_games','matched_trio_mean_js']].rename(columns={'pair':'Pair','matched_trio_games':'Same games','matched_trio_mean_js':'Shared-target divergence'})),mdtable(shared_results.rename(columns={'comparison':'Comparison','n':'Games','difference':'Divergence difference','p_value':'Raw HAC7 p','p_holm_three':'Holm-three p'})),
        'Curry and Poole have the most similar distributions over their other teammates. Green and Poole have the least similar. The Curry–Poole versus Green–Poole contrast passes the three-comparison correction; the other two contrasts do not. These are only 28 matched games and an exploratory family of tests. Excluding both compared players from both receiver sets prevents their mutual passes and impossible self-passes from automatically inflating dissimilarity. Similarity is role overlap, not offensive efficiency.',
        '## 7. Does probability weighting change the identified hubs?',
        f'Incoming-strength leaders differ between raw-count and probability networks in {changed}/{len(primaryranks)} complete games ({100*changed/len(primaryranks):.1f}%). The average rank correlation is {primaryranks.incoming_rank_spearman.mean():.3f}: the broad hierarchy usually persists, but the leading player sometimes changes. Directed-left-eigenvector leaders differ in {int(ecrows.directed_eigenvector_leader_changed.sum())}/{len(ecrows)} eligible irreducible games; mean rank correlation {ecrows.directed_eigenvector_rank_spearman.mean():.3f}.',
        f'PageRank is unchanged up to floating-point error (maximum difference {ranks.pagerank_max_difference.max():.3g}). Standard weighted PageRank already normalizes outgoing rows, so A and P contain the same transitions for that algorithm. Outgoing strength also cannot distinguish players because it is one for every defined row. Incoming probability strength and the appropriate directed left eigenvector are useful comparisons here. The previous undirected eigenvector style metric is a different definition and was not silently substituted.',
        'The preferred incoming leader stays the same in about 84% of games when sources making fewer than ten passes are omitted. Small-volume teammates therefore affect some rankings; the equal-source weighting is a substantive choice, not a universally superior centrality measure.',
        '## 8. Are preferences opponent-specific?',mdtable(opponent.loc[opponent.variant.eq('all_receivers'),['name','n','opponents','association_r_squared','adjusted_r_squared','circular_shift_p','p_holm_three']].rename(columns={'name':'Player','n':'Games','opponents':'Opponent groups','association_r_squared':'Raw association R²','adjusted_r_squared':'Adjusted R²','circular_shift_p':'Shift-reference p','p_holm_three':'Holm-three p'})),
        'No main test supports an opponent-specific pattern. The raw association R² around one-third looks substantial, but many opponent categories can explain that much variation by chance in small samples. Adjusted R² is negative, and observed association is not unusual relative to circularly shifted profiles. Fixed-core receiver sensitivities also show no clear pattern, with smaller samples. Circular-shift comparisons preserve profile order but still assume sufficient temporal exchangeability; season trends and injuries limit interpretation. This is not proof the Warriors never adapt to an opponent.',
        '## What this adds',
        'The useful result is a clearer distinction between volume, preference and role. Curry is frequently the broad receiving preference hub; Green concentrates his outgoing preferences more strongly; Curry and Poole have more similar other-target distributions. Probability weighting changes some leaders without changing PageRank. These findings describe organization and do not establish that a structure causes better shooting or wins.',
        'The detailed methods/audit report, all probability matrices, all eight figure sets, Fruchterman–Reingold network examples, CSVs and full updated script accompany this report. Separate within-question Holm corrections do not adjust all analyses conducted earlier in the project.'
    ]
    (out/'RESULTS_EXPLAINED.md').write_text('\n\n'.join(readable))

    (out/'DETAILED_ANALYSIS.md').write_text('\n\n'.join(report));print('Incoming leaders:',leadercounts.to_string(index=False),flush=True);print('Leader changes:',changed,'/',len(primaryranks),'mean rank rho',primaryranks.incoming_rank_spearman.mean(),'PR maxdiff',ranks.pagerank_max_difference.max(),flush=True)
    print('Green absence significant edges:',green.loc[green.p_holm_edges.lt(.05)].to_string(index=False),flush=True);print('Core targets:',shared_summary.to_string(index=False),flush=True);print('Opponent tests:',opponent.to_string(index=False),flush=True);print('Saved probability network ideas: '+str(out),flush=True);return 0



# ---------------------------------------------------------------------
# EIGHT INTEGRATED RESEARCH DESIGNS: OPPORTUNITY, PREFERENCES AND REPLACEMENT
# ---------------------------------------------------------------------
def integrated_probability(A):
    A=np.asarray(A,float);tot=A.sum(1)
    return np.divide(A,tot[:,None],out=np.zeros_like(A),where=tot[:,None]>0),tot>0


def integrated_hhi(v):
    v=np.asarray(v,float);q=v/v.sum();return float((q@q-1/len(q))/(1-1/len(q)))


def integrated_team_fit(data,x,y,label,controls=None):
    f=season_topics_fit(data,x,y,label,label,[x,*(controls or [])]);return f,next(v for v in f if v['term']==x)


def integrated_table(d):
    def cell(v):
        if pd.isna(v):return '—'
        return (f'{v:.5g}' if isinstance(v,(float,np.floating)) else str(v)).replace('|','/').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(map(str,d.columns))+' |','| '+' | '.join(['---']*len(d.columns))+' |',*['| '+' | '.join(cell(v) for v in row)+' |' for row in d.itertuples(index=False,name=None)]])


def integrated_scatter(data,x,y,path,title,xlabel=None,ylabel=None):
    import matplotlib.pyplot as plt
    from scipy.stats import linregress
    d=data.dropna(subset=[x,y]);fig,ax=plt.subplots(figsize=(11,8))
    if 'passing_style' in d:
        for style,col,marker,label in [('more_balanced','#2473b8','o','Balanced'),('more_star_centered','#e18a24','^','Star-centered')]:
            z=d.loc[d.passing_style.eq(style)];ax.scatter(z[x],z[y],color=col,marker=marker,label=label)
        ax.legend()
    else:ax.scatter(d[x],d[y],color='#2473b8',s=30,alpha=.7)
    fit=linregress(d[x],d[y]);xx=np.linspace(d[x].min(),d[x].max(),100);ax.plot(xx,fit.intercept+fit.slope*xx,color='#333333')
    if 'team' in d and len(d)<=35:
        for _,r in d.iterrows():ax.annotate(r.team,(r[x],r[y]),xytext=(4,4),textcoords='offset points',fontsize=8)
    ax.set(xlabel=xlabel or x.replace('_',' '),ylabel=ylabel or y.replace('_',' '),title=title+f'\nn={len(d)}; descriptive r={fit.rvalue:.3f}; R²={fit.rvalue**2:.3f}');ax.grid(alpha=.15);fig.tight_layout();season_topics_spread_labels(fig);fig.savefig(path.with_suffix('.pdf'));fig.savefig(path.with_suffix('.png'),dpi=140);plt.close(fig)


def integrated_study_1(ctx,out):
    players=ctx['players'];season=ctx['season'];rows=[]
    for _,r in season.iterrows():
        ids,A=ctx['networks'][r.team];P,valid=integrated_probability(A);pref=P.sum(0)/valid.sum();raw=A.sum(0)/A.sum();stats=players.loc[players.TEAM_ID.eq(r.team_id)].groupby('PLAYER_ID').agg(name=('PLAYER_NAME','first'),minutes=('MIN','sum'),fga=('FGA','sum')).reindex(ids)
        for j,pid in enumerate(ids):rows.append(dict(team=r.team,player_id=pid,name=stats.loc[pid,'name'],passing_style=r.passing_style,minutes_share=stats.loc[pid,'minutes']/stats.minutes.sum(),shot_share=stats.loc[pid,'fga']/stats.fga.sum(),raw_receiving_share=raw[j],preference_share=pref[j]))
    d=pd.DataFrame(rows);atomic_csv(d,out/'player_features.csv');models={'minutes_only':['minutes_share'],'minutes_raw':['minutes_share','raw_receiving_share'],'minutes_preference':['minutes_share','preference_share'],'minutes_raw_preference':['minutes_share','raw_receiving_share','preference_share']};reg=[]
    for label,predictors in models.items():
        x=predictors[-1];row,_=six_player_cluster_fit(d,x,'shot_share',predictors[:-1],label);reg.append(row)
    predictions=[];folds=[]
    for team in d.team.unique():
        train=d.loc[d.team.ne(team)];test=d.loc[d.team.eq(team)]
        for label,features in models.items():
            X=np.column_stack([np.ones(len(train)),train[features]]);b=np.linalg.lstsq(X,train.shot_share,rcond=None)[0];pred=np.column_stack([np.ones(len(test)),test[features]])@b;err=pred-test.shot_share.to_numpy()
            folds.append(dict(team=team,model=label,n=len(test),rmse=float(np.sqrt(np.mean(err**2))),mae=float(np.mean(np.abs(err)))))
            for pid,y,p in zip(test.player_id,test.shot_share,pred):predictions.append(dict(team=team,player_id=pid,model=label,observed=y,predicted=p))
    folds=pd.DataFrame(folds);pred=pd.DataFrame(predictions);summary=[]
    for label,g in pred.groupby('model'):
        error=g.predicted-g.observed;summary.append(dict(model=label,players=len(g),pooled_rmse=np.sqrt(np.mean(error**2)),pooled_mae=np.mean(np.abs(error)),mean_team_rmse=folds.loc[folds.model.eq(label),'rmse'].mean(),cv_r_squared=1-np.sum(error**2)/np.sum((g.observed-g.observed.mean())**2)))
    summary=pd.DataFrame(summary);atomic_csv(pd.DataFrame(reg),out/'clustered_regressions.csv');atomic_csv(folds,out/'leave_team_out_errors.csv');atomic_csv(pred,out/'leave_team_out_predictions.csv');atomic_csv(summary,out/'prediction_summary.csv')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(11,7));labels={'minutes_only':'Minutes only','minutes_raw':'Minutes +\nraw receptions','minutes_preference':'Minutes +\nreceiving preferences','minutes_raw_preference':'Minutes + raw receptions\n+ receiving preferences'};bars=ax.bar([labels[x] for x in summary.model],summary.pooled_rmse,color='#2473b8');ax.bar_label(bars,fmt='%.4f',padding=3);ax.set(ylabel='Held-out-team prediction error (RMSE in shot share)',title='1. Does receiving preference add information beyond minutes?');fig.tight_layout();fig.savefig(out/'model_comparison.pdf');fig.savefig(out/'model_comparison.png',dpi=140);plt.close(fig)
    main=next(v for v in reg if v['model']=='minutes_raw_preference')
    report=['## 1. Passing preference beyond minutes',integrated_table(summary),integrated_table(pd.DataFrame(reg)),
        '2025–26 player-team stints, all official participants. Main inference: coefficient on equal-source receiving preference after minutes share and raw receiving share, team fixed effects and two-way team/player clustered t errors. Cross-validation is separately pooled OLS, leaving one whole team out at a time; it cannot use an unknown held-out team fixed effect. Predictions are unconstrained share estimates and may be slightly negative; no post-hoc clipping. Transferred players can appear in different team folds, so this evaluates generalization to team contexts, not unseen player identities. Fold errors are descriptive, not independent training experiments. Same-season explanation of shot allocation is not prediction of future efficiency.']
    return dict(idea=1,test='Receiving preference coefficient beyond minutes and raw receptions',n=main['n'],effect=main['coefficient'],p_value=main['p_value']),report


def integrated_study_2(ctx,out):
    from scipy.spatial.distance import jensenshannon
    rows=[];candidates=[];players=ctx['players'];parent=ctx['parent'];hubs=pd.read_csv(parent/'hub_absence_resilience/team_season_metrics.csv')
    for _,r in hubs.loc[hubs.primary_eligible.eq(True)].iterrows():
        ids,A=ctx['networks'][r.team];present=pd.read_csv(parent/'hub_absence_resilience/present_networks'/r.team/'passing_matrix.csv',index_col=0);Pcount=present.to_numpy(float);B=A-Pcount;hub=ids.index(int(r.hub_id));own=players.loc[players.TEAM_ID.eq(r.team_id)];played=set(own.loc[own.PLAYER_ID.eq(r.hub_id),'GAME_ID']);names=own.drop_duplicates('PLAYER_ID').set_index('PLAYER_ID').PLAYER_NAME.to_dict();rank=[]
        for j,pid in enumerate(ids):
            if j==hub:continue
            gp=own.loc[own.PLAYER_ID.eq(pid)];n=int(gp.GAME_ID.isin(played).sum());take=[i for i in range(len(ids)) if i not in [hub,j]];v=Pcount[hub,take];w=Pcount[j,take]
            if n<10 or w.sum()<100 or v.sum()<=0:continue
            similarity=1-float(jensenshannon(v,w,base=2)**2);availability=float((~gp.GAME_ID.isin(played)).sum()/r.absent_games);row=dict(team=r.team,hub=r.hub_name,candidate=names[pid],player_id=pid,similarity=similarity,present_appearances=n,absent_period_availability=availability);candidates.append(row);rank.append(row)
        if not rank:raise ValueError('No adequately observed replacement candidate '+r.team)
        best=sorted(rank,key=lambda z:(-z['similarity'],z['player_id']))[0];take=[j for j in range(len(ids)) if j!=hub]
        while True:
            p=Pcount[np.ix_(take,take)];b=B[np.ix_(take,take)];keep=(p.sum(1)>0)&(b.sum(1)>0)
            if keep.all():break
            take=[i for i,k in zip(take,keep) if k]
        PP,_=integrated_probability(p);PB,_=integrated_probability(b);js=float(np.mean([jensenshannon(x,y,base=2)**2 for x,y in zip(PP,PB)]));panel=[ids[j] for j in take];mins=[]
        for flag in [True,False]:mins.append(own.loc[own.GAME_ID.isin(played).eq(flag)].groupby('PLAYER_ID').MIN.sum().reindex(panel,fill_value=0).to_numpy(float))
        rows.append(dict(team=r.team,hub=r.hub_name,passing_style=ctx['styles'][r.team],best_replacement=best['candidate'],replacement_similarity=best['similarity'],replacement_availability=best['absent_period_availability'],preference_disruption=js,
            adjusted_offensive_loss=r.adjusted_absence_penalty,minutes_distribution_change=float(jensenshannon(*mins,base=2)**2),log_pass_count_ratio=np.log(b.sum()/p.sum()),matched_nonhub_players=len(take)))
    d=pd.DataFrame(rows);fit,main=integrated_team_fit(d,'replacement_similarity','preference_disruption','replacement_similarity_primary');fits=list(fit)
    for y,controls,label in [('adjusted_offensive_loss',[],'offense_secondary'),('preference_disruption',['replacement_availability','minutes_distribution_change','log_pass_count_ratio'],'availability_minutes_count_sensitivity')]:fits.extend(integrated_team_fit(d,'replacement_similarity',y,label,controls)[0])
    atomic_csv(d,out/'team_replacements.csv');atomic_csv(pd.DataFrame(candidates),out/'candidate_similarities.csv');atomic_csv(pd.DataFrame(fits),out/'regressions.csv');integrated_scatter(d,'replacement_similarity','preference_disruption',out/'replacement_disruption','2. Does a similar replacement accompany less disruption?')
    report=['## 2. Replacement suitability and resilience',integrated_table(pd.DataFrame(fits).loc[lambda z:z.term.eq('replacement_similarity')][['model','n','coefficient','pearson_r','p_value_hc3']]),
        'Twenty-one saved hub-absence cases. Candidate similarity uses only hub-present passing to other teammates, excludes the two compared players from targets, requires >=10 hub-present appearances and >=100 observed candidate passes to those other targets (exposure criteria, no minute cutoff). Best similarity is selected without using absent passing preferences. Outcome is mean outgoing-row JS on the same iteratively positive-outgoing nonhub panel in present/absent networks. Absence availability and minute-distribution change are sensitivity controls, not part of primary predictor. This is retrospective structural association, not pre-injury prediction or evidence of causal replacement ability. Aggregate pass sample sizes differ; residual counts and co-absences can matter.']
    return dict(idea=2,test='Replacement similarity vs absent-period preference disruption',n=main['n'],effect=main['coefficient'],r=main['pearson_r'],r_squared=main['r_squared'],p_value=main['p_value_hc3']),report


def integrated_fixed_margin_draws(A,draws,seed,cache):
    """Labeled-pass destination rewiring, fixed row/column totals and no self-passes.
    Reversible 2-edge swaps AND 3-edge rotations avoid the directed-triangle trap.
    """
    import hashlib
    A=np.asarray(A,np.int64);n=len(A);fingerprint=hashlib.sha256(A.tobytes()+str((draws,seed,'2and3moves_v2')).encode()).hexdigest()
    if cache.exists():
        d=np.load(cache,allow_pickle=False)
        if str(d['source_hash'])==fingerprint:return {k:d[k] for k in d.files}
    rng=np.random.default_rng(seed);source=np.repeat(np.arange(n),A.sum(1)).tolist();target=[]
    for i in range(n):
        for j in range(n):target.extend([j]*int(A[i,j]))
    source_array=np.array(source);M=len(target);accepted=0;attempts=0
    def sweep(number):
        nonlocal accepted,attempts
        picks=rng.integers(0,M,size=(number,3));triple=rng.random(number)<.2;direction=rng.random(number)<.5
        for (u,v,w),three,forward in zip(picks,triple,direction):
            attempts+=1;r1=source[u];r2=source[v];d1=target[u];d2=target[v]
            if not three:
                if r1!=r2 and d1!=d2 and r1!=d2 and r2!=d1:target[u],target[v]=d2,d1;accepted+=1
            else:
                r3=source[w];d3=target[w]
                if len({r1,r2,r3})<3 or len({d1,d2,d3})<3:continue
                new=(d2,d3,d1) if forward else (d3,d1,d2)
                if r1!=new[0] and r2!=new[1] and r3!=new[2]:target[u],target[v],target[w]=new;accepted+=1
    sweep(20*M);stats=[];pair_sum=np.zeros_like(A,float);pair2=np.zeros_like(A,float);edge_sum=np.zeros_like(A,float)
    for k in range(draws):
        sweep(2*M);B=np.bincount(source_array*n+np.asarray(target),minlength=n*n).reshape(n,n)
        if not np.array_equal(B.sum(0),A.sum(0)) or not np.array_equal(B.sum(1),A.sum(1)) or np.diag(B).any():raise ValueError('Null margins or self-pass invariant failed')
        edge_sum+=B
        P,_=integrated_probability(B);product=P*P.T;stats.append(float(product.sum()/n));pair_sum+=product;pair2+=product**2
    stats=np.array(stats);mean=pair_sum/draws;sd=np.sqrt(np.maximum(pair2/draws-mean**2,0));lag=np.corrcoef(stats[:-1],stats[1:])[0,1] if np.std(stats)>0 else 0
    result=dict(samples=stats,pair_mean=mean,pair_sd=sd,edge_mean=edge_sum/draws,source_hash=np.array(fingerprint),accepted=np.array(accepted),attempts=np.array(attempts),lag1=np.array(lag));np.savez_compressed(cache,**result);return result


def integrated_study_3(ctx,out):
    root=ctx['warriors'];base=pd.read_csv(root/'role_checks/game_metrics.csv');base=base.loc[base.full_observed_network.eq(True)];cache=out/'null_cache';cache.mkdir(exist_ok=True);observed=[];nulls=[];diagnostics=[];pairrows=[];chains=[]
    for k,(_,r) in enumerate(base.iterrows(),1):
        g=int(r.game_number);f=pd.read_csv(root/'eigenvector_team_performance'/f'game_{g:02d}'/'passing_matrix.csv',index_col=0);ids=f.index.astype(int).to_numpy();A=f.to_numpy(np.int64);active=A.sum(0)+A.sum(1)>0;A=A[np.ix_(active,active)];ids=ids[active];P,valid=integrated_probability(A);obs=float(np.sum(P*P.T)/len(A));d=integrated_fixed_margin_draws(A,ctx['null_draws'],ctx['seed']+g,cache/f'game_{g:02d}.npz');observed.append(obs);nulls.append(d['samples'])
        diagnostics.append(dict(game_number=g,observed_reciprocity=obs,null_mean=d['samples'].mean(),null_sd=d['samples'].std(),lag1=float(d['lag1']),acceptance=float(d['accepted']/d['attempts'])))
        pg=pd.read_csv(root/f'game_{g:02d}'/'game_players.csv').set_index('PLAYER_ID').PLAYER_NAME.to_dict()
        for i in range(len(ids)):
            for j in range(i+1,len(ids)):
                if valid[i] and valid[j]:pairrows.append(dict(game_number=g,player_1=int(ids[i]),player_2=int(ids[j]),pair=pg[ids[i]]+' ↔ '+pg[ids[j]],observed=P[i,j]*P[j,i],null_expected=d['pair_mean'][i,j],excess=P[i,j]*P[j,i]-d['pair_mean'][i,j]))
        if k in [1,1+len(base)//2,len(base)]:
            second=integrated_fixed_margin_draws(A,199,ctx['seed']+100000+g,cache/f'chain_check_game_{g:02d}.npz')
            standard_error=np.sqrt(np.var(d['samples'],ddof=1)/len(d['samples'])+np.var(second['samples'],ddof=1)/len(second['samples']))
            chains.append(dict(game_number=g,primary_mean=d['samples'].mean(),independent_chain_mean=second['samples'].mean(),difference_in_nominal_MC_SE=(d['samples'].mean()-second['samples'].mean())/standard_error if standard_error>0 else 0))
        if k%10==0 or k==len(base):print(f'Fixed-margin nulls {k}/{len(base)} complete',flush=True)
    obs=float(np.mean(observed));null=np.mean(np.stack(nulls),axis=0);p=(1+int(np.sum(null>=obs)))/(len(null)+1);pairdata=pd.DataFrame(pairrows);summary=pairdata.groupby(['player_1','player_2','pair']).agg(games=('game_number','size'),observed_mean=('observed','mean'),null_mean=('null_expected','mean'),mean_excess=('excess','mean')).reset_index().sort_values('mean_excess',ascending=False)
    atomic_csv(pd.DataFrame(chains),out/'independent_chain_diagnostics.csv');atomic_csv(pd.DataFrame(diagnostics),out/'null_diagnostics.csv');atomic_csv(pairdata,out/'pair_game_excess.csv');atomic_csv(summary,out/'partnership_excess_summary.csv');atomic_csv(pd.DataFrame(dict(draw=np.arange(len(null)),global_null_reciprocity=null)),out/'global_null_draws.csv')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(11,7));ax.hist(null,bins=25,color='#9bb8d4');ax.axvline(obs,color='#c65a2a',lw=2,label='Observed');ax.set(xlabel='Mean game-level mutual preference mass / active players',ylabel='Null draws',title=f'3. Are partnerships more reciprocal than activity predicts?\nFixed row/column totals, no self-passes; one-sided Monte Carlo p={p:.4g}');ax.legend();fig.tight_layout();fig.savefig(out/'fixed_margin_null.pdf');fig.savefig(out/'fixed_margin_null.png',dpi=140);plt.close(fig)
    report=['## 3. Partnerships beyond passing opportunity',f'Observed mean reciprocity={obs:.6g}; null mean={null.mean():.6g}; excess={obs-null.mean():.6g}; one-sided Monte Carlo p={p:.5g}.',integrated_table(summary.loc[summary.games.ge(15)].head(12)),
        f'{len(base)} complete Warriors games. Each null preserves the exact outgoing and incoming pass totals, and forbids self-passes. Destination rewiring on labeled passes uses reversible two-edge swaps plus three-edge rotations; the latter allows directed triangle reversals. This is a degree/strength-constrained configuration reference, not uniform sampling over count tables. Burn-in 20 pass-count sweeps, two sweeps between {ctx["null_draws"]} samples per game; per-game acceptance and lag-one diagnostics saved. Correlated Monte Carlo draws limit numerical precision; finite burn-in is a diagnostic, not a mixing proof. Per-pair excesses descriptive, no cherry-picked significance claims. The global statistic is sum_ij Pij Pji / N; it is mutual preference, not actual two-pass sequences. Null draws condition on observed activity, not shared court time or defense.']
    return dict(idea=3,test='Excess mutual preference versus fixed-margin networks',n=len(base),effect=obs-null.mean(),p_value=p),report


def integrated_study_4(ctx,out):
    from scipy.spatial.distance import jensenshannon
    parent=ctx['parent'];old=pd.read_csv(parent/'eight_context_topics/prior_player_games.csv',dtype={'GAME_ID':str});current=ctx['players'];rows=[]
    for _,r in ctx['season'].iterrows():
        own0=old.loc[old.TEAM_ID.eq(r.team_id)].groupby('PLAYER_ID').MIN.sum();own1=current.loc[current.TEAM_ID.eq(r.team_id)].groupby('PLAYER_ID').MIN.sum();ids,A=ctx['networks'][r.team];common=sorted(set(own0.index)&set(own1.index));index={p:i for i,p in enumerate(common)};B=np.zeros((len(common),len(common)))
        for pid in common:
            path=parent/'eight_context_topics/profiles/prior'/r.team/(str(pid)+'.csv')
            if not path.exists():raise ValueError('Missing full-roster retained-player profile '+str(path))
            f=pd.read_csv(path)
            for _,v in f.iterrows():
                q=int(v.PASS_TEAMMATE_PLAYER_ID)
                if q in index and q!=pid:B[index[pid],index[q]]+=v.PASS
        J=[ids.index(p) for p in common];C=A[np.ix_(J,J)];take=list(range(len(common)))
        while True:
            b=B[np.ix_(take,take)];c=C[np.ix_(take,take)];keep=(b.sum(1)>0)&(c.sum(1)>0)
            if keep.all():break
            take=[i for i,k in zip(take,keep) if k]
        pb,_=integrated_probability(b);pc,_=integrated_probability(c);rows.append(dict(team=r.team,passing_style=r.passing_style,retained_previous_minutes=own0.reindex(common).sum()/own0.sum(),retained_current_minutes=own1.reindex(common).sum()/own1.sum(),retained_ids=len(common),matched_positive_sources=len(take),preference_change=float(np.mean([jensenshannon(x,y,base=2)**2 for x,y in zip(pb,pc)]))))
    d=pd.DataFrame(rows);f,main=integrated_team_fit(d,'retained_previous_minutes','preference_change','full_roster_continuity_primary');f.extend(integrated_team_fit(d,'retained_current_minutes','preference_change','current_minutes_retention_sensitivity')[0]);atomic_csv(d,out/'full_roster_2024_to_2025.csv');atomic_csv(pd.DataFrame(f),out/'primary_regressions.csv');integrated_scatter(d,'retained_previous_minutes','preference_change',out/'roster_continuity','4. Does roster continuity accompany stable passing preferences?')
    # Historical networks are all legacy >=10 MPG cohorts; do not claim full-roster coverage.
    historic=[];networks={};stats={};anonymous_features={}
    for file in sorted(ctx['root'].glob('202?-??/Regular_Season/*/passing_matrix.csv')):
        season=file.parts[-4];team=file.parts[-2];m=pd.read_csv(file,index_col=0);rs=pd.read_csv(file.parent/'roster_stats.csv').set_index('PLAYER_ID');mpg=rs.MIN/rs.GP
        historic.append(dict(season=season,team=team,players=len(rs),players_under10_mpg=int(mpg.lt(10).sum()),minimum_mpg=mpg.min(),scope='legacy rotation-only network'))
        networks[(season,team)]=(m.index.astype(int).tolist(),m.to_numpy(float));stats[(season,team)]=rs.MIN
        active=m.to_numpy(float).sum(0)+m.to_numpy(float).sum(1)>0;arr=m.to_numpy(float)[np.ix_(active,active)];Pr,valid=integrated_probability(arr);N=len(arr);pref=Pr.sum(0)/valid.sum()
        anonymous_features[(season,team)]=dict(season=season,team=team,receiving_preference_concentration=integrated_hhi(pref),first_choice_excess=(Pr[valid].max(1).mean()-1/(N-1))/(1-1/(N-1)),relative_reciprocity=(Pr*Pr.T).sum()/N*(N-1))

    transitions=[];excluded=[];seasons=sorted(set(s for s,t in networks))
    for previous,nextseason in zip(seasons,seasons[1:]):
        for team in ctx['season'].team:
            ids0,A=networks[(previous,team)];ids1,B=networks[(nextseason,team)];common=sorted(set(ids0)&set(ids1));take=list(range(len(common)));a=A[np.ix_([ids0.index(p) for p in common],[ids0.index(p) for p in common])];b=B[np.ix_([ids1.index(p) for p in common],[ids1.index(p) for p in common])]
            while True:
                aa=a[np.ix_(take,take)];bb=b[np.ix_(take,take)];keep=(aa.sum(1)>0)&(bb.sum(1)>0)
                if keep.all():break
                take=[i for i,k in zip(take,keep) if k]
            if len(take)<3:
                excluded.append(dict(team=team,transition=previous+'→'+nextseason,matched_players=len(take),reason='fewer than three matched positive-outgoing players'));continue
            pa,_=integrated_probability(aa);pb,_=integrated_probability(bb);transitions.append(dict(team=team,transition=previous+'→'+nextseason,retention=stats[(previous,team)].reindex(common).sum()/stats[(previous,team)].sum(),preference_change=float(np.mean([jensenshannon(x,y,base=2)**2 for x,y in zip(pa,pb)])),matched_players=len(take)))
    panel=pd.DataFrame(transitions);Z=pd.concat([pd.get_dummies(panel.team,drop_first=True,dtype=float),pd.get_dummies(panel.transition,drop_first=True,dtype=float)],axis=1);X=np.column_stack([np.ones(len(panel)),panel.retention,Z]);y=panel.preference_change.to_numpy();beta=np.linalg.lstsq(X,y,rcond=None)[0];inv=np.linalg.pinv(X.T@X);e=y-X@beta;meat=np.zeros((X.shape[1],X.shape[1]))
    for team in panel.team.unique():
        score=X[panel.team.eq(team)].T@e[panel.team.eq(team)];meat+=np.outer(score,score)
    G=panel.team.nunique();cov=G/(G-1)*(len(panel)-1)/(len(panel)-X.shape[1])*inv@meat@inv
    from scipy.stats import t
    secondary=pd.DataFrame([dict(model='legacy_rotation_only_team_and_transition_FE',n=len(panel),teams=G,coefficient=beta[1],standard_error=np.sqrt(cov[1,1]),p_value=2*t.sf(abs(beta[1]/np.sqrt(cov[1,1])),G-1))]);atomic_csv(pd.DataFrame(excluded),out/'legacy_excluded_transitions.csv');atomic_csv(pd.DataFrame(historic),out/'historical_roster_audit.csv');atomic_csv(panel,out/'legacy_rotation_transitions.csv');atomic_csv(secondary,out/'legacy_panel_sensitivity.csv')
    feature=pd.DataFrame(list(anonymous_features.values())).set_index(['season','team']);cols=['receiving_preference_concentration','first_choice_excess','relative_reciprocity'];scale=feature[cols].std(ddof=0);anonymous=[]
    for previous,nextseason in zip(seasons,seasons[1:]):
        for team in ctx['season'].team:
            common=set(networks[(previous,team)][0])&set(networks[(nextseason,team)][0]);retention=stats[(previous,team)].reindex(sorted(common)).sum()/stats[(previous,team)].sum();delta=(feature.loc[(nextseason,team),cols]-feature.loc[(previous,team),cols])/scale
            anonymous.append(dict(team=team,transition=previous+'→'+nextseason,retention=retention,anonymous_structure_change=float(np.sqrt(np.mean(delta.to_numpy(float)**2)))))
    anonymous=pd.DataFrame(anonymous);Z=pd.concat([pd.get_dummies(anonymous.team,drop_first=True,dtype=float),pd.get_dummies(anonymous.transition,drop_first=True,dtype=float)],axis=1);X=np.column_stack([np.ones(len(anonymous)),anonymous.retention,Z]);y=anonymous.anonymous_structure_change.to_numpy();beta=np.linalg.lstsq(X,y,rcond=None)[0];inv=np.linalg.pinv(X.T@X);e=y-X@beta;meat=np.zeros((X.shape[1],X.shape[1]));G=anonymous.team.nunique()
    for team in anonymous.team.unique():
        score=X[anonymous.team.eq(team)].T@e[anonymous.team.eq(team)];meat+=np.outer(score,score)
    cov=G/(G-1)*(len(anonymous)-1)/(len(anonymous)-X.shape[1])*inv@meat@inv;anonfit=pd.DataFrame([dict(model='legacy_rotation_anonymous_structure_team_transition_FE',n=len(anonymous),coefficient=beta[1],standard_error=np.sqrt(cov[1,1]),p_value=2*t.sf(abs(beta[1]/np.sqrt(cov[1,1])),G-1))])
    atomic_csv(feature.reset_index(),out/'legacy_anonymous_features.csv');atomic_csv(anonymous,out/'legacy_anonymous_transitions.csv');atomic_csv(anonfit,out/'legacy_anonymous_regression.csv')
    for values,outcome,label in [(panel,'preference_change','legacy_retained_partnerships'),(anonymous,'anonymous_structure_change','legacy_anonymous_shape')]:
        controls=np.column_stack([np.ones(len(values)),pd.get_dummies(values.team,drop_first=True,dtype=float),pd.get_dummies(values.transition,drop_first=True,dtype=float)]);x=values.retention.to_numpy();y=values[outcome].to_numpy();plot=pd.DataFrame(dict(team=values.team,x_residual=x-controls@np.linalg.lstsq(controls,x,rcond=None)[0],y_residual=y-controls@np.linalg.lstsq(controls,y,rcond=None)[0]))
        atomic_csv(plot,out/(label+'_plot_data.csv'));integrated_scatter(plot,'x_residual','y_residual',out/label,'4. Historical rotation-only continuity (legacy >=10 MPG cohort)','Retention after team/year baselines','Network change after team/year baselines')

    report=['## 4. Team identity and roster continuity',integrated_table(pd.DataFrame(f).loc[lambda z:z.term.eq(z.x)]),integrated_table(secondary),integrated_table(anonfit),
        'Anonymous historical comparison also retains all 150 transitions, including Utah: RMS standardized changes in incoming preference concentration, first-choice excess over a uniform-option baseline, and reciprocity relative to uniform passing. Features do not use player identities. Scaling uses the full historical descriptive sample, not a forecast. This is a secondary legacy-rotation-only association, with team/transition FE and team-clustered errors; three feature choices were specified as an exploratory extension. Full-roster anonymous historical networks are unavailable for omitted low-minute players.',
        'Primary 30-team comparison uses full-roster official logs and all saved retained-player profiles for 2024–25→2025–26. Retention is the share of previous minutes supplied by returning team-player IDs. Preferences compared only on an identical retained-player positive-outgoing induced panel. This isolates retained partnerships, not whole-roster structural similarity. Receiver availability, sample lengths and role changes can still confound.',
        'Historical audit: all 180 legacy networks contain only players averaging >=10 minutes/game. They are NOT pooled as full-roster observations. A clearly labeled secondary rotation-only panel has up to 150 year-to-year transitions, team and transition fixed effects, team-clustered t inference. No new minute cutoff applied to the primary study. Raw full-roster historical profiles are not available for all omitted players; no missing passes invented.']
    return dict(idea=4,test='Full-roster minutes retention vs retained-player preference change',n=main['n'],effect=main['coefficient'],r=main['pearson_r'],r_squared=main['r_squared'],p_value=main['p_value_hc3']),report


def integrated_study_5(ctx,out):
    from scipy.stats import spearmanr
    parent=ctx['parent'];edges=pd.read_csv(parent/'assist_network_ideas/assist_edges.csv');players=ctx['players'];rows=[];teams=[]
    for _,r in ctx['season'].iterrows():
        ids,A=ctx['networks'][r.team];pos={p:i for i,p in enumerate(ids)};B=np.zeros_like(A)
        for _,v in edges.loc[edges.team.eq(r.team)].iterrows():B[pos[int(v.assister_id)],pos[int(v.scorer_id)]]=v.assists
        P,valid=integrated_probability(A);pref=P.sum(0)/valid.sum();_,scores,_,_=eigenvector_passing_concentration(A,ids);stats=players.loc[players.TEAM_ID.eq(r.team_id)].groupby('PLAYER_ID').agg(name=('PLAYER_NAME','first'),minutes=('MIN','sum'))
        supply=A.sum(1)/A.sum();receive=A.sum(0)/A.sum();create=B.sum(1)/B.sum();finish=B.sum(0)/B.sum();ec=np.array([scores.get(pid,0) for pid in ids])
        for i,pid in enumerate(ids):rows.append(dict(team=r.team,player_id=pid,name=stats.loc[pid,'name'],passing_style=r.passing_style,minutes=stats.loc[pid,'minutes'],pass_out_share=supply[i],pass_in_share=receive[i],preference_in_share=pref[i],passing_ec_share=ec[i],assist_creator_share=create[i],assisted_scorer_share=finish[i],creation_enrichment=create[i]-supply[i],finishing_enrichment=finish[i]-receive[i]))
        teams.append(dict(team=r.team,passing_hub=stats.loc[ids[int(np.argmax(ec))],'name'],assist_creator=stats.loc[ids[int(np.argmax(create))],'name'],assisted_scorer=stats.loc[ids[int(np.argmax(finish))],'name'],same_passing_hub_creator=bool(np.argmax(ec)==np.argmax(create)),same_creator_finisher=bool(np.argmax(create)==np.argmax(finish)),pass_assist_creator_rank_r=spearmanr(supply,create).statistic,ec_creator_rank_r=spearmanr(ec,create).statistic,preference_finisher_rank_r=spearmanr(pref,finish).statistic))
    d=pd.DataFrame(rows);team=pd.DataFrame(teams);atomic_csv(d,out/'player_circulation_creation_roles.csv');atomic_csv(team,out/'team_role_overlap.csv');integrated_scatter(d,'pass_out_share','assist_creator_share',out/'circulation_creation','5. Passing circulation versus credited assisted creation')
    report=['## 5. Circulation and assisted creation',integrated_table(team),integrated_table(d.sort_values('creation_enrichment',ascending=False).head(12)[['team','name','pass_out_share','assist_creator_share','creation_enrichment']]),
        'All 30 2025–26 directed assist matrices are rebuilt from audited saved AST edges, distinct from all passes. Assist-provider shares and assisted-finisher shares compare different roles; differences from pass shares measure relative enrichment, not causal skill. Existing AST allocation deficits (one to nine assists in ten teams) remain documented. Assists omit missed opportunities and unassisted scoring. Raw passing EC is combined-involvement A+A.T for continuity with earlier analysis; receiving preference keeps direction. This is descriptive role mapping, with no new efficiency regression or causal role labels.']
    return None,report


def integrated_warriors_inputs(ctx):
    root=ctx['warriors'];base=pd.read_csv(root/'role_checks/game_metrics.csv');nets={};rosters={};ids_all=set();names={}
    for _,r in base.iterrows():
        g=int(r.game_number);f=pd.read_csv(root/'eigenvector_team_performance'/f'game_{g:02d}'/'passing_matrix.csv',index_col=0);ids=f.index.astype(int).tolist();nets[g]=(ids,f.to_numpy(float));ids_all.update(ids);pg=pd.read_csv(root/f'game_{g:02d}'/'game_players.csv');pg=pg.loc[pg.MIN.gt(0)].set_index('PLAYER_ID');rosters[g]=pg;names.update(pg.PLAYER_NAME.to_dict())
    ids_all=sorted(ids_all);pos={p:i for i,p in enumerate(ids_all)};arrays={}
    for g,(ids,A) in nets.items():
        B=np.zeros((len(ids_all),len(ids_all)));take=[pos[p] for p in ids];B[np.ix_(take,take)]=A;arrays[g]=B
    return base,ids_all,pos,arrays,rosters,names


def integrated_study_6(ctx,out):
    base,ids,pos,nets,rosters,names=integrated_warriors_inputs(ctx);complete=set(base.loc[base.full_observed_network.eq(True),'game_number'].astype(int));rows=[]
    for g in sorted(complete):
        if g-1 not in complete:continue
        A0=nets[g-1];A1=nets[g];O0=A0.sum(1);O1=A1.sum(1);P0,v0=integrated_probability(A0);P1,v1=integrated_probability(A1);common=v0&v1;enter=~v0&v1;exit=v0&~v1
        volume=np.sum(.5*(P0[common]+P1[common])*(O1[common]-O0[common])[:,None],axis=0);preference=np.sum(.5*(O0[common]+O1[common])[:,None]*(P1[common]-P0[common]),axis=0);activation=A1[enter].sum(0)-A0[exit].sum(0);delta=A1.sum(0)-A0.sum(0)
        if not np.allclose(volume+preference+activation,delta):raise ValueError('Receiving-count decomposition failed')
        for j,pid in enumerate(ids):
            rows.append(dict(game_number=g,previous_game_number=g-1,player_id=pid,name=names[pid],common_receiver=pid in rosters[g].index and pid in rosters[g-1].index,received_before=A0[:,j].sum(),received_after=A1[:,j].sum(),change=delta[j],supplier_volume_component=volume[j],supplier_preference_component=preference[j],pass_source_activation_component=activation[j]))
    d=pd.DataFrame(rows);common=d.loc[d.common_receiver];components=['supplier_volume_component','supplier_preference_component','pass_source_activation_component'];totals=np.array([common[c].abs().sum() for c in components]);summary=pd.DataFrame(dict(component=components,absolute_magnitude=totals,share_of_total_absolute_components=totals/totals.sum()))
    atomic_csv(d,out/'receiving_change_decomposition.csv');atomic_csv(summary,out/'component_magnitude_summary.csv');atomic_csv(common.reindex(common.change.abs().sort_values(ascending=False).index).head(30),out/'largest_receiving_changes.csv')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(11,7));ax.bar(['Supplier volume','Supplier preference','Pass-active source entry/exit'],100*summary.share_of_total_absolute_components,color=['#2473b8','#e18a24','#8e9aa6']);ax.set(ylabel='Share of absolute component magnitudes (%)',title='6. Why do players receive different numbers of passes?\nCommon receivers, fully observed consecutive Warriors games');fig.tight_layout();fig.savefig(out/'receiving_decomposition.pdf');fig.savefig(out/'receiving_decomposition.png',dpi=140);plt.close(fig)
    report=['## 6. Receiving changes: volume versus preference',integrated_table(summary),integrated_table(common.reindex(common.change.abs().sort_values(ascending=False).index).head(12)[['game_number','name','change',*components]]),
        'For each consecutive fully observed game pair, returning pass-active sources satisfy delta(O*P)=mean(P)*delta(O)+mean(O)*delta(P). Sources with zero outgoing passes in one game have undefined preferences there; their entry/exit is assigned to a separate activation component, not fabricated as a preference change. Components reconcile every receiver’s exact pass-count change. Summary uses receivers participating in both games. Absolute-magnitude percentages are NOT shares of net changes or variance explained; components can cancel. Preference component also includes changes in other available receivers and game roles. This is accounting, not an independent regression or causal mechanism test.']
    return None,report


def integrated_study_7(ctx,out):
    from scipy.spatial.distance import jensenshannon
    base,ids,pos,nets,rosters,names=integrated_warriors_inputs(ctx);complete=sorted(base.loc[base.full_observed_network.eq(True),'game_number'].astype(int));models=['historical_minutes','historical_receiving','personal_pooled_counts','personal_equal_game_probability','personal_rolling5_probability'];lossrows=[];predrows=[];prior_weight=.1
    for number,g in enumerate(complete):
        prior=complete[:number]
        if len(prior)<10:continue
        history=np.sum([nets[h] for h in prior],axis=0);mean_minutes={}
        for pid in rosters[g].index:
            values=[float(rosters[h].loc[pid,'MIN']) for h in prior if pid in rosters[h].index];mean_minutes[pid]=np.mean(values) if values else np.mean([v for pg in [rosters[h] for h in prior] for v in pg.MIN])
        for source in rosters[g].index:
            i=pos[source];training=[h for h in prior if nets[h][i].sum()>0]
            if len(training)<3 or nets[g][i].sum()<=0:continue
            targets=[p for p in rosters[g].index if p!=source];take=[pos[p] for p in targets];raw=nets[g][i,take];observed=raw/raw.sum();minutes=np.array([mean_minutes[p] for p in targets]);q0=minutes/minutes.sum()
            def shrink(v):
                v=np.asarray(v,float);return (1-prior_weight)*(v/v.sum())+prior_weight*q0 if v.sum()>0 else q0.copy()
            pooled=np.sum([nets[h][i] for h in training],axis=0);average=np.mean([nets[h][i]/nets[h][i].sum() for h in training],axis=0);rolling=np.mean([nets[h][i]/nets[h][i].sum() for h in training[-5:]],axis=0)
            Q={'historical_minutes':q0,'historical_receiving':shrink(history.sum(0)[take]),'personal_pooled_counts':shrink(pooled[take]),'personal_equal_game_probability':shrink(average[take]),'personal_rolling5_probability':shrink(rolling[take])}
            for label,q in Q.items():
                lossrows.append(dict(game_number=g,source_id=source,name=names[source],model=label,js_loss=float(jensenshannon(observed,q,base=2)**2),brier_loss=float(np.sum((observed-q)**2)),source_passes=raw.sum(),prior_source_games=len(training),latest_history_game=max(prior)))
            for j,target in enumerate(targets):predrows.append(dict(game_number=g,source_id=source,target_id=target,observed_probability=observed[j],**{label:Q[label][j] for label in models}))
        if number%15==0:print('Chronological forecast through game',g,flush=True)
    loss=pd.DataFrame(lossrows);bygame=loss.groupby(['game_number','model']).agg(js_loss=('js_loss','mean'),brier_loss=('brier_loss','mean'),sources=('source_id','size')).reset_index();summary=bygame.groupby('model').agg(test_games=('game_number','size'),mean_game_js=('js_loss','mean'),mean_game_brier=('brier_loss','mean')).reset_index();wide=bygame.pivot(index='game_number',columns='model',values='js_loss');comparisons=[]
    for baseline in ['historical_minutes','historical_receiving','personal_equal_game_probability','personal_rolling5_probability']:
        d=(wide.personal_pooled_counts-wide[baseline]).rename('loss_difference').reset_index();fit,_=warriors_role_hac(d,'loss_difference',[],'pooled_vs_'+baseline);r=fit[0];comparisons.append(dict(comparison='personal_pooled_counts minus '+baseline,n=len(d),mean_loss_difference=r['coefficient'],p_value=r['p_value_hac7']))
    compare=pd.DataFrame(comparisons);main=compare.iloc[0]
    reliability=[]
    for variant in ['sources_at_least_10_passes','pass_count_weighted_scoring']:
        z=loss.loc[loss.source_passes.ge(10)].copy() if variant=='sources_at_least_10_passes' else loss.copy()
        scores=[]
        for (g,label),v in z.groupby(['game_number','model']):scores.append(dict(game_number=g,model=label,js_loss=np.average(v.js_loss,weights=v.source_passes) if variant=='pass_count_weighted_scoring' else v.js_loss.mean()))
        score=pd.DataFrame(scores);w=score.pivot(index='game_number',columns='model',values='js_loss')
        diff=(w.personal_pooled_counts-w.historical_minutes).rename('difference').reset_index();rf,_=warriors_role_hac(diff,'difference',[],'forecast_reliability');reliability.append(dict(variant=variant,test_games=len(w),minutes_mean_js=w.historical_minutes.mean(),personal_pooled_mean_js=w.personal_pooled_counts.mean(),mean_difference=rf[0]['coefficient'],p_value=rf[0]['p_value_hac7']))
    atomic_csv(pd.DataFrame(reliability),out/'forecast_reliability_sensitivity.csv')
    for d,name in [(loss,'source_forecast_losses'),(bygame,'game_forecast_losses'),(summary,'forecast_model_summary'),(compare,'forecast_comparisons'),(pd.DataFrame(predrows),'forecast_edge_probabilities')]:atomic_csv(d,out/(name+'.csv'))
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(12,7));labels={'historical_minutes':'Past minutes','historical_receiving':'Past receiving\npopularity','personal_equal_game_probability':'Personal preferences\nequal-game average','personal_pooled_counts':'Personal preferences\npooled passes','personal_rolling5_probability':'Personal preferences\nlast five games'};bars=ax.bar([labels[x] for x in summary.model],summary.mean_game_js,color='#2473b8');ax.bar_label(bars,fmt='%.4f',padding=3);ax.set(ylabel='Mean next-game preference error (JS; lower = better)',title='7. Predicting Warriors passing preferences\n66 roster-conditional test games; earlier data only');fig.tight_layout();fig.savefig(out/'forecast_comparison.pdf');fig.savefig(out/'forecast_comparison.png',dpi=140);plt.close(fig)
    report=['## 7. Predicting the next observed passing network',integrated_table(summary),integrated_table(compare),
        'Forward chronological evaluation on fully observed Warriors games after >=10 earlier complete games; each evaluated source needs >=3 earlier positive-outgoing observations. Every prediction uses only earlier pass counts/profiles and earlier positive-minute appearances. Current actual participant list is supplied equally to ALL models: this is a roster-conditional benchmark, not unconditional pregame forecasting. Current minutes, shots and passes do not enter predictions; current passing determines only evaluation outcomes/eligibility. Newly seen receivers get the mean past player-minute prior.',
        'Baselines: previous mean receiver minutes, and pooled previous receiver popularity. Personal models: pooled raw counts (volume-weighted history), equal-game average row probabilities, last-five-source-game average probabilities. These are genuinely distinct multi-game estimators; normalizing a single count matrix twice would be identical. All non-minute profiles use fixed 10% historical-minutes shrinkage, chosen before evaluation, not tuned on target losses. JS and Brier losses evaluated per source then averaged equally per test game. Main inferential contrast is pooled-personal JS minus minutes-baseline JS, HAC7 using actual game gaps. Secondary comparisons unadjusted. Evaluation sensitivity omits source rows with fewer than ten actual passes or weights errors by actual pass counts; this changes scoring only, not predictions. This is an exploratory retrospective protocol after prior data exploration, not untouched external validation or demonstrated game outcome prediction.']
    return dict(idea=7,test='Chronological personal-preference forecast error minus minutes baseline',n=int(main['n']),effect=main['mean_loss_difference'],p_value=main['p_value']),report


def integrated_study_8(ctx,out):
    from scipy.stats import ttest_rel
    parent=ctx['parent'];comparison=pd.read_csv(parent/'six_network_topics/playoff_comparison.csv');rows=[];rng=np.random.default_rng(ctx['seed']+800)
    for _,r in comparison.loc[comparison.complete.eq(True)].iterrows():
        folder=parent/'six_network_topics/matched_playoff_networks'/r.team;f=pd.read_csv(folder/'regular_matrix.csv',index_col=0);g=pd.read_csv(folder/'playoff_matrix.csv',index_col=0);ids=f.index.astype(int).tolist();A=f.to_numpy(float);B=g.to_numpy(float)
        if g.index.astype(int).tolist()!=ids:raise ValueError('Playoff panel mismatch')
        Pa,va=integrated_probability(A);Pb,vb=integrated_probability(B);donors=va&vb;qa=Pa[donors].mean(0);qb=Pb[donors].mean(0);ec0,_,_,_=eigenvector_passing_concentration(A,ids);ec1,_,_,_=eigenvector_passing_concentration(B,ids)
        # Equal precision per source, sampled independently without replacement in both periods.
        sampling=int(300);preference_draws=[]
        for matrix in [A,B]:
            Q=np.zeros((sampling,len(ids)))
            for i in np.flatnonzero(donors):
                count=int(min(A[i].sum(),B[i].sum()));draw=rng.multivariate_hypergeometric(matrix[i].astype(np.int64),count,size=sampling);Q+=draw/count/donors.sum()
            h=(np.sum(Q*Q,axis=1)-1/len(ids))/(1-1/len(ids));preference_draws.append(h)
        rows.append(dict(team=r.team,passing_style=r.passing_style,matched_players=len(ids),matched_donors=int(donors.sum()),regular_preference_concentration=integrated_hhi(qa),playoff_preference_concentration=integrated_hhi(qb),preference_change=integrated_hhi(qb)-integrated_hhi(qa),
            regular_ec_concentration=ec0['eigenvector_concentration'],playoff_ec_concentration=ec1['eigenvector_concentration'],ec_change=ec1['eigenvector_concentration']-ec0['eigenvector_concentration'],minutes_change=r.minutes_concentration_change,
            equal_source_count_preference_change=float((preference_draws[1]-preference_draws[0]).mean())))
    d=pd.DataFrame(rows);test=ttest_rel(d.playoff_preference_concentration,d.regular_preference_concentration);fits=[]
    for x,y,label in [('minutes_change','ec_change','rotation_vs_count_EC_change'),('minutes_change','preference_change','rotation_vs_probability_change')]:fits.extend(integrated_team_fit(d,x,y,label)[0])
    equaltest=ttest_rel(d.equal_source_count_preference_change,np.zeros(len(d)));summary=pd.DataFrame([dict(model='paired_probability_concentration_primary',n=len(d),mean_change=d.preference_change.mean(),p_value=test.pvalue),dict(model='equal_source_sample_count_sensitivity',n=len(d),mean_change=d.equal_source_count_preference_change.mean(),p_value=equaltest.pvalue)])
    atomic_csv(d,out/'playoff_structure_changes.csv');atomic_csv(pd.DataFrame(fits),out/'rotation_change_regressions.csv');atomic_csv(summary,out/'paired_tests.csv');integrated_scatter(d,'minutes_change','ec_change',out/'playoff_rotation','8. Playoff rotation tightening and count-weighted centralization')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(11,8))
    for _,r in d.iterrows():ax.plot([0,1],[r.regular_preference_concentration,r.playoff_preference_concentration],marker='o',lw=1,color='#2473b8',alpha=.6);ax.annotate(r.team,(1,r.playoff_preference_concentration),xytext=(4,0),textcoords='offset points',fontsize=8)
    ax.set(xticks=[0,1],xticklabels=['Regular season','Playoffs'],ylabel='Equal-source receiving-preference concentration',title=f'8. Does playoff concentration remain after probability weighting?\nMatched players and donors; paired p={test.pvalue:.4g}');ax.set_xlim(-.1,1.2);fig.tight_layout();season_topics_spread_labels(fig);fig.savefig(out/'playoff_preferences.pdf');fig.savefig(out/'playoff_preferences.png',dpi=140);plt.close(fig)
    report=['## 8. Playoffs: rotation changes or preference changes?',integrated_table(summary),integrated_table(pd.DataFrame(fits).loc[lambda z:z.term.eq(z.x)]),
        'All 16 matched 2026 playoff teams. Count EC uses saved identical active player panels. Probability concentration uses incoming probabilities averaged across the SAME positive-outgoing donors in both contexts; recipient panel also fixed. Minutes changes are saved matched-player minute concentration differences. Primary paired t test is the mean change in probability concentration; this does not isolate causality or fully control rotation/role effects. Tournament teams are dependent opponents.',
        'Finite-count sensitivity equalizes each donor’s observed pass sample size between regular/playoff contexts using 300 hypergeometric draws, then averages the concentration difference. It is not a game bootstrap or sampling confidence interval. Probability weighting removes donor volume weighting but does not remove selection of recipients, shared court time or tactical roles. Count-EC/minutes association and remaining probability changes are complementary descriptions, not a causal decomposition.']
    return dict(idea=8,test='Paired playoff change in matched-donor probability concentration',n=len(d),effect=d.preference_change.mean(),p_value=test.pvalue),report


def run_integrated_network_study(argv):
    import json
    from datetime import datetime,timezone
    parser=argparse.ArgumentParser(description='Eight integrated opportunity, preference, replacement and continuity studies')
    parser.add_argument('--integrated-network-study',action='store_true');parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT_DIR);parser.add_argument('--only',type=int,nargs='+',choices=range(1,9));parser.add_argument('--null-draws',type=int,default=499);parser.add_argument('--seed',type=int,default=20261111)
    args=parser.parse_args(argv)
    if args.null_draws<99:raise ValueError('Use at least 99 null draws; default is 499')
    root=args.output_dir.expanduser();parent=root/'2025-26/team_season_topics';out=root/'integrated_network_study';out.mkdir(parents=True,exist_ok=True);season=pd.read_csv(parent/'team_season_metrics.csv');players=pd.read_csv(parent/'player_games.csv',dtype={'GAME_ID':str});networks={}
    for team in season.team:
        f=pd.read_csv(parent/'networks'/team/'passing_matrix.csv',index_col=0);networks[team]=(f.index.astype(int).tolist(),f.to_numpy(float))
    ctx=dict(root=root,parent=parent,warriors=root/'GSW_2021-22_82_games',season=season,players=players,networks=networks,styles=season.set_index('team').passing_style.to_dict(),seed=args.seed,null_draws=args.null_draws)
    studies={1:integrated_study_1,2:integrated_study_2,3:integrated_study_3,4:integrated_study_4,5:integrated_study_5,6:integrated_study_6,7:integrated_study_7,8:integrated_study_8}
    for k in args.only or list(studies):
        folder=out/f'study_{k}';folder.mkdir(parents=True,exist_ok=True);print(f'Starting integrated study {k}/8',flush=True);primary,report=studies[k](ctx,folder)
        if primary is not None:
            primary={key:(value.item() if isinstance(value,np.generic) else value) for key,value in primary.items()};primary={key:(None if isinstance(value,float) and not np.isfinite(value) else value) for key,value in primary.items()};atomic_json(primary,folder/'primary.json')
        (folder/'ANALYSIS.md').write_text('\n\n'.join(report));print(f'Completed integrated study {k}/8'+(' '+str(primary) if primary else ' (descriptive)'),flush=True)
    primary=[]
    for k in studies:
        path=out/f'study_{k}/primary.json'
        if path.exists():primary.append(json.loads(path.read_text()))
    results=pd.DataFrame(primary)
    if len(results):
        results['p_holm_primary_family']=np.nan;running=0
        for rank,i in enumerate(results.p_value.sort_values().index):running=max(running,min(1,float(results.loc[i,'p_value'])*(len(results)-rank)));results.loc[i,'p_holm_primary_family']=running
        atomic_csv(results,out/'primary_results.csv');print(results.to_string(index=False),flush=True)
    intro=['# Integrated passing-network research: eight studies',
        'All eight designs retain earlier analyses and CLI. Saved 2025–26 full-roster networks, 2021–22 Warriors games, matched 2026 playoffs, and audited historical networks. No new NBA downloads or soccer results. This is exploratory work following extensive previous topic selection.',
        'Six primary inferential tests (studies 1,2,3,4,7,8), Holm-corrected together when all studies are present. Studies 5 and 6 are descriptive/accounting analyses. Sensitivity p-values are unadjusted; this family correction does not cover all earlier research. Player, team, repeated-season and game-level observations require different inference; model-specific methods documented below. Raw and probability weight comparisons must distinguish genuinely different estimators from row-normalization invariance.',
        'Data audit: all 180 older league-season networks use legacy >=10 MPG cohorts. They appear only in a labeled rotation-only sensitivity. The main continuity test uses full-roster 2024–25→2025–26 official logs and retained-player profiles, without a new minute cutoff. Primary Warriors analyses use 76 complete networks from 81 valid games; forecast/decomposition samples require their own time/exposure eligibility. Older seven-game-series files also use a legacy 10-minute filter and are not pooled into these studies.',
        integrated_table(results) if len(results) else 'Primary studies not yet run.']
    for k in studies:
        path=out/f'study_{k}/ANALYSIS.md'
        if path.exists():intro.append(path.read_text())
    (out/'DETAILED_ANALYSIS.md').write_text('\n\n'.join(intro))
    required=[out/f'study_{k}/ANALYSIS.md' for k in studies]
    if all(p.exists() for p in required):
        cv=pd.read_csv(out/'study_1/prediction_summary.csv').set_index('model');replacement=pd.read_csv(out/'study_2/team_replacements.csv');null_d=pd.read_csv(out/'study_3/null_diagnostics.csv');legacy=pd.read_csv(out/'study_4/legacy_panel_sensitivity.csv');anonymous_result=pd.read_csv(out/'study_4/legacy_anonymous_regression.csv');roles=pd.read_csv(out/'study_5/team_role_overlap.csv');components=pd.read_csv(out/'study_6/component_magnitude_summary.csv');forecast=pd.read_csv(out/'study_7/forecast_model_summary.csv').set_index('model');playoff=pd.read_csv(out/'study_8/playoff_structure_changes.csv');family=results.set_index('idea')
        shot_reduction=100*(1-cv.loc['minutes_raw_preference','pooled_rmse']/cv.loc['minutes_raw','pooled_rmse']);forecast_reduction=100*(1-forecast.loc['personal_pooled_counts','mean_game_js']/forecast.loc['historical_minutes','mean_game_js']);popularity_reduction=100*(1-forecast.loc['personal_pooled_counts','mean_game_js']/forecast.loc['historical_receiving','mean_game_js'])
        explained=['# Eight integrated studies: results explained',
            'Completed using saved NBA data. These studies combine season networks, player statistics, actual hub absences, chronological Warriors games and matched playoffs. The purpose is to identify useful research questions and test whether their measures add information, not to select only strong lines. Six primary inferential tests are corrected together; studies 5 and 6 are descriptive. All results remain exploratory after extensive previous topic exploration.',
            '## 1. Do passing preferences add information beyond minutes?',
            integrated_table(cv.reset_index()[['model','pooled_rmse','cv_r_squared']]),
            f'Minutes alone account for much of between-player shooting allocation. Adding raw receiving share improves the held-out-team model; adding receiving preference to both produces another {shot_reduction:.1f}% RMSE reduction. That is a real but modest increment, not a claim that preference alone explains 92.5% of shooting. The full model’s preference coefficient passes the clustered test (Holm-family p={family.loc[1,"p_holm_primary_family"]:.4g}). Its sign is negative after controlling raw receptions, whereas the minutes-plus-preference coefficient is positive. The two measures overlap; this conditional sign should not be interpreted as a causal effect of being preferred. Models explain same-season shot shares, not future scoring efficiency.',
            '## 2. Does a similar backup make hub absence less disruptive?',
            f'The primary relationship points in the expected direction: r={family.loc[2,"r"]:.3f}, R²={100*family.loc[2,"r_squared"]:.1f}%, corrected p={family.loc[2,"p_holm_primary_family"]:.3f}. Teams with a similar passing-role candidate tend toward less disruption, but evidence is uncertain. The association weakens substantially after accounting for candidate availability, changes in minutes distribution and pass-count exposure. Offensive-loss evidence is also uncertain. This is not yet a strong main result.',
            '## 3. Are partnerships stronger than activity alone predicts?',
            f'Observed Warriors mutual-preference mass is {null_d.observed_reciprocity.mean():.5f}, compared with null mean {null_d.null_mean.mean():.5f}: about {100*(null_d.observed_reciprocity.mean()/null_d.null_mean.mean()-1):.1f}% higher. The one-sided Monte Carlo p is {family.loc[3,"p_value"]:.4g} and family-corrected p={family.loc[3,"p_holm_primary_family"]:.4g}. Each randomized network keeps each player’s exact pass-supplying and pass-receiving totals and forbids self-passes. The result therefore supports extra partnership organization beyond those activity totals. It does not establish intentional tactics or actual pass sequences. Simulation resolution is finite; per-pair excesses are descriptive. This is a promising network-science topic.',
            '## 4. Does roster continuity preserve passing identity?',
            f'The main full-roster 30-team comparison does not show a clear relationship: r={family.loc[4,"r"]:.3f}, corrected p={family.loc[4,"p_holm_primary_family"]:.3f}. A separate historical rotation-only panel has {int(legacy.n.iloc[0])} transitions and a negative within-team/year coefficient ({legacy.coefficient.iloc[0]:.4f}, unadjusted team-clustered p={legacy.p_value.iloc[0]:.4g}), consistent with continuity accompanying stability within those cohorts. These samples differ in years, coverage and analysis design, so this is not a full-roster replication. All 180 older season networks have a legacy >=10 MPG selection. One transition, Utah 2021–22→2022–23, has fewer than three matched positive-outgoing players and is excluded; the other 149 are retained.',
            f'The historical rotation-only dataset also supports an anonymous shape comparison across all 150 transitions: coefficient {anonymous_result.coefficient.iloc[0]:.4f}, unadjusted team-clustered p={anonymous_result.p_value.iloc[0]:.4g}. This ignores player names and compares receiving concentration, first-choice concentration and reciprocity. More retention accompanies more stable anonymous shape within those cohorts. Both historical findings remain secondary and limited to the inherited rotation-only data; they do not establish the same result for full rosters.',
            '## 5. Are circulators also creators and finishers?',
            f'The highest combined-passing-eigenvector player is also the leading assist provider in {int(roles.same_passing_hub_creator.sum())}/30 teams. The leading assist provider is also the main assisted finisher in only {int(roles.same_creator_finisher.sum())}/30. Thus circulation and final assisted creation often share a leader, while creation and finishing are usually split across players. The average within-team rank correlation between raw outgoing passing share and assist-provider share is {roles.pass_assist_creator_rank_r.mean():.3f}. This is role overlap, not efficiency or skill evidence; assisted scoring omits missed opportunities and unassisted baskets.',
            '## 6. Why do receiving counts change?',
            integrated_table(components),
            'Across common receivers in complete consecutive Warriors games, roughly 54.7% of absolute component magnitude comes from changing supplier preferences, 24.4% from changing supplier pass volumes and 20.9% from pass-active source entry/exit. Every observed receiving-count change reconciles exactly. These percentages describe absolute accounting components, not variance explained or percentages of net change: components can offset one another. Source activation includes a player moving between zero and positive outgoing passes, not just substitutions or roster arrivals. Preference changes can also reflect receiver availability.',
            '## 7. Do preferences predict the next game?',
            integrated_table(forecast.reset_index()),
            f'Personal pooled passing history reduces JS prediction error by {forecast_reduction:.1f}% versus historical receiver minutes and {popularity_reduction:.1f}% versus historical receiving popularity. The primary advantage survives the six-test correction (p={family.loc[7,"p_holm_primary_family"]:.4g}) and low-count/pass-weighted scoring sensitivities. Equal-game probability averaging is nearly as good; the raw difference from pooled history is not clear. A last-five-game window has higher mean error than all-history profiles. This is the clearest practical result in this suite.',
            'The participant list of the target game is supplied to all methods, so the result is conditional on knowing who participates. Actual target-game minutes and passing counts do not enter any prediction. Predictions use earlier complete games only, with an initial history requirement and fixed, untuned 10% historical-minute shrinkage. This is a retrospective, roster-conditional forecast of passing preferences, not a prediction of wins, and not external validation on untouched data.',
            '## 8. Do playoff changes reflect rotation or preferences?',
            f'Across 16 matched teams, changes in minutes concentration and count-weighted eigenvector concentration are strongly associated (descriptive r≈0.867, R²≈75.2%; supplementary HC3 p≈0.000014). Average count concentration rises by {playoff.ec_change.mean():.4f}. However, equal-source receiving-preference concentration has a much smaller mean change ({playoff.preference_change.mean():.4f}) and no clear paired shift after correction (p={family.loc[8,"p_holm_primary_family"]:.3f}). The equal-per-source pass-count sensitivity is also uncertain. This is consistent with rotation/opportunity changes accounting for an important part of count-based centralization, but it is not a causal decomposition. Unchanged mean concentration can hide substantial changes in specific passing connections.',
            '## Most useful research directions',
            'For a predictive project, study 7 offers a concrete task, chronological evaluation and meaningful baselines: “Can individual passing preferences predict later passing distributions beyond playing time and receiver popularity?” For network organization, study 3 offers a constrained null model: “Are passing partnerships more reciprocal than player activity alone predicts?” For regression, study 1 shows how minutes, volume and preference jointly describe shooting responsibility, with a modest incremental role for probability weighting.',
            'All graphs, full updated code, CSVs, audit results and simulation caches are included. Source data are saved official NBA inputs; no soccer results or new downloads are claimed.'
        ]
        (out/'RESULTS_EXPLAINED.md').write_text('\n\n'.join(explained))

    # Scientific figure collection; original vector PDFs also retained in each study folder.
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    figures=sorted(out.glob('study_*/*.png'))
    if figures:
        with PdfPages(out/'all_study_graphs.pdf') as pdf:
            for file in figures:
                fig,ax=plt.subplots(figsize=(12,9));ax.imshow(plt.imread(file));ax.axis('off');fig.tight_layout(pad=.05);pdf.savefig(fig);plt.close(fig)
    atomic_json(dict(created_at=datetime.now(timezone.utc).isoformat(),studies_requested=args.only or list(studies),primary_tests_available=len(results),null_draws=args.null_draws,seed=args.seed,
        primary_data_scope='2025-26 full-roster NBA teams, saved 2021-22 Warriors games, matched 2026 playoffs',historical_scope='Legacy >=10 MPG rotation-only networks used only in labeled sensitivity',no_new_downloads=True),out/'run_manifest.json')
    print('Saved integrated network study: '+str(out),flush=True);return 0



# ---------------------------------------------------------------------
# ORIGINAL 180 TEAM-SEASONS: SIMILAR OR COMPLEMENTARY PARTNERSHIPS?
# ---------------------------------------------------------------------
def original_partnership_profiles(A,pair_i=None,pair_j=None):
    """Third-party target divergence and mutual preference; direct pair edges excluded from profiles."""
    from scipy.special import entr
    A=np.asarray(A,float);N=len(A);outgoing=A.sum(1)
    if pair_i is None:
        i,j=np.triu_indices(N,1);keep=(outgoing[i]>0)&(outgoing[j]>0);pair_i=i[keep];pair_j=j[keep]
    i,j=pair_i,pair_j;u=A[i].copy();v=A[j].copy();u[np.arange(len(i)),j]=0;v[np.arange(len(i)),i]=0;nu=u.sum(1);nv=v.sum(1)
    U=np.divide(u,nu[:,None],out=np.zeros_like(u),where=nu[:,None]>0);V=np.divide(v,nv[:,None],out=np.zeros_like(v),where=nv[:,None]>0);mid=.5*(U+V)
    divergence=(np.sum(entr(mid),axis=1)-.5*np.sum(entr(U),axis=1)-.5*np.sum(entr(V),axis=1))/np.log(2);divergence[(nu<=0)|(nv<=0)]=np.nan
    mutual=A[i,j]*A[j,i]/(outgoing[i]*outgoing[j]);return i,j,divergence,mutual,np.minimum(nu,nv)


def original_partnership_r(x,y,minimum_other=None,threshold=0):
    good=np.isfinite(x)&np.isfinite(y)
    if minimum_other is not None:good&=minimum_other>=threshold
    if good.sum()<10:return np.nan
    a=x[good]-x[good].mean();b=y[good]-y[good].mean();den=np.sqrt((a@a)*(b@b));return float(a@b/den) if den>1e-18 else np.nan


def original_vectorized_partnership_null(A,draws,pilot,seed,cache):
    """Uniform labeled-pass matching reference, constrained against self-passes.
    Disjoint 2-edge swaps / 3-edge rotations each sweep are vectorized, reversible kernels.
    """
    import hashlib
    A=np.asarray(A,np.int64);N=len(A);key=hashlib.sha256(A.tobytes()+str((A.shape,draws,pilot,seed,'disjoint_2_3_v1')).encode()).hexdigest()
    if cache.exists():
        with np.load(cache,allow_pickle=False) as old:
            if str(old['source_hash'])==key:return {k:old[k] for k in old.files}
    rng=np.random.default_rng(seed);source=np.repeat(np.arange(N),A.sum(1));target=np.concatenate([np.repeat(np.arange(N),A[i]) for i in range(N)]);M=len(source);accepted=0;proposals=0
    def sweep(count=1):
        nonlocal accepted,proposals
        for _ in range(count):
            order=rng.permutation(M)
            if rng.random()<.8:
                index=order[:M//2*2].reshape(-1,2);u,v=index.T;r1,r2=source[u],source[v];d1,d2=target[u].copy(),target[v].copy();good=(r1!=r2)&(d1!=d2)&(r1!=d2)&(r2!=d1);target[u[good]]=d2[good];target[v[good]]=d1[good];accepted+=int(good.sum());proposals+=len(good)
            else:
                index=order[:M//3*3].reshape(-1,3)
                if not len(index):continue
                u,v,w=index.T;r1,r2,r3=source[u],source[v],source[w];d1,d2,d3=target[u].copy(),target[v].copy(),target[w].copy();forward=rng.random(len(u))<.5;n1=np.where(forward,d2,d3);n2=np.where(forward,d3,d1);n3=np.where(forward,d1,d2)
                good=(r1!=r2)&(r1!=r3)&(r2!=r3)&(d1!=d2)&(d1!=d3)&(d2!=d3)&(r1!=n1)&(r2!=n2)&(r3!=n3);target[u[good]]=n1[good];target[v[good]]=n2[good];target[w[good]]=n3[good];accepted+=int(good.sum());proposals+=len(good)
    i,j,xobs,yobs,otherobs=original_partnership_profiles(A);pair_count=len(i);sweep(20);pilot_y=np.zeros((pilot,pair_count));edge_mean=np.zeros_like(A,float)
    for k in range(pilot):
        sweep(2);B=np.bincount(source*N+target,minlength=N*N).reshape(N,N);_,_,_,y,_=original_partnership_profiles(B,i,j);pilot_y[k]=y;edge_mean+=B
    expected=pilot_y.mean(0);sweep(20);null_r=[];null_r50=[];null_raw_r=[];null_gap=[];audit_matrices=[];audit_x=[];audit_y=[];audit_other=[]
    for k in range(draws):
        sweep(2);B=np.bincount(source*N+target,minlength=N*N).reshape(N,N)
        if k%50==0 and (not np.array_equal(B.sum(0),A.sum(0)) or not np.array_equal(B.sum(1),A.sum(1)) or np.diag(B).any()):raise ValueError('Null pass margins changed')
        _,_,x,y,other=original_partnership_profiles(B,i,j);excess=y-expected;null_r.append(original_partnership_r(x,excess));null_r50.append(original_partnership_r(x,excess,other,50));null_raw_r.append(original_partnership_r(x,y));good=np.isfinite(x)
        q=np.quantile(x[good],[.25,.75]) if good.any() else [np.nan,np.nan];null_gap.append(float(excess[good&(x>=q[1])].mean()-excess[good&(x<=q[0])].mean()) if good.any() else np.nan)
        if k<5:audit_matrices.append(B);audit_x.append(x);audit_y.append(y);audit_other.append(other)
    null_r=np.array(null_r);lag=np.corrcoef(null_r[:-1],null_r[1:])[0,1] if np.isfinite(null_r).all() and np.std(null_r)>0 else 0;excess_obs=yobs-expected;good=np.isfinite(xobs);q=np.quantile(xobs[good],[.25,.75]) if good.any() else [np.nan,np.nan];obs_gap=float(excess_obs[good&(xobs>=q[1])].mean()-excess_obs[good&(xobs<=q[0])].mean()) if good.any() else np.nan
    result=dict(pair_i=i,pair_j=j,divergence=xobs,mutual=yobs,minimum_other_passes=otherobs,null_expected=expected,pilot_expected_se=pilot_y.std(0,ddof=1)/np.sqrt(pilot),pilot_mean_edges=edge_mean/pilot,
        observed_r=np.array(original_partnership_r(xobs,excess_obs)),observed_r50=np.array(original_partnership_r(xobs,excess_obs,otherobs,50)),observed_raw_r=np.array(original_partnership_r(xobs,yobs)),observed_gap=np.array(obs_gap),null_r=null_r,null_r50=np.array(null_r50),null_raw_r=np.array(null_raw_r),null_gap=np.array(null_gap),
        acceptance=np.array(accepted/proposals),lag1=np.array(lag),source_hash=np.array(key),audit_matrices=np.array(audit_matrices),audit_x=np.array(audit_x),audit_y=np.array(audit_y),audit_other=np.array(audit_other))
    np.savez_compressed(cache,**result);return result


def original_partnership_regression(pairs,label):
    """Network FE, equal network weights; team and team/pair two-way cluster inference."""
    from scipy.stats import t
    d=pairs.dropna(subset=['third_party_divergence','mutual_excess']).copy();groups=d.groupby('network');d['rx']=d.third_party_divergence-groups.third_party_divergence.transform('mean');d['ry']=d.mutual_excess-groups.mutual_excess.transform('mean');w=1/groups.network.transform('size').to_numpy(float);x=d.rx.to_numpy();y=d.ry.to_numpy();S=np.sum(w*x*x);beta=np.sum(w*x*y)/S;resid=y-beta*x;score=w*x*resid;n=len(d);k=d.network.nunique()+1
    def cluster_meat(keys):
        temp=d[keys].copy();temp['score']=score;s=temp.groupby(keys)['score'].sum();G=len(s);return G/(G-1)*(n-1)/(n-k)*float(s@s),G
    team,G=cluster_meat(['team']);dyad,H=cluster_meat(['pair_id']);intersection,I=cluster_meat(['team','pair_id']);var_team=team/S**2;var_two=(team+dyad-intersection)/S**2;corr=float(np.sum(w*x*y)/np.sqrt(np.sum(w*x*x)*np.sum(w*y*y)))
    return dict(model=label,pairs=n,networks=d.network.nunique(),teams=G,unique_pairs=H,slope=beta,within_r=corr,within_r_squared=corr*corr,team_cluster_se=np.sqrt(var_team),team_cluster_p=2*t.sf(abs(beta/np.sqrt(var_team)),G-1),
        two_way_cluster_se=np.sqrt(var_two) if var_two>0 else np.nan,two_way_cluster_p=2*t.sf(abs(beta/np.sqrt(var_two)),min(G,H)-1) if var_two>0 else np.nan),d


def original_mc_two_sided(observed,reference):
    reference=np.asarray(reference);reference=reference[np.isfinite(reference)]
    return min(1.,2*min((1+np.sum(reference>=observed))/(len(reference)+1),(1+np.sum(reference<=observed))/(len(reference)+1)))


def run_all_season_partnerships(argv):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from scipy.stats import spearmanr
    import json
    parser=argparse.ArgumentParser(description='All 180 original season networks: complementary or similar passing partnerships')
    parser.add_argument('--all-season-partnerships',action='store_true');parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT_DIR);parser.add_argument('--null-draws',type=int,default=499);parser.add_argument('--pilot-draws',type=int,default=199);parser.add_argument('--seed',type=int,default=20261112)
    args=parser.parse_args(argv)
    if min(args.null_draws,args.pilot_draws)<99:raise ValueError('Use at least 99 pilot and calibration draws')
    root=args.output_dir.expanduser();out=root/'all_seasons_partnerships';out.mkdir(parents=True,exist_ok=True);cache=out/'null_cache';cache.mkdir(exist_ok=True);summary=pd.read_csv(root/'2020s_summary/Regular_Season/all_seasons_team_metrics.csv').sort_values(['season','team']);networkrows=[];pairrows=[];nulls=[];nulls50=[];audits=[];inputfiles=[]
    for counter,(_,r) in enumerate(summary.iterrows(),1):
        folder=root/r.season/'Regular_Season'/r.team;file=folder/'passing_matrix.csv';f=pd.read_csv(file,index_col=0);ids=f.index.astype(int).to_numpy();A=f.to_numpy(float)
        if not np.isfinite(A).all() or (A<0).any() or not np.allclose(A,np.round(A)) or np.diag(A).any():raise ValueError('Invalid observed counts')
        roster=pd.read_csv(folder/'roster_stats.csv').set_index('PLAYER_ID');active=A.sum(0)+A.sum(1)>0;A=A[np.ix_(active,active)].astype(np.int64);ids=ids[active];network=r.season+'_'+r.team
        d=original_vectorized_partnership_null(A,args.null_draws,args.pilot_draws,args.seed+counter,cache/(network+'.npz'));expected=d['null_expected'];x=d['divergence'];excess=d['mutual']-expected;good=np.isfinite(x);observed=float(d['observed_r']);ref=d['null_r'];valid=np.isfinite(observed) and np.isfinite(ref).all();observed50=float(d['observed_r50']);valid50=np.isfinite(observed50) and np.isfinite(d['null_r50']).all()
        record=dict(network=network,team=r.team,season=r.season,active_players=len(A),defined_sources=int((A.sum(1)>0).sum()),recorded_passes=int(A.sum()),evaluable_pairs=int(good.sum()),
            observed_excess_r=observed,observed_raw_r=float(d['observed_raw_r']),mean_null_excess_r=float(np.nanmean(ref)),null_calibrated_fisher_difference=float(np.arctanh(np.clip(observed,-.999999,.999999))-np.nanmean(np.arctanh(np.clip(ref,-.999999,.999999)))),
            unadjusted_network_mc_p=original_mc_two_sided(observed,ref),eligible_global=valid,observed_excess_r_min50=observed50,eligible_min50_global=valid50,observed_quartile_gap=float(d['observed_gap']),
            null_lag1=float(d['lag1']),null_acceptance=float(d['acceptance']))
        networkrows.append(record);nulls.append(ref);nulls50.append(d['null_r50']);mpg=roster.MIN/roster.GP;audits.append(dict(network=network,roster_players=len(roster),players_below10_mpg=int(mpg.lt(10).sum()),minimum_mpg=mpg.min(),scope='original saved rotation cohort; no new minute filter'));inputfiles.append(dict(network=network,source=str(file),null_cache=str(cache/(network+'.npz'))))
        for n,(i,j) in enumerate(zip(d['pair_i'],d['pair_j'])):
            if not good[n]:continue
            pid,qid=sorted([int(ids[i]),int(ids[j])]);pairrows.append(dict(network=network,team=r.team,season=r.season,pair_id=str(pid)+'_'+str(qid),player_1=pid,player_2=qid,name_1=roster.loc[pid,'PLAYER_NAME'],name_2=roster.loc[qid,'PLAYER_NAME'],third_party_divergence=x[n],mutual_preference=d['mutual'][n],null_expected=expected[n],mutual_excess=excess[n],minimum_other_passes=d['minimum_other_passes'][n]))
        if counter%10==0 or counter==len(summary):
            atomic_csv(pd.DataFrame(networkrows),out/'network_results.csv');print(f'Original season networks {counter}/{len(summary)} complete',flush=True)
    networks=pd.DataFrame(networkrows);pairs=pd.DataFrame(pairrows);nulls=np.stack(nulls);nulls50=np.stack(nulls50);valid=networks.eligible_global.to_numpy(bool);Z=np.arctanh(np.clip(nulls[valid],-.999999,.999999));observed=float(np.mean(np.arctanh(np.clip(networks.loc[valid,'observed_excess_r'],-.999999,.999999))));reference=Z.mean(0);p=original_mc_two_sided(observed,reference)
    valid50=networks.eligible_min50_global.to_numpy(bool);obs50=float(np.mean(np.arctanh(np.clip(networks.loc[valid50,'observed_excess_r_min50'],-.999999,.999999))));ref50=np.arctanh(np.clip(nulls50[valid50],-.999999,.999999)).mean(0)
    primary=pd.DataFrame([dict(test='All original networks: null-calibrated relationship',networks=int(valid.sum()),observed_mean_fisher_z=observed,null_mean_fisher_z=reference.mean(),calibrated_difference=observed-reference.mean(),mc_two_sided_p=p),dict(test='At least 50 passes to other teammates per player',networks=int(valid50.sum()),observed_mean_fisher_z=obs50,null_mean_fisher_z=ref50.mean(),calibrated_difference=obs50-ref50.mean(),mc_two_sided_p=original_mc_two_sided(obs50,ref50))])
    regression,plot=original_partnership_regression(pairs,'All defined pair profiles');reg50,plot50=original_partnership_regression(pairs.loc[pairs.minimum_other_passes.ge(50)],'At least 50 passes to other teammates');regressions=pd.DataFrame([regression,reg50]);rng=np.random.default_rng(args.seed+90000);teams=networks.team.unique();bootstrap=[]
    team_values={team:networks.loc[networks.team.eq(team)&networks.eligible_global,'null_calibrated_fisher_difference'].to_numpy() for team in teams}
    for _ in range(5000):
        selected=rng.choice(teams,len(teams),replace=True);values=np.concatenate([team_values[team] for team in selected]);bootstrap.append(values.mean())
    ci=np.quantile(bootstrap,[.025,.975]);primary['team_block_ci_low']=np.nan;primary['team_block_ci_high']=np.nan;primary.loc[0,['team_block_ci_low','team_block_ci_high']]=ci
    atomic_csv(networks,out/'network_results.csv');atomic_csv(pairs,out/'pair_profiles_and_excess.csv');atomic_csv(primary,out/'primary_results.csv');atomic_csv(regressions,out/'regressions.csv');atomic_csv(plot,out/'within_network_plot_data.csv');atomic_csv(pd.DataFrame(audits),out/'roster_audit.csv');atomic_csv(pd.DataFrame(inputfiles),out/'source_manifest.csv');atomic_csv(pd.DataFrame(dict(draw=np.arange(args.null_draws),global_null_fisher_z=reference,global_null_min50_fisher_z=ref50)),out/'global_null_reference.csv')
    seasons=networks.groupby('season').agg(networks=('network','size'),mean_observed_r=('observed_excess_r','mean'),mean_null_r=('mean_null_excess_r','mean'),mean_calibrated_difference=('null_calibrated_fisher_difference','mean'),positive_difference_share=('null_calibrated_fisher_difference',lambda v:(v>0).mean())).reset_index();clubs=networks.groupby('team').agg(seasons=('season','size'),mean_calibrated_difference=('null_calibrated_fisher_difference','mean')).reset_index().sort_values('mean_calibrated_difference');atomic_csv(seasons,out/'season_summary.csv');atomic_csv(clubs,out/'team_summary.csv')
    with PdfPages(out/'all_seasons_partnerships.pdf') as pdf:
        def save(fig,name):
            fig.tight_layout();pdf.savefig(fig);fig.savefig(out/(name+'.pdf'));fig.savefig(out/(name+'.png'),dpi=150);plt.close(fig)
        fig,ax=plt.subplots(figsize=(11,8));ax.hist(reference,bins=25,color='#9bb8d4',label='Randomized reference');ax.axvline(observed,color='#c65c28',lw=2,label='Observed');ax.set(xlabel='Mean network Fisher-z correlation: profile difference vs partnership excess',ylabel='Calibration draws',title=f'Are partnerships associated with similar or different passing roles?\n{valid.sum()} original networks; two-sided Monte Carlo p={p:.4g}');ax.legend();save(fig,'global_null_test')
        fig,ax=plt.subplots(figsize=(11,8));ax.scatter(plot.rx,plot.ry,s=4,alpha=.08,color='#2473b8',rasterized=True);xx=np.linspace(plot.rx.min(),plot.rx.max(),100);ax.plot(xx,regression['slope']*xx,color='#c65c28',lw=2);ax.axhline(0,color='gray',lw=.5);ax.axvline(0,color='gray',lw=.5)
        ax.set(xlabel='Third-party target divergence relative to network mean',ylabel='Mutual-preference excess relative to network mean',title=f'Equal-network-weight regression, network fixed effects\nWithin r={regression["within_r"]:.3f}; team-cluster p={regression["team_cluster_p"]:.3g}');save(fig,'within_network_regression')
        fig,ax=plt.subplots(figsize=(11,8))
        for season,g in networks.groupby('season'):ax.scatter(g.mean_null_excess_r,g.observed_excess_r,label=season,s=25,alpha=.8)
        limits=[min(networks.mean_null_excess_r.min(),networks.observed_excess_r.min())-.05,max(networks.mean_null_excess_r.max(),networks.observed_excess_r.max())+.05];ax.plot(limits,limits,color='gray',ls='--');ax.set(xlabel='Mean randomized-network correlation',ylabel='Observed-network correlation',title='Observed versus expected association in each original network');ax.legend(fontsize=8);save(fig,'observed_vs_null_networks')
        fig,ax=plt.subplots(figsize=(12,7));ax.bar(seasons.season,seasons.mean_calibrated_difference,color='#2473b8');ax.axhline(0,color='gray',lw=.8);ax.set(ylabel='Mean observed minus null Fisher-z association',title='Association beyond activity constraints by season\nPositive favors complementary profiles; negative favors similar profiles');save(fig,'season_comparison')
        fig,ax=plt.subplots(figsize=(11,10));ax.barh(clubs.team,clubs.mean_calibrated_difference,color='#2473b8');ax.axvline(0,color='gray',lw=.8);ax.set(xlabel='Mean null-calibrated Fisher-z association across six seasons',title='Team descriptions; no individual-team significance claim');save(fig,'team_comparison')
        example=pairs.loc[pairs.network.eq('2021-22_GSW')];fig,ax=plt.subplots(figsize=(11,8));ax.scatter(example.third_party_divergence,example.mutual_excess,color='#9aa8b7',s=25)
        for ids,label,col in [({201939,203110},'Curry–Green','#168348'),({201939,1629673},'Curry–Poole','#2473b8'),({203110,1629673},'Green–Poole','#e18a24')]:
            q=example.loc[example.apply(lambda r:{r.player_1,r.player_2}==ids,axis=1)]
            if len(q):v=q.iloc[0];ax.scatter([v.third_party_divergence],[v.mutual_excess],color=col,s=65);ax.annotate(label,(v.third_party_divergence,v.mutual_excess),xytext=(4,5),textcoords='offset points')
        ax.axhline(0,color='gray',lw=.7);ax.set(xlabel='Difference in passing to OTHER teammates (JS divergence)',ylabel='Mutual preference minus fixed-activity expectation',title='Original Warriors 2021–22 season network\nIllustration only; this is not the 82-game network dataset');save(fig,'warriors_season_example')
    direction='complementary' if observed-reference.mean()>0 else 'similar';report=['# Original league-wide networks: similar or complementary partnerships?',
        f'Completed {len(networks)} team-season networks, 30 teams, six seasons (2020–21 through 2025–26). The original data inherit a >=10 MPG rotation cohort. No new minute cutoff imposed, no Warriors game-level networks substituted, and no new NBA downloads.',
        '## Main results',integrated_table(primary),integrated_table(regressions),integrated_table(seasons),
        f'The null-calibrated association points toward {direction} third-party passing profiles. This is a comparison with the stated activity-constrained reference, not a causal statement about player intentions or basketball effectiveness. Positive observed r alone is not sufficient: calibration recomputes the profile differences and partnership scores together in every null network to account for sampling and mathematical coupling.',
        '## Definitions',
        'A[i,j] = observed directed passes. P[i,j] = A[i,j]/outgoing_i. Pair strength is P[i,j]*P[j,i]. For each player pair, remove BOTH compared players from both receiving target sets, renormalize the remaining rows, and calculate JS divergence (base2). High divergence means different observed third-party targets; low divergence means similar targets. Self-pass impossibility and the direct pair edges do not enter the profile comparison. Undefined outside-target profiles are excluded with counts recorded.',
        f'Null networks preserve exact row and column pass totals and forbid self-passes. {args.pilot_draws} pilot samples estimate each pair’s activity-expected mutual score. {args.null_draws} further calibration samples estimate the correlation of third-party divergence with excess mutual strength. Independent disjoint two-edge swap and three-edge rotation kernels preserve the labeled-pass-matching reference. Twenty burn-in sweeps before pilot and calibration, two sweeps between samples; acceptance, lag-one correlations and audit matrices saved. Finite-chain Monte Carlo is approximate; pilot and calibration separated by burn-in, not claimed exactly independent.',
        'Primary global statistic is the equal-network mean Fisher-z correlation of third-party divergence with mutual-preference excess. A two-sided Monte Carlo comparison permits either similarity or complementarity. Team-block bootstrap resamples all seasons of the same franchise together; its interval describes variation across franchises and retains temporal grouping. Per-network p-values are exploratory unadjusted diagnostics; no claims of individual network significance. The min50 sensitivity requires >=50 observed passes to third parties for EACH player, and applies the same computability rule in null draws. This is a pass-exposure check, not a minutes filter.',
        'Secondary regression absorbs network fixed effects and gives each network equal total weight. Standard errors clustered by team, with two-way team/dyad clustering as a sensitivity, including correct team×dyad intersection aggregation across repeated seasons. Pairs sharing players across different teams may retain additional dependence. Primary randomization calibration remains the main test. Within-network R² is not an offensive-performance R².',
        '## Limits and interpretation',
        'Similarity is observed destination similarity, not identical positions, abilities or intentions. Complementarity means different target distributions, not proven tactical complementarity. Game or season aggregates cannot identify exact shared-court time; lineup overlap may explain some structure. The null controls pass activity but not joint court minutes, substitutions, opponent defense or possession order. Original rotation-cohort selection can change with playing time, and its cutoff limits full-roster generalization. Count totals may be smaller than full official team passes because omitted rotation players are outside these original matrices. No scoring, winning or causal efficiency claim follows.',
        'The 2021–22 Warriors figure is a familiar illustrative season network, not selected as evidence for the league-wide conclusion. All 180 network results, dyads, sensitivity results, reference draws and source manifests are retained, including weak or opposite directions. Existing project code and CLI preserved; new command: --all-season-partnerships.'
    ];(out/'DETAILED_ANALYSIS.md').write_text('\n\n'.join(report));atomic_json(dict(primary_statistic='equal-network mean Fisher-z excess-strength/profile-divergence correlation',networks=len(networks),teams=networks.team.nunique(),seasons=networks.season.nunique(),pilot_draws=args.pilot_draws,calibration_draws=args.null_draws,seed=args.seed,new_downloads=False),out/'run_manifest.json')
    print(primary.to_string(index=False),flush=True);print(regressions.to_string(index=False),flush=True);print(seasons.to_string(index=False),flush=True);print('Saved original-network partnership study: '+str(out),flush=True);return 0


# ---------------------------------------------------------------------
# SEASON STARTERS: FIVE MOST FREQUENT STARTERS, INDUCED PASSING NETWORKS
# ---------------------------------------------------------------------
def run_starter_passing_analysis(argv):
    """Season passing among five most frequent starters; not simultaneous-lineup passes."""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import networkx as nx
    parser=argparse.ArgumentParser(description='Starter-only season passing networks and partnership analysis')
    parser.add_argument('--starter-passing-analysis',action='store_true')
    parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--starter-counts-dir',type=Path)
    parser.add_argument('--starter-profiles-dir',type=Path)
    parser.add_argument('--null-draws',type=int,default=499)
    parser.add_argument('--pilot-draws',type=int,default=199)
    parser.add_argument('--seed',type=int,default=20261113)
    args=parser.parse_args(argv)
    if min(args.null_draws,args.pilot_draws)<99:raise ValueError('Use at least 99 pilot and calibration draws')
    root=args.output_dir.expanduser();out=root/'starter_passing_networks';out.mkdir(exist_ok=True,parents=True)
    counts=args.starter_counts_dir.expanduser() if args.starter_counts_dir else out/'starter_counts'
    counts.mkdir(exist_ok=True,parents=True)
    profiles=args.starter_profiles_dir.expanduser() if args.starter_profiles_dir else out/'profiles'
    profiles.mkdir(exist_ok=True,parents=True)
    summary=pd.read_csv(root/'2020s_summary/Regular_Season/all_seasons_team_metrics.csv').sort_values(['season','team'])
    selections=[];audit=[];metrics=[];pairrows=[];refs=[];refs50=[];graphs=[];failed=[]
    for k,r in enumerate(summary.itertuples(),1):
        network=r.season+'_'+r.team;folder=root/r.season/'Regular_Season'/r.team
        roster=pd.read_csv(folder/'roster_stats.csv').set_index('PLAYER_ID');countfile=counts/(network+'.csv')
        if not countfile.exists():
            q=leaguedashplayerstats.LeagueDashPlayerStats(season=r.season,season_type_all_star='Regular Season',team_id_nullable=int(roster.TEAM_ID.iloc[0]),starter_bench_nullable='Starters',per_mode_detailed='Totals',timeout=20)
            frame=q.get_data_frames()[0];frame['QUERY_TEAM']=r.team;frame['QUERY_TEAM_ID']=int(roster.TEAM_ID.iloc[0]);frame.to_csv(countfile,index=False)
            atomic_json(dict(parameters=q.parameters,fetched_at=utc_now(),payload=q.get_dict()),countfile.with_suffix('.json'))
        starts=pd.read_csv(countfile)
        expected_games=72 if r.season=='2020-21' else 82
        if starts.PLAYER_ID.duplicated().any() or starts.GP.sum()!=expected_games*5:raise ValueError('Starter-count validation failed: '+network)
        # NBA labels can name a traded player's final team even when GP is correctly team-filtered.
        # The queried team and the five-starts-per-game reconciliation establish the scope.
        starts['season_minutes']=starts.PLAYER_ID.map(roster.MIN)
        starts=starts.sort_values(['GP','MIN','PLAYER_ID'],ascending=[False,False,True]);chosen=starts.head(5).copy();ids=chosen.PLAYER_ID.astype(int).tolist()
        for rank,z in enumerate(chosen.itertuples(),1):
            selections.append(dict(network=network,season=r.season,team=r.team,selection_rank=rank,player_id=z.PLAYER_ID,name=z.PLAYER_NAME,starts=int(z.GP),starter_minutes=z.MIN,season_minutes=z.season_minutes,cutoff_tie=bool((starts.GP==chosen.GP.iloc[-1]).sum()>1)))
        full=pd.read_csv(folder/'passing_matrix.csv',index_col=0);full.index=full.index.astype(int);full.columns=full.columns.astype(int)
        missing=set(ids)-set(full.index)
        audit.append(dict(network=network,team=r.team,season=r.season,team_games=expected_games,total_team_starts=int(starts.GP.sum()),selected_players=len(ids),selected_players_missing_from_original=len(missing),starter_count_source=str(countfile),passing_source=str(folder/'passing_matrix.csv')))
        # Reconstruct all selected rows from full raw player profiles, recovering traded starters
        # and targets omitted from the historical rotation-only matrix.
        A=np.zeros((5,5),float);profilefolder=profiles/network;profilefolder.mkdir(exist_ok=True,parents=True)
        for i,pid in enumerate(ids):
            pf=profilefolder/(str(pid)+'.csv')
            if not pf.exists():
                client=NBAClient(Config(season=r.season,cache_dir=out/'raw_profile_cache',timeout=20,retries=2,min_delay=.2,max_delay=.4))
                try:frame,snapshot=client.passing(pid,int(roster.TEAM_ID.iloc[0]))
                finally:client.close()
                if frame.empty:raise ValueError('Missing actual starter profile: '+network+' '+str(pid))
                frame.to_csv(pf,index=False)
            frame=pd.read_csv(pf)
            if frame.empty or not frame.PLAYER_ID.eq(pid).all() or not frame.TEAM_ID.eq(int(roster.TEAM_ID.iloc[0])).all():raise ValueError('Invalid starter profile scope: '+str(pf))
            weights=frame.groupby('PASS_TEAMMATE_PLAYER_ID').PASS.sum()
            for j,qid in enumerate(ids):
                if i!=j:A[i,j]=float(weights.get(qid,0))
        known=[i for i,pid in enumerate(ids) if pid in full.index]
        original_edges=full.loc[[ids[i] for i in known],[ids[i] for i in known]].to_numpy(float)
        mismatch=int(np.sum(~np.isclose(A[np.ix_(known,known)],original_edges)))
        if mismatch:raise ValueError('Raw profiles disagree with existing original edges: '+network)
        audit[-1]['recovered_starter_profiles']=len(missing);audit[-1]['retained_edge_mismatches']=mismatch
        if not np.isfinite(A).all() or (A<0).any() or np.diag(A).any() or not np.allclose(A,np.round(A)):raise ValueError('Invalid passing counts: '+network)
        savefolder=out/'networks'/r.season/r.team;savefolder.mkdir(exist_ok=True,parents=True)
        atomic_csv(pd.DataFrame(A,index=ids,columns=ids),savefolder/'passing_matrix.csv',index=True)
        O=A.sum(1);P=np.divide(A,O[:,None],out=np.full_like(A,np.nan),where=O[:,None]>0)
        atomic_csv(pd.DataFrame(P,index=ids,columns=ids),savefolder/'passing_probabilities.csv',index=True)
        atomic_csv(chosen.rename(columns={'GP':'STARTS'}),savefolder/'selected_starters.csv')
        if (A.sum(0)+A.sum(1)==0).any() or (O==0).any():
            failed.append(dict(network=network,reason='Five selected starters do not all have observed outgoing passes',missing_ids=''));continue
        cache=out/'null_cache';cache.mkdir(exist_ok=True)
        d=original_vectorized_partnership_null(A,args.null_draws,args.pilot_draws,args.seed+k,cache/(network+'.npz'))
        x=d['divergence'];y=d['mutual']-d['null_expected'];rh,ch=compute_team_entropies(A);mi,nmi=passing_mutual_information(A)
        W=A+A.T;values,vectors=np.linalg.eigh(W);v=np.abs(vectors[:,-1]);v=v/v.sum();centrality=float(v@v)
        obs=float(d['observed_r']);ref=d['null_r'];valid=np.isfinite(obs) and np.isfinite(ref).all();obs50=float(d['observed_r50']);valid50=np.isfinite(obs50) and np.isfinite(d['null_r50']).all()
        metrics.append(dict(network=network,team=r.team,season=r.season,players=5,recorded_passes=int(A.sum()),original_recorded_passes=int(full.to_numpy().sum()),ratio_original_recorded_passes=A.sum()/full.to_numpy().sum(),receiver_entropy=rh,conditional_entropy=ch,normalized_mi=nmi,eigenvector_hhi=centrality,
            observed_excess_r=obs,mean_null_excess_r=float(np.nanmean(ref)),null_calibrated_fisher_difference=float(np.arctanh(np.clip(obs,-.999999,.999999))-np.nanmean(np.arctanh(np.clip(ref,-.999999,.999999)))),eligible_global=valid,observed_excess_r_min50=obs50,eligible_min50_global=valid50,null_lag1=float(d['lag1']),null_acceptance=float(d['acceptance'])))
        refs.append(ref);refs50.append(d['null_r50'])
        for n,(i,j) in enumerate(zip(d['pair_i'],d['pair_j'])):
            pid,qid=sorted([ids[i],ids[j]]);names=dict(zip(chosen.PLAYER_ID,chosen.PLAYER_NAME))
            pairrows.append(dict(network=network,team=r.team,season=r.season,pair_id=str(pid)+'_'+str(qid),player_1=pid,player_2=qid,name_1=names[pid],name_2=names[qid],third_party_divergence=x[n],mutual_preference=d['mutual'][n],null_expected=d['null_expected'][n],mutual_excess=y[n],minimum_other_passes=d['minimum_other_passes'][n]))
        graphs.append((r.season,r.team,A,ids,chosen,v))
        if k%10==0 or k==len(summary):print(f'Starter networks {k}/{len(summary)} complete',flush=True)
    networks=pd.DataFrame(metrics);pairs=pd.DataFrame(pairrows);selection=pd.DataFrame(selections)
    if networks.empty:raise ValueError('No complete starter networks available')
    refs=np.stack(refs);refs50=np.stack(refs50);valid=networks.eligible_global.to_numpy(bool);reference=np.arctanh(np.clip(refs[valid],-.999999,.999999)).mean(0);observed=np.arctanh(np.clip(networks.loc[valid,'observed_excess_r'],-.999999,.999999)).mean()
    eligible50=networks.eligible_min50_global.to_numpy(bool);obs50=np.arctanh(np.clip(networks.loc[eligible50,'observed_excess_r_min50'],-.999999,.999999)).mean() if eligible50.any() else np.nan;ref50=np.arctanh(np.clip(refs50[eligible50],-.999999,.999999)).mean(0) if eligible50.any() else np.full(args.null_draws,np.nan)
    primary=pd.DataFrame([dict(test='Five most frequent starters',networks=int(valid.sum()),observed_fisher_z=observed,null_fisher_z=reference.mean(),difference=observed-reference.mean(),mc_two_sided_p=original_mc_two_sided(observed,reference)),dict(test='At least 50 other-starter passes per player',networks=int(eligible50.sum()),observed_fisher_z=obs50,null_fisher_z=np.nanmean(ref50),difference=obs50-np.nanmean(ref50),mc_two_sided_p=original_mc_two_sided(obs50,ref50) if eligible50.any() else np.nan)])
    reg,plot=original_partnership_regression(pairs,'Five most frequent starters');reg50,_=original_partnership_regression(pairs.loc[pairs.minimum_other_passes.ge(50)],'At least 50 other-starter passes');regressions=pd.DataFrame([reg,reg50])
    franchise_values={team:g.null_calibrated_fisher_difference.to_numpy() for team,g in networks.loc[valid].groupby('team')};rng=np.random.default_rng(args.seed+90000);teams=list(franchise_values);boot=[np.concatenate([franchise_values[t] for t in rng.choice(teams,len(teams),replace=True)]).mean() for _ in range(5000)];ci=np.quantile(boot,[.025,.975]);primary['team_block_ci_low']=np.nan;primary['team_block_ci_high']=np.nan;primary.loc[0,['team_block_ci_low','team_block_ci_high']]=ci
    seasons=networks.groupby('season').agg(networks=('network','size'),mean_r=('observed_excess_r','mean'),mean_calibrated_difference=('null_calibrated_fisher_difference','mean'),mean_conditional_entropy=('conditional_entropy','mean'),mean_eigenvector_hhi=('eigenvector_hhi','mean')).reset_index()
    original=pd.read_csv(root/'all_seasons_partnerships/network_results.csv');comparison=networks.merge(original[['network','observed_excess_r','null_calibrated_fisher_difference']],on='network',suffixes=('_starters','_rotation'))
    for name,frame in [('network_metrics',networks),('selected_starters',selection),('source_audit',pd.DataFrame(audit)),('excluded_networks',pd.DataFrame(failed,columns=['network','reason','missing_ids'])),('pair_profiles_and_excess',pairs),('primary_results',primary),('regressions',regressions),('season_summary',seasons),('original_cohort_comparison',comparison),('global_null_reference',pd.DataFrame({'draw':np.arange(args.null_draws),'global_fisher_z':reference,'min50_fisher_z':ref50}))]:atomic_csv(frame,out/(name+'.csv'))
    def draw_network(ax,A,ids,chosen,v,title):
        G=nx.DiGraph();G.add_nodes_from(range(5))
        for i in range(5):
            for j in range(5):
                if A[i,j]>0:G.add_edge(i,j,weight=float(A[i,j]))
        G.remove_nodes_from(list(nx.isolates(G)))
        pos=nx.spring_layout(G.to_undirected(),seed=42,weight='weight');names=dict(zip(chosen.PLAYER_ID,chosen.PLAYER_NAME));starts=dict(zip(chosen.PLAYER_ID,chosen.GP));labels={i:names[ids[i]]+'\n'+str(int(starts[ids[i]]))+' starts' for i in G}
        maximum=A.max();nx.draw_networkx_nodes(G,pos,ax=ax,node_size=[900+4500*v[i] for i in G],node_color='#dceaf3',edgecolors='#34769e');nx.draw_networkx_labels(G,pos,labels,ax=ax,font_size=7)
        nx.draw_networkx_edges(G,pos,ax=ax,width=[.4+3.5*G[u][w]['weight']/maximum for u,w in G.edges()],alpha=.48,edge_color='#34769e',arrows=True,arrowsize=10,connectionstyle='arc3,rad=0.13',node_size=1800)
        ax.set_title(title,fontsize=11);ax.set_axis_off();ax.margins(.25)
    with PdfPages(out/'starter_networks_all_180.pdf') as gallery:
        for season in sorted(networks.season.unique()):
            gg=[g for g in graphs if g[0]==season]
            for start in range(0,len(gg),6):
                fig,axes=plt.subplots(3,2,figsize=(13,15))
                for ax,g in zip(axes.flat,gg[start:start+6]):draw_network(ax,g[2],g[3],g[4],g[5],g[1]+' '+g[0])
                for ax in list(axes.flat)[len(gg[start:start+6]):]:ax.set_axis_off()
                fig.suptitle('Five most frequent starters: season passing between selected players',fontsize=14);fig.text(.5,.012,'Directed edges: recorded passes; width: count; node size: combined passing eigenvector centrality.\nIncludes passes with substitutes on court. Not an exact five-player lineup network.',ha='center',fontsize=9);fig.tight_layout(rect=[0,.04,1,.96]);gallery.savefig(fig);plt.close(fig)
    with PdfPages(out/'starter_analysis_graphs.pdf') as pdf:
        def save(fig,name):
            fig.tight_layout();pdf.savefig(fig);fig.savefig(out/(name+'.png'),dpi=150);fig.savefig(out/(name+'.pdf'));plt.close(fig)
        fig,ax=plt.subplots(figsize=(11,8));ax.hist(reference,bins=25,color='#9bb8d4');ax.axvline(observed,color='#c65c28',lw=2,label='Observed');ax.legend();ax.set(xlabel='Equal-network mean Fisher-z association',ylabel='Calibration draws',title=f'Starter-only partnerships: comparison with activity-constrained networks\n{valid.sum()} networks; two-sided MC p={primary.iloc[0].mc_two_sided_p:.4g}');save(fig,'global_null_test')
        fig,ax=plt.subplots(figsize=(11,8));ax.scatter(plot.rx,plot.ry,s=12,alpha=.2,color='#2473b8');xx=np.linspace(plot.rx.min(),plot.rx.max(),100);ax.plot(xx,reg['slope']*xx,color='#c65c28');ax.axhline(0,color='gray',lw=.5);ax.axvline(0,color='gray',lw=.5);ax.set(xlabel='Other-starter target divergence relative to network mean',ylabel='Mutual-preference excess relative to network mean',title=f'Starter-pair regression; network fixed effects, equal network weights\nr={reg["within_r"]:.3f}; R²={reg["within_r_squared"]:.1%}; team-cluster p={reg["team_cluster_p"]:.3g}');save(fig,'within_network_regression')
        fig,ax=plt.subplots(figsize=(11,8));ax.scatter(comparison.null_calibrated_fisher_difference_rotation,comparison.null_calibrated_fisher_difference_starters,alpha=.6);limits=[min(comparison.null_calibrated_fisher_difference_rotation.min(),comparison.null_calibrated_fisher_difference_starters.min()),max(comparison.null_calibrated_fisher_difference_rotation.max(),comparison.null_calibrated_fisher_difference_starters.max())];ax.plot(limits,limits,color='gray',ls='--');ax.axhline(0,color='gray',lw=.5);ax.axvline(0,color='gray',lw=.5);ax.set(xlabel='Original rotation-cohort calibrated association',ylabel='Five-starter calibrated association',title='Same team-seasons, different player sets and target sets\nDescriptive comparison; the projection changes the meaning of profiles');save(fig,'rotation_vs_starters')
        fig,ax=plt.subplots(figsize=(11,7));ax.bar(seasons.season,seasons.mean_calibrated_difference,color='#2473b8');ax.axhline(0,color='gray',lw=.5);ax.set(ylabel='Observed minus null Fisher-z association',title='Starter-only association by season\nNegative: similarity tendency; positive: different-profile tendency');save(fig,'season_comparison')
        fig,ax=plt.subplots(figsize=(11,8));ax.scatter(networks.conditional_entropy,networks.eigenvector_hhi,alpha=.6);ax.set(xlabel='Conditional passing entropy, normalized by ln(4)',ylabel='Combined passing eigenvector HHI (five selected starters)',title='Descriptive starter-only passing structure\nHigher entropy: more varied next receiver; higher HHI: more centralized');save(fig,'entropy_and_centrality')
        example=next((g for g in graphs if g[0]=='2021-22' and g[1]=='GSW'),None)
        if example:
            fig,ax=plt.subplots(figsize=(11,9));draw_network(ax,example[2],example[3],example[4],example[5],'Warriors 2021–22: five most frequent starters');fig.text(.5,.03,'Season passes among these players; does not isolate all-five-together minutes.',ha='center');save(fig,'warriors_starter_network')
    report=['# Starter-only passing networks',
        f'Built {len(networks)} complete networks out of {len(summary)} original team-seasons, with {len(pairs)} pair observations. Five players selected by starts FOR THAT TEAM, then minutes in starting appearances as tie-breaker, then player ID. No position constraint imposed. These are the most frequent starters, not necessarily a five-player unit that often played together. Actual starting lineups vary across games.',
        'Pass counts come from full team-filtered raw passing profiles. Raw profiles recover selected starters omitted from historical matrices, particularly traded players. Every edge between players retained in the original matrix is checked for exact agreement. They include passes between selected starters while substitutes were also on court, and passes in games in which a selected player came off the bench. They are NOT filtered to starting appearances or simultaneous five-player minutes. No allocation or scaling by estimated shared minutes was performed. Passing probabilities are renormalized over the other four selected starters; they exclude passes to bench players.',
        'Starter metadata: NBA LeagueDashPlayerStats, StarterBench=Starters, TeamID set separately for every team-season. GP is the number of starts under this filter. Each team table reconciles to exactly five times the team games (72 in 2020–21, 82 thereafter). A traded player may retain a final-team label in a filtered response; the request TeamID and reconciled counts define attribution. Selected players missing from original matrices are recovered from full raw passing profiles; lower-ranked players never replace them. Source audits record these recoveries.',
        '## Results',integrated_table(primary),integrated_table(regressions),integrated_table(seasons),
        'Profiles compare passing to the other THREE selected starters after removing both compared players. Mutual strength is P(i,j)P(j,i), minus its mean from a fixed-row/column-total reference forbidding self-passes. Both profiles and mutual strength are recomputed in every null draw. Global reference uses an equal-network mean Fisher-z correlation; franchise-block bootstrap groups repeated seasons. Regression uses network fixed effects, equal network weights, team-cluster standard errors and two-way team/player-pair sensitivity. The 50-pass sensitivity excludes pairs without adequate observed other-starter pass counts and recalculates eligibility in null draws.',
        f'Randomization: {args.pilot_draws} pilot and {args.null_draws} calibration draws, 20 burn-in sweeps before each stage and two sweeps between samples. Finite-chain inference is approximate; diagnostics and audit matrices retained. With 499 calibration draws, the smallest conservative two-sided Monte Carlo p is 0.004.',
        '## Reading the outputs',
        'starter_networks_all_180.pdf shows every successfully constructed starter network using Fruchterman–Reingold placement. Arrows indicate pass direction, widths counts, node size combined A+A.T eigenvector share. Disconnected isolates are omitted from drawings. Layout distances and node placement are aesthetic, not spatial court positions. Starter selections and exact matrices are saved per team-season. Pass probabilities condition on the other four selected players, omitting passes to everyone else.',
        'starter_analysis_graphs.pdf contains the global reference test; within-network pair regression; original-cohort versus five-starter associations; season averages; descriptive entropy/centrality scatter; and the Warriors 2021–22 starter network.',
        'The original-versus-starter comparison changes BOTH pairs included and passing targets, and uses a separate constrained reference for each matrix. It is not a controlled estimate of a starter effect. Conditional entropy and centrality are derived from the same graph, so their scatter is descriptive, not independent evidence about performance.',
        '## Limits',
        'The historical matrices had a >=10-MPG rotation cohort, but this selection uses actual starts without a minutes cutoff and recovers selected players omitted from historical matrices. Full raw profiles supply all starter-to-starter edges. The chosen five may have little shared court time because of trades or injuries. Other-target profiles now have only three possible destinations, and each network has just ten pair comparisons. Fixed player count does not equal fixed playing opportunity. No causal, scoring-efficiency, or winning claim follows. Starter-only networks do not provide the exact simultaneous-lineup networks discussed earlier.'
    ];(out/'DETAILED_ANALYSIS.md').write_text('\n\n'.join(report))
    atomic_json(dict(created_at=utc_now(),networks=len(networks),requested=len(summary),selection='five largest team-specific start counts; minutes in starts then player ID tie-break',passes='season aggregate induced submatrix; not simultaneous lineup',null_draws=args.null_draws,pilot_draws=args.pilot_draws,seed=args.seed),out/'run_manifest.json')
    print(primary.to_string(index=False),flush=True);print(regressions.to_string(index=False),flush=True);print('Saved '+str(out),flush=True);return 0


# ---------------------------------------------------------------------
# FULL-ROSTER CONDITIONAL ENTROPY VERSUS EIGENVECTOR CONCENTRATION
# ---------------------------------------------------------------------
def run_full_roster_entropy_centrality(argv):
    """Thirty verified full rosters primary; legacy rotation cohorts separately labeled."""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from scipy.stats import linregress,spearmanr,t
    parser=argparse.ArgumentParser(description='Regress team eigenvector concentration on conditional passing entropy')
    parser.add_argument('--full-roster-entropy-centrality',action='store_true')
    parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT_DIR)
    args=parser.parse_args(argv);root=args.output_dir.expanduser();parent=root/'2025-26/team_season_topics';out=root/'full_roster_entropy_centrality';out.mkdir(exist_ok=True,parents=True)
    saved=pd.read_csv(parent/'team_season_metrics.csv');rows=[];sources=[];shares=[]
    if len(saved)!=30 or saved.team.nunique()!=30 or not saved.full_network_profiles.eq(True).all() or saved.missing_profile_count.ne(0).any():raise ValueError('Primary full-roster coverage not confirmed')
    def measure(file,team,season,scope):
        frame=pd.read_csv(file,index_col=0);ids=frame.index.astype(int).tolist()
        if frame.columns.astype(int).tolist()!=ids:raise ValueError('Matrix row/column IDs disagree')
        A=frame.to_numpy(float)
        if not np.isfinite(A).all() or (A<0).any() or np.diag(A).any():raise ValueError('Invalid pass counts')
        metrics,scores,selected,B=eigenvector_passing_concentration(A,ids)
        if not metrics['network_valid']:raise ValueError(scope+' '+season+' '+team+': '+metrics['network_error'])
        receiver,h=compute_team_entropies(B)
        rows.append(dict(scope=scope,team=team,season=season,conditional_entropy=h,receiver_entropy=receiver,eigenvector_concentration=metrics['eigenvector_concentration'],eigenvector_hhi=metrics['eigenvector_hhi'],active_players=len(selected),roster_players=len(ids),isolates_removed=len(ids)-len(selected),recorded_passes=int(B.sum()),top_player_id=metrics['top_eigenvector_player_id'],top_share=metrics['top_eigenvector_share']))
        sources.append(dict(scope=scope,team=team,season=season,path=str(file),sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
        shares.extend(dict(scope=scope,team=team,season=season,player_id=pid,eigenvector_share=value) for pid,value in scores.items())
        return rows[-1]
    for r in saved.itertuples():
        row=measure(parent/'networks'/r.team/'passing_matrix.csv',r.team,'2025-26','full_roster')
        if not np.isclose(row['conditional_entropy'],r.conditional_receiver_entropy) or not np.isclose(row['eigenvector_concentration'],r.eigenvector_concentration):raise ValueError('Recomputed primary metrics disagree with saved full-roster metrics')
    for file in sorted(root.glob('20*/Regular_Season/*/passing_matrix.csv')):measure(file,file.parent.name,file.parts[-4],'legacy_rotation')
    data=pd.DataFrame(rows);models=[]
    def fit(d,y,scope):
        x=d.conditional_entropy.to_numpy(float);values=d[y].to_numpy(float);X=np.column_stack([np.ones(len(d)),x]);inv=np.linalg.inv(X.T@X);beta=inv@X.T@values;res=values-X@beta;hat=np.sum(X*(X@inv),axis=1);u=res/(1-hat);hc3=inv@(X.T@(u[:,None]**2*X))@inv
        result=linregress(x,values);se=np.sqrt(hc3[1,1]);df=len(d)-2;kind='HC3';cov=hc3
        if scope=='legacy_rotation':
            score=pd.DataFrame({'team':d.team.to_numpy(),'s0':res,'s1':res*x}).groupby('team')[['s0','s1']].sum().to_numpy();G=len(score);cov=inv@(score.T@score)@inv*G/(G-1)*(len(d)-1)/(len(d)-2);se=np.sqrt(cov[1,1]);df=G-1;kind='team_cluster'
        robust_p=2*t.sf(abs(beta[1]/se),df);ci=t.ppf(.975,df)*se;rho,sp=spearmanr(x,values)
        record=dict(scope=scope,outcome=y,n=len(d),slope=beta[1],intercept=beta[0],r=result.rvalue,r_squared=result.rvalue**2,ordinary_ols_p=result.pvalue,uncertainty=kind,robust_se=se,robust_p=robust_p,slope_ci_low=beta[1]-ci,slope_ci_high=beta[1]+ci,spearman_rho=rho,spearman_p=sp)
        models.append(record);return record,cov,df
    fitted={}
    for scope,d in data.groupby('scope'):
        for y in ['eigenvector_concentration','eigenvector_hhi']:fitted[scope,y]=fit(d,y,scope)
    results=pd.DataFrame(models);atomic_csv(data,out/'network_metrics.csv');atomic_csv(results,out/'regressions.csv');atomic_csv(pd.DataFrame(sources),out/'source_manifest.csv');atomic_csv(pd.DataFrame(shares),out/'player_eigenvector_shares.csv')
    with PdfPages(out/'entropy_eigenvector_regressions.pdf') as pdf:
        for scope,d in data.groupby('scope',sort=False):
            for y,label in [('eigenvector_concentration','Normalized eigenvector concentration'),('eigenvector_hhi','Raw eigenvector HHI')]:
                model,cov,df=fitted[scope,y];fig,ax=plt.subplots(figsize=(11,8));xx=np.linspace(d.conditional_entropy.min()-.01,d.conditional_entropy.max()+.01,200);D=np.column_stack([np.ones(len(xx)),xx]);pred=model['intercept']+model['slope']*xx;band=t.ppf(.975,df)*np.sqrt(np.sum((D@cov)*D,axis=1));ax.fill_between(xx,pred-band,pred+band,color='#93b9d8',alpha=.3,label='95% interval for fitted mean');ax.plot(xx,pred,color='#c65c28',lw=2,label='OLS line')
                if scope=='full_roster':
                    ax.scatter(d.conditional_entropy,d[y],s=45,color='#2473b8')
                    for k,r in enumerate(d.itertuples()):ax.annotate(r.team,(r.conditional_entropy,getattr(r,y)),xytext=(4,4 if k%2==0 else -10),textcoords='offset points',fontsize=8)
                    title='2025–26: all 30 verified full-roster networks';note='No minutes or roster-count covariates. Isolates omitted. Counts combined as A+Aᵀ for eigenvectors.'
                else:
                    for season,g in d.groupby('season'):ax.scatter(g.conditional_entropy,g[y],s=24,alpha=.7,label=season)
                    title=f'{len(d)} original rotation-cohort networks, 2020–21 to 2025–26';note='Historical ≥10-MPG cohorts; these are NOT full rosters. Uncertainty clustered by franchise.'
                ax.set(xlabel='Conditional passing entropy H(receiver | passer), normalized',ylabel=label,title=title+f'\nr={model["r"]:.3f}; R²={model["r_squared"]:.1%}; {model["uncertainty"]} slope p={model["robust_p"]:.3g}')
                ax.legend(fontsize=8);fig.text(.5,.016,note+'\nBoth axes describe the same passing matrix; this is structural association, not offensive performance.',ha='center',fontsize=9);fig.tight_layout(rect=[0,.06,1,1]);pdf.savefig(fig);stem=scope+'_'+y;fig.savefig(out/(stem+'.png'),dpi=150);fig.savefig(out/(stem+'.pdf'));plt.close(fig)
    primary=results[(results.scope=='full_roster')&(results.outcome=='eigenvector_concentration')].iloc[0];raw=results[(results.scope=='full_roster')&(results.outcome=='eigenvector_hhi')].iloc[0];legacy=results[(results.scope=='legacy_rotation')&(results.outcome=='eigenvector_concentration')].iloc[0]
    report=['# Full-roster passing entropy and eigenvector concentration',
        'Primary analysis uses 30 verified full-roster 2025–26 passing matrices, including traded players and with no minutes cutoff. All profiles marked complete, no missing profiles; metrics independently recomputed from saved matrices and checked against saved full-roster results. The 180 historical rotation-cohort matrices are shown separately and not called full rosters. No new NBA downloads.',
        'Eigenvector centrality gives one score per player. The team-level outcome here is CONCENTRATION of those scores, not an individual player score. Directed passing counts A are combined as A+A.T for weighted eigenvectors, preserving the existing project definition. Eigenvector scores are L1-normalized to shares c. HHI=sum(c²); normalized concentration=(HHI−1/N)/(1−1/N), where N is the active nonisolated player count. Zero concentration means equal centrality, higher means more unequal centrality. Its theoretical share-distribution upper bound is not necessarily attained by a feasible undirected passing graph.',
        'The predictor is directed, outgoing-pass-weighted conditional entropy: H(receiver|passer)/ln(N−1). Higher means a more varied next receiver conditional on the passer. It does not demonstrate uncertainty about actual possession sequences. Isolates are omitted consistently for both metrics. Season matrices aggregate lineups and games, not simultaneous on-court groups.',
        '## Results',integrated_table(results),
        f'Primary full-roster result: r={primary.r:.6f}, R²={primary.r_squared:.6f}, slope={primary.slope:.6f}, HC3 p={primary.robust_p:.6g}; slope 95% interval [{primary.slope_ci_low:.6f}, {primary.slope_ci_high:.6f}]. A 0.10 increase in normalized conditional entropy corresponds to {primary.slope*.1:.6f} change in normalized concentration along the fitted line. This is association, not a causal effect.',
        f'Raw-HHI sensitivity on the SAME full rosters: r={raw.r:.6f}, R²={raw.r_squared:.6f}, HC3 p={raw.robust_p:.6g}. Roster-size normalization materially changes the measured association and must be disclosed. Entropy also uses an N-dependent normalization. No minutes or roster-count covariates are in these regressions; normalization is part of the metric definition.',
        f'Separate legacy rotation comparison: r={legacy.r:.6f}, R²={legacy.r_squared:.6f}, franchise-clustered p={legacy.robust_p:.6g}. These 180 team-seasons use the historical >=10-MPG cohorts; the pooled correlation is descriptive across seasons, and cluster uncertainty groups repeated franchises. It is not a 180-full-roster estimate or a season-fixed-effects estimate.',
        '## Interpretation',
        'Teams with higher conditional passing entropy tend to have more evenly distributed eigenvector centrality. This is a clear structural relationship. Both axes derive from the same passing matrix, and volume, roster composition, shared playing opportunities and metric normalization can contribute. It does not establish more efficient offense, better shot quality, higher wins, independent validation of a balanced style, or a causal mechanism. R² describes variation in centrality concentration in this sample, not variation in scoring or wins. Spearman statistics are exploratory rank-based descriptions; no multiple-testing selection was performed.',
        'The PDF has four plots: verified full rosters with normalized concentration; verified full rosters with raw HHI; historical rotation cohorts with normalized concentration; historical rotation cohorts with raw HHI. Confidence bands describe fitted-mean uncertainty, not prediction intervals for individual teams. Main uncertainty uses HC3 heteroskedasticity-robust errors with t reference (28 df); repeated-franchise legacy uncertainty uses cluster robust errors with 29 df. Ordinary OLS p-values are saved separately and are not interchangeable with robust p-values.',
        'Full code preserves previous CLI and adds --full-roster-entropy-centrality. From a delivered copy in another folder, supply --output-dir /Users/timothysu/Desktop/SportsAnalysis/nba_passing_output.'
    ];(out/'RESULTS_EXPLAINED.md').write_text('\n\n'.join(report));atomic_json(dict(created_at=utc_now(),primary_scope='30 full rosters, 2025-26',secondary_scope='180 legacy rotation cohorts, 2020-21 to 2025-26',x='normalized conditional passing entropy',y='normalized concentration of L1 weighted A+A.T eigenvectors',controls=[],new_downloads=False),out/'run_manifest.json')
    print(results.to_string(index=False),flush=True);print('Saved '+str(out),flush=True);return 0


def rebuild_historical_full_roster_metrics(root):
    """Use raw team-filtered roster responses, retaining traded players regardless of final-team labels."""
    destination=root/'full_roster_entropy_centrality/historical_full_rosters';destination.mkdir(exist_ok=True,parents=True)
    extra=destination/'additional_profiles';extra.mkdir(exist_ok=True);roster_index={};profile_index={}
    for path in list((Path.home()/'nba_passing_cache').glob('*.json'))+list(extra.glob('*.json')):
        snapshot=json.loads(path.read_text());q=snapshot.get('parameters',{})
        if q.get('DateFrom') or q.get('DateTo') or q.get('SeasonType')!='Regular Season' or q.get('PerMode')!='Totals' or q.get('Location') or q.get('Outcome') or q.get('SeasonSegment') or q.get('VsConference') or q.get('VsDivision') or int(q.get('LastNGames',0)) or int(q.get('OpponentTeamID',0)) or int(q.get('Month',0)):continue
        endpoint=snapshot.get('endpoint', 'playerdashptpass' if q.get('PlayerID') else '')
        if endpoint=='leaguedashplayerstats' and q.get('TeamID') and not q.get('StarterBench') and q.get('MeasureType','Base')=='Base':key=(q['Season'],int(q['TeamID']));index=roster_index
        elif endpoint=='playerdashptpass':key=(q['Season'],int(q.get('TeamID',0)),int(q.get('PlayerID',0)));index=profile_index
        else:continue
        stamp=snapshot.get('fetched_at','')
        if key not in index or stamp>index[key][0]:index[key]=(stamp,path)
    summary=pd.read_csv(root/'2020s_summary/Regular_Season/all_seasons_team_metrics.csv').sort_values(['season','team']);metrics_rows=[];profile_sources=[];matrix_sources=[];outside_edges=[]
    for k,r in enumerate(summary.itertuples(),1):
        historical=root/r.season/'Regular_Season'/r.team;oldroster=pd.read_csv(historical/'roster_stats.csv');team_id=int(oldroster.TEAM_ID.iloc[0]);key=(r.season,team_id)
        if key not in roster_index:raise ValueError('Missing full team-filtered roster cache: '+str(key))
        rosterpath=roster_index[key][1];rostersnapshot=json.loads(rosterpath.read_text());frame=extract_team_metrics(rostersnapshot['payload'],'LeagueDashPlayerStats',{'PLAYER_ID','PLAYER_NAME','GP','MIN'})
        # Query TeamID establishes team scope. Returned TEAM_ID can be a traded player's final team.
        roster=frame.loc[frame.GP.gt(0)&frame.MIN.gt(0)].copy().sort_values('PLAYER_ID')
        if roster.PLAYER_ID.duplicated().any():raise ValueError('Duplicate official participants')
        ids=roster.PLAYER_ID.astype(int).tolist();positions={pid:i for i,pid in enumerate(ids)};A=np.zeros((len(ids),len(ids)),int);received=A.copy();empty=[]
        for z in roster.itertuples():
            pid=int(z.PLAYER_ID);key=(r.season,team_id,pid)
            if key not in profile_index:
                query=playerdashptpass.PlayerDashPtPass(player_id=pid,team_id=team_id,season=r.season,season_type_all_star='Regular Season',per_mode_simple='Totals',timeout=20)
                snap=dict(endpoint='playerdashptpass',parameters=query.parameters,fetched_at=utc_now(),payload=query.get_dict());path=extra/(r.season+'_'+r.team+'_'+str(pid)+'.json');atomic_json(snap,path);profile_index[key]=(snap['fetched_at'],path)
            path=profile_index[key][1];snap=json.loads(path.read_text());made=validate_passing(snap['payload'],pid,team_id)
            if made.empty:empty.append(pid)
            for edge in made.itertuples():
                target=int(edge.PASS_TEAMMATE_PLAYER_ID)
                if target in positions:A[positions[pid],positions[target]]+=int(edge.PASS)
                elif edge.PASS>0:outside_edges.append(dict(season=r.season,team=r.team,passer_id=pid,receiver_id=target,passes=int(edge.PASS),reason='receiver outside official positive-minute team roster'))
            incoming=extract_team_metrics(snap['payload'],'PassesReceived',{'PLAYER_ID','TEAM_ID','PASS_TEAMMATE_PLAYER_ID','PASS'})
            if not ((incoming.PLAYER_ID==pid)&(incoming.TEAM_ID==team_id)).all():raise ValueError('Received profile scope mismatch')
            for edge in incoming.itertuples():
                passer=int(edge.PASS_TEAMMATE_PLAYER_ID)
                if passer!=pid and passer in positions:received[positions[passer],positions[pid]]+=int(edge.PASS)
            profile_sources.append(dict(season=r.season,team=r.team,player_id=pid,name=z.PLAYER_NAME,minutes=z.MIN,games=z.GP,roster_source=str(rosterpath),roster_sha256=hashlib.sha256(rosterpath.read_bytes()).hexdigest(),profile_source=str(path),profile_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),empty_outgoing_profile=made.empty))
        if not np.array_equal(A,received):raise ValueError('Full roster passes-made and passes-received disagree: '+r.season+' '+r.team)
        original=pd.read_csv(historical/'passing_matrix.csv',index_col=0);original.index=original.index.astype(int);original.columns=original.columns.astype(int);common=[pid for pid in ids if pid in original.index];ix=[positions[pid] for pid in common]
        if not np.array_equal(A[np.ix_(ix,ix)],original.loc[common,common].to_numpy()):raise ValueError('Retained historical edges changed')
        folder=destination/r.season/r.team;folder.mkdir(exist_ok=True,parents=True);matrixpath=folder/'passing_matrix.csv';atomic_csv(pd.DataFrame(A,index=ids,columns=ids),matrixpath,index=True);atomic_csv(roster,folder/'roster.csv')
        metrics,scores,selected,B=eigenvector_passing_concentration(A,ids)
        if not metrics['network_valid']:raise ValueError('Invalid full network: '+metrics['network_error'])
        receiver,h=compute_team_entropies(B);outgoing_empty_confirmed=sum(received[positions[pid]].sum()==0 for pid in empty)
        metrics_rows.append(dict(scope='full_roster',season=r.season,team=r.team,conditional_entropy=h,receiver_entropy=receiver,eigenvector_concentration=metrics['eigenvector_concentration'],eigenvector_hhi=metrics['eigenvector_hhi'],roster_players=len(ids),active_players=len(selected),isolates_removed=len(ids)-len(selected),recorded_passes=int(A.sum()),empty_outgoing_profiles=len(empty),empty_rows_confirmed_by_receivers=outgoing_empty_confirmed,received_mismatch_cells=0,retained_edge_mismatches=0))
        matrix_sources.append(dict(scope='full_roster',season=r.season,team=r.team,path=str(matrixpath),sha256=hashlib.sha256(matrixpath.read_bytes()).hexdigest()))
        if k%30==0:print(f'Full-roster reconstruction {k}/{len(summary)} validated',flush=True)
    result=pd.DataFrame(metrics_rows)
    saved=pd.read_csv(root/'full_roster_entropy_centrality/network_metrics.csv');saved=saved.loc[saved.scope.eq('full_roster')];check=result.loc[result.season.eq('2025-26')].merge(saved,on='team',suffixes=('_rebuilt','_saved'))
    if len(check)!=30 or not np.allclose(check.conditional_entropy_rebuilt,check.conditional_entropy_saved) or not np.allclose(check.eigenvector_concentration_rebuilt,check.eigenvector_concentration_saved):raise ValueError('2025–26 full-roster reproduction failed')
    atomic_csv(result,destination/'network_metrics.csv');atomic_csv(pd.DataFrame(profile_sources),destination/'profile_sources.csv');atomic_csv(pd.DataFrame(matrix_sources),destination/'source_manifest.csv');atomic_csv(pd.DataFrame(outside_edges,columns=['season','team','passer_id','receiver_id','passes','reason']),destination/'excluded_outside_roster_edges.csv');return result


# ---------------------------------------------------------------------
# SEPARATE SEASON REGRESSIONS: CONDITIONAL ENTROPY AND CENTRALITY
# ---------------------------------------------------------------------
def run_season_entropy_centrality(argv):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from scipy.stats import linregress,t
    parser=argparse.ArgumentParser(description='Separate season entropy/eigenvector-concentration regressions')
    parser.add_argument('--season-entropy-centrality',action='store_true');parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT_DIR)
    args=parser.parse_args(argv);root=args.output_dir.expanduser();source=root/'full_roster_entropy_centrality';out=source/'by_season';out.mkdir(exist_ok=True,parents=True)
    if not (source/'network_metrics.csv').exists():run_full_roster_entropy_centrality(['--full-roster-entropy-centrality','--output-dir',str(root)])
    manifest=pd.read_csv(source/'source_manifest.csv')
    if any(hashlib.sha256(Path(r.path).read_bytes()).hexdigest()!=r.sha256 for r in manifest.itertuples()):
        run_full_roster_entropy_centrality(['--full-roster-entropy-centrality','--output-dir',str(root)])
    data=pd.read_csv(source/'network_metrics.csv');legacy=data[data.scope.eq('legacy_rotation')].copy();full=rebuild_historical_full_roster_metrics(root);data=pd.concat([full,legacy],ignore_index=True);models=[];fitted={}
    def fit(d,y,scope,season):
        x=d.conditional_entropy.to_numpy(float);z=d[y].to_numpy(float);X=np.column_stack([np.ones(len(d)),x]);bread=np.linalg.inv(X.T@X);b=bread@X.T@z;e=z-X@b;h=np.sum(X*(X@bread),axis=1);u=e/(1-h);cov=bread@(X.T@(X*u[:,None]**2))@bread;se=np.sqrt(cov[1,1]);df=len(d)-2;q=linregress(x,z);ci=t.ppf(.975,df)*se
        result=dict(scope=scope,season=season,outcome=y,n=len(d),slope=b[1],intercept=b[0],r=q.rvalue,r_squared=q.rvalue**2,ols_p=q.pvalue,hc3_se=se,hc3_p=2*t.sf(abs(b[1]/se),df),slope_ci_low=b[1]-ci,slope_ci_high=b[1]+ci)
        models.append(result);fitted[scope,season,y]=(result,cov,df)
    seasons=sorted(legacy.season.unique())
    if len(seasons)!=6 or not legacy.groupby('season').size().eq(30).all():raise ValueError('Expected six original seasons with 30 networks each')
    for season,g in legacy.groupby('season'):
        for y in ['eigenvector_concentration','eigenvector_hhi']:fit(g,y,'legacy_rotation',season)
    for season,g in full.groupby('season'):
        for y in ['eigenvector_concentration','eigenvector_hhi']:fit(g,y,'full_roster',season)
    results=pd.DataFrame(models);results['holm_p_within_six_seasons']=np.nan
    for scope in ['full_roster','legacy_rotation']:
        for y in ['eigenvector_concentration','eigenvector_hhi']:
            index=results.index[results.scope.eq(scope)&results.outcome.eq(y)];order=results.loc[index].hc3_p.sort_values().index;adjusted=np.maximum.accumulate([(len(order)-j)*results.loc[i,'hc3_p'] for j,i in enumerate(order)]);results.loc[order,'holm_p_within_six_seasons']=np.minimum(adjusted,1)
    atomic_csv(results,out/'season_regressions.csv');atomic_csv(data,out/'network_metrics.csv')
    def draw(ax,d,scope,season,y,labels=False,common=False):
        model,cov,df=fitted[scope,season,y];xx=np.linspace(d.conditional_entropy.min(),d.conditional_entropy.max(),120);X=np.column_stack([np.ones(len(xx)),xx]);pred=X@np.array([model['intercept'],model['slope']]);band=t.ppf(.975,df)*np.sqrt(np.sum(X*(X@cov),axis=1));ax.fill_between(xx,pred-band,pred+band,color='#91b8d4',alpha=.25);ax.plot(xx,pred,color='#c65c28',lw=1.8);ax.scatter(d.conditional_entropy,d[y],s=26 if common else 38,color='#2473b8',alpha=.85)
        if labels:
            for k,r in enumerate(d.itertuples()):ax.annotate(r.team,(r.conditional_entropy,getattr(r,y)),xytext=(3,4 if k%2==0 else -9),textcoords='offset points',fontsize=7)
        ax.set_title(f'{season} · n={len(d)}\nr={model["r"]:.3f}; R²={model["r_squared"]:.1%}; HC3 p={model["hc3_p"]:.3g}',fontsize=10 if common else 13)
        ax.set_xlabel('Conditional passing entropy',fontsize=9 if common else 11);ax.set_ylabel('Normalized EC concentration' if y=='eigenvector_concentration' else 'Raw eigenvector HHI',fontsize=9 if common else 11)
        if common:
            population=full if scope=='full_roster' else legacy;ax.set_xlim(population.conditional_entropy.min()-.015,population.conditional_entropy.max()+.015);span=population[y].max()-population[y].min();ax.set_ylim(max(0,population[y].min()-.15*span),population[y].max()+.15*span)
    with PdfPages(out/'season_regressions.pdf') as pdf:
        for y in ['eigenvector_concentration','eigenvector_hhi']:
            fig,axes=plt.subplots(2,3,figsize=(16,10))
            for ax,season in zip(axes.flat,seasons):draw(ax,full[full.season.eq(season)],'full_roster',season,y,common=True)
            fig.suptitle('Separate season regressions: full-roster passing networks',fontsize=16);fig.text(.5,.012,'All official positive-minute participants, including traded players; no minutes cutoff or covariates.\nFixed axes across seasons; 95% HC3 fitted-mean bands. Both metrics describe the same passing graph.',ha='center',fontsize=10);fig.tight_layout(rect=[0,.055,1,.95]);pdf.savefig(fig);fig.savefig(out/('six_seasons_'+y+'.png'),dpi=150);fig.savefig(out/('six_seasons_'+y+'.pdf'));plt.close(fig)
        for season,d in full.groupby('season'):
            for y in ['eigenvector_concentration','eigenvector_hhi']:
                fig,ax=plt.subplots(figsize=(11,8));draw(ax,d,'full_roster',season,y,labels=True);fig.suptitle('Verified full-roster coverage, including traded players; no minutes cutoff',fontsize=13);fig.text(.5,.015,'No covariates. Bands show fitted-mean uncertainty, not individual-team prediction intervals.\nA structural association does not establish better offense or causality.',ha='center',fontsize=9);fig.tight_layout(rect=[0,.06,1,.94]);pdf.savefig(fig);stem='full_roster_'+season+'_'+y;fig.savefig(out/(stem+'.png'),dpi=150);fig.savefig(out/(stem+'.pdf'));plt.close(fig)
        for y in ['eigenvector_concentration','eigenvector_hhi']:
            fig,axes=plt.subplots(2,3,figsize=(16,10))
            for ax,season in zip(axes.flat,seasons):draw(ax,legacy[legacy.season.eq(season)],'legacy_rotation',season,y,common=True)
            fig.suptitle('Separate REFERENCE: original rotation cohorts, not full rosters',fontsize=16);fig.text(.5,.015,'Historical ≥10-MPG selection; separate six-test family for each outcome.',ha='center');fig.tight_layout(rect=[0,.04,1,.95]);pdf.savefig(fig);fig.savefig(out/('legacy_six_seasons_'+y+'.png'),dpi=150);fig.savefig(out/('legacy_six_seasons_'+y+'.pdf'));plt.close(fig)
    primary=results[results.outcome.eq('eigenvector_concentration')&results.scope.eq('full_roster')];raw=results[results.outcome.eq('eigenvector_hhi')&results.scope.eq('full_roster')];comparison=results[results.scope.eq('legacy_rotation')]
    report=['# Full-roster conditional passing entropy versus centrality: each season',
        'Six separate full-roster fits, one per season 2020–21 through 2025–26, with 30 teams each. Historical full rosters reconstructed from cached team-filtered NBA roster responses and passing profiles, including traded players despite final-team labels in roster rows. No minutes cutoff. Original rotation-cohort results are retained separately as references.',
        '## Full-roster normalized concentration',integrated_table(primary),
        '## Full-roster raw-HHI sensitivity',integrated_table(raw),
        '## Separate original rotation reference',integrated_table(comparison),
        'Reconstruction includes every official player with GP>0 and total MIN>0 in that team-season. Raw roster query TeamID establishes team attribution; returned player labels may name a final team after a trade. Nine uncached very-short-appearance profiles were checked against NBA. Empty outgoing profiles are retained as zero rows only after comparison with every other player’s PassesReceived records. All retained directed edges agree exactly between made and received views, and all edges common to original matrices agree. Six empty outgoing profiles across six team-seasons are independently corroborated by receiver profiles. Exactly one recorded pass to a player outside that team’s official roster was excluded and documented (2024–25 LAL, LeBron James to Mike Conley). Rebuilt 2025–26 metrics reproduce the previous verified full-roster results.',
        'X is directed conditional passing entropy H(receiver|passer)/ln(N−1). Y is concentration of weighted A+A.T eigenvector shares: (sum(c²)−1/N)/(1−1/N). Raw HHI=sum(c²) is a separate sensitivity. Isolates are omitted consistently. Higher X means broader next-receiver distributions conditional on the passer; higher Y means more centrality concentrated in fewer players.',
        'Each model is unadjusted OLS with an intercept and HC3 robust uncertainty (t reference, 28 df). There are no minutes or player-count covariates. Each season has one observation per team; teams recur BETWEEN fits, so the six seasons are not independent replications. Holm adjusts six slope tests separately within each scope/outcome family: full normalized, full raw, legacy normalized, legacy raw. No selection of only favorable seasons.',
        'PDF: two six-panel full-roster overview plots, twelve full-roster individual season plots, then two explicitly separate legacy rotation overviews. Fixed axes within overview figures. Individual pages label teams. Confidence bands describe uncertainty in the fitted mean, not individual-team prediction intervals. CSV includes slopes, intercepts, r, R², ordinary OLS p, HC3 errors and p, slope intervals and corrected p.',
        'Both variables derive from the same passing graph. Exposure, roster composition and N-dependent normalization may contribute. Compare the raw-HHI sensitivity before describing the association as independent of normalization. This is not a test of scoring, winning, causal effects or superior offense. Differences across seasons are descriptive; no formal between-season slope-difference test was performed.',
        'Profile provenance, hashes, official roster rows, all full matrices, zero-profile checks and outside-roster exclusions are saved under historical_full_rosters. Full code preserves previous commands and adds --season-entropy-centrality. The reconstruction uses the existing ~/nba_passing_cache plus the saved additional_profiles directory; missing profiles are requested from NBA rather than silently invented.'
    ];(out/'RESULTS_EXPLAINED.md').write_text('\n\n'.join(report));atomic_json(dict(created_at=utc_now(),season_fits=6,teams_each=30,primary_scope='full rosters reconstructed from official team-filtered responses',secondary_scope='legacy rotation cohorts',controls=[],standard_errors='HC3',multiple_testing='Holm within each six-season scope/outcome family'),out/'run_manifest.json')
    print(results.to_string(index=False),flush=True);print('Saved '+str(out),flush=True);return 0


# ---------------------------------------------------------------------
# EIGHT FULL-ROSTER ROLE, IDENTITY AND PERFORMANCE STUDIES
# ---------------------------------------------------------------------
def full_eight_hhi(values,normalize=False):
    v=np.asarray(values,float);total=v.sum()
    if total<=0:return np.nan
    h=float(np.sum((v/total)**2));return (h-1/len(v))/(1-1/len(v)) if normalize and len(v)>1 else h


def full_eight_fit(data,x,y,label,fixed=None,controls=None,equal_group=None,two_way=False):
    """FWL association, cluster inference; repeated team-pair intersections aggregated."""
    from scipy.stats import t
    fixed=fixed or [];controls=controls or [];required=list(dict.fromkeys([x,y,'team',*fixed,*controls,*(['pair_id'] if two_way else []),*([equal_group] if equal_group else [])]));d=data.dropna(subset=required).copy()
    Z=[np.ones((len(d),1))]
    if controls:Z.append(d[controls].to_numpy(float))
    for key in fixed:Z.append(pd.get_dummies(d[key],drop_first=True,dtype=float).to_numpy())
    Z=np.column_stack(Z);xx=d[x].to_numpy(float);yy=d[y].to_numpy(float);X=np.column_stack([Z[:,0],xx,Z[:,1:]]);w=1/d.groupby(equal_group)[equal_group].transform('size').to_numpy(float) if equal_group else np.ones(len(d));sw=np.sqrt(w);rank=np.linalg.matrix_rank(X*sw[:,None]);beta=np.linalg.lstsq(X*sw[:,None],yy*sw,rcond=None)[0];bread=np.linalg.pinv(X.T@(w[:,None]*X));res=yy-X@beta;score=w[:,None]*X*res[:,None]
    def cluster(keys):
        temp=d[keys].reset_index(drop=True).copy();temp['row']=np.arange(len(d));groups=temp.groupby(keys).row.apply(list);s=np.stack([score[index].sum(0) for index in groups]);G=len(s);return (s.T@s)*G/(G-1)*(len(d)-1)/(len(d)-rank),G
    meat,G=cluster(['team']);cov=bread@meat@bread;team_se=float(np.sqrt(max(0,cov[1,1])));df=G-1;p=2*t.sf(abs(beta[1]/team_se),df) if team_se>0 else np.nan;two_se=two_p=np.nan
    if two_way:
        other,H=cluster(['pair_id']);intersection,I=cluster(['team','pair_id']);var=(bread@(meat+other-intersection)@bread)[1,1]
        if var>0:two_se=float(np.sqrt(var));two_p=float(2*t.sf(abs(beta[1]/two_se),min(G,H)-1))
    rx=xx-Z@np.linalg.lstsq(Z*sw[:,None],xx*sw,rcond=None)[0];ry=yy-Z@np.linalg.lstsq(Z*sw[:,None],yy*sw,rcond=None)[0];r=float(np.sum(w*rx*ry)/np.sqrt(np.sum(w*rx**2)*np.sum(w*ry**2)))
    d['x_residual']=rx;d['y_residual']=ry;d['analysis_weight']=w
    result=dict(model=label,x=x,y=y,n=len(d),teams=G,slope=float(beta[1]),partial_r=r,partial_r_squared=r*r,team_cluster_se=team_se,p_value=float(p),two_way_se=two_se,two_way_p=two_p,slope_ci_low=float(beta[1]-t.ppf(.975,df)*team_se),slope_ci_high=float(beta[1]+t.ppf(.975,df)*team_se),fixed_effects='+'.join(fixed),covariates='+'.join(controls),equal_weight_groups=equal_group or '')
    return result,d


def full_eight_context(root,out,pilot,draws,seed):
    """Build assist edges, role metrics and null-adjusted mutual pairs from validated full rosters."""
    from scipy.stats import spearmanr
    historical=root/'full_roster_entropy_centrality/historical_full_rosters';metric=pd.read_csv(historical/'network_metrics.csv');sources=pd.read_csv(historical/'profile_sources.csv');nets={};teamrows=[];players=[];edges=[];pairrows=[];audits=[];rosters={};seasons=sorted(metric.season.unique());statsdir=out/'team_stats';statsdir.mkdir(exist_ok=True,parents=True);nullcache=out/'null_cache';nullcache.mkdir(exist_ok=True)
    stats={}
    for season in seasons:
        for measure in ['Base','Advanced']:
            file=statsdir/(season+'_'+measure+'.csv')
            if not file.exists():
                from nba_api.stats.endpoints import leaguedashteamstats
                query=leaguedashteamstats.LeagueDashTeamStats(season=season,season_type_all_star='Regular Season',measure_type_detailed_defense=measure,per_mode_detailed='Totals',timeout=20);f=query.get_data_frames()[0]
                atomic_csv(f,file);atomic_json(dict(parameters=query.parameters,fetched_at=utc_now(),payload=query.get_dict()),file.with_suffix('.json'))
            f=pd.read_csv(file)
            if len(f)!=30 or not f.GP.eq(72 if season=='2020-21' else 82).all():raise ValueError('Incomplete official season totals')
            stats[season,measure]=f.set_index('TEAM_ID')
    for k,r in enumerate(metric.sort_values(['season','team']).itertuples(),1):
        season,team=r.season,r.team;folder=historical/season/team;frame=pd.read_csv(folder/'passing_matrix.csv',index_col=0);ids=frame.index.astype(int).tolist();A=frame.to_numpy(float);N=len(ids);index={pid:i for i,pid in enumerate(ids)};rs=pd.read_csv(folder/'roster.csv').set_index('PLAYER_ID').loc[ids];rosters[season,team]=rs;tid=int(pd.read_csv(root/season/'Regular_Season'/team/'roster_stats.csv').TEAM_ID.iloc[0]);S=np.zeros_like(A)
        own=sources[(sources.season==season)&(sources.team==team)]
        if len(own)!=N:raise ValueError('Full profile manifest mismatch')
        for z in own.itertuples():
            path=Path(z.profile_source)
            if hashlib.sha256(path.read_bytes()).hexdigest()!=z.profile_sha256:raise ValueError('Profile content changed')
            payload=json.loads(path.read_text())['payload'];f=validate_passing(payload,int(z.player_id),tid)
            for e in f.itertuples():
                pid=int(e.PASS_TEAMMATE_PLAYER_ID)
                if pid in index:S[index[int(z.player_id)],index[pid]]+=float(e.AST)
        if (S<0).any() or np.diag(S).any() or (S>A).any():raise ValueError('Invalid assist edge counts')
        base=stats[season,'Base'].loc[tid];advanced=stats[season,'Advanced'].loc[tid];box_ast=float(rs.AST.sum());box_fga=float(rs.FGA.sum());box_minutes=float(rs.MIN.sum())
        if not np.isclose(box_ast,base.AST) or not np.isclose(box_fga,base.FGA):raise ValueError('Player totals do not reconcile to official team AST/FGA')
        O=A.sum(1);R=A.sum(0);P=np.divide(A,O[:,None],out=np.zeros_like(A),where=O[:,None]>0);m,scores,active,B=eigenvector_passing_concentration(A,ids);c=np.array([scores.get(pid,0) for pid in ids]);order=np.argsort(-c,kind='stable');top1,top2=order[:2];gap=float((c[top1]-c[top2])/(c[top1]+c[top2]));role=rs.AST.to_numpy()/box_ast-rs.FGA.to_numpy()/box_fga;scale=np.sqrt(np.mean(role**2));zrole=role/scale
        def perron(M):
            vals,vec=np.linalg.eig(M);radius=float(np.max(np.abs(vals)));candidate=np.flatnonzero(np.abs(vals-radius)<1e-7*max(1,radius))
            if len(candidate)!=1:raise ValueError('Nonunique directed Perron eigenvector')
            v=np.real(vec[:,candidate[0]]);v*=1 if v.sum()>0 else -1
            if v.min()<-1e-7:raise ValueError('Invalid Perron sign')
            v=np.maximum(v,0);v/=v.sum()
            if not np.allclose(M@v,radius*v,atol=1e-7):raise ValueError('Directed eigenvector residual')
            return v
        distributed=perron(A);received=perron(A.T);sep=float(.5*np.abs(distributed-received).sum());volume_sep=float(.5*np.abs(O/A.sum()-R/A.sum()).sum());reciprocity=float(np.sum(P*P.T)/max(1,(O>0).sum()));fga_hhi=full_eight_hhi(rs.FGA);fga_norm=full_eight_hhi(rs.FGA,True)
        teamrows.append(dict(network=season+'_'+team,season=season,team=team,team_id=tid,roster_players=N,active_players=len(active),conditional_entropy=r.conditional_entropy,eigenvector_concentration=m['eigenvector_concentration'],eigenvector_hhi=m['eigenvector_hhi'],top1_id=ids[top1],top1_name=rs.iloc[top1].PLAYER_NAME,top1_share=c[top1],top2_id=ids[top2],top2_name=rs.iloc[top2].PLAYER_NAME,top2_share=c[top2],top2_gap=gap,top2_combined=c[top1]+c[top2],directional_ec_separation=sep,volume_role_separation=volume_sep,reciprocity=reciprocity,shooting_hhi=fga_hhi,shooting_concentration=fga_norm,minutes_hhi=full_eight_hhi(rs.MIN),minutes_concentration=full_eight_hhi(rs.MIN,True),passes=float(A.sum()),tracked_assists=float(S.sum()),tracked_assist_coverage=S.sum()/box_ast,assists_per100passes=100*S.sum()/A.sum(),ast_to=base.AST/base.TOV,off_rating=advanced.OFF_RATING,turnovers_per100=100*base.TOV/advanced.POSS,passes_per100=100*A.sum()/advanced.POSS,official_assists=base.AST,official_fga=base.FGA,minutes=box_minutes))
        names=rs.PLAYER_NAME.to_dict()
        for i,pid in enumerate(ids):players.append(dict(network=season+'_'+team,season=season,team=team,player_id=pid,name=names[pid],minutes=rs.iloc[i].MIN,games=rs.iloc[i].GP,assists=rs.iloc[i].AST,fga=rs.iloc[i].FGA,creator_finisher_role=role[i],standardized_role=zrole[i],combined_ec=c[i],outgoing_ec=distributed[i],incoming_ec=received[i],directional_role=distributed[i]-received[i],passes_made=O[i],passes_received=R[i]))
        for i,j in zip(*np.where(A>0)):edges.append(dict(network=season+'_'+team,season=season,team=team,passer_id=ids[i],receiver_id=ids[j],passer=names[ids[i]],receiver=names[ids[j]],passes=A[i,j],assists=S[i,j],assists_per100passes=100*S[i,j]/A[i,j]))
        d=original_vectorized_partnership_null(A,draws,pilot,seed+k,nullcache/(season+'_'+team+'.npz'))
        for n,(i,j) in enumerate(zip(d['pair_i'],d['pair_j'])):
            pid,qid=sorted([ids[i],ids[j]]);pairrows.append(dict(network=season+'_'+team,season=season,team=team,pair_id=str(pid)+'_'+str(qid),player_1=pid,player_2=qid,name_1=names[pid],name_2=names[qid],role_complementarity=-zrole[i]*zrole[j],role_distance=abs(zrole[i]-zrole[j]),opposite_roles=bool(role[i]*role[j]<0),mutual_preference=d['mutual'][n],null_expected=d['null_expected'][n],null_expected_se=d['pilot_expected_se'][n],mutual_excess=d['mutual'][n]-d['null_expected'][n],minimum_player_passes=min(O[i],O[j])))
        audits.append(dict(network=season+'_'+team,full_roster_players=N,player_assists=box_ast,official_assists=base.AST,player_fga=box_fga,official_fga=base.FGA,tracked_assists=S.sum(),tracked_ast_minus_box=S.sum()-base.AST,null_acceptance=float(d['acceptance']),null_lag1=float(d['lag1']),pass_count=int(A.sum())))
        nets[season,team]=(ids,A,S,c,distributed,received)
        if k%10==0:print(f'Eight-study preparation {k}/180 networks complete',flush=True)
    teamdata=pd.DataFrame(teamrows);playerdata=pd.DataFrame(players);edgedata=pd.DataFrame(edges);pairs=pd.DataFrame(pairrows)
    for stem,d in [('team_metrics',teamdata),('player_roles',playerdata),('directed_edges',edgedata),('mutual_pairs',pairs),('data_audit',pd.DataFrame(audits))]:atomic_csv(d,out/(stem+'.csv'))
    return dict(teams=teamdata,players=playerdata,edges=edgedata,pairs=pairs,networks=nets,rosters=rosters,seasons=seasons)


def run_eight_full_roster_studies(argv):
    from scipy.spatial.distance import jensenshannon
    from scipy.stats import spearmanr
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import networkx as nx
    parser=argparse.ArgumentParser(description='Eight full-roster passing role and identity studies')
    parser.add_argument('--eight-full-roster-studies',action='store_true');parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT_DIR);parser.add_argument('--pilot-draws',type=int,default=399);parser.add_argument('--null-draws',type=int,default=199);parser.add_argument('--seed',type=int,default=20261114)
    args=parser.parse_args(argv)
    if min(args.pilot_draws,args.null_draws)<99:raise ValueError('Use at least99 null draws')
    root=args.output_dir.expanduser();out=root/'eight_full_roster_studies';out.mkdir(exist_ok=True,parents=True)
    ctx=full_eight_context(root,out,args.pilot_draws,args.null_draws,args.seed);teams=ctx['teams'];players=ctx['players'];edges=ctx['edges'];pairs=ctx['pairs'];models=[];primary=[];reports=[];plot_specs=[]
    def model(study,d,x,y,label,fixed=None,controls=None,equal_group=None,two_way=False,is_primary=False):
        result,plot=full_eight_fit(d,x,y,label,fixed,controls,equal_group,two_way);result['study']=study;result['primary']=is_primary;models.append(result);folder=out/f'study_{study}';folder.mkdir(exist_ok=True);atomic_csv(plot,folder/(label+'_plot_data.csv'));plot_specs.append((study,label,result,plot))
        if is_primary:primary.append(result.copy())
        return result,plot
    # Studies are specified before results: six primary tests; ideas4 and8 descriptive.
    model(1,pairs,'role_complementarity','mutual_excess','creator_finisher_partnerships',['network'],equal_group='network',two_way=True,is_primary=True)
    model(1,pairs[pairs.minimum_player_passes.ge(50)],'role_complementarity','mutual_excess','creator_finisher_min50',['network'],equal_group='network',two_way=True)
    model(1,pairs,'role_distance','mutual_excess','role_distance_sensitivity',['network'],equal_group='network',two_way=True)
    # Returning-player preferences compared on the same retained receiving targets.
    continuity=[];source_changes=[];excluded=[]
    for old,new in zip(ctx['seasons'],ctx['seasons'][1:]):
        for team in sorted(teams.team.unique()):
            ids0,A,*_=ctx['networks'][old,team];ids1,B,*_=ctx['networks'][new,team];common=sorted(set(ids0)&set(ids1));rs0=ctx['rosters'][old,team];rs1=ctx['rosters'][new,team];a=A[np.ix_([ids0.index(pid) for pid in common],[ids0.index(pid) for pid in common])];b=B[np.ix_([ids1.index(pid) for pid in common],[ids1.index(pid) for pid in common])];good=(a.sum(1)>0)&(b.sum(1)>0);values=[];values50=[]
            for i,pid in enumerate(common):
                if not good[i]:continue
                js=float(jensenshannon(a[i],b[i],base=2)**2);values.append(js)
                if min(a[i].sum(),b[i].sum())>=50:values50.append(js)
                source_changes.append(dict(team=team,transition=old+'→'+new,player_id=pid,preference_change=js,passes_old=a[i].sum(),passes_new=b[i].sum(),retained_targets=len(common)))
            if len(values)<3:excluded.append(dict(team=team,transition=old+'→'+new,reason='fewer than3 matched positive-source rows'));continue
            before=teams[(teams.team==team)&(teams.season==old)].iloc[0];after=teams[(teams.team==team)&(teams.season==new)].iloc[0]
            continuity.append(dict(team=team,transition=old+'→'+new,retained_players=len(common),matched_sources=len(values),retention_old=rs0.loc[common,'MIN'].sum()/rs0.MIN.sum(),retention_new=rs1.loc[common,'MIN'].sum()/rs1.MIN.sum(),retention=.5*(rs0.loc[common,'MIN'].sum()/rs0.MIN.sum()+rs1.loc[common,'MIN'].sum()/rs1.MIN.sum()),preference_change=np.mean(values),preference_change_min50=np.mean(values50) if len(values50)>=3 else np.nan,matched_sources_min50=len(values50),centrality_change=abs(after.eigenvector_concentration-before.eigenvector_concentration)))
    continuity=pd.DataFrame(continuity);folder=out/'study_2';folder.mkdir(exist_ok=True);atomic_csv(continuity,folder/'team_transitions.csv');atomic_csv(pd.DataFrame(source_changes),folder/'returning_player_changes.csv');atomic_csv(pd.DataFrame(excluded,columns=['team','transition','reason']),folder/'excluded_transitions.csv')
    model(2,continuity,'retention','preference_change','roster_identity',['team','transition'],is_primary=True)
    model(2,continuity,'retention','preference_change_min50','roster_identity_min50',['team','transition'])
    model(2,continuity,'retention','centrality_change','roster_centrality_change',['team','transition'])
    # Top-two hierarchy; concentration held constant in the performance model.
    model(3,teams,'top2_gap','ast_to','top_two_hierarchy_ast_to',['team','season'],['eigenvector_concentration'],is_primary=True)
    model(3,teams,'top2_gap','off_rating','top_two_hierarchy_offense',['team','season'],['eigenvector_concentration'])
    model(3,teams,'top2_gap','turnovers_per100','top_two_hierarchy_turnovers',['team','season'],['eigenvector_concentration'])
    # Assist yield: descriptive, all positive-pass edges retained; fixed100-pass shrinkage display.
    folder=out/'study_4';folder.mkdir(exist_ok=True);seasonrate=edges.groupby('season').assists.sum()/edges.groupby('season').passes.sum();edges=edges.copy();edges['season_reference_rate']=edges.season.map(seasonrate);edges['shrunk_assists_per100passes']=100*(edges.assists+100*edges.season_reference_rate)/(edges.passes+100);atomic_csv(edges,folder/'pass_to_assist_yields.csv');atomic_csv(teams[['season','team','passes','tracked_assists','assists_per100passes','tracked_assist_coverage']],folder/'team_pass_to_assist_yields.csv');examples=edges[edges.passes.ge(100)].sort_values('shrunk_assists_per100passes',ascending=False).head(30);atomic_csv(examples,folder/'high_yield_examples_min100passes.csv')
    # Directional EC separation, not combined prominence or pure volume.
    model(5,teams,'directional_ec_separation','ast_to','directional_roles_ast_to',['team','season'],is_primary=True)
    model(5,teams,'volume_role_separation','ast_to','volume_roles_ast_to',['team','season'])
    model(5,teams,'directional_ec_separation','off_rating','directional_roles_offense',['team','season'])
    # Passing concentration versus shooting concentration; raw-HHI sensitivity is explicit.
    model(6,teams,'eigenvector_concentration','shooting_concentration','passing_shooting_concentration',['team','season'],is_primary=True)
    model(6,teams,'eigenvector_hhi','shooting_hhi','passing_shooting_raw_hhi',['team','season'])
    model(6,teams,'eigenvector_concentration','shooting_concentration','passing_shooting_minutes_sensitivity',['team','season'],['minutes_concentration'])
    model(6,teams,'eigenvector_hhi','shooting_hhi','passing_shooting_raw_minutes_sensitivity',['team','season'],['minutes_hhi'])
    folder=out/'study_6';folder.mkdir(exist_ok=True);quadrants=teams.copy();quadrants['passing_percentile']=quadrants.groupby('season').eigenvector_concentration.rank(pct=True);quadrants['shooting_percentile']=quadrants.groupby('season').shooting_concentration.rank(pct=True);quadrants['passing_group']=np.where(quadrants.passing_percentile>.5,'more concentrated','more distributed');quadrants['shooting_group']=np.where(quadrants.shooting_percentile>.5,'more concentrated','more distributed');atomic_csv(quadrants,folder/'passing_shooting_profiles.csv')
    # Same franchise + same player pair in consecutive seasons; transition FE prevents level artifacts.
    persistence=[]
    for old,new in zip(ctx['seasons'],ctx['seasons'][1:]):
        a=pairs[pairs.season.eq(old)];b=pairs[pairs.season.eq(new)];joined=a.merge(b,on=['team','pair_id'],suffixes=('_old','_new'))
        for z in joined.itertuples():persistence.append(dict(team=z.team,pair_id=z.pair_id,transition=old+'→'+new,network_transition=z.team+'_'+old+'_'+new,name_1=z.name_1_old,name_2=z.name_2_old,previous_excess=z.mutual_excess_old,next_excess=z.mutual_excess_new,previous_mutual=z.mutual_preference_old,next_mutual=z.mutual_preference_new,minimum_passes=min(z.minimum_player_passes_old,z.minimum_player_passes_new)))
    persistence=pd.DataFrame(persistence);folder=out/'study_7';folder.mkdir(exist_ok=True);atomic_csv(persistence,folder/'returning_pair_transitions.csv')
    model(7,persistence,'previous_excess','next_excess','partnership_persistence',['network_transition'],equal_group='network_transition',two_way=True,is_primary=True)
    model(7,persistence[persistence.minimum_passes.ge(50)],'previous_excess','next_excess','partnership_persistence_min50',['network_transition'],equal_group='network_transition',two_way=True)
    # Same-entropy cases selected WITHOUT using performance or passing-to-assist outcomes.
    cases=[];features=['eigenvector_concentration','top2_gap','reciprocity','directional_ec_separation']
    for season,g in teams.groupby('season'):
        scale=g[features].std(ddof=0).replace(0,1);best=None
        for i in range(len(g)):
            for j in range(i+1,len(g)):
                a,b=g.iloc[i],g.iloc[j];difference=abs(a.conditional_entropy-b.conditional_entropy)
                if difference>.005:continue
                distance=float(np.sqrt(np.mean(((a[features]-b[features])/scale).to_numpy(float)**2)))
                if best is None or distance>best[0]:best=(distance,a,b,difference)
        if best is not None:
            distance,a,b,difference=best;record=dict(season=season,team_a=a.team,team_b=b.team,entropy_gap=difference,structure_distance=distance)
            for col in ['conditional_entropy',*features,'ast_to','off_rating','assists_per100passes']:record[col+'_a']=a[col];record[col+'_b']=b[col]
            cases.append(record)
    cases=pd.DataFrame(cases);folder=out/'study_8';folder.mkdir(exist_ok=True);atomic_csv(cases,folder/'matched_entropy_cases.csv')
    results=pd.DataFrame(models);family=pd.DataFrame(primary);family['primary_p']=np.where(family.study.isin([1,7]),family.two_way_p,family.p_value);order=family.primary_p.sort_values().index;family['holm_p']=np.nan;adjusted=np.maximum.accumulate([(len(order)-j)*family.loc[i,'primary_p'] for j,i in enumerate(order)]);family.loc[order,'holm_p']=np.minimum(1,adjusted);atomic_csv(results,out/'all_regressions.csv');atomic_csv(family,out/'primary_results.csv')
    # Plot and report generation follows below.
    titles={1:'1. Do creator–finisher roles accompany stronger partnerships?',2:'2. Does roster continuity preserve passing identity?',3:'3. Does top-two passing hierarchy relate to performance?',5:'5. Does directional role separation relate to performance?',6:'6. Does passing concentration accompany shooting concentration?',7:'7. Do passing partnerships persist across seasons?'}
    figurepaths=[];study_figs={i:[] for i in range(1,9)}
    import contextlib
    with contextlib.ExitStack() as stack:
        collection=stack.enter_context(PdfPages(out/'all_eight_studies.pdf'))
        study_pdf_handles={i:stack.enter_context(PdfPages(out/f'study_{i}'/'graphs.pdf')) for i in range(1,9)}
        def save(fig,study,name):
            folder=out/f'study_{study}';folder.mkdir(exist_ok=True);fig.tight_layout(rect=[0,.085,1,.94]);file=folder/(name+'.pdf');fig.savefig(file);fig.savefig(folder/(name+'.png'),dpi=140);collection.savefig(fig);study_pdf_handles[study].savefig(fig);plt.close(fig);figurepaths.append(file);study_figs[study].append(file)
        for study,label,result,plot in plot_specs:
            fig,ax=plt.subplots(figsize=(11,8));ax.scatter(plot.x_residual,plot.y_residual,s=5 if len(plot)>1000 else 24,alpha=.09 if len(plot)>1000 else .6,color='#2473b8',rasterized=True);xx=np.linspace(plot.x_residual.min(),plot.x_residual.max(),200);ax.plot(xx,result['slope']*xx,color='#c65c28',lw=2);ax.axhline(0,color='gray',lw=.5);ax.axvline(0,color='gray',lw=.5)
            p=result['two_way_p'] if study in [1,7] else result['p_value'];inference='team/dyad' if study in [1,7] else 'team';suffix='PRIMARY' if result['primary'] else 'Secondary / sensitivity'
            ax.set(xlabel=result['x'].replace('_',' ')+' after stated baselines',ylabel=result['y'].replace('_',' ')+' after stated baselines',title=titles[study]+f'\n{suffix}: partial r={result["partial_r"]:.3f}; R²={result["partial_r_squared"]:.1%}; {inference} p={p:.3g}')
            fig.text(.5,.014,f'n={result["n"]}; fixed effects: {result["fixed_effects"]}; covariates: {result["covariates"] or "none"}\nThe plotted R² concerns residual association, not total performance explained or out-of-sample prediction.',ha='center',fontsize=8);fig.subplots_adjust(bottom=.12);save(fig,study,label)
        fig,ax=plt.subplots(figsize=(11,8));ax.scatter(teams.top2_share,teams.top1_share,c=teams.top2_gap,cmap='viridis',s=35,alpha=.7);lo=min(teams.top2_share.min(),teams.top1_share.min());hi=max(teams.top2_share.max(),teams.top1_share.max());ax.plot([lo,hi],[lo,hi],ls='--',color='gray');ax.set(xlabel='Second-largest combined passing eigenvector share',ylabel='Largest combined passing eigenvector share',title='3. Top-two hierarchy is a continuum\nNear diagonal: similar top-two shares; farther above: more separated top hub');save(fig,3,'top_two_structure')
        fig,ax=plt.subplots(figsize=(11,8));ax.scatter(edges.passes,edges.assists_per100passes,s=3,alpha=.06,color='#2473b8',rasterized=True);ax.set_xscale('log');ax.set(xlabel='Recorded passes on directed connection (log scale)',ylabel='Credited assists per100 passes',title='4. Passing volume and assist yield on every positive-pass connection\nDescriptive; low-count rates are noisy, no regression inference');save(fig,4,'directed_pass_assist_yield')
        fig,ax=plt.subplots(figsize=(11,8));values=[teams.loc[teams.season.eq(s),'assists_per100passes'].to_numpy() for s in ctx['seasons']];ax.boxplot(values,tick_labels=ctx['seasons']);ax.set(ylabel='Team tracked assists per100 recorded passes',title='4. Team pass-to-assist yield by season\nTracking assists, not all useful passes or potential assists');save(fig,4,'team_pass_assist_yield')
        fig,ax=plt.subplots(figsize=(11,8));example=players[(players.season=='2025-26')&(players.team=='GSW')];ax.scatter(example.outgoing_ec,example.incoming_ec,c=example.directional_role,cmap='coolwarm',s=45);upper=max(example.outgoing_ec.max(),example.incoming_ec.max())*1.1;ax.plot([0,upper],[0,upper],ls='--',color='gray')
        for z in example.nlargest(8,'combined_ec').itertuples():ax.annotate(z.name,(z.outgoing_ec,z.incoming_ec),xytext=(3,4),textcoords='offset points',fontsize=8)
        ax.set(xlabel='Outgoing (right) passing eigenvector share',ylabel='Incoming (left) passing eigenvector share',title='5. Warriors2025–26: directed passing prominence\nThese are network roles, not positions, shooting roles or per-minute ability');save(fig,5,'warriors_directional_roles')
        fig,ax=plt.subplots(figsize=(11,8))
        for season,g in teams.groupby('season'):ax.scatter(g.eigenvector_concentration,g.shooting_concentration,label=season,s=24,alpha=.7)
        ax.set(xlabel='Normalized combined passing eigenvector concentration',ylabel='Normalized player FGA-share concentration',title='6. Passing concentration versus shooting concentration\nPooled descriptive view; the primary regression separately absorbs team and season baselines');ax.legend();save(fig,6,'passing_shooting_descriptive')
        fig,ax=plt.subplots(figsize=(11,8));ax.scatter(persistence.previous_excess,persistence.next_excess,s=4,alpha=.07,color='#2473b8',rasterized=True);lim=[min(persistence.previous_excess.min(),persistence.next_excess.min()),max(persistence.previous_excess.max(),persistence.next_excess.max())];ax.plot(lim,lim,ls='--',color='gray');ax.set(xlabel='Previous-season mutual-preference excess',ylabel='Next-season mutual-preference excess',title='7. Returning teammate partnerships\nDiagonal marks unchanged excess; primary regression uses transition baselines');save(fig,7,'partnership_persistence_descriptive')
        for case in cases.itertuples():
            fig,axes=plt.subplots(1,2,figsize=(15,8))
            for ax,team,tag in [(axes[0],case.team_a,'a'),(axes[1],case.team_b,'b')]:
                ids,A,S,c,_,_=ctx['networks'][case.season,team];G=nx.DiGraph();G.add_nodes_from(range(len(ids)))
                for i,j in zip(*np.where(A>0)):G.add_edge(i,j,weight=float(A[i,j]))
                G.remove_nodes_from(list(nx.isolates(G)));combined=nx.from_numpy_array(A+A.T);combined.remove_nodes_from(list(nx.isolates(combined)));pos=nx.spring_layout(combined,weight='weight',seed=42);mx=A.max();node_sizes=[45+4500*c[i] for i in G]
                nx.draw_networkx_edges(G,pos,ax=ax,alpha=.18,edge_color='#476b85',node_size=node_sizes,width=[.15+2*np.sqrt(G[u][v]['weight']/mx) for u,v in G.edges()],arrows=True,arrowsize=5,connectionstyle='arc3,rad=.1');nx.draw_networkx_nodes(G,pos,ax=ax,node_size=node_sizes,node_color='#cfe3ef',edgecolors='#34769e');name=ctx['rosters'][case.season,team].PLAYER_NAME.to_dict();top=np.argsort(-c)[:5];
                for yy,i in zip(np.linspace(.86,.22,5),sorted(top,key=lambda i:-pos[i][1])):
                    if i in G:ax.annotate(name[ids[i]],xy=pos[i],xytext=(.02,yy),textcoords='axes fraction',fontsize=9,ha='left',va='center',bbox=dict(boxstyle='round,pad=.2',fc='white',ec='none',alpha=.9),arrowprops=dict(arrowstyle='-',color='#536879',lw=.7))
                ax.set_title(team+f' · entropy={getattr(case,"conditional_entropy_"+tag):.4f}\nEC concentration={getattr(case,"eigenvector_concentration_"+tag):.4f}; top-two gap={getattr(case,"top2_gap_"+tag):.3f}');ax.set_axis_off();ax.margins(.2)
            fig.suptitle('8. Similar entropy, contrasting structure: '+case.season,fontsize=15);fig.text(.5,.025,f'Entropy difference={case.entropy_gap:.5f}; selected by network-feature contrast, not performance.\nAll positive edges retained; only five most central players labeled. FR positions are aesthetic.',ha='center',fontsize=9);fig.subplots_adjust(bottom=.1);save(fig,8,'same_entropy_'+case.season)
    family=family.sort_values('study');explanation=['# Eight full-roster passing studies: results and interpretation',
        'All180 full-roster team-seasons,30 teams,2020–21 through2025–26. No minutes cutoff. Official positive-minute participants include traded players; graph isolates omitted where required by eigenvectors. Six primary inferential tests were specified, one for studies1,2,3,5,6,7; Holm correction across these six. Studies4 and8 are descriptive. All secondary models are retained, not selected by p-value. Secondary p-values are unadjusted exploratory results. Pair studies use two-way team/dyad p-values in the primary correction; other studies use team-clustered p-values.',
        '## Six primary tests',integrated_table(family[['study','model','n','slope','partial_r','partial_r_squared','primary_p','holm_p']])]
    method_text={
      1:('Do creator–finisher roles accompany stronger partnerships?',
         'Player role = share of official team assists minus share of official team FGA, standardized by the network RMS role. Positive is creator-leaning; negative is finisher-leaning. Pair complementarity = negative product of the two standardized roles: positive for opposite roles, negative for similar-sided roles. Y = P(i,j)P(j,i) minus its fixed-margin reference expectation. Network fixed effects and equal total weight per network; team/dyad clustered inference. A role-distance alternative and minimum50-pass exposure sensitivity are retained. These box-score roles describe activities, not ability, positions or tactical intent. Assists also arise from passing, so roles are not wholly independent of the network. Shared-court exposure is not controlled.'),
      2:('Does roster continuity preserve passing identity?',
         'Retention is the average of previous and current season minute shares supplied by returning player IDs. Y is equal-source mean JS divergence between returning players’ rows after conditioning both seasons on the same returning receiving targets. At least3 comparable positive-outgoing rows required. Team and transition fixed effects; team-cluster inference. Minimum50 passes to retained targets in BOTH seasons is a sensitivity. Excluding changed targets isolates returning relationships but does not describe whole-roster change. Secondary absolute normalized-centrality change addresses anonymous shape.'),
      3:('Does top-two hierarchy relate to performance?',
         'X = (largest−second-largest combined EC share)/(largest+second-largest share). Higher means a more separated leading hub. This is continuous top-two hierarchy, not a verified two-hub or single-hub classification. A small gap can also occur in broadly balanced teams. Primary Y = official team assists/turnovers. Team and season fixed effects plus overall normalized EC concentration; team-cluster inference. Offensive rating and turnovers per100 possessions are secondary outcomes. Conditioning on concentration helps distinguish hierarchy at similar overall concentration; it is not causal.'),
      4:('Which passing connections produce credited assists?',
         'For every positive-pass edge, yield =100*tracking assists/passes. Raw rates and fixed100-pass shrinkage toward the season pooled rate are saved. The shrinkage display is (assists+100*season_rate)/(passes+100), not a calibrated probability model or confidence interval. Illustrative high-yield examples require100 passes; no players are removed from the underlying networks. Team yields and tracking-to-official-assist coverage are audited. This captures final credited assists, not missed opportunities, potential assists, earlier useful passes or total shot quality. No inferential claim from edge rankings.'),
      5:('Do differentiated directional passing roles relate to performance?',
         'Compute right(A) outgoing and right(A.T) incoming Perron eigenvector shares from DIRECTED counts, checking uniqueness, nonnegative signs and eigenvector residuals. Team role separation =0.5*sum(abs(outgoing_share−incoming_share));0 means identical incoming/outgoing profiles. Primary Y = official AST/TO, team/season fixed effects, team-cluster inference. Volume-share separation and offensive rating are secondary. Directed prominence is not the same as pure pass volume, finishing ability or a player’s position. Aggregate minutes and lineup overlap can affect roles.'),
      6:('Does passing concentration accompany shooting concentration?',
         'Primary X = normalized combined EC concentration; Y = normalized HHI of official player FGA shares. Higher Y means shots are concentrated in fewer players. Team/season fixed effects, team-cluster inference. Raw passing HHI versus raw FGA HHI is a sensitivity because both normalizations depend on roster size. Additional sensitivity models condition on minutes-share concentration (normalized with normalized outcomes; raw with raw outcomes). These were added to investigate shared playing opportunities after the primary association was observed; the pre-specified primary model and six-test family are unchanged. They are exploratory adjustments, not causal estimates. Descriptive four-group displays split each season at median percentile; these are relative labels, not natural clusters. High passing concentration with low shooting concentration is a possible distributor-to-many-finishers structure, not proof of that mechanism.'),
      7:('Do teammate partnerships persist across seasons?',
         'Match identical player pairs within the same franchise in consecutive seasons. X = previous-season null-adjusted mutual preference; Y = next-season excess. Franchise-transition fixed effects and equal total weight per transition; team/dyad clustered inference with correctly aggregated team×dyad intersections across seasons. Minimum50 total outgoing passes by each player in BOTH seasons is a sensitivity. Stable excess can reflect recurring shared minutes and roles; no causal chemistry or forecasting-performance claim follows.'),
      8:('Can the same entropy hide different structures?',
         'Within each season, select one team pair with entropy difference<=0.005 maximizing RMS standardized distance in EC concentration, top-two gap, reciprocity and directional EC separation. Performance and assist yield are NOT used in selection. The most contrasting pair is deliberately selected, so the cases are illustrations, not representative random samples or a significance test. All active nodes and positive directed pass edges are drawn; only top5 EC players labeled. FR placement has no court-position interpretation.')}
    for study in range(1,9):
        question,methods=method_text[study];text=['## '+str(study)+'. '+question,methods]
        if study in family.study.values:
            f=family[family.study.eq(study)].iloc[0];direction='positive' if f.slope>0 else 'negative';evidence='survives the six-primary-test Holm correction' if f.holm_p<.05 else 'does not survive the six-primary-test Holm correction'
            text.append(f'Primary result: n={int(f.n)}, partial r={f.partial_r:.4f}, partial R²={100*f.partial_r_squared:.2f}%, slope={f.slope:.6g}, primary clustered p={f.primary_p:.6g}, corrected p={f.holm_p:.6g}. The estimated association is {direction} and {evidence}. Partial R² concerns residual variation after the specified baselines, not total variation in performance.');text.append(integrated_table(results[results.study.eq(study)]))
        elif study==4:
            text.append(f'Descriptive sample: {len(edges)} directed positive-pass edges. Across all team-seasons, tracking assist yield is {100*edges.assists.sum()/edges.passes.sum():.3f} assists per100 passes; this pools counts and is not an equal-team mean. Tracking/official-assist ratios range {teams.tracked_assist_coverage.min():.4f} to {teams.tracked_assist_coverage.max():.4f}.');text.append(integrated_table(examples.head(12)[['season','team','passer','receiver','passes','assists','assists_per100passes','shrunk_assists_per100passes']]))
        else:text.append(integrated_table(cases))
        text.extend(['Figures: '+', '.join(path.name for path in study_figs[study]),'Detailed data and secondary/sensitivity models are retained in this study folder and the top-level CSVs.']);(out/f'study_{study}'/'RESULTS_EXPLAINED.md').write_text('\n\n'.join(text));explanation.extend(text)
    explanation.extend(['## Shared limits and provenance',
        f'Null baselines: {args.pilot_draws} pilot samples estimate expected pair mutual products, {args.null_draws} calibration samples retained by the shared sampler;20 sweeps burn before each stage,2 between samples, two-edge swaps and three-edge rotations preserve exact outgoing and incoming counts and forbid self-passes. Finite-chain Monte Carlo and pilot estimation add uncertainty; reported regression errors do not fully propagate pilot uncertainty. More draws are a sensitivity, not a guarantee of exactness. Null inputs and audit matrices are cached.',
        'Player FGA and AST sums reconcile to official team totals in all180 networks. Tracking-assist totals are separately compared with official box-score assists; any differences are recorded, not silently substituted. Raw profile hashes checked. All sources are season aggregates, so no exact shared-court minutes, five-player lineup identity, possession order or causal effect is inferred. Players in a season roster need not have played together. Franchise clustering handles repeated seasons imperfectly with only30 clusters; pair-cluster sensitivities are also reported for dyad models. No claim of out-of-sample predictive performance.',
        'Outputs: all_eight_studies.pdf; primary_results.csv; all_regressions.csv; team_metrics.csv; player_roles.csv; directed_edges.csv; mutual_pairs.csv; data_audit.csv; each study’s CSVs and figures. Full code preserves existing CLI and adds --eight-full-roster-studies. Official historical team performance totals were downloaded as needed; passing profiles and full matrices are the validated saved data.'
    ]);(out/'RESULTS_EXPLAINED.md').write_text('\n\n'.join(explanation));atomic_json(dict(created_at=utc_now(),networks=len(teams),seasons=ctx['seasons'],teams=teams.team.nunique(),player_team_seasons=len(players),positive_directed_edges=len(edges),pair_observations=len(pairs),primary_studies=[1,2,3,5,6,7],descriptive_studies=[4,8],pilot_draws=args.pilot_draws,calibration_draws=args.null_draws,seed=args.seed),out/'run_manifest.json')
    print(family[['study','n','slope','partial_r','partial_r_squared','primary_p','holm_p']].to_string(index=False),flush=True);print('Saved all eight studies: '+str(out),flush=True);return 0


# ---------------------------------------------------------------------
# DEEPER STUDIES 4,5,8: YIELD STABILITY, DIRECTIONAL ROLES, ENTROPY LIMITS
# ---------------------------------------------------------------------
def run_deep_passing_458(argv):
    from scipy.stats import f as fdist,t
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import contextlib
    parser=argparse.ArgumentParser(description='Deeper studies of assist yield,directional roles,and similar entropy')
    parser.add_argument('--deep-passing-458',action='store_true');parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT_DIR)
    args=parser.parse_args(argv);root=args.output_dir.expanduser();source=root/'eight_full_roster_studies';out=source/'deep_458';out.mkdir(exist_ok=True,parents=True)
    teams=pd.read_csv(source/'team_metrics.csv');players=pd.read_csv(source/'player_roles.csv');edges=pd.read_csv(source/'directed_edges.csv');seasons=sorted(teams.season.unique())
    if len(teams)!=180 or len(players)!=3922:raise ValueError('Expected complete six-season inputs')
    atomic_json({name:dict(path=str(source/name),sha256=hashlib.sha256((source/name).read_bytes()).hexdigest()) for name in ['team_metrics.csv','player_roles.csv','directed_edges.csv']},out/'input_manifest.json')
    for i in [4,5,8]:(out/f'study_{i}').mkdir(exist_ok=True)
    models=[];plots=[];primary=[]
    def fit(study,data,x,y,label,fixed,controls=None,group=None,primary_test=False):
        result,plot=full_eight_fit(data,x,y,label,fixed,controls,group,True);result['study']=study;result['primary']=primary_test;models.append(result);atomic_csv(plot,out/f'study_{study}'/(label+'_plot_data.csv'));plots.append((study,label,result,plot))
        if primary_test:primary.append(dict(study=study,test=label,n=result['n'],partial_r=result['partial_r'],partial_r_squared=result['partial_r_squared'],p=result['two_way_p']))
        return result,plot
    # 4: consecutive-season directed links, exposure sensitivity, past-only benchmark predictions.
    transitions=[]
    for old,new in zip(seasons,seasons[1:]):
        prior=edges[edges.season.eq(old)];current=edges[edges.season.eq(new)];joined=prior.merge(current,on=['team','passer_id','receiver_id'],suffixes=('_old','_new'));league=prior.assists.sum()/prior.passes.sum();passer=prior.groupby(['team','passer_id'])[['assists','passes']].sum();receiver=prior.groupby(['team','receiver_id'])[['assists','passes']].sum()
        for z in joined.itertuples():
            p=passer.loc[(z.team,z.passer_id)];q=receiver.loc[(z.team,z.receiver_id)];n=z.passes_new;k=z.assists_new;preds=dict(league=league,passer=(p.assists+100*league)/(p.passes+100),receiver=(q.assists+100*league)/(q.passes+100),connection=(z.assists_old+100*league)/(z.passes_old+100))
            record=dict(team=z.team,transition=old+'→'+new,network_transition=z.team+'_'+old+'_'+new,pair_id=str(z.passer_id)+'_'+str(z.receiver_id),passer_id=z.passer_id,receiver_id=z.receiver_id,passer=z.passer_old,receiver=z.receiver_old,passes_old=z.passes_old,passes_new=z.passes_new,assists_old=z.assists_old,assists_new=z.assists_new,yield_old=100*z.assists_old/z.passes_old,yield_new=100*k/n,minimum_passes=min(z.passes_old,z.passes_new))
            for name,prob in preds.items():record['pred_'+name]=prob;record['brier_'+name]=(k*(1-prob)**2+(n-k)*prob**2)/n;record['rate_sqerr_'+name]=(k/n-prob)**2
            transitions.append(record)
    transitions=pd.DataFrame(transitions);atomic_csv(transitions,out/'study_4/directed_link_transitions.csv');main4=transitions[transitions.minimum_passes.ge(100)].copy()
    fit(4,main4,'yield_old','yield_new','assist_yield_persistence_min100',['network_transition'],group='network_transition',primary_test=True)
    for threshold in [50,200]:fit(4,transitions[transitions.minimum_passes.ge(threshold)],'yield_old','yield_new','assist_yield_persistence_min'+str(threshold),['network_transition'],group='network_transition')
    forecast=[]
    for (team,transition),g in main4.groupby(['team','transition']):
        for name in ['league','passer','receiver','connection']:forecast.append(dict(team=team,transition=transition,model=name,connections=len(g),passes=g.passes_new.sum(),brier=np.average(g['brier_'+name],weights=g.passes_new),rate_rmse=np.sqrt(np.average(g['rate_sqerr_'+name],weights=g.passes_new))))
    forecast=pd.DataFrame(forecast);atomic_csv(forecast,out/'study_4/past_only_prediction_losses.csv');fs=forecast.groupby('model').agg(team_transitions=('brier','size'),mean_brier=('brier','mean'),mean_transition_rate_rmse=('rate_rmse','mean')).reset_index();atomic_csv(fs,out/'study_4/past_only_prediction_summary.csv')
    receiver_totals=edges.groupby(['season','team','receiver_id','receiver'])[['passes','assists']].sum().reset_index();receiver_totals['assists_per100passes_received']=100*receiver_totals.assists/receiver_totals.passes;passer_totals=edges.groupby(['season','team','passer_id','passer'])[['passes','assists']].sum().reset_index();passer_totals['assists_per100passes_made']=100*passer_totals.assists/passer_totals.passes;atomic_csv(receiver_totals,out/'study_4/receiver_assist_yield.csv');atomic_csv(passer_totals,out/'study_4/passer_assist_yield.csv')
    stable=main4.copy();stable['combined_passes']=stable.passes_old+stable.passes_new;stable['combined_yield']=100*(stable.assists_old+stable.assists_new)/stable.combined_passes;atomic_csv(stable.sort_values('combined_yield',ascending=False),out/'study_4/repeated_connections.csv')
    # 5: compare network orientation to independent box-score role measures, with opportunity controls.
    players=players.merge(teams[['network','passes','minutes']],on='network',validate='many_to_one');players['minutes_share']=players.minutes_x/players.minutes_y;players['volume_role']=(players.passes_made-players.passes_received)/players.passes;players['pair_id']=players.player_id.astype(str)
    fit(5,players,'directional_role','creator_finisher_role','directional_vs_boxscore_role',['network'],['minutes_share','combined_ec'],'network',True)
    fit(5,players,'directional_role','creator_finisher_role','directional_vs_boxscore_unadjusted',['network'],group='network')
    fit(5,players,'volume_role','creator_finisher_role','volume_vs_boxscore_role',['network'],['minutes_share','combined_ec'],'network')
    fit(5,players,'directional_role','creator_finisher_role','directional_role_beyond_volume',['network'],['minutes_share','combined_ec','volume_role'],'network')
    fit(5,players,'directional_role','volume_role','directional_vs_volume',['network'],group='network')
    role_trans=[]
    for old,new in zip(seasons,seasons[1:]):
        z=players[players.season.eq(old)].merge(players[players.season.eq(new)],on=['team','player_id'],suffixes=('_old','_new'))
        for r in z.itertuples():role_trans.append(dict(team=r.team,pair_id=str(r.player_id),name=r.name_old,transition=old+'→'+new,network_transition=r.team+'_'+old+'_'+new,role_old=r.directional_role_old,role_new=r.directional_role_new,minutes_share_old=r.minutes_share_old,minutes_share_new=r.minutes_share_new))
    role_trans=pd.DataFrame(role_trans);fit(5,role_trans,'role_old','role_new','directional_role_persistence',['network_transition'],['minutes_share_old','minutes_share_new'],'network_transition');atomic_csv(role_trans,out/'study_5/player_role_transitions.csv');atomic_csv(players,out/'study_5/player_roles_enriched.csv')
    # 8: all close-entropy pairs, not just deliberately contrasting examples.
    features=['eigenvector_concentration','top2_gap','reciprocity','directional_ec_separation'];comparisons=[]
    for season,g in teams.groupby('season'):
        scale=g[features].std(ddof=0).replace(0,1)
        for i in range(len(g)):
            for j in range(i+1,len(g)):
                a,b=g.iloc[i],g.iloc[j];record=dict(season=season,team_a=a.team,team_b=b.team,entropy_gap=abs(a.conditional_entropy-b.conditional_entropy),close_entropy=abs(a.conditional_entropy-b.conditional_entropy)<=.005,off_rating_gap=abs(a.off_rating-b.off_rating))
                for col in features:record[col+'_gap']=abs(a[col]-b[col])
                record['structure_distance']=float(np.sqrt(np.mean(((a[features]-b[features])/scale).to_numpy(float)**2)));comparisons.append(record)
    comparisons=pd.DataFrame(comparisons);near=comparisons[comparisons.close_entropy].copy();atomic_csv(comparisons,out/'study_8/all_same_season_pairs.csv');atomic_csv(near,out/'study_8/all_close_entropy_pairs.csv');summary=near.groupby('season').agg(pairs=('team_a','size'),median_centrality_gap=('eigenvector_concentration_gap','median'),maximum_centrality_gap=('eigenvector_concentration_gap','max'),median_off_rating_gap=('off_rating_gap','median'),fraction_ec_gap_atleast001=('eigenvector_concentration_gap',lambda x:(x>=.01).mean()),fraction_off_rating_gap_atleast5=('off_rating_gap',lambda x:(x>=5).mean())).reset_index();atomic_csv(summary,out/'study_8/close_entropy_summary.csv')
    # Joint association: four network features after entropy and team/season baselines.
    joint=[];joint_coefs=[]
    def joint_fit(minutes=False):
        controls=['conditional_entropy']+(['minutes_concentration'] if minutes else []);Z=np.column_stack([np.ones(len(teams)),teams[controls].to_numpy(float),pd.get_dummies(teams.team,drop_first=True,dtype=float),pd.get_dummies(teams.season,drop_first=True,dtype=float)]);X=np.column_stack([Z,teams[features].to_numpy(float)]);y=teams.off_rating.to_numpy();b=np.linalg.lstsq(X,y,rcond=None)[0];e=y-X@b;bread=np.linalg.pinv(X.T@X);G=teams.team.nunique();rank=np.linalg.matrix_rank(X);score=np.stack([X[teams.team.eq(team)].T@e[teams.team.eq(team)] for team in teams.team.unique()]);cov=bread@(score.T@score)@bread*G/(G-1)*(len(teams)-1)/(len(teams)-rank);q=len(features);v=cov[-q:,-q:];F=float(b[-q:]@np.linalg.pinv(v)@b[-q:]/q);p=float(fdist.sf(F,q,G-1));base_res=y-Z@np.linalg.lstsq(Z,y,rcond=None)[0];partial=1-float(e@e/(base_res@base_res));label='joint_structure_beyond_entropy'+('_minutes' if minutes else '')
        result=dict(model=label,n=len(teams),features=q,F=F,df1=q,df2=G-1,p=p,partial_r_squared=partial,minutes_adjusted=minutes);joint.append(result)
        for i,feature in enumerate(features):se=float(np.sqrt(cov[-q+i,-q+i]));joint_coefs.append(dict(model=label,feature=feature,coefficient=b[-q+i],team_cluster_se=se,unadjusted_p=2*t.sf(abs(b[-q+i]/se),G-1)))
        if not minutes:primary.append(dict(study=8,test=label,n=len(teams),partial_r=np.nan,partial_r_squared=partial,p=p))
    joint_fit();joint_fit(True);joint=pd.DataFrame(joint);atomic_csv(joint,out/'study_8/joint_structure_tests.csv');atomic_csv(pd.DataFrame(joint_coefs),out/'study_8/joint_structure_coefficients.csv')
    # Held-out-franchise check; season means are known cross-sectional baselines, not forecasts.
    designs={'season_only':[], 'entropy':['conditional_entropy'],'structure':features,'entropy_structure':['conditional_entropy',*features]};predictions=[];season_dummies=pd.get_dummies(teams.season,drop_first=True,dtype=float).to_numpy()
    for name,columns in designs.items():
        X=np.column_stack([np.ones(len(teams)),season_dummies,*([teams[columns].to_numpy(float)] if columns else [])]);y=teams.off_rating.to_numpy()
        for team in sorted(teams.team.unique()):
            test=teams.team.eq(team).to_numpy();b=np.linalg.lstsq(X[~test],y[~test],rcond=None)[0];pred=X[test]@b
            for (_,r),p in zip(teams.loc[test].iterrows(),pred):predictions.append(dict(team=team,season=r.season,model=name,actual=r.off_rating,predicted=p,squared_error=(r.off_rating-p)**2))
    predictions=pd.DataFrame(predictions);cv=predictions.groupby('model').squared_error.agg(['sum','mean']).reset_index();baseline=float(cv.loc[cv.model.eq('season_only'),'sum'].iloc[0]);cv['rmse']=np.sqrt(cv['mean']);cv['r_squared_vs_season_only']=1-cv['sum']/baseline;atomic_csv(predictions,out/'study_8/heldout_franchise_predictions.csv');atomic_csv(cv,out/'study_8/heldout_franchise_summary.csv')
    family=pd.DataFrame(primary);order=family.p.sort_values().index;family['holm_p']=np.nan;family.loc[order,'holm_p']=np.minimum(1,np.maximum.accumulate([(3-i)*family.loc[k,'p'] for i,k in enumerate(order)]));atomic_csv(family,out/'primary_results.csv');atomic_csv(pd.DataFrame(models),out/'all_regressions.csv')
    # Graphs and readable explanations.
    names={4:'Assist-yield stability',5:'Incoming/outgoing passing roles',8:'What entropy misses'}
    with contextlib.ExitStack() as stack:
        pdf=stack.enter_context(PdfPages(out/'deep_studies_4_5_8.pdf'));pdfs={i:stack.enter_context(PdfPages(out/f'study_{i}'/'graphs.pdf')) for i in [4,5,8]}
        def save(fig,study,name):
            fig.tight_layout(rect=[0,.08,1,.94])
            if not name.startswith('same_entropy_'):season_topics_spread_labels(fig)
            pdf.savefig(fig);pdfs[study].savefig(fig);fig.savefig(out/f'study_{study}'/(name+'.pdf'));fig.savefig(out/f'study_{study}'/(name+'.png'),dpi=145);plt.close(fig)
        # Restore the six original same-entropy network comparisons in the deeper PDFs.
        import networkx as nx
        network_cases=pd.read_csv(source/'study_8/matched_entropy_cases.csv')
        atomic_csv(network_cases,out/'study_8/matched_entropy_network_cases.csv')
        for case in network_cases.itertuples():
            fig,axes=plt.subplots(1,2,figsize=(15,8))
            for ax,team,tag in [(axes[0],case.team_a,'a'),(axes[1],case.team_b,'b')]:
                frame=pd.read_csv(root/'full_roster_entropy_centrality/historical_full_rosters'/case.season/team/'passing_matrix.csv',index_col=0);ids=frame.index.astype(int).tolist();A=frame.to_numpy(float);own=players[(players.season==case.season)&(players.team==team)].set_index('player_id');c=own.reindex(ids).combined_ec.to_numpy();node_names=own.name.to_dict();G=nx.DiGraph();G.add_nodes_from(range(len(ids)))
                for i,j in zip(*np.where(A>0)):G.add_edge(i,j,weight=float(A[i,j]))
                G.remove_nodes_from(list(nx.isolates(G)));combined=nx.from_numpy_array(A+A.T);combined.remove_nodes_from(list(nx.isolates(combined)));pos=nx.spring_layout(combined,weight='weight',seed=42);sizes=[45+4500*c[i] for i in G];maximum=A.max()
                nx.draw_networkx_edges(G,pos,ax=ax,alpha=.18,edge_color='#476b85',node_size=sizes,width=[.15+2*np.sqrt(G[u][v]['weight']/maximum) for u,v in G.edges()],arrows=True,arrowsize=5,connectionstyle='arc3,rad=.1');nx.draw_networkx_nodes(G,pos,ax=ax,node_size=sizes,node_color='#cfe3ef',edgecolors='#34769e')
                top=np.argsort(-c)[:5]
                for yy,i in zip(np.linspace(.86,.22,5),sorted(top,key=lambda i:-pos[i][1])):
                    if i in G:ax.annotate(node_names[ids[i]],xy=pos[i],xytext=(.02,yy),textcoords='axes fraction',fontsize=9,ha='left',va='center',bbox=dict(boxstyle='round,pad=.2',fc='white',ec='none',alpha=.9),arrowprops=dict(arrowstyle='-',color='#536879',lw=.7))
                ax.set_title(team+f' · entropy={getattr(case,"conditional_entropy_"+tag):.4f}\nEC concentration={getattr(case,"eigenvector_concentration_"+tag):.4f}; top-two gap={getattr(case,"top2_gap_"+tag):.3f}');ax.set_axis_off();ax.margins(.2)
            fig.suptitle('8. Similar entropy, contrasting networks: '+case.season,fontsize=15);fig.text(.5,.025,f'Entropy difference={case.entropy_gap:.5f}; original comparison selected by network contrast, not performance.\nAll positive edges retained; node size: combined eigenvector share; top5 players labeled. FR layout is aesthetic.',ha='center',fontsize=9);save(fig,8,'same_entropy_'+case.season)
        for study,label,result,data in plots:
            fig,ax=plt.subplots(figsize=(11,8));ax.scatter(data.x_residual,data.y_residual,s=6 if len(data)>1000 else 20,alpha=.13 if len(data)>1000 else .6,color='#2473b8',rasterized=True);xx=np.linspace(data.x_residual.min(),data.x_residual.max(),150);ax.plot(xx,result['slope']*xx,color='#c65c28');ax.axhline(0,color='gray',lw=.5);ax.axvline(0,color='gray',lw=.5);ax.set(xlabel=result['x'].replace('_',' ')+' after stated baselines',ylabel=result['y'].replace('_',' ')+' after stated baselines',title=names[study]+f'\npartial r={result["partial_r"]:.3f}; R²={result["partial_r_squared"]:.1%}; two-way clustered p={result["two_way_p"]:.3g}');fig.text(.5,.02,'Fixed effects: '+result['fixed_effects']+'; covariates: '+(result['covariates'] or 'none')+'\nPartial association, not a causal effect or total offensive performance explained.',ha='center',fontsize=8);save(fig,study,label)
        fig,ax=plt.subplots(figsize=(10,7));ax.bar(fs.model,fs.mean_brier,color='#2473b8');ax.set(ylabel='Mean team-transition per-pass Brier loss (lower is better)',title='4. Next-season assist yield: past-only benchmark probabilities\nSame connections with at least100 passes in both seasons');save(fig,4,'past_only_benchmark_losses')
        fig,axes=plt.subplots(1,2,figsize=(14,7));example=players[(players.team=='GSW')&(players.season=='2021-22')].sort_values('directional_role');axes[0].barh(example.name,example.directional_role,color=np.where(example.directional_role>0,'#2473b8','#c65c28'));axes[0].axvline(0,color='gray',lw=.5);axes[0].set(xlabel='Outgoing minus incoming eigenvector share',title='Warriors2021–22: direction of prominence');axes[1].scatter(example.creator_finisher_role,example.directional_role,s=45)
        for r in example.nlargest(8,'combined_ec').itertuples():axes[1].annotate(r.name,(r.creator_finisher_role,r.directional_role),xytext=(3,4),textcoords='offset points',fontsize=7)
        axes[1].margins(x=.2,y=.15);axes[1].set(xlabel='Assist share minus FGA share',ylabel='Outgoing minus incoming EC share',title='Same players: network role versus box-score role');fig.text(.5,.015,'Left: positive means greater outgoing EC. Right: positive box-score role means more assist share than shot share.\nThese are different definitions, not interchangeable labels for ability or court position.',ha='center',fontsize=9);save(fig,5,'warriors_2021_roles')
        fig,ax=plt.subplots(figsize=(11,7));ax.hist(comparisons.eigenvector_concentration_gap,bins=25,alpha=.45,density=True,label='All same-season team pairs');ax.hist(near.eigenvector_concentration_gap,bins=20,alpha=.6,density=True,label='Entropy gap<=0.005');ax.set(xlabel='Absolute normalized centrality-concentration gap',ylabel='Density',title='8. Similar entropy narrows, but does not eliminate, structural differences');ax.legend();save(fig,8,'close_entropy_structural_gaps')
        fig,ax=plt.subplots(figsize=(11,7));ax.scatter(near.eigenvector_concentration_gap,near.off_rating_gap,s=22,alpha=.6);ax.set(xlabel='Centrality-concentration difference among close-entropy teams',ylabel='Absolute offensive-rating difference',title='8. All close-entropy pairs: descriptive, dependent comparisons\nNo independent-pair significance test or causal interpretation');save(fig,8,'all_close_entropy_performance_gaps')
        fig,ax=plt.subplots(figsize=(11,7));ax.bar(cv.model,cv.rmse,color='#2473b8');ax.set(ylabel='Held-out-franchise RMSE (points per100 possessions)',title='8. Can structure add information beyond entropy?\nAll six seasons of each franchise held out together; lower is better');fig.text(.5,.015,'Same-season full-year features; this is a held-out-franchise check, not an advance-season forecast.\nOverlapping training folds make ordinary paired p-values inappropriate; losses are descriptive.',ha='center',fontsize=9);save(fig,8,'heldout_franchise_performance')
    report=['# Deeper studies4,5,8',
        'All180 full-roster team-seasons; no new NBA downloads. Three new primary tests were chosen before examining these deeper model results:4 directed-link assist-yield persistence at100 passes each season;5 directional EC role alignment with box-score role after minutes share/combined prominence;8 four-feature joint offensive-rating association beyond entropy and team/season baselines. Holm correction across these three; all other tests exploratory.',
        '## Primary results',integrated_table(family),
        '## 4. Do productive assist connections stay productive?',integrated_table(pd.DataFrame(models).query('study==4')),integrated_table(fs),
        f'{len(main4)} repeated directed connections have at least100 passes in BOTH seasons. Exposure filters apply to this rate-comparison sample, not the original full networks. Sensitivities use50 and200. Match passer/receiver IDs within the same franchise across consecutive years; franchise-transition baselines and equal total weight per transition; team/directed-dyad clustered uncertainty. Positive persistence may reflect stable finishing roles and shared playing situations, not causal chemistry.',
        'Past-only benchmark: previous-season pooled league, passer, receiver or connection assist probability; each individual rate uses a100-pass prior toward the previous-season league rate. Current-year outcomes never enter predictions. Score future observed assists/nonassists with exact aggregate Brier loss, weighted by current passes within a team-transition and equally across team-transitions. Current exposure eligibility uses the observed later season; this is a conditional test on known recurring connections, not a forecast of who will play or connect. Loss comparisons are descriptive and do not establish a pair-specific advantage if receiver history performs as well or better. Tracking credited assists omit missed chances and earlier creation; tracking/box-score totals differ as documented in the parent analysis.',
        '## 5. What do directional roles actually mean?',integrated_table(pd.DataFrame(models).query('study==5')),
        'Player network role is outgoing(right A) minus incoming(right A.T) eigenvector share. Positive means outgoing prominence is greater; negative means incoming prominence is greater. Primary outcome is official assist share minus official FGA share; controls are team-minute share and combined A+A.T eigenvector share; network fixed effects, equal network total weight, team/player clustered errors. This tests descriptive role alignment, not team efficiency or player ability. Assist-based roles remain related to the passing data.',
        'Volume role=(passes made−passes received)/team passes. Compare its model with the EC model and the incremental EC-after-volume model before claiming eigenvectors add information. A high correlation between EC orientation and volume orientation means their interpretations overlap. Player role persistence is a secondary same-franchise consecutive-season association after old/new minute shares and transition baselines. The earlier team AST/TO result remains weak and is NOT overturned by player role alignment.',
        '## 8. How often can close entropy hide differences?',integrated_table(summary),integrated_table(joint),integrated_table(cv),
        f'All2610 same-season team pairs examined; {len(near)} have entropy gap<=0.005. Unlike the earlier six deliberately contrasting examples, these tables include EVERY eligible pair. Pairs sharing a team are dependent; distribution summaries and gap plots are descriptive, not independent-pair tests. Thresholds0.005 entropy,0.01 centrality and5 offensive-rating points are descriptive reference values, not universal cutoffs.',
        'Joint test adds EC concentration, top-two hierarchy, reciprocity and directional EC separation to entropy plus team and season fixed effects, using a four-parameter franchise-clustered Wald F with29 denominator df. Minutes concentration is an exploratory sensitivity. Joint partial R² is in-sample after baselines; individual feature p-values are exploratory, correlated coefficients can be unstable, and association is not causal.',
        'Held-out-franchise OLS compares season-only, entropy, structure and entropy+structure, using all six seasons of one franchise as the test set. The feature sets were fixed, with no tuning to the test losses. These are full-season contemporaneous features, not predictors available before the season. RMSE is points per100 possessions; R²_vs_season_only compares held-out sums of squared errors with the held-out season-only baseline. Negative values mean worse than that baseline. The joint in-sample model has team fixed effects; the held-out-franchise models cannot fit an effect for an unseen franchise and therefore use season baselines without team fixed effects. These answer different questions. Do not confuse a significant in-sample joint test with improved held-out performance. Cross-validation training sets overlap, so no ordinary paired significance test is attached to loss differences.',
        '## Shared limitations',
        'Full-season aggregates mix lineups, trades and playing time, without exact shared-court exposure or pass sequences. Matching same IDs does not establish uninterrupted teammate status. Team/player or team/dyad clustering only partially handles complex dependence. Three new primary tests are adjusted together; the many secondary diagnostics are exploratory, with unadjusted p-values. Earlier eight-study results are preserved separately. Raw inputs, transition rows, prediction errors, all models and all eligible entropy pairs are saved.',
        'Full updated source retains previous CLI and adds --deep-passing-458. The six original study8 network comparisons are restored as the first six pages of the combined and study8 PDFs. Readable graphs are in deep_studies_4_5_8.pdf and study4/5/8 graphs.pdf. The enriched-role columns minutes_x and minutes_y are player minutes and team minutes, respectively.'
    ];(out/'METHODS_AND_RESULTS.md').write_text('\n\n'.join(report));atomic_json(dict(created_at=utc_now(),networks=180,directed_link_transitions=len(transitions),main_min100_connections=len(main4),close_entropy_pairs=len(near),primary_tests=[4,5,8],controls_and_scope='see METHODS_AND_RESULTS.md'),out/'run_manifest.json');print(family.to_string(index=False),flush=True);print(fs.to_string(index=False),flush=True);print(cv.to_string(index=False),flush=True);print('Saved deeper458 studies: '+str(out),flush=True);return 0


# ---------------------------------------------------------------------
# PROVISIONAL PASSING STYLE: MINUTES-RELATIVE CENTRALIZATION AND HUB SHAPE
# ---------------------------------------------------------------------
def passing_structure_label(ratio,gap1,gap2,rules):
    if ratio<=rules['balanced_ratio_max']:return 'more_balanced'
    if ratio<rules['centralized_ratio_min']:return 'intermediate'
    if gap1>=rules['single_gap_min'] and gap1>=gap2:return 'one_hub_leaning'
    if gap2>=rules['dual_gap_min'] and gap2>gap1:return 'two_hub_leaning'
    return 'centralized_core'


def run_passing_structure_classifier(argv):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from scipy.stats import spearmanr
    import networkx as nx
    parser=argparse.ArgumentParser(description='Transparent provisional star-centered versus balanced passing classifier')
    parser.add_argument('--classify-passing-structure',action='store_true');parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT_DIR);parser.add_argument('--reference-rules',type=Path)
    args=parser.parse_args(argv);root=args.output_dir.expanduser();source=root/'eight_full_roster_studies';out=root/'passing_structure_classification';out.mkdir(exist_ok=True,parents=True);teams=pd.read_csv(source/'team_metrics.csv');players=pd.read_csv(source/'player_roles.csv');rows=[];inputs=[]
    if len(teams)!=180 or teams.groupby('season').size().ne(30).any():raise ValueError('Expected180 validated full-roster networks')
    for r in teams.itertuples():
        own=players[players.network.eq(r.network)].sort_values(['combined_ec','player_id'],ascending=[False,True]);c=own.combined_ec.to_numpy(float);m=own.minutes.to_numpy(float)/own.minutes.sum()
        if not np.isclose(c.sum(),1) or not np.isclose(c@c,r.eigenvector_hhi) or not np.isclose(m@m,r.minutes_hhi):raise ValueError('Share/HHI consistency failed')
        rows.append(dict(network=r.network,season=r.season,team=r.team,active_players=r.active_players,eigenvector_concentration=r.eigenvector_concentration,ec_hhi=r.eigenvector_hhi,minutes_hhi=r.minutes_hhi,effective_passing_hubs=1/r.eigenvector_hhi,effective_minutes_players=1/r.minutes_hhi,centralization_ratio=r.eigenvector_hhi/r.minutes_hhi,hub1_id=int(own.player_id.iloc[0]),hub1_name=own.name.iloc[0],hub1_ec=c[0],hub1_minutes_share=m[0],hub1_premium=c[0]-m[0],hub2_id=int(own.player_id.iloc[1]),hub2_name=own.name.iloc[1],hub2_ec=c[1],hub2_minutes_share=m[1],top2_ec=c[:2].sum(),top2_minutes_share=m[:2].sum(),top2_premium=c[:2].sum()-m[:2].sum(),gap1=(c[0]-c[1])/(c[0]+c[1]),gap2=(c[1]-c[2])/(c[1]+c[2])))
    data=pd.DataFrame(rows)
    fitted=dict(balanced_ratio_max=float(data.centralization_ratio.quantile(.25)),centralized_ratio_min=float(data.centralization_ratio.quantile(.75)),single_gap_min=float(data.gap1.quantile(.75)),dual_gap_min=float(data.gap2.quantile(.75)))
    rules=json.loads(args.reference_rules.read_text())['thresholds'] if args.reference_rules else fitted
    data['passing_style']=data.apply(lambda r:passing_structure_label(r.centralization_ratio,r.gap1,r.gap2,rules),axis=1)
    data['broad_style']=np.where(data.centralization_ratio<=rules['balanced_ratio_max'],'more_balanced',np.where(data.centralization_ratio>=rules['centralized_ratio_min'],'more_centralized','intermediate'))
    raw_rules=dict(balanced_ratio_max=float(data.eigenvector_concentration.quantile(.25)),centralized_ratio_min=float(data.eigenvector_concentration.quantile(.75)),single_gap_min=rules['single_gap_min'],dual_gap_min=rules['dual_gap_min'])
    if args.reference_rules:raw_rules=json.loads(args.reference_rules.read_text()).get('observed_thresholds',raw_rules)
    data['observed_passing_style']=data.apply(lambda r:passing_structure_label(r.eigenvector_concentration,r.gap1,r.gap2,raw_rules),axis=1)
    data['observed_broad_style']=np.where(data.eigenvector_concentration<=raw_rules['balanced_ratio_max'],'more_balanced',np.where(data.eigenvector_concentration>=raw_rules['centralized_ratio_min'],'more_centralized','intermediate'))
    data['minutes_relative_style']=data.passing_style
    data['views_agree']=data.observed_broad_style.eq(data.broad_style)
    data['consensus_style']=np.where(data.views_agree,data.passing_style,'metric_sensitive')
    data['centralization_percentile']=data.centralization_ratio.rank(pct=True)
    # Sensitivity to descriptive cutoffs, not a probability that a label is true.
    fine_matches=np.zeros(len(data));broad_matches=np.zeros(len(data));grid=[]
    for qb in [.20,.25,.30]:
        for qc in [.70,.75,.80]:
            for qg in [.65,.75,.85]:
                alt=dict(balanced_ratio_max=float(data.centralization_ratio.quantile(qb)),centralized_ratio_min=float(data.centralization_ratio.quantile(qc)),single_gap_min=float(data.gap1.quantile(qg)),dual_gap_min=float(data.gap2.quantile(qg)))
                labels=data.apply(lambda r:passing_structure_label(r.centralization_ratio,r.gap1,r.gap2,alt),axis=1);broad=np.where(data.centralization_ratio<=alt['balanced_ratio_max'],'more_balanced',np.where(data.centralization_ratio>=alt['centralized_ratio_min'],'more_centralized','intermediate'));fine_matches+=labels.eq(data.passing_style).to_numpy();broad_matches+=broad==data.broad_style.to_numpy();grid.append(dict(balanced_quantile=qb,centralized_quantile=qc,hierarchy_quantile=qg,**alt))
    data['label_agreement_across_cutoffs']=fine_matches/len(grid);data['broad_agreement_across_cutoffs']=broad_matches/len(grid);data['threshold_stable']=data.label_agreement_across_cutoffs.ge(.8);data['label_status']=np.where(data.threshold_stable,'threshold-stable','borderline / threshold-sensitive')
    # Held-out-franchise reference-threshold check.
    loo=[]
    for r in data.itertuples():
        train=data[data.team.ne(r.team)];local=dict(balanced_ratio_max=float(train.centralization_ratio.quantile(.25)),centralized_ratio_min=float(train.centralization_ratio.quantile(.75)),single_gap_min=float(train.gap1.quantile(.75)),dual_gap_min=float(train.gap2.quantile(.75)));loo.append(passing_structure_label(r.centralization_ratio,r.gap1,r.gap2,local))
    data['heldout_franchise_label']=loo;data['heldout_franchise_agrees']=data.passing_style.eq(data.heldout_franchise_label)
    for name in ['team_metrics.csv','player_roles.csv']:inputs.append(dict(path=str(source/name),sha256=hashlib.sha256((source/name).read_bytes()).hexdigest()))
    atomic_csv(data,out/'all_team_season_styles.csv');atomic_csv(data[data.season.eq('2025-26')].sort_values('centralization_ratio',ascending=False),out/'2025-26_styles.csv');atomic_csv(pd.DataFrame(grid),out/'cutoff_sensitivity_grid.csv');atomic_csv(pd.DataFrame(inputs),out/'source_manifest.csv');atomic_json(dict(thresholds=rules,observed_thresholds=raw_rules,reference_networks=180,reference_seasons=sorted(data.season.unique()),reference_note='Empirical relative quartiles, not validated natural star/balanced classes',interpretation='Passing structure only; minutes ratio is opportunity comparison, not exact on-court exposure control',threshold_sensitivity='27 descriptive cutoff combinations; agreement is not a statistical confidence probability'),out/'classification_rules.json')
    counts=pd.crosstab(data.season,data.passing_style);atomic_csv(counts.reset_index(),out/'style_counts_by_season.csv')
    labels={'more_balanced':'More distributed relative to minutes','intermediate':'Intermediate','one_hub_leaning':'Centralized: one-hub leaning','two_hub_leaning':'Centralized: two-hub leaning','centralized_core':'Centralized: broader core'};colors={'more_balanced':'#168348','intermediate':'#9aa5af','one_hub_leaning':'#c65c28','two_hub_leaning':'#7556a6','centralized_core':'#2473b8'}
    examples=[]
    for style in ['one_hub_leaning','two_hub_leaning','more_balanced','centralized_core','intermediate']:
        g=data[data.passing_style.eq(style)].copy()
        if g.empty:continue
        if style=='one_hub_leaning':chosen=g.sort_values(['threshold_stable','gap1','centralization_ratio'],ascending=False).iloc[0]
        elif style=='two_hub_leaning':chosen=g.sort_values(['threshold_stable','gap2','centralization_ratio'],ascending=False).iloc[0]
        elif style=='more_balanced':chosen=g.sort_values(['threshold_stable','centralization_ratio'],ascending=[False,True]).iloc[0]
        elif style=='centralized_core':chosen=g.sort_values(['threshold_stable','centralization_ratio'],ascending=False).iloc[0]
        else:chosen=g.iloc[np.argmin(abs(g.centralization_ratio-g.centralization_ratio.median()))]
        examples.append(chosen)
    examples=pd.DataFrame(examples);atomic_csv(examples,out/'illustrative_examples.csv')
    with PdfPages(out/'passing_structure_styles.pdf') as pdf:
        def save(fig,name,network=False):
            fig.tight_layout(rect=[0,.085,1,.94])
            if not network:season_topics_spread_labels(fig)
            pdf.savefig(fig);fig.savefig(out/(name+'.pdf'));fig.savefig(out/(name+'.png'),dpi=145);plt.close(fig)
        fig,ax=plt.subplots(figsize=(12,8));latest=data[data.season.eq('2025-26')]
        for style,g in latest.groupby('passing_style'):ax.scatter(g.centralization_ratio,g.gap1,color=colors[style],s=55,label=labels[style])
        for r in latest.itertuples():ax.annotate(r.team,(r.centralization_ratio,r.gap1),xytext=(4,4),textcoords='offset points',fontsize=8)
        ax.axvline(rules['balanced_ratio_max'],color='#168348',ls='--');ax.axvline(rules['centralized_ratio_min'],color='#c65c28',ls='--');ax.axhline(rules['single_gap_min'],color='gray',ls=':',lw=.8);ax.set(xlabel='Passing HHI / minutes HHI (higher: extra concentration)',ylabel='Top-player separation: (EC1−EC2)/(EC1+EC2)',title='2025–26 passing structures: provisional relative labels');ax.legend(fontsize=8);fig.text(.5,.018,'Reference cutoffs use all180 full-roster networks. No basketball ability or superstar status is inferred.\nMiddle cases remain intermediate; label stability is saved rather than forcing a binary split.',ha='center',fontsize=9);save(fig,'2025-26_structure_map')
        fig,ax=plt.subplots(figsize=(12,8))
        for style,g in data.groupby('passing_style'):ax.scatter(g.eigenvector_concentration,g.centralization_ratio,color=colors[style],s=25,alpha=.7,label=labels[style])
        ax.axvline(raw_rules['balanced_ratio_max'],color='#168348',ls=':',lw=.8);ax.axvline(raw_rules['centralized_ratio_min'],color='#c65c28',ls=':',lw=.8);ax.axhline(1,color='gray',ls=':');ax.axhline(rules['balanced_ratio_max'],color='#168348',ls='--');ax.axhline(rules['centralized_ratio_min'],color='#c65c28',ls='--');ax.set(xlabel='Original normalized EC concentration',ylabel='EC HHI / minutes HHI',title='Raw network centralization and minutes-relative centralization are different');ax.legend(fontsize=8);fig.text(.5,.02,'The ratio compares distribution inequality, not per-minute passing ability or exact shared-court exposure.',ha='center');save(fig,'raw_vs_minutes_relative')
        fig,ax=plt.subplots(figsize=(12,8));ax.scatter(data.hub1_minutes_share,data.hub1_ec,c=[colors[x] for x in data.passing_style],s=30,alpha=.7);lo=min(data.hub1_minutes_share.min(),data.hub1_ec.min());hi=max(data.hub1_minutes_share.max(),data.hub1_ec.max());ax.plot([lo,hi],[lo,hi],color='gray',ls='--')
        for r in examples.itertuples():ax.annotate(r.network,(r.hub1_minutes_share,r.hub1_ec),xytext=(4,4),textcoords='offset points',fontsize=8)
        ax.set(xlabel='Leading passing hub’s share of team minutes',ylabel='Same player’s share of combined eigenvector centrality',title='Prominence above playing-time share\nDiagonal is a descriptive equality reference, not an expected passing model');save(fig,'hub_prominence_vs_minutes')
        fig,ax=plt.subplots(figsize=(12,7));bottom=np.zeros(len(counts))
        for style in labels:
            if style in counts:ax.bar(counts.index,counts[style],bottom=bottom,color=colors[style],label=labels[style]);bottom+=counts[style].to_numpy()
        ax.set(ylabel='Teams',title='Passing-style counts by season using one fixed reference');ax.legend(fontsize=8);save(fig,'styles_by_season')
        for r in examples.itertuples():
            own=players[players.network.eq(r.network)].sort_values(['combined_ec','player_id'],ascending=[False,True]);c=own.combined_ec.to_numpy();m=own.minutes.to_numpy()/own.minutes.sum();shown=list(range(min(8,len(own))));cs=list(c[shown]);ms=list(m[shown]);names=own.name.iloc[shown].tolist()
            if len(own)>8:cs.append(sum(c[8:]));ms.append(sum(m[8:]));names.append('All other players')
            fig,ax=plt.subplots(figsize=(13,8));x=np.arange(len(names));ax.bar(x-.18,cs,width=.36,label='Eigenvector share',color=colors[r.passing_style]);ax.bar(x+.18,ms,width=.36,label='Minutes share',color='#b9c2c9');ax.set_xticks(x,names,rotation=30,ha='right',fontsize=8);ax.set(ylabel='Share',title=r.network+' · '+labels[r.passing_style]+f'\nRatio={r.centralization_ratio:.3f}; leading hubs: {r.hub1_name} / {r.hub2_name}');ax.legend();fig.text(.5,.02,f'Cutoff agreement={r.label_agreement_across_cutoffs:.0%}; {r.label_status}.\nOther players are aggregated for display only; all players are included in scores.',ha='center',fontsize=9);save(fig,'example_shares_'+r.network)
            frame=pd.read_csv(root/'full_roster_entropy_centrality/historical_full_rosters'/r.season/r.team/'passing_matrix.csv',index_col=0);ids=frame.index.astype(int).tolist();A=frame.to_numpy();score=own.set_index('player_id').combined_ec;name=own.set_index('player_id').name;G=nx.DiGraph();G.add_nodes_from(range(len(ids)))
            for i,j in zip(*np.where(A>0)):G.add_edge(i,j,weight=float(A[i,j]))
            G.remove_nodes_from(list(nx.isolates(G)));H=nx.from_numpy_array(A+A.T);H.remove_nodes_from(list(nx.isolates(H)));pos=nx.spring_layout(H,weight='weight',seed=42);sizes=[60+5000*score[ids[i]] for i in G];fig,ax=plt.subplots(figsize=(12,9));nx.draw_networkx_edges(G,pos,ax=ax,node_size=sizes,width=[.15+2*np.sqrt(G[i][j]['weight']/A.max()) for i,j in G.edges()],alpha=.18,edge_color='#536d80',arrowsize=5,connectionstyle='arc3,rad=.1');nx.draw_networkx_nodes(G,pos,ax=ax,node_size=sizes,node_color='#d3e4ef',edgecolors=colors[r.passing_style]);top=[ids.index(int(pid)) for pid in own.player_id.head(5)]
            for yy,i in zip(np.linspace(.86,.20,5),sorted(top,key=lambda i:-pos[i][1])):ax.annotate(name[ids[i]],xy=pos[i],xytext=(.02,yy),textcoords='axes fraction',fontsize=9,ha='left',bbox=dict(fc='white',ec='none',alpha=.9),arrowprops=dict(arrowstyle='-',color='#536d80',lw=.7))
            ax.set_title(r.network+' · '+labels[r.passing_style]);ax.axis('off');ax.margins(.2);fig.text(.5,.02,'Node size: combined EC share; arrows: actual pass direction and volume; all positive edges included.\nFR layout is aesthetic. These are passing hubs, not a ranking of basketball star quality.',ha='center',fontsize=9);save(fig,'example_network_'+r.network,True)
    current=data[data.season.eq('2025-26')];rankrho=spearmanr(data.eigenvector_concentration,data.centralization_ratio).statistic
    report=['# Can a passing network be called star-centered or balanced?',
        'Yes as a transparent operational description of PASSING STRUCTURE, not an objective label of team superstar quality or scoring dependence. We should not force every team into two categories. This classifier keeps intermediate cases and separates one-hub, two-hub and broader-core patterns. Labels are provisional and relative to the180-network reference.',
        '## Main score',
        'Compute L1 eigenvector shares c from combined passing counts A+A.T. Passing HHI=sum(c²); effective passing hubs=1/HHI. Compute minute shares m from official team-player season minutes; minutes HHI=sum(m²); effective minutes players=1/minutesHHI. The score R=passingHHI/minutesHHI=effective_minutes_players/effective_passing_hubs. R=1 means equal inequality of the two distributions, NOT identical player shares. Higher R means more concentrated passing prominence relative to how minutes are distributed. This is an opportunity comparison, not a causal adjustment or an exact shared-minutes model.',
        f'Reference quartiles: balanced R≤{rules["balanced_ratio_max"]:.6f}; centralized R≥{rules["centralized_ratio_min"]:.6f}; middle cases intermediate. These cutoffs intentionally describe relative tails and therefore are not discovered natural classes or universally validated star thresholds. They do not force a50/50 split, but do identify reference quartiles by construction.',
        '## Which hub shape?',
        f'Rank shares c1≥c2≥c3. First gap=(c1−c2)/(c1+c2); second gap=(c2−c3)/(c2+c3). Among high-R networks, one-hub leaning requires first gap≥{rules["single_gap_min"]:.6f} and first≥second. Two-hub leaning requires second gap≥{rules["dual_gap_min"]:.6f} and second>first. Otherwise label centralized broader core. A two-hub leaning label describes a visible drop after the second player, not two NBA superstars.',
        '## Observed versus minutes-relative views',
        f'The observed_passing_style column applies the same hub-shape rules to raw normalized EC concentration, with reference quartiles {raw_rules["balanced_ratio_max"]:.6f} and {raw_rules["centralized_ratio_min"]:.6f}. The minutes_relative_style column describes extra concentration relative to playing opportunity. Unequal minutes can be part of a real star-centered style, so neither view replaces the other. consensus_style is metric_sensitive where their broad labels differ. These are descriptive relative tails, not discovered natural classes.',
        '## Current season',integrated_table(current[['team','observed_passing_style','minutes_relative_style','consensus_style','hub1_name','hub2_name','centralization_ratio','hub1_ec','top2_ec','label_status']].sort_values('centralization_ratio',ascending=False)),
        '## Illustrations',integrated_table(examples[['network','passing_style','hub1_name','hub2_name','centralization_ratio','gap1','gap2','label_agreement_across_cutoffs']]),
        '## Stability and limits',
        f'{int(data.threshold_stable.sum())}/180 fine labels agree with at least80% of27 reasonable cutoff combinations. {int(data.heldout_franchise_agrees.sum())}/180 agree when reference thresholds are estimated excluding that franchise. These are sensitivity checks, not confidence probabilities or validation against a true star-based label. Borderline cases remain visibly marked. Raw normalized concentration versus minutes-relative score has descriptive rank correlation {rankrho:.4f}; they answer different questions.',
        'The graphs retain all positive-minute roster participants, omit true graph isolates from EC, and use full season counts; no10-minute filter is imposed. Trading/injuries can make a season network combine players who did not share court time. Passing centrality can identify a center or connector instead of the leading scorer. Unequal minutes can be part of a real star-centered offense, so retain the raw EC concentration and top-player shares alongside the minutes-relative score. Do not interpret the label as a complete judgment of the offense, player skill, efficiency or winning.',
        'For a binary downstream comparison, use only clearly marked threshold-stable more-balanced versus more-centralized tail cases and disclose the excluded intermediate/borderline teams. Keeping the continuous R score generally loses less information. Outcomes were not used to choose the thresholds; this analysis does not claim a significant performance difference between categories.',
        'Outputs: all_team_season_styles.csv;2025-26_styles.csv;classification_rules.json;cutoff_sensitivity_grid.csv;illustrative_examples.csv;passing_structure_styles.pdf. Full source preserves earlier commands and adds --classify-passing-structure. A saved rules JSON can be supplied with --reference-rules to keep thresholds fixed for a new dataset.'
    ];(out/'METHOD_AND_RESULTS.md').write_text('\n\n'.join(report));print(pd.DataFrame([rules]).to_string(index=False),flush=True);print(data.passing_style.value_counts().to_string(),flush=True);print(current[['team','passing_style','hub1_name','hub2_name','centralization_ratio','label_status']].sort_values('centralization_ratio',ascending=False).to_string(index=False),flush=True);print('Saved '+str(out),flush=True);return 0

def run_decade(argv=None):
    import sys
    arguments = list(sys.argv[1:] if argv is None else argv)
    if '--pass-strength-entropy-regressions' in arguments:
        from pass_strength_entropy_regressions import main as pass_strength_regression_main
        return pass_strength_regression_main(arguments)
    if '--ec-hub-study' in arguments:
        from ec_hub_study import main as ec_hub_main
        return ec_hub_main(arguments)
    if '--split-role-hub-study' in arguments:
        from split_role_hub_study import main as split_role_hub_main
        return split_role_hub_main(arguments)
    if '--all-season-hub-study' in arguments:
        from all_season_hub_study import main as all_season_hub_main
        return all_season_hub_main(arguments)
    if '--hub-redistribution-study' in arguments:
        from hub_redistribution_study import main as hub_redistribution_main
        return hub_redistribution_main(arguments)
    if '--ball-control-dependency' in arguments:
        from ball_control_dependency import main as ball_control_dependency_main
        return ball_control_dependency_main(arguments)
    if '--ball-control-removed-networks' in arguments:
        from ball_control_removed_networks import main as ball_control_removed_networks_main
        return ball_control_removed_networks_main(arguments)
    if '--ball-control-forecast' in arguments:
        from ball_control_forecast import main as ball_control_forecast_main
        return ball_control_forecast_main(arguments)
    if '--creator-dependency' in arguments:
        from creator_dependency import main as creator_dependency_main
        return creator_dependency_main(arguments)
    if '--creator-removed-networks' in arguments:
        from creator_removed_networks import main as creator_removed_networks_main
        return creator_removed_networks_main(arguments)
    if '--creator-forecast' in arguments:
        from creator_forecast import main as creator_forecast_main
        return creator_forecast_main(arguments)
    if '--prospective-absence' in arguments:
        from prospective_absence import main as prospective_absence_main
        return prospective_absence_main(arguments)
    if '--minutes-adjusted-networks' in arguments:
        from minutes_adjusted_networks import main as minutes_adjusted_networks_main
        return minutes_adjusted_networks_main(arguments)
    if '--predicted-actual-networks' in arguments:
        from predicted_actual_networks import main as predicted_actual_networks_main
        return predicted_actual_networks_main(arguments)
    if '--volume-player-dependency' in arguments:
        from volume_player_dependency import main as volume_player_dependency_main
        return volume_player_dependency_main(arguments)
    if '--volume-removed-networks' in arguments:
        from volume_removed_networks import main as volume_removed_networks_main
        return volume_removed_networks_main(arguments)
    if '--ec-player-dependency' in arguments:
        from ec_player_dependency import main as ec_player_dependency_main
        return ec_player_dependency_main(arguments)
    if '--ec-removed-networks' in arguments:
        from ec_removed_networks import main as ec_removed_networks_main
        return ec_removed_networks_main(arguments)
    if '--onoff-dependency' in arguments:
        from onoff_dependency import main as onoff_dependency_main
        return onoff_dependency_main(arguments)
    if '--onoff-star-networks' in arguments:
        from onoff_star_networks import main as onoff_star_networks_main
        return onoff_star_networks_main(arguments)
    if '--league-network-changes' in arguments:
        from league_network_changes import main as league_network_changes_main
        return league_network_changes_main(arguments)
    if '--hub-entropy-regressions' in arguments:
        from hub_entropy_regressions import main as hub_entropy_regressions_main
        return hub_entropy_regressions_main(arguments)
    if '--league-star-presence-entropy' in arguments:
        from league_star_presence_entropy import main as league_presence_main
        return league_presence_main(arguments)
    if '--star-presence-entropy' in arguments:
        from star_presence_entropy import main as star_presence_entropy_main
        return star_presence_entropy_main(arguments)
    if '--dependency-network-cases' in arguments:
        from dependency_network_cases import main as dependency_cases_main
        return dependency_cases_main(arguments)
    if '--star-dependency-study' in arguments:
        from star_dependency_study import main as star_dependency_main
        return star_dependency_main(arguments)
    if '--revised-methodology' in arguments:
        from revised_methodology import main as revised_methodology_main
        return revised_methodology_main(arguments)
    if '--court-zone-networks' in arguments:
        from court_zone_networks import main as court_zone_network_main
        return court_zone_network_main(arguments)
    if '--court-zone-regressions' in arguments:
        from court_zone_regressions import main as court_zone_regression_main
        return court_zone_regression_main(arguments)
    if '--four-metric-regressions' in arguments:
        from four_metric_regressions import main as four_metric_main
        return four_metric_main(arguments)
    if '--structure-validation' in arguments:
        from structure_validation import main as structure_validation_main
        return structure_validation_main(arguments)
    if '--raw-count-star-entropy' in arguments:
        from raw_count_star_entropy import main as raw_count_star_main
        return raw_count_star_main(arguments)
    if '--action-weighted-star-entropy' in arguments:
        from action_weighted_star_entropy import main as weighted_star_main
        return weighted_star_main(arguments)
    if '--combined-star-entropy' in arguments:
        from combined_star_entropy import main as combined_star_main
        return combined_star_main(arguments)
    if '--star-entropy-regressions' in arguments:
        from star_entropy_regressions import main as star_entropy_main
        return star_entropy_main(arguments)
    if '--fg-attempt-networks' in arguments:
        from fg_attempt_networks import main as fg_attempt_main
        return fg_attempt_main(arguments)
    if '--lineup-entropy-analysis' in arguments:
        import importlib.util
        module_path = Path(__file__).resolve().with_name('lineup_entropy_analysis.py')
        spec = importlib.util.spec_from_file_location('_sportsanalysis_lineup_entropy', module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.main(arguments)
    if '--lineup-role-networks' in arguments:
        import importlib.util
        module_path = Path(__file__).resolve().with_name('lineup_role_networks.py')
        spec = importlib.util.spec_from_file_location('_sportsanalysis_lineup_roles', module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.main(arguments)
    if '--position-networks' in arguments:
        import importlib.util
        module_path = Path(__file__).resolve().with_name('position_networks.py')
        spec = importlib.util.spec_from_file_location('_sportsanalysis_position_networks', module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.main(arguments)
    if '--entropy-structure-profile' in arguments:
        import importlib.util
        module_path = Path(__file__).resolve().with_name('entropy_structure_profiles.py')
        spec = importlib.util.spec_from_file_location('_sportsanalysis_entropy_structure', module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.main(arguments)
    if '--offensive-structure-profile' in arguments:
        import importlib.util
        module_path = Path(__file__).resolve().with_name('offensive_structure_profiles.py')
        spec = importlib.util.spec_from_file_location('_sportsanalysis_offensive_structure', module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.main(arguments)
    if '--usage-starness' in arguments:
        import importlib.util
        module_path = Path(__file__).resolve().with_name('usage_analysis.py')
        spec = importlib.util.spec_from_file_location('_sportsanalysis_usage_starness', module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.main(arguments)
    if '--basket-regressions' in arguments:
        import importlib.util
        module_path = Path(__file__).resolve().with_name('basket_regressions.py')
        spec = importlib.util.spec_from_file_location('_sportsanalysis_basket_regressions', module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.main(arguments)
    if '--basket-networks' in arguments:
        import importlib.util
        module_path = Path(__file__).resolve().with_name('basket_networks.py')
        spec = importlib.util.spec_from_file_location('_sportsanalysis_basket_networks', module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.main(arguments)
    if '--sportvu-passing-sequences' in arguments:
        import importlib.util
        module_path=Path(__file__).resolve().with_name('sportvu_passing_sequences.py')
        spec=importlib.util.spec_from_file_location('_sportsanalysis_sportvu_sequences',module_path)
        module=importlib.util.module_from_spec(spec)
        sys.modules[spec.name]=module
        spec.loader.exec_module(module)
        return module.main(arguments)
    if '--classify-passing-structure' in arguments:
        return run_passing_structure_classifier(arguments)
    if '--deep-passing-458' in arguments:
        return run_deep_passing_458(arguments)
    if '--eight-full-roster-studies' in arguments:
        return run_eight_full_roster_studies(arguments)
    if '--season-entropy-centrality' in arguments:
        return run_season_entropy_centrality(arguments)
    if '--full-roster-entropy-centrality' in arguments:
        return run_full_roster_entropy_centrality(arguments)
    if '--starter-passing-analysis' in arguments:
        return run_starter_passing_analysis(arguments)
    if '--all-season-partnerships' in arguments:
        return run_all_season_partnerships(arguments)
    if '--integrated-network-study' in arguments:
        return run_integrated_network_study(arguments)
    if '--warriors-probability-ideas' in arguments:
        return run_warriors_probability_ideas(arguments)
    if '--warriors-markov-checks' in arguments:
        return run_warriors_markov_checks(arguments)
    if '--warriors-entropy-ideas' in arguments:
        return run_warriors_entropy_ideas(arguments)
    if '--warriors-role-checks' in arguments:
        return run_warriors_role_checks(arguments)
    if '--passing-lineup-balance' in arguments:
        return run_passing_lineup_balance(arguments)
    if '--passing-hub-followups' in arguments:
        return run_passing_hub_followups(arguments)
    if '--assist-creator-followups' in arguments:
        return run_assist_creator_followups(arguments)
    if '--assist-network-ideas' in arguments:
        return run_assist_network_ideas(arguments)
    if '--centrality-idea-suite' in arguments:
        return run_centrality_idea_suite(arguments)
    if '--hub-successor-analysis' in arguments:
        return run_hub_successor_analysis(arguments)
    if '--four-tactical-questions' in arguments:
        return run_four_tactical_questions(arguments)
    if '--passing-volume-by-style' in arguments:
        return run_passing_volume_by_style(arguments)
    if '--eight-context-topics' in arguments:
        return run_eight_context_topics(arguments)
    if '--six-network-topics' in arguments:
        return run_six_network_topics(arguments)
    if '--roles-and-familiarity' in arguments:
        return run_roles_familiarity(arguments)
    if '--defense-passing-disruption' in arguments:
        return run_defense_disruption(arguments)
    if '--hub-absence-resilience' in arguments:
        return run_hub_absence_resilience(arguments)
    if '--team-season-five-topics' in arguments:
        return run_five_team_season_topics(arguments)
    if '--team-season-topics' in arguments:
        return run_team_season_topics(arguments)
    if '--warriors-open-shots' in arguments:
        return run_warriors_open_shots(arguments)
    if '--warriors-eigenvector-performance' in arguments:
        return run_warriors_eigenvector_performance(arguments)
    if '--warriors-stability' in arguments:
        return run_warriors_stability(arguments)
    if '--warriors-82' in arguments:
        return run_warriors_82(arguments)
    if '--rebuild-saved-10min' in arguments:
        return rebuild_saved_ten_minute_analysis(arguments)
    if '--series-analysis' in arguments:
        return run_series_analysis(arguments)
    return _original_run_decade_with_series(arguments)



if __name__ == "__main__":
    raise SystemExit(run_decade())
