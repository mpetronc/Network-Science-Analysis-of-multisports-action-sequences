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




_original_run_decade_with_series = run_decade

def run_decade(argv=None):
    import sys
    arguments = list(sys.argv[1:] if argv is None else argv)
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
