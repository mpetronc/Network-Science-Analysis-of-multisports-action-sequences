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
    # Include additional receivers discovered in passing profiles.
    for frame in profiles.values():
        for row in frame.to_dict(orient="records"):
            receiver = int(row["PASS_TEAMMATE_PLAYER_ID"])
            if (
                receiver not in ids
                and row["PASS"] > 0
                and not config.rotation_only
            ):
                ids.add(receiver)
                names[receiver] = str(
                    row.get("PASS_TO", receiver)
                )
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
        settings or {},
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
    if not config.rotation_only:
        known = {
            int(pid)
            for pid in roster["PLAYER_ID"]
        }
        for _ in range(3):
            new = set()
            for frame in profiles.values():
                new |= set(
                    frame.loc[
                        frame["PASS"] > 0,
                        "PASS_TEAMMATE_PLAYER_ID",
                    ].astype(int)
                )
            new -= known
            if not new:
                break
            for pid in sorted(new):
                known.add(pid)
                extras_fetched.append(pid)
                download(
                    pid,
                    f"unlisted player {pid} (traded or part-season)",
                )
    settings = {
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
    graph = build_graph(data)
    fig, ax = plt.subplots(figsize=(12, 10))
    positions = nx.spring_layout(
        graph.to_undirected(),
        seed=42,
        weight=None,
    )
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
    labels = short_labels(data)
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
    ax.axis("off")
    fig.text(
        0.5,
        0.025,
        "Source: NBA.com | Node size: total incoming and outgoing activity | "
        "Edge width: tracked passes\n"
        "Gray nodes: unknown outgoing passing profile",
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
    return graph
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


def run_decade(argv=None) -> int:
    """Run every selected 2020s NBA season, preserving each season's outputs."""
    import sys

    decade_parser = argparse.ArgumentParser(add_help=False)
    decade_parser.add_argument("--all-2020s", action="store_true")
    decade_parser.add_argument("--start-year", type=int, default=2020)
    decade_parser.add_argument("--end-year", type=int, default=2025)
    batch_args, remaining = decade_parser.parse_known_args(argv)

    if not batch_args.all_2020s:
        if "--start-year" in remaining or "--end-year" in remaining:
            decade_parser.error("--start-year and --end-year require --all-2020s")
        return main(remaining)

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
        decade_parser.error("Do not specify --season together with --all-2020s")

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
    for year in range(start, end + 1):
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
    for item in statuses:
        path = base / item["season"] / mode
        if rotation:
            path = path / rotation
        csv_path = path / "team_metrics.csv"
        if csv_path.exists():
            frame = pd.read_csv(csv_path)
            if not frame.empty:
                frame["season"] = item["season"]
                frames.append(frame)

    decade_dir = base / "2020s_summary" / mode
    if rotation:
        decade_dir = decade_dir / rotation
    decade_dir.mkdir(parents=True, exist_ok=True)
    atomic_csv(pd.DataFrame(statuses), decade_dir / "season_run_status.csv")

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        combined["season_start_year"] = combined["season"].str[:4].astype(int)
        combined = combined.sort_values(["season_start_year", "team"])
        atomic_csv(combined, decade_dir / "all_seasons_team_metrics.csv")
        plot_decade_trends(combined, decade_dir)
        plot_decade_team_scatter(combined, decade_dir)
        LOG.info("Combined decade results: %s", decade_dir.resolve())
    else:
        LOG.warning("No season metrics were available for decade aggregation")

    return int(any(item["exit_code"] != 0 for item in statuses))


def plot_decade_trends(frame: pd.DataFrame, output_dir: Path) -> None:
    """Plot year-by-year league medians for the three existing measures."""
    import matplotlib.pyplot as plt

    metrics = [
        ("observed_only_receiver_entropy", "Receiver entropy"),
        ("observed_only_conditional_receiver_entropy", "Conditional receiver entropy"),
        ("usage_hhi_starness", "Usage HHI starness"),
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


if __name__ == "__main__":
    raise SystemExit(run_decade())
