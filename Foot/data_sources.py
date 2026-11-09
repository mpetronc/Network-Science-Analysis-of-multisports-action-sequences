"""Download the fixed Premier League inputs into a reusable local cache."""

from concurrent.futures import ThreadPoolExecutor
import gzip
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

STATSBOMB_REF = "4b73468fc5b0f1950f9f66fada70ad3a4f9327cb"
STATSBOMB_URL = f"https://raw.githubusercontent.com/hudl/open-data/{STATSBOMB_REF}/data"
OPENPLAY_URL = "https://github.com/nlbair/openplay/releases/download/data-v0.1.0"
YEARS = tuple(range(2014, 2026))
SEASON_FORMAT = "{start}-{end:02d}"


def season_name(year):
    return SEASON_FORMAT.format(start=year, end=(year + 1) % 100)


def event_file(year):
    return f"events_eng_{year % 100:02d}{(year + 1) % 100:02d}.parquet"


def read_json(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


class DataSources:
    def __init__(self, directory, *, offline=False):
        self.directory = Path(directory)
        self.offline = offline

    def fetch(self, url, path):
        if path.is_file() and path.stat().st_size:
            return path
        if self.offline:
            raise FileNotFoundError(f"Missing cached input {path}. Run once without --offline.")
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=path.parent) as temporary_dir:
            download = Path(temporary_dir) / "download"
            subprocess.run(
                ["curl", "--fail", "--location", "--silent", "--show-error",
                 "--retry", "3", "--max-time", "90", "--compressed",
                 "--output", str(download), url],
                check=True,
            )
            if not download.stat().st_size:
                raise OSError(f"Empty response for {url}")
            if path.suffix == ".gz":
                compressed = Path(temporary_dir) / "compressed"
                with download.open("rb") as source, gzip.open(compressed, "wb", compresslevel=3) as target:
                    shutil.copyfileobj(source, target)
                download = compressed
            download.replace(path)
        return path

    def statsbomb_matches(self):
        path = self.directory / "statsbomb" / "matches.json"
        matches = read_json(self.fetch(f"{STATSBOMB_URL}/matches/2/27.json", path))
        if len(matches) != 380 or len({row["match_id"] for row in matches}) != 380:
            raise ValueError("Expected 380 distinct StatsBomb Premier League fixtures.")
        return sorted(matches, key=lambda row: (row["match_date"], row["kick_off"], row["match_id"]))

    def statsbomb_paths(self, matches, *, kind="events"):
        def get(row):
            match_id = row["match_id"]
            path = self.directory / "statsbomb" / kind / f"{match_id}.json.gz"
            return self.fetch(f"{STATSBOMB_URL}/{kind}/{match_id}.json", path)

        with ThreadPoolExecutor(max_workers=6) as pool:
            return list(pool.map(get, matches))

    def openplay(self, filename):
        return self.fetch(f"{OPENPLAY_URL}/{filename}", self.directory / "openplay" / filename)

    def preload(self):
        matches = self.statsbomb_matches()
        self.statsbomb_paths(matches)
        print(f"Cached StatsBomb events for {len(matches)} fixtures.", flush=True)
        self.statsbomb_paths(matches, kind="lineups")
        files = ["matches.parquet", "lineups.parquet", "known_gaps.csv", "data_issues.csv"]
        files.extend(event_file(year) for year in YEARS)
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(self.openplay, files))
        print(f"Cached OpenPlay match list, coverage notes and {len(YEARS)} seasons.", flush=True)
