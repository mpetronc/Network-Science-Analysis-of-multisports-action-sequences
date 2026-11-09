# NBA passing network analysis

`graphit.GPT.py` analyzes season-aggregated NBA passing profiles. It does not
reconstruct ordered possessions or action sequences. `nba_api` is a community
client, not an NBA-supported SDK. Data source: [NBA.com](https://www.nba.com/stats).

Install `requirements.txt` into your PyCharm project interpreter, then run
`graphit.GPT.py` for all teams in the 2025–26 regular season. Python 3.10+ is required.
League-wide runs save plots without opening a window for every team.

```sh
python -m pip install -r requirements.txt
python graphit.GPT.py
python graphit.GPT.py --team GSW
python graphit.GPT.py --team BOS --no-show
python graphit.GPT.py --team ALL --no-show
python graphit.GPT.py --team GSW --refresh --no-show
python graphit.GPT.py --team GSW --max-cache-age-hours 24 --no-show
python -m unittest discover -s . -p 'test_graphit.py' -v
```

## Results and interpretation

Results go into `nba_passing_output/<season>/<season type>/`, beside the script
unless `--output-dir` specifies another folder. After **each team**, the pipeline
saves that team's files and updates these league-wide CSVs before plotting or
requesting the next team:

- `team_metrics.csv`: one row per saved team, with entropy metrics, pass totals,
  coverage, timestamps, and status. Failed teams have `NA` metrics, rather than
  invented zero values. If a failed attempt has earlier saved results, those
  remain with status `previous_results` and an explanation in `latest_attempt_error`.
- `all_team_passes.csv`: one row per directed passer–receiver pair, with team,
  season, player IDs/names, counts, profile availability, and source. Self-pairs
  are excluded. Known zero counts remain zero; unknown outgoing counts are `NA`.

Each team also gets its own subfolder containing:

- `passing_network.png`: directed graph, with NBA.com attribution.
- `passing_network.graphml`: reusable directed graph data for NetworkX, Gephi,
  or other GraphML-compatible tools. Includes every player (even isolates),
  player IDs/names, display labels, available-profile flags, missing-data reasons,
  known passing involvement, and directed edges weighted by recorded passes.
- `passing_matrix.csv`: player IDs identify rows/columns; `NA` means unknown
  outgoing passes. The diagonal is zero because self-passes are excluded.
- `players.csv`: player names and outgoing-profile availability/reasons.
- `passes.csv`: that team's passer–receiver pairs, in the same format as
  `all_team_passes.csv`.
- `metrics.csv`: one CSV row of team metrics and coverage.
- `metrics.json`: metrics, coverage, source, query parameters, and snapshot dates.

The graph data is saved with the CSV checkpoint before plotting or moving to
the next team. GraphML stores observed positive edges only; an absent edge from
a player with an unavailable outgoing profile is unknown. `passing_matrix.csv`
and `passes.csv` retain the explicit `NA` values for those unknown counts.
The PNG is also saved for each team during the run.

To reopen a saved graph in Python:

```python
from pathlib import Path
import networkx as nx

path = Path("NBA/nba_passing_output/2025-26/Regular_Season/GSW/passing_network.graphml")
graph = nx.read_graphml(path, node_type=int)
```

`run_summary.json` is updated after each team and records successes, failures,
plot errors, and unprocessed teams. CSVs are written to temporary files, flushed,
and replaced atomically, so an interrupted replacement keeps the previous file.
Completed teams remain saved if another team fails or the run is interrupted.
Re-running replaces each team's rows without appending duplicates, and the
combined CSVs retain other teams saved in the same season folder. Cached player
responses can be reused on a restart; the script still visits the selected teams.
A nonzero exit status indicates incomplete profiles, failed teams, plot errors,
or stopped downloads; any available CSV results are still saved.

Receiver entropy is `H(receiver) / ln(n)`. Conditional receiver entropy is
`H(receiver | passer) / ln(n - 1)`, weighted by each passer's share of recorded
passes. For two players, conditional entropy is zero. With no passes, entropy
is undefined and stored as JSON `null`.

Every season participant with positive games/minutes stays in the network,
including players whose downloads fail. Known passes to those players are
retained. Receivers absent from the roster response are also retained and
flagged as missing outgoing profiles. Empty tracking profiles are treated as
unavailable, since they do not establish that a player made zero passes.

When profiles are missing, the main entropy fields are `null`; separate
`observed_only_*` values are explicitly provisional. Even complete player
downloads do not establish tracking coverage for every game/pass. Playing time,
lineup availability, roster size, and missing tracking affect interpretation;
these metrics alone do not identify offensive style or tactical unpredictability.
The conditional measure is not labeled an entropy rate: no stationary Markov
model of actual possession sequences is assumed.

## Requests and caching

Requests are sequential, spaced by 2–3.5 seconds, and retried at most four times
with exponential backoff. HTTP 401/403 stops downloads. HTTP 429 respects
`Retry-After` when supplied; persistent rate limits stop the run. A requested
cooldown longer than five minutes stops the run instead of retrying early.
Three consecutive exhausted request failures also stop the run. No unsupported
socket-reset method or access-block bypass is used. Headers/timing do not
guarantee access.

JSON snapshots are stored in `~/nba_passing_cache`, keyed by the endpoint and all
resolved query parameters (including API defaults), including team, player,
season, season type, and totals mode. Each snapshot records its retrieval time,
library version, and source. Writes are
atomic. Old `.pkl` files are not reused. By default, valid snapshots are retained
for reproducibility; use `--refresh` or `--max-cache-age-hours` when fresh data is
needed. The roster query is filtered by team and season on the server to include
historical participants, rather than treating a current roster as a season roster.

The library's MIT license covers its code; downloaded data is subject to the
[NBA terms](https://www.nba.com/termsofuse). Review permitted uses before public
publication or redistribution. Cache files and generated data are ignored by Git
by default. Nothing in this repository guarantees NBA authorization or access.
