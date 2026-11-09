# Premier League and NBA passing study

Run **football.py** from PyCharm using the project interpreter. The default run reads the saved NBA 2025–26 regular-season exports, the 2015–16 StatsBomb Premier League season, and the 2014–15 through 2025–26 OpenPlay Premier League release. It saves the tables and figures under **Foot/output**.

From the project terminal:

```sh
python -m pip install -r Foot/requirements.txt
python Foot/football.py --offline
python Foot/regression.py
python Foot/followup.py
python Foot/season_reports.py
python Foot/season_exports.py
python -m unittest discover -s Foot -p 'test_football.py'
```

Use **--offline** when the input cache already exists. A fresh run can download the public data with system **curl**, or you can cache it first with **--download-only**. **--cache**, **--output**, **--nba-dir**, and **--nba-season** let you select other paths or a different saved NBA season. For the recorded-receiver analysis alone, use **--statsbomb-only --output Foot/output/statsbomb_only**. **--no-plots** regenerates the tables alone. Cached data and generated output stay local.

## What to open

| Output | Contents |
|---|---|
| `premier_league_2020-21_to_2025-26/` | The six Premier League seasons matching the NBA period. Each club-season has a weighted PNG and GraphML network, pass matrix, edge list, player and roster tables, and metrics in CSV and JSON. The package includes both roster versions, season-wide tables, trend plots and a 38-match Manchester City case study. |
| `season_reports/` | Football-only season trends, match metrics, minutes sensitivity, and a Leicester City stability report. Use `season_reports.py --club "Club Name"` for another club from the 2015–16 StatsBomb season. |
| `star_entropy_followup.png` and `star_entropy_followup_summary.json` | A lineup-constrained passing shuffle and a chronological forecast check. The companion CSVs give simulated slopes, club entropy gaps and held-out predictions. Run `followup.py` from the cached event files and saved StatsBomb team table. |
| `star_entropy_regression.csv` and `star_entropy_regression.png` | Regressions of passing entropy on PageRank concentration. Run `regression.py` to rebuild them from saved results alone. |
| `nba_football_comparison.png` | Shared conditional receiver entropy and PageRank concentration chart. Hollow NBA markers show teams with partial tracking profiles or prior results. |
| `football_networks/` and `nba_networks/` | Passing maps with node sizes set by weighted PageRank and arrow widths scaled to pass counts within each team. Football maps include every player in the network, with a numbered PageRank roster. Pitch markers start at average pass locations and are spaced where they crowd. Players without a pass location appear below the pitch. Maps display up to 30 directed routes. The football GraphML files retain all edges and the original average locations. |
| `nba_football_team_season_metrics.csv` | Team entropies, PageRank HHI, normalized concentration, coverage, and within-league percentiles where at least five teams have complete profiles. NBA usage HHI appears when supplied by the saved export. |
| `statsbomb_team_match_metrics.csv` and `statsbomb_match_passes.csv.gz` | The two team networks for every fixture, with a count for each directed player pair. |
| `statsbomb_players.csv` and `nba_players.csv` | Passes sent and received, per-passer entropy, and PageRank. |
| `statsbomb_recipient_prediction.csv` and `statsbomb_possession_patterns.png` | Held-out completed-pass recipient prediction and the most frequent four recorded on-ball actions through the first shot of a possession. |
| `openplay_tactical_prediction.png` and `openplay_team_season_prediction.csv` | Held-out final-third channel prediction and its descriptive relationship with earlier non-penalty shot concentration. |
| `receiver_benchmark_teams.csv` and `receiver_benchmark_summary.json` | Cross-provider pass alignment, receiver agreement, entropy errors, and PageRank ranking agreement on the 2015–16 overlap. |
| `openplay_inferred_team_networks.csv` and `openplay_inferred_players.csv` | Separate estimated networks for all 12 seasons when the benchmark meets the extension criteria. Coverage is listed for each club-season. |

## Six-season exports

Run `season_exports.py` with the cached OpenPlay events. The default package contains 120 club-seasons for 2020–21 through 2025–26 in each of `all_players/` and `minutes_filter/`. The filter uses 18.75 minutes by default, as in the football season report. Every included player gets a node, with all positive inferred routes in GraphML and up to 30 displayed in the PNG. Each season folder also contains `team_metrics.csv` and `all_team_passes.csv`.

Passing weights count completed passes with inferred recipients. In a matrix, zero means zero assigned passes for that pair. Player and team coverage columns show the size of the unresolved portion. `coverage.csv` lists fixture availability and exclusions. The 2015–16 StatsBomb season remains the recorded-receiver benchmark.

The `manchester_city_2021-22/` folder contains the 38 match networks, receiving-share stability and win/draw/loss summaries in both roster versions. Football uses non-penalty shot concentration for its shooting metric.

## Methods

- The shared metrics use the NBA formulas for receiver entropy and conditional receiver entropy, with natural logarithms and the same roster-based denominators. Each pass carries its recorded count. Weighted receiving PageRank uses damping of 0.85. Team concentration rescales PageRank HHI from the uniform roster baseline `1/n` to the single-player maximum of 1. Unknown NBA passes stay flagged and only observed edges contribute to provisional values.
- StatsBomb networks use completed passes with a recorded recipient and pool the counts from 38 matches per club. The recipient model learns from earlier kickoff dates and is evaluated on later dates. It compares a passer-only model with one that also uses ball origin and the preceding action in the possession. Both models restrict predictions to on-pitch teammates. The output lists the eligible share and log loss in bits.
- OpenPlay predicts right, centre or left **among completed passes crossing into the final third**. The baseline uses club and pass origin. The second model also uses the previous recorded on-ball action. Both learn from earlier fixtures only, using prior seasons and the first half of the current season by kickoff date. The later fixtures are scored before being added to training. The destination only supplies the target. Shot HHI uses each player's share of the club's earlier non-penalty shots. The chart describes an association. Provider-flagged partial matches are excluded and listed in the coverage table.
- The receiver benchmark joins the overlap by fixture and maps players by club and shirt number. Passes align one to one by passer, period, time and pitch coordinates. The heuristic guesses the next plausible teammate on the ball within eight seconds and 14 normalized pitch units of the destination, and abstains when the evidence fails those checks. Agreement is measured on aligned passes with a guess. Club entropy and PageRank comparisons use the complete recorded network and the full inferred subset. The extended estimates require at least 80% alignment and inference coverage, 95% receiver agreement, median absolute conditional entropy error at most 0.02 and median club hub rank correlation at least 0.95. The overlap measures agreement in 2015–16, with later seasons identified as inferred.
- The star concentration regression estimates how conditional receiver entropy changes per 0.01 increase in normalized PageRank HHI. It fits recorded StatsBomb team-season values, inferred OpenPlay values with season effects, and inferred OpenPlay values with club and season effects. The latter uses clubs observed in at least two seasons and compares changes within a club after accounting for changes shared across a season. The 95% intervals resample entire clubs. This preserves repeated club observations but does not account for dependence through shared opponents. The NBA fit uses observed passes and is shown as provisional because of incomplete player profiles. Both measures come from the same passing network, so the regressions describe an association.
- The regression follow-up shuffles StatsBomb receivers inside each match and recorded lineup while preserving every player's pass totals sent and received. It excludes self-passes and keeps passes with uncertain lineups fixed. Four chains generate 400 reference seasons after warmup. The chronological check uses inferred OpenPlay receivers, trains on earlier seasons, and compares forecasting late-season entropy from early-season entropy with and without early hub concentration. Match dates determine the split. Provider-flagged partial games are excluded.
- The football season report includes mutual information in nats and divided by log(roster size), and effective receivers as exp(raw receiver entropy). Each report includes all players and a minutes-filtered cohort. The default threshold is 18.75 minutes, the same fraction of 90 minutes as 10 minutes of a 48-minute NBA game. For season networks the threshold uses average minutes per appearance. For match networks it uses minutes in that match. The filter retains only passes with two eligible endpoints. StatsBomb minutes use lineup stints and the elapsed match clock. OpenPlay minutes use its published lineup totals.
- The club stability report compares consecutive games using base-2 Jensen–Shannon divergence over receiving shares. One variant includes anyone who appeared in either match, and the other includes the players who appeared in both. The win, draw and loss summaries are descriptive. The season regressions use non-penalty shot concentration, with separate fits by receiver source and minutes cohort.
- The NBA comparison uses season totals. The two chronological prediction studies use football match events. Team-level ranks and plots carry the season and league labels so readers can distinguish roster and coverage differences.

## Data

- [StatsBomb Open Data, Premier League 2015–16](https://github.com/hudl/open-data/blob/master/data/matches/2/27.json). The code pins the source commit. Follow [StatsBomb's attribution terms](https://github.com/hudl/open-data#terms--conditions) for publication.
- [OpenPlay (Noah Bair), data-v0.1.0](https://github.com/nlbair/openplay/releases/tag/data-v0.1.0). The release reports 4,556 available fixtures and publishes the data under CC BY 4.0.
- [NBA.com tracking](https://www.nba.com/stats/players/passing). The study reads the existing NBA script's local exports.
