# Passing-sequence networks from movement tracking

Created an exploratory sequence-network pipeline using the requested sealneaward/nba-movement-data repository, pinned to commit `5c9ee6e1387e052c19ab2d22e12120c2210cbe54`.

## What was processed

Three historical Warriors games, with both teams included:

| Date | Game | GSW inferred passes | Opponent inferred passes |
|---|---|---:|---:|
|2016-01-04|Charlotte at Golden State|222|157|
|2016-01-05|Golden State at Lakers|164|141|
|2016-01-08|Golden State at Portland|204|131|

Total:1019 retained inferred passes, including 590 Warriors passes.862 match both stricter and more relaxed settings. There are 608 continuous passing chains,243 with at least two passes, and 411 ordered three-player transitions.79 distinct game/team/five-player lineup matrices were also saved.

This is a three-game 2015–16 pilot, not the full repository or the earlier 2020 s season dataset.

## Why this is a sequence network

The old season matrices counted how often A passed to B but did not establish what happened immediately before or after.

Here, ball/player positions are used to infer time-ordered transfers. For example, the inferred pattern **Stephen Curry → Draymond Green → Klay Thompson** appears five times in the Warriors sample.

Outputs include:

1. **First-order network:** nodes are players; arrows count inferred passes.
2. **Second-order network:** nodes are ordered pairs, such as Curry→Green. An arrow to Green→Thompson represents that inferred three-player sequence.
3. **Tracked-lineup networks:** each matrix groups retained passes when the same five teammates are tracked on court. Substitutions are never crossed.
4. **Sequence list:** timestamps, player IDs/names, lineup IDs, chain IDs and threshold-sensitivity flags.

The second-order figures show a 14-state projection for readability. Full state matrices and triple counts are saved. Missing outgoing probability rows remain undefined rather than being made absorbing states. First-/second-order training entropy differences do not by themselves prove a memory effect or the Markov property.

## How passes were inferred

The tracker identifies a plausible owner only when the ball stays close to one player, sufficiently farther from others, for a minimum duration. It then retains same-team changes with free-flight evidence, plausible movement and stable lineups.

The pipeline breaks sequences at tracking gaps, substitutions, play-by-play stoppages, likely shots/high balls, opponent control and rejected transfers. It removes overlapping duplicate event windows first; otherwise the same physical pass could be counted more than once.

The primary settings are 3 ft ownership radius,0.75 ft nearest-player margin,0.12 s stable support, at least 4 ft ball displacement, at most 1.2 s transfer interval and ball height no greater than 8.5 ft. A pass also needs a frame where the ball is farther than 3 ft from every player. These are heuristic settings, not learned pass probabilities.

Consequently, short passes, handoffs, lobs, near-shot passes and inbound/stopped-clock passes can be omitted. Threshold agreement is not a probability that a pass is correct.

## Important validation limit

These are **movement-inferred and incomplete sequences**, not official annotated passes.

A limited check detected 55 of 145 credited assister→shooter pairs in the supplied play-by-play within the specified preceding-time window. This is not an estimate of precision or comprehensive recall: final assists are only one subset of passes, event times are rounded and the extractor intentionally removes near-event frames.

The result should be used as an exploratory prototype until more video or manually labeled tracking segments are reviewed. The saved contact sheets show selected inferred transfers for visual review; they do not validate every pass.

## Files and reproduction

Open [the complete 16-page graph PDF](graphs/passing_sequence_networks.pdf). Individual PNG/PDF graphs and tracking contact sheets are in `graphs/`. Compact sequence tables, network matrices, settings, and validation checks are in `results/`. Five-player matrices are under `results/games/GAMEID/lineup_networks/`.

Install Python dependencies and libarchive (`bsdtar`, for sample extraction), then run from this folder:

```sh
python3 -m pip install -r requirements.txt
python3 fetch_sportvu_sample.py --dest sportvu_sequence_data/raw
python3 sportvu_passing_sequences.py --tracking-dir sportvu_sequence_data/raw --output-dir regenerated_results
python3 tests/test_sequence_extraction.py
```

Raw tracking archives, extracted coordinates, and frame caches are excluded because they are large. The downloader retrieves the three pinned games and verifies their Git blob hashes. No original repository setup scripts are run. `regenerated_results` includes the larger frame-level audits.

The synthetic test checks pass chronology, sequence breaks, shots, dribbles, opposing control, substitutions, tracking gaps, and overlapping event windows. It does not establish accuracy on all NBA passes. The saved actual-data checks are exploratory and have the limits described above.

Source: [sealneaward/nba-movement-data](https://github.com/sealneaward/nba-movement-data), which credits neilmj for the original data. This folder is an independent addition; existing SportsAnalysis files are preserved.
