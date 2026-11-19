# First-20 passing hubs across the six saved seasons

Run from the SportsAnalysis project:

    python teamgraphs.py --all-season-hub-study --workers 4

The existing commands remain available. Seasons default to 2020-21 through 2025-26. Use --seasons to select a subset. --report-only rebuilds reports from completed seasons without requesting new data. Completed seasons and validated profiles are reused on resume.

The hub is the first-20-game leader in passes sent plus received. The predictor is first-20 passing-only eigenvector concentration, calculated on A+A.T with L1 eigenvector shares and normalized HHI. The original network uses training games in which the hub appeared. Predictions use only training minutes, earlier replacement gains and previous-season NBA position labels. Outgoing hub and incoming hub edges are reassigned by forecast extra-minute shares, excluding self-passes and preserving pass mass. Shooting edges never enter these networks or EC.

Later absent games provide the held-out outcome. Original, predicted and observed networks use the same within-team entropy reference. Fits require at least 15 early hub-present games and 10 analyzed later absent games. Equal-pass-count and raw-count regressions are exported separately. The later-present baseline is a sensitivity, not training data. Pooled fits include season indicators and franchise-clustered uncertainty; they do not treat seasons from the same franchise as independent observations.

Tracking gaps are not zero passes. Empty profiles require independent zero confirmation. Ambiguous multi-game absence windows are split around affected appearances before declaring data unavailable. Unresolved absence windows are excluded as whole windows; superseded source windows are retained for audit. Official and analyzed absence counts are separate. If exclusions prevent a validated later-present reference, it is stored as NaN and omitted from that sensitivity fit. The main three-network comparison can still use the remaining verified absence games.

Dependencies: numpy, pandas, scipy, matplotlib, networkx, requests, nba_api. pypdf combines the individual season PDFs and adds a season label on every page. The numerical verifier uses pdfinfo. Official team/player game logs must be present in nba_passing_output/star_dependency_study/game_cache; fetch_dependency_games.py retrieves them if needed.

Outputs: per-season 30-team network PDFs, entropy PDFs and matrices; combined reports; all_team_season_entropy_and_ec.csv; season_regressions.csv; pooled_season_adjusted_regressions.csv; prediction_performance_by_season.csv; mechanical_baseline_diagnostics.csv; profile_tracking_coverage.csv; tracking exclusions and validation records.

The replacement rule remains an exploratory assumption. Its weights were carried over rather than selected to maximize the regressions. Aggregate entropy can reflect lineup changes and mathematical node-removal effects. Higher entropy is not necessarily better offense, and these comparisons do not identify causal star dependency. R-squared describes fitted association, not out-of-sample regression accuracy. Equal-pass-count draws are not independent games or confidence intervals for causal effects.
