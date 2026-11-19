# Passing hub network study

All 30 NBA teams across six seasons (2020-21 through 2025-26): **180 three-network comparisons**. This folder groups the graphs, source matrices, regressions, prediction comparisons, replacement roles, methodology and code packages.

## Start here

- [All 180 original/predicted/actual network panels (PDF)](prediction_comparisons/all_180_split_role_networks.pdf)
- [Prediction comparisons and diagnostics (PDF)](prediction_comparisons/passing_role_model_comparison.pdf)
- [Prediction comparison data](prediction_comparisons/prediction_comparison.csv)
- [Team-by-team interpretation](prediction_comparisons/ALL_TEAM_COMPARISONS.md)
- [Player minutes and replacement passing roles](replacement_roles/minutes_and_passing_role_weights.csv)
- [Replacement model methodology](methodology/replacement_model_methodology.md)
- [Baseline, hub selection and entropy methodology](methodology/baseline_and_entropy_methodology.md)
- [Passing-involvement regressions (PDF)](regressions/passes_in_plus_out/pass_strength_entropy_regressions.pdf)
- [Passing-involvement regression methodology and results](regressions/passes_in_plus_out/METHOD_AND_RESULTS.md)
- [Eigenvector-index regressions (PDF)](regressions/eigenvector_centrality/all_seasons_regressions.pdf)
- [Every team's entropy changes (PDF)](entropy_changes/all_30_teams_entropy_changes.pdf)
- [Entropy-change analysis](entropy_changes/ANALYSIS.md)

## Browse every team's three networks

- [2020-21 - all 30 teams](networks/2020-21/)
- [2021-22 - all 30 teams](networks/2021-22/)
- [2022-23 - all 30 teams](networks/2022-23/)
- [2023-24 - all 30 teams](networks/2023-24/)
- [2024-25 - all 30 teams](networks/2024-25/)
- [2025-26 - all 30 teams](networks/2025-26/)

Each team folder includes the three-network image, original/predicted/actual passing matrices, first-20 selection matrix, and frozen minutes and passing-role specifications. Missing actual comparisons have no observed network; any placeholder matrix must not be interpreted as evidence of zero passing. The primary sample is 96 eligible team-seasons; 171 have observed absence comparisons.

## Method in plain language

1. Identify the passing hub by the largest passes sent plus received during the team's first 20 games. This is the restored volume-selected hub study, not the alternative EC-selected-player study or NBA tracking touches selection.
2. Build the original network from early hub-present games. All selection and role inputs precede the first-20 cutoff.
3. Estimate who replaces the hub's minutes using early rotation and position evidence, respecting 240 team minutes and a 48-minute individual limit for a regulation-length forecast.
4. Estimate replacement sending and receiving roles separately using forecast total minutes and shrunk early passes per minute. Early absence evidence is blended conservatively when available. These weights can differ from the shares of extra minutes.
5. Keep existing nonhub connections fixed. Redistribute outgoing hub edges across replacement senders and incoming hub edges across replacement receivers. Exclude self-passes and preserve pass mass. Remove the hub's row and column.
6. Build the actual network from later tracked games when the hub did not play. Size nodes using each graph's own passing involvement; fixed Fruchterman-Reingold positions aid visual comparison.
7. Compare outgoing conditional passing entropy and actual edge-share differences. Equal-pass-count observed entropy changes are used in the regressions; full-network levels appear in the scenario plots. Forecast entropy errors and network edge-share errors are separate outcomes.
8. Relate observed entropy change to early hub passing involvement, with season adjustment and franchise-clustered pooled inference. The original first-20 team EC-concentration-index regressions are included as a separately labelled alternative predictor; EC is calculated on passing edges only.

## Key results and limits

The passes-in-plus-out involvement regression uses 96 eligible team-seasons: season-adjusted r=0.383, R2=14.6%, Holm-corrected pooled p=0.0007. A later hub-present baseline gives r=0.368, R2=13.5%, corrected p=0.0027. These are structural associations, not causal effects or proof of better offense. Corrections cover the declared pooled tests, not every exploratory analysis across the project.

The separate passing-role forecast improves median absolute entropy error from 0.0433 to 0.0289, with 62/96 improvements. Median edge-share error changes from 0.4944 to 0.4981, with only 34/96 improvements. Similar entropy does not imply similar connections. The method has not achieved an overall improvement in predicting who passes to whom.

Higher conditional entropy describes more diverse receiver choices given the passer; it does not establish equal player involvement or offensive quality. Mechanical hub removal, roster mixing, opponents, co-absence, temporal change and limited absence samples remain relevant. The work is retrospective and exploratory, not independent external validation.

## Replacement-role data

The replacement-role ledger records projected minutes, minute replacement shares, sender replacement shares and receiver replacement shares for every player. Team-specific specifications and frozen predictions are next to the network matrices. The methodology documents describe the shrinkage and routing rules in detail.

## Code and provenance

- [Complete split-role model code](code/split_role_hub_full_code.zip)
- [Complete regression code](code/pass_strength_regressions_full_code.zip)
- [Entropy graphs, data and plotting code](code/team_entropy_changes_graphs_and_data.zip)
- [Working project modules](../SportsAnalysis/)

These are the already calculated saved results. No new NBA downloads, hub reselection or recalculation were performed for this upload. Some scripts retain paths from the original computer; adjust those to rerun elsewhere. Validation and source hashes are included in `validation/`; `UPLOAD_MANIFEST.json` lists the new files and their Git blob hashes. Earlier SportsAnalysis publications remain available.
