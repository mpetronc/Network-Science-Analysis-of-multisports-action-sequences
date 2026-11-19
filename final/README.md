# Final: NBA passing-hub network study

**Chosen model:** first20 passes-sent-plus-received hub, separate replacement minutes/sending/receiving roles, and a historical adjustment learned from **other franchises in the immediately previous season only**.

**Evaluation:** at least10 tracked hub-absent games after game20. There are100 qualifying team-seasons and85 with historical training. All180 original/predicted/actual illustrations are retained across30 teams and six seasons; the default galleries and error summaries identify the qualifying sample.

## Open the results

- [All180 three-network comparisons: complete PDF](networks/all_180_team_networks.pdf)
- [Browse networks by season](networks/README.md)
- [Local interactive gallery](networks/index.html) — download/open locally to filter by team, season and eligibility; GitHub displays HTML as source.
- [Prediction comparison report](prediction_comparisons/TEAM_BY_TEAM.md)
- [Prediction error graphs](prediction_comparisons/initial_vs_final_errors.png)
- [Model comparison graph](prediction_comparisons/model_comparison.png)
- [All prediction errors: CSV](prediction_comparisons/all_team_prediction_errors.csv)
- [Only10+ absence comparisons: CSV](prediction_comparisons/ten_plus_absence_prediction_errors.csv)
- [Replacement minutes and passing responsibilities](replacement_roles/all_player_minutes_and_passing_roles.csv)
- [Historical adjustments to player roles](replacement_roles/historical_player_role_changes.csv)
- [Complete methodology and formulas](methodology/COMPLETE_METHODOLOGY.md)
- [Reproduce the calculations](REPRODUCING.md)
- [Download code and calculation inputs](code_and_data.zip)

## Team network galleries

Each panel reads **original early hub present → final predicted hub absent → actual later hub absent**. Node sizes reflect each panel's own passing involvement; isolates are omitted. Forecasts use the first20 target games plus permitted previous-season examples. Later target games evaluate predictions. Missing actual networks are labelled unavailable.

|Season|10+ absence cases|Default gallery|All teams|Download|
|---|---:|---|---|---|
|2020-21|15|[Qualifying networks](networks/2020-21/README.md)|[All30 illustrations](networks/2020-21/all_teams.md)|[Season PDF](networks/2020-21_all_30_teams.pdf)|
|2021-22|22|[Qualifying networks](networks/2021-22/README.md)|[All30 illustrations](networks/2021-22/all_teams.md)|[Season PDF](networks/2021-22_all_30_teams.pdf)|
|2022-23|19|[Qualifying networks](networks/2022-23/README.md)|[All30 illustrations](networks/2022-23/all_teams.md)|[Season PDF](networks/2022-23_all_30_teams.pdf)|
|2023-24|12|[Qualifying networks](networks/2023-24/README.md)|[All30 illustrations](networks/2023-24/all_teams.md)|[Season PDF](networks/2023-24_all_30_teams.pdf)|
|2024-25|14|[Qualifying networks](networks/2024-25/README.md)|[All30 illustrations](networks/2024-25/all_teams.md)|[Season PDF](networks/2024-25_all_30_teams.pdf)|
|2025-26|18|[Qualifying networks](networks/2025-26/README.md)|[All30 illustrations](networks/2025-26/all_teams.md)|[Season PDF](networks/2025-26_all_30_teams.pdf)|

The2020–21 predictions use the initial model because2019–20 network training is not saved. They are kept in the100-case evaluation but excluded from the85-case historical comparison.

## Prediction results on the same85 cases

Lower errors are better. Entropy errors are normalized units, not percentages of incorrect predictions.

|Model|n|Mean connection error|Median connection error|Median entropy error|
|---|---:|---:|---:|---:|
|Initial separate-role|85|0.5186|0.5018|0.0286|
|Final previous-season-only|85|0.4996|0.4850|0.0293|
|Simple deletion|85|0.5321|0.5172|0.0602|

The final model improves connection error for **67 of85** cases and worsens it for18. The improvement is modest; entropy is not consistently more accurate. Detailed means, medians and latest-season checks are in [model_comparison.csv](prediction_comparisons/model_comparison.csv). These are retrospective comparisons, not independent new-season validation.

## Observed entropy regressions

These regressions use **observed**, not predicted, entropy changes and are regenerated for the final10+ absence filter. They do not measure forecast accuracy.

- [Passing-involvement regression PDF: eight pages](regressions/passing_involvement/pass_strength_entropy_regressions.pdf)
- [Passing-involvement statistics](regressions/passing_involvement/pooled_regressions.csv)
- [Alternative passing-only EC-index regressions](regressions/eigenvector_concentration/ec_entropy_regressions.pdf)
- [Regression input and eligibility](regressions/regression_input_with_final_filter.csv)

For the100-case early-present reference, season-adjusted r=0.346, within-season R²=12.0%, Holm-corrected pooled p=0.0070. For the91-case later-present reference, r=0.365, R²=13.3%, corrected p=0.0040. Pooled uncertainty clusters by franchise. The additional later-reference requirement is at least15 later present games and a valid reference.

These numbers differ from the earlier96-case report because the current filter adds four cases with fewer than15 early present games. The source metadata retain the old historical-training eligibility; a separate temporary copy changes evaluation eligibility for these regressions. No target-season later outcomes enter target forecasts.

## Package contents and provenance

- `networks/`:180 full-resolution PNGs, complete PDF, six season PDFs, galleries and graph summary.
- `prediction_model/`:180 final predicted matrices, frozen training specifications, coefficients and temporal/franchise audit.
- `data/nba_passing_output/`:original and actual matrices, first20 inputs, frozen initial role forecasts, minute specifications, player names and source metadata needed for the calculations.
- `replacement_roles/`:player minute allocations, sender/receiver shares and learned role multipliers.
- `prediction_comparisons/`:same-sample accuracy summaries and charts.
- `regressions/`:observed passing-involvement and EC-index association outputs.
- `code/`:standalone selected-model and reproduction scripts.
- `validation/`:source hashes, reconstruction checks, sample checks and publication manifest.

The package preserves the chosen predictions exactly. Its learning uses only the immediately previous season, excludes the target franchise and does not use names/IDs as learned features. Historical labels legitimately come from that completed previous season. New-player availability, opponent effects, roster mixing, approximate replacement weights and shared entropy/centrality inputs remain limitations. Higher entropy does not establish better offense or causal star dependency.
