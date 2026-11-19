# First-20 passing hubs and role-weighted redistribution | 2025–26

## Definitions

The passing hub is the player with the largest sum of outgoing and incoming directed pass counts in the first twenty regular-season games (ties: lowest player ID). The EC index is a separate team measure: L1 eigenvector shares on A+Aᵀ, concentration (sum(p_i²)−1/N)/(1−1/N), using only active passing nodes in that first-20 network. No shooting edges or later-game data enter identity/index.

Original panel: observed first-20 games in which the selected hub appeared. Middle: counterfactual passing-share prediction frozen after game20. Third: all later missed games within the valid team window. Later hub-present games are also exported as a concurrent-reference sensitivity; they never train the prediction.

## Replacement minutes and depth

Official NBA CommonTeamRoster positions from the COMPLETED PREVIOUS season are used, so current/future roster moves are not exposed to the model. Candidates are only players observed in training. These are broad G/F/C position labels, not official historical depth-chart ordering. Depth rank is inferred from first-20 hub-present minutes within the overlapping position group. Missing/rookie position labels receive a neutral fallback.

Convert training hub-present minutes to a 240-minute regulation rotation. Remove the hub. Candidate fallback weight = available minute capacity (48−baseline minutes) × position match / inferred position depth rank. Position match is 1 for overlapping known positions, 0.5 if either is unknown and 0.25 otherwise. Among matching candidates, depth ranks follow baseline minutes excluding the hub.

If earlier hub-absent games occur WITHIN the first 20, calculate positive teammate minute gains relative to hub-present games, normalize, and blend with fallback weights using n_early_abs/(n_early_abs+5). This is a declared shrinkage convention, not tuned on later results. Allocate the removed hub minutes by those weights, capping each player at48 and redistributing excess among uncapped players. Actual later minutes, later roster arrivals and later passing are never inputs.

## Passing redistribution

Keep all nonhub-to-nonhub edges unchanged. For each outgoing hub edge to j, distribute its weight across replacements, excluding j to avoid a self-pass. For each incoming edge from i to the hub, distribute it across replacement receivers excluding i. Renormalize replacement weights for each excluded endpoint. The hub row/column and diagonal are zero; total team pass mass is preserved. This assumes replacements inherit the hub’s connections; it is a model to test, not observed behavior. Minute weights are NOT also multiplied into the nonhub edges, which would double-count replacement activity.

The forecast is a relative passing-pattern counterfactual, not a forecast of game pace or total pass volume. Node areas and edge widths normalize within context. Player layouts can use the displayed union for readability, but layout does not enter any prediction or metric.

## Entropy and regression

Outgoing conditional entropy H(receiver|passer), weighted by passes, is exported in a fixed within-team normalized form using log2(N_union_active−1), where union uses training-present and observed-later-absence players for evaluation. All three panels share the denominator. Original raw matrices remain available.

Primary Y = actual later-absence entropy − first-20 hub-present entropy, after 500 equal-total-pass-count subsamples of the two observed integer-count matrices. X = first-20 passing-only EC concentration. At least15 training hub-present and10 later hub-absent games are required for the fit. Low-exposure teams remain illustrated but never enter that line. Raw-count results are a sensitivity.

An early/late comparison can mix absence effects with time drift. Therefore a separate sensitivity compares later hub-absent versus later hub-present games, requiring ≥15 later hub-present games as well. That reference uses later data only as an outcome comparator, never to select the hub or construct the prediction. No opponent/lineup controls are imposed; neither difference is a causal effect.

OLS/HC3 t uncertainty, conventional Pearson p and Holm across the four declared analyses are exported. This follows extensive earlier exploration and remains exploratory. Subsampling ranges are not confidence intervals for causal behavior.

## Coverage and validation

{
  "teams": 30,
  "observed_later_absence_pairs": 28,
  "primary_regression_teams": 18,
  "later_present_sensitivity_teams": 18,
  "median_predicted_edge_distance": 0.5255547081734979,
  "median_deleted_edge_distance": 0.5579388138991443,
  "redistribution_improves_teams": 13,
  "primary_mean_actual_entropy_change": 0.04719322340822493
}

A trade-safe evaluation window stops14 days before the first recorded other-team appearance when applicable; this retrospective cohort guard is not used to forecast a trade. Empty tracking profiles require independent zero confirmation. All frozen prediction hashes are retained. No observed absence is represented as unavailable, not entropy zero.

## Limitations

First20 is a small training sample; role order is a proxy, not an official NBA depth chart. Candidate availability and teammate roles can change after game20. The prediction stays fixed through the later season, making long absences and roster changes difficult. Missing position information is explicit. Temporal drift, co-absence, opponent strength and finite counts remain. Higher entropy means greater recipient variety, not better offense.

## Interpretation of this run
28 teams have an observed later-absence comparison; 18 meet the declared exposure requirements. Primary equal-pass-count regression: r=0.404, R2=0.163, HC3 p=0.102 (Holm p=0.188). The slope is +0.0118 normalized entropy per +0.01 EC index. This is a positive but uncertain association, not evidence of a causal effect.

Using later hub-present games as the reference gives r=0.496, R2=0.246, HC3 p=0.0184; Holm-corrected p=0.0563 across the four reported tests. This sensitivity reduces broad early/late drift but does not control opponents, other injuries, or roster changes. The primary early-reference result remains the primary result.

The replacement model reduces edge-share total-variation error for 13/18 eligible teams. Median error is 0.526 versus 0.558 for simply deleting the hub. Error is still large: 0 means identical edge shares and 1 disjoint edge shares. The model is a modest improvement over deletion, not a validated realistic replacement mechanism. Available-minute capacity and missing position labels can give excessive weight to minor rotation players; early absence role estimates are sparse.

Largest primary entropy increases among eligible teams: LAL +0.1171, CHI +0.0956, OKC +0.0782. BKN is the only eligible decline (-0.0096). These are aggregate passing-pattern differences; they do not show better offense. Raw entropy in bits is exported alongside normalized entropy.
