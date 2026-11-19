# Separate minutes and passing roles: six-season benchmark

All 180 forecasts retain the previously frozen minutes allocation, hub identity and EC index. Extra minutes still use early rotations, previous-season broad position labels and earlier replacement evidence. Passing responsibility is now separate from those extra-minute shares.

For each nonhub player with positive early hub-present minutes and forecast minutes, estimate outgoing passes per minute and incoming passes per minute separately. Shrink each rate toward its nonhub team peer rate using 48 prior minutes. Forecast total minutes × each shrunk rate gives separate sender and receiver weights. Total forecast minutes, rather than extra minutes, allow an existing starter to inherit playmaking without receiving much additional court time.

If early hub-absent games exist, shrink each player’s absence rate toward their own shrunk present rate using 48 prior minutes. Positive rate gains × forecast minutes form additional role evidence. Blend with base role weights using n_early_abs/(n_early_abs+5), separately for sending and receiving. If there is no positive gain evidence, use base weights. The 48-minute and five-game constants were fixed before this benchmark and were not optimized against its errors.

Keep nonhub-to-nonhub edges fixed. Distribute each hub-to-j edge among replacement senders using sender weights, excluding j. Distribute each i-to-hub edge among replacement receivers using receiver weights, excluding i. Renormalize each route. Pass mass is preserved and the hub row, column and diagonal are zero.

Touches and potential assists are NOT used: complete first-20 tracking for those measures is not saved across all six seasons. Passes per minute measure recorded passing activity, not the quality of playmaking. Position is not a hard restriction on the passing-role step.

All new matrices and input hashes are frozen before evaluation matrices are read. Original and actual networks are unchanged. The prior 96-team-season exposure sample is the primary benchmark; all 171 observed comparisons are also exported. Missing-data exclusions, entropy denominators and candidates are inherited. Original observed EC–entropy regressions therefore remain unchanged.

This is retrospective model development using previously examined absence outcomes, not fresh external validation. A later untouched season is needed to validate a final chosen rule. No weights were tuned to maximize improvement. Low-minute rates, static minutes, later injuries and roster moves remain limitations. Early-only replacement candidates are unchanged to isolate the passing-role change.

Primary benchmark: median entropy error changes from 0.0433 to 0.0289, with 62/96 improving. Median edge-share error changes from 0.4944 to 0.4981, with 34/96 improving and 62/96 worsening. This version is closer on entropy but is not an overall improvement in predicting who passes to whom.

{
  "sample": "eligible",
  "season": "ALL",
  "n": 96,
  "median_edge_old": 0.4944261429779898,
  "median_edge_new": 0.49813221622490705,
  "median_paired_edge_change": 0.005945028346271419,
  "teams_edge_improved": 34,
  "teams_edge_worsened": 62,
  "median_entropy_old": 0.04330406365657308,
  "median_entropy_new": 0.02889050956620015,
  "median_paired_entropy_change": -0.02218874417556055,
  "teams_entropy_improved": 62
}

Negative new-minus-old error means improvement. Total-variation error compares normalized edge shares, with 0 identical and 1 disjoint. Entropy error is absolute difference under the same inherited within-team reference; different networks can have similar entropy, so both errors matter.