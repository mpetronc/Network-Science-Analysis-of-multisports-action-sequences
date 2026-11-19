# All-season first-20 passing-hub study

{
  "excluded_absence_games": 2,
  "unavailable_later_present_references": 3,
  "seasons": [
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
    "2025-26"
  ],
  "team_seasons": 180,
  "eligible_team_seasons": 96,
  "observed_pairs": 171,
  "pooled": [
    {
      "analysis": "primary_equal_count",
      "n": 96,
      "seasons": 6,
      "franchises": 30,
      "season_adjusted_r": 0.3618455449059748,
      "within_season_r_squared": 0.13093219836830183,
      "slope": 1.187590890064913,
      "slope_per_001": 0.011875908900649131,
      "franchise_clustered_p": 0.0013306535005050146,
      "ci_low": 0.5037065637796443,
      "ci_high": 1.8714752163501818
    },
    {
      "analysis": "late_present_equal_count",
      "n": 88,
      "seasons": 6,
      "franchises": 30,
      "season_adjusted_r": 0.20775787001502602,
      "within_season_r_squared": 0.043163332553180445,
      "slope": 0.559471468579852,
      "slope_per_001": 0.00559471468579852,
      "franchise_clustered_p": 0.018131446910898665,
      "ci_low": 0.10261334341402573,
      "ci_high": 1.0163295937456782
    }
  ]
}

Same study as the 2025–26 run: hub=max passes sent+received in first20; passing-only EC normalized HHI on A+A.T; predictions use only training rotations and previous-season NBA position metadata. Outgoing conditional entropy H(receiver|passer), weighted by each player’s outgoing pass share, with fixed within-team normalization. The reference denominator log2(N−1) uses the active union of training-present and observed absent players for retrospective evaluation. It does not alter frozen forecast edges; raw entropy in bits is also exported. The normalized forecast entropy therefore uses an evaluation reference, not a roster size known in advance. No minutes cutoff. Each panel sizes nodes by its own passing involvement and omits isolates.

2020–21 had 72 scheduled games per team, other saved seasons 82. All use first20 training and remaining observed eligible games, with retrospective trade censor as previously documented. At least15 early hub-present and10 later absent games for regression. No observed absence is unavailable, never zero entropy. Unresolved empty tracking windows are excluded as whole absence windows, not individual rows filled with zero; official and analyzed absence counts and dates are recorded. When exclusions prevent validating subtraction from full-season totals, the later-present reference is unavailable (NaN), and that team is omitted from that sensitivity fit. See tracking_exclusions.json and evaluation_tracking_scope.json in each affected season. No observed later-present games yields a known empty matrix and undefined reference entropy, rather than a tracking-data gap.

Primary outcome remains later absence minus first20 hub-present, equal-pass-count resampling. Later-present reference is sensitivity. Four fits per season, with Holm across all reported season tests in addition to within-season corrections. Thresholds and model choices are carried over from exploratory prior analysis.

Pooled fits include season indicators and franchise-clustered sandwich uncertainty with small-sample correction and t(G−1). Repeated franchise seasons are not treated as fully independent. Holm adjustment is also exported across the two pooled tests. The reported season-adjusted r and within-season R2 use season-demeaned x and y. They are not the R2 of a model including season indicators.

Limitations remain: first20 identity instability; missing broad position labels; arbitrary unvalidated replacement-role weights; roster mixing; opponent/co-absence confounding; shared network inputs and mechanical node-removal effects. More entropy is not evidence of improved performance. Neither aggregate differences nor regressions identify causal star dependency.

The 500 pass-count subsamples equalize count exposure; they are not 500 independent team observations or game-level uncertainty intervals. An additional descriptive diagnostic compares actual changes with simple deletion, and actual absent entropy minus deleted-network entropy. It is not a causal adjustment or a model of tactical adaptation. Complete retrieval audits do not guarantee that tracking recorded every pass in every game. profile_tracking_coverage.csv compares the passing-profile G field with official appearances: a lower G can reflect zero-activity games or incomplete tracking and is a flag, not proof of missing passes. Aggregate profiles cannot distinguish those explanations. Only official team participants within each query date window can be teammate recipients. Any discarded foreign or out-of-window recipient edges are listed in excluded_invalid_recipient_edges.csv, while raw profiles are retained. Full raw profiles, source snapshots, replacement allocations, frozen specifications, matrices, coverage and season-level reports are in each season folder. Previously validated 2025–26 results are reused unchanged; earlier seasons require independent date-filtered retrieval. Missing downloads are not treated as zero or replaced with full-season matrices.