# Team-by-team entropy changes
All 180 restored volume-selected hub team-seasons are shown, including those outside the 96-observation primary regression. No networks, hubs, entropy estimators, or regression results were changed. No NBA downloads were made.

Six league pages show the exact equal-pass-count actual-minus-early-present entropy changes used in the regression. Grey bars mark comparisons outside the primary sample; missing values have no bar. These exploratory bars have no confidence intervals and do not establish individual-team statistical significance.

Thirty franchise pages show six season panels each: early hub-present network, the latest split-role replacement forecast, and later actual hub-absent network. Full-network entropy levels differ from pass-count-matched regression differences. Predicted entropy is not used as an outcome in the centrality regression. A common within-team-season denominator is inherited; denominators and pooled rosters can differ across seasons. No direct year-to-year performance trend is inferred.

Higher entropy means a more diverse receiver distribution conditional on sender, averaged over recorded passes. It does not prove better offense, more equal player involvement, or less reliance on stars. Distinct passing patterns can have the same entropy. Removing a hub can mechanically change entropy; later roster mixing, opponents, co-absences and temporal drift may also matter.

The regression is positive and modest, with exploratory pooled season-adjusted, franchise-clustered inference inherited from the prior report. Leave-one-franchise-out correlations are descriptive robustness checks, not fresh confirmatory tests. Multiple exploratory analyses have already been considered.

{
  "eligible_n": 96,
  "positive_changes": 90,
  "negative_changes": 6,
  "median_change": 0.05423225630333943,
  "mean_change": 0.05452686487426517,
  "leave_one_franchise_r_min": 0.3420580657767239,
  "leave_one_franchise_r_max": 0.4105051747946131,
  "prediction_median_abs_entropy_error": 0.028890509566200206,
  "prediction_overestimates_count": 32,
  "prediction_underestimates_count": 64,
  "largest_positive": [
    {
      "season": "2022-23",
      "team": "DAL",
      "hub_name": "Luka Don\u010di\u0107",
      "actual_equal_count_change": 0.16392953623721837
    },
    {
      "season": "2021-22",
      "team": "POR",
      "hub_name": "Damian Lillard",
      "actual_equal_count_change": 0.15096125180029218
    },
    {
      "season": "2024-25",
      "team": "CHA",
      "hub_name": "LaMelo Ball",
      "actual_equal_count_change": 0.14632685921602054
    },
    {
      "season": "2020-21",
      "team": "OKC",
      "hub_name": "Shai Gilgeous-Alexander",
      "actual_equal_count_change": 0.1235962029693477
    },
    {
      "season": "2021-22",
      "team": "ORL",
      "hub_name": "Cole Anthony",
      "actual_equal_count_change": 0.12314152332494581
    }
  ],
  "largest_negative": [
    {
      "season": "2020-21",
      "team": "WAS",
      "hub_name": "Bradley Beal",
      "actual_equal_count_change": -0.09285395030645303
    },
    {
      "season": "2020-21",
      "team": "MIN",
      "hub_name": "D'Angelo Russell",
      "actual_equal_count_change": -0.08082814041149029
    },
    {
      "season": "2023-24",
      "team": "NOP",
      "hub_name": "Brandon Ingram",
      "actual_equal_count_change": -0.049566701250260435
    },
    {
      "season": "2023-24",
      "team": "MIA",
      "hub_name": "Kyle Lowry",
      "actual_equal_count_change": -0.019665059811054847
    },
    {
      "season": "2025-26",
      "team": "BKN",
      "hub_name": "Nic Claxton",
      "actual_equal_count_change": -0.009637574897711613
    }
  ]
}