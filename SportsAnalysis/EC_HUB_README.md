Run: python teamgraphs.py --ec-hub-study

# EC-selected passing hubs: all six seasons

Hub = player with the largest weighted eigenvector centrality on A+A.T using the first20 regular-season team games. This is the same passing-only symmetric EC definition used for team EC concentration. No shooting edges enter. Normalizing individual EC scores does not change their ranking.

First20 EC selects the same person as passes-in-plus-out for168/180 team-seasons;12 selections change. Original networks use early hub-present games. Predictions use only earlier minutes, positions and passing-role evidence. Actual networks use later hub-absent games within the valid evaluation window. Changed hubs require rebuilt played/missed partitions; they are not relabeled old networks.

The separate sending/receiving replacement model and the minutes-only benchmark are both retained. Positions, prior strengths, exclusion rules and eligibility thresholds were not tuned against these new outcomes. There are170 observed comparisons,97 primary-eligible team-seasons, and91 later-present sensitivity comparisons. The team EC index itself is unchanged because the first20 all-player network is unchanged.

Circle areas show passing involvement in each graph, not individual EC. An EC-selected hub need not have the largest circle or be the leading scorer. These are passing-network hubs, not validated measures of overall basketball star quality.

Main pooled result: within-season r=0.3515, R2=12.35%, franchise-clustered p=0.001475, Holm p=0.002950. Later-present baseline: r=0.1893, R2=3.59%, clustered/Holm p=0.03355. These are exploratory associations with repeated franchises accounted for, not causal effects.

Network graphs: passing_roles/all_180_split_role_networks.pdf. Observed regressions: ec_baseline/all_seasons_regressions.pdf. New replacement benchmark: passing_roles/passing_role_model_comparison.pdf. Full selection list: hub_selection_comparison.csv.

## Changed selections

|Season|Team|Previous volume hub|New EC hub|
|---|---|---|---|
|2021-22|GSW|Stephen Curry|Draymond Green|
|2021-22|IND|T.J. McConnell|Domantas Sabonis|
|2022-23|BOS|Jayson Tatum|Marcus Smart|
|2022-23|CHI|DeMar DeRozan|Nikola Vučević|
|2022-23|LAL|Russell Westbrook|Anthony Davis|
|2022-23|MIL|Giannis Antetokounmpo|Jrue Holiday|
|2022-23|TOR|Scottie Barnes|Fred VanVleet|
|2023-24|CLE|Darius Garland|Evan Mobley|
|2023-24|POR|Shaedon Sharpe|Malcolm Brogdon|
|2024-25|ORL|Jalen Suggs|Franz Wagner|
|2025-26|CHA|Collin Sexton|Miles Bridges|
|2025-26|HOU|Amen Thompson|Alperen Sengun|