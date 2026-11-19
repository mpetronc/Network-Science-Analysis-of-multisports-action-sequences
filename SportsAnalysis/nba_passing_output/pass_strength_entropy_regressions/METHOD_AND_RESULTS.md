# Entropy change versus passes-in-plus-out centrality

Uses the restored pre-EC player-selection study: first-20 passes-sent-plus-received leader. No tracking TOUCHES, no EC-selected hubs, and no replacement-model prediction errors are substituted for the outcome.

Primary X=(hub passes sent + hub passes received)/(all first20 team passes). It is the fraction of recorded team passes involving the hub. With self-passes excluded, a given hub can occur only once in a pass, so no double counting occurs in its numerator. The alternative node-strength share divides by twice team passes; that rescales X by one-half and yields identical r, R2 and p. Raw first20 passes-in-plus-out is a declared sensitivity.

Primary Y=observed later hub-absent entropy minus early hub-present entropy, using the existing500 equal-pass-count subsamples. Positive means higher outgoing conditional entropy without the hub. The later hub-present reference is a sensitivity. The180-team-season source has96 primary-eligible observations and88 later-reference eligible observations, with the same earlier15-present/10-absent criteria and tracking/trade exclusions. Actual matrices, selected hubs and replacement forecasts are unchanged.

Six per-season HC3 fits for each of four declared analyses are exported (24 tests, Holm correction across24). Four pooled fits include season indicators and franchise-clustered sandwich uncertainty with finite-sample correction and t(G-1) inference. Holm correction across the four pooled tests is reported separately. Pooled r and R2 use within-season-demeaned variables; they are not total model R2 including season effects.

This remains exploratory after extensive previous analyses. Hub involvement is cumulative activity over the first20 team games and reflects availability. Raw totals additionally reflect passing volume/pace; normalized involvement reduces that scale difference. X and the original-network entropy share inputs, and node removal mechanically changes entropy. Temporal drift, roster mixing, co-absence and opponents remain. These are structural associations, not causal player impact, offensive efficiency or prediction accuracy.

n,seasons,franchises,within_season_r,within_season_r_squared,slope,clustered_p,slope_ci_low,slope_ci_high,analysis,x,y,effect_per_10_percentage_points,effect_per_1000_passes,holm_four_pooled_tests
96,6,30,0.3827058050326908,0.14646373320571995,0.25742728489953687,0.00017877691581699067,0.13483603468613659,0.38001853511293715,normalized_early,hub_pass_involvement_fraction,actual_equal_count_change,0.025742728489953688,,0.0007151076632679627
88,6,30,0.3680580425003627,0.13546672264919882,0.20324937231131535,0.0013325670326386188,0.08618854566362724,0.3203101989590035,normalized_later,hub_pass_involvement_fraction,concurrent_equal_count_change,0.020324937231131535,,0.0026651340652772376
96,6,30,0.29796583257604803,0.08878363738273748,3.318629294825776e-05,0.0003075246966253895,1.6618251030352964e-05,4.975433486616255e-05,raw_early,hub_passes_in_plus_out,actual_equal_count_change,,0.033186292948257756,0.0009225740898761686
88,6,30,0.2913229743840825,0.08486907540398879,2.6364452483785643e-05,0.006268512524176641,8.069574382622157e-06,4.4659330584949125e-05,raw_later,hub_passes_in_plus_out,concurrent_equal_count_change,,0.026364452483785644,0.006268512524176641
