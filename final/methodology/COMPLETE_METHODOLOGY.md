# Final methodology: previous-season-only historical role prediction

## Research questions

How does passing structure differ between early games with a passing hub and later games without that hub? Can an early-data replacement forecast, adjusted using other franchises in the immediately previous season, predict the later hub-absent network? Does greater early hub involvement accompany larger observed entropy change?

The prediction and association questions are separate: the entropy-centrality regressions use observed networks, not predicted entropy.

## Data and time separation

Thirty teams across 2020-21 through2025-26 produce180 team-seasons. NBA tracking aggregates provide directed player-to-player pass counts; official appearances define hub-present and hub-absent games. Matrices are whole-game aggregates, including bench time when the hub appeared. They do not isolate passes made while that hub was physically on court, or reconstruct complete possession sequences. No minimum player-minute cutoff is imposed. Self-passes and shooting/basket edges are excluded; displayed isolates are omitted.

For each target team, the first20 team games define all target training inputs. Later passes/minutes do not construct its prediction. The immediately previous season supplies historical network outcomes and first20 features from other franchises only. The target franchise is excluded; no older network season substitutes for a missing adjacent season. Historical feature records retain their previously frozen prior-position metadata. Names/IDs align matrices but are not learned features. Players may recur on other franchises; this is not leave-player-out testing.

The first saved season has no2019-20 network training and retains the initial separate-role forecast. A retrospective trade safeguard and tracking exclusions are inherited from the saved study. Missing observations are undefined, not zero passes or zero entropy. Empty tracking requires independent zero confirmation; availability audits cannot prove the feeds recorded every pass.

## Hub and observed matrices

Let A20 contain all first20 passes and P contain passes during first20 games where the hub appeared. E=A20-P is the early hub-absent matrix. For player i, strength is c_i=sum_j A20_ij+sum_j A20_ji. The hub h maximizes c_i, breaking ties by lowest ID. Hub involvement C_h=c_h/sum_ij A20_ij is the fraction of team passes involving that player. Tracking TOUCHES, EC-selected players and on/off ratings are not substituted for this definition.

T is the actual matrix from later tracked hub-absent games. Multiple absence spells, lineups and opponents are pooled. Original/actual matrices are unchanged by model choice.

## Replacement minutes

Let m_i be early hub-present minute totals. Baseline b_i=240*m_i/sum_j m_j. Remove the hub. Nonhub candidates require positive early hub-present minutes. Available capacity is max(0,48-b_i).

Fallback score is capacity*position_match/depth_rank, normalized across candidates. Prior-season broad G/F/C labels give match1 for overlap,0.5 when either is unknown,0.25 for known nonoverlap. Matching candidates rank by baseline minutes; other ranks are1. This is inferred rotation depth, not an official historical depth chart.

If early hub-absent games exist, g_i=max(0,early_absent_MPG_i-early_present_MPG_i). Normalize positive gains and blend with fallback weights using alpha=n_early_absent/(n_early_absent+5). Without positive gain evidence alpha=0. Allocate removed hub minutes using these weights, cap each candidate at48, and redistribute excess among uncapped candidates. Forecast f_h=0, sum_i f_i=240. This is a regulation-length scenario, not an overtime or game-specific availability forecast.

## Separate sending and receiving roles

For outgoing and incoming counts separately, estimate r_i=(c_i+48*r_peer)/(m_i+48), where r_peer is eligible nonhub peers' pooled pass count divided by pooled minutes. Initial role scores are f_i*r_i, normalized separately. Total forecast minutes allow existing starters to inherit responsibility without large increases in playing time.

For early absence evidence, r_abs_i=(c_abs_i+48*r_i)/(m_abs_i+48). Positive gains f_i*max(r_abs_i-r_i,0), normalized, are blended with initial role shares using n_early_absent/(n_early_absent+5), separately for sending and receiving. Without positive evidence retain initial shares. Forecast minutes provide an activity-weight fallback when necessary.

## Initial predicted network M

Keep nonhub-to-nonhub edges unchanged initially. Remove hub edges. Redistribute each h-to-j edge among replacement senders, excluding j; redistribute each i-to-h edge among replacement receivers, excluding i. Renormalize each route to prevent self-passes. If route weights vanish, use forecast-minute weights excluding that endpoint.

For distinct nonhub i,j, M_ij=P_ij+P_hj*s_i/(1-s_j)+P_ih*r_j/(1-r_i), with sender shares s and receiver shares r and the stated fallbacks. M has zero diagonal/hub row/hub column and sum(M)=sum(P). Counts may be fractional expected weights. This predicts structure at the original volume, not future pace.

## Previous-season historical correction

For every target, historical training uses only original-primary-eligible comparisons (at least15 early present and10 later absent games) from the immediately previous season, excluding the target franchise. The latest user filter changes evaluation membership, not these frozen historical training criteria.

For each active historical candidate, calculate initial sender/receiver shares p_i and actual later-absence shares a_i. Actual denominators include all observed later passes, including later players not in the early candidate set. Fit targets y_i=ln[(a_i+epsilon)/(p_i+epsilon)], separately for sending and receiving, with epsilon=0.05/n and n active forecast candidates.

Thirteen features are: ln(max(n*predicted_sender_share,1e-5)), the analogous receiver feature, n*original_sender_share, n*original_receiver_share, forecast_minutes/48, extra_minutes/48, G/F/C/unknown indicators, position match, ln(n), and hub involvement. Feature means and standard deviations are fitted using permitted historical rows only; standardized denominators have floor0.01.

Two weighted ridge fits minimize sum_i w_i*(y_i-beta0-x_i*beta)^2+10*sum_k beta_k^2; the intercept is unpenalized. Each historical team-season has total weight1, with player weight1/n. Ridge10, smoothing0.05 and multiplier caps were fixed before this benchmark, not optimized against target outcomes.

For target features, log corrections are clipped to [-ln4,ln4], producing sender factors a_i and receiver factors b_j between0.25 and4 before normalization. Set Btilde_ij=M_ij*a_i*b_j, then B_ij=Btilde_ij*sum(M)/sum(Btilde). Existing connections can strengthen/weaken, but zero initial edges and later-arriving players cannot be added. The factors do not enforce independent exact predicted marginals, and realized share changes are not bounded by the raw factor caps after joint normalization. Minutes are unchanged by this historical layer.

## Prediction errors and current evaluation sample

Evaluate targets with at least10 analyzed/tracked later hub-absent games and a nonempty observed absence matrix. This gives100 cases;85 have previous-season training. The15 earliest-season eligible cases use the initial fallback. No15-early-present threshold is imposed on these final evaluation summaries.

Connection error is TV=0.5*sum_ij abs(B_ij/sum(B)-T_ij/sum(T)), ranging from0 for identical edge-share distributions to1 for disjoint distributions. Entropy error is abs(Hstar(B)-Hstar(T)). Export means, medians and paired counts on identical samples; compare initial/older-minute/deletion baselines and separate historical-copy baselines only on their available common observations.

The optional historical copy deliberately uses the same franchise's previous same-hub absence network, when its prior comparison meets the historical exposure criterion. It is mapped to the active first20 forecast roster and renormalized; unseen edges receive zero. Retained historical pass mass is exported. Unavailable copies are missing, not silent current-model fallbacks. These copies are benchmarks only, not inputs to the learned fit.

## Conditional passing entropy

For a matrix X, p_ij=X_ij/sum_j X_ij and H(X)=-sum_i [sum_j X_ij/sum_ij X_ij]*sum_j p_ij*log2(p_ij). Zero terms contribute0, and zero-outgoing players have zero weight. This is receiver diversity conditional on sender, not equal player involvement or offensive quality.

Normalize Hstar=H/log2(Nref-1), where Nref is the active union of original and actual absent players. Original/predicted/actual panels share the same team-season reference. Later roster information thus enters the retrospective evaluation reference, not forecast edges. Raw bit entropy is also in the source metadata. Empty observed networks have undefined entropy.

## Observed entropy-change regressions

Regressions are regenerated using the final10+ absence filter (100 cases), rather than reusing the former96-case results. Primary Y is the average of500 without-replacement equal-pass-count differences Hstar(T_sample)-Hstar(P_sample), using K=min(totalP,totalT) passes in each draw. This reduces unequal-count exposure, but does not match games, lineups or opponents. Draws are not independent team observations or causal confidence intervals.

Primary X is first20 hub involvement C_h. Raw hub pass totals are a sensitivity. A later-present reference is another sensitivity, requiring at least15 later present games and a validated entropy reference (91 cases in this final package). First20 passing-only EC concentration is an alternative predictor: calculate eigenvector shares summing to1 on A20+A20.T among active players, then EC_index=(sum_i p_i^2-1/N)/(1-1/N). EC does not select the hub or enter the historical prediction features.

Pooled OLS includes season indicators and franchise-clustered sandwich uncertainty, with finite-sample correction and t(G-1) inference. Reported pooled r/R2 use within-season-demeaned variables, not total fit including season effects. Seasonal fits use HC3 uncertainty. Holm corrections use four pooled involvement specifications and24 seasonal tests as separate families; EC has two pooled and12 seasonal tests. These corrections do not erase the project's broader exploratory history.

## Presentation and limitations

Three-panel graphs show original, final predicted and actual networks. Node areas use each panel's own (incoming+outgoing)/(2*team passes), and isolates are omitted. Fixed Fruchterman-Reingold positions use the displayed union for readability only; layout is not a model input. All180 illustrations are retained; analysis and default galleries focus on the100 ten-plus cases.

Prediction matrices and settings are frozen before target-season evaluation reads. Previous-season outcomes are legitimate historical labels for subsequent targets. However, these seasons have been repeatedly examined during development; this is retrospective testing rather than fresh independent validation. Replacement assumptions, fixed roles, incomplete availability, opponents, other injuries, roster change, shared network inputs and mechanical hub removal remain limitations. Similar entropy can conceal different networks. More entropy does not mean better offense. The model does not establish causal star impact.
