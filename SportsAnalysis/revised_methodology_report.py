"""Generate readable methodology and results from the saved revised analysis."""
from pathlib import Path
import pandas as pd,numpy as np
NAMES={'passing_entropy_excess':'Activity-adjusted passing diversity','shooting_entropy_role_excess':'Position-adjusted shooting-zone diversity','passing_ec_excess':'Activity-adjusted passing EC concentration','shooting_hhi_unbiased':'Shooting concentration (FGA HHI)'}

def write_report(out):
 out=Path(out);d=pd.read_csv(out/'revised_metrics.csv');r=pd.read_csv(out/'pooled_regressions.csv');ctx=pd.read_csv(out/'opportunity_conditioned_sensitivity.csv');temporal=pd.read_csv(out/'temporal_checks.csv');op=pd.read_csv(out/'temporal_opportunity_check.csv').iloc[0];absence=pd.read_csv(out/'absence_checks.csv');groups=pd.read_csv(out/'leader_interactions.csv');influence=pd.read_csv(out/'leave_one_franchise_out_association.csv');players=pd.read_csv(out/'position_shooting_reference.csv')
 methodology='''# Revised methodology: offensive organization and reference comparisons

## Data and scope

180 whole regular-season team networks: 30 NBA teams in each season 2020–21 through 2025–26. Original full rosters and directed pass counts retained, including traded participants with their team-filtered counts. No minutes cutoff. Shooting inputs are player attempts across 14 official NBA half-court zones plus backcourt/heaves; every attempt includes makes and misses. Exact shot counts and original passing inputs were previously verified, and current source hashes are checked against the stored null baseline. Position labels are archived team-season PG, SG, SF, PF or C. They are broad nominal positions, not measured lineup roles. New files are separate from previous analyses.

## Four primary measures

### 1. Activity-adjusted passing diversity

First calculate pass-weighted conditional entropy H(receiver | passer), normalized by log2(Npassing−1). Then subtract its mean across 500 independent reference networks for that team-season. Outgoing pass totals are fixed exactly in every simulation; incoming totals match in expectation, not exactly per draw. No self-passes. Expected recipient probabilities are fitted by zero-diagonal iterative proportional fitting to observed outgoing/incoming totals. Every retained original null draw was checked for correct row sums and zero diagonal.

New score = observed normalized passing entropy − mean reference normalized entropy. Negative means more restricted pair choices than this activity-only mixing model. A higher score is closer to activity-based mixing; it does NOT necessarily mean higher raw passing entropy. Matching counts in simulations addresses finite-count entropy differences relative to this reference. It does not remove lineup, shared-minute or schedule effects.

### 2. Position-adjusted shooting-zone diversity

For each shooter, calculate entropy across the 15 court zones divided by log2(15). Build position-specific shot-zone reference distributions using ONLY the first five seasons (2020–21 through 2024–25). Exclude each focal player's entire development-period history from both their role reference and its league smoothing prior. Thus a player's own shot choices do not set their peer comparison. Smooth the role counts with 30 effective attempts distributed by a leave-player-out league prior; the global prior has one pseudocount per zone. These fixed choices are not tuned for regression strength.

For each player-team-season simulate 500 independent zone-count samples with exactly their observed number of attempts. Subtract expected sample entropy from observed entropy, then average player differences weighted by FGA. The result is position-relative zone diversity, not shot efficiency. Count-matched simulation handles sample-size effects under this specified model. Five nominal positions do not capture all differences in player skill, tactical roles or lineup assignments. Reference-estimation uncertainty is not fully propagated by the regression confidence bands.

### 3. Activity-adjusted passing concentration

EC is calculated on the original player-to-player passing matrix A+Aᵀ only. Shooting nodes/weights never enter. Convert EC values into L1 shares and calculate normalized HHI. Subtract mean EC concentration from the same 500 passing-reference networks. A positive score means passing prominence is more concentrated than expected from this activity baseline. A negative score means less concentrated. This is a signed reference difference, not a 0–1 star probability. Original EC remains available in the data.

### 4. Shooting concentration

Use player FGA shares, without free throws or turnovers. Primary measure is the pair-based finite-count HHI: sum_i FGA_i(FGA_i−1) / [team FGA(team FGA−1)]. It is the fraction of pairs of distinct recorded attempts attributable to the same shooter. Under a multinomial sampling interpretation it corrects the upward finite-count bias of ordinary sample HHI. Real basketball attempts need not be independent; this is not a general correction for playing-time or injury effects. Ordinary HHI, shooter-count-normalized HHI, top-one share and top-two share are retained as sensitivities. Ordinary HHI and its effective-shooter count are also retained. Using raw HHI avoids making primary results depend on a chosen roster-count normalization.

## Models and inference

All six unordered pairs of these four measures are tested, with the passing-diversity/FGA-concentration comparison designated as the focal relationship before reading the revised fit. Primary models have no minutes or roster regression controls. Pooled inference uses franchise-clustered standard errors (30 franchises), with t reference df=29. Holm correction covers the six primary tests. Separate, explicitly labeled families cover leader-group fits, interaction tests, temporal association tests, per-season focal tests, opportunity sensitivities and absence tests. Sensitivity rows are not selected to maximize significance. These corrections cover the stated new test families, not all historical questions explored in this project; confirmatory evidence still needs genuinely new data.

Shared leader means the SAME player leads passing EC and FGA totals. This replaces comparisons based on the earlier FGA+0.44FTA+TOV workload leader. Group lines are descriptive; the formal interaction tests determine whether slopes demonstrably differ.

## Retrospective temporal check

Fit coefficients on 150 team-seasons from the first five seasons and apply them without refitting to the 30 team-seasons from 2025–26. Shooting-position reference counts also exclude the latest season. Report frozen-model prediction error and R² separately from the correlation and HC3 slope test fitted within the 30 latest-season observations. R² relative to the test mean and squared-error improvement relative to the development mean are both exported; they are not interchangeable.

The latest season was already explored in the project, so this is NOT an untouched holdout, external validation, or a prospective forecast. Predictors use season-wide observed activity and are concurrent structural measurements. Truly confirmatory testing requires a genuinely unseen season or independent data.

## Opportunity-conditioned sensitivity

As a secondary question, include roster player count and minutes concentration (HHI of all roster MIN shares), with and without season indicators. These are clearly separated from the requested primary regressions without controls. They ask whether an association remains beyond opportunity distribution. Playing opportunities may themselves be part of star-centered organization, so conditioning changes the estimand; it is not automatically the causally correct model. Compare a frozen opportunity-only model with a frozen opportunity-plus-entropy model on the latest season.

## Additional robustness checks

Repeat comparisons with the first and second halves of the Monte Carlo draws; use unnormalized entropy differences in bits; calculate exact KL divergence between observed pass-row probabilities and the fitted activity reference; use alternative FGA concentration summaries; remove each franchise in turn for influence checks; and fit the focal relationship separately by season. KL identity H(fitted mixing)−H(observed) is independently checked. These checks address numerical noise and definition choices, not all basketball confounders. Monte Carlo error is not uncertainty about a causal effect or about the suitability of the null model.

## Absence check improved as far as saved data allows

21 compatible teams from 2025–26. Rebuild entropy/EC predictors and their reference networks using ONLY hub-present games. This removes hub-absent games from the network predictor. Reuse existing adjusted offensive absence penalties; do not claim a fresh outcome model. Positive penalty means offense was better with the hub present. Saved adjustments include home court, opponent defense excluding the game, weighted absences of three other key teammates, game number and back-to-backs (varying controls only).

Hub identity and eligibility were selected retrospectively from season totals. Hub-present games can occur after an absence, and appearances do not equal on-court possessions. Other injuries are only partly covered; penalties have different uncertainty; 21 selected teams are not the entire league. These are observational checks, not causal validation or pre-injury predictions. No shooting-absence metric is claimed without matched shooting data for those windows.

## Remaining limits

We cannot reconstruct exact simultaneous-lineup passing or observed pass-to-shot sequences from season aggregate matrices. Old movement examples and lineup analyses do not provide these timestamps for all 180 current team-seasons. The reference permits pairs who may have limited shared-court exposure. Position labels are coarse; benchmarks are model choices. CIs condition on estimated references and do not include every layer of reference/position uncertainty. Metrics remain descriptions of organization, not validated star dependency, player quality or offensive effectiveness. No assertion that a stronger regression makes a measure better.
'''
 (out/'METHODOLOGY.md').write_text(methodology)
 summary='''# New results after methodological checks

All results are exploratory structural associations. The baseline adjustment changes the question and can change signs; this is not evidence that a causal effect reversed.

## Six revised pairwise associations (180 team-seasons)

| Pair | r | R² | Franchise-robust p | Holm p (six tests) |
|---|---:|---:|---:|---:|
'''
 for z in r.itertuples():summary+=f'| {NAMES[z.x]} vs {NAMES[z.outcome]} | {z.r:.3f} | {z.r_squared:.1%} | {z.robust_p:.3g} | {z.holm_p:.3g} |\n'
 x='passing_entropy_excess';y='shooting_hhi_unbiased';f=r[(r.x==x)&(r.outcome==y)].iloc[0];ct=ctx[(ctx.x==x)&(ctx.y==y)];test=temporal[(temporal.x==x)&(temporal.y==y)].iloc[0]
 raw_r=float(np.corrcoef(d.passing_entropy,d[y])[0,1]);summary+=f'''
## What changed in the main relationship?

Using the SAME new FGA-HHI outcome, raw passing entropy has r={raw_r:.3f}, whereas passing entropy relative to its activity baseline has r={f.r:.3f} (R²={f.r_squared:.1%}). Most adjusted entropy values are negative: higher means less restricted than the team-specific activity benchmark, not higher absolute entropy. Teams concentrating attempts often have passing closer to this benchmark, while lower-concentration teams show larger entropy shortfalls. This could reflect rotation continuity, roles, pair preferences or other opportunity structure; it is not a causal explanation. The earlier −0.653 correlation used shooter-count-normalized HHI, so it should not be compared to +0.644 as if only one definition changed. Both raw-HHI and normalized-HHI sensitivity results are supplied.

The focal association is positive in all six seasons. Removing each franchise leaves r between {influence.r.min():.3f} and {influence.r.max():.3f}. Separate Monte Carlo halves give nearly identical results. Thus the result is not simply one franchise or simulation noise.

## Strong total association, weak incremental evidence

| Secondary opportunity comparison | Partial r | Partial R² | Robust p | Holm p (12 sensitivity tests) |
|---|---:|---:|---:|---:|
'''
 for z in ct.itertuples():summary+=f'| Roster count + minutes HHI'+(' + season indicators' if z.season_FE else '')+f' | {z.partial_r:.3f} | {z.partial_r_squared:.1%} | {z.robust_p:.3g} | {z.holm_p:.3g} |\n'
 summary+=f'''
These sensitivities weaken the evidence that passing-diversity differences add beyond the distribution of playing opportunities. They do not prove playing time causes everything, and controlling opportunity variables changes the substantive question.

## Latest-season retrospective check

First-five-season entropy-only model: latest-season r={test.test_r:.3f}; frozen-model prediction R² relative to latest-season mean={test.predictive_r_squared_vs_test_mean:.1%}. Squared-error improvement relative to predicting the development mean={test.predictive_r_squared_vs_train_mean:.1%}. The two percentages have different denominators.

The opportunity-only model has RMSE {op.opportunity_only_baseline_rmse:.5f}; opportunity plus entropy has RMSE {op.test_rmse:.5f}. Adding entropy changes the baseline's squared error by {-op.incremental_R2_vs_opportunity_only:+.1%} (worse in this check). The additional entropy feature therefore does not show incremental predictive value here, despite the strong entropy-only correlation. This is retrospective, not an untouched validation or season-ahead forecast.

## Shooting diversity and leader groups

Position-adjusted shooting-zone entropy has little pooled association with the other three revised metrics; none of its primary comparisons survives correction. Shared passing/FGA leaders: {int(d.leader_alignment.eq('Shared leader').sum())}; separate leaders: {int(d.leader_alignment.eq('Separate leaders').sum())}. None of the six slope-interaction tests survives Holm correction. Visible group-line differences are not a confirmed distinction.

## Hub-present-only absence results

| Predictor | n | r with adjusted offensive penalty | Robust p | Holm p (four tests) |
|---|---:|---:|---:|---:|
'''
 for z in absence.itertuples():summary+=f'| {z.x.replace("_"," ")} | {z.n} | {z.r:.3f} | {z.robust_p:.3g} | {z.holm_p:.3g} |\n'
 summary+='''
No passing measure clearly validates causal star dependency in this selected sample. Predictor contamination from absent games has been reduced, but retrospective hub selection, concurrent measurement and limited sample/availability controls remain.

## What the revised methodology supports

Describe passing preferences and shooting-zone diversity relative to explicit activity/position baselines. Keep passing concentration and shooting concentration separate. Present both the primary total associations and opportunity-conditioned checks. The data support an organization/playing-opportunity study; current evidence does not support branding these scores as validated star dependency or claiming an entropy feature improves prediction beyond minutes and roster structure.

Files: new_data.csv is the clean 180-row dataset; revised_metrics.csv includes all original/reference values and diagnostics; revised_analysis.pdf shows all six pairs, retrospective temporal checks and leader groups; primary_comparison.pdf isolates the central baseline/opportunity issue; absence_present_only.pdf shows the limited absence check. Source hashes, reference counts, fixed settings, runtime versions and full sensitivity tables are included.
'''
 (out/'RESULTS.md').write_text(summary)
 (out/'README.md').write_text('Revised offensive-organization analysis. Start with RESULTS.md and METHODOLOGY.md. Clean data: new_data.csv (180 rows). New scores are signed differences from explicit references, except shooting HHI; negative reference differences are not negative raw entropy. Run python3 teamgraphs.py --revised-methodology from SportsAnalysis. Prior results and graphs are preserved. Requires existing court_zone_networks data, stored independent passing null baselines and archived position metadata. Both primary and opportunity-conditioned sensitivity results are shown; latest-season tests are retrospective.\n')
