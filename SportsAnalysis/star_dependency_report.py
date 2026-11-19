"""Readable results and methodology for the prior-defined leader absence study."""
from pathlib import Path
import json,hashlib,platform
import numpy as np,pandas as pd

def write_report(out,root,cache):
 out=Path(out);elig=pd.read_csv(out/'eligibility.csv');fit=pd.read_csv(out/'primary_panel_results.csv');sens=pd.read_csv(out/'panel_sensitivities.csv');specific=pd.read_csv(out/'star_specific_share_models.csv');ep=pd.read_csv(out/'absence_episodes.csv');ad=pd.read_csv(out/'adaptation/episode_shooting_adaptation.csv');tests=pd.read_csv(out/'adaptation/episode_adaptation_tests.csv');passing=pd.read_csv(out/'adaptation/limited_passing_adaptation.csv');pt=pd.read_csv(out/'adaptation/limited_passing_adaptation_tests.csv');ex=pd.read_csv(out/'adaptation/excluded_episode_windows.csv');conf=pd.read_csv(out/'adaptation/shot_log_disagreements.csv');penalties=pd.read_csv(out/'team_season_penalties.csv')
 rows=[]
 for policy in ['passing_hub','leading_shooter']:
  data=pd.read_csv(out/(policy+'_primary_games.csv.gz'),dtype={'GAME_ID':str});rows.append(data[['season','team','GAME_ID']])
 unique_games=len(pd.concat(rows).drop_duplicates());unique_units=len(pd.concat(rows)[['season','team']].drop_duplicates())
 methodology='''# Star dependency: revised absence-study methodology

## Research question and operational definition

Do teams lose offensive efficiency when a previously identified leader does not appear, and do prior passing/shooting structures predict a larger decline? Dependency is represented by a conditional offensive-rating penalty during nonappearance, not by entropy or centrality alone. The estimates are observational, not randomized causal player impacts.

## Data and time ordering

Official NBA regular-season team basic, team advanced and player game logs cover 2020–21 through 2025–26. Outcomes use the five follow-up seasons 2021–22 through 2025–26. The immediately preceding season supplies each leader ID and baseline structure. All 30 teams are considered each follow-up year (150 candidate team-seasons per definition). Complete team/player game counts and player-sum points, FGA and FGM are checked. Official advanced OFF_RATING is the game outcome; possessions are positive and source game keys match.

Primary leader definition: the previous season's highest passing-only eigenvector-centrality share. Secondary definition: the previous season's largest total FGA. These identify a passing hub and leading shooter, not necessarily the league's best player or highest per-minute scorer. EC uses A+Aᵀ from player-to-player passes only; shooting edges never enter. Shooting HHI is calculated from prior player FGA shares, without FTA/TOV. No combined EC/usage score or arbitrary 50/50 weights is used.

We use the first 10 team games as an availability landmark. The prior-defined leader must be observed on that team at least once in this window and must not already have appeared for a different team. Evaluation starts after both teams in a game have completed their first 10 games. This ensures key lists and their historical weights are defined before follow-up. Actual appearance/nonappearance statuses are realized participation from game logs, not recorded pregame injury announcements, so this is not a fully pregame prediction model. Stars absent throughout the opening window are excluded even if they return later; this limits generalization.

If the leader subsequently appears for another team, follow-up is censored 14 days before that first other-team appearance. This is a transparent conservative proxy for avoiding trade-related nonappearances, not an exact transaction date. Primary team-seasons require at least 15 leader appearances and five nonappearances in the remaining observation window. Sensitivities include one-to-four-nonappearance units, late-season exclusions and high co-absence exclusions. No player-minute cutoff is imposed; any logged appearance counts, including brief appearances. The selection threshold addresses within-team estimability, not star quality.

## Outcome and context controls

Outcome: official game offensive rating, points per 100 possessions. Compare games in the same team-season, with a separate intercept and linear game-number trend for each team-season. Other controls: nominal home venue, rest (capped at three off days), own and opponent back-to-back status, pregame opponent defensive strength, and missing key teammates/opponents.

Opponent defense uses only preceding games, shrunk to the preceding season's defensive rate with 1,000 prior possessions (approximately 10 games). The focal game and future games do not enter this rating. Key-player lists use the top three previous-season NBA-minute players observed on each team's opening-10-game roster, including newcomers from other teams. Their previous-season minutes determine fixed weights; own-star controls exclude the focal leader. Actual current-game minutes are not used as an availability covariate. This only partly controls simultaneous absences and current roster quality; breakouts, new rookies and other role changes remain.

## Primary model

OffRating = team-season intercept + team-season linear trend + context controls + absence coefficient + absence×prior passing EC concentration + absence×prior shooting HHI + absence×shared-leader indicator + error.

Shared leader means the same player led prior EC and prior FGA. Continuous traits are centered and divided by their standard deviations across eligible team-seasons; the shared-leader indicator is centered but kept in binary units. The main absence coefficient therefore evaluates the predicted penalty at average prior structure in the eligible cohort. A one-standard-deviation interaction describes how that penalty changes with concentration, conditional on the other included traits. The shared-leader interaction is the conditional difference between shared and separate leadership. It is not a causal effect of centralization.

The coefficient on absence is an offensive-rating change; dependency_penalty is its negative, so positive means offense is worse when the leader does not appear. Main inference uses one game-level panel rather than regressing noisy estimated team penalties against indices. Team-season estimates are supplied for descriptive plots, with HC3 uncertainty and identifiability checks; they should not be ranked as precise causal player effects.

## Inference and robustness

Franchise-clustered standard errors account for repeated team seasons/games (30 franchises); t reference has 29 df. Additional two-way franchise/game and franchise/player covariance checks address shared games and players changing franchises across years. Primary Holm correction covers eight tests: absence plus three structure interactions under each of the two leader definitions. The two samples overlap, often sharing the same leader, so their game counts and effects cannot be added or treated as independent. Ordinary 95% CIs are pointwise; Bonferroni familywise CIs are also exported.

Sensitivity models: remove team-specific trends; include units with one-to-four nonappearances; exclude games with high other-key unavailability; exclude the final 10 team games; restrict to at least 70% of prior roster minutes represented in the opening window; add prior minutes-concentration moderation; test EC/HHI interactions separately; use opening-10-game shooting HHI; and use prior activity-adjusted passing entropy/EC as alternatives. Star-specific EC/FGA-share models are exported separately with their own Holm family. Sensitivity p values are exploratory and not a menu from which to select the strongest result. Previous activity-adjusted passing inputs depend only on the prior season's network; future-fitted position baselines are NOT used in this prior-season prediction design.

This is a new exploratory observational analysis after extensive project exploration, not an untouched holdout, preregistration, or external validation. Features are time-aligned before follow-up, but the research design itself has been developed retrospectively. In-sample fit/partial R² are not prospective forecast accuracy.

## Absence episodes and adaptation

Consecutive nonappearance runs of at least two observed team games are identified. A preceding five-game window must exist with at least four leader appearances. No episode is selected based on points, wins or offensive rating. Optional five-game return windows require at least four appearances and are only used for the descriptive return plot. Right-censored runs are flagged; the observed run length is not claimed to be its full eventual duration. Episode outcome adjustments come from the panel model and are not a causal event study.

For shooting adaptation, REMOVE THE LEADER FROM BOTH pre and absence comparisons. Primary comparison uses the same other players with at least one appearance in each window. FGA HHI is the pair-based estimator sum_i n_i(n_i−1)/[N(N−1)], which corrects finite-count upward bias under a multinomial interpretation. Top-shooter share is also compared. An all-remaining-player union sensitivity retains rotation changes instead of conditioning on a common core. Another sensitivity requires the leader to appear in ALL five pre-window games instead of at least four. Common-core coverage and shooter counts are exported.

Shooting-zone entropy uses the 15 official court zones. For each remaining shooter with attempts in both windows, sample without replacement down to the smaller FGA count in that pair of windows. Repeat 200 times and use identical per-player sample weights in both contexts. This addresses unequal count effects; simulation spread is sampling sensitivity, not a confidence interval for behavioral change. Episode mean tests cluster by franchise and apply Holm across eight adaptation outcomes (four under each leader definition). Episodes and policy samples overlap and are not independent.

Full game-window passing networks are not available across the five seasons. A limited secondary check uses only the nine compatible existing 2025–26 presence/absence caches whose hub ID matches the prior-defined hub. The hub is removed in both contexts, the remaining positive-outgoing panel is fixed, and outgoing pass counts are matched per player using 200 without-replacement samples. These are WHOLE-SEASON hub-present versus hub-absent contexts, not episode before/during networks. No exact shared-court or pass-to-shot sequence claim is made.

## Data disagreements and remaining limitations

Two official player-game FGA counts disagreed with shot-chart event counts in games touched by the episode extraction. Fresh official shot-chart queries confirmed the same counts. No missing shot location was invented. One affected focal episode window is excluded from location-based shooting adaptation; the other discrepancy concerned a nonfocal side of a used game. Both are audited. Main game-log performance models remain unchanged because their team/player totals agree independently; they do not depend on shot-chart location counts.

Nonappearance is not confirmed injury: it can include rest, suspension, illness or coaching decisions. Appearing is not on-court possession exposure. Team-specific trends, pregame opponents and key-availability controls cannot remove all selection/confounding. Prior-season leaders and structures can become outdated as roles/rosters change. Early-season/full-season absences and departed leaders are not fully represented. Coarse key lists omit some important absences; nominal home labels can include neutral venues. All estimated loss, redistribution and entropy patterns remain observational. More even sharing during an absence does not establish that balanced play causes the performance decline.

The method now evaluates actual performance loss and avoids using absence-period data to choose the leader or primary structure predictors. It can show conditional dependency estimates; it does not automatically validate an EC/HHI index as a universal star-dependency score.
'''
 (out/'METHODOLOGY.md').write_text(methodology)
 summary=f'''# Star dependency: results of the five-season study

## Main finding

Teams show an adjusted offensive decline when their prior-defined leader does not appear. The network/shot concentration measures are less reliable at identifying WHICH teams decline most.

The passing-hub sample has 92 eligible team-seasons and 6,449 game observations; the leading-shooter sample has 99 and 6,958. They overlap: {unique_units} unique team-seasons and {unique_games:,} unique team-game observations across the two policies. All 30 franchises are represented. The 150 candidate team-seasons per policy are screened by explicit retention, trade-censor and appearance-variation rules.

| Prior-defined missing leader | Eligible team-seasons | Games | Estimated decline, points/100 | Pointwise 95% CI | Holm p (eight primary tests) |
|---|---:|---:|---:|---|---:|
'''
 labels={'passing_hub':'Passing EC leader','leading_shooter':'FGA leader'}
 for policy in labels:
  z=fit[(fit.policy==policy)&fit.term.eq('absence')].iloc[0];summary+=f'| {labels[policy]} | {z.team_seasons} | {z.n_games:,} | {z.dependency_penalty:.2f} | {z.penalty_ci_low:.2f} to {z.penalty_ci_high:.2f} | {z.holm_p:.3g} |\n'
 block=fit[fit.term.eq('absence')]
 summary+='\nGame-to-game offensive variation remains large. The absence-plus-interaction block accounts for only '+', '.join(f'{r.absence_block_partial_r_squared:.2%}' for r in block.itertuples())+' of residual game-level variation after context effects, respectively. Full in-sample model R², including team-season effects/trends and controls, is '+', '.join(f'{r.full_model_r_squared:.1%}' for r in block.itertuples())+'. A statistically supported mean penalty is not the same as highly accurate game forecasting.\n'
 summary+='''
Positive penalty means lower offensive rating during nonappearance, conditional on the model. It is evaluated at the eligible cohort's mean prior structure. The two estimates are not a formal test that one leader type matters more: samples and absences overlap, with different compositions. All core sensitivity versions retain a positive average penalty; exclusions of high co-absences and late-season games reduce its size.

## Does prior structure predict a larger loss?

| Leader definition | Prior feature | Penalty change | Raw p | Holm p |
|---|---|---:|---:|---:|
'''
 for z in fit[fit.term.ne('absence')].itertuples():
  name={'absence_x_pre_ec':'Passing EC concentration (per 1 SD)','absence_x_pre_shooting_hhi':'Shooting HHI (per 1 SD)','absence_x_shared_leader':'Shared versus separate leader'}[z.term];summary+=f'| {labels[z.policy]} | {name} | {z.dependency_penalty:+.2f} points/100 | {z.p:.3g} | {z.holm_p:.3g} |\n'
 summary+='''
Higher prior passing concentration is suggestively associated with a larger absence penalty, but neither primary EC interaction survives the eight-test correction. Shooting HHI and shared-leader status are not confirmed moderators. Baseline-adjusted EC/entropy and player-specific share models are sensitivity checks, not independent discoveries. Stronger conclusions cannot be obtained by choosing whichever sensitivity has the smallest uncorrected p value.

This is a material distinction from the earlier entropy–starness regressions: an absence penalty is now measured from performance, and concentration is tested as its prior predictor. Evidence for the average absence penalty is stronger than evidence that our structural scores explain its variation.

## Do remaining players find a replacement star or redistribute?

The leader is excluded from BOTH comparison windows. Common remaining players account for approximately 92–94% of the other players' shot attempts. Results:

| Prior-defined leader | Episode comparisons | Remaining-player FGA HHI before | During absence | Mean HHI change | Episodes with lower HHI |
|---|---:|---:|---:|---:|---:|
'''
 for policy,g in ad.groupby('policy'):
  summary+=f'| {labels[policy]} | {len(g)} | {g.pre_FGA_HHI.mean():.3f} | {g.absent_FGA_HHI.mean():.3f} | {g.FGA_HHI_change.mean():+.4f} | {g.FGA_HHI_change.lt(0).mean():.1%} |\n'
 summary+='''
Both HHI decreases survive the eight-test adaptation correction. The all-remaining-player union sensitivity has the same direction. The highest remaining shooter's share falls by about 2.6 percentage points on average. Thus the observed tendency is more evenly distributed shooting among remaining teammates, rather than systematically funneling shots toward one replacement. A different top remaining shooter can still emerge: its identity changes in roughly 56–57% of comparisons. That identity change alone does not establish a new star.

Shooting-zone entropy changes are small and uncertain after count matching. Redistribution WHO takes shots can occur without a substantial change in the variety of zones FROM WHICH each remaining player shoots. The adaptation comparisons are descriptive, use short preceding windows, and do not establish that redistribution caused or prevented the offensive decline.

## What about passing adaptation?

Only nine compatible 2025–26 season-context pairs are available. Count-matched passing entropy change and remaining-player EC-concentration change have intervals including zero, with corrected p values well above .05. No firm claim about a new passing hub versus passing balance is supported. These are aggregate presence/absence contexts, not episode-specific or lineup-specific networks.

## Episode and data-quality scope

140 passing-hub and 165 leading-shooter episode comparisons meet the availability-window rules before the location-data check. One passing-hub episode window has an official shot-chart/game-log discrepancy and is excluded from shooting-location adaptation, leaving 139 and 165 comparisons. Episode cohorts overlap and cannot be added as independent events. Optional return plots have 89 complete passing-hub windows and 101 complete leading-shooter windows; they are selected descriptive subsets. Right-censored runs remain flagged. All other eligible performance games remain in the main analysis.

## Interpretation

The strongest supported statement is: **NBA teams in these retained-leader cohorts tend to lose offensive efficiency during leader nonappearance, while the remaining players share shot attempts more evenly.** The study does not yet establish a reliable network-based rule for predicting which teams are most dependent. Higher passing centralization is a candidate for genuinely new-data testing, not a confirmed universal index. Missing-player reasons and lineup exposure remain important unresolved limitations.

Read METHODOLOGY.md for chronology, controls, selection, model equations and limitations. Primary p values are franchise clustered and Holm corrected; displayed 95% intervals are pointwise. Prior broad project exploration means these remain exploratory results, even though leader IDs and predictors are time-aligned to the previous season.
'''
 (out/'RESULTS.md').write_text(summary)
 (out/'README.md').write_text('Start with RESULTS.md and METHODOLOGY.md. Graphs: star_dependency_results.pdf and adaptation/adaptation_results.pdf. Primary statistics: primary_panel_results.csv. Team-season descriptive data: team_season_penalties.csv; episodes: absence_episodes.csv. Compressed game panels and pregame-control data are included. The two leader-policy samples overlap. Run from SportsAnalysis: python3 teamgraphs.py --star-dependency-study. Cache downloader: python3 fetch_dependency_games.py. Full original SportsAnalysis prior networks and shot caches are required for regeneration. No causal or validated universal starness-index claim.\n')
 # Compact data table and complete reproducibility information.
 penalties.to_csv(out/'new_team_season_data.csv',index=False)
 versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__);(out/'runtime_versions.json').write_text(json.dumps(versions,indent=2))
 provenance=[]
 for f in Path(__file__).resolve().parent.glob('star_dependency*.py'):provenance.append(dict(path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
 (out/'computation_source_hashes.json').write_text(json.dumps(provenance,indent=2))
