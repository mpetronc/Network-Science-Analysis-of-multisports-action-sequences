from pathlib import Path
import pandas as pd

def write_report(out):
 out=Path(out);d=pd.read_csv(out/'nine_network_case_metrics.csv');selected=pd.read_csv(out/'selected_six_cases.csv');ranking=pd.read_csv(out/'all_92_performance_ranking.csv')
 text='''# Contrasting network cases: offensive drops, rises and near-zero estimates

Nine eligible 2025–26 passing-hub cases have compatible full-season hub-present/absent network caches. Case selection uses performance point estimates only: two largest estimated drops, two largest positive changes, and two remaining estimates closest to zero. This is an outcome-selected illustration, not a new independent validation or a regression sample. The complete 92-team-season performance ranking is exported separately; its global extremes need not have matched network caches.

| Group | Team and missing hub | Estimated offensive change (absent−present), pts/100 | Nominal 95% interval | Count-matched outgoing entropy change | Count-matched EC concentration change |
|---|---|---:|---|---:|---:|
'''
 for z in selected.itertuples():text+=f'| {z.case_group} | {z.team}: {z.star_name} | {z.offense_change:+.2f} | {z.change_ci_low:+.2f} to {z.change_ci_high:+.2f} | {z.entropy_change:+.4f} | {z.EC_concentration_change:+.4f} |\n'
 drop=d.sort_values('offense_change').iloc[0];rise=d.sort_values('offense_change').iloc[-1];flat=d.assign(distance=d.offense_change.abs()).sort_values('distance').iloc[0]
 text+=f'''
## Three central illustrations

**{drop.team}, without {drop.star_name}: largest estimated drop among available paired networks.** Offense changes {drop.offense_change:+.2f} points/100, with interval {drop.change_ci_low:+.2f} to {drop.change_ci_high:+.2f}. Count-matched remaining-player passing entropy increases {drop.entropy_change:+.4f}, while EC concentration changes {drop.EC_concentration_change:+.4f}. The expected matched nonhub EC leader changes from {drop.present_nonstar_passing_leader} to {drop.absent_passing_leader}. The top two absent-context EC shares are separated by only {100*drop.absent_top2_EC_gap:.2f} percentage points; that numerical leader is not necessarily a distinct replacement star. The remaining network can become more diverse and less centralized while offensive production falls. This does not mean diversity caused the loss; a broad network may not replace the missing hub's particular functions.

**{rise.team}, without {rise.star_name}: largest estimated rise among the nine.** The point estimate is {rise.offense_change:+.2f} points/100, but its interval spans {rise.change_ci_low:+.2f} to {rise.change_ci_high:+.2f}. This is not evidence that the team truly improves without its hub. Remaining-player passing entropy changes {rise.entropy_change:+.4f} and EC concentration {rise.EC_concentration_change:+.4f}. Its expected matched EC leader changes from {rise.present_nonstar_passing_leader} to {rise.absent_passing_leader}. Greater concentration among the remaining players is compatible with a noisy positive performance estimate; it does not establish a beneficial replacement mechanism.

**{flat.team}, without {flat.star_name}: closest-to-zero point estimate.** Change {flat.offense_change:+.2f}, with interval {flat.change_ci_low:+.2f} to {flat.change_ci_high:+.2f}. Count-matched entropy changes {flat.entropy_change:+.4f}, EC concentration {flat.EC_concentration_change:+.4f}; its expected matched EC leader changes from {flat.present_nonstar_passing_leader} to {flat.absent_passing_leader}. A near-zero point estimate with this uncertainty does not establish unchanged performance. We checked an explicitly illustrative ±2-points/100 equivalence margin; no candidate meets the Holm-corrected equivalence criterion.

## What the network pages show

Each of the nine available cases has three aligned passing diagrams: full hub-present network; that context with the hub removed and outgoing pass counts matched; and hub-absent context with the same remaining-player panel/counts. The hub is removed from BOTH matched panels so apparent redistribution is not an automatic deletion artifact. Match each remaining player's outgoing total to min(present, absent). Edge weights in the displayed matched graphs are the EXPECTED values from this without-replacement matching, computed as original row probability × matched row total. They can be fractional, not newly observed passes. Both matched panels use the same positions, node sizes and edge-width scale; the full-roster first panel has its own edge-width scale. Fruchterman–Reingold placement uses passing edges only.

Stored entropy/EC change summaries are 200-draw count-matched averages from the prior adaptation analysis. Because entropy and EC are nonlinear, those averages need not exactly equal metrics of the displayed expected graph; both definitions are explicitly exported. Bars show the top 12 players' expected-graph EC shares with full matched-panel denominators, and the largest changes in FGA shares among the same remaining players. EC is ONLY on passing edges. No shooting edge enters the centrality calculation. Shooting shares are descriptive full-season presence/absence contexts, not performance controls. Gray bars show changes in minutes share, so a shot-share increase can be compared with increased playing opportunity. Player tables also export attempts per 36 minutes. These comparisons cannot control shared-court time or infer tactical intent; other teammates may also miss games.

## Uncertainty and scope

Performance estimates use the prior-defined hub study's post-opening-10-game, trade-censored, context-adjusted sample. Network caches summarize ALL saved presence/absence games across the season. They do not align exactly to that stricter performance window or to individual absence episodes. Other teammate availability, roster changes and shared-court exposure can still drive network differences. Per-donor count matching does not control recipient co-appearance. These case diagrams cannot establish what caused a performance loss or gain.

Ranks use uncertain point estimates; being largest in this nine-case set is not proof of being the most dependent team. The nine-case individual-test Holm correction and illustrative equivalence-test correction are exported. They do not correct the entire project's historical search or the selection of all 92 prior estimates. Nonsignificance is not equivalence. The complete all-92 ranking may include apparent improvements with wide intervals and role players identified as passing hubs; high EC is not synonymous with being the team's best player.

## Useful research conclusion

Network diversity, concentration and offensive effectiveness should be measured separately. These contrasting cases show why becoming more balanced does not necessarily preserve a missing player's contribution. Conversely, noisy apparent improvements or near-zero estimates should not be treated as proof of successful adaptation. Use the cases to propose specific role-replacement questions, then evaluate those questions in independent, time-aligned episode data.

Files: network_case_comparisons.pdf has the nine-case uncertainty overview, nine detailed pages (the six highlighted cases first) and an all-nine change comparison. selected_six_cases.csv and nine_network_case_metrics.csv contain all values; per-team matrices and player_role_changes.csv allow inspection of re-routing and shot-share changes. No new causal regression on the outcome-selected cases is claimed.
'''
 (out/'RESULTS_AND_METHOD.md').write_text(text)
 (out/'README.md').write_text('Start with RESULTS_AND_METHOD.md and network_case_comparisons.pdf. The groups refer to estimated offense changes during prior-defined passing-hub nonappearance. Paired networks are available for nine 2025–26 cases; all-92 ranking is separate. Rises and closest-to-zero estimates are not verified improvement/equivalence. Shared panel/outgoing counts and passing-only EC are enforced. Existing outputs remain intact. Run python3 teamgraphs.py --dependency-network-cases from SportsAnalysis.\n')
