from pathlib import Path
import pandas as pd,numpy as np,json

def write_report(out):
 out=Path(out);d=pd.read_csv(out/'all_30_team_entropy.csv');valid=d[d.status.isin(['complete_cached','complete_new'])];missing=d[~d.status.isin(['complete_cached','complete_new'])];same=valid[valid.same_as_prior_season_hub]
 text=f'''# Full-league hub-present versus hub-absent passing entropy

All 30 teams are listed for the 2025–26 regular season. {len(valid)} have a validated observed presence/absence comparison. {int(d.absent_games.eq(0).sum())} hubs played all 82 games, so their observed absence entropy is unavailable, not zero. {int(d.status.isin(['missing_downloaded_network','unverified_profile_or_partition']).sum())} other teams lack a verified network partition.

## Hub definition and comparability

The all-league descriptive rule is each team's FULL-SEASON 2025–26 passing-only EC leader. This is retrospective, not a star selected before an absence, and is not used to validate a predictive dependency index. Current-year EC can identify a connector/role player, not necessarily the team's best player. The earlier nine-case cohort used a prior-defined hub and a matching network cache. Those nine have the same hub under both rules; their original observed entropy values are independently checked unchanged. For other teams the current hub can differ from the previous season's hub; both IDs/names and a same-hub flag are exported. Do not merge these definitions in the prospective-style absence study. {len(same)} available comparisons also retain the prior-season hub identity.

## EC ranking

Teams are ordered by the full-season passing-only eigenvector centrality concentration index (passing starness), highest first. EC is computed on A+Aᵀ using passing edges only. Player EC scores are normalized to shares p_i, then the index is (sum(p_i²) − 1/N)/(1 − 1/N), where N is the number of active passing-network players. Higher values mean centrality is more concentrated across players; zero means equal centrality. This is the existing passing-starness index, not the top player's EC share. Both the hub share and unadjusted EC HHI remain in the CSV for reference. Minutes and availability can affect this descriptive index; ordering does not control for them or establish dependency.

## Calculation

Outgoing conditional entropy H(receiver | passer), weighted by each player's outgoing pass total. With-hub includes the hub's outgoing passes and passes to the hub. Actually-without uses recorded passing from games without that hub. No shooting edges enter. Raw bits are primary; normalized scores use the SAME denominator within each team: log2(N_union_active−1). The union is taken from the full season. No per-context donor-population change is hidden in the denominator.

The deletion check zeroes the hub's row and column in the SAME observed present network. It is a synthetic diagnostic, not an observed absence. Algebraically, full absence minus full presence equals deletion difference plus additional context difference in bits. The latter also includes player composition and volume-weight differences, not merely tactics. Earlier fixed-nonhub/donor-matched results remain separate.

For sample-size sensitivity, independently subsample both full networks without replacement to the smaller TOTAL number of passes, 500 times. Matching every individual donor would erase the hub's row and defeat this total with/without comparison. Subsampling diagnostics are not confidence intervals for causal behavior. Source partitions are checked for nonnegative counts, zero diagonal, no star passes in absence, and exact full=present+absent. Missing profiles are not silently filled with zero. Independently observed zero passing is permitted.

## League descriptive results

{int(valid.entropy_change.gt(0).sum())} teams have higher observed entropy without the hub; {int(valid.entropy_change.lt(0).sum())} have lower. Mean normalized change across available teams: {valid.entropy_change.mean():+.4f}; median: {valid.entropy_change.median():+.4f}. Equal-total-pass mean change: {valid.equal_total_entropy_change.mean():+.4f}. These are descriptive team summaries, not causal league-average effects. Low-exposure cases with fewer than 10 absence games are flagged, rather than excluded from an all-team request.

| EC rank | Team | Passing hub | EC concentration index | Present / absent games | With hub | Actually without | Change | Equal-total change | Status |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|
'''
 def fmt(v):return f'{v:.4f}' if pd.notna(v) else 'N/A'
 for z in d.itertuples():text+=f'| {z.ec_rank} | {z.team} | {z.hub_name} | {z.passing_ec_concentration:.4f} | {z.present_games}/{z.absent_games} | {fmt(z.with_hub_entropy)} | {fmt(z.without_hub_entropy)} | {fmt(z.entropy_change)} | {fmt(z.equal_total_entropy_change)} | {z.status} |\n'
 flips=valid[np.sign(valid.entropy_change)!=np.sign(valid.equal_total_entropy_change)].team.tolist()
 text+=f'\nAfter equal-total sampling, {int(valid.equal_total_entropy_change.gt(0).sum())} teams increase and {int(valid.equal_total_entropy_change.lt(0).sum())} decrease. Direction changes for '+(', '.join(flips) if flips else 'none')+'. The league mean changes from slightly negative to near zero/slightly positive, showing the importance of unequal sample counts. This does not establish a null effect; team patterns vary and exposure remains unmatched.\n'
 text+='''
## Coverage and uncertainty

The comparison concerns full-season contexts, not exact before/after episodes or common on-court lineups. Player availability, donor activity mix, role changes, game schedules and other absences remain different. A hub missing only two or three games gives a weak basis for generalizing behavior; sampling equal pass counts cannot fix that. Higher entropy measures varied recipient choices, not offensive quality or better performance. The synthetic deletion score can be calculated for a never-absent hub but does not provide observed absence evidence.

The earlier cached networks cover 21 teams. Additional low-absence teams are reconstructed from date-filtered NBA PlayerDashPtPass Totals for disjoint absence blocks and subtracted from a consistent full-season matrix. All required player profiles must be complete. Empty profiles count as zero only when independent saved daily passing records confirm zero for all official appearances. Positive passes to unknown roster IDs, date filters with impossible GP, negative subtraction, or any pass to/from the absent hub invalidate the partition. Profile-vs-daily total differences are audited because these are distinct saved tracking outputs, not assumed equal. Detailed source/cache audits and download progress are retained.

Files: all_30_team_entropy.csv is the full coverage table; available_team_comparisons.csv contains verified pairs; unchanged_hub_definition_subset.csv marks available same-prior-hub identities but may also contain a never-absent hub with N/A. all_teams_star_presence_entropy.pdf shows all clubs and the league comparison. Prior studies and original nine-case results are preserved. The all-team rule is not substituted into the previous prior-season-defined performance-loss model.
'''
 (out/'RESULTS_AND_METHOD.md').write_text(text);(out/'summary.json').write_text(json.dumps(dict(season='2025-26',listed_teams=30,verified_pairs=len(valid),no_absence_teams=d.loc[d.absent_games.eq(0),'team'].tolist(),other_unverified_teams=missing.loc[missing.absent_games.gt(0),'team'].tolist(),higher_entropy_without=int(valid.entropy_change.gt(0).sum()),lower_entropy_without=int(valid.entropy_change.lt(0).sum()),equal_total_higher=int(valid.equal_total_entropy_change.gt(0).sum()),equal_total_lower=int(valid.equal_total_entropy_change.lt(0).sum()),direction_changed_teams=flips,low_absence_teams=valid.loc[valid.few_absence_games,'team'].tolist(),mean_change=float(valid.entropy_change.mean()),median_change=float(valid.entropy_change.median()),equal_total_mean_change=float(valid.equal_total_entropy_change.mean()),hub_rule='Current full-season passing-only EC; retrospective descriptive',same_prior_hub_verified_pairs=len(same)),indent=2))
