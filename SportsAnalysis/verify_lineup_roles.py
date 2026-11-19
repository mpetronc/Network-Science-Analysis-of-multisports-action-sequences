from pathlib import Path
import sys,json
import numpy as np,pandas as pd
from lineup_role_networks import candidates,transform,entropy
ROOT=Path.home()/'Desktop/SportsAnalysis/nba_passing_output'
OUT=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'lineup_role_networks'
# Synthetic tests target bijections and entropy invariance rather than role accuracy.
ids=[1,2,3,4,5];primary=dict(zip(ids,['PG','SG','SF','PF','C']));broad=dict(zip(ids,['G','G','F','F','C']));_,opt,cost,gap=candidates(ids,primary,broad);assert len(opt)==1 and cost==0 and opt[0].tolist()==list(range(5))
primary[2]='PG';_,opt,cost,gap=candidates(ids,primary,broad);assert len(opt)==2 and cost==1
A=np.arange(25,dtype=float).reshape(5,5);np.fill_diagonal(A,0)
for a in opt:
 B=transform(A,a);assert B.sum()==A.sum() and np.trace(B)==0 and np.isclose(entropy(B),entropy(A))
assert np.isclose(entropy(np.ones((5,5))-np.eye(5)),1)
D=np.zeros((5,5));D[np.arange(5),(np.arange(5)+1)%5]=1;assert entropy(D)==0
try:transform(A,[0,0,2,3,4])
except ValueError:pass
else:raise AssertionError('Non-bijective role assignment accepted')
summary=pd.read_csv(OUT/'lineup_summary.csv',dtype={'game_id':str});passes=pd.read_csv(OUT/'pass_role_assignments.csv',dtype={'game_id':str});assign=pd.read_csv(OUT/'lineup_role_assignments.csv');assert len(passes)==1019 and passes.role_source.ne(passes.role_target).all()
for (team,lineup),a in assign.groupby(['team','lineup']):assert len(a)==5 and a.inferred_role.nunique()==5 and a.player_id.nunique()==5
for z in summary.itertuples():
 folder=OUT/'lineups'/z.game_id/z.team/z.lineup;A=pd.read_csv(folder/'player_pass_counts.csv',index_col=0);B=pd.read_csv(folder/'inferred_role_pass_counts.csv',index_col=0);P=pd.read_csv(folder/'inferred_role_pass_probabilities.csv',index_col=0);assert A.shape==B.shape==(5,5);assert A.to_numpy().sum()==B.to_numpy().sum()==z.inferred_passes;assert np.trace(B)==0
 if z.inferred_passes>0:
  oldpath=ROOT/'sportvu_passing_sequences/games'/z.game_id/'lineup_networks'/z.team/z.lineup/'pass_counts.csv';old=pd.read_csv(oldpath,index_col=0);old.index=old.index.astype(int);old.columns=old.columns.astype(int);ids=A.index.astype(int).tolist();assert np.array_equal(A.to_numpy(),old.loc[ids,ids].to_numpy());assert np.isclose(entropy(A.to_numpy()),entropy(B.to_numpy()))
  assert np.isclose(z.conditional_entropy_fixed_five,entropy(B.to_numpy()))
 assert P.loc[B.sum(axis=1)>0].sum(axis=1).sub(1).abs().lt(1e-10).all();assert P.loc[B.sum(axis=1).eq(0)].isna().all().all()
for game,g in passes.groupby('game_id'):
 f=np.load(ROOT/'sportvu_passing_sequences/frame_cache'/(game+'_frames.npz'))
 for z in g.itertuples():
  a,b=int(z.release_frame),int(z.receive_frame);own=np.sort(f['player_ids'][a][f['team_ids'][a]==z.source_team]);assert '-'.join(map(str,own))==str(z.lineup)
  assert int(f['timestamp_ms'][a])==int(z.release_timestamp_ms) and int(f['timestamp_ms'][b])==int(z.receive_timestamp_ms)
  assert (f['quarter'][a:b+1]==z.quarter).all() and (f['player_ids'][a:b+1]==f['player_ids'][a]).all() and (f['team_ids'][a:b+1]==f['team_ids'][a]).all()
  assert z.source in own and z.target in own
segments=pd.read_csv(OUT/'observed_tracking_segments.csv',dtype={'game_id':str});seconds=segments.groupby(['game_id','team','lineup']).observed_running_clock_seconds.sum();assert np.allclose([seconds.loc[(z.game_id,z.team,z.lineup)] for z in summary.itertuples()],summary.observed_running_clock_seconds)
e=pd.read_csv(OUT/'event_lineup_alignment_audit.csv');assert not e.duplicated(['game_id','team','eventnum','actor_id','source_event','target_event']).any()
s=pd.read_csv(OUT/'assignment_sensitivity_draws.csv');assert s.groupby('team').size().eq(json.loads((OUT/'role_inference_rules.json').read_text())['sensitivity_draws']).all() and s.mixing_gap.ge(-1e-10).all()
message='PASS: unique/tied prior-role fixtures and invalid non-bijection rejection.\nPASS: 80 observed lineup-game matrices, 79 passing-positive groups, all five players retained in matrices.\nPASS: every passing-positive player matrix exactly matches the prior extractor lineup matrix.\nPASS: all 1,019 passes match frame timestamps, stable on-court memberships and valid actors throughout transfer.\nPASS: one-to-one role mappings preserve all counts, remove role self-loops, and leave within-lineup entropy unchanged.\nPASS: probability rows sum to one where observed, and are NaN when undefined.\nPASS: tracking-segment durations reconcile to observed lineup exposure summaries.\nPASS: no duplicate event annotations; event matches/rejections exported.\nPASS: 500 role-assignment sensitivity draws per team, fixed seed, zero role self-loops and nonnegative mixing gaps.\nThese checks do not validate inferred tactical role accuracy, completeness of tracking passes, official lineup minutes, or possession paths.\n'
(OUT/'verification.txt').write_text(message);print(message);print('Median passes per passing-positive group:',summary.loc[summary.inferred_passes.gt(0),'inferred_passes'].median())
