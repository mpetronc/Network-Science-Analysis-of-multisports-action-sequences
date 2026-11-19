from pathlib import Path
import sys,json
import numpy as np,pandas as pd
from lineup_entropy_analysis import H,smooth_row_model,marginal_model,loss,posterior_mean_entropy
ROOT=Path.home()/'Desktop/SportsAnalysis/nba_passing_output'
OUT=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'lineup_role_networks/entropy_analysis'
A=np.ones((5,5))-np.eye(5);assert np.isclose(float(H(A)),1);assert np.allclose(H(np.stack([A,A*2])),1)
D=np.zeros((5,5));D[np.arange(5),(np.arange(5)+1)%5]=1;assert np.isclose(float(H(D)),0)
perm=[3,1,4,0,2];assert np.isclose(float(H(D[np.ix_(perm,perm)])),float(H(D)))
for P in [smooth_row_model(A,.5),marginal_model(A,.5)]:assert np.allclose(P.sum(1),1) and np.trace(P)==0 and np.isclose(loss(A,P),2)
assert 0<posterior_mean_entropy(A,.5)<1
bad=A.copy();bad[0,0]=1
try:H(bad)
except ValueError:pass
else:raise AssertionError('Player self-pass accepted')
audit=pd.read_csv(OUT/'sample_size_and_entropy_audit.csv',dtype={'game_id':str});assert len(audit)==79 and audit.passes.sum()==1019;old=pd.read_csv(ROOT/'lineup_role_networks/lineup_summary.csv',dtype={'game_id':str});j=audit.merge(old,on=['game_id','team','lineup'],validate='one_to_one');assert np.allclose(j.raw_conditional_entropy,j.conditional_entropy_fixed_five);assert audit.uniform_reference_mean.between(0,1).all()
case=pd.read_csv(OUT/'recurring_lineup_case.csv',dtype={'game_id':str});players=pd.read_csv(OUT/'recurring_lineup_players.csv');assert len(case)==3 and case.passes.sum()==143 and players.common_outgoing_sample.sum()==34
settings=json.loads((OUT/'analysis_settings.json').read_text());draws=pd.read_csv(OUT/'recurring_lineup_subsampling_draws.csv',dtype={'game_id':str});assert draws.groupby('game_id').size().eq(settings['draws']).all()
# Rebuild every held-game probability exclusively from the other two matrices.
scores=pd.read_csv(OUT/'leave_one_game_out_prediction.csv',dtype={'held_game':str});matrices=[]
for z in case.itertuples():matrices.append(pd.read_csv(ROOT/'lineup_role_networks/lineups'/z.game_id/z.team/z.lineup/'player_pass_counts.csv',index_col=0).to_numpy(float))
for held,z in enumerate(case.itertuples()):
 train=sum((a for i,a in enumerate(matrices) if i!=held),np.zeros((5,5)))
 for alpha in [.1,.5,1.]:
  for name,P in [('uniform_four_receivers',(np.ones((5,5))-np.eye(5))/4),('receiver_popularity',marginal_model(train,alpha)),('passer_specific_receivers',smooth_row_model(train,alpha))]:
   r=scores[(scores.held_game==z.game_id)&scores.alpha.eq(alpha)&scores.model.eq(name)].iloc[0];assert r.training_passes==143-r.test_passes;assert np.isclose(loss(matrices[held],P),r.held_logloss_bits_per_pass)
agg=pd.read_csv(OUT/'pooled_held_game_prediction.csv')
for z in agg.itertuples():
 a=scores[scores.model.eq(z.model)&scores.alpha.eq(z.alpha)];assert np.isclose(np.average(a.held_logloss_bits_per_pass,weights=a.test_passes),z.pooled_held_logloss_bits)
message='PASS: uniform/deterministic, vectorized and label-permutation entropy fixtures; self-pass rejection.\nPASS: all 79 raw lineup entropies reproduce earlier player matrices; 1,019 passes retained.\nPASS: case selected by repeated-lineup sample volume, 143 passes across three games.\nPASS: matched source sample is 34 passes; diagnostic draw counts reconcile.\nPASS: all held-game predictions independently rebuilt using only the other two games.\nPASS: probability rows, self-exclusions and pooled weighted log losses verified.\nSampling ranges are not confidence intervals. No population significance, official pass completeness or causal conclusions claimed.\n'
(OUT/'verification.txt').write_text(message);print(message)
