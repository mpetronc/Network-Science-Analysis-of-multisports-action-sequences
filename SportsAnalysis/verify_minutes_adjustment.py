from pathlib import Path
import argparse,json,hashlib,re
import pandas as pd,numpy as np

def tv(A,B):return abs(A/A.sum()-B/B.sum()).sum()/2
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args(argv);out=a.data_dir;root=a.output_dir;d=pd.read_csv(out/'team_comparison_metrics.csv');m=pd.read_csv(out/'minute_allocations_and_predictions.csv');assert len(d)==30 and d.regression_eligible.sum()==24
 for z in d.itertuples():
  folder=out/z.team;frames=[pd.read_csv(folder/(name+'_matrix.csv'),index_col=0) for name in ['deletion','sender_minutes','both_endpoints_minutes','actual']];D,M,R,A=[f.to_numpy() for f in frames];ids=frames[0].index.astype(int).tolist();star=ids.index(z.star_id);q=m[m.team.eq(z.team)].set_index('player_id').reindex(ids);mp=q.present_total_minutes.to_numpy(copy=True);ma=q.absent_total_minutes.to_numpy();mp[star]=0;pool=D.sum()/mp.sum();rate=np.divide(D.sum(1),mp,out=np.full(len(mp),pool),where=mp>0);target=rate*ma;target[star]=0
  assert np.allclose(M.sum(1),target) and np.allclose(R.sum(1),target);assert np.allclose(q.minutes_based_predicted_passes,M.sum(1));assert all(np.isfinite(B).all() and (B>=0).all() and not np.diag(B).any() and B[star].sum()+B[:,star].sum()==0 for B in [D,M,R,A]);assert np.all(M[ma==0]==0) and np.all(M[:,ma==0]==0)
  independent=np.zeros_like(M)
  for i in range(len(ids)):
   if target[i]==0:continue
   mask=ma>0;mask[star]=False;mask[i]=False;k=D[i].copy();k[~mask]=0
   if k.sum()==0:k=ma*mask
   independent[i]=target[i]*k/k.sum()
  assert np.allclose(independent,M);original=pd.read_csv(root/'onoff_dependency/predicted_actual_networks'/z.team/'deleted_counts.csv',index_col=0).to_numpy();assert np.array_equal(D,original)
  assert np.isclose(tv(D,A),z.deletion_edge_tv) and np.isclose(tv(M,A),z.sender_minutes_edge_tv);assert np.isclose(z.positive_nonstar_minute_gains+z.negative_nonstar_minute_changes,z.star_present_minutes_per_team_game+z.absent_team_recorded_minutes_per_game-z.present_team_recorded_minutes_per_game)
  C=pd.read_csv(folder/'common_deletion_matrix.csv',index_col=0);CM=pd.read_csv(folder/'common_sender_minutes_matrix.csv',index_col=0);B=pd.read_csv(folder/'common_actual_matrix.csv',index_col=0);assert C.index.tolist()==CM.index.tolist()==B.index.tolist();assert np.isclose(z.common_deletion_conditional_js,z.common_sender_minutes_conditional_js,atol=1e-10)
  for i in range(len(C)):
   x=C.to_numpy()[i];y=CM.to_numpy()[i]
   if x.sum()>0:assert np.allclose(x/x.sum(),y/y.sum())
 for z in pd.read_csv(out/'source_manifest.csv').itertuples():assert hashlib.sha256(Path(z.path).read_bytes()).hexdigest()==z.sha256
 assert b'30' in re.findall(rb'/Count\s+(\d+)',(out/'all_30_minutes_adjusted_networks.pdf').read_bytes());assert len(list(out.glob('*/*_minutes_models.png')))==30
 result=dict(teams=30,qualified=24,network_pages=30,baseline_unchanged=True,independent_sender_model_reconstruction=True,row_targets_equal_rate_times_recorded_minutes=True,unavailable_players_zero=True,no_star_or_self_passes=True,minute_change_identity_verified=True,common_panel_recipient_probabilities_preserved=True,sources_unchanged=True);(out/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
