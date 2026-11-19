from pathlib import Path
import argparse,json,re
import numpy as np,pandas as pd
from scipy.stats import entropy,linregress

def H(M):
 q=M/M.sum();return entropy(q.ravel(),base=2)-entropy(q.sum(1),base=2)
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args(argv);out=a.data_dir;root=a.output_dir;d=pd.read_csv(out/'network_metrics.csv');coverage=pd.read_csv(out/'coverage_all_30.csv');prior=pd.read_csv(root/'court_zone_networks/regressions/four_metrics.csv');prior=prior[prior.season.eq('2024-25')].set_index('team');assert len(coverage)==30 and len(d)==19
 for z in d.itertuples():
  assert z.star_id==prior.loc[z.team,'passing_leader_id'];folder=out/'networks'/z.team;P=pd.read_csv(folder/'present_passing_matrix.csv',index_col=0);A=pd.read_csv(folder/'absent_passing_matrix.csv',index_col=0);D=pd.read_csv(folder/'synthetic_deleted_passing_matrix.csv',index_col=0);ids=P.index.astype(int).tolist();i=ids.index(z.star_id);B=P.to_numpy().copy();B[i]=0;B[:,i]=0;assert np.array_equal(B,D.to_numpy());T=P.to_numpy()+A.to_numpy();n=np.count_nonzero(T.sum(0)+T.sum(1));den=np.log2(n-1);assert np.isclose(H(P.to_numpy())/den,z.with_hub_entropy) and np.isclose(H(A.to_numpy())/den,z.without_hub_entropy) and np.isclose(H(B)/den,z.deleted_entropy);assert A.loc[z.star_id].sum()+A[str(z.star_id)].sum()==0
  assert np.isclose(z.entropy_change,z.deletion_entropy_change+z.additional_context_change)
  if z.window_games==82:
   F=pd.read_csv(root/'2025-26/team_season_topics/networks'/z.team/'passing_matrix.csv',index_col=0);assert np.array_equal(T,F.to_numpy())
 plan=pd.read_csv(out/'network_download_plan.csv')
 for team,q in plan.groupby('team'):
  folder=out/'networks'/team;A=pd.read_csv(folder/'absent_passing_matrix.csv',index_col=0);P=pd.read_csv(folder/'present_passing_matrix.csv',index_col=0);ids=A.index.astype(int).tolist();ix={pid:i for i,pid in enumerate(ids)};mat={kind:np.zeros(A.shape) for kind in q.kind.unique()}
  for z in q.itertuples():
   f=out/'passing_profiles'/team/(z.start+'_'+z.end)/(str(z.player_id)+'.csv');assert json.loads(f.with_suffix('.json').read_text())['complete']
   for r in pd.read_csv(f).itertuples():
    if r.PASS>0:mat[z.kind][ix[z.player_id],ix[int(r.PASS_TEAMMATE_PLAYER_ID)]]+=r.PASS
  assert np.array_equal(mat['absent'],A.to_numpy())
  if 'total' in mat:assert np.array_equal(mat['total'],A.to_numpy()+P.to_numpy())
 sample=d[d.regression_eligible];assert len(sample)==14 and sample.present_games.ge(15).all() and sample.absent_games.ge(10).all()
 for z in pd.read_csv(out/'regression_statistics.csv').itertuples():
  y=sample.equal_total_entropy_change if z.measure=='equal_total' else sample.entropy_change;y=abs(y) if z.outcome=='absolute' else y;fit=linregress(sample.prior_on_off_gap,y);assert np.isclose(fit.rvalue,z.r) and np.isclose(fit.slope,z.slope) and np.isclose(fit.pvalue,z.pearson_p)
 assert b'19' in re.findall(rb'/Count\s+(\d+)',(out/'network_changes/highest_ec_player_three_networks.pdf').read_bytes());assert len(list((out/'network_changes').glob('*/*network_changes.png')))==19
 result=dict(listed_teams=30,available_pairs=19,regression_teams=14,network_pages=19,strict_prior_ec_selection_verified=True,synthetic_deletion_verified=True,independent_entropy_verified=True,new_raw_profile_reconstruction_verified=True,ols_checked_against_scipy=True);(out/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
