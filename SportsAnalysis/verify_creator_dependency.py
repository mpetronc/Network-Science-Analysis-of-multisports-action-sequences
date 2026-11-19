from pathlib import Path
import argparse,json,re
import pandas as pd,numpy as np
from scipy.stats import entropy,linregress

def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--data-dir',type=Path,required=True);a=p.parse_args(argv);root=a.output_dir;out=a.data_dir;c=pd.read_csv(out/'all_creator_candidates.csv');coverage=pd.read_csv(out/'coverage_all_30.csv');s=pd.read_csv(out/'selected_stars.csv');d=pd.read_csv(out/'network_metrics.csv');assert coverage.team.nunique()==30 and len(s)==len(d)==15
 for z in coverage.itertuples():
  f=c[c.team.eq(z.team)];q=f[f.MIN.ge(1000)].sort_values(['potential_assists_per36','PLAYER_ID'],ascending=[False,True]);assert int(q.iloc[0].PLAYER_ID)==z.star_id;assert np.isclose(z.prior_potential_assists_per36,36*z.prior_potential_assists/z.prior_tracking_minutes)
  raw=pd.read_csv(out/'tracking_2024-25'/(z.team+'.csv'));source=json.loads((out/'tracking_2024-25'/(z.team+'.source.json')).read_text());assert source['parameters']['Season']=='2024-25' and source['parameters']['PtMeasureType']=='Passing';assert source['parameters']['TeamID']==int(raw.TEAM_ID.iloc[0]);assert raw.PLAYER_ID.isin(c[c.team.eq(z.team)].PLAYER_ID).all()
 for z in d.itertuples():
  folder=out/'networks'/z.team;P=pd.read_csv(folder/'present_passing_matrix.csv',index_col=0);A=pd.read_csv(folder/'absent_passing_matrix.csv',index_col=0);D=pd.read_csv(folder/'synthetic_deleted_passing_matrix.csv',index_col=0);ids=P.index.astype(int).tolist();hi=ids.index(z.star_id);expected=P.to_numpy().copy();expected[hi]=0;expected[:,hi]=0;assert np.array_equal(expected,D.to_numpy());T=P.to_numpy()+A.to_numpy();den=np.log2(np.count_nonzero(T.sum(0)+T.sum(1))-1)
  for M,value in [(P,z.with_hub_entropy),(D,z.deleted_entropy),(A,z.without_hub_entropy)]:
   q=M.to_numpy()/M.to_numpy().sum();h=entropy(q.ravel(),base=2)-entropy(q.sum(1),base=2);assert np.isclose(h/den,value)
  assert A.loc[z.star_id].sum()+A[str(z.star_id)].sum()==0
  if z.window_games==82:
   F=pd.read_csv(root/'2025-26/team_season_topics/networks'/z.team/'passing_matrix.csv',index_col=0);assert np.array_equal(T,F.to_numpy())
 plan=pd.read_csv(out/'network_download_plan.csv')
 for team,group in plan.groupby('team'):
  folder=out/'networks'/team;P=pd.read_csv(folder/'present_passing_matrix.csv',index_col=0);A=pd.read_csv(folder/'absent_passing_matrix.csv',index_col=0);ids=P.index.astype(int).tolist();ix={pid:i for i,pid in enumerate(ids)};reconstructed={kind:np.zeros(P.shape) for kind in group.kind.unique()}
  for r in group.itertuples():
   f=out/'passing_profiles'/team/(r.start+'_'+r.end)/(str(r.player_id)+'.csv');assert json.loads(f.with_suffix('.json').read_text())['complete']
   for edge in pd.read_csv(f).itertuples():
    if edge.PASS>0:reconstructed[r.kind][ix[r.player_id],ix[int(edge.PASS_TEAMMATE_PLAYER_ID)]]+=edge.PASS
  assert np.array_equal(reconstructed['absent'],A.to_numpy())
 sample=d[d.regression_eligible];assert len(sample)==12
 for z in pd.read_csv(out/'regression_statistics.csv').itertuples():
  y=sample.equal_total_entropy_change if z.measure=='equal_total' else sample.entropy_change;y=abs(y) if z.outcome=='absolute' else y;fit=linregress(sample.prior_on_off_gap,y);assert np.isclose(fit.rvalue,z.r) and np.isclose(fit.slope,z.slope)
 assert b'15' in re.findall(rb'/Count\s+(\d+)',(out/'network_changes/primary_creator_three_networks.pdf').read_bytes())
 result=dict(teams=30,creator_selections_verified=True,selection_uses_previous_season_potential_assists_per36=True,minimum_tracking_minutes=1000,actual_pairs=15,regression_teams=12,network_pages=15,raw_profile_reconstruction_verified=True,entropy_independently_verified=True,regressions_checked_against_scipy=True);(out/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
