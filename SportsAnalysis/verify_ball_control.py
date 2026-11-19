from pathlib import Path
import argparse,json,re
import numpy as np,pandas as pd
from scipy.stats import entropy,linregress

def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--data-dir',type=Path,required=True);a=p.parse_args(argv);out=a.data_dir;root=a.output_dir;c=pd.read_csv(out/'all_ball_control_candidates.csv');coverage=pd.read_csv(out/'coverage_all_30.csv');d=pd.read_csv(out/'network_metrics.csv');assert len(coverage)==30 and len(d)==19;assert not any('percentile' in col or 'score' in col for col in c.columns)
 for z in coverage.itertuples():
  q=c[c.team.eq(z.team)&c.MIN.ge(1000)].sort_values(['TOUCHES','TIME_OF_POSS','PLAYER_ID'],ascending=[False,False,True]);assert int(q.iloc[0].PLAYER_ID)==z.star_id;assert np.isclose(q.iloc[0].TOUCHES,z.prior_touches) and np.isclose(q.iloc[0].TIME_OF_POSS,z.prior_time_of_possession);assert z.selection_rule=='touches_first';f=out/'tracking_2024-25'/(z.team+'.source.json');snap=json.loads(f.read_text());assert snap['parameters']['Season']=='2024-25' and snap['parameters']['PtMeasureType']=='Possessions'
 assert abs(c.TIME_OF_POSS-c.TOUCHES*c.AVG_SEC_PER_TOUCH/60).le(c.TOUCHES*.005/60+.11).all()
 for z in d.itertuples():
  folder=out/'networks'/z.team;P=pd.read_csv(folder/'present_passing_matrix.csv',index_col=0);A=pd.read_csv(folder/'absent_passing_matrix.csv',index_col=0);D=pd.read_csv(folder/'synthetic_deleted_passing_matrix.csv',index_col=0);ids=P.index.astype(int).tolist();i=ids.index(z.star_id);pred=P.to_numpy().copy();pred[i]=0;pred[:,i]=0;assert np.array_equal(pred,D.to_numpy());assert A.loc[z.star_id].sum()+A[str(z.star_id)].sum()==0;T=P.to_numpy()+A.to_numpy();den=np.log2(np.count_nonzero(T.sum(0)+T.sum(1))-1)
  for M,h in [(P,z.with_hub_entropy),(D,z.deleted_entropy),(A,z.without_hub_entropy)]:
   q=M.to_numpy()/M.to_numpy().sum();assert np.isclose((entropy(q.ravel(),base=2)-entropy(q.sum(1),base=2))/den,h)
  if z.window_games==82:
   F=pd.read_csv(root/'2025-26/team_season_topics/networks'/z.team/'passing_matrix.csv',index_col=0);assert np.array_equal(T,F.to_numpy())
 sample=d[d.regression_eligible];assert len(sample)==15
 for z in pd.read_csv(out/'regression_statistics.csv').itertuples():
  y=sample.equal_total_entropy_change if z.measure=='equal_total' else sample.entropy_change;y=abs(y) if z.outcome=='absolute' else y;f=linregress(sample.prior_on_off_gap,y);assert np.isclose(f.rvalue,z.r) and np.isclose(f.slope,z.slope)
 assert b'19' in re.findall(rb'/Count\s+(\d+)',(out/'network_changes/ball_control_three_networks.pdf').read_bytes())
 r=dict(teams=30,strict_highest_touches_verified=True,possession_time_tie_break_verified=True,no_combined_score_used=True,time_of_possession_units_minutes_verified=True,actual_pairs=19,regression_teams=15,network_pages=19,entropy_independently_checked=True,regression_checked_against_scipy=True);(out/'verification.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
if __name__=='__main__':main()
