from pathlib import Path
import argparse,json,hashlib,re
import pandas as pd,numpy as np
from scipy.stats import entropy,linregress

def conditional(M):
 q=M/M.sum();rows=q.sum(1);return float(entropy(q.ravel(),base=2)-entropy(rows,base=2))
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'nba_passing_output');p.add_argument('--data-dir',type=Path);a=p.parse_args(argv);root=a.output_dir;out=a.data_dir or root/'onoff_dependency';d=pd.read_csv(out/'network_metrics.csv');candidates=pd.read_csv(out/'all_prior_candidates.csv');assert len(d)==30 and d.team.nunique()==30;profilemanifest=[]
 for f in (out/'passing_profiles').glob('*/*/*.json'):
  z=json.loads(f.read_text());assert z['complete'],str(f);csv=f.with_suffix('.csv');profilemanifest.append(dict(path=str(csv),sha256=hashlib.sha256(csv.read_bytes()).hexdigest(),audit_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),receiver_side_confirmation='receiver_profiles' in z))
 for z in d.itertuples():
  q=candidates[candidates.team.eq(z.team)&candidates.eligible_minutes&candidates.retained_first10];assert np.isclose(z.prior_on_off_gap,q.gap.max());assert z.prior_on_minutes>=1000 and z.prior_off_minutes>=500
  folder=out/'networks'/z.team;P=pd.read_csv(folder/'present_passing_matrix.csv',index_col=0);A=pd.read_csv(folder/'absent_passing_matrix.csv',index_col=0);assert list(P.index)==list(A.index) and list(P.columns)==list(A.columns);T=P.to_numpy()+A.to_numpy();n=np.count_nonzero(T.sum(0)+T.sum(1));den=np.log2(n-1);assert np.isclose(conditional(P.to_numpy())/den,z.with_hub_entropy);assert np.isclose(conditional(A.to_numpy())/den,z.without_hub_entropy);assert np.isclose(z.entropy_change,z.without_hub_entropy-z.with_hub_entropy)
  assert z.present_games+z.absent_games==z.window_games;assert A.loc[z.star_id].sum()+A[str(z.star_id)].sum()==0
  if z.window_games==82:
   full=pd.read_csv(root/'2025-26/team_season_topics/networks'/z.team/'passing_matrix.csv',index_col=0);assert np.array_equal(full.to_numpy(),T)
  else:assert z.window_end<z.trade_censor_date
  U=pd.read_csv(out/'network_changes'/z.team/'present_per_1000_passes.csv',index_col=0).to_numpy();V=pd.read_csv(out/'network_changes'/z.team/'absent_per_1000_passes.csv',index_col=0).to_numpy();assert np.isclose(U.sum(),1000) and np.isclose(V.sum(),1000)
 plan=pd.read_csv(out/'network_download_plan.csv')
 for team,group in plan.groupby('team'):
  folder=out/'networks'/team;F=pd.read_csv(folder/'present_passing_matrix.csv',index_col=0);A=pd.read_csv(folder/'absent_passing_matrix.csv',index_col=0);ids=F.index.astype(int).tolist();ix={pid:i for i,pid in enumerate(ids)};reconstructed={kind:np.zeros(F.shape) for kind in group.kind.unique()}
  for r in group.itertuples():
   raw=pd.read_csv(out/'passing_profiles'/team/(r.start+'_'+r.end)/(str(r.player_id)+'.csv'))
   for edge in raw.itertuples():
    if edge.PASS>0:reconstructed[r.kind][ix[r.player_id],ix[int(edge.PASS_TEAMMATE_PLAYER_ID)]]+=edge.PASS
  assert np.array_equal(reconstructed['absent'],A.to_numpy())
  if 'total' in reconstructed:assert np.array_equal(reconstructed['total'],F.to_numpy()+A.to_numpy())
 sample=d[d.regression_eligible];s=pd.read_csv(out/'regression_statistics.csv');assert len(s)==4 and s.n.eq(len(sample)).all() and sample.present_games.ge(15).all() and sample.absent_games.ge(10).all()
 for z in s.itertuples():
  y=sample.equal_total_entropy_change.to_numpy() if z.measure=='equal_total' else sample.entropy_change.to_numpy();y=abs(y) if z.outcome=='absolute' else y;f=linregress(sample.prior_on_off_gap,y);assert np.isclose(f.slope,z.slope) and np.isclose(f.rvalue,z.r) and np.isclose(f.pvalue,z.pearson_p)
 games=pd.read_csv(out/'game_comparison_data.csv')
 for z in d.itertuples():
  g=games[games.TEAM_ID.eq(z.team_id)];on=g[g.star_played];off=g[~g.star_played];assert len(on)==z.present_games and len(off)==z.absent_games;assert np.isclose(100*on.PTS.sum()/on.POSS.sum(),z.game_played_ortg) and np.isclose(100*off.PTS.sum()/off.POSS.sum(),z.game_missed_ortg)
 for file,n in [(out/'onoff_dependency_summary.pdf',4),(out/'network_changes/onoff_selected_star_networks.pdf',30)]:assert str(n).encode() in re.findall(rb'/Count\s+(\d+)',file.read_bytes())
 assert len(list((out/'onoff').glob('*/*.csv')))==60;pd.DataFrame(profilemanifest).to_csv(out/'profile_source_manifest.csv',index=False)
 result=dict(teams=30,regression_teams=len(sample),profiles_verified=len(profilemanifest),summary_pages=4,network_pages=30,previous_season_selection_verified=True,independent_entropy_calculation_verified=True,full_season_partitions_verified=True,played_missed_rating_aggregation_verified=True,regressions_checked_against_scipy=True);(out/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
