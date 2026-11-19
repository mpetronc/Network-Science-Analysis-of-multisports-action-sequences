"""Independent source, reference-population and frozen-model integrity checks."""
from pathlib import Path
import argparse,itertools
import numpy as np,pandas as pd

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('destination',type=Path);p.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'nba_passing_output');p.add_argument('--position-file',type=Path,default=Path(__file__).resolve().parent/'position_network_metadata/player_positions.csv');a=p.parse_args(argv);out=a.destination;root=a.output_dir
 d=pd.read_csv(out/'revised_metrics.csv');clean=pd.read_csv(out/'new_data.csv');players=pd.read_csv(out/'position_shooting_reference.csv');results=pd.read_csv(out/'pooled_regressions.csv');temp=pd.read_csv(out/'temporal_checks.csv');ref=pd.read_csv(out/'development_position_zone_counts.csv',index_col=0);pos=pd.read_csv(a.position_file).set_index(['season','team','player_id']).position
 assert len(d)==len(clean)==180 and not d.duplicated(['team','season']).any() and len(results)==6 and len(temp)==6
 assert d.season.nunique()==6 and d.groupby('season').size().eq(30).all()
 assert np.isfinite(d.select_dtypes('number')).all().all() and np.isfinite(clean.select_dtypes('number')).all().all()
 assert np.allclose(d.passing_entropy_excess,d.passing_entropy-d.null_entropy_mean) and np.allclose(d.passing_ec_excess,d.passing_starness-d.null_ec_mean,atol=1e-8)
 expected={r:np.zeros(len(ref)) for r in ref};own={};weighted=[]
 for z in d.itertuples():
  folder=root/'full_roster_entropy_centrality/historical_full_rosters'/z.season/z.team;A=pd.read_csv(folder/'passing_matrix.csv',index_col=0);R=pd.read_csv(folder/'roster.csv').set_index('PLAYER_ID').loc[A.index.astype(int)];C=pd.read_csv(root/'court_zone_networks'/z.season/z.team/'player_zone_attempts.csv',index_col=0).reindex(index=A.index.astype(int),columns=ref.index)
  F=C.sum(axis=1).to_numpy();assert np.array_equal(F,R.FGA.to_numpy())
  assert np.isclose(np.sum(F*(F-1))/(F.sum()*(F.sum()-1)),z.shooting_hhi_unbiased)
  active=A.to_numpy().sum(0)+A.to_numpy().sum(1)>0;B=A.to_numpy()[np.ix_(active,active)];_,V=np.linalg.eigh(B+B.T);v=np.abs(V[:,-1]);v/=v.sum();ec=(v@v-1/len(v))/(1-1/len(v));assert np.isclose(ec,z.passing_starness,atol=1e-8)
  if z.season<='2024-25':
   for pid,c in zip(C.index,C.to_numpy()):
    role=pos.loc[(z.season,z.team,pid)];expected[role]+=c;own[(int(pid),role)]=own.get((int(pid),role),0)+float(c.sum())
  minute=R.MIN.to_numpy();assert np.isclose(np.sum((minute/minute.sum())**2),z.minutes_hhi)
  g=players[(players.team==z.team)&(players.season==z.season)];assert int(g.fga.sum())==int(F.sum())
  value=np.average(g.entropy_excess,weights=g.fga);assert np.isclose(value,z.shooting_entropy_role_excess)
 assert np.array_equal(ref.to_numpy(),pd.DataFrame(expected,index=ref.index).to_numpy())
 for z in players.itertuples():assert np.isclose(z.peer_reference_shots,ref[z.position].sum()-own.get((int(z.player_id),z.position),0))
 train=d[d.season<='2024-25'];test=d[d.season.eq('2025-26')];assert len(train)==150 and len(test)==30
 for z in temp.itertuples():
  b=np.polyfit(train[z.x],train[z.y],1);pred=np.polyval(b,test[z.x]);err=np.sum((test[z.y]-pred)**2);assert np.isclose(1-err/np.sum((test[z.y]-test[z.y].mean())**2),z.predictive_r_squared_vs_test_mean)
 for z in results.itertuples():assert np.isclose(np.corrcoef(d[z.x],d[z.outcome])[0,1],z.r)
 labels=[0,0,0,1,1,2];exact=np.mean([i==j for i,j in itertools.combinations(labels,2)]);counts=np.array([3,2,1]);assert np.isclose(exact,np.sum(counts*(counts-1))/(len(labels)*(len(labels)-1)))
 for file,n in [('revised_analysis.pdf',3),('primary_comparison.pdf',1),('absence_present_only.pdf',1)]:assert f'/Count {n}'.encode() in (out/file).read_bytes()
 report=dict(team_seasons=180,development_team_seasons=150,temporal_test_team_seasons=30,position_training_reference_excludes_test=True,focal_player_history_excluded_from_position_reference=True,EC_independently_recomputed_from_passing_only=True,HHI_matches_FGA_pairs=True,frozen_predictions_reproduced=True,primary_correlations_reproduced=True,all_scores_finite=True)
 import json
 (out/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
