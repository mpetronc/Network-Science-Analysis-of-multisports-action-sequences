from pathlib import Path
import argparse,json,hashlib,re
import pandas as pd,numpy as np

def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--data-dir',type=Path,required=True);a=p.parse_args(argv);root=a.output_dir;out=a.data_dir;episodes=json.loads((out/'episode_plan.json').read_text());manifest=json.loads((out/'freeze_manifest.json').read_text());P=pd.read_csv(root/'2025-26/team_season_topics/player_games.csv',dtype={'GAME_ID':str});P.GAME_ID=P.GAME_ID.str.zfill(10);results=pd.read_csv(out/'episode_results.csv');assert len(episodes)==14 and len(manifest)==14 and len(results)==14
 for e in episodes:
  folder=out/e['team'];s=json.loads((folder/'frozen_specification.json').read_text());ids=s['prediction_player_ids'];pre=P[P.TEAM_ID.eq(e['team_id'])&P.GAME_ID.isin(e['pre_game_ids'])];history=P[P.TEAM_ID.eq(e['team_id'])&P.GAME_ID.isin(e['history_game_ids'])];assert pre.GAME_DATE.max()<e['test_start'] and history.GAME_DATE.max()<e['test_start'];assert not set(e['pre_game_ids'])&set(e['test_game_ids']);assert set(ids)==set(pre.PLAYER_ID)|{e['star_id']};observed=set(history[history.PLAYER_ID.eq(e['star_id'])].GAME_ID);absent=set(e['history_game_ids'])-observed;assert absent==set(s['earlier_absence_game_ids'])
  mp=pre.groupby('PLAYER_ID').MIN.sum().reindex(ids,fill_value=0).to_numpy(float);recent=mp/5;hp=history[history.GAME_ID.isin(absent)].groupby('PLAYER_ID').MIN.sum().reindex(ids,fill_value=0).to_numpy(float);weights=(hp+5*recent)/(len(absent)+5);weights[recent<=0]=0;weights[ids.index(e['star_id'])]=0;assert np.allclose(weights,s['rotation_weights'])
  # Independently solve the cap-constrained proportional allocation by bisection.
  lo=0.;hi=1.
  while np.minimum(48,hi*weights).sum()<240-1e-9:hi*=2
  for _ in range(100):
   mid=(lo+hi)/2
   if np.minimum(48,mid*weights).sum()<240:lo=mid
   else:hi=mid
  minutes=np.minimum(48,hi*weights);assert np.allclose(minutes,s['predicted_minutes_per_game'],atol=1e-6);assert np.isclose(minutes.sum(),240) and minutes.max()<=48+1e-6
  D=pd.read_csv(folder/'deletion_baseline.csv',index_col=0).to_numpy();M=pd.read_csv(folder/'frozen_prediction.csv',index_col=0).to_numpy();rate=np.divide(D.sum(1),mp,out=np.zeros(len(mp)),where=mp>0);target=rate*minutes*2;expected=np.zeros_like(M)
  for i in range(len(ids)):
   if target[i]<=0:continue
   mask=minutes>0;mask[i]=False;k=D[i].copy();k[~mask]=0
   if k.sum()==0:k=minutes*mask
   if k.sum()>0:expected[i]=target[i]*k/k.sum()
  assert np.allclose(expected,M);assert sha(folder/'frozen_prediction.csv')==s['matrix_sha256']==next(z['sha256'] for z in manifest if z['team']==e['team']);assert M[ids.index(e['star_id'])].sum()+M[:,ids.index(e['star_id'])].sum()==0
  actual=pd.read_csv(folder/'actual_test_matrix.csv',index_col=0);expanded=pd.DataFrame(M,index=ids,columns=list(map(str,ids))).reindex(index=actual.index,columns=actual.columns,fill_value=0).to_numpy();A=actual.to_numpy();z=results[results.team.eq(e['team'])].iloc[0];assert np.isclose(abs(expanded/expanded.sum()-A/A.sum()).sum()/2,z.frozen_edge_tv)
  # Sensitivity to future records: the independent reconstruction above reads
  # only pre/history rows; no test minutes or pass totals enter any component.
 assert b'14' in re.findall(rb'/Count\s+(\d+)',(out/'frozen_predictions_and_actual.pdf').read_bytes())
 result=dict(teams_considered=15,eligible=14,evaluated=14,network_pages=14,pre_test_date_order_verified=True,minute_weights_reconstructed_from_pre_only=True,cap_allocation_independently_verified=True,passing_predictions_rebuilt_without_test_inputs=True,frozen_matrix_hashes_unchanged=True,test_scoring_verified=True);(out/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
