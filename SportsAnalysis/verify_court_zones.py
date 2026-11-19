"""Verify exported season court-zone graphs against original pass/shot totals."""
from pathlib import Path
import argparse
import numpy as np,pandas as pd
from court_zone_networks import ZONES

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('destination',type=Path);p.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'nba_passing_output');a=p.parse_args();rows=[]
 for folder in sorted(a.destination.glob('20*/*')):
  if not (folder/'edges.csv').exists():continue
  team,season=folder.name,folder.parent.name;original=a.output_dir/'full_roster_entropy_centrality/historical_full_rosters'/season/team
  matrix=pd.read_csv(original/'passing_matrix.csv',index_col=0);ids=matrix.index.astype(int).tolist();r=pd.read_csv(original/'roster.csv').set_index('PLAYER_ID').loc[ids];edges=pd.read_csv(folder/'edges.csv');passes=edges[edges.edge_type.eq('pass')].copy();shots=edges[edges.edge_type.eq('shot_attempt')].copy()
  passes['source']=passes.source.astype(int);passes['target']=passes.target.astype(int)
  recovered=passes.pivot(index='source',columns='target',values='raw_count').reindex(index=ids,columns=ids).fillna(0).to_numpy(float)
  assert np.array_equal(recovered,matrix.to_numpy(float))
  assert set(shots.target)<=set(ZONES) and not edges.source.astype(str).isin(ZONES).any()
  assert np.array_equal(shots.weight,shots.raw_count) and np.array_equal(passes.weight,passes.raw_count)
  assert not (edges.source.astype(str)==edges.target.astype(str)).any()
  shots['source']=shots.source.astype(int)
  assert np.array_equal(shots.groupby('source').raw_count.sum().reindex(ids,fill_value=0),r.FGA)
  assert np.array_equal(shots.groupby('source').made.sum().reindex(ids,fill_value=0),r.FGM)
  counts=pd.read_csv(folder/'player_zone_attempts.csv',index_col=0);makes=pd.read_csv(folder/'player_zone_makes.csv',index_col=0)
  assert list(counts.columns)==list(ZONES) and np.array_equal(counts.sum(axis=1).to_numpy(),r.FGA)
  assert np.array_equal(makes.sum(axis=1).to_numpy(),r.FGM) and (makes.to_numpy()<=counts.to_numpy()).all()
  for file in ['network.png','network.pdf','court_zone_map.png','court_zone_map.pdf']:assert (folder/file).stat().st_size>1000
  rows.append(dict(team=team,season=season,passes_unchanged=True,FGA_FGM_preserved=True,only_zone_sinks=True,raw_edge_weights=True,no_self_loops=True))
 assert len(rows)==180,f'Expected 180 complete graph sets, got {len(rows)}'
 data=(a.destination/'all_court_zone_networks.pdf').read_bytes();assert b'/Count 180' in data
 pd.DataFrame(rows).to_csv(a.destination/'independent_integrity_audit.csv',index=False)
 print('Verified all 180 graph sets and PDF pages: unchanged passes, complete FGA/FGM, raw weights, only zone sinks, no self-loops.')
if __name__=='__main__':main()
