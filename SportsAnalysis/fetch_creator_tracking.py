from pathlib import Path
import argparse,json
from concurrent.futures import ThreadPoolExecutor,as_completed
import pandas as pd,numpy as np
from nba_api.stats.endpoints import leaguedashptstats
from nba_api.stats.static import teams
from teamgraphs import NBAClient,Config,extract_team_metrics

def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'nba_passing_output');p.add_argument('--destination',type=Path,default=Path(__file__).resolve().parent/'nba_passing_output/creator_dependency');a=p.parse_args(argv);root=a.output_dir;out=a.destination;out.mkdir(parents=True,exist_ok=True)
 def one(t):
  file=out/'tracking_2024-25'/(t['abbreviation']+'.csv');file.parent.mkdir(exist_ok=True)
  if file.exists():return dict(team=t['abbreviation'],status='cached',rows=len(pd.read_csv(file)))
  c=NBAClient(Config(season='2024-25',cache_dir=out/'tracking_cache',timeout=25,retries=2,min_delay=.5,max_delay=1))
  def validate(payload):
   f=extract_team_metrics(payload,'LeagueDashPtStats',{'PLAYER_ID','PLAYER_NAME','TEAM_ID','GP','MIN','POTENTIAL_AST','PASSES_MADE'});assert len(f)>0 and not f.PLAYER_ID.duplicated().any();roster=pd.read_csv(root/'full_roster_entropy_centrality/historical_full_rosters/2024-25'/t['abbreviation']/'roster.csv').set_index('PLAYER_ID');assert set(f.PLAYER_ID)<=set(roster.index);official=roster.reindex(f.PLAYER_ID);assert np.all(f.GP.to_numpy()<=official.GP.to_numpy());assert np.all(f.MIN.to_numpy()<=official.MIN.to_numpy()+1.01);f['REPORTED_TEAM_ID']=f.TEAM_ID;f['TEAM_ID']=t['id'];f['official_minutes']=official.MIN.to_numpy();f['minute_coverage_fraction']=np.divide(f.MIN.to_numpy(),official.MIN.to_numpy(),out=np.zeros(len(f)),where=official.MIN.to_numpy()>0);assert np.isfinite(f[['MIN','GP','POTENTIAL_AST','PASSES_MADE']]).all().all();assert f[['MIN','GP','POTENTIAL_AST','PASSES_MADE']].ge(0).all().all();assert f.POTENTIAL_AST.le(f.PASSES_MADE).all();return f
  try:
   f,snapshot=c.fetch(leaguedashptstats.LeagueDashPtStats,dict(season='2024-25',season_type_all_star='Regular Season',player_or_team='Player',pt_measure_type='Passing',per_mode_simple='Totals',team_id_nullable=t['id']),validate);f.to_csv(file,index=False);file.with_suffix('.source.json').write_text(json.dumps(snapshot));return dict(team=t['abbreviation'],status='complete',rows=len(f))
  except Exception as exc:return dict(team=t['abbreviation'],status='failed',error=repr(exc))
  finally:c.close()
 records=[]
 with ThreadPoolExecutor(max_workers=3) as pool:
  for f in as_completed([pool.submit(one,t) for t in teams.get_teams()]):records.append(f.result());pd.DataFrame(records).to_csv(out/'tracking_download_progress.csv',index=False);print(records[-1],flush=True)
if __name__=='__main__':main()
