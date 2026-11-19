from pathlib import Path
import argparse,json
from concurrent.futures import ThreadPoolExecutor,as_completed
import pandas as pd,numpy as np
from nba_api.stats.endpoints import teamplayeronoffsummary
from nba_api.stats.static import teams
from teamgraphs import NBAClient,Config,extract_team_metrics

def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--destination',type=Path,default=Path(__file__).resolve().parent/'nba_passing_output/onoff_dependency');a=p.parse_args(argv);out=a.destination;out.mkdir(parents=True,exist_ok=True)
 def one(season,team):
  file=out/'onoff'/season/(team['abbreviation']+'.csv');file.parent.mkdir(parents=True,exist_ok=True)
  if file.exists():return dict(season=season,team=team['abbreviation'],rows=len(pd.read_csv(file)),status='cached')
  client=NBAClient(Config(season=season,cache_dir=out/'api_cache',timeout=25,retries=2,min_delay=.5,max_delay=1))
  def validate(payload):
   parts=[]
   for status,name in [('on','PlayersOnCourtTeamPlayerOnOffSummary'),('off','PlayersOffCourtTeamPlayerOnOffSummary')]:
    f=extract_team_metrics(payload,name,{'VS_PLAYER_ID','VS_PLAYER_NAME','MIN','OFF_RATING','TEAM_ID','GP'});assert not f.empty and f.TEAM_ID.eq(team['id']).all() and not f.VS_PLAYER_ID.duplicated().any();assert np.isfinite(f[['MIN','OFF_RATING','GP']]).all().all();assert f.MIN.ge(0).all() and f.OFF_RATING.ge(0).all();f['condition']=status;parts.append(f)
   return pd.concat(parts,ignore_index=True)
  try:
   f,snap=client.fetch(teamplayeronoffsummary.TeamPlayerOnOffSummary,dict(team_id=team['id'],season=season,season_type_all_star='Regular Season',per_mode_detailed='Totals',measure_type_detailed_defense='Advanced'),validate);f.to_csv(file,index=False);file.with_suffix('.source.json').write_text(json.dumps(snap));return dict(season=season,team=team['abbreviation'],rows=len(f),status='complete')
  except Exception as e:return dict(season=season,team=team['abbreviation'],status='failed',error=repr(e))
  finally:client.close()
 records=[]
 with ThreadPoolExecutor(max_workers=2) as pool:
  futures=[pool.submit(one,s,t) for s in ['2024-25','2025-26'] for t in teams.get_teams()]
  for f in as_completed(futures):
   r=f.result();records.append(r);pd.DataFrame(records).to_csv(out/'onoff_download_progress.csv',index=False);print(r,flush=True)
 return 0
if __name__=='__main__':raise SystemExit(main())
