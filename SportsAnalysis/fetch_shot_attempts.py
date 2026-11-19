from pathlib import Path
import pandas as pd,json,time
from nba_api.stats.endpoints import shotchartdetail
from nba_api.stats.static import teams
import argparse
parser=argparse.ArgumentParser(description='Download complete NBA FGA records using separate team requests; league-wide responses can truncate.')
parser.add_argument('--cache',type=Path,default=Path(__file__).resolve().parent/'nba_passing_output/shot_attempt_cache')
parser.add_argument('--seasons',nargs='+',default=['2020-21','2021-22','2022-23','2023-24','2024-25','2025-26'])
args=parser.parse_args();root=args.cache.expanduser();root.mkdir(parents=True,exist_ok=True)
for season in args.seasons:
 parts=[]
 for team in teams.get_teams():
  p=root/f'{season}_{team["abbreviation"]}.csv.gz'
  if not p.exists():
   for retry in range(3):
    try:
     q=shotchartdetail.ShotChartDetail(team_id=team['id'],player_id=0,season_nullable=season,season_type_all_star='Regular Season',context_measure_simple='FGA',timeout=35)
     df=q.get_data_frames()[0]
     assert len(df)>1000 and df.SHOT_ATTEMPTED_FLAG.eq(1).all()
     df.to_csv(p,index=False,compression='gzip');break
    except Exception as e:
     if retry==2: raise
     time.sleep(2)
   print(season,team['abbreviation'],len(df),flush=True)
  parts.append(pd.read_csv(p));time.sleep(.15)
 frame=pd.concat(parts,ignore_index=True)
 frame.to_csv(root/f'{season}_shot_attempts.csv.gz',index=False,compression='gzip')
 (root/f'{season}_source.json').write_text(json.dumps(dict(endpoint='https://stats.nba.com/stats/shotchartdetail',season=season,context_measure='FGA',season_type='Regular Season',scope='30 separate TeamID requests',player_id=0,team_ids=[t['id'] for t in teams.get_teams()],rows=len(frame),fetched_utc=pd.Timestamp.now(tz='UTC').isoformat()),indent=2))
