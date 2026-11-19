"""Download official regular-season team/player game logs for a prior-defined hub study."""
from pathlib import Path
import argparse,time,json,hashlib
import pandas as pd
from nba_api.stats.endpoints import leaguegamelog,teamgamelogs

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache',type=Path,default=Path(__file__).resolve().parent/'nba_passing_output/star_dependency_study/game_cache');p.add_argument('--existing-root',type=Path,default=Path(__file__).resolve().parent/'nba_passing_output');a=p.parse_args();root=a.cache;root.mkdir(parents=True,exist_ok=True);manifest=[]
 for season in ['2020-21','2021-22','2022-23','2023-24','2024-25','2025-26']:
  for kind in ['team_base','team_advanced','player_base']:
   dest=root/f'{season}_{kind}.csv.gz';meta=root/f'{season}_{kind}_source.json'
   if dest.exists() and meta.exists():
    manifest.append(json.loads(meta.read_text()));print('Cached',season,kind,flush=True);continue
   existing=None
   if season=='2025-26':
    existing=a.existing_root/'2025-26/team_season_topics'/({'team_base':'team_games.csv','player_base':'player_games.csv','team_advanced':'hub_absence_resilience/official_game_advanced.csv'}[kind])
   if existing and existing.exists():
    frame=pd.read_csv(existing,dtype={'GAME_ID':str});record=dict(season=season,kind=kind,source='Previously downloaded NBA stats cache',source_path=str(existing),source_sha256=hashlib.sha256(existing.read_bytes()).hexdigest())
   else:
    for retry in range(3):
     try:
      if kind=='team_advanced':q=teamgamelogs.TeamGameLogs(season_nullable=season,season_type_nullable='Regular Season',measure_type_player_game_logs_nullable='Advanced',per_mode_simple_nullable='Totals',timeout=35)
      else:q=leaguegamelog.LeagueGameLog(season=season,season_type_all_star='Regular Season',player_or_team_abbreviation='P' if kind=='player_base' else 'T',timeout=35)
      frame=q.get_data_frames()[0];record=dict(season=season,kind=kind,endpoint=q.endpoint,parameters=q.parameters,fetched_utc=pd.Timestamp.now(tz='UTC').isoformat());break
     except Exception:
      if retry==2:raise
      time.sleep(2)
   expected_games=1080 if season=='2020-21' else 1230
   assert frame.TEAM_ID.nunique()==30 and frame.GAME_ID.nunique()==expected_games,(season,kind,len(frame),frame.GAME_ID.nunique())
   unique=['TEAM_ID','GAME_ID']+(['PLAYER_ID'] if kind=='player_base' else []);assert not frame.duplicated(unique).any()
   if kind=='team_advanced':assert {'POSS','OFF_RATING'}<=set(frame) and frame.POSS.gt(0).all()
   if kind!='player_base':assert len(frame)==2*expected_games
   frame.to_csv(dest,index=False,compression={'method':'gzip','mtime':0});record.update(rows=len(frame),games=expected_games,sha256=hashlib.sha256(dest.read_bytes()).hexdigest());meta.write_text(json.dumps(record,indent=2));manifest.append(record);print(season,kind,len(frame),'rows',flush=True);time.sleep(.3)
 (root/'download_manifest.json').write_text(json.dumps(manifest,indent=2));print('All six seasons complete',flush=True)
if __name__=='__main__':main()
