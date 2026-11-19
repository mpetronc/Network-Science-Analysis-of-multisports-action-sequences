"""Confirm empty outgoing profiles only with complete independent daily zero counts."""
from pathlib import Path
import json,sys
import pandas as pd
from nba_api.stats.endpoints import leaguedashptstats
from teamgraphs import NBAClient,Config,season_topics_daily_passing

def confirm(root,out,season):
 root=Path(root);out=Path(out);base=root/'star_dependency_study/game_cache';players=pd.read_csv(base/(season+'_player_base.csv.gz'),dtype={'GAME_ID':str});games=pd.read_csv(base/(season+'_team_base.csv.gz'));players.GAME_DATE=players.GAME_DATE.str[:10];games.GAME_DATE=games.GAME_DATE.str[:10];folder=out/'independent_daily_zero_checks';folder.mkdir(exist_ok=True);client=NBAClient(Config(season=season,cache_dir=folder/'cache',timeout=25,retries=2,min_delay=.5,max_delay=1));fixed=[]
 try:
  for file in (out/'passing_profiles').rglob('*.json'):
   info=json.loads(file.read_text())
   if info.get('complete') or info.get('error')!='Empty profile without independently verified zero':continue
   appearances=players[players.TEAM_ID.eq(info['team_id'])&players.PLAYER_ID.eq(info['player_id'])&players.GAME_DATE.between(info['start'],info['end'])];assert len(appearances)==info['expected_games'];proof=[];zero=True
   for date in sorted(appearances.GAME_DATE.unique()):
    target=folder/(date+'.csv')
    if target.exists():f=pd.read_csv(target)
    else:
     participation=players[players.GAME_DATE.eq(date)];assert not participation.PLAYER_ID.duplicated().any();actual=participation.set_index('PLAYER_ID').TEAM_ID.to_dict();expected=set(games.loc[games.GAME_DATE.eq(date),'TEAM_ID']);day=pd.Timestamp(date).strftime('%m/%d/%Y')
     f,snapshot=client.fetch(leaguedashptstats.LeagueDashPtStats,dict(season=season,season_type_all_star='Regular Season',player_or_team='Player',per_mode_simple='Totals',pt_measure_type='Passing',date_from_nullable=day,date_to_nullable=day),lambda p:season_topics_daily_passing(p,expected,actual));f.to_csv(target,index=False);target.with_suffix('.source.json').write_text(json.dumps(snapshot,indent=2))
    q=f[f.PLAYER_ID.eq(info['player_id'])&f.TEAM_ID.eq(info['team_id'])]
    if len(q)!=1 or q.PASSES_MADE.iloc[0]!=0:zero=False;break
    proof.append(str(target))
   if zero and len(proof)==len(appearances):info.update(complete=True,error='',zero_confirmation='Independent daily NBA Passing tracking has an explicit zero PASSES_MADE row for every official appearance in this window',daily_zero_proof=proof);file.write_text(json.dumps(info,indent=2));fixed.append((info['team'],info['player_id'],info['start']))
 finally:client.close()
 print('Independent daily zero confirmations',fixed,flush=True)
 return fixed
if __name__=='__main__':confirm(Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3])
