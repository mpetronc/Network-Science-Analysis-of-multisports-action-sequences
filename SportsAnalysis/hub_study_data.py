"""Audited dated passing profiles and prior-season roster positions."""
from pathlib import Path
import json,shutil
from concurrent.futures import ThreadPoolExecutor,as_completed
import pandas as pd,numpy as np
from nba_api.stats.endpoints import playerdashptpass,commonteamroster
from nba_api.stats.static import teams
from teamgraphs import NBAClient,Config,validate_passing,extract_team_metrics
from corroborate_empty_profiles import main as corroborate

def atomic_csv(frame,file):
 temp=file.with_suffix('.tmp.csv');frame.to_csv(temp,index=False);temp.replace(file)
class StudyData:
 def __init__(self,root,out):
  self.root=Path(root);self.out=Path(out);self.out.mkdir(parents=True,exist_ok=True);base=self.root/'2025-26/team_season_topics';self.players=pd.read_csv(base/'player_games.csv',dtype={'GAME_ID':str});self.games=pd.read_csv(base/'team_games.csv',dtype={'GAME_ID':str});self.players.GAME_ID=self.players.GAME_ID.str.zfill(10);self.games.GAME_ID=self.games.GAME_ID.str.zfill(10);self.games=self.games.sort_values(['GAME_DATE','GAME_ID']);self.daily=pd.concat([pd.read_csv(f).assign(date=f.stem) for f in (base/'daily_passing').glob('*.csv')],ignore_index=True);self.plan=[];self.cache_sources=[self.root/'onoff_dependency/passing_api_cache',self.root/'onoff_dependency/prospective_absence/passing_api_cache',self.root/'creator_dependency/passing_api_cache',self.root/'creator_dependency/prospective_absence/passing_api_cache',self.root/'ball_control_dependency/passing_api_cache',self.root/'ball_control_dependency/prospective_absence/passing_api_cache',self.root/'league_star_presence_entropy/api_cache',base/'hub_absence_resilience/cache']
 def add_window(self,team,tid,kind,games):
  if len(games)==0:return
  start=games.GAME_DATE.min();end=games.GAME_DATE.max();players=self.players[self.players.TEAM_ID.eq(tid)&self.players.GAME_ID.isin(games.GAME_ID)]
  for pid,part in players.groupby('PLAYER_ID'):
   d=self.daily[self.daily.TEAM_ID.eq(tid)&self.daily.PLAYER_ID.eq(pid)&self.daily.date.between(start,end)];self.plan.append(dict(team=team,team_id=int(tid),kind=kind,player_id=int(pid),start=start,end=end,expected_games=len(part),daily_observed_games=len(d),daily_passes=float(d.PASSES_MADE.sum())))
 def file(self,r):return self.out/'passing_profiles'/r['team']/(r['start']+'_'+r['end'])/(str(r['player_id'])+'.csv')
 def download(self,kinds):
  columns=['team','team_id','kind','player_id','start','end','expected_games','daily_observed_games','daily_passes'];atomic_csv(pd.DataFrame(self.plan,columns=columns),self.out/'network_download_plan.csv');tasks=[r for r in self.plan if r['kind'] in kinds];records=[]
  def one(r):
   file=self.file(r);file.parent.mkdir(parents=True,exist_ok=True);audit=file.with_suffix('.json')
   if file.exists() and audit.exists():
    info=json.loads(audit.read_text())
    if info.get('complete'):return dict(r,complete=True,error='')
   c=NBAClient(Config(season='2025-26',cache_dir=self.out/'passing_api_cache',timeout=25,retries=2,min_delay=.5,max_delay=1));params=dict(player_id=r['player_id'],team_id=r['team_id'],season='2025-26',season_type_all_star='Regular Season',per_mode_simple='Totals',date_from_nullable=pd.Timestamp(r['start']).strftime('%m/%d/%Y'),date_to_nullable=pd.Timestamp(r['end']).strftime('%m/%d/%Y'))
   def valid(payload):
    f=validate_passing(payload,r['player_id'],r['team_id'])
    if len(f) and f.G.gt(r['expected_games']).any():raise ValueError('Date-filtered GP exceeds official window appearances')
    return f
   try:
    query=playerdashptpass.PlayerDashPtPass(**params,get_request=False);target=c.cache.path_for(query.endpoint,dict(query.parameters))
    if not target.exists():
     for folder in self.cache_sources:
      source=folder/target.name
      if source.exists():target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target);break
    f,_=c.fetch(playerdashptpass.PlayerDashPtPass,params,valid);zero=r['daily_observed_games']==r['expected_games'] and r['daily_passes']==0;complete=len(f)>0 or zero;error='' if complete else 'Empty profile without independently verified zero'
   except Exception as exc:f=pd.DataFrame(columns=['PASS_TEAMMATE_PLAYER_ID','PASS']);complete=False;error=repr(exc)
   finally:c.close()
   f.to_csv(file,index=False);info=dict(r,complete=complete,error=error,profile_passes=float(f.PASS.sum()));audit.write_text(json.dumps(info,indent=2));return info
  with ThreadPoolExecutor(max_workers=6) as pool:
   for i,f in enumerate(as_completed([pool.submit(one,r) for r in tasks]),1):
    records.append(f.result())
    if i%50==0 or i==len(tasks):atomic_csv(pd.DataFrame(records),self.out/('progress_'+'_'.join(sorted(kinds))+'.csv'));print('/'.join(kinds),i,'/',len(tasks),'complete',sum(r['complete'] for r in records),flush=True)
  if tasks:corroborate(['--data-dir',str(self.out)])
 def matrix(self,team,kind,ids):
  M=np.zeros((len(ids),len(ids)));ix={pid:i for i,pid in enumerate(ids)}
  for r in self.plan:
   if r['team']!=team or r['kind']!=kind:continue
   f=self.file(r);audit=json.loads(f.with_suffix('.json').read_text());assert audit['complete'],str(f)
   for edge in pd.read_csv(f).itertuples():
    if edge.PASS>0:assert int(edge.PASS_TEAMMATE_PLAYER_ID) in ix,(team,'receiver outside roster',edge.PASS_TEAMMATE_PLAYER_ID);M[ix[r['player_id']],ix[int(edge.PASS_TEAMMATE_PLAYER_ID)]]+=edge.PASS
  assert np.isfinite(M).all() and (M>=0).all() and not np.diag(M).any();return M
 def positions(self):
  folder=self.out/'prior_roster_positions';folder.mkdir(exist_ok=True);records=[]
  def one(t):
   file=folder/(t['abbreviation']+'.csv')
   if file.exists():return pd.read_csv(file)
   c=NBAClient(Config(season='2024-25',cache_dir=self.out/'position_cache',timeout=25,retries=2,min_delay=.5,max_delay=1))
   try:
    def valid(payload):
     f=extract_team_metrics(payload,'CommonTeamRoster',{'TeamID','PLAYER_ID','PLAYER','POSITION'});assert not f.empty and f.TeamID.eq(t['id']).all();return f
    f,snap=c.fetch(commonteamroster.CommonTeamRoster,dict(team_id=t['id'],season='2024-25'),valid);f.to_csv(file,index=False);file.with_suffix('.source.json').write_text(json.dumps(snap));return f
   finally:c.close()
  with ThreadPoolExecutor(max_workers=3) as pool:
   for f in as_completed([pool.submit(one,t) for t in teams.get_teams()]):records.append(f.result())
  frame=pd.concat(records,ignore_index=True);positions={}
  for pid,part in frame.groupby('PLAYER_ID'):
   value=' '.join(part.POSITION.fillna('').astype(str));value=value.replace('Guard','G').replace('Forward','F').replace('Center','C');positions[int(pid)]=sorted(set(value)&set('GFC'))
  (self.out/'prior_position_map.json').write_text(json.dumps(positions,indent=2));return positions

def first20(root,out):
 data=StudyData(root,out)
 for t in teams.get_teams():
  g=data.games[data.games.TEAM_ID.eq(t['id'])].head(20);assert len(g)==20;data.add_window(t['abbreviation'],t['id'],'first20',g)
 data.download({'first20'});positions=data.positions();return data,positions
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--destination',type=Path,required=True);a=p.parse_args();first20(a.output_dir,a.destination)
