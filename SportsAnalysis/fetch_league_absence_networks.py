"""Complete missing 2025–26 descriptive EC-hub absence networks; reuse validated cached teams."""
from pathlib import Path
import argparse,json,time
from concurrent.futures import ThreadPoolExecutor,as_completed
import pandas as pd,numpy as np
from nba_api.stats.endpoints import playerdashptpass
from nba_api.stats.static import teams
from teamgraphs import NBAClient,Config,validate_passing,DataUnavailable

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'nba_passing_output');p.add_argument('--destination',type=Path);a=p.parse_args(argv);root=a.output_dir;out=a.destination or root/'league_star_presence_entropy';out.mkdir(parents=True,exist_ok=True);source=root/'2025-26/team_season_topics';legacy=source/'hub_absence_resilience';metrics=pd.read_csv(root/'court_zone_networks/regressions/four_metrics.csv');metrics=metrics[metrics.season.eq('2025-26')];P=pd.read_csv(source/'player_games.csv',dtype={'GAME_ID':str});G=pd.read_csv(source/'team_games.csv',dtype={'GAME_ID':str});G['date']=pd.to_datetime(G.GAME_DATE).dt.normalize();daily=pd.concat([pd.read_csv(f).assign(date=f.stem) for f in (source/'daily_passing').glob('*.csv')],ignore_index=True);mapping={t['abbreviation']:t['id'] for t in teams.get_teams()};plan=[];coverage=[]
 for z in metrics.sort_values('team').itertuples():
  team=z.team;tid=mapping[team];hub=int(z.passing_leader_id);games=G[G.TEAM_ID.eq(tid)].sort_values(['date','GAME_ID']).copy();present=set(P[P.TEAM_ID.eq(tid)&P.PLAYER_ID.eq(hub)].GAME_ID);games['present']=games.GAME_ID.isin(present);nabs=int((~games.present).sum());cache=legacy/'present_networks'/team/'passing_matrix.csv';audit=cache.with_name('network_audit.json');cached=False
  if cache.exists() and audit.exists():
   info=json.loads(audit.read_text());cached=info.get('coverage_complete',False) and int(info['hub_id'])==hub
  coverage.append(dict(team=team,team_id=tid,hub_id=hub,hub_name=z.passing_leader_name,present_games=len(present),absent_games=nabs,validated_legacy_network=cached,status='validated_existing_network' if cached else ('no_absence_games' if nabs==0 else 'download_required')))
  if cached or nabs==0:continue
  games['block']=games.present.ne(games.present.shift()).cumsum()
  for _,block in games[~games.present].groupby('block'):
   start=block.date.min().date().isoformat();end=block.date.max().date().isoformat()
   for pid,part in P[P.TEAM_ID.eq(tid)&P.GAME_ID.isin(block.GAME_ID)].groupby('PLAYER_ID'):
    observed=daily[daily.TEAM_ID.eq(tid)&daily.PLAYER_ID.eq(pid)&daily.date.between(start,end)]
    plan.append(dict(team=team,team_id=tid,hub_id=hub,player_id=int(pid),start=start,end=end,expected_games=len(part),daily_observed_games=len(observed),daily_passes=float(observed.PASSES_MADE.sum())))
 pd.DataFrame(coverage).to_csv(out/'coverage_plan.csv',index=False);pd.DataFrame(plan).to_csv(out/'download_plan.csv',index=False);print('Missing profiles:',len(plan),'for',len(set(r['team'] for r in plan)),'teams',flush=True)
 c=NBAClient(Config(season='2025-26',cache_dir=out/'api_cache',timeout=30,retries=2,min_delay=.5,max_delay=1))
 def one(r):
  folder=out/'absence_profiles'/r['team']/(r['start']+'_'+r['end']);folder.mkdir(parents=True,exist_ok=True);file=folder/(str(r['player_id'])+'.csv');audit=file.with_suffix('.json')
  if file.exists() and audit.exists():
   saved=json.loads(audit.read_text())
   if saved.get('complete'):return saved
  params=dict(player_id=r['player_id'],team_id=r['team_id'],season='2025-26',season_type_all_star='Regular Season',per_mode_simple='Totals',date_from_nullable=pd.Timestamp(r['start']).strftime('%m/%d/%Y'),date_to_nullable=pd.Timestamp(r['end']).strftime('%m/%d/%Y'))
  def validate(payload):
   f=validate_passing(payload,r['player_id'],r['team_id'])
   if len(f) and pd.to_numeric(f.G).gt(r['expected_games']).any():raise ValueError('Date-filter GP exceeds official appearances')
   return f
  try:
   frame,_=c.fetch(playerdashptpass.PlayerDashPtPass,params,validate);confirmed_zero=(r['daily_observed_games']==r['expected_games'] and r['daily_passes']==0);complete=len(frame)>0 or confirmed_zero;error='' if complete else 'Empty tracking profile without independently verified zero'
  except DataUnavailable as e:frame=pd.DataFrame(columns=['PLAYER_ID','TEAM_ID','PASS_TEAMMATE_PLAYER_ID','PASS']);complete=False;error=str(e)
  frame.to_csv(file,index=False);record=dict(r,complete=complete,error=error,profile_passes=float(frame.PASS.sum()),daily_difference=float(frame.PASS.sum()-r['daily_passes']));audit.write_text(json.dumps(record,indent=2));return record
 records=[]
 try:
  with ThreadPoolExecutor(max_workers=2) as pool:
   futures=[pool.submit(one,r) for r in plan]
   for i,f in enumerate(as_completed(futures),1):
    records.append(f.result())
    if i%20==0 or i==len(futures):pd.DataFrame(records).to_csv(out/'download_progress.csv',index=False);print(f'{i}/{len(futures)} profiles; complete={sum(r["complete"] for r in records)}',flush=True)
 finally:c.close()
 # Build only complete new networks, never replacing unknown profiles with zero.
 built=[]
 for row in coverage:
  if row['validated_legacy_network'] or row['absent_games']==0:continue
  team=row['team'];tasks=[r for r in records if r['team']==team];full=pd.read_csv(source/'networks'/team/'passing_matrix.csv',index_col=0);ids=full.index.astype(int).tolist();ix={pid:i for i,pid in enumerate(ids)};A=np.zeros(full.shape);errors=[]
  for task in tasks:
   if not task['complete']:errors.append(task['error']);continue
   file=out/'absence_profiles'/team/(task['start']+'_'+task['end'])/(str(task['player_id'])+'.csv');f=pd.read_csv(file)
   for r in f.itertuples():
    receiver=int(r.PASS_TEAMMATE_PLAYER_ID)
    if receiver not in ix:
     if r.PASS>0:errors.append('Receiver outside roster: '+str(receiver))
    else:A[ix[task['player_id']],ix[receiver]]+=r.PASS
  T=full.to_numpy(float);B=T-A
  if (B<0).any():errors.append('Absence counts exceed full-season edge counts')
  hi=ix[row['hub_id']]
  if A[hi].sum()+A[:,hi].sum()>0:errors.append('Absent hub has passes')
  if not np.isfinite(A).all() or (A<0).any() or np.diag(A).any():errors.append('Invalid absent counts')
  target=out/'new_networks'/team;target.mkdir(parents=True,exist_ok=True);record=dict(row,complete=not errors,errors=errors,profiles=len(tasks),total_absent_passes=float(A.sum()),total_present_passes=float(B.sum()))
  if not errors:
   pd.DataFrame(A,index=ids,columns=ids).to_csv(target/'absent_passing_matrix.csv');pd.DataFrame(B,index=ids,columns=ids).to_csv(target/'present_passing_matrix.csv')
  (target/'audit.json').write_text(json.dumps(record,indent=2));built.append(record);print(team,'verified' if not errors else str(errors),flush=True)
 print('Finished',flush=True);return 0
if __name__=='__main__':raise SystemExit(main())
