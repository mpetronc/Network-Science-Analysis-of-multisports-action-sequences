from pathlib import Path
import argparse,json,shutil
from concurrent.futures import ThreadPoolExecutor,as_completed
import pandas as pd,numpy as np
from nba_api.stats.endpoints import playerdashptpass
from teamgraphs import NBAClient,Config,validate_passing,DataUnavailable

def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'nba_passing_output');p.add_argument('--data-dir',type=Path);a=p.parse_args(argv);root=a.output_dir;out=a.data_dir or root/'volume_selected_dependency';source=root/'2025-26/team_season_topics';P=pd.read_csv(source/'player_games.csv',dtype={'GAME_ID':str});G=pd.read_csv(source/'team_games.csv',dtype={'GAME_ID':str});G['date']=pd.to_datetime(G.GAME_DATE);selected=pd.read_csv(out/'selected_stars.csv').fillna('');daily=pd.concat([pd.read_csv(f).assign(date=f.stem) for f in (source/'daily_passing').glob('*.csv')],ignore_index=True);legacy=pd.read_csv(root/'league_star_presence_entropy/all_30_team_entropy.csv').set_index('team');cachebases=[root/'league_star_presence_entropy/api_cache',source/'hub_absence_resilience/cache',root/'onoff_dependency/passing_api_cache',root/'ec_selected_dependency/passing_api_cache'];records=[];plan=[];reuse={}
 for z in selected.itertuples():
  old=legacy.loc[z.team];folder=out/'networks'/z.team;folder.mkdir(parents=True,exist_ok=True)
  prior_ec=pd.read_csv(root/'ec_selected_dependency/selected_stars.csv');match_ec=prior_ec[prior_ec.team.eq(z.team)&prior_ec.star_id.eq(z.star_id)&prior_ec.window_start.eq(z.window_start)&prior_ec.window_end.eq(z.window_end)]
  if len(match_ec)==1:
   saved=root/'ec_selected_dependency/networks'/z.team;assert json.loads((saved/'audit.json').read_text())['complete']
   for label in ['present','absent']:shutil.copy2(saved/(label+'_passing_matrix.csv'),folder/(label+'_passing_matrix.csv'))
   (folder/'audit.json').write_text(json.dumps(dict(team=z.team,star_id=z.star_id,complete=True,reused_same_player_and_window=True,source=str(saved)),indent=2));reuse[z.team]=True;continue
  existing=pd.read_csv(root/'onoff_dependency/selected_stars.csv');match=existing[existing.team.eq(z.team)&existing.star_id.eq(z.star_id)&existing.window_start.eq(z.window_start)&existing.window_end.eq(z.window_end)]
  if len(match)==1:
   saved=root/'onoff_dependency/networks'/z.team;info=json.loads((saved/'audit.json').read_text());assert info['complete']
   for label in ['present','absent']:shutil.copy2(saved/(label+'_passing_matrix.csv'),folder/(label+'_passing_matrix.csv'))
   (folder/'audit.json').write_text(json.dumps(dict(team=z.team,star_id=z.star_id,complete=True,reused_same_player_and_window=True,source=str(saved),present_games=z.present_games,absent_games=z.absent_games),indent=2));reuse[z.team]=True;continue
  if z.window_games==82 and int(old.hub_id)==z.star_id and old.status in ['complete_cached','complete_new']:
   full=pd.read_csv(source/'networks'/z.team/'passing_matrix.csv',index_col=0);T=full.to_numpy(float)
   if old.status=='complete_cached':B=pd.read_csv(source/'hub_absence_resilience/present_networks'/z.team/'passing_matrix.csv',index_col=0).to_numpy(float);A=T-B
   else:B=pd.read_csv(root/'league_star_presence_entropy/new_networks'/z.team/'present_passing_matrix.csv',index_col=0).to_numpy(float);A=pd.read_csv(root/'league_star_presence_entropy/new_networks'/z.team/'absent_passing_matrix.csv',index_col=0).to_numpy(float)
   for label,M in [('present',B),('absent',A)]:pd.DataFrame(M,index=full.index,columns=full.columns).to_csv(folder/(label+'_passing_matrix.csv'))
   (folder/'audit.json').write_text(json.dumps(dict(team=z.team,complete=True,reused_same_player_and_window=True,star_id=z.star_id,present_games=z.present_games,absent_games=z.absent_games),indent=2));reuse[z.team]=True;continue
  games=G[G.TEAM_ID.eq(z.team_id)&G.date.between(z.window_start,z.window_end)].sort_values(['date','GAME_ID']).copy();played=set(P[P.TEAM_ID.eq(z.team_id)&P.PLAYER_ID.eq(z.star_id)].GAME_ID);games['present']=games.GAME_ID.isin(played);games['block']=games.present.ne(games.present.shift()).cumsum();groups=[('absent',block) for _,block in games[~games.present].groupby('block')]
  if z.window_games!=82:groups.append(('total',games))
  for kind,block in groups:
   start=block.date.min().date().isoformat();end=block.date.max().date().isoformat()
   for pid,part in P[P.TEAM_ID.eq(z.team_id)&P.GAME_ID.isin(block.GAME_ID)].groupby('PLAYER_ID'):
    observed=daily[daily.TEAM_ID.eq(z.team_id)&daily.PLAYER_ID.eq(pid)&daily.date.between(start,end)];plan.append(dict(team=z.team,team_id=z.team_id,kind=kind,player_id=int(pid),start=start,end=end,expected_games=len(part),daily_observed_games=len(observed),daily_passes=float(observed.PASSES_MADE.sum())))
 pd.DataFrame(plan).to_csv(out/'network_download_plan.csv',index=False);print('Required profiles',len(plan),'teams',len(set(r['team'] for r in plan)),'reused',list(reuse),flush=True)
 def one(r):
  file=out/'passing_profiles'/r['team']/(r['start']+'_'+r['end'])/(str(r['player_id'])+'.csv');file.parent.mkdir(parents=True,exist_ok=True);audit=file.with_suffix('.json')
  if file.exists() and audit.exists():
   info=json.loads(audit.read_text())
   if info['complete']:return dict(r,**{k:v for k,v in info.items() if k not in r})
  c=NBAClient(Config(season='2025-26',cache_dir=out/'passing_api_cache',timeout=25,retries=2,min_delay=.5,max_delay=1));params=dict(player_id=r['player_id'],team_id=r['team_id'],season='2025-26',season_type_all_star='Regular Season',per_mode_simple='Totals',date_from_nullable=pd.Timestamp(r['start']).strftime('%m/%d/%Y'),date_to_nullable=pd.Timestamp(r['end']).strftime('%m/%d/%Y'))
  def validate(payload):
   f=validate_passing(payload,r['player_id'],r['team_id'])
   if len(f) and pd.to_numeric(f.G).gt(r['expected_games']).any():raise ValueError('Date-filter GP exceeds official appearances')
   return f
  try:
   query=playerdashptpass.PlayerDashPtPass(**params,get_request=False);resolved=dict(query.parameters);target=c.cache.path_for(query.endpoint,resolved)
   if not target.exists():
    for base in cachebases:
     src=base/target.name
     if src.exists():target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,target);break
   f,_=c.fetch(playerdashptpass.PlayerDashPtPass,params,validate);zero=r['daily_observed_games']==r['expected_games'] and r['daily_passes']==0;complete=len(f)>0 or zero;error='' if complete else 'Empty profile without independently verified zero'
  except Exception as e:f=pd.DataFrame(columns=['PASS_TEAMMATE_PLAYER_ID','PASS']);complete=False;error=repr(e)
  finally:c.close()
  f.to_csv(file,index=False);info=dict(r,complete=complete,error=error,profile_passes=float(f.PASS.sum()),daily_difference=float(f.PASS.sum()-r['daily_passes']));audit.write_text(json.dumps(info,indent=2));return info
 with ThreadPoolExecutor(max_workers=6) as pool:
  futures=[pool.submit(one,r) for r in plan]
  for i,f in enumerate(as_completed(futures),1):
   records.append(f.result())
   if i%50==0 or i==len(plan):pd.DataFrame(records).to_csv(out/'network_download_progress.csv',index=False);print(i,'/',len(plan),'complete',sum(r['complete'] for r in records),flush=True)
 from corroborate_empty_profiles import main as corroborate
 corroborate(['--data-dir',str(out)])
 for r in records:
  audit=out/'passing_profiles'/r['team']/(r['start']+'_'+r['end'])/(str(r['player_id'])+'.json')
  updated=json.loads(audit.read_text());r.update(complete=updated['complete'],error=updated['error'])
 pd.DataFrame(records).to_csv(out/'network_download_progress.csv',index=False)
 for z in selected.itertuples():
  if z.team in reuse:continue
  full=pd.read_csv(source/'networks'/z.team/'passing_matrix.csv',index_col=0);ids=full.index.astype(int).tolist();ix={pid:i for i,pid in enumerate(ids)};mat={'absent':np.zeros(full.shape),'total':full.to_numpy(float).copy() if z.window_games==82 else np.zeros(full.shape)};errors=[];tasks=[r for r in records if r['team']==z.team]
  for r in tasks:
   if not r['complete']:errors.append(r['error']);continue
   f=pd.read_csv(out/'passing_profiles'/z.team/(r['start']+'_'+r['end'])/(str(r['player_id'])+'.csv'))
   for edge in f.itertuples():
    receiver=int(edge.PASS_TEAMMATE_PLAYER_ID)
    if receiver not in ix:
     if edge.PASS>0:errors.append('Positive receiver outside roster '+str(receiver))
    else:mat[r['kind']][ix[r['player_id']],ix[receiver]]+=edge.PASS
  T=mat['total'];A=mat['absent'];B=T-A
  if B.min() < -1e-8:errors.append('Absence exceeds total counts')
  hi=ix[z.star_id]
  if A[hi].sum()+A[:,hi].sum()>0:errors.append('Absent star has passes')
  if not np.isfinite(T).all() or not np.isfinite(A).all() or A.min()<0 or np.diag(T).any() or np.diag(A).any():errors.append('Invalid matrix')
  folder=out/'networks'/z.team;folder.mkdir(exist_ok=True)
  if not errors:
   for label,M in [('present',B),('absent',A),('total',T)]:pd.DataFrame(M,index=ids,columns=ids).to_csv(folder/(label+'_passing_matrix.csv'))
  audit=dict(team=z.team,star_id=z.star_id,complete=not errors,errors=errors,profiles=len(tasks),present_games=z.present_games,absent_games=z.absent_games,window_games=z.window_games);(folder/'audit.json').write_text(json.dumps(audit,indent=2));print(z.team,'VERIFIED' if not errors else errors,flush=True)
if __name__=='__main__':main()
