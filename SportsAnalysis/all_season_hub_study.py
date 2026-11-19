"""Extend the frozen first-20 passing-hub study to all six saved seasons."""
from pathlib import Path
import argparse,subprocess,sys,os,json,shutil,time,hashlib
SEASONS=['2020-21','2021-22','2022-23','2023-24','2024-25','2025-26']

def generate(project,out,season,workers=4):
 code=out/'source';code.mkdir(parents=True,exist_ok=True);prior=f'{int(season[:4])-1}-{int(season[:4])%100:02d}'
 for name in ['hub_study_data.py','hub_redistribution_model.py','hub_redistribution_report.py','hub_redistribution_study.py','corroborate_empty_profiles.py']:
  s=(project/name).read_text().replace('2024-25','__PRIOR_SEASON__').replace('2025-26',season).replace('2025–26',season).replace('__PRIOR_SEASON__',prior)
  if name=='hub_study_data.py':
   s='from hub_thread_local_http import install as install_worker_http\ninstall_worker_http()\n'+s
   s=s.replace('max_workers=6',f'max_workers={workers}')
   s=s.replace("   f=self.file(r);audit=", "   if (r['team'],r['kind'],r['start'],r['end']) in getattr(self,'excluded_windows',set()):continue\n   f=self.file(r);audit=")
   s=s.replace(";tasks=[r for r in self.plan", ";from all_season_hub_cached_totals import seed;seed(self);tasks=[r for r in self.plan")
   s=s.replace("  if tasks:corroborate(['--data-dir',str(self.out)])", "  if tasks:corroborate(['--data-dir',str(self.out)]);from confirm_hub_empty_daily import confirm;confirm(self.root,self.out,'"+season+"')")
   first=s.index('   for edge in pd.read_csv(f).itertuples():');last=s.index('  assert np.isfinite(M)',first);s=s[:first]+"   from hub_tracking_cleanup import add_profile;add_profile(self,M,ix,r,pd.read_csv(f))\n"+s[last:]
   start=s.index("base=self.root/");end=s.index(';self.plan=[]',start)
   s=s[:start]+"base=self.root/'star_dependency_study/game_cache';self.players=pd.read_csv(base/'"+season+"_player_base.csv.gz',dtype={'GAME_ID':str});self.games=pd.read_csv(base/'"+season+"_team_base.csv.gz',dtype={'GAME_ID':str});self.players.GAME_ID=self.players.GAME_ID.str.zfill(10);self.games.GAME_ID=self.games.GAME_ID.str.zfill(10);self.players.GAME_DATE=self.players.GAME_DATE.str[:10];self.games.GAME_DATE=self.games.GAME_DATE.str[:10];self.games=self.games.sort_values(['GAME_DATE','GAME_ID']);self.daily=pd.DataFrame(columns=['TEAM_ID','PLAYER_ID','date','PASSES_MADE'])"+s[end:]
  if name=='corroborate_empty_profiles.py':
   s=s.replace("   if snap is None:complete=False;break", "   raw_source=json.loads(audit.read_text()).get('source_raw_cache')\n   if snap is None and raw_source:snap=json.loads(Path(raw_source).read_text())\n   if snap is None:complete=False;break")
   s=s.replace("source=str(client.cache.path_for(query.endpoint,dict(query.parameters)))","source=raw_source or str(client.cache.path_for(query.endpoint,dict(query.parameters)))")
  if name=='hub_redistribution_report.py':
   s=s.replace("pd.read_csv(root/'"+season+"/team_season_topics/player_games.csv')","pd.read_csv(root/'star_dependency_study/game_cache/"+season+"_player_base.csv.gz')")
  if name=='hub_redistribution_study.py':
   start=s.index(' caches={}');end=s.index(" data.download({'later_absence','eval_total'})",start)
   s=s[:start]+''' evaluation={}
 for z in meta:
  team=z['team'];g=data.games[data.games.TEAM_ID.eq(z['team_id'])].sort_values(['GAME_DATE','GAME_ID']).reset_index(drop=True);g['ordinal']=np.arange(len(g));g=g[g.GAME_DATE.between(z['window_start'],z['window_end'])];ids=sorted(data.players.loc[data.players.TEAM_ID.eq(z['team_id']),'PLAYER_ID'].unique().tolist());target=g[g.GAME_ID.isin(z['later_absent_game_ids'])];add_blocks(data,team,z['team_id'],'later_absence',target);data.add_window(team,z['team_id'],'eval_total',g);evaluation[team]=dict(ids=ids,total=None,absent=None,source='Audited season-specific date-filtered total and later-absence profiles')
'''+s[end:]
  if name=='hub_redistribution_study.py':
   s=s.replace("data.download({'later_absence','eval_total'});results=[]", "data.download({'later_absence','eval_total'});from hub_tracking_exclusions import prepare;unavailable_reference=prepare(data,meta);results=[]")
   s=s.replace("T=e['total'] if e['total'] is not None else data.matrix(team,'eval_total',ids)", "T=None if team in unavailable_reference else (e['total'] if e['total'] is not None else data.matrix(team,'eval_total',ids))")
   s=s.replace("LP=T-all20-A;assert A.min()>=-1e-8 and LP.min()>=-1e-8", "LP=np.full_like(A,np.nan) if team in unavailable_reference else T-all20-A;assert A.min()>=-1e-8 and (team in unavailable_reference or LP.min()>=-1e-8)")
  (code/name).write_text(s)
 for helper in ['all_season_hub_cached_totals.py','confirm_hub_empty_daily.py','hub_thread_local_http.py','hub_tracking_exclusions.py','hub_tracking_cleanup.py']:shutil.copy2(Path(__file__).resolve().parent/helper,code/helper)
 (code/'source_hashes.json').write_text(json.dumps({f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in code.glob('*.py')},indent=2))
 return code

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--all-season-hub-study',action='store_true');p.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'nba_passing_output');p.add_argument('--destination',type=Path);p.add_argument('--seasons',nargs='+',default=SEASONS);p.add_argument('--workers',type=int,choices=range(1,7),default=4);p.add_argument('--report-only',action='store_true');a=p.parse_args(argv);project=a.output_dir.parent;out=a.destination or a.output_dir/'all_season_hub_redistribution';out.mkdir(parents=True,exist_ok=True);status=[]
 for season in a.seasons:
  dest=out/season
  if (dest/'summary.json').exists() and (dest/'all_30_hub_networks.pdf').exists():
   status.append(dict(season=season,complete=True,source='Previously completed identical per-season study'));print('Reusing completed',season,flush=True);continue
  if season=='2025-26' and (a.output_dir/'hub_redistribution_2025_26/validation.json').exists():
   shutil.copytree(a.output_dir/'hub_redistribution_2025_26',dest,dirs_exist_ok=True);status.append(dict(season=season,complete=True,source='Previously validated identical first-20 study'));continue
  if a.report_only:
   status.append(dict(season=season,complete=False,source='Not yet complete; missing date-filtered NBA data'));continue
  code=generate(project,dest,season,a.workers);env=dict(os.environ,PYTHONPATH=str(code)+os.pathsep+str(project),MPLCONFIGDIR='/private/tmp/sportsanalysis-mpl');command=[sys.executable,'-B',str(code/'hub_redistribution_study.py'),'--output-dir',str(a.output_dir),'--destination',str(dest)]
  rc=1
  for attempt in range(3):
   with (dest/'run.log').open('a') as log:
    print('Starting',season,'attempt',attempt+1,flush=True);rc=subprocess.call(command,env=env,stdout=log,stderr=subprocess.STDOUT)
   if rc==0:break
  status.append(dict(season=season,complete=rc==0,source='Season-specific NBA date-filtered passing profiles',returncode=rc));(out/'run_status.json').write_text(json.dumps(status,indent=2));print('Season result',season,rc,flush=True)
 (out/'run_status.json').write_text(json.dumps(status,indent=2))
 from all_season_hub_report import report
 report(a.output_dir,out)
 return 0 if all(z['complete'] for z in status) else 1
if __name__=='__main__':raise SystemExit(main())
