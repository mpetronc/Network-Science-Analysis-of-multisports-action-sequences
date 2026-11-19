"""First-20 passing-only EC leaders, rebuilt absence comparisons and split-role forecasts."""
from pathlib import Path
import argparse,json,shutil,subprocess,sys,os,hashlib
import pandas as pd,numpy as np
from all_season_hub_study import generate,SEASONS


def selection_table(root,baseline):
    names={season:pd.read_csv(root/'star_dependency_study/game_cache'/(season+'_player_base.csv.gz')).drop_duplicates('PLAYER_ID').set_index('PLAYER_ID').PLAYER_NAME.to_dict() for season in SEASONS}
    rows=[]
    for file in sorted(baseline.glob('*/*/first20_all_passing.csv')):
        season,team=file.relative_to(baseline).parts[:2];frame=pd.read_csv(file,index_col=0);ids=frame.index.astype(int).tolist();A=frame.to_numpy();active=A.sum(0)+A.sum(1)>0;indices=np.flatnonzero(active);_,v=np.linalg.eigh((A+A.T)[np.ix_(active,active)]);c=abs(v[:,-1]);c/=c.sum();k=sorted(range(len(indices)),key=lambda k:(-c[k],ids[indices[k]]))[0];pid=ids[indices[k]];old=json.loads((file.parent/'selection.json').read_text());rows.append(dict(season=season,team=team,volume_hub_id=old['hub_id'],volume_hub_name=old['hub_name'],ec_hub_id=pid,ec_hub_name=names[season][pid],ec_share=float(c[k]),changed=pid!=old['hub_id']))
    return pd.DataFrame(rows)


def ec_source(project,dest,season,changed,workers,original):
    source=generate(project,dest,season,workers)
    f=source/'hub_redistribution_study.py';s=f.read_text();needle="i=sorted(range(len(ids)),key=lambda i:(-strength[i],ids[i]))[0]";replacement="ec_select=ec_index(A20,ids);i=sorted(range(len(ids)),key=lambda i:(-ec_select['shares'].get(ids[i],0),ids[i]))[0]";assert needle in s;s=s.replace(needle,replacement)
    s=s.replace('for t in teams.get_teams():',f"for t in [t for t in teams.get_teams() if t['abbreviation'] in {sorted(changed)!r}]:",1)
    s=s.replace('from hub_redistribution_report import report;report(root,out,a.skip_network_plots);',"print('Rebuilt EC-selected teams',len(results),flush=True);")
    f.write_text(s)
    f=source/'hub_redistribution_report.py';s=f.read_text().replace('Hub identity: most passes sent + received during first 20 games','Hub identity: highest passing-only EC during first 20 games').replace('The passing hub is the player with the largest sum of outgoing and incoming directed pass counts in the first twenty regular-season games','The passing hub is the player with the largest eigenvector centrality on the symmetric passing-only A+Aᵀ network in the first twenty regular-season games');f.write_text(s)
    return source


def merge_season(original,dest,changed):
    old=pd.read_csv(original/'coverage_and_hubs.csv');new=pd.read_csv(dest/'coverage_and_hubs.csv');combined=pd.concat([old[~old.team.isin(changed)],new],ignore_index=True);assert len(combined)==30 and combined.team.nunique()==30;combined['hub_selection_rule']='Highest first-20 passing-only eigenvector centrality on A+A.T';combined.to_csv(dest/'coverage_and_hubs.csv',index=False)
    for name in ['replacement_minutes_and_weights.csv','network_download_plan.csv']:
        old=pd.read_csv(original/name);new=pd.read_csv(dest/name);combined=pd.concat([old[~old.team.isin(changed)],new],ignore_index=True)
        if name=='network_download_plan.csv':combined=combined.drop_duplicates(['team','kind','player_id','start','end'])
        combined.to_csv(dest/name,index=False)
    for name in ['selection_plan.json','freeze_manifest.json']:
        old=json.loads((original/name).read_text());new=json.loads((dest/name).read_text());(dest/name).write_text(json.dumps([r for r in old if r['team'] not in changed]+new,indent=2))
    old=json.loads((original/'tracking_exclusions.json').read_text()) if (original/'tracking_exclusions.json').exists() else [];new=json.loads((dest/'tracking_exclusions.json').read_text());(dest/'tracking_exclusions.json').write_text(json.dumps([r for r in old if r['team'] not in changed]+new,indent=2))
    # Remove stale, hub-specific profile wrappers from the active plan; raw API caches stay available.
    plan=pd.read_csv(dest/'network_download_plan.csv');required={(r.team,r.start+'_'+r.end,str(r.player_id)) for r in plan.itertuples()}
    for file in (dest/'passing_profiles').glob('*/*/*.json'):
        team,window,pid=file.relative_to(dest/'passing_profiles').parts;pid=Path(pid).stem
        if (team,window,pid) not in required:
            target=dest/'unused_profile_cache'/team/window;target.mkdir(parents=True,exist_ok=True)
            for suffix in ['.json','.csv']:
                f=file.with_suffix(suffix)
                if f.exists():shutil.move(str(f),str(target/f.name))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--ec-hub-study',action='store_true');p.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'nba_passing_output');p.add_argument('--baseline',type=Path);p.add_argument('--destination',type=Path);p.add_argument('--workers',type=int,choices=range(1,7),default=4);a=p.parse_args(argv);root=a.output_dir;baseline=a.baseline or root/'all_season_hub_redistribution';out=a.destination or root/'ec_hub_analysis';out.mkdir(parents=True,exist_ok=True);table=selection_table(root,baseline);assert len(table)==180;table.to_csv(out/'hub_selection_comparison.csv',index=False);newbase=out/'ec_baseline';newbase.mkdir(exist_ok=True);status=[]
    for season in SEASONS:
        old=baseline/season;dest=newbase/season;changed=set(table.loc[table.season.eq(season)&table.changed,'team']);marker=dest/'ec_rebuild_complete.json'
        if marker.exists():status.append(dict(season=season,complete=True,changed=len(changed)));continue
        if not dest.exists():shutil.copytree(old,dest)
        # Stale plots from the volume-selected definition must not be presented as EC results.
        for file in [dest/'all_30_hub_networks.pdf',*dest.glob('*/*_three_networks.png')]:
            if file.exists():file.unlink()
        source=ec_source(root.parent,dest,season,changed,a.workers,old)
        if changed:
            env=dict(os.environ,PYTHONPATH=str(source)+os.pathsep+str(root.parent),MPLCONFIGDIR='/private/tmp/sportsanalysis-mpl');command=[sys.executable,'-B',str(source/'hub_redistribution_study.py'),'--output-dir',str(root),'--destination',str(dest),'--skip-network-plots'];rc=1
            for attempt in range(3):
                with (dest/'ec_run.log').open('a') as log:print('Rebuilding',season,sorted(changed),'attempt',attempt+1,flush=True);rc=subprocess.call(command,env=env,stdout=log,stderr=subprocess.STDOUT)
                if rc==0:break
            if rc:raise RuntimeError('Incomplete EC rebuild: '+season+'; validated downloads are saved for resume')
            merge_season(old,dest,changed)
        else:
            coverage=pd.read_csv(dest/'coverage_and_hubs.csv');coverage['hub_selection_rule']='Highest first-20 passing-only eigenvector centrality on A+A.T';coverage.to_csv(dest/'coverage_and_hubs.csv',index=False)
        # Recompute entropy and observed regressions using all thirty verified contexts.
        env=dict(os.environ,PYTHONPATH=str(source)+os.pathsep+str(root.parent),MPLCONFIGDIR='/private/tmp/sportsanalysis-mpl');code="from hub_redistribution_report import report;report("+repr(str(root))+","+repr(str(dest))+",True)"
        subprocess.run([sys.executable,'-B','-c',code],env=env,check=True)
        marker.write_text(json.dumps(dict(season=season,changed_teams=sorted(changed),hub_rule='argmax passing-only EC from first20 A+A.T'),indent=2));status.append(dict(season=season,complete=True,changed=len(changed)));(out/'run_status.json').write_text(json.dumps(status,indent=2))
    from all_season_hub_report import report
    import all_season_hub_report as reporting
    original_merge=reporting.merge_pdf
    def merge_available(files,target):
        if all(Path(f).exists() for f in files):original_merge(files,target)
    reporting.merge_pdf=merge_available
    report(root,newbase)
    method=newbase/'METHOD_AND_RESULTS.md';text=method.read_text();text=text.replace('hub=max passes sent+received in first20','hub=highest first-20 passing-only eigenvector centrality on A+A.T');text=text.replace('Previously validated 2025–26 results are reused unchanged; earlier seasons require independent date-filtered retrieval.','Same-player contexts are reused after first-20 EC identity checks; changed hubs have rebuilt date-filtered contexts.');method.write_text(text)
    import ec_hub_role_report
    sys.modules['split_role_hub_report']=ec_hub_role_report
    from split_role_hub_study import main as role_study
    role_study(['--output-dir',str(root),'--baseline',str(newbase),'--destination',str(out/'passing_roles')])
    print('EC-selected hub study complete',str(out),flush=True);return 0

if __name__=='__main__':raise SystemExit(main())
