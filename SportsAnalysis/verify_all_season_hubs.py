"""Independent numerical and scope checks for the multi-season study."""
from pathlib import Path
import json,sys,importlib.util
import numpy as np,pandas as pd
from scipy.stats import t

def main(out,project):
 out=Path(out);project=Path(project)
 source=(project/'verify_hub_redistribution.py').read_text();needle=" for f in (out/'passing_profiles').rglob('*.json'):assert json.loads(f.read_text())['complete'],f";replacement=""" declared={(r['team'],r['kind'],r['start'],r['end']) for r in json.loads((out/'tracking_exclusions.json').read_text())} if (out/'tracking_exclusions.json').exists() else set()
 for f in (out/'passing_profiles').rglob('*.json'):
  info=json.loads(f.read_text());assert info['complete'] or (info['team'],info['kind'],info['start'],info['end']) in declared,f""";assert needle in source;namespace={'__name__':'independent_single_verifier'};exec(compile(source.replace(needle,replacement).replace('complete raw audits','complete used-profile audits with declared exclusions'),str(project/'verify_hub_redistribution.py'),'exec'),namespace)
 data=pd.read_csv(out/'all_team_season_entropy_and_ec.csv');assert not data.duplicated(['season','team']).any();checked=[]
 for season,d in data.groupby('season'):
  assert len(d)==30;folder=out/season;namespace['main'](folder)
  for file in folder.glob('*/frozen_specification.json'):
   s=json.loads(file.read_text());assert s['position_source_season']==f'{int(season[:4])-1}-{int(season[:4])%100:02d}'
   selection=s['selection'];assert selection['first20_hub_present_games']+selection['first20_hub_absent_games']==20
  for audit in (folder/'passing_profiles').rglob('*.json'):
   info=json.loads(audit.read_text())
   if 'original_parameters' in info:
    params=info['original_parameters'];assert params['Season']==season and not params['DateFrom'] and not params['DateTo'] and params['PerMode']=='Totals';assert info['kind']=='eval_total' and info['scope_equivalence']
  exclusions=json.loads((folder/'tracking_exclusions.json').read_text()) if (folder/'tracking_exclusions.json').exists() else [];excluded={(r['team'],r['kind'],r['start'],r['end']) for r in exclusions};plan=pd.read_csv(folder/'network_download_plan.csv')
  official=pd.read_csv(project/'nba_passing_output/star_dependency_study/game_cache'/(season+'_team_base.csv.gz'),dtype={'GAME_ID':str});official.GAME_DATE=official.GAME_DATE.str[:10];official.GAME_ID=official.GAME_ID.str.zfill(10)
  player_logs=pd.read_csv(project/'nba_passing_output/star_dependency_study/game_cache'/(season+'_player_base.csv.gz'),dtype={'GAME_ID':str});player_logs.GAME_ID=player_logs.GAME_ID.str.zfill(10);scopes={}
  for z in d.itertuples():
   spec=json.loads((folder/z.team/'frozen_specification.json').read_text());selection=spec['selection'];ids=selection['training_player_ids'];mp=player_logs[player_logs.TEAM_ID.eq(z.team_id)&player_logs.GAME_ID.isin(selection['train_present_game_ids'])].groupby('PLAYER_ID').MIN.sum().reindex(ids,fill_value=0).to_numpy();assert np.allclose(mp,spec['training_present_minutes']);assert np.allclose(240*mp/mp.sum(),spec['base_minutes'])
  if season!='2025-26':
   for z in d.itertuples():
    selection=json.loads((folder/z.team/'selection.json').read_text());used=[]
    windows={(r['start'],r['end']) for r in plan[plan.team.eq(z.team)&plan.kind.eq('later_absence')].to_dict('records') if (r['team'],r['kind'],r['start'],r['end']) not in excluded}
    for start,end in windows:used.extend(official.loc[official.TEAM_ID.eq(z.team_id)&official.GAME_DATE.between(start,end),'GAME_ID'].tolist())
    assert len(used)==len(set(used))==z.later_hub_absent_games and set(used)<=set(selection['later_absent_game_ids'])
    frame=pd.read_csv(folder/z.team/'actual_evaluation.csv',index_col=0);ids=frame.index.astype(int).tolist();ix={pid:i for i,pid in enumerate(ids)};A=np.zeros(frame.shape)
    for r in plan[plan.team.eq(z.team)&plan.kind.eq('later_absence')].to_dict('records'):
     if (r['team'],r['kind'],r['start'],r['end']) in excluded:continue
     f=folder/'passing_profiles'/r['team']/(r['start']+'_'+r['end'])/(str(r['player_id'])+'.csv');assert json.loads(f.with_suffix('.json').read_text())['complete']
     scope=(z.team_id,r['start'],r['end'])
     if scope not in scopes:scopes[scope]=set(player_logs.loc[player_logs.TEAM_ID.eq(z.team_id)&player_logs.GAME_DATE.str[:10].between(r['start'],r['end']),'PLAYER_ID'])
     for edge in pd.read_csv(f).itertuples():
      receiver=int(edge.PASS_TEAMMATE_PLAYER_ID)
      if edge.PASS>0 and receiver in ix and receiver in scopes[scope]:A[ix[r['player_id']],ix[receiver]]+=edge.PASS
    assert np.allclose(A,frame.to_numpy())
    if hasattr(z,'official_later_hub_absent_games'):assert z.official_later_hub_absent_games-z.later_hub_absent_games==z.excluded_later_absence_games
    if hasattr(z,'later_present_reference_available') and z.later_present_reference_available==False:
     ref=pd.read_csv(folder/z.team/'later_present_evaluation.csv',index_col=0);assert (ref.isna().all().all() or (z.later_hub_present_games==0 and np.allclose(ref,0))) and not z.concurrent_eligible
  checked.append(season)
 p=pd.read_csv(out/'pooled_season_adjusted_regressions.csv')
 for z in p.itertuples():
  col,eligible=('actual_equal_count_change','primary_eligible') if z.analysis=='primary_equal_count' else ('concurrent_equal_count_change','concurrent_eligible');q=data[data[eligible]];xx=q.ec_index-q.groupby('season').ec_index.transform('mean');yy=q[col]-q.groupby('season')[col].transform('mean');slope=np.dot(xx,yy)/np.dot(xx,xx);assert np.isclose(slope,z.slope) and len(q)==z.n
  residual=yy-slope*xx;scores=pd.DataFrame({'team':q.team,'score':xx*residual}).groupby('team').score.sum();G=len(scores);k=q.season.nunique()+1;variance=G/(G-1)*(len(q)-1)/(len(q)-k)*np.dot(scores,scores)/np.dot(xx,xx)**2;se=np.sqrt(variance);pvalue=2*t.sf(abs(slope/se),G-1);assert np.isclose(pvalue,z.franchise_clustered_p) and np.isclose(slope-t.ppf(.975,G-1)*se,z.ci_low) and np.isclose(slope+t.ppf(.975,G-1)*se,z.ci_high)
 assert namespace['pages'](out/'all_team_season_networks.pdf')==30*len(checked)
 assert namespace['pages'](out/'all_team_season_entropy_changes.pdf')==7*len(checked)
 assert namespace['pages'](out/'all_seasons_regressions.pdf')==6
 result=dict(passed=True,seasons=checked,team_seasons=len(data),checks='Independent per-team hub selection, EC, edge redistribution, outgoing entropy, regression, scope, frozen hashes; pooled slope, franchise-clustered uncertainty and confidence intervals by independent within-season score calculation; combined PDF page counts');(out/'validation.json').write_text(json.dumps(result,indent=2));print(result)
if __name__=='__main__':main(sys.argv[1],sys.argv[2])
