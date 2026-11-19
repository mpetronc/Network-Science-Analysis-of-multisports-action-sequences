"""2025–26 first-20 passing hubs, role redistribution, entropy and EC regressions."""
from pathlib import Path
import argparse,json,hashlib
import pandas as pd,numpy as np
from nba_api.stats.static import teams
from hub_study_data import first20
from hub_redistribution_model import ec_index,redistribute

def sha(file):return hashlib.sha256(Path(file).read_bytes()).hexdigest()
def dump_matrix(M,ids,file):pd.DataFrame(M,index=ids,columns=ids).to_csv(file)
def add_blocks(data,team,tid,kind,g):
 if len(g)==0:return
 # Group by consecutive TEAM schedule positions, not calendar-adjacent dates.
 for _,part in g.groupby(g.ordinal.diff().ne(1).cumsum()):data.add_window(team,tid,kind,part)

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--hub-redistribution-study',action='store_true');p.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'nba_passing_output');p.add_argument('--destination',type=Path);p.add_argument('--skip-network-plots',action='store_true');a=p.parse_args(argv);root=a.output_dir;out=a.destination or root/'hub_redistribution_2025_26';data,positions=first20(root,out);coverage=[];meta=[]
 for t in teams.get_teams():
  team=t['abbreviation'];tid=t['id'];g=data.games[data.games.TEAM_ID.eq(tid)].sort_values(['GAME_DATE','GAME_ID']).reset_index(drop=True);g['ordinal']=np.arange(len(g));train=g.head(20);pp=data.players[data.players.TEAM_ID.eq(tid)&data.players.GAME_ID.isin(train.GAME_ID)];ids=sorted(pp.PLAYER_ID.unique().tolist());A20=data.matrix(team,'first20',ids);strength=A20.sum(0)+A20.sum(1);i=sorted(range(len(ids)),key=lambda i:(-strength[i],ids[i]))[0];hub=ids[i];name=data.players.loc[data.players.PLAYER_ID.eq(hub),'PLAYER_NAME'].iloc[0];ec=ec_index(A20,ids);played=set(data.players[data.players.TEAM_ID.eq(tid)&data.players.PLAYER_ID.eq(hub)].GAME_ID);abs_early=train[~train.GAME_ID.isin(played)];add_blocks(data,team,tid,'early_absence',abs_early);other=data.players[data.players.PLAYER_ID.eq(hub)&data.players.TEAM_ID.ne(tid)];other_date=pd.to_datetime(other.GAME_DATE).min() if len(other) else pd.NaT;censor=other_date-pd.Timedelta(days=14) if pd.notna(other_date) else pd.NaT;window=g[g.GAME_DATE.lt(str(censor.date()))] if pd.notna(censor) else g;later=window[window.ordinal.ge(20)];late_abs=later[~later.GAME_ID.isin(played)];late_pre=later[later.GAME_ID.isin(played)];folder=out/team;folder.mkdir(exist_ok=True);dump_matrix(A20,ids,folder/'first20_all_passing.csv');info=dict(team=team,team_id=tid,hub_id=int(hub),hub_name=name,first20_last_date=train.GAME_DATE.iloc[-1],first20_passes_in_out=float(strength[i]),first20_passes_out=float(A20[i].sum()),first20_passes_in=float(A20[:,i].sum()),ec_index=ec['index'],ec_hhi=ec['hhi'],ec_active_players=ec['n'],hub_ec_share=ec['shares'][hub],first20_hub_present_games=int(20-len(abs_early)),first20_hub_absent_games=len(abs_early),later_hub_present_games=len(late_pre),later_hub_absent_games=len(late_abs),window_start=g.GAME_DATE.iloc[0],window_end=window.GAME_DATE.iloc[-1],trade_censor_date=str(censor.date()) if pd.notna(censor) else '',training_player_ids=ids,train_game_ids=train.GAME_ID.tolist(),train_present_game_ids=train.loc[train.GAME_ID.isin(played),'GAME_ID'].tolist(),train_absent_game_ids=abs_early.GAME_ID.tolist(),later_absent_game_ids=late_abs.GAME_ID.tolist(),later_present_game_ids=late_pre.GAME_ID.tolist());meta.append(info);(folder/'selection.json').write_text(json.dumps(info,indent=2));print('Selected',team,name,flush=True)
 (out/'selection_plan.json').write_text(json.dumps(meta,indent=2));data.download({'early_absence'});frozen=[];allocations=[]
 for z in meta:
  folder=out/z['team'];ids=z['training_player_ids'];A20=pd.read_csv(folder/'first20_all_passing.csv',index_col=0).to_numpy();early=data.matrix(z['team'],'early_absence',ids);P=A20-early;assert P.min()>=0;hi=ids.index(z['hub_id']);assert early[hi].sum()+early[:,hi].sum()==0;pre=data.players[data.players.TEAM_ID.eq(z['team_id'])&data.players.GAME_ID.isin(z['train_present_game_ids'])];mp=pre.groupby('PLAYER_ID').MIN.sum().reindex(ids,fill_value=0).to_numpy(float);absmins=data.players[data.players.TEAM_ID.eq(z['team_id'])&data.players.GAME_ID.isin(z['train_absent_game_ids'])].groupby('PLAYER_ID').MIN.sum().reindex(ids,fill_value=0).to_numpy(float);historic=dict(per_game=absmins/max(1,z['first20_hub_absent_games']),present_per_game=mp/z['first20_hub_present_games']);M,spec=redistribute(P,mp,historic,z['first20_hub_absent_games'],hi,ids,positions);D=P.copy();D[hi]=0;D[:,hi]=0
  for name,B in [('hub_present_training',P),('deleted_training',D),('frozen_redistribution',M)]:dump_matrix(B,ids,folder/(name+'.csv'))
  spec.update(selection=z,training_present_minutes=mp.tolist(),training_absent_minutes=absmins.tolist(),position_source_season='2024-25',freeze_stage='Before later-game network retrieval',frozen_sha256=sha(folder/'frozen_redistribution.csv'));(folder/'frozen_specification.json').write_text(json.dumps(spec,indent=2));frozen.append(dict(team=z['team'],sha256=spec['frozen_sha256']))
  for i,pid in enumerate(ids):allocations.append(dict(team=z['team'],player_id=pid,player_name=data.players.loc[data.players.PLAYER_ID.eq(pid),'PLAYER_NAME'].iloc[0],is_hub=i==hi,prior_position='/'.join(positions.get(pid,[])) or 'Unknown',baseline_minutes=spec['base_minutes'][i],forecast_minutes=spec['forecast_minutes'][i],extra_minutes=spec['extra_minutes'][i],replacement_weight=spec['replacement_weights'][i],position_match=spec['position_match'][i],depth_rank=spec['position_depth_rank'][i],historical_blend=spec['historical_blend']))
 (out/'freeze_manifest.json').write_text(json.dumps(frozen,indent=2));pd.DataFrame(allocations).to_csv(out/'replacement_minutes_and_weights.csv',index=False);print('FROZEN all predictions',len(frozen),flush=True)
 # Evaluation phase: reuse exact full-window partitions where available.
 caches={}
 for name in ['onoff_dependency','ec_selected_dependency','volume_selected_dependency','creator_dependency','ball_control_dependency']:
  f=root/name/'selected_stars.csv'
  if f.exists():caches[name]=pd.read_csv(f)
 legacy=pd.read_csv(root/'league_star_presence_entropy/all_30_team_entropy.csv').set_index('team');evaluation={}
 for z in meta:
  team=z['team'];folder=out/team;full=pd.read_csv(root/'2025-26/team_season_topics/networks'/team/'passing_matrix.csv',index_col=0);ids=full.index.astype(int).tolist();found=False
  for name,table in caches.items():
   match=table[table.team.eq(team)&table.star_id.eq(z['hub_id'])&table.window_start.eq(z['window_start'])&table.window_end.eq(z['window_end'])]
   if len(match):
    base=root/name/'networks'/team;audit=json.loads((base/'audit.json').read_text());assert audit['complete'];pp=pd.read_csv(base/'present_passing_matrix.csv',index_col=0);aa=pd.read_csv(base/'absent_passing_matrix.csv',index_col=0);assert pp.index.astype(int).tolist()==ids and aa.index.astype(int).tolist()==ids;evaluation[team]=dict(ids=ids,total=pp.to_numpy()+aa.to_numpy(),absent=aa.to_numpy(),source=str(base));found=True;break
  if not found and len(data.games[data.games.TEAM_ID.eq(z['team_id'])&data.games.GAME_DATE.between(z['window_start'],z['window_end'])])==82 and legacy.loc[team,'hub_id']==z['hub_id'] and legacy.loc[team,'status'] in ['complete_cached','complete_new']:
   if legacy.loc[team,'status']=='complete_cached':pp=pd.read_csv(root/'2025-26/team_season_topics/hub_absence_resilience/present_networks'/team/'passing_matrix.csv',index_col=0);A=full.to_numpy()-pp.to_numpy()
   else:A=pd.read_csv(root/'league_star_presence_entropy/new_networks'/team/'absent_passing_matrix.csv',index_col=0).to_numpy()
   evaluation[team]=dict(ids=ids,total=full.to_numpy(),absent=A,source='validated legacy same player/full-season partition');found=True
  if not found:
   g=data.games[data.games.TEAM_ID.eq(z['team_id'])].sort_values(['GAME_DATE','GAME_ID']).reset_index(drop=True);g['ordinal']=np.arange(len(g));g=g[g.GAME_DATE.between(z['window_start'],z['window_end'])];target=g[g.GAME_ID.isin(z['later_absent_game_ids'])];add_blocks(data,team,z['team_id'],'later_absence',target)
   if len(g)==82:evaluation[team]=dict(ids=ids,total=full.to_numpy(),absent=None,source='downloaded later absence profiles')
   else:data.add_window(team,z['team_id'],'eval_total',g);evaluation[team]=dict(ids=ids,total=None,absent=None,source='downloaded trade-censored total/absence profiles')
 data.download({'later_absence','eval_total'});results=[]
 for z in meta:
  team=z['team'];folder=out/team;e=evaluation[team];ids=e['ids'];trainids=z['training_player_ids'];early_train=data.matrix(team,'early_absence',trainids);early=pd.DataFrame(early_train,index=trainids,columns=trainids).reindex(index=ids,columns=ids,fill_value=0).to_numpy();T=e['total'] if e['total'] is not None else data.matrix(team,'eval_total',ids);all20=pd.read_csv(folder/'first20_all_passing.csv',index_col=0).reindex(index=ids,columns=list(map(str,ids)),fill_value=0).to_numpy();P=pd.read_csv(folder/'hub_present_training.csv',index_col=0).reindex(index=ids,columns=list(map(str,ids)),fill_value=0).to_numpy();M=pd.read_csv(folder/'frozen_redistribution.csv',index_col=0).reindex(index=ids,columns=list(map(str,ids)),fill_value=0).to_numpy();A=e['absent']-early if e['absent'] is not None else data.matrix(team,'later_absence',ids);LP=T-all20-A;assert A.min()>=-1e-8 and LP.min()>=-1e-8;A=np.maximum(A,0);LP=np.maximum(LP,0);hi=ids.index(z['hub_id']);assert A[hi].sum()+A[:,hi].sum()==0;assert sha(folder/'frozen_redistribution.csv')==next(r['sha256'] for r in frozen if r['team']==team)
  for name,B in [('original',P),('predicted',M),('actual',A),('later_present',LP)]:dump_matrix(B,ids,folder/(name+'_evaluation.csv'))
  row={k:v for k,v in z.items() if not isinstance(v,list)};row.update(status='verified_comparison' if A.sum()>0 else 'no_later_observed_absence',evaluation_source=e['source']);results.append(row)
 pd.DataFrame(results).to_csv(out/'coverage_and_hubs.csv',index=False);from hub_redistribution_report import report;report(root,out,a.skip_network_plots);return 0
if __name__=='__main__':raise SystemExit(main())
