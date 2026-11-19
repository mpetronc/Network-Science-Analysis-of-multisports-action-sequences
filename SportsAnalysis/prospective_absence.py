"""Frozen pre-absence passing predictions evaluated on two subsequent missed games."""
from pathlib import Path
import argparse,json,hashlib,shutil,inspect
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np,pandas as pd,networkx as nx
from scipy.spatial.distance import jensenshannon
from nba_api.stats.endpoints import playerdashptpass
from teamgraphs import NBAClient,Config,validate_passing
from corroborate_empty_profiles import main as corroborate
from star_presence_entropy import bits,active_n
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def digest(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def cap_allocate(weights,total=240,cap=48):
 weights=np.asarray(weights,float);use=weights>0;assert use.sum()*cap>=total;result=np.zeros(len(weights));remaining=float(total)
 while remaining>1e-8:
  active=use&(result==0);proposal=remaining*weights[active]/weights[active].sum();index=np.flatnonzero(active);capped=proposal>=cap
  if not capped.any():result[index]=proposal;break
  result[index[capped]]=cap;remaining-=cap*capped.sum()
 assert np.isclose(result.sum(),total) and (result<=cap+1e-8).all();return result

def pre_minutes(pre,history,star,ids):
 recent=pre.groupby('PLAYER_ID').MIN.sum().reindex(ids,fill_value=0).to_numpy(float)/5;absids=history.attrs['absence_game_ids'];historical=history[history.GAME_ID.isin(absids)].groupby('PLAYER_ID').MIN.sum().reindex(ids,fill_value=0).to_numpy(float);weights=(historical+5*recent)/(len(absids)+5);eligible=recent>0;eligible[np.array(ids)==star]=False;weights[~eligible]=0;return cap_allocate(weights),recent,weights,historical

def network_predict(D,mp,minutes):
 rate=np.divide(D.sum(1),mp,out=np.zeros(len(mp)),where=mp>0);target=rate*minutes;M=np.zeros_like(D,float);fallback=np.zeros(len(mp),bool)
 for i in range(len(mp)):
  if target[i]<=0:continue
  allowed=minutes>0;allowed[i]=False;k=D[i].copy();k[~allowed]=0
  if k.sum()==0:k=minutes*allowed;fallback[i]=True
  if k.sum()>0:M[i]=target[i]*k/k.sum()
 return M,rate,fallback

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prospective-absence',action='store_true');p.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'nba_passing_output');p.add_argument('--destination',type=Path);p.add_argument('--skip-network-plots',action='store_true');p.add_argument('--star-source',type=Path);p.add_argument('--star-definition',default='prior-season on/off-selected player');a=p.parse_args(argv);root=a.output_dir;out=a.destination or root/'onoff_dependency/prospective_absence';out.mkdir(parents=True,exist_ok=True);starfile=a.star_source or root/'onoff_dependency/selected_stars.csv';stars=pd.read_csv(starfile);P=pd.read_csv(root/'2025-26/team_season_topics/player_games.csv',dtype={'GAME_ID':str});G=pd.read_csv(root/'2025-26/team_season_topics/team_games.csv',dtype={'GAME_ID':str});P.GAME_ID=P.GAME_ID.str.zfill(10);G.GAME_ID=G.GAME_ID.str.zfill(10);G=G.sort_values(['GAME_DATE','GAME_ID']);daily=pd.concat([pd.read_csv(f).assign(date=f.stem) for f in (root/'2025-26/team_season_topics/daily_passing').glob('*.csv')],ignore_index=True);coverage=[];episodes=[];plan=[];pd.DataFrame([dict(path=str(f),sha256=digest(f)) for f in [starfile,root/'2025-26/team_season_topics/player_games.csv',root/'2025-26/team_season_topics/team_games.csv']]).to_csv(out/'source_manifest.csv',index=False)
 for z in stars.itertuples():
  g=G[G.TEAM_ID.eq(z.team_id)&G.GAME_DATE.between(z.window_start,z.window_end)].sort_values(['GAME_DATE','GAME_ID']).reset_index(drop=True);appearance=set(P[P.TEAM_ID.eq(z.team_id)&P.PLAYER_ID.eq(z.star_id)].GAME_ID);present=g.GAME_ID.isin(appearance).to_numpy();start=None
  for i in range(20,len(g)-1):
   if not present[i] and not present[i+1] and present[i-1] and present[i-5:i].sum()>=3:start=i;break
  if start is None:coverage.append(dict(team=z.team,star_name=z.star_name,status='no_eligible_two_game_episode_after_game20_with_3of5_pre_presence'));continue
  pre=g.iloc[start-5:start];test=g.iloc[start:start+2];history=g.iloc[max(0,start-20):start];earlier_abs=history.loc[~history.GAME_ID.isin(appearance),'GAME_ID'].tolist();pidlist=sorted(set(P[P.TEAM_ID.eq(z.team_id)&P.GAME_ID.isin(pre.GAME_ID)].PLAYER_ID)|{z.star_id});episode=dict(team=z.team,team_id=z.team_id,star_id=z.star_id,star_name=z.star_name,start_date=test.GAME_DATE.iloc[0],pre_start=pre.GAME_DATE.iloc[0],pre_end=pre.GAME_DATE.iloc[-1],test_start=test.GAME_DATE.iloc[0],test_end=test.GAME_DATE.iloc[-1],pre_game_ids=pre.GAME_ID.tolist(),test_game_ids=test.GAME_ID.tolist(),history_game_ids=history.GAME_ID.tolist(),earlier_absence_game_ids=earlier_abs,baseline_star_appearances=int(present[start-5:start].sum()),prediction_player_ids=pidlist);assert episode['pre_end']<episode['test_start'];episodes.append(episode);coverage.append(dict(team=z.team,star_name=z.star_name,status='planned',start_date=episode['start_date']))
  for kind,block in [('pre',pre),('test',test)]:
   for pid,part in P[P.TEAM_ID.eq(z.team_id)&P.GAME_ID.isin(block.GAME_ID)].groupby('PLAYER_ID'):
    f=daily[daily.TEAM_ID.eq(z.team_id)&daily.PLAYER_ID.eq(pid)&daily.date.between(block.GAME_DATE.iloc[0],block.GAME_DATE.iloc[-1])];plan.append(dict(team=z.team,team_id=z.team_id,kind=kind,player_id=int(pid),start=block.GAME_DATE.iloc[0],end=block.GAME_DATE.iloc[-1],expected_games=len(part),daily_observed_games=len(f),daily_passes=float(f.PASSES_MADE.sum())))
 (out/'episode_plan.json').write_text(json.dumps(episodes,indent=2));pd.DataFrame(coverage).to_csv(out/'coverage_all_30.csv',index=False);pd.DataFrame(plan).to_csv(out/'network_download_plan.csv',index=False);print('Eligible episodes',len(episodes),'profiles',len(plan),flush=True)
 caches=[root/'creator_dependency/prospective_absence/passing_api_cache',root/'ball_control_dependency/passing_api_cache',root/'onoff_dependency/prospective_absence/passing_api_cache',root/'creator_dependency/passing_api_cache',root/'onoff_dependency/passing_api_cache',root/'ec_selected_dependency/passing_api_cache',root/'volume_selected_dependency/passing_api_cache',root/'league_star_presence_entropy/api_cache',root/'2025-26/team_season_topics/hub_absence_resilience/cache'];records=[]
 def fetch(r):
  file=out/'passing_profiles'/r['team']/(r['start']+'_'+r['end'])/(str(r['player_id'])+'.csv');file.parent.mkdir(parents=True,exist_ok=True);audit=file.with_suffix('.json')
  if file.exists() and audit.exists():
   info=json.loads(audit.read_text())
   if info.get('complete'):return info
  c=NBAClient(Config(season='2025-26',cache_dir=out/'passing_api_cache',timeout=25,retries=2,min_delay=.5,max_delay=1));params=dict(player_id=r['player_id'],team_id=r['team_id'],season='2025-26',season_type_all_star='Regular Season',per_mode_simple='Totals',date_from_nullable=pd.Timestamp(r['start']).strftime('%m/%d/%Y'),date_to_nullable=pd.Timestamp(r['end']).strftime('%m/%d/%Y'))
  def validate(payload):
   f=validate_passing(payload,r['player_id'],r['team_id'])
   if len(f) and f.G.gt(r['expected_games']).any():raise ValueError('Profile GP exceeds window appearances')
   return f
  try:
   query=playerdashptpass.PlayerDashPtPass(**params,get_request=False);target=c.cache.path_for(query.endpoint,dict(query.parameters))
   if not target.exists():
    for cache in caches:
     candidate=cache/target.name
     if candidate.exists():target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(candidate,target);break
   f,_=c.fetch(playerdashptpass.PlayerDashPtPass,params,validate);zero=r['daily_observed_games']==r['expected_games'] and r['daily_passes']==0;complete=len(f)>0 or zero;error='' if complete else 'Empty profile without independently verified zero'
  except Exception as exc:f=pd.DataFrame(columns=['PASS_TEAMMATE_PLAYER_ID','PASS']);complete=False;error=repr(exc)
  finally:c.close()
  f.to_csv(file,index=False);info=dict(r,complete=complete,error=error,profile_passes=float(f.PASS.sum()));audit.write_text(json.dumps(info,indent=2));return info
 def download(kind):
  tasks=[r for r in plan if r['kind']==kind]
  with ThreadPoolExecutor(max_workers=6) as pool:
   for i,f in enumerate(as_completed([pool.submit(fetch,r) for r in tasks]),1):
    records.append(f.result())
    if i%50==0 or i==len(tasks):pd.DataFrame(records).to_csv(out/'download_progress.csv',index=False);print(kind,i,'/',len(tasks),'complete',sum(x['complete'] for x in records),flush=True)
  corroborate(['--data-dir',str(out)])
 def matrix(e,kind,ids):
  tasks=[r for r in plan if r['team']==e['team'] and r['kind']==kind];M=np.zeros((len(ids),len(ids)));ix={pid:i for i,pid in enumerate(ids)}
  for r in tasks:
   f=out/'passing_profiles'/r['team']/(r['start']+'_'+r['end'])/(str(r['player_id'])+'.csv');audit=json.loads(f.with_suffix('.json').read_text());assert audit['complete'],str(f)
   for edge in pd.read_csv(f).itertuples():
    if edge.PASS>0:assert int(edge.PASS_TEAMMATE_PLAYER_ID) in ix;M[ix[r['player_id']],ix[int(edge.PASS_TEAMMATE_PLAYER_ID)]]+=edge.PASS
  assert np.isfinite(M).all() and (M>=0).all() and not np.diag(M).any();return M
 # Stage 1: only pre-window passing is downloaded; all predictions are frozen before
 # any test passing profile is requested by this run.
 download('pre');frozen=[];minute_rows=[];failed=[]
 for e in episodes:
  try:
   ids=e['prediction_player_ids'];B=matrix(e,'pre',ids);star=ids.index(e['star_id']);D=B.copy();D[star]=0;D[:,star]=0;pre=P[P.TEAM_ID.eq(e['team_id'])&P.GAME_ID.isin(e['pre_game_ids'])].copy();history=P[P.TEAM_ID.eq(e['team_id'])&P.GAME_ID.isin(e['history_game_ids'])].copy();history.attrs['absence_game_ids']=e['earlier_absence_game_ids'];assert pre.GAME_DATE.max()<e['start_date'] and history.GAME_DATE.max()<e['start_date'];minutes,recent,weights,historical=pre_minutes(pre,history,e['star_id'],ids);mp=pre.groupby('PLAYER_ID').MIN.sum().reindex(ids,fill_value=0).to_numpy(float);M,rate,fallback=network_predict(D,mp,minutes*2);folder=out/e['team'];folder.mkdir(exist_ok=True)
   for name,C in [('pre_original',B),('deletion_baseline',D),('frozen_prediction',M)]:pd.DataFrame(C,index=ids,columns=ids).to_csv(folder/(name+'.csv'))
   specification=dict(e,predicted_minutes_per_game=minutes.tolist(),recent_minutes_per_game=recent.tolist(),rotation_weights=weights.tolist(),previous_absence_minutes=historical.tolist(),prior_absence_games=len(e['earlier_absence_game_ids']),smoothing_pseudo_games=5,regulated_team_minutes=240,per_player_cap=48,baseline_outgoing_rates=rate.tolist(),recipient_fallback=fallback.tolist(),freeze_stage='Before test profile download',matrix_sha256=digest(folder/'frozen_prediction.csv'));(folder/'frozen_specification.json').write_text(json.dumps(specification,indent=2));frozen.append(e)
   for i,pid in enumerate(ids):minute_rows.append(dict(team=e['team'],player_id=pid,star=pid==e['star_id'],recent_minutes_per_game=recent[i],prior_absence_games=len(e['earlier_absence_game_ids']),prior_absence_total_minutes=historical[i],smoothed_weight=weights[i],predicted_minutes_per_game=minutes[i],passing_rate=rate[i],fallback=bool(fallback[i])))
  except Exception as exc:failed.append(dict(team=e['team'],stage='pre_freeze',error=repr(exc)))
 pd.DataFrame(minute_rows).to_csv(out/'frozen_minute_predictions.csv',index=False);(out/'freeze_manifest.json').write_text(json.dumps([dict(team=e['team'],sha256=digest(out/e['team']/'frozen_prediction.csv')) for e in frozen],indent=2));print('FROZEN predictions',len(frozen),'before test downloads',flush=True)
 # Stage 2: evaluation and an explicitly post-outcome actual-minutes benchmark.
 download('test');rows=[];evaluated=[];allocation=[];names=P.drop_duplicates('PLAYER_ID').set_index('PLAYER_ID').PLAYER_NAME.to_dict()
 for e in frozen:
  try:
   folder=out/e['team'];spec=json.loads((folder/'frozen_specification.json').read_text());assert digest(folder/'frozen_prediction.csv')==spec['matrix_sha256'];testplayers=P[P.TEAM_ID.eq(e['team_id'])&P.GAME_ID.isin(e['test_game_ids'])];ids=sorted(set(e['prediction_player_ids'])|set(testplayers.PLAYER_ID));ix={pid:i for i,pid in enumerate(ids)};F=pd.read_csv(folder/'frozen_prediction.csv',index_col=0);oldids=F.index.astype(int).tolist()
   def expand(name):
    old=pd.read_csv(folder/(name+'.csv'),index_col=0);return old.reindex(index=ids,columns=list(map(str,ids)),fill_value=0).to_numpy(float)
   D=expand('deletion_baseline');M=expand('frozen_prediction');B=expand('pre_original');A=matrix(e,'test',ids);hi=ix[e['star_id']];assert A[hi].sum()+A[:,hi].sum()==0;pre=P[P.TEAM_ID.eq(e['team_id'])&P.GAME_ID.isin(e['pre_game_ids'])];mp=pre.groupby('PLAYER_ID').MIN.sum().reindex(ids,fill_value=0).to_numpy(float);actualminutes=testplayers.groupby('PLAYER_ID').MIN.sum().reindex(ids,fill_value=0).to_numpy(float);oracle,orate,ofallback=network_predict(D,mp,actualminutes);assert min(D.sum(),M.sum(),A.sum(),oracle.sum())>0;den=np.log2(active_n(B+A)-1)
   def tv(C):return float(abs(C/C.sum()-A/A.sum()).sum()/2)
   pred_minutes=pd.Series(spec['predicted_minutes_per_game'],index=e['prediction_player_ids']).reindex(ids,fill_value=0).to_numpy();actualpg=actualminutes/2;actualshare=actualpg/actualpg.sum();predshare=pred_minutes/240;row=dict(team=e['team'],star_name=e['star_name'],start_date=e['start_date'],baseline_star_appearances=e['baseline_star_appearances'],prior_absence_games=spec['prior_absence_games'],baseline_edge_tv=tv(D),frozen_edge_tv=tv(M),actual_minutes_benchmark_edge_tv=tv(oracle),frozen_improvement=tv(D)-tv(M),minute_share_tv=float(abs(predshare-actualshare).sum()/2),actual_team_minutes_per_game=actualpg.sum(),unseen_actual_player_pass_share=float(A.sum(1)[[pid not in oldids for pid in ids]].sum()/A.sum()),frozen_entropy_error=float(bits(M)-bits(A))/den,baseline_entropy_error=float(bits(D)-bits(A))/den,actual_minutes_entropy_error=float(bits(oracle)-bits(A))/den,first_test_date=e['test_start'],pre_last_date=e['pre_end']);rows.append(row);evaluated.append((e,ids,D,M,oracle,A));pd.DataFrame(A,index=ids,columns=ids).to_csv(folder/'actual_test_matrix.csv');pd.DataFrame(oracle,index=ids,columns=ids).to_csv(folder/'actual_minutes_benchmark.csv')
   for i,pid in enumerate(ids):allocation.append(dict(team=e['team'],player_id=pid,player_name=names.get(pid,str(pid)),predicted_minutes_per_game=pred_minutes[i],actual_minutes_per_game=actualpg[i],error_minutes_per_game=pred_minutes[i]-actualpg[i],known_before_episode=pid in oldids))
   assert digest(folder/'frozen_prediction.csv')==spec['matrix_sha256']
  except Exception as exc:failed.append(dict(team=e['team'],stage='test_scoring',error=repr(exc)))
 d=pd.DataFrame(rows);d.to_csv(out/'episode_results.csv',index=False);pd.DataFrame(allocation).to_csv(out/'minute_forecast_errors.csv',index=False);pd.DataFrame(failed,columns=['team','stage','error']).to_csv(out/'excluded_data_failures.csv',index=False);summary=dict(listed_teams=len(stars),star_definition=a.star_definition,eligible_episodes=len(episodes),frozen_predictions=len(frozen),evaluated_episodes=len(d),median_baseline_edge_tv=float(d.baseline_edge_tv.median()),median_frozen_edge_tv=float(d.frozen_edge_tv.median()),median_actual_minutes_benchmark_edge_tv=float(d.actual_minutes_benchmark_edge_tv.median()),frozen_improved_teams=int(d.frozen_improvement.gt(1e-8).sum()),frozen_tied_teams=int(d.frozen_improvement.abs().le(1e-8).sum()),frozen_worsened_teams=int(d.frozen_improvement.lt(-1e-8).sum()),median_paired_improvement=float(d.frozen_improvement.median()),mean_paired_improvement=float(d.frozen_improvement.mean()),median_minute_share_tv=float(d.minute_share_tv.median()));(out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
 if not a.skip_network_plots:
  with PdfPages(out/'frozen_predictions_and_actual.pdf') as pdf:
   for e,ids,D,M,O,A in evaluated:
    G=nx.from_numpy_array((D+D.T+A+A.T)>0);G=G.subgraph(np.flatnonzero((D+A).sum(0)+(D+A).sum(1)>0)).copy();opts=dict(seed=42,weight=None,iterations=350)
    if 'method' in inspect.signature(nx.spring_layout).parameters:opts['method']='force'
    pos=nx.spring_layout(G,**opts);coords=np.array(list(pos.values()));lo=coords.min(0)-.25;hi=coords.max(0)+.25;fig,axes=plt.subplots(2,2,figsize=(19,14));scale=max((1000*C/C.sum()).max() for C in [D,M,O,A])
    for ax,C,title in zip(axes.flat,[D,M,O,A],['Pre-window deletion baseline','Frozen: pre-absence minutes prediction','Actual-minutes benchmark (outcome-known)','Actual next two missed games']):
     active=np.flatnonzero(C.sum(0)+C.sum(1)>0);H=nx.DiGraph();H.add_nodes_from(active);edges=[(i,j) for i in active for j in active if C[i,j]>0];H.add_edges_from(edges);size=[12000*(C.sum(0)[i]+C.sum(1)[i])/(2*C.sum()) for i in H];nx.draw_networkx_edges(H,pos,edgelist=edges,width=[.1+4*1000*C[i,j]/C.sum()/scale for i,j in edges],alpha=.25,edge_color='#3f7f9e',arrows=True,arrowsize=8,node_size=size,connectionstyle='arc3,rad=.07',ax=ax);nx.draw_networkx_nodes(H,pos,node_size=size,node_color='#dcebef',edgecolors='#536b76',ax=ax);nx.draw_networkx_labels(H,pos,labels={i:str(names.get(ids[i],ids[i])).replace(' ','\n',1) for i in H},font_size=6.5,ax=ax);ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[1],hi[1]);ax.axis('off');ax.set_title(title)
    fig.suptitle(f"{e['team']} | {e['star_name']} | absence starts {e['start_date']}\nBaseline {e['pre_start']} to {e['pre_end']}; test {e['test_start']} to {e['test_end']}",fontsize=16);fig.text(.5,.02,'Frozen model uses only earlier minutes/passing; star assumed unavailable for both test games. New test players receive zero forecast minutes.\nActual-minutes benchmark uses test minutes for known baseline players, but no test passing edges. Node areas and edge widths normalized within context.',ha='center',fontsize=10);fig.tight_layout(rect=[0,.07,1,.94]);fig.savefig(out/e['team']/'forecast_networks.png',dpi=120);pdf.savefig(fig);plt.close(fig)
 fig,ax=plt.subplots(figsize=(11,9));x=d.baseline_edge_tv;y=d.frozen_edge_tv;ax.scatter(x,y)
 for z in d.itertuples():ax.annotate(z.team,(z.baseline_edge_tv,z.frozen_edge_tv),xytext=(4,3),textcoords='offset points',fontsize=8)
 limit=max(x.max(),y.max())+.03;ax.plot([0,limit],[0,limit],ls='--',c='grey');ax.set_xlabel('Pre-window deletion distance');ax.set_ylabel('Frozen pre-absence prediction distance');ax.set_title('Below diagonal: predicted minutes improve the two-game forecast');fig.tight_layout();fig.savefig(out/'forecast_comparison.pdf');fig.savefig(out/'forecast_comparison.png',dpi=150);plt.close(fig)
 text=['# Frozen pre-absence network test',f'One prespecified first qualifying two-game absence episode per team, after team game 20, with a star appearance in the immediately preceding game and ≥3 appearances in the preceding five team games. {len(stars)} candidate teams considered using {a.star_definition}; exclusions listed. Target spells are identified retrospectively for evaluation; the forecast is conditional on the star being out for both games. It does not forecast injury occurrence or duration.','## Inputs frozen before target passing download','Passing rates/preferences use only the preceding five TEAM games; some windows contain an earlier absence and are not purely star-present. Player candidates come only from these five games. The rotation history is the preceding 20 team games. Earlier-star-absence minutes are shrunk toward the current five-game rotation using five pseudo-games: weight_i = (sum of minutes in earlier missed games + 5 × recent mean minutes)/(number of earlier missed games + 5). This fixed convention is not tuned against test outcomes. With no earlier absences, proportional minute allocation can leave normalized passing shares unchanged; those cases are ties, not successful adaptation forecasts.','Set star weight to zero; exclude players with no positive minutes in the recent window. Allocate 240 regulation minutes proportionally to the remaining weights, iteratively cap any player at 48 and redistribute the remainder among uncapped players. No test-game minutes, participation list, or passes determine the frozen weights. Known future roster departures are not supplied; this conservative model can predict minutes for someone who will not appear. Unseen test players get zero.','Outgoing rate = pre-window nonstar outgoing passes / pre-window recorded minutes. Multiply by predicted minutes for two games, and retain pre-window recipient probabilities among forecast-active teammates. Unsupported recipient rows use predicted-minute weights; unknown baseline minute rates are zero, not learned from test outcomes. Prediction matrices and specifications are hashed and frozen BEFORE this run requests target passing. All date cutoffs and manifests are exported.','The original prior-star selection uses an early-season retained-first-ten check; this is not an opening-night prediction. Existing approximate trade-censor windows define the retrospective eligible cohort, not forecast features. Only strictly earlier rows enter minute prediction and passing estimation.','## Scoring','Directed-edge share total variation is 0 for identical shares and 1 for disjoint support. The frozen model, deletion baseline and actual-minutes benchmark use exactly the same short pre/test windows. Actual-minutes benchmark is explicitly post-outcome and cannot be used as evidence of forecasting accuracy. Its known-player rates remain pre-window-only; unseen actual players have zero rate.','This experiment evaluates two games per team; it is a noisy short-horizon pilot, not the earlier ≥10-missed-game season regression. No small-sample teams are inserted into the old regressions. The 240-minute prediction assumes regulation; actual overtime minutes are not known in advance.','## Results',json.dumps(summary,indent=2),'## Limitations','Only one spell per eligible team, fixed short baseline and two-game target. Other injuries, roster moves, opponent strength and shared-court opportunity are unmodelled. Aggregated totals cannot identify exact lineups. No hyperparameters selected by maximizing test fit; alternative models should be tested on separate future episodes. Full player and profile audits retained.']
 (out/'METHOD_AND_RESULTS.md').write_text('\n\n'.join(text));return 0
if __name__=='__main__':raise SystemExit(main())
