"""Remaining-player adaptation during prior-defined leader nonappearance episodes."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np,pandas as pd
from scipy.stats import t
from scipy.sparse.csgraph import connected_components
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from court_zone_networks import ZONES,classify_zones

def H(C):
 C=np.asarray(C,float);n=C.sum(-1);p=np.divide(C,n[...,None],out=np.zeros_like(C),where=n[...,None]>0);return -np.sum(p*np.log2(np.maximum(p,1e-300)),axis=-1)
def hhi(F):
 n=F.sum();return float(np.sum(F*(F-1))/(n*(n-1))) if n>1 else np.nan

def mean_cluster(g,column):
 y=g[column].dropna();z=g.loc[y.index];n=len(y);mu=float(y.mean());sums=(y-mu).groupby(z.team).sum();G=len(sums);se=float(np.sqrt(G/(G-1)*np.sum(sums**2)/n**2));critical=t.ppf(.975,G-1)
 return dict(n_episodes=n,franchises=G,mean=mu,se=se,ci_low=mu-critical*se,ci_high=mu+critical*se,p=2*t.sf(abs(mu/se),G-1))
def correct(d):
 order=d.p.sort_values().index;d.loc[order,'holm_p']=np.minimum(1,np.maximum.accumulate([(len(order)-j)*d.loc[i,'p'] for j,i in enumerate(order)]));return d

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'nba_passing_output');p.add_argument('--study-dir',type=Path);p.add_argument('--cache',type=Path);p.add_argument('--draws',type=int,default=200);a=p.parse_args(argv);root=a.output_dir;study=a.study_dir or root/'star_dependency_study';cache=a.cache or study/'game_cache';out=study/'adaptation';out.mkdir(parents=True,exist_ok=True);ep=pd.read_csv(study/'absence_episodes.csv',dtype={'pre_games':str,'absence_game_ids':str,'return_games':str});rng=np.random.default_rng(20261118);plotrng=np.random.default_rng(20261119);rows=[];sources=[];disagreements=[];excluded=[]
 for season,e in ep.groupby('season'):
  f=cache/f'{season}_player_base.csv.gz';P=pd.read_csv(f,dtype={'GAME_ID':str});P.GAME_ID=P.GAME_ID.str.zfill(10);names=P.drop_duplicates('PLAYER_ID').set_index('PLAYER_ID').PLAYER_NAME.to_dict();appear=P.groupby(['TEAM_ID','GAME_ID']).PLAYER_ID.agg(set).to_dict();needed=set(';'.join(e.pre_games.tolist()+e.absence_game_ids.tolist()).split(';'))
  shotfile=root/'shot_attempt_cache'/f'{season}_shot_attempts.csv.gz';shots=pd.read_csv(shotfile,dtype={'GAME_ID':str});shots.GAME_ID=shots.GAME_ID.str.zfill(10);shots=shots[shots.GAME_ID.isin(needed)].copy();shots['zone']=classify_zones(shots)
  C=shots.groupby(['TEAM_ID','GAME_ID','PLAYER_ID','zone']).size().unstack(fill_value=0).reindex(columns=ZONES,fill_value=0);lookup={key:g.droplevel([0,1]) for key,g in C.groupby(level=[0,1])};ids=P.groupby(['TEAM_ABBREVIATION']).TEAM_ID.first().to_dict()
  # Verify source shots match every used player-game FGA, including zero-attempt appearances.
  truth=P[P.GAME_ID.isin(needed)].set_index(['TEAM_ID','GAME_ID','PLAYER_ID']).FGA;observed=C.sum(axis=1).reindex(truth.index,fill_value=0);bad=truth.ne(observed);badpairs=set((int(tid),game) for tid,game,pid in truth.index[bad]);
  for (tid,game,pid),count in truth[bad].items():disagreements.append(dict(season=season,TEAM_ID=tid,GAME_ID=game,PLAYER_ID=pid,game_log_FGA=int(count),shot_chart_FGA=int(observed.loc[(tid,game,pid)])))
  for z in e.itertuples():
   tid=int(ids[z.team]);pre=z.pre_games.split(';');absent=z.absence_game_ids.split(';');
   if any((tid,g) in badpairs for g in pre+absent):
    excluded.append(dict(policy=z.policy,season=season,team=z.team,start_date=z.start_date,reason='Official shot chart/player game log FGA disagreement in compared team window'));continue
   assert all(int(z.star_id) not in appear[(tid,g)] for g in absent);preplayers=set().union(*[appear[(tid,g)] for g in pre]);absplayers=set().union(*[appear[(tid,g)] for g in absent]);common=sorted((preplayers&absplayers)-{int(z.star_id)});union=sorted((preplayers|absplayers)-{int(z.star_id)})
   def combine(games):
    frames=[lookup[(tid,g)] for g in games if (tid,g) in lookup];return pd.concat(frames).groupby(level=0).sum().reindex(index=union,columns=ZONES,fill_value=0).fillna(0)
   A=combine(pre);B=combine(absent);assert z.star_id not in A.index and z.star_id not in B.index
   Ac=A.reindex(common).to_numpy(np.int64);Bc=B.reindex(common).to_numpy(np.int64);F=Ac.sum(1);G=Bc.sum(1)
   if len(common)<3 or min(F.sum(),G.sum())<=1:continue
   entpre=np.zeros(a.draws);entabs=np.zeros(a.draws);positive=(F>0)&(G>0);take=np.minimum(F,G);take[~positive]=0;total=take.sum()
   if total:
    for i,n in enumerate(take):
     if n:
      sa=rng.multivariate_hypergeometric(Ac[i],int(n),size=a.draws);sb=rng.multivariate_hypergeometric(Bc[i],int(n),size=a.draws);entpre+=n/total*H(sa)/np.log2(len(ZONES));entabs+=n/total*H(sb)/np.log2(len(ZONES))
   preleader=common[int(np.argmax(F))];absleader=common[int(np.argmax(G))];rows.append(dict(policy=z.policy,season=season,team=z.team,unit=z.unit,star_id=z.star_id,star_name=z.star_name,start_date=z.start_date,absence_games=z.absence_games,right_censored=z.right_censored,pre_window_all_star_present=all(int(z.star_id) in appear[(tid,g)] for g in pre),common_nonstar_players=len(common),entropy_matched_shooters=int(positive.sum()),entropy_matched_attempts=int(total),pre_matched_attempts=int(F.sum()),absent_matched_attempts=int(G.sum()),pre_nonstar_attempt_coverage=F.sum()/A.to_numpy().sum(),absent_nonstar_attempt_coverage=G.sum()/B.to_numpy().sum(),pre_FGA_HHI=hhi(F),absent_FGA_HHI=hhi(G),FGA_HHI_change=hhi(G)-hhi(F),FGA_HHI_union_change=hhi(B.sum(axis=1).to_numpy())-hhi(A.sum(axis=1).to_numpy()),top1_share_change=G.max()/G.sum()-F.max()/F.sum(),shooting_zone_entropy_change=float((entabs-entpre).mean()) if total else np.nan,shooting_entropy_MC_sd=float((entabs-entpre).std(ddof=1)) if total else np.nan,pre_remaining_leader_id=preleader,absent_remaining_leader_id=absleader,pre_remaining_leader=names.get(preleader,str(preleader)),absent_remaining_leader=names.get(absleader,str(absleader)),remaining_leader_changed=preleader!=absleader,adjusted_absence_loss=z.adjusted_absence_loss))
  for file in [f,shotfile]:sources.append(dict(path=str(file),sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
 pd.DataFrame(disagreements,columns=['season','TEAM_ID','GAME_ID','PLAYER_ID','game_log_FGA','shot_chart_FGA']).to_csv(out/'shot_log_disagreements.csv',index=False);pd.DataFrame(excluded,columns=['policy','season','team','start_date','reason']).to_csv(out/'excluded_episode_windows.csv',index=False);data=pd.DataFrame(rows);data.to_csv(out/'episode_shooting_adaptation.csv',index=False);summary=[]
 for policy,g in data.groupby('policy'):
  for metric in ['FGA_HHI_change','FGA_HHI_union_change','top1_share_change','shooting_zone_entropy_change']:
   result=mean_cluster(g,metric);result.update(policy=policy,metric=metric);summary.append(result)
 summary=correct(pd.DataFrame(summary));summary.to_csv(out/'episode_adaptation_tests.csv',index=False)
 strict=[]
 for policy,g in data[data.pre_window_all_star_present].groupby('policy'):
  for metric in ['FGA_HHI_change','FGA_HHI_union_change','top1_share_change','shooting_zone_entropy_change']:
   result=mean_cluster(g,metric);result.update(policy=policy,metric=metric);strict.append(result)
 correct(pd.DataFrame(strict)).to_csv(out/'strict_pre_presence_sensitivity.csv',index=False)
 with PdfPages(out/'adaptation_results.pdf') as pdf:
  fig,axes=plt.subplots(1,2,figsize=(14,6))
  for ax,metric,title in [(axes[0],'FGA_HHI_change','Shooting concentration among common remaining players'),(axes[1],'shooting_zone_entropy_change','Count-matched shooting-zone diversity')]:
   for j,(policy,g) in enumerate(data.groupby('policy')):
    vals=g[metric].dropna();ax.scatter(np.full(len(vals),j)+plotrng.uniform(-.12,.12,len(vals)),vals,s=15,alpha=.3);q=summary[(summary.policy==policy)&(summary.metric==metric)].iloc[0];ax.errorbar(j,q['mean'],yerr=[[q['mean']-q.ci_low],[q.ci_high-q['mean']]],fmt='o',c='black',capsize=6)
   ax.axhline(0,c='grey',ls='--');ax.set_xticks([0,1],['Prior leading shooter','Prior passing hub']);ax.set_title(title);ax.set_ylabel('During absence minus preceding five games')
  fig.suptitle('What changes after removing the leader from BOTH comparisons?');fig.text(.5,.02,'Common nonstar teammates only. Entropy uses equal per-player FGA samples across windows; HHI uses pair-count correction.\nUncertainty for episode means clusters by franchise. Multiple episodes overlap and are not independent. Descriptive, not causal mediation.',ha='center',fontsize=9);fig.tight_layout(rect=[0,.09,1,.93]);fig.savefig(out/'shooting_adaptation.png',dpi=150);pdf.savefig(fig);plt.close(fig)
  # Passing adaptation is limited to compatible saved 2025–26 season-context networks.
  eligibility=pd.read_csv(study/'eligibility.csv');covfile=root/'offensive_structure_profiles/absence_validation_coverage.csv';cov=pd.read_csv(covfile);oldhub=cov.set_index('team').hub_id.to_dict();passing=[]
  for z in eligibility[(eligibility.policy=='passing_hub')&(eligibility.season=='2025-26')&eligibility.eligible].itertuples():
   if z.team not in oldhub or int(oldhub[z.team])!=int(z.star_id):continue
   fullfile=root/'2025-26/team_season_topics/networks'/z.team/'passing_matrix.csv';presentfile=root/'2025-26/team_season_topics/hub_absence_resilience/present_networks'/z.team/'passing_matrix.csv'
   if not fullfile.exists() or not presentfile.exists():continue
   full=pd.read_csv(fullfile,index_col=0);pres=pd.read_csv(presentfile,index_col=0);assert full.index.astype(str).tolist()==pres.index.astype(str).tolist() and full.columns.astype(str).tolist()==pres.columns.astype(str).tolist();ids=full.index.astype(int).tolist();PP=pres.to_numpy(float);AA=full.to_numpy(float)-PP;assert AA.min()>=-1e-8;AA=np.maximum(AA,0);star=ids.index(int(z.star_id));assert AA[star].sum()+AA[:,star].sum()==0;take=[i for i in range(len(ids)) if i!=star]
   while True:
    P=PP[np.ix_(take,take)];A=AA[np.ix_(take,take)];keep=(P.sum(1)>0)&(A.sum(1)>0)
    if keep.all():break
    take=[i for i,k in zip(take,keep) if k]
   if len(take)<3:continue
   samples=np.minimum(P.sum(1),A.sum(1)).astype(int);hp=[];ha=[];cp=[];ca=[]
   for rep in range(a.draws):
    R=np.array([rng.multivariate_hypergeometric(v.astype(int),int(n)) for v,n in zip(P,samples)]);S=np.array([rng.multivariate_hypergeometric(v.astype(int),int(n)) for v,n in zip(A,samples)]);hp.append(np.average(H(R),weights=samples)/np.log2(len(take)-1));ha.append(np.average(H(S),weights=samples)/np.log2(len(take)-1))
    if connected_components((R+R.T)>0,directed=False)[0]==1 and connected_components((S+S.T)>0,directed=False)[0]==1:
     vals=[]
     for M in [R,S]:
      _,V=np.linalg.eigh(M+M.T);v=np.abs(V[:,-1]);v/=v.sum();vals.append((v@v-1/len(v))/(1-1/len(v)))
     cp.append(vals[0]);ca.append(vals[1])
   passing.append(dict(team=z.team,season='2025-26',star_id=z.star_id,star_name=z.star_name,common_nonstar_players=len(take),matched_passes_per_context=int(samples.sum()),entropy_change=np.mean(ha)-np.mean(hp),EC_concentration_change=np.mean(ca)-np.mean(cp) if len(cp) else np.nan,valid_connected_EC_draws=len(cp),contexts='All saved hub-present vs hub-absent season games, not episode windows; same nonhub panel, equal outgoing counts'))
   for f in [fullfile,presentfile]:sources.append(dict(path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
  passed=pd.DataFrame(passing);passed.to_csv(out/'limited_passing_adaptation.csv',index=False)
  if len(passed)>=3:
   tests=[]
   for metric in ['entropy_change','EC_concentration_change']:
    r=mean_cluster(passed,metric);r['metric']=metric;tests.append(r)
   correct(pd.DataFrame(tests)).to_csv(out/'limited_passing_adaptation_tests.csv',index=False)
   fig,axes=plt.subplots(1,2,figsize=(13,5))
   for ax,metric,title in [(axes[0],'entropy_change','Count-matched passing entropy change'),(axes[1],'EC_concentration_change','Count-matched remaining-player EC concentration change')]:
    ax.bar(passed.team,passed[metric]);ax.axhline(0,c='grey');ax.tick_params(axis='x',rotation=60);ax.set_title(title,fontsize=11)
   fig.suptitle('Limited 2025–26 passing adaptation: prior-defined hubs matching existing caches');fig.text(.5,.015,'Hub removed in BOTH contexts. Same remaining-player panel; equal outgoing pass samples per player.\nThese are whole-season presence/absence contexts, not episode before/during networks; exposure and roles can still differ.',ha='center',fontsize=9);fig.tight_layout(rect=[0,.09,1,.9]);fig.savefig(out/'limited_passing_adaptation.png',dpi=150);pdf.savefig(fig);plt.close(fig)
 pd.DataFrame(sources).to_csv(out/'source_manifest.csv',index=False);(out/'adaptation_settings.json').write_text(json.dumps(dict(seed=20261118,rarefaction_draws=a.draws,shot_zones=15,common_player_rule='Nonstar players with a logged appearance in both pre and absence windows; positive FGA in both for count-matched entropy',passing_limit='2025–26 matching saved hub ID; season contexts not episode networks'),indent=2));print(summary.to_string(index=False));print('Passing compatible teams:',len(passing));print('Shot conflicts:',len(disagreements),'excluded episode windows:',len(excluded));return 0
if __name__=='__main__':raise SystemExit(main())
