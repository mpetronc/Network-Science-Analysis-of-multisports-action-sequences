"""Prior-season-defined star nonappearance study: five follow-up seasons, observational."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np,pandas as pd
from scipy.stats import t
from nba_api.stats.static import teams
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

SEASONS=['2020-21','2021-22','2022-23','2023-24','2024-25','2025-26']

def holm(frame,p='p',column='holm_p'):
 order=frame[p].sort_values().index;frame.loc[order,column]=np.minimum(1,np.maximum.accumulate([(len(order)-j)*frame.loc[i,p] for j,i in enumerate(order)]));return frame

def cluster_cov(X,res,groups,rank_full):
 groups=np.asarray(groups);unique,inverse=np.unique(groups,return_inverse=True);scores=np.zeros((len(unique),X.shape[1]));np.add.at(scores,inverse,X*res[:,None]);bread=np.linalg.inv(X.T@X);G=len(unique);n=len(res)
 return bread@(scores.T@scores)@bread*G/(G-1)*(n-1)/(n-rank_full)

def fit_panel(d,traits,with_trends=True):
 # Residualize FE/trends and pregame controls jointly, then regress treatment/interactions.
 groups=pd.Categorical(d.unit);D=np.eye(len(groups.categories))[groups.codes];trend=(d.game_number.to_numpy(float)-41)/41
 controls=['home','b2b','rest','opp_b2b','opp_def_pregame','other_missing','opponent_missing'];C=d[controls].to_numpy(float);C[:,4]=(C[:,4]-110)/10
 Z=np.column_stack([D,D*trend[:,None],C]) if with_trends else np.column_stack([D,C]);absent=d.absent.to_numpy(float);X=np.column_stack([absent]+[absent*d[f'{v}_z'].to_numpy(float) for v in traits]);y=d.off_rating.to_numpy(float)
 coef=np.linalg.lstsq(Z,np.column_stack([y,X]),rcond=None)[0];residual=np.column_stack([y,X])-Z@coef;yr=residual[:,0];XR=residual[:,1:];assert np.linalg.matrix_rank(XR)==XR.shape[1]
 b=np.linalg.lstsq(XR,yr,rcond=None)[0];fullZ=np.linalg.lstsq(Z,y-X@b,rcond=None)[0];res=y-X@b-Z@fullZ;rank=np.linalg.matrix_rank(Z)+len(b)
 club=cluster_cov(XR,res,d.team.to_numpy(),rank);game=cluster_cov(XR,res,d.GAME_ID.to_numpy(),rank);intersection=cluster_cov(XR,res,(d.team+'_'+d.GAME_ID).to_numpy(),rank);two=club+game-intersection;player=cluster_cov(XR,res,d.star_id.to_numpy(),rank);clubplayer=cluster_cov(XR,res,(d.team+'_'+d.star_id.astype(str)).to_numpy(),rank);team_player=club+player-clubplayer
 # Record both; primary franchise clustering (30 possible franchises), two-way sensitivity.
 G=d.team.nunique();df=G-1;names=['absence']+['absence_x_'+v for v in traits];rows=[]
 for j,name in enumerate(names):
  se=float(np.sqrt(max(club[j,j],0)));critical=t.ppf(.975,df);s2=float(np.sqrt(two[j,j])) if two[j,j]>0 else np.nan;sp=float(np.sqrt(team_player[j,j])) if team_player[j,j]>0 else np.nan
  rows.append(dict(term=name,n_games=len(d),team_seasons=d.unit.nunique(),franchises=G,coefficient_absence_effect=b[j],dependency_penalty=-b[j],se=se,p=2*t.sf(abs(b[j]/se),df),penalty_ci_low=-b[j]-critical*se,penalty_ci_high=-b[j]+critical*se,two_way_se=s2,two_way_p=2*t.sf(abs(b[j]/s2),df) if s2>0 else np.nan,team_player_se=sp,team_player_p=2*t.sf(abs(b[j]/sp),df) if sp>0 else np.nan,trends=with_trends,df=df,full_model_r_squared=1-np.sum(res**2)/np.sum((y-y.mean())**2),absence_block_partial_r_squared=1-np.sum(res**2)/np.sum(yr**2)))
 return pd.DataFrame(rows),b,club,y-Z@fullZ,res

def prepare(root,cache,out):
 base=[];adv=[];players=[];sources=[];teamids={v['abbreviation']:v['id'] for v in teams.get_teams()}
 for season in SEASONS:
  frames=[]
  for kind in ['team_base','team_advanced','player_base']:
   f=cache/f'{season}_{kind}.csv.gz';v=pd.read_csv(f,dtype={'GAME_ID':str});v.GAME_ID=v.GAME_ID.str.zfill(10);v['season']=season;frames.append(v);sources.append(dict(path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
  b,a,p=frames;a=a[['season','TEAM_ID','GAME_ID','POSS','OFF_RATING']];g=b.merge(a,on=['season','TEAM_ID','GAME_ID'],validate='one_to_one');assert len(g)==len(b) and g.POSS.gt(0).all()
  sums=p.groupby(['TEAM_ID','GAME_ID'])[['PTS','FGA','FGM']].sum().reindex(pd.MultiIndex.from_frame(g[['TEAM_ID','GAME_ID']]))
  assert np.array_equal(sums.to_numpy(),g[['PTS','FGA','FGM']].to_numpy()),f'Incomplete player logs {season}'
  g['date']=pd.to_datetime(g.GAME_DATE).dt.normalize();g=g.sort_values(['TEAM_ID','date','GAME_ID']);g['game_number']=g.groupby('TEAM_ID').cumcount()+1;g['home']=g.MATCHUP.str.contains('vs.',regex=False).astype(int);gap=g.groupby('TEAM_ID').date.diff().dt.days;g['b2b']=gap.eq(1).astype(int);g['rest']=(gap-1).clip(0,3).fillna(3)
  opp=g[['GAME_ID','TEAM_ID','PTS','POSS','b2b','game_number']].rename(columns={'TEAM_ID':'opponent_id','PTS':'opponent_points','POSS':'opponent_possessions','b2b':'opp_b2b','game_number':'opponent_game_number'})
  g=g.merge(opp,on='GAME_ID');g=g[g.TEAM_ID!=g.opponent_id].copy();assert len(g)==len(b)
  base.append(g);players.append(p)
 games=pd.concat(base,ignore_index=True).sort_values(['season','TEAM_ID','date']);players=pd.concat(players,ignore_index=True);games['team']=games.TEAM_ID.map({v:k for k,v in teamids.items()});games['off_rating']=games.OFF_RATING
 # Strength is strictly pregame, shrunk to the immediately preceding season's defense.
 annual=games.groupby(['season','TEAM_ID'])[['opponent_points','opponent_possessions']].sum();annual['drtg']=100*annual.opponent_points/annual.opponent_possessions
 games['def_pregame']=np.nan
 for (season,tid),g in games.groupby(['season','TEAM_ID'],sort=False):
  prior=SEASONS[SEASONS.index(season)-1] if season!=SEASONS[0] else None
  rate=float(annual.loc[(prior,tid),'drtg']) if prior else 110.;points=g.opponent_points.cumsum()-g.opponent_points;poss=g.opponent_possessions.cumsum()-g.opponent_possessions
  games.loc[g.index,'def_pregame']=100*(points+1000*rate/100)/(poss+1000)
 lookup=games.set_index(['season','TEAM_ID','GAME_ID']).def_pregame
 games['opp_def_pregame']=[lookup.loc[(s,t,g)] for s,t,g in zip(games.season,games.opponent_id,games.GAME_ID)]
 metricsfile=root/'court_zone_networks/regressions/four_metrics.csv';metrics=pd.read_csv(metricsfile);sources.append(dict(path=str(metricsfile),sha256=hashlib.sha256(metricsfile.read_bytes()).hexdigest()));appearance=players.groupby(['season','TEAM_ID','GAME_ID']).PLAYER_ID.agg(set).to_dict();names=players.drop_duplicates('PLAYER_ID').set_index('PLAYER_ID').PLAYER_NAME.to_dict()
 priorinfo={};firstten={};opponent_keys={};prior_minutes=players.groupby(['season','PLAYER_ID']).MIN.sum()
 for season in SEASONS[1:]:
  prior=SEASONS[SEASONS.index(season)-1]
  for team,tid in teamids.items():
   g=games[(games.season==season)&(games.TEAM_ID==tid)];early=set(g[g.game_number<=10].GAME_ID);retained=set().union(*[appearance[(season,tid,x)] for x in early]);firstten[(season,tid)]=retained
   f=root/'full_roster_entropy_centrality/historical_full_rosters'/prior/team/'roster.csv';R=pd.read_csv(f);sources.append(dict(path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest()));z=metrics[(metrics.season==prior)&(metrics.team==team)].iloc[0]
   Afile=f.with_name('passing_matrix.csv');A=pd.read_csv(Afile,index_col=0);active=A.to_numpy().sum(0)+A.to_numpy().sum(1)>0;B=A.to_numpy()[np.ix_(active,active)];_,V=np.linalg.eigh(B+B.T);v=np.abs(V[:,-1]);v/=v.sum();ec=dict(zip(A.index.astype(int).to_numpy()[active],v));sources.append(dict(path=str(Afile),sha256=hashlib.sha256(Afile.read_bytes()).hexdigest()))
   priorinfo[(season,tid)]=(R,z,ec);keymins=prior_minutes.loc[prior].reindex(sorted(retained)).dropna().sort_values(ascending=False,kind='stable').head(3);opponent_keys[(season,tid)]=(keymins/keymins.sum()).to_dict()
 policies={};coverage=[]
 adjustedfile=root/'court_zone_networks/revised_methodology/revised_metrics.csv';adjusted=pd.read_csv(adjustedfile).set_index(['season','team']);sources.append(dict(path=str(adjustedfile),sha256=hashlib.sha256(adjustedfile.read_bytes()).hexdigest()))
 for policy in ['passing_hub','leading_shooter']:
  records=[]
  for (season,tid),(R,z,ec) in priorinfo.items():
   team=z.team;pid=int(z.passing_leader_id if policy=='passing_hub' else z.shooting_leader_id);star_name=names.get(pid,str(pid));g=games[(games.season==season)&(games.TEAM_ID==tid)].copy().sort_values('date');retained=firstten[(season,tid)];earlycut=g[g.game_number==10].date.iloc[0];other=players[(players.season==season)&(players.PLAYER_ID==pid)&(players.TEAM_ID!=tid)].copy();other['date']=pd.to_datetime(other.GAME_DATE).dt.normalize();already_other=bool((other.date<=earlycut).any());censor=other.date.min()-pd.Timedelta(days=14) if len(other) else pd.NaT
   ok=pid in retained and not already_other;reason='retained in first 10 games' if ok else 'prior leader not observed on team in first 10 games or already changed teams'
   if pd.notna(censor):g=g[g.date<censor]
   g=g[(g.game_number>10)&(g.opponent_game_number>10)].copy()
   if not ok:coverage.append(dict(policy=policy,season=season,team=team,star_id=pid,star_name=star_name,retained=False,eligible=False,reason=reason,present=0,absent=0,trade_censor=str(censor)));continue
   prior=SEASONS[SEASONS.index(season)-1];keymins=prior_minutes.loc[prior].reindex(sorted(retained-{pid})).dropna().sort_values(ascending=False,kind='stable').head(3);weights=(keymins/keymins.sum()).to_dict()
   g['absent']=[int(pid not in appearance[(season,tid,game)]) for game in g.GAME_ID];g['other_missing']=[sum(w for p,w in weights.items() if p not in appearance[(season,tid,game)]) for game in g.GAME_ID];g['opponent_missing']=[sum(w for p,w in opponent_keys[(season,opp)].items() if p not in appearance[(season,opp,game)]) for opp,game in zip(g.opponent_id,g.GAME_ID)]
   present=int(g.absent.eq(0).sum());absent=int(g.absent.eq(1).sum());eligible=present>=15 and absent>=5
   coverage.append(dict(policy=policy,season=season,team=team,star_id=pid,star_name=star_name,retained=True,eligible=eligible,reason='primary eligible' if eligible else 'fewer than 15 appearances or 5 nonappearances after landmark/censor',present=present,absent=absent,trade_censor=str(censor)))
   F=R.set_index('PLAYER_ID').FGA;shares=F/F.sum();mins=R.MIN/R.MIN.sum();g['star_id']=pid;g['star_name']=star_name;g['prior_season']=z.season;g['landmark_end_date']=earlycut;g['unit']=season+'_'+team;g['pre_ec']=z.passing_starness;g['pre_shooting_hhi']=z.shooting_hhi;g['shared_leader']=float(z.passing_leader_id==z.shooting_leader_id);g['star_ec_share']=ec.get(pid,0);g['star_fga_share']=float(shares.loc[pid]);g['pre_minutes_hhi']=float(mins@mins);adjust=adjusted.loc[(z.season,team)];g['pre_entropy_excess']=adjust.passing_entropy_excess;g['pre_ec_excess']=adjust.passing_ec_excess;earlygames=set(games[(games.season==season)&(games.TEAM_ID==tid)&(games.game_number<=10)].GAME_ID);earlyF=players[(players.season==season)&(players.TEAM_ID==tid)&players.GAME_ID.isin(earlygames)].groupby('PLAYER_ID').FGA.sum();q=earlyF/earlyF.sum();g['landmark_shooting_hhi']=float(q@q);g['prior_roster_continuity']=float(R[R.PLAYER_ID.isin(retained)].MIN.sum()/R.MIN.sum());g['policy']=policy;g['primary_eligible']=eligible;records.append(g)
  data=pd.concat(records,ignore_index=True)
  units=data[data.primary_eligible].drop_duplicates('unit');standards={}
  for col in ['pre_ec','pre_shooting_hhi','shared_leader','star_ec_share','star_fga_share','pre_minutes_hhi','pre_entropy_excess','pre_ec_excess','landmark_shooting_hhi','prior_roster_continuity']:
   mean=units[col].mean();sd=units[col].std(ddof=0) if col!='shared_leader' else 1.;data[col+'_z']=(data[col]-mean)/sd;standards[col]=dict(mean=mean,sd=sd)
  policies[policy]=(data,standards)
 games.to_csv(out/'pregame_team_data.csv.gz',index=False,compression='gzip');pd.DataFrame(coverage).to_csv(out/'eligibility.csv',index=False);pd.DataFrame(sources).to_csv(out/'source_manifest.csv',index=False)
 return policies,games,players,appearance,priorinfo

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--star-dependency-study',action='store_true');p.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'nba_passing_output');p.add_argument('--cache',type=Path);p.add_argument('--destination',type=Path);a=p.parse_args(argv);root=a.output_dir;cache=a.cache or root/'star_dependency_study/game_cache';out=a.destination or root/'star_dependency_study';out.mkdir(parents=True,exist_ok=True)
 policies,games,players,appearance,priorinfo=prepare(root,cache,out);allfits=[];sensitivity=[];specific=[];modelstore={};penalties=[];episodes=[]
 for policy,(data,standards) in policies.items():
  data.to_csv(out/f'{policy}_eligible_landmark_games.csv.gz',index=False,compression='gzip');(out/(policy+'_feature_scaling.json')).write_text(json.dumps(standards,indent=2));d=data[data.primary_eligible].copy().reset_index(drop=True)
  traits=['pre_ec','pre_shooting_hhi','shared_leader'];fit,b,cov,adjusted,res=fit_panel(d,traits);fit['policy']=policy;allfits.append(fit);d['context_adjusted_off_rating']=adjusted;modelstore[policy]=(d,fit,b,cov,standards);d.to_csv(out/f'{policy}_primary_games.csv.gz',index=False,compression='gzip')
  for label,sub,trends in [('no_team_specific_trends',d,False),('include_1_to_4_absence_units',data.groupby('unit').filter(lambda g:g.absent.eq(0).sum()>=15 and g.absent.eq(1).sum()>=1).copy().reset_index(drop=True),True),('exclude_high_other_absence',d[d.other_missing<=.5].copy().reset_index(drop=True),True),('exclude_last_10_games',d[d.game_number<=72].copy().reset_index(drop=True),True),('prior_roster_continuity_at_least_70pct',d[d.prior_roster_continuity>=.7].copy().reset_index(drop=True),True)]:
   f,*_=fit_panel(sub,traits,with_trends=trends);f['policy']=policy;f['sensitivity']=label;sensitivity.append(f)
  f,*_=fit_panel(d,['pre_ec','pre_shooting_hhi','shared_leader','pre_minutes_hhi']);f['policy']=policy;f['sensitivity']='absence_interaction_with_prior_minutes_concentration';sensitivity.append(f)
  f,*_=fit_panel(d,['star_ec_share','star_fga_share']);f['policy']=policy;specific.append(f)
  for cols,label in [(['pre_ec'],'EC_interaction_alone'),(['pre_ec','landmark_shooting_hhi','shared_leader'],'first_10_game_shooting_HHI'),(['pre_shooting_hhi'],'shooting_HHI_interaction_alone'),(['pre_entropy_excess','pre_ec_excess','pre_shooting_hhi','shared_leader'],'baseline_adjusted_passing_features')]:
   f,*_=fit_panel(d,cols);f['policy']=policy;f['sensitivity']=label;sensitivity.append(f)
  # Unit estimates are descriptive noisy points; primary inference is the one-stage panel above.
  for unit,g in d.groupby('unit'):
   X=np.column_stack([np.ones(len(g)),g.absent,g.game_number/82,g.home,g.b2b,g.rest,g.opp_b2b,(g.opp_def_pregame-110)/10,g.other_missing,g.opponent_missing]);v=g.off_rating.to_numpy();coef=np.linalg.lstsq(X,v,rcond=None)[0];err=v-X@coef;bread=np.linalg.pinv(X.T@X);hat=np.sum(X*(X@bread),1);u=err/np.maximum(1-hat,1e-6);cv=bread@(X.T@(X*u[:,None]**2))@bread;se=np.sqrt(max(cv[1,1],0));nuisance=np.delete(X,1,axis=1);rx=g.absent.to_numpy()-nuisance@np.linalg.lstsq(nuisance,g.absent,rcond=None)[0];identified=bool(np.sum(rx**2)>1e-8);
   if not identified:coef[1]=np.nan;se=np.nan
   z=g.iloc[0];penalties.append(dict(policy=policy,unit=unit,season=z.season,team=z.team,star_id=z.star_id,star_name=z.star_name,present=int(g.absent.eq(0).sum()),absent=int(g.absent.eq(1).sum()),pre_ec=z.pre_ec,pre_shooting_hhi=z.pre_shooting_hhi,individually_identified=identified,penalty=-coef[1],se=se,ci_low=-coef[1]-t.ppf(.975,len(g)-np.linalg.matrix_rank(X))*se,ci_high=-coef[1]+t.ppf(.975,len(g)-np.linalg.matrix_rank(X))*se))
  # Consecutive nonappearance episodes, with a 5-game preceding window (>=4 appearances).
  for unit,g in d.groupby('unit'):
   g=g.sort_values('game_number').copy();numbers=g.game_number.to_numpy();flags=g.absent.to_numpy();start=0
   while start<len(g):
    if flags[start]==0:start+=1;continue
    end=start+1
    while end<len(g) and flags[end]==1 and numbers[end]==numbers[end-1]+1:end+=1
    run=g.iloc[start:end];pre=g.iloc[max(0,start-5):start];post=g.iloc[end:end+5]
    if len(run)>=2 and len(pre)==5 and pre.absent.sum()<=1 and pre.game_number.iloc[-1]==run.game_number.iloc[0]-1 and np.diff(pre.game_number).max()==1:
     z=run.iloc[0];validpost=len(post)==5 and post.absent.sum()<=1 and post.game_number.iloc[0]==run.game_number.iloc[-1]+1 and np.diff(post.game_number).max()==1
     episodes.append(dict(policy=policy,unit=unit,season=z.season,team=z.team,star_id=z.star_id,star_name=z.star_name,start_date=str(z.date.date()),absence_games=len(run),pre_games=';'.join(pre.GAME_ID),absence_game_ids=';'.join(run.GAME_ID),return_games=';'.join(post.GAME_ID) if validpost else '',adjusted_absence_loss=pre.context_adjusted_off_rating.mean()-run.context_adjusted_off_rating.mean(),adjusted_pre_level=pre.context_adjusted_off_rating.mean(),adjusted_absence_level=run.context_adjusted_off_rating.mean(),adjusted_return_level=post.context_adjusted_off_rating.mean() if validpost else np.nan,raw_absence_loss=pre.off_rating.mean()-run.off_rating.mean(),right_censored=bool(end==len(g)),return_window_complete=bool(validpost)))
    start=end
 fit=holm(pd.concat(allfits,ignore_index=True));critical=t.ppf(1-.05/(2*len(fit)),fit.df);fit['bonferroni95_penalty_ci_low']=fit.dependency_penalty-critical*fit.se;fit['bonferroni95_penalty_ci_high']=fit.dependency_penalty+critical*fit.se;fit.to_csv(out/'primary_panel_results.csv',index=False);pd.concat(sensitivity,ignore_index=True).to_csv(out/'panel_sensitivities.csv',index=False);holm(pd.concat(specific,ignore_index=True)).to_csv(out/'star_specific_share_models.csv',index=False);pd.DataFrame(penalties).to_csv(out/'team_season_penalties.csv',index=False);ep=pd.DataFrame(episodes);ep.to_csv(out/'absence_episodes.csv',index=False)
 with PdfPages(out/'star_dependency_results.pdf') as pdf:
  fig,ax=plt.subplots(figsize=(11,5));points=[]
  for policy,label in [('passing_hub','Prior-season passing hub'),('leading_shooter','Prior-season leading shooter')]:
   f=fit[(fit.policy==policy)&(fit.term=='absence')].iloc[0];points.append((label,f))
  for i,(label,f) in enumerate(points):ax.errorbar(f.dependency_penalty,i,xerr=[[f.dependency_penalty-f.penalty_ci_low],[f.penalty_ci_high-f.dependency_penalty]],fmt='o',capsize=5,label=f'{label}: {f.dependency_penalty:.2f} points/100, Holm p={f.holm_p:.3g}')
  ax.set_yticks(range(2),[x[0] for x in points]);ax.set_ylim(-.6,1.6);ax.axvline(0,c='grey',ls='--');ax.set_xlabel('Estimated offensive decline when prior-defined leader does not appear');ax.set_title('Five-season leader nonappearance study\nAdjusted observational estimates; 95% franchise-clustered intervals');ax.legend(fontsize=8,loc='lower right');fig.tight_layout();fig.savefig(out/'pooled_absence_penalties.png',dpi=150);pdf.savefig(fig);plt.close(fig)
  for policy,(d,f,b,cov,standards) in modelstore.items():
   fig,axes=plt.subplots(1,2,figsize=(15,6));q=pd.DataFrame(penalties);q=q[q.policy==policy]
   for j,(x,title) in enumerate([('pre_ec','Prior-season passing EC concentration'),('pre_shooting_hhi','Prior-season shooting HHI')]):
    ax=axes[j];ax.scatter(q[x],q.penalty,s=22,alpha=.6);xx=np.linspace(d[x].min(),d[x].max(),100);X=np.zeros((len(xx),4));X[:,0]=1;X[:,j+1]=(xx-standards[x]['mean'])/standards[x]['sd'];pred=-X@b;se=np.sqrt(np.maximum(np.einsum('ij,jk,ik->i',X,cov,X),0));critical=t.ppf(.975,d.team.nunique()-1);ax.plot(xx,pred,c='#ba4337');ax.fill_between(xx,pred-critical*se,pred+critical*se,color='#ba4337',alpha=.12);ax.axhline(0,c='grey',lw=.8);ax.set_xlabel(title);ax.set_ylabel('Offensive absence penalty (points/100)');test=f.iloc[j+1];adjusted_p=fit[(fit.policy==policy)&(fit.term==test.term)].holm_p.iloc[0];ax.set_title(f'Interaction raw p={test.p:.3g}; Holm p={adjusted_p:.3g}\nPenalty change per 1 SD: {test.dependency_penalty:.2f}')
   fig.suptitle(policy.replace('_',' ').title()+': does prior structure predict dependency?');fig.text(.5,.02,'Points: noisy within-team-season estimates. Red line: one-stage interaction model holding other features at reference means.\nTeam-season FE/trends; pregame opponent defense, venue, rest, own/opponent key availability. EC excludes shooting edges. Observational, not causal.',ha='center',fontsize=9);fig.tight_layout(rect=[0,.08,1,.93]);fig.savefig(out/(policy+'_structure_dependency.png'),dpi=150);pdf.savefig(fig);plt.close(fig)
  fig,axes=plt.subplots(1,2,figsize=(14,6))
  for ax,(policy,label) in zip(axes,[('passing_hub','Prior passing hub'),('leading_shooter','Prior leading shooter')]):
   e=ep[ep.policy==policy];complete=e.dropna(subset=['adjusted_return_level']);levels=complete[['adjusted_pre_level','adjusted_absence_level','adjusted_return_level']].to_numpy()
   for values in levels:ax.plot([0,1,2],values,c='grey',alpha=.12,lw=.6)
   if len(levels):ax.plot([0,1,2],levels.mean(0),c='#b64537',marker='o',lw=3)
   ax.set_xticks([0,1,2],['Preceding 5 games','Nonappearance run','5 return games']);ax.set_title(f'{label}: {len(complete)} complete episodes');ax.set_ylabel('Context-adjusted offensive rating (centered)')
  fig.suptitle('Episode patterns: descriptive before/absence/return comparisons');fig.text(.5,.02,'Not a causal event study: completed episodes are selected and lengths vary. Previous-period performance does not select episodes.\nContext adjustment from the panel model. Repeated episodes are related; individual lines are not independent estimates.',ha='center',fontsize=9);fig.tight_layout(rect=[0,.08,1,.93]);fig.savefig(out/'episode_patterns.png',dpi=150);pdf.savefig(fig);plt.close(fig)
 (out/'study_settings.json').write_text(json.dumps(dict(outcome_seasons=SEASONS[1:],star_definition='Prior-season EC leader (primary), prior-season FGA leader (secondary)',landmark=10,primary_min_present=15,primary_min_absent=5,trade_censor_days_before_other_team_appearance=14,pregame_defense_prior_possessions=1000,primary_control_trends='team-season-specific linear trend',clustering='franchise; two-way franchise/game sensitivity',definition='Recorded nonappearance, not confirmed injury or on/off-court possessions'),indent=2))
 from star_dependency_adaptation import main as adaptation_main
 adaptation_main(['--output-dir',str(root),'--study-dir',str(out),'--cache',str(cache)])
 from star_dependency_report import write_report
 write_report(out,root,cache)
 print('ELIGIBILITY\n'+pd.read_csv(out/'eligibility.csv').groupby(['policy','eligible']).size().to_string());print('PRIMARY\n'+fit[['policy','term','n_games','team_seasons','dependency_penalty','penalty_ci_low','penalty_ci_high','p','holm_p','two_way_p']].to_string(index=False));print('EPISODES\n'+ep.groupby('policy').size().to_string());return 0
if __name__=='__main__':raise SystemExit(main())
