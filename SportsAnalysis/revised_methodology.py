"""Activity/position-reference structure metrics, corrected inference and temporal checks."""
from pathlib import Path
import argparse,hashlib,itertools,json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import t
from entropy_structure_profiles import regress
from passing_null_test import expected_counts,metrics as null_metrics
from court_zone_networks import ZONES
from revised_methodology_context import context_fit,temporal_context,focal_figure

COLS={'passing_entropy_excess':'Passing diversity minus activity baseline','shooting_entropy_role_excess':'Shooting-zone diversity minus position baseline','passing_ec_excess':'Passing EC concentration minus activity baseline','shooting_hhi_unbiased':'Shooting concentration (FGA HHI, bias-corrected)'}
PAIRS=list(itertools.combinations(COLS,2));PRIMARY=('passing_entropy_excess','shooting_hhi_unbiased')

def holm(frame,p='robust_p',out='holm_p'):
 if len(frame)==0:return frame
 order=frame[p].sort_values().index;frame.loc[order,out]=np.minimum(1,np.maximum.accumulate([(len(order)-j)*frame.loc[i,p] for j,i in enumerate(order)]));return frame

def entropy_rows(C):
 C=np.asarray(C,float);tot=C.sum(-1);P=np.divide(C,tot[...,None],out=np.zeros_like(C),where=tot[...,None]>0)
 return -np.sum(P*np.log2(np.maximum(P,1e-300)),axis=-1)

def band(ax,g,x,y,label,color):
 fit,b,cov,_=regress(g,y,x=x);xx=np.linspace(g[x].min(),g[x].max(),120);X=np.column_stack([np.ones(len(xx)),xx-fit['entropy_mean']]);mu=X@b;se=np.sqrt(np.maximum(np.einsum('ij,jk,ik->i',X,cov,X),0));c=t.ppf(.975,fit['df']);ax.plot(xx,mu,color=color,label=label);ax.fill_between(xx,mu-c*se,mu+c*se,color=color,alpha=.1);return fit

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--revised-methodology',action='store_true');p.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'nba_passing_output');p.add_argument('--position-file',type=Path,default=Path(__file__).resolve().parent/'position_network_metadata/player_positions.csv');p.add_argument('--destination',type=Path);p.add_argument('--draws',type=int,default=500);args=p.parse_args(argv)
 root=args.output_dir;out=args.destination or root/'court_zone_networks/revised_methodology';out.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(20261117);sources=[]
 def record(file):sources.append(dict(path=str(file),sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
 metricfile=root/'court_zone_networks/regressions/four_metrics.csv';nullfolder=root/'fg_attempt_networks/structure_validation/null_model';nullfile=nullfolder/'team_null_metrics.csv';nullsource=nullfolder/'source_manifest.csv';nullnpz=nullfolder/'null_draws.npz'
 d=pd.read_csv(metricfile);null=pd.read_csv(nullfile);nullmanifest=pd.read_csv(nullsource);nullhashes={(Path(path).parent.parent.name,Path(path).parent.name):sha for path,sha in zip(nullmanifest.path,nullmanifest.sha256)};assert len(nullhashes)==180;draws=np.load(nullnpz);assert draws['entropy'].shape[0]==len(null)==180
 null['passing_entropy_half1']=null.observed_entropy-draws['entropy'][:,:250].mean(1);null['passing_entropy_half2']=null.observed_entropy-draws['entropy'][:,250:].mean(1)
 null['passing_ec_half1']=null.observed_ec-draws['ec'][:,:250].mean(1);null['passing_ec_half2']=null.observed_ec-draws['ec'][:,250:].mean(1)
 null['passing_entropy_MC_se']=draws['entropy'].std(1,ddof=1)/np.sqrt(draws['entropy'].shape[1]);null['passing_ec_MC_se']=draws['ec'].std(1,ddof=1)/np.sqrt(draws['ec'].shape[1])
 null=null.rename(columns={'entropy_excess':'passing_entropy_excess','ec_excess':'passing_ec_excess'})
 d=d.merge(null,on=['team','season'],validate='one_to_one');assert len(d)==180 and np.allclose(d.passing_entropy,d.observed_entropy) and np.allclose(d.passing_starness,d.observed_ec,atol=1e-8)
 for f in [metricfile,nullfile,nullsource,nullnpz,args.position_file]:record(f)
 positions=pd.read_csv(args.position_file);assert not positions.duplicated(['season','team','player_id']).any();poslookup=positions.set_index(['season','team','player_id'])
 assert set(positions.position)<=set(['PG','SG','SF','PF','C'])
 playerrows=[];matrices={};TRAIN_END='2024-25';TEST='2025-26';rolecounts={p:np.zeros(len(ZONES)) for p in ['PG','SG','SF','PF','C']};individual={}
 for row in d.itertuples():
  folder=root/'full_roster_entropy_centrality/historical_full_rosters'/row.season/row.team;matrixfile=folder/'passing_matrix.csv';rosterfile=folder/'roster.csv';zonefile=root/'court_zone_networks'/row.season/row.team/'player_zone_attempts.csv'
  assert hashlib.sha256(matrixfile.read_bytes()).hexdigest()==nullhashes[(row.season,row.team)],'Passing source changed since null baseline'
  A=pd.read_csv(matrixfile,index_col=0);R=pd.read_csv(rosterfile).set_index('PLAYER_ID').loc[A.index.astype(int)];C=pd.read_csv(zonefile,index_col=0).reindex(index=A.index.astype(int),columns=ZONES);assert np.isfinite(C.to_numpy()).all() and np.array_equal(C.sum(axis=1),R.FGA)
  matrices[(row.team,row.season)]=(A,R,C)
  for pid,counts in zip(C.index,C.to_numpy(float)):
   info=poslookup.loc[(row.season,row.team,pid)];assert float(info.fga)==counts.sum();role=info.position
   if row.season<=TRAIN_END:
    rolecounts[role]+=counts;individual[(int(pid),role)]=individual.get((int(pid),role),np.zeros(len(ZONES)))+counts
  for f in [matrixfile,rosterfile,zonefile]:record(f)
 pooled=sum(rolecounts.values());teamrows=[]
 # Every player's role reference excludes that player's entire development-period history.
 for row in d.itertuples():
  A,R,C=matrices[(row.team,row.season)];F=C.sum(axis=1).to_numpy(float);teamnull=np.zeros(args.draws);observed=[];shot_n=len(F[F>0]);total=F.sum();shares=F/total
  for pid,c in zip(C.index,C.to_numpy(float)):
   role=poslookup.loc[(row.season,row.team,pid),'position'];peer=rolecounts[role]-individual.get((int(pid),role),np.zeros(len(ZONES)));assert (peer>=0).all();player_global=sum((individual.get((int(pid),r),np.zeros(len(ZONES))) for r in rolecounts),np.zeros(len(ZONES)));global_peer=pooled-player_global;prior=(global_peer+1)/(global_peer.sum()+len(ZONES));prob=(peer+30*prior)/(peer.sum()+30);n=int(c.sum())
   obs=float(entropy_rows(c))/np.log2(len(ZONES))
   if n:
    sim=rng.multinomial(n,prob,size=args.draws);hs=entropy_rows(sim)/np.log2(len(ZONES));teamnull+=n/total*hs;observed.append(n/total*obs)
    playerrows.append(dict(team=row.team,season=row.season,player_id=pid,name=R.loc[pid,'PLAYER_NAME'],position=role,fga=n,observed_zone_entropy=obs,expected_position_entropy=hs.mean(),entropy_excess=obs-hs.mean(),peer_reference_shots=peer.sum()))
  minutes=R.MIN.to_numpy(float);assert np.isfinite(minutes).all() and (minutes>=0).all() and minutes.sum()>0;minutes_hhi=float(np.sum((minutes/minutes.sum())**2));rawhhi=float(shares@shares);unbiased=float(np.sum(F*(F-1))/(total*(total-1)));norm=(unbiased-1/shot_n)/(1-1/shot_n)
  assert np.isclose(sum(observed),row.shooting_entropy,atol=1e-12)
  teamrows.append(dict(team=row.team,season=row.season,shooting_entropy_role_excess=sum(observed)-teamnull.mean(),shooting_role_expected_entropy=teamnull.mean(),shooting_role_MC_se=teamnull.std(ddof=1)/np.sqrt(args.draws),shooting_role_half1=sum(observed)-teamnull[:args.draws//2].mean(),shooting_role_half2=sum(observed)-teamnull[args.draws//2:].mean(),shooting_hhi_unbiased=unbiased,shooting_hhi_normalized_unbiased=norm,shooting_hhi_raw=rawhhi,shooting_top1_share=shares.max(),shooting_top2_share=np.sort(shares)[-2:].sum(),effective_shooters=1/rawhhi,minutes_hhi=minutes_hhi))
 d=d.merge(pd.DataFrame(teamrows),on=['team','season'],validate='one_to_one');d['leader_alignment']=np.where(d.shared_passing_shooting_leader,'Shared leader','Separate leaders');d['sample']=np.where(d.season.eq(TEST),'Retrospective temporal test','Development')
 d.to_csv(out/'revised_metrics.csv',index=False);pd.DataFrame(playerrows).to_csv(out/'position_shooting_reference.csv',index=False)
 # Store frozen reference counts; smoothing is fixed in advance, not tuned for regression strength.
 pd.DataFrame(rolecounts,index=ZONES).to_csv(out/'development_position_zone_counts.csv');(out/'settings.json').write_text(json.dumps(dict(seed=20261117,shooting_null_draws=args.draws,passing_null_draws=500,development_end=TRAIN_END,test=TEST,role_smoothing_shots=30,primary_x=PRIMARY[0],primary_y=PRIMARY[1],null_model='outgoing exact, incoming in expectation; self-passes forbidden',holdout_status='retrospective; latest season was already explored earlier in project'),indent=2))
 fits=[];groups=[];interactions=[];fe=[];temporal=[];sensitivity=[]
 for x,y in PAIRS:
  fit,*_=regress(d,y,x=x);fit.update(analysis='pooled',primary=(x,y)==PRIMARY);fits.append(fit)
  for group,g in d.groupby('leader_alignment'):
   fit,*_=regress(g,y,x=x);fit.update(group=group);groups.append(fit)
  fit,*_=regress(d,y,x=x,interaction='alignment');interactions.append(fit)
  fit,*_=regress(d,y,x=x,season_fe=True);fe.append(fit)
  train=d[d.season<=TRAIN_END];test=d[d.season.eq(TEST)];fit,b,_,_=regress(train,y,x=x);pred=b[0]+b[1]*(test[x].to_numpy()-fit['entropy_mean']);base=np.full(len(test),train[y].mean());SSE=np.sum((test[y].to_numpy()-pred)**2);den=np.sum((test[y].to_numpy()-base)**2);testfit,*_=regress(test,y,x=x)
  temporal.append(dict(x=x,y=y,train_n=len(train),test_n=len(test),train_slope=b[1],train_r=fit['r'],test_r=testfit['r'],test_association_hc3_p=testfit['robust_p'],test_rmse=np.sqrt(SSE/len(test)),baseline_train_mean_rmse=np.sqrt(den/len(test)),predictive_r_squared_vs_train_mean=1-SSE/den,predictive_r_squared_vs_test_mean=1-SSE/np.sum((test[y]-test[y].mean())**2),status='retrospective temporal check, not untouched or prospective validation'))
  for half in [1,2]:
   xx={'passing_entropy_excess':f'passing_entropy_half{half}','shooting_entropy_role_excess':f'shooting_role_half{half}','passing_ec_excess':f'passing_ec_half{half}'}.get(x,x);yy={'passing_entropy_excess':f'passing_entropy_half{half}','shooting_entropy_role_excess':f'shooting_role_half{half}','passing_ec_excess':f'passing_ec_half{half}'}.get(y,y)
   fit,*_=regress(d,yy,x=xx);fit.update(analysis=f'null_draw_half_{half}',original_x=x,original_y=y);sensitivity.append(fit)
 # Primary relationship sensitivity to starness definition; no minute/roster regression controls.
 for y in ['shooting_hhi_raw','shooting_hhi_normalized_unbiased','shooting_top1_share','shooting_top2_share']:
  fit,*_=regress(d,y,x=PRIMARY[0]);fit.update(analysis='shooting_concentration_definition');sensitivity.append(fit)
 primary=pd.DataFrame(fits);holm(primary);primary.to_csv(out/'pooled_regressions.csv',index=False)
 holm(pd.DataFrame(groups)).to_csv(out/'leader_group_regressions.csv',index=False);holm(pd.DataFrame(interactions),p='interaction_p').to_csv(out/'leader_interactions.csv',index=False);pd.DataFrame(fe).to_csv(out/'season_FE_sensitivity.csv',index=False)
 temp=pd.DataFrame(temporal);holm(temp,p='test_association_hc3_p');temp.to_csv(out/'temporal_checks.csv',index=False);pd.DataFrame(sensitivity).to_csv(out/'definition_and_MC_sensitivity.csv',index=False)
 contexts=[]
 for x,y in PAIRS:
  for year in [False,True]:contexts.append(context_fit(d,x,y,['roster_players','minutes_hhi'],season_fe=year))
 holm(pd.DataFrame(contexts)).to_csv(out/'opportunity_conditioned_sensitivity.csv',index=False)
 pd.DataFrame([temporal_context(d,PRIMARY[0],PRIMARY[1],['roster_players','minutes_hhi'])]).to_csv(out/'temporal_opportunity_check.csv',index=False)
 seasonrows=[]
 for season,g in d.groupby('season'):
  fit,*_=regress(g,PRIMARY[1],x=PRIMARY[0]);fit['season']=season;seasonrows.append(fit)
 holm(pd.DataFrame(seasonrows)).to_csv(out/'primary_by_season.csv',index=False)
 # Remove entropy-normalization effects as a sensitivity and verify the KL grouping identity.
 bits_rows=[]
 for row in d.itertuples():
  A,_,_=matrices[(row.team,row.season)];a=A.to_numpy(float);active=a.sum(0)+a.sum(1)>0;a=a[np.ix_(active,active)];E=expected_counts(a);KL=float(np.sum(np.where(a>0,a*np.log2(np.maximum(np.divide(a,E,out=np.ones_like(a),where=E>0),1e-300)),0))/a.sum());expected_h=float(np.sum(E.sum(1)*entropy_rows(E))/E.sum())
  assert np.isclose(KL,expected_h-row.passing_entropy_bits,atol=1e-8)
  bits_rows.append(dict(team=row.team,season=row.season,passing_entropy_excess_bits=row.passing_entropy_excess*np.log2(row.passing_players-1),passing_preference_KL_bits=KL))
 extra=d[['team','season','shooting_hhi_unbiased']].merge(pd.DataFrame(bits_rows),on=['team','season'],validate='one_to_one');extra.to_csv(out/'passing_bits_and_KL_sensitivity_metrics.csv',index=False)
 bitfits=[]
 for x in ['passing_entropy_excess_bits','passing_preference_KL_bits']:
  fit,*_=regress(extra,'shooting_hhi_unbiased',x=x);bitfits.append(fit)
 pd.DataFrame(bitfits).to_csv(out/'passing_bits_and_KL_sensitivity.csv',index=False)

 with PdfPages(out/'revised_analysis.pdf') as pdf:
  for mode in ['pooled','temporal','leader_groups']:
   fig,axes=plt.subplots(2,3,figsize=(19,12))
   for ax,(x,y) in zip(axes.flat,PAIRS):
    if mode=='pooled':
     for label,g in d.groupby('sample'):ax.scatter(g[x],g[y],s=22,alpha=.65,label=label)
     fit=band(ax,d,x,y,'Pooled regression','#b54236');ax.set_title(f'r={fit["r"]:.3f} | R²={fit["r_squared"]:.1%} | p={fit["robust_p"]:.3g}',fontsize=11)
    elif mode=='temporal':
     train=d[d.season<=TRAIN_END];test=d[d.season.eq(TEST)];ax.scatter(test[x],test[y],s=35,color='#c37a31',label='2025–26 observed');fit=band(ax,train,x,y,'Fit on earlier 150 team-seasons','#236c91');q=temp[(temp.x==x)&(temp.y==y)].iloc[0];ax.set_title(f'Test r={q.test_r:.3f} | frozen-model test R²={q.predictive_r_squared_vs_test_mean:.1%}',fontsize=10)
    else:
     for (group,g),color in zip(d.groupby('leader_alignment'),['#bb7834','#227d97']):
      ax.scatter(g[x],g[y],s=22,alpha=.6,color=color);fit=band(ax,g,x,y,f'{group}: n={len(g)}, r={np.corrcoef(g[x],g[y])[0,1]:.3f}',color)
    ax.set_xlabel(COLS[x],fontsize=8);ax.set_ylabel(COLS[y],fontsize=8);ax.grid(alpha=.15);ax.legend(fontsize=6)
   fig.suptitle({'pooled':'Revised measures: activity and position reference differences','temporal':'Retrospective temporal check: coefficients frozen after first five seasons','leader_groups':'Separate lines for shared versus separate passing/FGA leaders'}[mode],fontsize=17)
   fig.text(.5,.015,'Negative excess means below the specified reference, not invalid entropy. Shooting concentration is FGA HHI with finite-count correction.\nMain models have no minutes/roster covariates. Pooled inference clusters franchises; Holm tests exported. Baselines do not control lineup co-appearance.\nLatest season was previously explored: temporal results are retrospective. These describe structure, not validated causal star dependency.',ha='center',fontsize=9);fig.tight_layout(rect=[0,.09,1,.95]);fig.savefig(out/(mode+'.png'),dpi=150);pdf.savefig(fig);plt.close(fig)
 # Fix absence-window contamination of X as far as existing data allows: use hub-present networks only.
 coveragefile=root/'offensive_structure_profiles/absence_validation_coverage.csv';absence=[];absence_rows=[]
 if coveragefile.exists():
  record(coveragefile);
  for name in ['analysis_audit.json','control_definitions.json','team_season_metrics.csv','eligibility.csv']:
   auditfile=root/'2025-26/team_season_topics/hub_absence_resilience'/name
   if auditfile.exists():record(auditfile)
  coverage=pd.read_csv(coveragefile);coverage=coverage[coverage.eligible_for_exploratory_check.eq(True)]
  for row in coverage.itertuples():
   file=root/'2025-26/team_season_topics/hub_absence_resilience/present_networks'/row.team/'passing_matrix.csv';record(file);A=pd.read_csv(file,index_col=0).to_numpy(float);active=A.sum(0)+A.sum(1)>0;A=A[np.ix_(active,active)].astype(np.int64);oh,oc=null_metrics(A);E=expected_counts(A);prob=np.divide(E,E.sum(1)[:,None],out=np.zeros_like(E),where=E.sum(1)[:,None]>0);hs=[];cs=[]
   for rep in range(args.draws):
    B=np.zeros_like(A)
    for i,total in enumerate(A.sum(1)):
     if total:B[i]=rng.multinomial(int(total),prob[i])
    h,c=null_metrics(B);hs.append(h);cs.append(c)
   absence_rows.append(dict(team=row.team,season='2025-26',hub_name=row.hub_name,hub_present_games=row.hub_present_games,hub_absent_games=row.hub_absent_games,adjusted_offensive_penalty=row.adjusted_offensive_penalty,present_passing_entropy_excess=oh-np.mean(hs),present_passing_ec_excess=oc-np.mean(cs),present_passing_entropy=oh,present_passing_ec=oc))
  absent=pd.DataFrame(absence_rows);absent.to_csv(out/'hub_present_only_absence_metrics.csv',index=False)
  for x in ['present_passing_entropy_excess','present_passing_ec_excess','present_passing_entropy','present_passing_ec']:
   fit,*_=regress(absent,'adjusted_offensive_penalty',x=x);absence.append(fit)
  holm(pd.DataFrame(absence)).to_csv(out/'absence_checks.csv',index=False)
  fig,axes=plt.subplots(1,2,figsize=(13,6))
  for ax,x in zip(axes,['present_passing_entropy_excess','present_passing_ec_excess']):
   ax.scatter(absent[x],absent.adjusted_offensive_penalty);fit=band(ax,absent,x,'adjusted_offensive_penalty','Present-network predictor','#b54236');ax.set_xlabel(x.replace('_',' '));ax.set_ylabel('Saved adjusted offensive absence penalty');ax.set_title(f'n={len(absent)}, r={fit["r"]:.3f}, p={fit["robust_p"]:.3f}')
  fig.suptitle('Exploratory absence check: predictors use hub-present games only');fig.text(.5,.015,'Removes hub-absent games from the network predictor, but is not prospective: selected hubs and sample are retrospective.\nSaved outcome adjustments reused; unequal outcome precision and incomplete availability controls remain.',ha='center',fontsize=9);fig.tight_layout(rect=[0,.09,1,.93]);fig.savefig(out/'absence_present_only.png',dpi=150);fig.savefig(out/'absence_present_only.pdf');plt.close(fig)
 focal_figure(d,out)
 influence=[]
 for team in sorted(d.team.unique()):
  g=d[d.team.ne(team)];influence.append(dict(excluded_franchise=team,n=len(g),r=float(np.corrcoef(g[PRIMARY[0]],g[PRIMARY[1]])[0,1])))
 pd.DataFrame(influence).to_csv(out/'leave_one_franchise_out_association.csv',index=False)
 clean=['season','team','sample']+list(COLS)+['passing_entropy','shooting_entropy','passing_starness','shooting_hhi_raw','passing_leader_name','shooting_leader_name','leader_alignment','passes','fga','roster_players','passing_players','shooting_players','minutes_hhi']
 d[clean].to_csv(out/'new_data.csv',index=False)
 import inspect,platform
 for function in [regress,null_metrics,context_fit]:record(Path(inspect.getfile(function)).resolve())
 for name in ['revised_methodology.py','revised_methodology_report.py']:
  file=Path(__file__).resolve().with_name(name)
  if file.exists():record(file)
 (out/'runtime_versions.json').write_text(json.dumps(dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,matplotlib=matplotlib.__version__),indent=2))
 pd.DataFrame(sources).to_csv(out/'source_manifest.csv',index=False)
 # Older shared helper uses entropy-specific column names; make exported generic fits unambiguous.
 for name in ['pooled_regressions.csv','leader_group_regressions.csv','leader_interactions.csv','season_FE_sensitivity.csv','definition_and_MC_sensitivity.csv','primary_by_season.csv','passing_bits_and_KL_sensitivity.csv','absence_checks.csv']:
  file=out/name
  if file.exists():
   table=pd.read_csv(file).rename(columns={'entropy_mean':'x_mean','fitted_intercept_at_entropy_mean':'fitted_intercept_at_x_mean','slope_per_0_1_entropy':'slope_per_0_1_x'});table.to_csv(file,index=False)
 from revised_methodology_report import write_report
 write_report(out)

 print('POOLED\n'+primary[['x','outcome','n','r','r_squared','robust_p','holm_p']].to_string(index=False),flush=True)
 print('TEMPORAL\n'+temp[['x','y','test_r','predictive_r_squared_vs_train_mean','holm_p']].to_string(index=False),flush=True)
 print('INTERACTIONS\n'+pd.read_csv(out/'leader_interactions.csv')[['x','outcome','interaction_p','holm_p']].to_string(index=False),flush=True)
 if absence:print('ABSENCE\n'+pd.read_csv(out/'absence_checks.csv')[['x','n','r','robust_p','holm_p']].to_string(index=False),flush=True)
 return 0
if __name__=='__main__':raise SystemExit(main())
