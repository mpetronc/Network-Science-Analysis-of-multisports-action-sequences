"""Four separate passing/shooting metrics and every unordered pairwise regression."""
from pathlib import Path
import argparse,hashlib,itertools,json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import t
from court_zone_networks import ZONES,build,classify_zones
from nba_api.stats.static import teams
from teamgraphs import eigenvector_passing_concentration
from entropy_structure_profiles import conditional_entropy,regress

LABELS={'passing_entropy':'Passing entropy','shooting_entropy':'Shooting zone entropy','passing_starness':'Passing starness (EC concentration)','shooting_starness':'Shooting starness (FGA concentration)'}
PAIRS=list(itertools.combinations(LABELS,2))

def holm(frame,indices,column):
 order=frame.loc[indices,'robust_p'].sort_values().index
 frame.loc[order,column]=np.minimum(1,np.maximum.accumulate([(len(order)-j)*frame.loc[i,'robust_p'] for j,i in enumerate(order)]))

def metrics(matrix,roster,edges):
 ids=matrix.index.astype(int).tolist();assert ids==matrix.columns.astype(int).tolist()
 A=matrix.to_numpy(float);r=roster.set_index('PLAYER_ID').loc[ids]
 # EC must use ONLY the original player-to-player passing matrix.
 # Independently reconstruct it from pass-tagged edges; shot edges never enter.
 pass_edges=edges.loc[edges.edge_type.eq('pass')].copy()
 pass_edges['source']=pd.to_numeric(pass_edges.source,errors='raise').astype(int)
 pass_edges['target']=pd.to_numeric(pass_edges.target,errors='raise').astype(int)
 if not set(pass_edges.source).union(pass_edges.target)<=set(ids):
  raise ValueError('Passing EC cannot include shooting/category nodes')
 pass_matrix=pass_edges.pivot(index='source',columns='target',values='raw_count').reindex(index=ids,columns=ids).fillna(0).to_numpy(float)
 if not np.array_equal(A,pass_matrix):
  raise ValueError('EC input must match original passing edges exactly; shooting weights forbidden')
 ec,scores,active,sub=eigenvector_passing_concentration(A,ids)
 if not ec['network_valid']:raise ValueError(ec['network_error'])
 hp,hbits,n,P=conditional_entropy(A)
 shots=edges.loc[edges.edge_type.eq('shot_attempt')].copy()
 S=shots.pivot(index='source',columns='target',values='raw_count').fillna(0);S.index=S.index.astype(int)
 if not set(S.index)<=set(ids):raise ValueError('Unknown shooter')
 S=S.reindex(index=ids,columns=list(ZONES),fill_value=0).fillna(0).to_numpy(float)
 F=S.sum(1)
 if not np.array_equal(F,r.FGA.to_numpy(float)) or not np.isfinite(S).all() or (S<0).any():raise ValueError('Invalid or incomplete shot counts')
 probs=np.divide(S,F[:,None],out=np.zeros_like(S),where=F[:,None]>0)
 bits=float(-np.sum(S*np.log2(np.maximum(probs,1e-300)))/F.sum())
 shares=F/F.sum();nshoot=int(np.count_nonzero(F));hhi=float(shares@shares)
 if nshoot<2:raise ValueError('At least two shooters required')
 concentration=(hhi-1/nshoot)/(1-1/nshoot)
 # Supplementary shooter-allocation entropy is distinct from primary shot-choice entropy.
 shooterbits=float(-np.sum(shares*np.log2(np.maximum(shares,1e-300))))
 top=int(np.argmax(F));leader=min(scores,key=lambda pid:(-scores[pid],pid))
 result=dict(passing_entropy=hp,shooting_entropy=bits/np.log2(len(ZONES)),passing_starness=ec['eigenvector_concentration'],shooting_starness=concentration,passing_entropy_bits=hbits,shooting_entropy_bits=bits,shooting_hhi=hhi,shooting_starness_full_roster=(hhi-1/len(ids))/(1-1/len(ids)),shooting_allocation_entropy=shooterbits/np.log2(nshoot),shooting_allocation_entropy_bits=shooterbits,passing_players=n,shooting_players=nshoot,roster_players=len(ids),passes=P,fga=F.sum(),passing_leader_id=leader,passing_leader_name=r.loc[leader,'PLAYER_NAME'],shooting_leader_id=ids[top],shooting_leader_name=r.iloc[top].PLAYER_NAME,shooting_top_share=shares[top],shared_passing_shooting_leader=leader==ids[top])
 assert all(0<=result[x]<=1 for x in LABELS)
 return result

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--court-zone-regressions',action='store_true');p.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'nba_passing_output');p.add_argument('--destination',type=Path);args=p.parse_args(argv)
 root=args.output_dir;out=args.destination or root/'court_zone_networks/regressions';out.mkdir(parents=True,exist_ok=True);rows=[];sources=[]
 mapping={t['abbreviation']:t['id'] for t in teams.get_teams()};loaded=None
 for file in sorted((root/'full_roster_entropy_centrality/historical_full_rosters').glob('20*/*/passing_matrix.csv')):
  season,team=file.parent.parent.name,file.parent.name;rosterfile=file.with_name('roster.csv');cache=root/'shot_attempt_cache'/f'{season}_shot_attempts.csv.gz'
  if loaded!=season:allshots=pd.read_csv(cache);classify_zones(allshots);loaded=season
  matrix=pd.read_csv(file,index_col=0);roster=pd.read_csv(rosterfile)
  original,graph,counts,makes,audit=build(matrix,roster,allshots.loc[allshots.TEAM_ID.eq(mapping[team])])
  edges=pd.DataFrame([dict(source=u,target=v,**a) for u,v,a in graph.edges(data=True)])
  row=metrics(matrix,roster,edges);row.update(team=team,season=season);rows.append(row)
  for f in [file,rosterfile,cache]:sources.append(dict(team=team,season=season,path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
 d=pd.DataFrame(rows)
 if len(d)!=180 or not d.groupby('season').size().eq(30).all():raise ValueError('Expected all 180 team-seasons')
 oldfile=root/'fg_attempt_networks/four_metric_regressions/four_metrics.csv'
 if oldfile.exists():
  old=pd.read_csv(oldfile);comparison=d.merge(old[['team','season','passing_entropy','passing_starness','shooting_starness','shooting_entropy']],on=['team','season'],suffixes=('_new','_old'),validate='one_to_one')
  for name in ['passing_entropy','passing_starness','shooting_starness']:
   assert np.allclose(comparison[name+'_new'],comparison[name+'_old'],atol=1e-12)
  comparison[['team','season','shooting_entropy_old','shooting_entropy_new']].to_csv(out/'shooting_entropy_before_after.csv',index=False)
 d.to_csv(out/'four_metrics.csv',index=False);pd.DataFrame(sources).to_csv(out/'source_manifest.csv',index=False)
 d[list(LABELS)].corr().to_csv(out/'pearson_r_matrix.csv');(d[list(LABELS)].corr()**2).to_csv(out/'r_squared_matrix.csv')
 results=[];sensitivity=[]
 with PdfPages(out/'all_pairwise_regressions.pdf') as pdf:
  for season in ['ALL']+sorted(d.season.unique()):
   sample=d if season=='ALL' else d[d.season.eq(season)];fig,axes=plt.subplots(2,3,figsize=(18,11))
   for ax,(x,y) in zip(axes.flat,PAIRS):
    fit,b,cov,_=regress(sample,y,x=x);fit.update(season=season,model='unadjusted',intercept=b[0]-b[1]*sample[x].mean(),slope_per_0_01_x=b[1]/100);results.append(fit)
    for label,g in sample.groupby('season'):ax.scatter(g[x],g[y],s=23,alpha=.7,label=label)
    xx=np.linspace(sample[x].min(),sample[x].max(),120);X=np.column_stack([np.ones(len(xx)),xx-fit['entropy_mean']]);pred=X@b;band=t.ppf(.975,fit['df'])*np.sqrt(np.maximum(np.einsum('ij,jk,ik->i',X,cov,X),0));ax.plot(xx,pred,color='#b33c32');ax.fill_between(xx,pred-band,pred+band,color='#b33c32',alpha=.12)
    if season!='ALL':
     for z in sample.itertuples():ax.annotate(z.team,(getattr(z,x),getattr(z,y)),fontsize=5,xytext=(2,2),textcoords='offset points')
    ax.set_xlabel(LABELS[x]);ax.set_ylabel(LABELS[y]);ax.set_title(f'r={fit["r"]:.3f} | R²={fit["r_squared"]:.1%}\nslope={b[1]:.4f} | robust p={fit["robust_p"]:.3g}',fontsize=11);ax.grid(alpha=.15)
   if season=='ALL':axes.flat[0].legend(fontsize=6)
   fig.suptitle(f'{season} | All six passing/shooting-zone metric pairs | n={len(sample)}',fontsize=18)
   fig.text(.5,.015,'Shooting zone entropy: court-zone choice given shooter. Shooting starness: concentration of FGA shares. Passing starness: passing-only EC concentration.\nNo minutes/roster controls. Pooled p/CI cluster by franchise; per-season HC3. Shading: 95% confidence band for fitted mean. Shared inputs can create mathematical coupling.',ha='center',fontsize=10)
   fig.tight_layout(rect=[0,.075,1,.95]);fig.savefig(out/(season+'_six_regressions.png'),dpi=150);pdf.savefig(fig);plt.close(fig)
 for x,y in PAIRS:
  fit,*_=regress(d,y,x=x,season_fe=True);fit.update(model='season_FE');sensitivity.append(fit)
 # Normalization sensitivities; not part of the six primary pairs.
 for x,y in [('shooting_entropy','shooting_starness_full_roster'),('shooting_allocation_entropy','shooting_starness')]:
  fit,*_=regress(d,y,x=x);fit.update(model='supplementary_definition');sensitivity.append(fit)
 r=pd.DataFrame(results)
 for season in r.season.unique():holm(r,r.index[r.season.eq(season)],'holm_p_within_six')
 holm(r,r.index,'holm_p_all42')
 r.to_csv(out/'regression_results.csv',index=False);pd.DataFrame(sensitivity).to_csv(out/'sensitivity_models.csv',index=False)
 fig,ax=plt.subplots(figsize=(9,8));corr=d[list(LABELS)].corr();im=ax.imshow(corr,vmin=-1,vmax=1,cmap='RdBu_r');names=['Passing entropy','Shooting zone entropy','Passing starness','Shooting starness'];ax.set_xticks(range(4),names,rotation=30,ha='right');ax.set_yticks(range(4),names)
 for i in range(4):
  for j in range(4):ax.text(j,i,f'{corr.iloc[i,j]:.3f}',ha='center',va='center',color='white' if abs(corr.iloc[i,j])>.6 else 'black')
 fig.colorbar(im,ax=ax,label='Pearson r');ax.set_title('Four-metric correlations | 180 team-seasons');fig.tight_layout();fig.savefig(out/'correlation_matrix.png',dpi=160);fig.savefig(out/'correlation_matrix.pdf');plt.close(fig)
 (out/'README.md').write_text('''Four metrics, six unordered pairwise regressions overall and within each of six seasons (42 fits). All metrics lie in [0,1]. Primary shooting entropy is H(court zone | shooter), FGA-weighted, divided by log2(15); categories are 14 NBA half-court zones plus backcourt/heaves, makes and misses included. It is within-shooter shot-location diversity, not how evenly players share attempts or shot quality; geographic resolution is limited to the fixed NBA zones. Passing entropy is H(receiver|passer), pass-weighted, divided by log2(Npassing-1). Passing starness is normalized HHI of L1 eigenvector centrality shares on original A+A.T, active passing nodes only. Shooting starness is normalized HHI of player FGA shares, normalized for the number of positive-FGA shooters. Neither is validated causal dependency. Shooting starness deliberately uses FGA only, not previous FGA+0.44FTA+TOV workload proxy or official USG%. Zero-FGA players do not change primary shooter-count denominator; full-roster normalization is exported as a sensitivity. All source FGA/FGM/3PA/3PM checked per player; raw counts, no 50/50 scaling, no minutes cutoff.\n\nEach unordered pair is fit once using metric order passing entropy, shooting entropy, passing starness, shooting starness; reversing axes preserves Pearson r and simple R² but changes OLS slope. No multiple-predictor or causal model is implied. Pooled p/CIs cluster by franchise; per-season HC3. Holm p supplied within each six-test family and across all 42 primary fits. Season-FE sensitivities separate; r/R² in those rows remain raw bivariate summaries, full_model_r_squared is full fit. Slope per 0.01 X exported. Negative fitted values outside observed data are not valid metric values.\n\nShooting allocation entropy (entropy of shooter FGA shares) is exported only as a supplementary definition because it and FGA HHI summarize the SAME share distribution, so a strong negative relationship is substantially mathematical. Similarly passing entropy/EC share source counts; previous randomization showed a strong relationship can arise from activity alone. Primary conditional shooting entropy avoids the direct same-share-definition comparison, but cannot eliminate all coupling. All 180 team-seasons include traded players' team-filtered counts. Source hashes retained; prior code and output folders preserved.\n''')
 print(r[r.season.eq('ALL')][['x','outcome','n','r','r_squared','slope','robust_p','holm_p_within_six']].to_string(index=False));return 0
if __name__=='__main__':raise SystemExit(main())
