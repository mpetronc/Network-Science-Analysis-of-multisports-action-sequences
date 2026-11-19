"""Regress whole-season conditional entropy against passing EC concentration."""
from pathlib import Path
import argparse,sys,json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import t
from teamgraphs import eigenvector_passing_concentration
from entropy_structure_profiles import conditional_entropy,regress

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--star-entropy-regressions',action='store_true');p.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'nba_passing_output');p.add_argument('--destination',type=Path);a=p.parse_args(argv)
 root=a.output_dir;out=a.destination or root/'fg_attempt_networks/entropy_star_regressions';out.mkdir(parents=True,exist_ok=True);rows=[];manifest=[]
 for file in sorted((root/'full_roster_entropy_centrality/historical_full_rosters').glob('20*/*/passing_matrix.csv')):
  season,team=file.parent.parent.name,file.parent.name;matrix=pd.read_csv(file,index_col=0);A=matrix.to_numpy(float);ids=matrix.index.astype(int).tolist()
  assert ids==matrix.columns.astype(int).tolist()
  ec,scores,active,sub=eigenvector_passing_concentration(A,ids)
  if not ec['network_valid']: raise ValueError(f'{season} {team}: {ec["network_error"]}')
  passing,bits,n,P=conditional_entropy(A)
  edgefile=root/'fg_attempt_networks'/season/team/'edges.csv';edges=pd.read_csv(edgefile);shots=edges[edges.edge_type.eq('shot_attempt')];cats=['OTHER_2PT','THREE_PT','LAYUP_DUNK']
  S=shots.pivot(index='source',columns='target',values='raw_count').fillna(0);S.index=S.index.astype(int);S=S.reindex(index=ids,columns=cats,fill_value=0).fillna(0).to_numpy(float)
  assert np.isclose(shots.weight.sum(),P) and S.sum()>0
  # Rows are players; the three sinks have no outgoing transitions. Do not invent sink self-loops.
  values={}
  for ratio in [.5,1.,2.]:
   B=np.column_stack([A,S*P/S.sum()*ratio]);r=B.sum(1);prob=np.divide(B,r[:,None],out=np.zeros_like(B),where=r[:,None]>0)
   h=float(-np.sum(B*np.log2(np.maximum(prob,1e-300)))/B.sum())
   # n positive-activity players; each can reach n-1 other players plus 3 sinks.
   n_aug=int(np.count_nonzero(A.sum(0)+A.sum(1)+S.sum(1)));values[f'augmented_entropy_ratio_{ratio:g}']=h/np.log2(n_aug+2)
   if ratio==1: values['augmented_entropy_bits']=h
  top=max(scores,key=scores.get)
  rows.append(dict(team=team,season=season,ec_concentration=ec['eigenvector_concentration'],top_ec_share=scores[top],passing_entropy=passing,passing_entropy_bits=bits,players=n,passes=P,fga=S.sum(),**values))
  for f in [file,edgefile]:manifest.append(dict(path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
 d=pd.DataFrame(rows);d.to_csv(out/'team_season_metrics.csv',index=False);pd.DataFrame(manifest).to_csv(out/'source_manifest.csv',index=False)
 results=[]
 measures=[('augmented_entropy_ratio_1','Passing + three FGA end nodes'),('passing_entropy','Passing only')]
 with PdfPages(out/'entropy_vs_star_dependency.pdf') as pdf:
  for season in ['ALL']+sorted(d.season.unique()):
   sample=d if season=='ALL' else d[d.season.eq(season)];fig,axes=plt.subplots(1,2,figsize=(15,6))
   for ax,(x,title) in zip(axes,measures):
    fit,b,cov,terms=regress(sample,'ec_concentration',x=x);fit.update(season=season,model='unadjusted');results.append(fit)
    xx=np.linspace(sample[x].min(),sample[x].max(),150);X=np.column_stack([np.ones(len(xx)),xx-fit['entropy_mean']]);y=X@b;band=t.ppf(.975,fit['df'])*np.sqrt(np.maximum(np.einsum('ij,jk,ik->i',X,cov,X),0))
    for label,g in sample.groupby('season'): ax.scatter(g[x],g.ec_concentration,s=30,alpha=.75,label=label)
    ax.plot(xx,y,color='#b7432d');ax.fill_between(xx,y-band,y+band,color='#b7432d',alpha=.12)
    if season!='ALL':
     for row in sample.itertuples():ax.annotate(row.team,(getattr(row,x.replace('.','_'),np.nan) if '.' in x else getattr(row,x),row.ec_concentration),fontsize=6)
    ax.set_title(f'{title}\nn={len(sample)}, r={fit["r"]:.3f}, R²={fit["r_squared"]:.1%}\nslope={fit["slope"]:.4f}, robust p={fit["robust_p"]:.3g}',fontsize=11)
    ax.set_xlabel('Normalized conditional entropy');ax.set_ylabel('Passing EC concentration (star dependency proxy)');ax.grid(alpha=.15)
    if season=='ALL':ax.legend(fontsize=7)
   fig.suptitle(f'{season} | Whole-season entropy and passing star concentration',fontsize=15)
   fig.text(.5,.015,'No minutes or roster controls. EC uses original passing network only. Pooled p/CI clustered by franchise; season p/CI use HC3.\nEntropy of augmented graph is weighted one-step choice entropy, not an absorbing-chain entropy rate. Association does not establish dependence or causation.',ha='center',fontsize=9)
   fig.tight_layout(rect=[0,.07,1,.93]);fig.savefig(out/f'{season}_entropy_star.png',dpi=150);pdf.savefig(fig);plt.close(fig)
 for x in ['augmented_entropy_ratio_0.5','augmented_entropy_ratio_1','augmented_entropy_ratio_2','passing_entropy']:
  for fe in [False,True]:
   fit,*_=regress(d,'ec_concentration',x=x,season_fe=fe);fit.update(season='ALL',model='sensitivity_season_FE' if fe else 'sensitivity_unadjusted');results.append(fit)
 result=pd.DataFrame(results)
 # Holm adjustment across the 14 primary overall/per-season regressions.
 primary=result.index[result.model.eq('unadjusted')];order=result.loc[primary,'robust_p'].sort_values().index
 result.loc[order,'holm_p_primary14']=np.minimum(1,np.maximum.accumulate([(len(order)-j)*result.loc[i,'robust_p'] for j,i in enumerate(order)]))
 result.to_csv(out/'regression_results.csv',index=False)
 (out/'README.md').write_text('''Conditional entropy is the outgoing-weighted average of each player row's Shannon entropy. Passing-only is divided by log2(N-1); augmented by log2(N+2) (N-1 possible player recipients and three shot sinks). Shot weight sums to total passes at ratio 1. All shot categories include misses. Zero-outdegree terminal nodes contribute no row entropy, with no fabricated self-loops. This is not stationary Markov entropy rate: an absorbing chain eventually has no action. Augmented entropy depends on the chosen pass/shot scaling; ratios 0.5 and 2 are exported as sensitivity checks. Normalized entropies have different choice spaces and should not be compared as an effect size.\n\nStar dependency proxy = normalized HHI of L1 eigenvector shares on original A+A.T, active passing players only. This is structural concentration, not an observed loss when a star is absent, skill or official usage. Both metrics share source passing counts, so correlations can partly reflect mathematical coupling. No controls in the primary plots; season fixed effects are separately labeled sensitivity models. Repeated franchises use cluster-robust p and 95% confidence bands; per-season tests use HC3. Holm-corrected p values for 14 primary comparisons are exported. Plain r/R² describe the raw bivariate association even in the separately labeled season-FE sensitivity rows; full_model_r_squared is the adjusted model fit. Source hashes retained.\n''')
 print(result[result.model.eq('unadjusted')][['season','x','r','r_squared','slope','robust_p','holm_p_primary14']].to_string(index=False))
 return 0
if __name__=='__main__':raise SystemExit(main())
