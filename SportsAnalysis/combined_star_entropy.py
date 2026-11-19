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
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--combined-star-entropy',action='store_true');p.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'nba_passing_output');p.add_argument('--destination',type=Path);a=p.parse_args(argv)
 root=a.output_dir;out=a.destination or root/'fg_attempt_networks/combined_star_entropy_regressions';out.mkdir(parents=True,exist_ok=True);rows=[];manifest=[]
 metrics_file=root/'fg_attempt_networks/entropy_star_regressions/team_season_metrics.csv'
 profile_file=root/'basket_networks/usage_starness/team_season_starness_and_usage.csv'
 metrics=pd.read_csv(metrics_file);profile=pd.read_csv(profile_file)
 keep=['team','season','passing_ec','usage_hhi','passing_percentile','usage_percentile','starness_index','starness_passing_weight_0.25','starness_passing_weight_0.75','same_passing_usage_leader']
 d=metrics.merge(profile[keep],on=['team','season'],validate='one_to_one')
 assert len(d)==180 and np.allclose(d.ec_concentration,d.passing_ec,atol=1e-8)
 assert np.allclose(d.starness_index,.5*d.passing_percentile+.5*d.usage_percentile)
 assert d.starness_index.between(0,100).all()
 d.to_csv(out/'team_season_metrics.csv',index=False)
 for file in [metrics_file,profile_file]:manifest.append(dict(path=str(file),sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
 pd.DataFrame(manifest).to_csv(out/'source_manifest.csv',index=False)
 results=[]
 measures=[('augmented_entropy_ratio_1','Passing + three FGA end nodes'),('passing_entropy','Passing only')]
 with PdfPages(out/'entropy_vs_combined_starness.pdf') as pdf:
  for season in ['ALL']+sorted(d.season.unique()):
   sample=d if season=='ALL' else d[d.season.eq(season)];fig,axes=plt.subplots(1,2,figsize=(15,6))
   for ax,(x,title) in zip(axes,measures):
    fit,b,cov,terms=regress(sample,'starness_index',x=x);fit.update(season=season,model='unadjusted');results.append(fit)
    xx=np.linspace(sample[x].min(),sample[x].max(),150);X=np.column_stack([np.ones(len(xx)),xx-fit['entropy_mean']]);y=X@b;band=t.ppf(.975,fit['df'])*np.sqrt(np.maximum(np.einsum('ij,jk,ik->i',X,cov,X),0))
    for label,g in sample.groupby('season'): ax.scatter(g[x],g.starness_index,s=30,alpha=.75,label=label)
    ax.plot(xx,y,color='#b7432d');ax.fill_between(xx,y-band,y+band,color='#b7432d',alpha=.12)
    if season!='ALL':
     for row in sample.itertuples():ax.annotate(row.team,(getattr(row,x.replace('.','_'),np.nan) if '.' in x else getattr(row,x),row.starness_index),fontsize=6)
    ax.set_title(f'{title}\nn={len(sample)}, r={fit["r"]:.3f}, R²={fit["r_squared"]:.1%}\nslope={fit["slope"]:.4f}, robust p={fit["robust_p"]:.3g}',fontsize=11)
    ax.set_xlabel('Normalized conditional entropy');ax.set_ylabel('Combined centrality + usage index (0–100)');ax.grid(alpha=.15)
    if season=='ALL':ax.legend(fontsize=7)
   fig.suptitle(f'{season} | Whole-season entropy and combined starness',fontsize=15)
   fig.text(.5,.015,'No minutes or roster controls. Index = 50% passing EC percentile + 50% usage-concentration percentile. Pooled p/CI clustered by franchise; season p/CI use HC3.\nEntropy of augmented graph is weighted one-step choice entropy, not an absorbing-chain entropy rate. Association does not establish dependence or causation.',ha='center',fontsize=9)
   fig.tight_layout(rect=[0,.07,1,.93]);fig.savefig(out/f'{season}_entropy_star.png',dpi=150);pdf.savefig(fig);plt.close(fig)
 for x in ['augmented_entropy_ratio_0.5','augmented_entropy_ratio_1','augmented_entropy_ratio_2','passing_entropy']:
  for fe in [False,True]:
   fit,*_=regress(d,'starness_index',x=x,season_fe=fe);fit.update(season='ALL',model='sensitivity_season_FE' if fe else 'sensitivity_unadjusted');results.append(fit)
 result=pd.DataFrame(results)
 # Holm adjustment across the 14 primary overall/per-season regressions.
 primary=result.index[result.model.eq('unadjusted')];order=result.loc[primary,'robust_p'].sort_values().index
 result.loc[order,'holm_p_primary14']=np.minimum(1,np.maximum.accumulate([(len(order)-j)*result.loc[i,'robust_p'] for j,i in enumerate(order)]))
 result.to_csv(out/'regression_results.csv',index=False)
 (out/'README.md').write_text("""Entropy vs combined centrality-and-usage starness. Reuses the frozen 180-team-season starness reference from basket_networks/usage_starness; no re-ranking. Index = 0.5 * passing EC concentration percentile + 0.5 * usage HHI percentile, 0–100. EC uses only original player passing network A+A.T. Usage HHI uses player season shares of FGA + 0.44*FTA + TOV; this is a workload-concentration proxy, not official USG%. A high index can have different players leading passing and usage; leader alignment is retained in the metrics CSV. It is not validated causal star dependency.

Entropy and its sensitivity values reuse the verified recent three-shot-node analysis. Conditional entropy is weighted one-step row entropy; normalized by log2(N+2) with three shot sinks, or log2(N-1) for passing only. Sinks have no outgoing transitions and no self-loops. Shot-edge total is equal to passes at ratio 1, with 0.5 and 2 sensitivity models. This is not absorbing-chain entropy rate. Primary regressions have no minutes or roster controls. Repeated-franchise pooled p values/CI are cluster robust, per-season p/CI HC3; 14 primary comparisons also have Holm corrected p values. Season fixed effects and shot-scale sensitivities are separately labeled. Raw r/R² in FE rows remain bivariate summaries; use full_model_r_squared for full FE fit. Both metrics share source passing/shot data; association can partly be mathematical coupling, not independent predictive validation.
""")
 print(result[result.model.eq('unadjusted')][['season','x','r','r_squared','slope','robust_p','holm_p_primary14']].to_string(index=False))
 return 0
if __name__=='__main__':raise SystemExit(main())
