"""Regress whole-season entropy against centrality/usage weighted by pass and FGA counts."""
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
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw-count-star-entropy',action='store_true');p.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'nba_passing_output');p.add_argument('--destination',type=Path);a=p.parse_args(argv)
 root=a.output_dir;out=a.destination or root/'fg_attempt_networks/raw_count_star_entropy_regressions';out.mkdir(parents=True,exist_ok=True);rows=[];manifest=[]
 metrics_file=root/'fg_attempt_networks/entropy_star_regressions/team_season_metrics.csv'
 profile_file=root/'basket_networks/usage_starness/team_season_starness_and_usage.csv'
 metrics=pd.read_csv(metrics_file);profile=pd.read_csv(profile_file)
 keep=['team','season','passing_ec','usage_hhi','passing_percentile','usage_percentile','starness_index','starness_passing_weight_0.25','starness_passing_weight_0.75','same_passing_usage_leader']
 d=metrics.merge(profile[keep],on=['team','season'],validate='one_to_one')
 assert len(d)==180 and np.allclose(d.ec_concentration,d.passing_ec,atol=1e-8)
 assert np.allclose(d.starness_index,.5*d.passing_percentile+.5*d.usage_percentile)
 assert d.starness_index.between(0,100).all()
 d['equal_weight_starness_index']=d.starness_index
 d['pass_weight']=d.passes/(d.passes+d.fga)
 d['shot_weight']=d.fga/(d.passes+d.fga)
 d['starness_index']=d.pass_weight*d.passing_percentile+d.shot_weight*d.usage_percentile
 assert np.allclose(d.pass_weight+d.shot_weight,1)
 assert d.starness_index.between(0,100).all()
 raw_rows=[]
 for row in d.itertuples():
  folder=root/'full_roster_entropy_centrality/historical_full_rosters'/row.season/row.team
  matrix_file=folder/'passing_matrix.csv';edge_file=root/'fg_attempt_networks'/row.season/row.team/'edges.csv'
  matrix=pd.read_csv(matrix_file,index_col=0);ids=matrix.index.astype(int).tolist();A=matrix.to_numpy(float)
  assert ids==matrix.columns.astype(int).tolist()
  edges=pd.read_csv(edge_file);shots=edges[edges.edge_type.eq('shot_attempt')]
  S=shots.pivot(index='source',columns='target',values='raw_count').fillna(0);S.index=S.index.astype(int)
  assert set(S.index)<=set(ids)
  S=S.reindex(index=ids,columns=['OTHER_2PT','THREE_PT','LAYUP_DUNK'],fill_value=0).fillna(0).to_numpy(float)
  assert np.isfinite(A).all() and np.isfinite(S).all() and (A>=0).all() and (S>=0).all()
  assert A.sum()==row.passes and S.sum()==row.fga
  active=A.sum(0)+A.sum(1)+S.sum(1)>0;B=np.column_stack([A,S]);totals=B.sum(1)
  prob=np.divide(B,totals[:,None],out=np.zeros_like(B),where=totals[:,None]>0)
  bits=float(-np.sum(B*np.log2(np.maximum(prob,1e-300)))/B.sum())
  # Independently verify Shannon's grouping identity: action-type choice + recipient/category choice.
  ps=A.sum(1);fs=S.sum(1);mix=np.column_stack([ps,fs]);mix_prob=np.divide(mix,totals[:,None],out=np.zeros_like(mix),where=totals[:,None]>0)
  action_bits=float(-np.sum(mix*np.log2(np.maximum(mix_prob,1e-300)))/B.sum())
  shot_prob=np.divide(S,fs[:,None],out=np.zeros_like(S),where=fs[:,None]>0)
  shot_bits=float(-np.sum(S*np.log2(np.maximum(shot_prob,1e-300)))/S.sum())
  expected=action_bits+row.pass_weight*row.passing_entropy_bits+row.shot_weight*shot_bits
  assert np.isclose(bits,expected,atol=1e-10)
  raw_rows.append(dict(team=row.team,season=row.season,raw_count_entropy=bits/np.log2(int(active.sum())+2),raw_count_entropy_bits=bits,pass_shot_choice_entropy_bits=action_bits,shot_category_entropy_bits=shot_bits,raw_shot_to_pass_ratio=S.sum()/A.sum()))
  for file in [matrix_file,edge_file]:manifest.append(dict(path=str(file),sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
 d=d.merge(pd.DataFrame(raw_rows),on=['team','season'],validate='one_to_one')
 assert d.raw_count_entropy.between(0,1).all()
 d.to_csv(out/'team_season_metrics.csv',index=False)
 for file in [metrics_file,profile_file]:manifest.append(dict(path=str(file),sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
 pd.DataFrame(manifest).to_csv(out/'source_manifest.csv',index=False)
 results=[]
 measures=[('raw_count_entropy','Actual passes + FGA counts'),('passing_entropy','Passing only')]
 with PdfPages(out/'entropy_vs_raw_count_starness.pdf') as pdf:
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
    ax.set_xlabel('Normalized conditional entropy');ax.set_ylabel('Action-weighted centrality + usage index (0–100)');ax.grid(alpha=.15)
    if season=='ALL':ax.legend(fontsize=7)
   fig.suptitle(f'{season} | Actual-count entropy and action-weighted starness',fontsize=15)
   fig.text(.5,.015,'No minutes or roster controls. Index weights: team passes/(passes+FGA) for EC; FGA/(passes+FGA) for usage. Pooled p/CI clustered by franchise; season p/CI use HC3.\nEntropy now uses raw passing and FGA counts, with no rescaling of shot edges. Association does not establish dependence or causation.',ha='center',fontsize=9)
   fig.tight_layout(rect=[0,.07,1,.93]);fig.savefig(out/f'{season}_entropy_star.png',dpi=150);pdf.savefig(fig);plt.close(fig)
 for x in ['raw_count_entropy','augmented_entropy_ratio_0.5','augmented_entropy_ratio_1','augmented_entropy_ratio_2','passing_entropy']:
  for fe in [False,True]:
   fit,*_=regress(d,'starness_index',x=x,season_fe=fe);fit.update(season='ALL',model='sensitivity_season_FE' if fe else 'sensitivity_unadjusted');results.append(fit)
 result=pd.DataFrame(results)
 # Holm adjustment across the 14 primary overall/per-season regressions.
 primary=result.index[result.model.eq('unadjusted')];order=result.loc[primary,'robust_p'].sort_values().index
 result.loc[order,'holm_p_primary14']=np.minimum(1,np.maximum.accumulate([(len(order)-j)*result.loc[i,'robust_p'] for j,i in enumerate(order)]))
 result.to_csv(out/'regression_results.csv',index=False)
 # Compare the new raw-count model with the previous equal-total-edge model, using identical Y scores.
 comparison=[]
 for label,x in [('Actual raw counts','raw_count_entropy'),('Previous equal total pass/shot weights','augmented_entropy_ratio_1'),('Passing only','passing_entropy')]:
  fit,*_=regress(d,'starness_index',x=x);fit['entropy_definition']=label;comparison.append(fit)
 pd.DataFrame(comparison).to_csv(out/'before_after_comparison.csv',index=False)
 (out/'README.md').write_text("""Actual-count conditional entropy vs action-weighted starness. All 180 team-seasons use the original directed pass counts A and player shot-category FGA counts S with NO rescaling. Categories: other 2PT, 3PT, layup/dunk. Both makes and misses count. Each player row is normalized by its outgoing passes plus FGA. H = -sum(B_ij log2(B_ij / row_total_i))/sum(B), B=[A,S]. Divide by log2(N+2), where N includes all nonisolated player nodes (N-1 possible other players plus three shot sinks). Terminal nodes have no outgoing transitions and no fabricated self-loops. This is one-step conditional entropy, not stationary Markov entropy rate. Exported raw_count_entropy_bits also gives unnormalized bits. Shannon grouping identity independently verifies every team: H = H(pass/shot choice|player) + overall pass share * passing conditional bits + overall FGA share * shot-category conditional bits.

Starness is unchanged from the previous requested version: P/(P+FGA)*passing EC percentile + FGA/(P+FGA)*usage concentration percentile, using frozen 180-team-season ranks. EC is original passing A+A.T, excluding passing isolates; usage concentration is HHI of FGA+0.44FTA+TOV shares, not official USG%. This combines descriptive concentration, not validated causal dependency. A pass and a shot are counted events, not possession fractions; possessions can contain several passes or shots. Pass and FGA totals are verified against source files.

Primary graphs compare actual-count augmented entropy and passing-only entropy overall and separately for each season. No minutes or roster controls. Pooled inference clusters repeated franchises; per-season HC3. Holm-corrected p values cover 14 primary comparisons. Season fixed effects and old shot-scaling sensitivity definitions are separately labeled. Raw r/R² in FE sensitivity rows remain bivariate, while full_model_r_squared describes full model fit. Before/after comparisons keep Y unchanged. Prior graph drawings and outputs remain in their original folders; this new regression uses raw_count, not normalized weight, from the saved edges. Both axes share inputs; a stronger correlation is not independent validation or evidence of improved performance.
""")
 print(result[result.model.eq('unadjusted')][['season','x','r','r_squared','slope','robust_p','holm_p_primary14']].to_string(index=False))
 return 0
if __name__=='__main__':raise SystemExit(main())
