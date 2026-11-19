"""Descriptive team-level regressions of entropy change on passing EC concentration."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def holm(p):
 p=np.asarray(p);order=np.argsort(p);adjusted=np.empty(len(p));adjusted[order]=np.minimum(1,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))));return adjusted

class Fit:
 def __init__(self,x,y):
  X=np.column_stack([np.ones(len(x)),x]);self.inv=np.linalg.inv(X.T@X);self.params=self.inv@X.T@y;res=y-X@self.params;h=np.sum((X@self.inv)*X,axis=1);self.cov=self.inv@(X.T@((res/(1-h))[:,None]**2*X))@self.inv;self.df=len(x)-2;self.se=np.sqrt(np.diag(self.cov));self.pvalues=2*st.t.sf(np.abs(self.params/self.se),self.df);self.rsquared=1-(res@res)/np.sum((y-y.mean())**2)
 def conf_int(self):
  width=st.t.ppf(.975,self.df)*self.se;return np.column_stack([self.params-width,self.params+width])
 def prediction(self,x):
  X=np.column_stack([np.ones(len(x)),x]);mean=X@self.params;width=st.t.ppf(.975,self.df)*np.sqrt(np.maximum(0,np.sum((X@self.cov)*X,axis=1)));return pd.DataFrame({'mean':mean,'mean_ci_lower':mean-width,'mean_ci_upper':mean+width})

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--hub-entropy-regressions',action='store_true');p.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'nba_passing_output');p.add_argument('--input',type=Path);p.add_argument('--destination',type=Path);a=p.parse_args(argv)
 source=a.input or a.output_dir/'league_star_presence_entropy/all_30_team_entropy.csv';out=a.destination or a.output_dir/'league_star_presence_entropy/centrality_regressions';out.mkdir(parents=True,exist_ok=True)
 allteams=pd.read_csv(source);d=allteams[allteams.status.isin(['complete_cached','complete_new'])].copy();assert len(d)==27 and d.team.nunique()==27;assert np.isfinite(d[['passing_ec_concentration','entropy_change','equal_total_entropy_change']]).all().all()
 excluded=d[d.absent_games.lt(10)].copy();excluded.to_csv(out/'excluded_low_absence_teams.csv',index=False);d=d[d.absent_games.ge(10)].copy();assert len(d)==21
 d['equal_total_absolute_change']=d.equal_total_entropy_change.abs();d['raw_absolute_change']=d.entropy_change.abs();d.to_csv(out/'regression_team_data.csv',index=False)
 results=[];models={}
 for measure,column in [('equal_total','equal_total_entropy_change'),('raw','entropy_change')]:
  for sample,frame in [('at_least_10_absence_games',d)]:
   x=frame.passing_ec_concentration.to_numpy()
   for outcome in ['signed','absolute']:
    y=frame[column].to_numpy();y=np.abs(y) if outcome=='absolute' else y;model=Fit(x,y);r,pearson_p=st.pearsonr(x,y);rho,rho_p=st.spearmanr(x,y);ci=model.conf_int()[1];key=(measure,sample,outcome);models[key]=(frame,x,y,model)
    results.append(dict(measure=measure,sample=sample,outcome=outcome,n=len(frame),r=r,r_squared=model.rsquared,slope=model.params[1],slope_per_001=model.params[1]*.01,intercept=model.params[0],hc3_slope_ci_low=ci[0],hc3_slope_ci_high=ci[1],hc3_p=model.pvalues[1],pearson_p=pearson_p,spearman_rho=rho,spearman_p=rho_p))
 stats=pd.DataFrame(results);stats['holm_p_all_4']=holm(stats.hc3_p);primary=stats.measure.eq('equal_total')&stats['sample'].eq('at_least_10_absence_games');stats['holm_p_primary_2']=np.nan;stats.loc[primary,'holm_p_primary_2']=holm(stats.loc[primary,'hc3_p']);stats.to_csv(out/'regression_statistics.csv',index=False)
 # Leave-one-team-out sensitivity checks use the same predefined regressions.
 loo=[]
 for key,(frame,x,y,model) in models.items():
  for i,team in enumerate(frame.team):
   keep=np.arange(len(x))!=i;fit=Fit(x[keep],y[keep]);loo.append(dict(measure=key[0],sample=key[1],outcome=key[2],removed_team=team,slope=fit.params[1],hc3_p=fit.pvalues[1],r_squared=fit.rsquared))
 pd.DataFrame(loo).to_csv(out/'leave_one_team_out.csv',index=False)
 with PdfPages(out/'centrality_entropy_regressions.pdf') as pdf:
  for measure in ['equal_total','raw']:
   fig,axes=plt.subplots(1,2,figsize=(15,6.5))
   for row,sample in enumerate(['at_least_10_absence_games']):
    for col,outcome in enumerate(['signed','absolute']):
     ax=axes[col];frame,x,y,model=models[(measure,sample,outcome)];s=stats[(stats.measure==measure)&(stats['sample']==sample)&(stats.outcome==outcome)].iloc[0];ax.scatter(x,y,c='#287b9b',s=45,alpha=.85)
     for team,xx,yy in zip(frame.team,x,y):ax.annotate(team,(xx,yy),xytext=(4,3),textcoords='offset points',fontsize=8)
     grid=np.linspace(x.min(),x.max(),150);pred=model.prediction(grid);ax.plot(grid,pred['mean'],c='#aa343b');ax.fill_between(grid,pred.mean_ci_lower,pred.mean_ci_upper,color='#aa343b',alpha=.12);ax.axhline(0,c='grey',lw=.7);ax.set_xlabel('Passing-only EC concentration index');ax.set_ylabel('Entropy absent − present' if outcome=='signed' else '|Entropy absent − present|');ax.set_title(f'{"Direction" if outcome=="signed" else "Magnitude"} | At least 10 absence games\nn={len(frame)}; r={s.r:+.3f}; R²={s.r_squared:.3f}; HC3 p={s.hc3_p:.3f}\nSlope per +0.01 EC={s.slope_per_001:+.4f}',fontsize=11);ax.grid(alpha=.15)
   fig.suptitle(f'2025–26: EC concentration and hub-absence entropy change\n{"Equal-total-pass analysis (primary)" if measure=="equal_total" else "Raw full-network analysis (sensitivity)"}',fontsize=16);fig.text(.5,.015,'Only teams with at least 10 hub-absence games enter points, fits and statistics. Lines: OLS; shading: HC3 95% confidence interval for fitted mean.\nCurrent-season hub selection; unmatched game contexts. Association and structural change, not causal dependency or offensive harm.',ha='center',fontsize=10);fig.tight_layout(rect=[0,.07,1,.92]);fig.savefig(out/f'{measure}_regressions.png',dpi=170);pdf.savefig(fig);plt.close(fig)
 text=['# Passing centralization and hub-absence entropy change','2025–26 descriptive team comparison. 27 observed pairs exist; all regressions now use only the 21 teams with at least 10 absence games. BOS, HOU, MIL, MIN, NYK and PHX are excluded from points, fits and statistics at the user’s request. LAC, NOP and WAS lack actual hub-absent games and are not assigned zeros.','## Method','X is the existing full-season passing-only EC concentration index: (sum(p_i²)−1/N)/(1−1/N), using L1 EC shares on A+Aᵀ among active passing players. No shooting edges enter. H is outgoing conditional entropy H(receiver|passer), with the same full-season active-player denominator log2(N−1) in both contexts. The present network includes the hub. The absent network uses actual absent games.','Primary Y is the mean entropy difference after 500 independent without-replacement equal-total-pass subsamples. Signed change tests direction; the absolute value of that mean difference tests magnitude. It is not the mean absolute difference across draws, which would incorporate sampling noise differently. Matching pass counts does not match game context or donor activity. The raw-network counterparts are sensitivity checks; the same 21-team sample is used throughout.','OLS fits use HC3 robust standard errors and t-based 95% slope intervals. Pearson r and OLS R² describe the fitted linear association. The main reported p value tests the slope with HC3; conventional Pearson p is separately exported. Holm adjustment is provided for the two primary tests and, conservatively, the four retained analyses. The ≥10-game restriction was requested after reviewing the earlier results; adjusted p values for this revision do not erase that exploratory history.','|Pass sample|Team sample|Outcome|n|r|R²|Slope per +.01 EC|HC3 p|Holm p (4)|','|---|---|---|---:|---:|---:|---:|---:|---:|']
 for z in stats.itertuples():text.append(f'|{z.measure}|{z.sample}|{z.outcome}|{z.n}|{z.r:+.3f}|{z.r_squared:.3f}|{z.slope_per_001:+.4f}|{z.hc3_p:.4f}|{z.holm_p_all_4:.4f}|')
 text+=['## Interpretation and limits','A positive signed slope means centralized teams tend to show a more positive entropy change without the hub. A positive magnitude slope means more centralized teams tend to change more in either direction. Neither implies worse offense. EC and entropy share passing-network inputs, which can create mathematical coupling. EC and hub identity use the same full season, including absence contexts, so this is retrospective. Minutes, opponents, other absences, trades and scheduling are not controlled. Absolute change is nonnegative and noisy; an exploratory straight line is not a full measurement-error model. Leave-one-team-out slopes and p values are exported to expose influential teams. Equal-total sampling ranges do not provide independent team observations or causal confidence intervals.','The previous-season-defined extension is a separate future analysis: it requires rebuilding presence/absence partitions around those hubs. Existing current-hub networks cannot simply be relabelled.','Files: regression_statistics.csv; regression_team_data.csv; leave_one_team_out.csv; centrality_entropy_regressions.pdf; equal_total_regressions.png; raw_regressions.png.']
 text+=['## Reading the uncertainty','Compare HC3 slope p, conventional Pearson p, and both Holm corrections in the exported statistics. A result that crosses 0.05 under different reasonable choices should be described as exploratory and method-sensitive. Compare signed and absolute outcomes separately: a directional trend does not demonstrate larger structural changes or performance losses.']
 (out/'METHOD_AND_RESULTS.md').write_text('\n\n'.join(text));(out/'verification.json').write_text(json.dumps(dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),observed_teams=27,regression_teams=len(d),excluded_teams=excluded.team.tolist(),stricter_teams=int(d.absent_games.ge(10).sum()),analyses=len(stats),primary_tests=2),indent=2));print(stats.to_string(index=False));return 0
if __name__=='__main__':raise SystemExit(main())
