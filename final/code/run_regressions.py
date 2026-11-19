"""Reproduce observed entropy-change regressions for the final ten-plus-absence sample."""
from pathlib import Path
import argparse,tempfile,json,hashlib
import numpy as np,pandas as pd
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pass_strength_entropy_regressions import main as passing_main,pooled_fit
from statistics_helpers import Fit,holm


def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--package-root',type=Path,default=Path(__file__).resolve().parent.parent);p.add_argument('--destination',type=Path);a=p.parse_args(argv);package=a.package_root;source=package/'data/nba_passing_output/all_season_hub_redistribution';out=a.destination or package/'regressions';out.mkdir(parents=True,exist_ok=True)
 d=pd.read_csv(source/'all_team_season_entropy_and_ec.csv');d['primary_eligible']=d.later_hub_absent_games.ge(10)&d.actual_passes.gt(0);d['concurrent_eligible']=d.primary_eligible&d.later_hub_present_games.ge(15)&d.entropy_later_present.notna()&d.concurrent_equal_count_change.notna();assert d.primary_eligible.sum()==100
 # Metadata copy changes evaluation eligibility only. All first20 matrices are linked read-only.
 with tempfile.TemporaryDirectory(prefix='nba-final-regressions-') as temp:
  base=Path(temp);d.to_csv(base/'all_team_season_entropy_and_ec.csv',index=False)
  for z in d.itertuples():
   target=base/z.season/z.team;target.mkdir(parents=True);(target/'first20_all_passing.csv').symlink_to((source/z.season/z.team/'first20_all_passing.csv').resolve())
  passing_main(['--baseline',str(base),'--destination',str(out/'passing_involvement')])
 # Separately labelled first20 passing-only EC-index predictor; same observed outcomes.
 dest=out/'eigenvector_concentration';dest.mkdir(exist_ok=True);pooled=[];seasonal=[];models={}
 specs=[('early_present','actual_equal_count_change','primary_eligible'),('later_present','concurrent_equal_count_change','concurrent_eligible')]
 for label,ycol,elig in specs:
  q=d[d[elig]].copy();result,xc,yc=pooled_fit(q,'ec_index',ycol);result.update(analysis=label,x='ec_index',y=ycol);pooled.append(result);models[label]=(q,xc,yc,result)
  for season,g in q.groupby('season'):
   f=Fit(g.ec_index.to_numpy(),g[ycol].to_numpy());r,pv=pearsonr(g.ec_index,g[ycol]);ci=f.conf_int()[1];seasonal.append(dict(analysis=label,season=season,n=len(g),r=r,r_squared=f.rsquared,slope=f.params[1],hc3_p=f.pvalues[1],pearson_p=pv,ci_low=ci[0],ci_high=ci[1]))
 s=pd.DataFrame(seasonal);s['holm_12_season_tests']=holm(s.hc3_p);ps=pd.DataFrame(pooled);ps['holm_two_pooled_tests']=holm(ps.clustered_p);s.to_csv(dest/'season_regressions.csv',index=False);ps.to_csv(dest/'pooled_regressions.csv',index=False)
 with PdfPages(dest/'ec_entropy_regressions.pdf') as pdf:
  for label,ycol,elig in specs:
   fig,axes=plt.subplots(2,3,figsize=(17,10),sharex=True,sharey=True)
   for ax,(season,g) in zip(axes.flat,d[d[elig]].groupby('season')):
    fit=Fit(g.ec_index.to_numpy(),g[ycol].to_numpy());row=s[s.analysis.eq(label)&s.season.eq(season)].iloc[0];ax.scatter(g.ec_index,g[ycol],s=24)
    for team,x,y in zip(g.team,g.ec_index,g[ycol]):ax.annotate(team,(x,y),fontsize=6,xytext=(2,2),textcoords='offset points')
    grid=np.linspace(g.ec_index.min(),g.ec_index.max(),100);pred=fit.prediction(grid);ax.plot(grid,pred['mean'],c='firebrick');ax.fill_between(grid,pred.mean_ci_lower,pred.mean_ci_upper,color='firebrick',alpha=.12);ax.axhline(0,c='grey',lw=.6);ax.set_title(f'{season} | n={len(g)} | r={row.r:+.3f} | R²={row.r_squared:.3f}\nHC3 p={row.hc3_p:.3f}',fontsize=10);ax.set_xlabel('First20 passing-only EC concentration index');ax.set_ylabel('Actual absent minus present entropy')
   fig.suptitle('EC-index association | '+label.replace('_',' ')+' reference | 10+ later absences',fontsize=15);fig.tight_layout(rect=[0,0,1,.93]);fig.savefig(dest/(label+'_by_season.png'),dpi=150);pdf.savefig(fig);plt.close(fig)
  for label,ycol,elig in specs:
   q,xc,yc,result=models[label];fig,ax=plt.subplots(figsize=(12,8))
   for color,(season,g) in zip(plt.cm.tab10(np.linspace(0,1,q.season.nunique())),q.groupby('season')):ax.scatter(xc.loc[g.index],yc.loc[g.index],color=color,label=season,alpha=.75)
   grid=np.linspace(xc.min(),xc.max(),100);ax.plot(grid,result['slope']*grid,c='firebrick');lo=grid*result['slope_ci_low'];hi=grid*result['slope_ci_high'];ax.fill_between(grid,np.minimum(lo,hi),np.maximum(lo,hi),color='firebrick',alpha=.1);row=ps[ps.analysis.eq(label)].iloc[0];ax.axhline(0,c='grey',lw=.6);ax.axvline(0,c='grey',lw=.6);ax.set_xlabel('EC index minus season average');ax.set_ylabel('Observed entropy change minus season average');ax.set_title(f"{label.replace('_',' ')} | n={len(q)} | within-season r={row.within_season_r:+.3f} | R²={row.within_season_r_squared:.3f}\nFranchise-clustered p={row.clustered_p:.4f} | Holm p={row.holm_two_pooled_tests:.4f}");ax.legend();fig.tight_layout();fig.savefig(dest/(label+'_pooled.png'),dpi=150);pdf.savefig(fig);plt.close(fig)
 d.to_csv(out/'regression_input_with_final_filter.csv',index=False)
 (dest/'METHOD.md').write_text('First20 passing-only EC normalized HHI on A+A.T is an alternative predictor, not a rule for hub identity. Hub identity remains maximum passes sent+received. Outcome is observed later absence minus early present entropy with inherited 500 equal-pass-count draws. Ten-plus tracked later absences and a nonempty observed matrix define the primary 100-case sample; later-present sensitivity also needs15 later present games and a validated reference. Pooled models adjust for season and cluster by franchise. Holm correction is across two pooled EC tests; twelve seasonal tests form a separate family. These observed associations do not measure forecast error and are unchanged by which predicted network is used. The broader exploration remains retrospective.\n')
 (out/'validation.json').write_text(json.dumps(dict(passed=True,primary_n=int(d.primary_eligible.sum()),later_reference_n=int(d.concurrent_eligible.sum()),evaluation_rule='10+ analyzed later absences; no minimum early-present count',prediction_model_uses_regression=False,source_metadata_sha256=hashlib.sha256((source/'all_team_season_entropy_and_ec.csv').read_bytes()).hexdigest()),indent=2));print('Final regression sample:',int(d.primary_eligible.sum()),'later reference:',int(d.concurrent_eligible.sum()),flush=True)

if __name__=='__main__':main()
