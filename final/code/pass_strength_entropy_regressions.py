"""Observed hub-absence entropy changes versus first-20 passes-in-plus-out centrality."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np,pandas as pd
from scipy.stats import pearsonr,t,linregress
from statistics_helpers import Fit,holm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def pooled_fit(q,xcol,ycol):
    x=q[xcol].to_numpy();y=q[ycol].to_numpy();dummy=pd.get_dummies(q.season,drop_first=True,dtype=float).to_numpy();X=np.column_stack([np.ones(len(q)),x,dummy]);inv=np.linalg.inv(X.T@X);b=inv@X.T@y;e=y-X@b;meat=np.zeros((X.shape[1],X.shape[1]));groups=q.team.unique();G=len(groups)
    for team in groups:
        mask=q.team.eq(team).to_numpy();score=X[mask].T@e[mask];meat+=np.outer(score,score)
    cov=G/(G-1)*(len(q)-1)/(len(q)-X.shape[1])*inv@meat@inv;se=np.sqrt(cov[1,1]);p=2*t.sf(abs(b[1]/se),G-1);ci=t.ppf(.975,G-1)*se
    xc=q[xcol]-q.groupby('season')[xcol].transform('mean');yc=q[ycol]-q.groupby('season')[ycol].transform('mean');r=pearsonr(xc,yc).statistic
    # Independent within-season slope and scalar clustered variance.
    slope=np.dot(xc,yc)/np.dot(xc,xc);scores=pd.DataFrame({'team':q.team,'s':xc*(yc-slope*xc)}).groupby('team').s.sum();v=G/(G-1)*(len(q)-1)/(len(q)-X.shape[1])*np.dot(scores,scores)/np.dot(xc,xc)**2;assert np.isclose(slope,b[1]) and np.isclose(np.sqrt(v),se)
    return dict(n=len(q),seasons=q.season.nunique(),franchises=G,within_season_r=r,within_season_r_squared=r*r,slope=b[1],clustered_p=p,slope_ci_low=b[1]-ci,slope_ci_high=b[1]+ci),xc,yc


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pass-strength-entropy-regressions',action='store_true');p.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'nba_passing_output');p.add_argument('--baseline',type=Path);p.add_argument('--destination',type=Path);a=p.parse_args(argv);base=a.baseline or a.output_dir/'all_season_hub_redistribution';out=a.destination or a.output_dir/'pass_strength_entropy_regressions';out.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(base/'all_team_season_entropy_and_ec.csv');assert len(d)==180;fractions=[];raw=[];totals=[]
    for z in d.itertuples():
        frame=pd.read_csv(base/z.season/z.team/'first20_all_passing.csv',index_col=0);A=frame.to_numpy();ids=frame.index.astype(int).tolist();i=ids.index(z.hub_id);strength=A.sum(0)+A.sum(1);assert strength[i]==strength.max();assert np.isclose(strength[i],z.first20_passes_in_out);S=A.sum();assert S>0 and 0<=strength[i]/S<=1
        raw.append(float(strength[i]));totals.append(float(S));fractions.append(float(strength[i]/S))
    d['hub_passes_in_plus_out']=raw;d['first20_team_passes']=totals;d['hub_pass_involvement_fraction']=fractions;d['hub_pass_involvement_percent']=100*d.hub_pass_involvement_fraction;d['hub_share_of_total_node_strength']=d.hub_pass_involvement_fraction/2
    d.to_csv(out/'team_season_data.csv',index=False)
    specs=[('normalized_early','hub_pass_involvement_fraction','actual_equal_count_change','primary_eligible'),('normalized_later','hub_pass_involvement_fraction','concurrent_equal_count_change','concurrent_eligible'),('raw_early','hub_passes_in_plus_out','actual_equal_count_change','primary_eligible'),('raw_later','hub_passes_in_plus_out','concurrent_equal_count_change','concurrent_eligible')]
    seasonal=[];pooled=[];models={}
    for label,xcol,ycol,elig in specs:
        q=d[d[elig]].copy();assert np.isfinite(q[[xcol,ycol]]).all().all();result,xc,yc=pooled_fit(q,xcol,ycol);result.update(analysis=label,x=xcol,y=ycol,effect_per_10_percentage_points=result['slope']*.1 if label.startswith('normalized') else np.nan,effect_per_1000_passes=result['slope']*1000 if label.startswith('raw') else np.nan);pooled.append(result);models[label]=(q,xc,yc,result)
        for season,g in q.groupby('season'):
            x=g[xcol].to_numpy();y=g[ycol].to_numpy();fit=Fit(x,y);r,pval=pearsonr(x,y);ci=fit.conf_int()[1];check=linregress(x,y);assert np.isclose(check.slope,fit.params[1]) and np.isclose(check.rvalue,r)
            seasonal.append(dict(analysis=label,season=season,n=len(g),r=r,r_squared=fit.rsquared,intercept=fit.params[0],slope=fit.params[1],hc3_p=fit.pvalues[1],pearson_p=pval,slope_ci_low=ci[0],slope_ci_high=ci[1]))
    s=pd.DataFrame(seasonal);s['holm_all_season_tests']=holm(s.hc3_p);s.to_csv(out/'season_regressions.csv',index=False);ps=pd.DataFrame(pooled);ps['holm_four_pooled_tests']=holm(ps.clustered_p);ps.to_csv(out/'pooled_regressions.csv',index=False)
    with PdfPages(out/'pass_strength_entropy_regressions.pdf') as pdf:
        for label,xcol,ycol,elig in specs:
            fig,axes=plt.subplots(2,3,figsize=(17,10),sharex=True,sharey=True)
            for ax,(season,g) in zip(axes.flat,d[d[elig]].groupby('season')):
                x=g[xcol].to_numpy();y=g[ycol].to_numpy();fit=Fit(x,y);factor=100 if label.startswith('normalized') else 1;ax.scatter(x*factor,y,s=27)
                for team,xx,yy in zip(g.team,x,y):ax.annotate(team,(xx*factor,yy),fontsize=6,xytext=(2,2),textcoords='offset points')
                grid=np.linspace(x.min(),x.max(),100);pred=fit.prediction(grid);ax.plot(grid*factor,pred['mean'],c='firebrick');ax.fill_between(grid*factor,pred.mean_ci_lower,pred.mean_ci_upper,alpha=.12,color='firebrick');row=s[s.analysis.eq(label)&s.season.eq(season)].iloc[0];ax.axhline(0,c='grey',lw=.6);ax.set_title(f'{season} | n={len(g)}, r={row.r:+.3f}, R²={row.r_squared:.3f}\nHC3 p={row.hc3_p:.3f}',fontsize=11);ax.set_xlabel('First-20 passes involving hub (%)' if factor==100 else 'First-20 hub passes sent + received');ax.set_ylabel('Actual absent − present entropy')
            fig.suptitle('Passes-in + passes-out centrality | '+label.replace('_',' '),fontsize=16);fig.tight_layout(rect=[0,.02,1,.94]);fig.savefig(out/(label+'_by_season.png'),dpi=150);pdf.savefig(fig);plt.close(fig)
        for label,xcol,ycol,elig in specs:
            q,xc,yc,result=models[label];factor=100 if label.startswith('normalized') else 1;fig,ax=plt.subplots(figsize=(12,8))
            for color,(season,g) in zip(plt.cm.tab10(np.linspace(0,1,q.season.nunique())),q.groupby('season')):ax.scatter(xc.loc[g.index]*factor,yc.loc[g.index],label=season,color=color,alpha=.7)
            grid=np.linspace(xc.min(),xc.max(),100);ax.plot(grid*factor,grid*result['slope'],c='firebrick',label='Season-adjusted fit');lo=grid*result['slope_ci_low'];hi=grid*result['slope_ci_high'];ax.fill_between(grid*factor,np.minimum(lo,hi),np.maximum(lo,hi),color='firebrick',alpha=.1);ax.axhline(0,c='grey',lw=.6);ax.axvline(0,c='grey',lw=.6);row=ps[ps.analysis.eq(label)].iloc[0];ax.set_title(f'{label.replace("_"," ")} | season-adjusted pooled association\nn={row.n}, within-season r={row.within_season_r:+.3f}, R²={row.within_season_r_squared:.3f}\nFranchise-clustered p={row.clustered_p:.4f}; Holm p={row.holm_four_pooled_tests:.4f}');ax.set_xlabel('Hub pass involvement minus season mean (percentage points)' if factor==100 else 'Hub passes sent + received minus season mean');ax.set_ylabel('Entropy change minus season mean');ax.legend();fig.tight_layout();fig.savefig(out/(label+'_pooled.png'),dpi=150);pdf.savefig(fig);plt.close(fig)
    method=['# Entropy change versus passes-in-plus-out centrality','Uses the restored pre-EC player-selection study: first-20 passes-sent-plus-received leader. No tracking TOUCHES, no EC-selected hubs, and no replacement-model prediction errors are substituted for the outcome.',
            'Primary X=(hub passes sent + hub passes received)/(all first20 team passes). It is the fraction of recorded team passes involving the hub. With self-passes excluded, a given hub can occur only once in a pass, so no double counting occurs in its numerator. The alternative node-strength share divides by twice team passes; that rescales X by one-half and yields identical r, R2 and p. Raw first20 passes-in-plus-out is a declared sensitivity.',
            f'Primary Y=observed later hub-absent entropy minus early hub-present entropy, using the existing500 equal-pass-count subsamples. Positive means higher outgoing conditional entropy without the hub. The later hub-present reference is a sensitivity. The {len(d)}-team-season source has {int(d.primary_eligible.sum())} primary-eligible observations and {int(d.concurrent_eligible.sum())} later-reference eligible observations, with the ten-plus tracked later absence evaluation filter and inherited tracking/trade exclusions. No early-present count threshold is imposed for evaluation; the later-present sensitivity requires at least15 later present games. Actual matrices, selected hubs and replacement forecasts are unchanged.',
            'Six per-season HC3 fits for each of four declared analyses are exported (24 tests, Holm correction across24). Four pooled fits include season indicators and franchise-clustered sandwich uncertainty with finite-sample correction and t(G-1) inference. Holm correction across the four pooled tests is reported separately. Pooled r and R2 use within-season-demeaned variables; they are not total model R2 including season effects.',
            'This remains exploratory after extensive previous analyses. Hub involvement is cumulative activity over the first20 team games and reflects availability. Raw totals additionally reflect passing volume/pace; normalized involvement reduces that scale difference. X and the original-network entropy share inputs, and node removal mechanically changes entropy. Temporal drift, roster mixing, co-absence and opponents remain. These are structural associations, not causal player impact, offensive efficiency or prediction accuracy.',ps.to_csv(index=False)]
    (out/'METHOD_AND_RESULTS.md').write_text('\n\n'.join(method));(out/'validation.json').write_text(json.dumps(dict(passed=True,team_seasons=180,primary_n=int(d.primary_eligible.sum()),later_reference_n=int(d.concurrent_eligible.sum()),checks='First20 hub strengths reconstructed and verified; highest-strength identity checked; seasonal slopes/r checked with SciPy; pooled slopes and clustered SEs checked independently by within-season scalar scores',source_sha256=hashlib.sha256((base/'all_team_season_entropy_and_ec.csv').read_bytes()).hexdigest()),indent=2));print(ps.to_string(index=False),flush=True);return 0

if __name__=='__main__':raise SystemExit(main())
