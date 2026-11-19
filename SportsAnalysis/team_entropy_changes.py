"""Plot all 180 restored passing-hub team-seasons without changing any networks."""
from pathlib import Path
import sys,json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import pearsonr
ROOT=Path('/Users/timothysu/Documents/Codex/2026-11-05/referenced-chatgpt-conversation-this-is-an-2')
SOURCE=ROOT/'outputs/pass_strength_entropy_regressions'
SPLIT=Path('/Users/timothysu/Desktop/SportsAnalysis/nba_passing_output/split_role_hub_redistribution')
OUT=ROOT/'outputs/pass_strength_team_changes';OUT.mkdir(parents=True,exist_ok=True)
d=pd.read_csv(SOURCE/'team_season_data.csv');pred=pd.read_csv(SPLIT/'prediction_comparison.csv')
assert not d.duplicated(['season','team']).any() and len(d)==180
p=pred[['season','team','hub_id','entropy_original','entropy_actual','entropy_new']].rename(columns={'entropy_original':'check_original','entropy_actual':'check_actual','hub_id':'check_hub_id','entropy_new':'entropy_predicted_split_role'})
d=d.merge(p,on=['season','team'],validate='one_to_one');assert len(d)==180
assert (d.hub_id==d.check_hub_id).all()
for a,b in [('entropy_original','check_original'),('entropy_actual','check_actual')]:assert np.allclose(d[a],d[b],equal_nan=True)
d['split_role_predicted_change']=d.entropy_predicted_split_role-d.entropy_original
d['split_role_entropy_error']=d.entropy_actual-d.entropy_predicted_split_role
d.drop(columns=['check_original','check_actual','check_hub_id']).sort_values(['season','team']).to_csv(OUT/'all_team_entropy_changes.csv',index=False)
q=d[d.primary_eligible].copy();xc=q.hub_pass_involvement_fraction-q.groupby('season').hub_pass_involvement_fraction.transform('mean');yc=q.actual_equal_count_change-q.groupby('season').actual_equal_count_change.transform('mean');slope=np.dot(xc,yc)/np.dot(xc,xc)
q['regression_residual']=yc-slope*xc
q.sort_values('actual_equal_count_change',ascending=False).to_csv(OUT/'eligible_team_changes_ranked.csv',index=False)
leave=[]
for team in sorted(q.team.unique()):
 z=q[q.team.ne(team)];x=z.hub_pass_involvement_fraction-z.groupby('season').hub_pass_involvement_fraction.transform('mean');y=z.actual_equal_count_change-z.groupby('season').actual_equal_count_change.transform('mean');leave.append({'excluded_franchise':team,'n':len(z),'within_season_r':pearsonr(x,y).statistic})
pd.DataFrame(leave).to_csv(OUT/'leave_one_franchise_out.csv',index=False)
summary={'eligible_n':len(q),'positive_changes':int((q.actual_equal_count_change>0).sum()),'negative_changes':int((q.actual_equal_count_change<0).sum()),'median_change':q.actual_equal_count_change.median(),'mean_change':q.actual_equal_count_change.mean(),'leave_one_franchise_r_min':min(z['within_season_r'] for z in leave),'leave_one_franchise_r_max':max(z['within_season_r'] for z in leave),'prediction_median_abs_entropy_error':q.split_role_entropy_error.abs().median(),'prediction_overestimates_count':int((q.split_role_entropy_error<0).sum()),'prediction_underestimates_count':int((q.split_role_entropy_error>0).sum())}
pd.DataFrame([{'season':s,'n':len(g),'increased':sum(g.actual_equal_count_change>0),'decreased':sum(g.actual_equal_count_change<0),'median_change':g.actual_equal_count_change.median()} for s,g in q.groupby('season')]).to_csv(OUT/'season_change_summary.csv',index=False)
summary['largest_positive']=q.nlargest(5,'actual_equal_count_change')[['season','team','hub_name','actual_equal_count_change']].to_dict('records');summary['largest_negative']=q.nsmallest(5,'actual_equal_count_change')[['season','team','hub_name','actual_equal_count_change']].to_dict('records')
(OUT/'analysis_summary.json').write_text(json.dumps(summary,indent=2))
seasons=sorted(d.season.unique());teams=sorted(d.team.unique());assert len(teams)==30
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
with PdfPages(OUT/'all_30_teams_entropy_changes.pdf') as pdf:
 for season in seasons:
  g=d[d.season.eq(season)].sort_values('team');fig,ax=plt.subplots(figsize=(13,13));y=np.arange(30)
  for yy,z in zip(y,g.itertuples()):
   v=z.actual_equal_count_change
   if np.isfinite(v):
    color=('#197c68' if v>=0 else '#b54050') if z.primary_eligible else '#abb0b7'
    ax.barh(yy,v,color=color,height=.66);ax.text(v+(.003 if v>=0 else -.003),yy,f'{v:+.3f}',ha='left' if v>=0 else 'right',va='center',fontsize=9)
   else:ax.text(.003,yy,'No observed comparison',va='center',fontsize=8,color='#777777')
  labels=[f'{z.team} | {z.hub_name} | {z.later_hub_absent_games} absent'+(' *' if not z.primary_eligible else '') for z in g.itertuples()]
  ax.set_yticks(y,labels);ax.invert_yaxis();ax.axvline(0,color='#424242',lw=.8);ax.set_xlim(-.14,.22);ax.grid(axis='x',alpha=.15);ax.set_xlabel('Equal-pass-count normalized entropy change: actual absent minus early present')
  ax.set_title(f'{season}: entropy change for every team\nGreen = increase; red = decrease; grey / * = outside primary regression sample',pad=16)
  fig.text(.5,.015,'Hub = first-20 passes sent + received leader. Later absences only. Missing comparisons are not zero.\nBars use the same equal-pass-count outcome as the regression; they do not measure offensive quality.',ha='center',fontsize=9)
  fig.tight_layout(rect=[0,.055,1,1]);fig.savefig(OUT/f'{season}_all_team_changes.png',dpi=150);pdf.savefig(fig);plt.close(fig)
 # One page per franchise, six independent team-season comparisons, no temporal lines across seasons.
 for team in teams:
  g=d[d.team.eq(team)].set_index('season');fig,axes=plt.subplots(2,3,figsize=(15,9),sharey=True)
  for ax,season in zip(axes.flat,seasons):
   z=g.loc[season];values=[z.entropy_original,z.entropy_predicted_split_role,z.entropy_actual];observed=np.isfinite(z.entropy_actual)
   if not observed:values[2]=np.nan
   ax.plot([0,1,2],values,color='#bac2c9',lw=1.5,zorder=1)
   for x,val,c in zip([0,1,2],values,['#356ba0','#a75f15','#197c68']):
    if np.isfinite(val):ax.scatter(x,val,color=c,s=65,zorder=3);ax.annotate(f'{val:.3f}',(x,val),xytext=(0,9),textcoords='offset points',ha='center',fontsize=9)
   if not observed:ax.text(2,.15,'No actual\ncomparison',ha='center',fontsize=8)
   ax.set_xticks([0,1,2],['Present','Predicted\nabsent','Actual\nabsent']);ax.set_xlim(-.35,2.35);ax.set_ylim(0,1);ax.grid(axis='y',alpha=.15);ax.set_ylabel('Normalized outgoing conditional entropy')
   ax.set_title(f'{season} | {z.hub_name}\n{int(z.later_hub_absent_games)} tracked absent games | '+('primary eligible' if z.primary_eligible else 'outside primary sample'),fontsize=9)
   if np.isfinite(z.actual_equal_count_change):ax.text(.04,.05,f'Regression ΔH = {z.actual_equal_count_change:+.3f}',transform=ax.transAxes,fontsize=9)
  fig.suptitle(f'{team}: original, replacement forecast, and observed entropy',fontsize=16)
  fig.text(.5,.02,'Dots show full-network entropy levels; the forecast uses separate minutes and passing-role redistribution.\nRegression ΔH uses equal-pass-count sampling, so it can differ slightly from the actual-minus-present dot difference.\nConnecting lines guide comparison of scenarios; they are not game-by-game trajectories. Seasons may have different hubs and normalization references.',ha='center',fontsize=9)
  fig.tight_layout(rect=[0,.095,1,.95]);fig.savefig(OUT/f'{team}_six_season_entropy.png',dpi=140);pdf.savefig(fig);plt.close(fig)
method='''# Team-by-team entropy changes
All 180 restored volume-selected hub team-seasons are shown, including those outside the 96-observation primary regression. No networks, hubs, entropy estimators, or regression results were changed. No NBA downloads were made.

Six league pages show the exact equal-pass-count actual-minus-early-present entropy changes used in the regression. Grey bars mark comparisons outside the primary sample; missing values have no bar. These exploratory bars have no confidence intervals and do not establish individual-team statistical significance.

Thirty franchise pages show six season panels each: early hub-present network, the latest split-role replacement forecast, and later actual hub-absent network. Full-network entropy levels differ from pass-count-matched regression differences. Predicted entropy is not used as an outcome in the centrality regression. A common within-team-season denominator is inherited; denominators and pooled rosters can differ across seasons. No direct year-to-year performance trend is inferred.

Higher entropy means a more diverse receiver distribution conditional on sender, averaged over recorded passes. It does not prove better offense, more equal player involvement, or less reliance on stars. Distinct passing patterns can have the same entropy. Removing a hub can mechanically change entropy; later roster mixing, opponents, co-absences and temporal drift may also matter.

The regression is positive and modest, with exploratory pooled season-adjusted, franchise-clustered inference inherited from the prior report. Leave-one-franchise-out correlations are descriptive robustness checks, not fresh confirmatory tests. Multiple exploratory analyses have already been considered.
'''
(OUT/'README.md').write_text(method+'\n'+json.dumps(summary,indent=2));assert len(list(OUT.glob('*_six_season_entropy.png')))==30;assert len(list(OUT.glob('*_all_team_changes.png')))==6
print(json.dumps(summary,indent=2))
