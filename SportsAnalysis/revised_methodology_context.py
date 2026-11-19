"""Clearly labeled opportunity-conditioned sensitivity models; not primary estimands."""
import numpy as np,pandas as pd
from scipy.stats import t

def context_fit(d,x,y,features,season_fe=False):
 controls=[np.ones(len(d))]+[d[c].to_numpy(float) for c in features]
 if season_fe:
  dummy=pd.get_dummies(d.season,drop_first=True,dtype=float);controls.extend(dummy[c].to_numpy() for c in dummy)
 C=np.column_stack(controls);xx=d[x].to_numpy(float);yy=d[y].to_numpy(float);X=np.column_stack([C,xx]);assert np.linalg.matrix_rank(X)==X.shape[1]
 b=np.linalg.lstsq(X,yy,rcond=None)[0];res=yy-X@b;bread=np.linalg.inv(X.T@X);G=d.team.nunique();meat=np.zeros_like(bread)
 for team in d.team.unique():
  k=d.team.eq(team).to_numpy();u=X[k].T@res[k];meat+=np.outer(u,u)
 cov=bread@meat@bread*G/(G-1)*(len(d)-1)/(len(d)-X.shape[1]);se=float(np.sqrt(cov[-1,-1]));df=G-1;critical=t.ppf(.975,df)
 xr=xx-C@np.linalg.lstsq(C,xx,rcond=None)[0];yr=yy-C@np.linalg.lstsq(C,yy,rcond=None)[0];r=float(np.corrcoef(xr,yr)[0,1])
 return dict(x=x,y=y,n=len(d),controls=' + '.join(features),season_FE=season_fe,slope=b[-1],partial_r=r,partial_r_squared=r*r,robust_p=2*t.sf(abs(b[-1]/se),df),ci_low=b[-1]-critical*se,ci_high=b[-1]+critical*se,standard_errors='franchise-clustered',df=df,interpretation='Sensitivity: association conditional on opportunity distribution; different estimand from primary total association')

def temporal_context(d,x,y,features,train_end='2024-25',test='2025-26'):
 train=d[d.season<=train_end];held=d[d.season.eq(test)];cols=[x]+features
 X=np.column_stack([np.ones(len(train))]+[train[c].to_numpy(float) for c in cols]);T=np.column_stack([np.ones(len(held))]+[held[c].to_numpy(float) for c in cols]);v=train[y].to_numpy();b=np.linalg.lstsq(X,v,rcond=None)[0];pred=T@b
 B=np.column_stack([np.ones(len(train))]+[train[c].to_numpy(float) for c in features]);Q=np.column_stack([np.ones(len(held))]+[held[c].to_numpy(float) for c in features]);baseline=Q@np.linalg.lstsq(B,v,rcond=None)[0];actual=held[y].to_numpy();sse=np.sum((actual-pred)**2);base_sse=np.sum((actual-baseline)**2)
 return dict(x=x,y=y,train_n=len(train),test_n=len(held),controls=' + '.join(features),test_rmse=np.sqrt(sse/len(held)),opportunity_only_baseline_rmse=np.sqrt(base_sse/len(held)),incremental_R2_vs_opportunity_only=1-sse/base_sse,predictive_R2_vs_test_mean=1-sse/np.sum((actual-actual.mean())**2),status='retrospective temporal sensitivity; coefficients fit only to development seasons')

def focal_predictions(d,x='passing_entropy_excess',y='shooting_hhi_unbiased'):
 train=d[d.season<='2024-25'];held=d[d.season.eq('2025-26')];result=held[['team','season',y]].copy()
 for label,cols in [('entropy_only',[x]),('opportunity_only',['roster_players','minutes_hhi']),('opportunity_plus_entropy',['roster_players','minutes_hhi',x])]:
  X=np.column_stack([np.ones(len(train))]+[train[c].to_numpy(float) for c in cols]);T=np.column_stack([np.ones(len(held))]+[held[c].to_numpy(float) for c in cols]);b=np.linalg.lstsq(X,train[y],rcond=None)[0];result['predicted_'+label]=T@b
 return result

def focal_figure(d,out):
 import matplotlib.pyplot as plt
 pred=focal_predictions(d);pred.to_csv(out/'temporal_focal_predictions.csv',index=False)
 fig,axes=plt.subplots(2,2,figsize=(14,11));y='shooting_hhi_unbiased'
 for ax,x,title in [(axes[0,0],'passing_entropy','Raw passing entropy'),(axes[0,1],'passing_entropy_excess','Passing entropy minus activity reference')]:
  ax.scatter(d[x],d[y],s=21,alpha=.65);b=np.polyfit(d[x],d[y],1);xx=np.linspace(d[x].min(),d[x].max(),100);ax.plot(xx,np.polyval(b,xx),c='#b54236');r=np.corrcoef(d[x],d[y])[0,1];ax.set_title(f'{title}\nr={r:.3f}, R²={r*r:.1%}');ax.set_xlabel(title);ax.set_ylabel('Shooting concentration (pair-based FGA HHI)')
 C=np.column_stack([np.ones(len(d)),d.roster_players,d.minutes_hhi]);x=d.passing_entropy_excess.to_numpy();v=d[y].to_numpy();xr=x-C@np.linalg.lstsq(C,x,rcond=None)[0];yr=v-C@np.linalg.lstsq(C,v,rcond=None)[0];ax=axes[1,0];ax.scatter(xr,yr,s=21,alpha=.65);b=np.polyfit(xr,yr,1);xx=np.linspace(xr.min(),xr.max(),100);ax.plot(xx,np.polyval(b,xx),c='#b54236');r=np.corrcoef(xr,yr)[0,1];ax.set_title(f'Opportunity-conditioned sensitivity\nPartial r={r:.3f}, partial R²={r*r:.1%}');ax.set_xlabel('Passing entropy difference after roster/minutes comparison');ax.set_ylabel('Shooting HHI after roster/minutes comparison')
 ax=axes[1,1]
 for label,c in [('opportunity_only','#d28a35'),('opportunity_plus_entropy','#287891')]:
  rmse=np.sqrt(np.mean((pred['predicted_'+label]-pred[y])**2));ax.scatter(pred[y],pred['predicted_'+label],s=30,alpha=.65,label=f'{label.replace("_"," ")}: RMSE={rmse:.5f}',c=c)
 limits=[min(pred[y].min(),pred.filter(like='predicted_').min().min()),max(pred[y].max(),pred.filter(like='predicted_').max().max())];ax.plot(limits,limits,c='grey',ls='--');ax.set_xlabel('Observed 2025–26 shooting HHI');ax.set_ylabel('Prediction from first five seasons');change=np.sum((pred.predicted_opportunity_plus_entropy-pred[y])**2)/np.sum((pred.predicted_opportunity_only-pred[y])**2)-1;ax.set_title(f'Does entropy improve the opportunity baseline?\nAdding entropy changes squared error by {change:+.1%}');ax.legend(fontsize=8)
 fig.suptitle('Primary comparison: why the baseline and opportunity checks matter',fontsize=16)
 fig.text(.5,.02,'Panels use the same FGA-HHI outcome. Controls are secondary sensitivities, not causal corrections or replacements for the primary model.\nTemporal coefficients are frozen on 150 earlier team-seasons; 30 latest-season points were already explored previously, so this is retrospective.',ha='center',fontsize=9);fig.tight_layout(rect=[0,.07,1,.94]);fig.savefig(out/'primary_comparison.png',dpi=150);fig.savefig(out/'primary_comparison.pdf');plt.close(fig)
