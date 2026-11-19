from pathlib import Path
import numpy as np,pandas as pd
from hub_entropy_regressions import Fit,holm
import matplotlib.pyplot as plt

def run(out,d,sample):
 out=Path(out);rows=[];fig,axes=plt.subplots(1,2,figsize=(14,6))
 for ax,outcome in zip(axes,['signed','absolute']):
  x=sample.prior_on_off_gap.to_numpy();y=sample.deletion_entropy_change.to_numpy();y=np.abs(y) if outcome=='absolute' else y;fit=Fit(x,y);r=np.corrcoef(x,y)[0,1];rows.append(dict(outcome=outcome,n=len(x),r=r,r_squared=fit.rsquared,hc3_p=fit.pvalues[1],slope=fit.params[1]));ax.scatter(x,y)
  for team,xx,yy in zip(sample.team,x,y):ax.annotate(team,(xx,yy),fontsize=8,xytext=(4,3),textcoords='offset points')
  grid=np.linspace(x.min(),x.max(),100);ax.plot(grid,fit.prediction(grid)['mean'],c='firebrick');ax.axhline(0,color='grey',lw=.7);ax.set_xlabel('volume-selected player prior on/off ORtg gap');ax.set_ylabel('Deleted − played entropy' if outcome=='signed' else '|Deleted − played entropy|');ax.set_title(f'Synthetic removal: {outcome} | n={len(x)}\nr={r:+.3f}; R²={fit.rsquared:.3f}; HC3 p={fit.pvalues[1]:.3f}')
 fig.suptitle('Mechanical deletion only: not observed player absence');fig.tight_layout();fig.savefig(out/'synthetic_deletion_regressions.pdf');fig.savefig(out/'synthetic_deletion_regressions.png',dpi=150);plt.close(fig);s=pd.DataFrame(rows);s['holm_2']=holm(s.hc3_p);s.to_csv(out/'synthetic_deletion_statistics.csv',index=False);assert np.allclose(d.entropy_change,d.deletion_entropy_change+d.additional_context_change)
