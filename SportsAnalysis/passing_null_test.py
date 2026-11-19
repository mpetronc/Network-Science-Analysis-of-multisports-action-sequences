"""Independent passing null: fixed outgoing totals, expected incoming totals, forbidden self-passes."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from entropy_structure_profiles import regress

def metrics(A):
 row=A.sum(1);p=np.divide(A,row[:,None],out=np.zeros_like(A,dtype=float),where=row[:,None]>0)
 h=float(-np.sum(A*np.log2(np.maximum(p,1e-300)))/A.sum()/np.log2(len(A)-1))
 _,vec=np.linalg.eigh(A+A.T);v=np.abs(vec[:,-1]);v/=v.sum();return h,float(((v@v)-1/len(v))/(1-1/len(v)))
def expected_counts(A):
 row=A.sum(1);col=A.sum(0);E=np.ones_like(A,dtype=float);np.fill_diagonal(E,0)
 for iteration in range(10000):
  E*=np.divide(row,E.sum(1),out=np.zeros_like(row,dtype=float),where=E.sum(1)>0)[:,None]
  E*=np.divide(col,E.sum(0),out=np.zeros_like(col,dtype=float),where=E.sum(0)>0)[None,:]
  if max(np.abs(E.sum(1)-row).max(),np.abs(E.sum(0)-col).max())<1e-8:break
 else:raise ValueError('Expected-margin fitting did not converge')
 assert not np.diag(E).any();return E

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'nba_passing_output');p.add_argument('--destination',type=Path);p.add_argument('--draws',type=int,default=500);a=p.parse_args(argv);root=a.output_dir;out=a.destination or root/'fg_attempt_networks/structure_validation/null_model';out.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(20261116);rows=[];allh=[];allc=[];sources=[]
 for k,file in enumerate(sorted((root/'full_roster_entropy_centrality/historical_full_rosters').glob('20*/*/passing_matrix.csv')),1):
  A=pd.read_csv(file,index_col=0).to_numpy(float);active=A.sum(0)+A.sum(1)>0;A=A[np.ix_(active,active)].astype(np.int64);row=A.sum(1);E=expected_counts(A);prob=np.divide(E,E.sum(1)[:,None],out=np.zeros_like(E),where=E.sum(1)[:,None]>0);oh,oc=metrics(A);hs=[];cs=[];incoming=[]
  for draw in range(a.draws):
   B=np.zeros_like(A)
   for i,total in enumerate(row):
    if total:B[i]=rng.multinomial(int(total),prob[i])
   assert np.array_equal(B.sum(1),row) and not np.diag(B).any() and (B>=0).all()
   h,c=metrics(B);hs.append(h);cs.append(c);incoming.append(B.sum(0))
  hs=np.array(hs);cs=np.array(cs);allh.append(hs);allc.append(cs)
  rows.append(dict(team=file.parent.name,season=file.parent.parent.name,observed_entropy=oh,observed_ec=oc,null_entropy_mean=hs.mean(),null_entropy_sd=hs.std(ddof=1),entropy_excess=oh-hs.mean(),entropy_z=(oh-hs.mean())/hs.std(ddof=1),null_ec_mean=cs.mean(),ec_excess=oc-cs.mean(),null_entropy_q025=np.quantile(hs,.025),null_entropy_q975=np.quantile(hs,.975),conditional_lower_tail_p=(1+np.count_nonzero(hs<=oh))/(a.draws+1),mean_incoming_absolute_error=float(np.abs(np.mean(incoming,axis=0)-A.sum(0)).mean())))
  sources.append(dict(path=str(file),sha256=hashlib.sha256(file.read_bytes()).hexdigest()));print(f'{k}/180',flush=True)
 H=np.array(allh);C=np.array(allc);d=pd.DataFrame(rows);d.to_csv(out/'team_null_metrics.csv',index=False);np.savez_compressed(out/'null_draws.npz',entropy=H,ec=C)
 nullr=np.array([np.corrcoef(H[:,j],C[:,j])[0,1] for j in range(a.draws)]);observed=float(np.corrcoef(d.observed_entropy,d.observed_ec)[0,1]);fits=[]
 for x,y in [('entropy_excess','observed_ec'),('entropy_excess','ec_excess')]:
  fit,*_=regress(d,y,x=x);fits.append(fit)
 pd.DataFrame(fits).to_csv(out/'excess_regressions.csv',index=False);pd.DataFrame(dict(draw=range(a.draws),pooled_null_r=nullr)).to_csv(out/'null_correlation_draws.csv',index=False)
 summary=dict(draws=a.draws,seed=20261116,observed_pooled_r=observed,null_r_mean=float(nullr.mean()),null_r_q025=float(np.quantile(nullr,.025)),null_r_q975=float(np.quantile(nullr,.975)),observed_lower_than_null_025=int((d.observed_entropy<d.null_entropy_q025).sum()),median_entropy_excess=float(d.entropy_excess.median()),method='Independent per-row multinomial draws. Outgoing counts exact; incoming counts matched in expectation by zero-diagonal iterative proportional fitting. Plug-in conditional null, not external validation.')
 (out/'summary.json').write_text(json.dumps(summary,indent=2));pd.DataFrame(sources).to_csv(out/'source_manifest.csv',index=False)
 fig,axes=plt.subplots(1,2,figsize=(13,5));axes[0].scatter(d.null_entropy_mean,d.observed_entropy,s=18);limits=[min(d.observed_entropy.min(),d.null_entropy_mean.min()),max(d.observed_entropy.max(),d.null_entropy_mean.max())];axes[0].plot(limits,limits,c='grey');axes[0].set_xlabel('Mean randomized conditional entropy');axes[0].set_ylabel('Observed conditional entropy');axes[1].hist(nullr,bins=25);axes[1].axvline(observed,c='red',label=f'Observed r={observed:.3f}');axes[1].set_xlabel('Entropy–EC r across randomized team-seasons');axes[1].legend();fig.suptitle('Independent volume-based passing null | 500 draws/team');fig.tight_layout();fig.savefig(out/'null_model.png',dpi=150);fig.savefig(out/'null_model.pdf');plt.close(fig)
 (out/'README.md').write_text('''500 independent multinomial networks per team-season, seed 20261116. Exact observed row totals (outgoing pass counts), no self-passes; incoming totals match observed counts in expectation, not in each draw. Expected counts fit maximum-entropy mixing subject to both margins and a zero diagonal using iterative proportional fitting. Independent row-multinomial draws use these expected recipient probabilities. Thus observed passer–receiver preferences are removed while sender activity is held exact and receiver activity matched on average. Does not preserve lineups, minutes together, game schedules or possession sequences; shot edges not randomized. Null parameters are estimated on the same data, so interpret conditional tail scores and pooled null-r comparisons as model checks, not external/causal validation. No claim of exact incoming-margin preservation. Mean simulated incoming-count deviations exported. Raw observed-versus-null correlation is not an effect of star absence. Mean-null subtraction is a reference comparison, not a causal adjustment. Excess regressions retain franchise-clustered p values but are exploratory. A preliminary exact-margin MCMC approach was discarded due to highly correlated draws; only this independent-draw model informs the report.\n''');print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
