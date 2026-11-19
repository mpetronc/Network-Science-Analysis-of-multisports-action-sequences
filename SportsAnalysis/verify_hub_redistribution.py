"""Independent invariants and numerical checks for the completed study."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
from scipy.stats import linregress
import subprocess,re
def pages(file):
 return int(re.search(r"Pages:\s+(\d+)",subprocess.check_output(["pdfinfo",str(file)],text=True)).group(1))

def entropy(B):
 total=B.sum();r=B.sum(1);p=B/total
 nz=B>0
 return float(-(p[nz]*np.log2((B/np.maximum(r[:,None],1))[nz])).sum())

def main(out):
 out=Path(out);d=pd.read_csv(out/'team_entropy_and_ec.csv');assert len(d)==30
 for z in d.itertuples():
  folder=out/z.team;s=json.loads((folder/'frozen_specification.json').read_text());ids=s['selection']['training_player_ids'];h=ids.index(z.hub_id)
  B=pd.read_csv(folder/'first20_all_passing.csv',index_col=0).to_numpy();strength=B.sum(0)+B.sum(1);assert strength[h]==strength.max()
  active=strength>0;C=(B+B.T)[np.ix_(active,active)];v=np.ones(len(C));v/=np.linalg.norm(v)
  for _ in range(1000):
   nv=C@v;nv/=np.linalg.norm(nv)
   if np.linalg.norm(nv-v)<1e-12:break
   v=nv
  shares=nv/nv.sum();index=(shares@shares-1/len(C))/(1-1/len(C));assert np.isclose(index,z.ec_index,atol=1e-10)
  P=pd.read_csv(folder/'hub_present_training.csv',index_col=0).to_numpy();M=pd.read_csv(folder/'frozen_redistribution.csv',index_col=0).to_numpy();assert np.isclose(M.sum(),P.sum());assert np.allclose(M[h],0) and np.allclose(M[:,h],0) and np.allclose(np.diag(M),0)
  forecast=np.array(s['forecast_minutes']);assert np.isclose(forecast.sum(),240) and forecast.max()<=48+1e-8
  assert np.isclose(sum(s['replacement_weights']),1)
  R=P.copy();R[h]=0;R[:,h]=0;w=np.array(s['replacement_weights'])
  for j in range(len(ids)):
   if j==h:continue
   v=w.copy();v[j]=0;v[h]=0
   if v.sum()==0:v=forecast.copy();v[j]=0;v[h]=0
   R[:,j]+=P[h,j]*v/v.sum()
  for i in range(len(ids)):
   if i==h:continue
   v=w.copy();v[i]=0;v[h]=0
   if v.sum()==0:v=forecast.copy();v[i]=0;v[h]=0
   R[i,:]+=P[i,h]*v/v.sum()
  assert np.allclose(R,M)
  assert hashlib.sha256((folder/'frozen_redistribution.csv').read_bytes()).hexdigest()==s['frozen_sha256']
  for name,column in [('original','entropy_original'),('predicted','entropy_predicted'),('actual','entropy_actual')]:
   A=pd.read_csv(folder/(name+'_evaluation.csv'),index_col=0).to_numpy();assert A.min()>=0 and not np.diag(A).any()
   if A.sum():assert np.isclose(entropy(A)/z.normalization_log2,getattr(z,column),atol=1e-11)
  assert z.primary_eligible==(z.first20_hub_present_games>=15 and z.later_hub_absent_games>=10 and z.actual_passes>0)
 stats=pd.read_csv(out/'ec_entropy_regressions.csv')
 mapping={'primary_equal_count':('actual_equal_count_change','primary_eligible'),'raw_training_reference':('actual_change','primary_eligible'),'late_present_equal_count':('concurrent_equal_count_change','concurrent_eligible'),'late_present_raw':('concurrent_period_change','concurrent_eligible')}
 for row in stats.itertuples():
  col,elig=mapping[row.analysis];q=d[d[elig]];r=linregress(q.ec_index,q[col]);assert np.isclose(r.slope,row.slope) and np.isclose(r.rvalue,row.r) and np.isclose(r.rvalue**2,row.r_squared) and len(q)==row.n
 for f in (out/'passing_profiles').rglob('*.json'):assert json.loads(f.read_text())['complete'],f
 assert pages(out/'all_30_hub_networks.pdf')==30
 assert pages(out/'entropy_changes_and_regressions.pdf')==7
 result={'teams_verified':30,'checks':'Hub selection, independent EC power iteration, minute constraints, edge redistribution, frozen hashes, independent outgoing entropy, regression slopes/r/R2, complete raw audits, PDF page counts','passed':True};(out/'validation.json').write_text(json.dumps(result,indent=2));print(result)
if __name__=='__main__':
 import sys
 main(sys.argv[1])
