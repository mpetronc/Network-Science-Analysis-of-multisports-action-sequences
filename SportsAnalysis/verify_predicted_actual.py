from pathlib import Path
import argparse,json,re,hashlib
import numpy as np,pandas as pd
from scipy.stats import entropy,linregress

def H(M):
 q=M/M.sum();return entropy(q.ravel(),base=2)-entropy(q.sum(1),base=2)
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,required=True);a=p.parse_args(argv);out=a.data_dir;d=pd.read_csv(out/'prediction_actual_metrics.csv');nodes=pd.read_csv(out/'player_role_changes.csv');assert len(d)==30 and d.team.nunique()==30;assert len(list(out.glob('*/*_three_networks.png')))==30
 for z in d.itertuples():
  folder=out/z.team;frames=[pd.read_csv(folder/(name+'_counts.csv'),index_col=0) for name in ['original','deleted','actual']];P,D,A=[f.to_numpy(float) for f in frames];ids=frames[0].index.astype(int).tolist();hi=ids.index(z.star_id);expected=P.copy();expected[hi]=0;expected[:,hi]=0;assert np.array_equal(expected,D);assert A[hi].sum()+A[:,hi].sum()==0;assert all(np.isfinite(M).all() and (M>=0).all() and not np.diag(M).any() for M in [P,D,A]);N=np.count_nonzero((P+A).sum(0)+(P+A).sum(1));den=np.log2(N-1)
  assert np.allclose([H(M)/den for M in [P,D,A]],[z.entropy_original,z.entropy_deleted,z.entropy_actual]);assert np.isclose(z.entropy_prediction_error,z.actual_entropy_change-z.predicted_entropy_change);assert np.isclose(z.edge_total_variation,abs(D/D.sum()-A/A.sum()).sum()/2)
  group=nodes[nodes.team.eq(z.team)].set_index('player_id').reindex(ids)
  for name,M in zip(['original','deleted','actual'],[P,D,A]):
   share=(M.sum(0)+M.sum(1))/(2*M.sum());assert np.allclose(share,group[name+'_involvement_share']);assert np.allclose(16000*share,group[name+'_node_area']);assert np.isclose(group[name+'_node_area'].sum(),16000)
  scaled=pd.read_csv(folder/'deleted_scaled_to_actual_total.csv',index_col=0).to_numpy();assert np.allclose(scaled,D/D.sum()*A.sum())
 primary=d[d.regression_eligible];assert len(primary)==24 and primary.present_games.ge(15).all() and primary.absent_games.ge(10).all();fit=linregress(primary.predicted_entropy_change,primary.actual_entropy_change);s=json.loads((out/'change_regression.json').read_text());assert np.isclose(fit.rvalue,s['r']) and np.isclose(fit.slope,s['slope']);assert np.isfinite(d.select_dtypes('number')).all().all();assert d.edge_total_variation.between(0,1).all() and d.node_total_variation.between(0,1).all() and d.common_donor_js_bits.between(0,1).all()
 for z in pd.read_csv(out/'source_manifest.csv').itertuples():assert hashlib.sha256(Path(z.path).read_bytes()).hexdigest()==z.sha256
 assert b'30' in re.findall(rb'/Count\s+(\d+)',(out/'all_30_original_deleted_actual.pdf').read_bytes())
 result=dict(teams=30,network_pages=30,qualified_teams=24,own_graph_node_areas_verified=True,all_node_area_scales_equal_16000=True,isolates_zero_area=True,synthetic_deletion_verified=True,independent_entropy_verified=True,edge_share_distances_verified=True,regression_checked_against_scipy=True,sources_unchanged=True);(out/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
