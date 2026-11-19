"""Independent checks for separate passing roles, conserved edges and held-out scoring."""
from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd
from star_presence_entropy import bits


def main(root,baseline,out):
    root,baseline,out=map(Path,[root,baseline,out])
    results=pd.read_csv(out/'prediction_comparison.csv')
    assert len(results)==180 and results.eligible.sum()==96 and results.observed.sum()==171
    manifest=json.loads((out/'freeze_manifest.json').read_text());assert len(manifest)==180
    for z in results.itertuples():
        folder=out/z.season/z.team;src=baseline/z.season/z.team
        s=json.loads((folder/'frozen_specification.json').read_text());old=json.loads((src/'frozen_specification.json').read_text());ids=s['training_player_ids'];hub=ids.index(z.hub_id)
        P=pd.read_csv(src/'hub_present_training.csv',index_col=0).to_numpy();A20=pd.read_csv(src/'first20_all_passing.csv',index_col=0).to_numpy();E=A20-P
        mp=np.asarray(s['training_present_minutes']);ma=np.asarray(s['training_absent_minutes']);fm=np.asarray(s['forecast_minutes']);valid=(mp>0)&(fm>0);valid[hub]=False
        assert np.allclose(fm,old['forecast_minutes']) and np.allclose(s['minute_weights'],old['replacement_weights'])
        roles=[]
        for key,axis in [('sender',1),('receiver',0)]:
            counts=P.sum(axis);prior=counts[valid].sum()/mp[valid].sum();rate=(counts+48*prior)/(mp+48);rate[~valid]=0
            base=fm*rate*valid
            if base.sum()==0:base=fm*valid
            base/=base.sum();ar=(E.sum(axis)+48*rate)/(ma+48);gain=fm*np.clip(ar-rate,0,None)*valid;alpha=s['first20_hub_absent_games']/(s['first20_hub_absent_games']+5) if s['first20_hub_absent_games'] and gain.sum()>0 else 0
            expected=(1-alpha)*base+alpha*(gain/gain.sum() if gain.sum()>0 else base)
            assert np.allclose(expected,s[key]['weights']) and expected[hub]==0 and not expected[~valid].any();roles.append(expected)
        # Independently reconstruct each ordered source/recipient edge.
        M=P.copy();M[hub]=0;M[:,hub]=0
        for i in range(len(ids)):
            for j in range(len(ids)):
                if i==hub or j==hub or i==j:continue
                w=roles[0].copy();w[j]=0
                if w.sum()==0:w=fm*valid;w[j]=0
                v=roles[1].copy();v[i]=0
                if v.sum()==0:v=fm*valid;v[i]=0
                M[i,j]+=P[hub,j]*w[i]/w.sum()+P[i,hub]*v[j]/v.sum()
        f=folder/'frozen_passing_role_prediction.csv';saved=pd.read_csv(f,index_col=0).to_numpy();assert np.allclose(M,saved)
        assert hashlib.sha256(f.read_bytes()).hexdigest()==s['prediction_sha256']==next(r['sha256'] for r in manifest if r['season']==z.season and r['team']==z.team)
        assert np.isclose(M.sum(),P.sum()) and not np.diag(M).any() and not M[hub].any() and not M[:,hub].any()
        eval_frame=pd.read_csv(folder/'predicted_evaluation.csv',index_col=0);A=pd.read_csv(src/'actual_evaluation.csv',index_col=0).to_numpy();B=eval_frame.to_numpy();assert np.isclose(bits(B)/z.denominator,z.entropy_new)
        if A.sum()>0:
            assert np.isclose(.5*np.abs(B/B.sum()-A/A.sum()).sum(),z.edge_error_new)
            assert np.isclose(abs(bits(B)-bits(A))/z.denominator,z.entropy_error_new)
        # No future arrivals acquire forecast edges.
        future=[i for i,pid in enumerate(eval_frame.index.astype(int)) if pid not in ids]
        assert not B[future,:].any() and not B[:,future].any()
    result=dict(passed=True,forecasts=180,observed_comparisons=171,primary_benchmark=96,
                checks='Unchanged training-only minutes; independently recomputed role weights, every predicted edge, conserved mass, zero self/hub routes, frozen hashes, independent entropy and held-out errors; future roster arrivals have zero forecast weight')
    (out/'validation.json').write_text(json.dumps(result,indent=2));print(result)

if __name__=='__main__':
    import sys
    main(*sys.argv[1:])
