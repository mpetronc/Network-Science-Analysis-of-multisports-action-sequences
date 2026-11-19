"""Independent first-20 EC selection, rebuilt cohort, forecast and regression checks."""
from pathlib import Path
import json
import numpy as np,pandas as pd
from scipy.stats import linregress,t


def main(root,original,base,roles):
    root,original,base,roles=map(Path,[root,original,base,roles]);data=pd.read_csv(base/'all_team_season_entropy_and_ec.csv');old=pd.read_csv(original/'all_team_season_entropy_and_ec.csv').set_index(['season','team']);changes=pd.read_csv(base.parent/'hub_selection_comparison.csv');changed={(r.season,r.team) for r in changes[changes.changed].itertuples()};assert len(data)==180 and len(changed)==12
    logs={}
    for season in sorted(data.season.unique()):
        players=pd.read_csv(root/'star_dependency_study/game_cache'/(season+'_player_base.csv.gz'),dtype={'GAME_ID':str});players.GAME_ID=players.GAME_ID.str.zfill(10);players.GAME_DATE=players.GAME_DATE.str[:10];logs[season]=players
    for z in data.itertuples():
        folder=base/z.season/z.team;Aframe=pd.read_csv(folder/'first20_all_passing.csv',index_col=0);A=Aframe.to_numpy();ids=Aframe.index.astype(int).tolist();assert np.array_equal(A,pd.read_csv(original/z.season/z.team/'first20_all_passing.csv',index_col=0).to_numpy());active=np.flatnonzero(A.sum(0)+A.sum(1)>0);S=(A+A.T)[np.ix_(active,active)];v=np.ones(len(S));v/=np.linalg.norm(v)
        for _ in range(2000):
            nv=S@v;nv/=np.linalg.norm(nv)
            if np.linalg.norm(nv-v)<1e-13:break
            v=nv
        scores=nv/nv.sum();hi=ids.index(z.hub_id);assert hi in active and np.isclose(scores[np.flatnonzero(active==hi)[0]],scores.max(),atol=1e-10);assert np.isclose(z.ec_index,old.loc[(z.season,z.team),'ec_index'])
        spec=json.loads((folder/'frozen_specification.json').read_text());assert spec['selection']['hub_id']==z.hub_id
        original_context=pd.read_csv(folder/'original_evaluation.csv',index_col=0);actual=pd.read_csv(folder/'actual_evaluation.csv',index_col=0);assert original_context.index.equals(actual.index);ix={pid:i for i,pid in enumerate(actual.index.astype(int))};assert not actual.to_numpy()[ix[z.hub_id]].any() and not actual.to_numpy()[:,ix[z.hub_id]].any()
        if (z.season,z.team) not in changed:
            assert np.array_equal(actual,pd.read_csv(original/z.season/z.team/'actual_evaluation.csv',index_col=0))
            assert np.allclose(pd.read_csv(roles/z.season/z.team/'predicted_evaluation.csv',index_col=0),pd.read_csv(root/'split_role_hub_redistribution'/z.season/z.team/'predicted_evaluation.csv',index_col=0))
        else:
            selection=json.loads((folder/'selection.json').read_text());p=logs[z.season];hub_games=set(p.loc[p.TEAM_ID.eq(z.team_id)&p.PLAYER_ID.eq(z.hub_id),'GAME_ID']);assert not hub_games.intersection(selection['later_absent_game_ids']);assert set(selection['train_present_game_ids'])<=hub_games
            exclusions=json.loads((base/z.season/'tracking_exclusions.json').read_text());excluded={(r['team'],r['kind'],r['start'],r['end']) for r in exclusions};plan=pd.read_csv(base/z.season/'network_download_plan.csv');R=np.zeros(actual.shape);scope={}
            for r in plan[plan.team.eq(z.team)&plan.kind.eq('later_absence')].to_dict('records'):
                if (r['team'],r['kind'],r['start'],r['end']) in excluded:continue
                f=base/z.season/'passing_profiles'/z.team/(r['start']+'_'+r['end'])/(str(r['player_id'])+'.csv');assert json.loads(f.with_suffix('.json').read_text())['complete'];key=(r['start'],r['end'])
                if key not in scope:scope[key]=set(p.loc[p.TEAM_ID.eq(z.team_id)&p.GAME_DATE.between(*key),'PLAYER_ID'])
                for edge in pd.read_csv(f).itertuples():
                    receiver=int(edge.PASS_TEAMMATE_PLAYER_ID)
                    if edge.PASS>0 and receiver in scope[key] and receiver in ix:R[ix[r['player_id']],ix[receiver]]+=edge.PASS
            assert np.allclose(R,actual.to_numpy())
    # Reuse the independent role-weight/edge reconstruction with this cohort's actual sample sizes.
    source=(root.parent/'verify_split_role_hubs.py').read_text();source=source.replace('results.eligible.sum()==96 and results.observed.sum()==171',f"results.eligible.sum()=={int(data.primary_eligible.sum())} and results.observed.sum()=={int(data.actual_passes.gt(0).sum())}");source=source.replace('observed_comparisons=171',f'observed_comparisons={int(data.actual_passes.gt(0).sum())}').replace('primary_benchmark=96',f'primary_benchmark={int(data.primary_eligible.sum())}');namespace={'__name__':'ec_role_verifier'};exec(compile(source,'independent_role_verifier','exec'),namespace);namespace['main'](root,base,roles)
    pooled=pd.read_csv(base/'pooled_season_adjusted_regressions.csv')
    for z in pooled.itertuples():
        col,elig=('actual_equal_count_change','primary_eligible') if z.analysis=='primary_equal_count' else ('concurrent_equal_count_change','concurrent_eligible');q=data[data[elig]];x=q.ec_index-q.groupby('season').ec_index.transform('mean');y=q[col]-q.groupby('season')[col].transform('mean');slope=np.dot(x,y)/np.dot(x,x);res=y-slope*x;scores=pd.DataFrame({'team':q.team,'score':x*res}).groupby('team').score.sum();G=len(scores);k=q.season.nunique()+1;se=np.sqrt(G/(G-1)*(len(q)-1)/(len(q)-k)*np.dot(scores,scores)/np.dot(x,x)**2);p=2*t.sf(abs(slope/se),G-1);assert np.isclose(slope,z.slope) and np.isclose(p,z.franchise_clustered_p)
    result=dict(passed=True,team_seasons=180,changed_hubs=12,unchanged_hubs=168,eligible=int(data.primary_eligible.sum()),observed=int(data.actual_passes.gt(0).sum()),checks='Independent EC power iteration; unchanged first20 matrices and EC index; rebuilt absence rows from validated profiles; unchanged-case equality; independent split-role weights/edges and pooled clustered inference');(base.parent/'validation.json').write_text(json.dumps(result,indent=2));print(result)

if __name__=='__main__':
    import sys
    main(*sys.argv[1:])
