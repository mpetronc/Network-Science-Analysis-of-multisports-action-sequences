"""Immediately-previous-season, target-franchise-excluded role prediction benchmark."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FEATURES=['log_sender_share','log_receiver_share','present_sender_share','present_receiver_share','forecast_minutes_fraction','extra_minutes_fraction','position_G','position_F','position_C','unknown_position','position_match','log_active_players','hub_involvement']
RIDGE=10.0
MAX_FACTOR=4.0


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fit(X,Y,w):
    wn=w/w.sum();mu=wn@X;scale=np.maximum(np.sqrt(wn@((X-mu)**2)),.01);Z=np.column_stack([np.ones(len(X)),(X-mu)/scale]);penalty=np.eye(Z.shape[1])*RIDGE;penalty[0,0]=0
    coef=np.linalg.solve(Z.T@(w[:,None]*Z)+penalty,Z.T@(w[:,None]*Y));return dict(mu=mu,scale=scale,coef=coef)


def predict(M,X,model,active):
    if model is None:return M.copy(),np.zeros((len(M),2))
    Z=np.column_stack([np.ones(len(X)),(X-model['mu'])/model['scale']]);logs=np.clip(Z@model['coef'],-np.log(MAX_FACTOR),np.log(MAX_FACTOR));full=np.zeros((len(M),2));full[active]=logs
    B=M*np.exp(full[:,0,None]+full[None,:,1]);B*=M.sum()/B.sum();return B,full


def entropy(M,den):
    if M.sum()<=0:return np.nan
    p=np.divide(M,M.sum(1)[:,None],out=np.zeros_like(M),where=M.sum(1)[:,None]>0);nz=M>0
    return float(-(M[nz]/M.sum()*np.log2(p[nz])).sum()/den)


def tv(M,A):return float(np.abs(M/M.sum()-A/A.sum()).sum()/2) if M.sum()>0 and A.sum()>0 else np.nan


def frame(path):
    f=pd.read_csv(path,index_col=0);f.index=f.index.astype(int);f.columns=f.columns.astype(int);return f


def align(M,source_ids,ids):return pd.DataFrame(M,index=source_ids,columns=source_ids).reindex(index=ids,columns=ids,fill_value=0).to_numpy(float,copy=True)


def tests():
    M=np.array([[0.,2,3],[4,0,2],[1,3,0]]);X=np.arange(9).reshape(3,3)/10;active=np.arange(3);model=dict(mu=np.zeros(3),scale=np.ones(3),coef=np.array([[0.,0.],[.1,-.1],[.2,.1],[-.1,.1]]));B,_=predict(M,X,model,active)
    assert np.isclose(B.sum(),M.sum()) and not np.diag(B).any() and np.all(B>=0)
    order=np.array([2,0,1]);C,_=predict(M[np.ix_(order,order)],X[order],model,active);assert np.allclose(C,B[np.ix_(order,order)])
    assert np.array_equal(predict(M,X,None,active)[0],M)
    Y=np.column_stack([X[:,0],X[:,1]]);f=fit(X,Y,np.ones(3));assert np.isfinite(f['coef']).all()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'nba_passing_output');p.add_argument('--destination',type=Path);p.add_argument('--self-check',action='store_true');a=p.parse_args(argv);tests()
    if a.self_check:print('Role correction, permutation and fallback checks passed');return 0
    root=a.output_dir;base=root/'all_season_hub_redistribution';current=root/'split_role_hub_redistribution';out=a.destination or root/'previous_season_history';out.mkdir(parents=True,exist_ok=True)
    meta=pd.read_csv(base/'all_team_season_entropy_and_ec.csv');assert len(meta)==180;ledger=pd.read_csv(current/'minutes_and_passing_role_weights.csv').set_index(['season','team','player_id']);specs=sorted(base.glob('*/*/frozen_specification.json'));training={};hashes={}
    # Build target features only from first20 counts and frozen role forecasts.
    for path in specs:
        season,team=path.relative_to(base).parts[:2];spec=json.loads(path.read_text());sel=spec['selection'];ids=sel['training_player_ids'];P=frame(path.parent/'hub_present_training.csv');F=frame(path.parent/'first20_all_passing.csv');M=frame(current/season/team/'frozen_passing_role_prediction.csv');assert P.index.tolist()==F.index.tolist()==M.index.tolist()==ids
        P=P.to_numpy(float);F=F.to_numpy(float);M=M.to_numpy(float);hub=ids.index(sel['hub_id']);assert M[hub].sum()+M[:,hub].sum()==0
        active=np.flatnonzero(M.sum(1)+M.sum(0)>0);n=len(active);ps=M.sum(1)/M.sum();pr=M.sum(0)/M.sum();os=P.sum(1)/P.sum();orr=P.sum(0)/P.sum();hubshare=(F.sum(1)[hub]+F.sum(0)[hub])/F.sum();X=[]
        for i in active:
            r=ledger.loc[(season,team,ids[i])];position=str(r.prior_position);X.append([np.log(max(ps[i]*n,1e-5)),np.log(max(pr[i]*n,1e-5)),os[i]*n,orr[i]*n,r.forecast_minutes/48,r.extra_minutes/48,float('G' in position),float('F' in position),float('C' in position),float(position=='Unknown'),r.position_match,np.log(n),hubshare])
        source=[path,path.parent/'hub_present_training.csv',path.parent/'first20_all_passing.csv',current/season/team/'frozen_passing_role_prediction.csv'];hashes.update({str(f):sha(f) for f in source})
        training[(season,team)]=dict(ids=ids,hub=hub,hub_id=sel['hub_id'],hub_name=sel['hub_name'],cutoff=sel['first20_last_date'],P=P,M=M,active=active,X=np.array(X),sender=ps,receiver=pr,n=n)
    outcomes={}
    def historical_outcome(key):
        if key not in outcomes:
            f=frame(base/key[0]/key[1]/'actual_evaluation.csv');outcomes[key]=(f.index.tolist(),f.to_numpy(float));hashes[str(base/key[0]/key[1]/'actual_evaluation.csv')]=sha(base/key[0]/key[1]/'actual_evaluation.csv')
        return outcomes[key]
    frozen=[];coefrows=[];node_rows=[];seasons=sorted(meta.season.unique())
    # For every target, historical outcomes are strictly from completed earlier seasons.
    for season in seasons:
        for team in sorted(meta.team.unique()):
            target=training[(season,team)];year=int(season[:4]);required_previous=f'{year-1}-{year%100:02d}';eligible=meta[meta.primary_eligible & meta.season.eq(required_previous) & meta.team.ne(team)];keys=list(zip(eligible.season,eligible.team));assert all(s==required_previous and t!=team for s,t in keys);Xs=[];Ys=[];Ws=[]
            for key in keys:
                h=training[key];oid,A=historical_outcome(key);A=align(A,oid,h['ids']);actual_total=outcomes[key][1].sum();active=h['active'];eps=.05/h['n'];true_send=A.sum(1)[active]/actual_total;true_receive=A.sum(0)[active]/actual_total
                Y=np.column_stack([np.log((true_send+eps)/(h['sender'][active]+eps)),np.log((true_receive+eps)/(h['receiver'][active]+eps))]);Xs.append(h['X']);Ys.append(Y);Ws.append(np.full(h['n'],1/h['n']))
            model=fit(np.vstack(Xs),np.vstack(Ys),np.concatenate(Ws)) if keys else None
            B,factors=predict(target['M'],target['X'],model,target['active']);assert np.isclose(B.sum(),target['M'].sum()) and not np.diag(B).any() and B[target['hub']].sum()+B[:,target['hub']].sum()==0
            dest=out/season/team;dest.mkdir(parents=True,exist_ok=True);pd.DataFrame(B,index=target['ids'],columns=target['ids']).to_csv(dest/'predicted_training.csv')
            if model is not None:
                (dest/'learned_coefficients.json').write_text(json.dumps({k:v.tolist() for k,v in model.items()},indent=2))
                for name,coef in zip(['intercept']+FEATURES,model['coef']):coefrows.append(dict(season=season,team=team,feature=name,sender_coefficient=coef[0],receiver_coefficient=coef[1]))
            for i,pid in enumerate(target['ids']):node_rows.append(dict(season=season,team=team,player_id=pid,sender_multiplier=float(np.exp(factors[i,0])),receiver_multiplier=float(np.exp(factors[i,1])),original_sender_share=target['sender'][i],predicted_sender_share=B.sum(1)[i]/B.sum(),original_receiver_share=target['receiver'][i],predicted_receiver_share=B.sum(0)[i]/B.sum()))
            # Separate, intentionally simple same-franchise historical-copy baseline.
            previous=seasons[seasons.index(season)-1] if seasons.index(season)>0 else None;copy_available=False;retained=np.nan;copy_reason='No previous saved season';copy_sha=None
            if previous:
                prev=meta[meta.season.eq(previous)&meta.team.eq(team)].iloc[0]
                if int(prev.hub_id)!=target['hub_id']:copy_reason='Previous selected hub differs: no comparable same-hub absence network'
                elif not prev.primary_eligible:copy_reason='Previous same-hub comparison does not meet exposure criteria'
                else:
                    pid,H=historical_outcome((previous,team));C=align(H,pid,target['ids']);allowed=target['M'].sum(0)+target['M'].sum(1)>0;C[~allowed]=0;C[:,~allowed]=0;retained=float(C.sum()/H.sum())
                    if C.sum()>0:
                        C*=target['M'].sum()/C.sum();copy_available=True;copy_reason='Previous same-hub absent passes copied on intersection with active first20 forecast roster; renormalized';pd.DataFrame(C,index=target['ids'],columns=target['ids']).to_csv(dest/'historical_copy_training.csv');copy_sha=sha(dest/'historical_copy_training.csv')
                    else:copy_reason='No retained passing edges after roster intersection'
            rec=dict(season=season,team=team,hub_id=target['hub_id'],hub_name=target['hub_name'],first20_cutoff=target['cutoff'],required_previous_season=required_previous,history_policy='Immediately preceding season only',historical_team_seasons=len(keys),historical_training_keys=[list(k) for k in keys],target_franchise_excluded=True,all_training_seasons_strictly_earlier=True,all_training_seasons_immediately_previous=True,identity_features_used=False,feature_names=FEATURES,ridge_penalty=RIDGE,multiplier_cap=MAX_FACTOR,label_pseudomass=.05,prediction_sha256=sha(dest/'predicted_training.csv'),copy_available=copy_available,copy_reason=copy_reason,copy_retained_previous_pass_share=retained,copy_sha256=copy_sha,freeze_stage='Prediction frozen before reading target-season absence outcomes')
            (dest/'frozen_specification.json').write_text(json.dumps(rec,indent=2));frozen.append(rec)
        print('Frozen target forecasts',season,flush=True)
    (out/'freeze_manifest.json').write_text(json.dumps(frozen,indent=2));pd.DataFrame(coefrows).to_csv(out/'learned_coefficients.csv',index=False);pd.DataFrame(node_rows).to_csv(out/'predicted_player_role_changes.csv',index=False)
    # All target forecasts are frozen. Target outcomes now used only for evaluation.
    rows=[]
    for r in frozen:
        season,team=r['season'],r['team'];t=training[(season,team)];dest=out/season/team;z=meta[meta.season.eq(season)&meta.team.eq(team)].iloc[0];A=frame(base/season/team/'actual_evaluation.csv');ids=A.index.tolist();A=A.to_numpy(float);assert sha(dest/'predicted_training.csv')==r['prediction_sha256'];B=frame(dest/'predicted_training.csv').reindex(index=ids,columns=ids,fill_value=0).to_numpy(float);M=align(t['M'],t['ids'],ids);P=align(t['P'],t['ids'],ids);D=P.copy();h=ids.index(t['hub_id']);D[h]=0;D[:,h]=0;pd.DataFrame(B,index=ids,columns=ids).to_csv(dest/'predicted_evaluation.csv');den=z.normalization_log2
        row=dict(season=season,team=team,hub_name=t['hub_name'],eligible=bool(z.later_hub_absent_games>=10 and z.actual_passes>0),observed=bool(A.sum()>0),historical_team_seasons=r['historical_team_seasons'],has_history=r['historical_team_seasons']>0,copy_available=r['copy_available'],copy_reason=r['copy_reason'],copy_retained_previous_pass_share=r['copy_retained_previous_pass_share'],entropy_actual=entropy(A,den))
        models={'current':M,'learned':B,'deletion':D}
        if r['copy_available']:
            assert sha(dest/'historical_copy_training.csv')==r['copy_sha256'];models['copy']=frame(dest/'historical_copy_training.csv').reindex(index=ids,columns=ids,fill_value=0).to_numpy(float)
        for name in ['current','learned','deletion','copy']:
            C=models.get(name);row['edge_'+name]=tv(C,A) if C is not None else np.nan;row['entropy_error_'+name]=abs(entropy(C,den)-row['entropy_actual']) if C is not None else np.nan
        row['edge_change']=row['edge_learned']-row['edge_current'];row['entropy_error_change']=row['entropy_error_learned']-row['entropy_error_current'];rows.append(row)
    d=pd.DataFrame(rows);d.to_csv(out/'team_season_comparison.csv',index=False);summaries=[]
    samples={'all_primary':d[d.eligible],'primary_with_history':d[d.eligible & d.has_history],'latest_season_primary':d[d.eligible & d.season.eq(seasons[-1])],'comparable_copy_subset':d[d.eligible & d.has_history & d.copy_available]}
    for sample,q in samples.items():
        for model in ['current','learned','deletion','copy']:
            g=q[q['edge_'+model].notna()];delta=g['edge_'+model]-g.edge_current
            summaries.append(dict(sample=sample,model=model,n=len(g),mean_connection_error=g['edge_'+model].mean(),median_connection_error=g['edge_'+model].median(),median_entropy_error=g['entropy_error_'+model].median(),mean_entropy_error=g['entropy_error_'+model].mean(),improved_vs_current=int((delta<-1e-8).sum()),worsened_vs_current=int((delta>1e-8).sum()),unchanged_vs_current=int((delta.abs()<=1e-8).sum()),median_paired_connection_change=delta.median()))
    summary=pd.DataFrame(summaries);summary.to_csv(out/'performance_summary.csv',index=False)
    pd.DataFrame([dict(season=s,n=len(g),mean_current=g.edge_current.mean(),mean_learned=g.edge_learned.mean(),median_current=g.edge_current.median(),median_learned=g.edge_learned.median(),improved=int((g.edge_change<-1e-8).sum()),worsened=int((g.edge_change>1e-8).sum())) for s,g in d[d.eligible].groupby('season')]).to_csv(out/'season_summary.csv',index=False)
    q=samples['primary_with_history'];fig,axes=plt.subplots(1,2,figsize=(14,6))
    for ax,metric,title in [(axes[0],'edge','Connection error'),(axes[1],'entropy_error','Absolute normalized entropy error')]:
        for latest,color,label in [(False,'#879baa','Earlier target seasons'),(True,'#197c68',seasons[-1])]:
            g=q[q.season.eq(seasons[-1])==latest];ax.scatter(g[metric+'_current'],g[metric+'_learned'],c=color,label=label,alpha=.8)
        limit=max(q[metric+'_current'].max(),q[metric+'_learned'].max())*1.06;ax.plot([0,limit],[0,limit],c='firebrick',label='Equal error');ax.set(xlim=(0,limit),ylim=(0,limit),xlabel='Current first20 separate-role model',ylabel='Previous-season-only role model',title=title);ax.legend(fontsize=8)
    fig.suptitle(f'{len(q)} eligible cases | training uses previous season only | below diagonal = improvement');fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(out/'current_vs_cross_franchise.png',dpi=155);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(14,6));labels={'current':'Current','learned':'Previous-season learned','deletion':'Deletion','copy':'Previous same-hub copy'}
    for ax,sample,title in [(axes[0],'primary_with_history','All eligible cases with historical training'),(axes[1],'comparable_copy_subset','Same observations with comparable historical copy')]:
        g=summary[summary['sample'].eq(sample)&summary.model.isin(['current','learned','deletion'] if sample=='primary_with_history' else ['current','learned','copy'])];ax.bar(np.arange(len(g)),g.mean_connection_error,color=['#346e9d','#197c68','#bb8b3a'][:len(g)]);ax.set_xticks(np.arange(len(g)),[labels[m] for m in g.model],rotation=12);ax.set_title(title+f'\nn={int(g.n.iloc[0])}' if len(g) else title);ax.set_ylabel('Mean connection error (lower is better)')
    fig.tight_layout();fig.savefig(out/'baseline_comparisons.png',dpi=155);plt.close(fig)
    for path,h in hashes.items():assert sha(path)==h
    assert d.eligible.sum()==100 and d.observed.sum()==171
    validation=dict(passed=True,team_seasons=180,primary_n=100,primary_with_history=int((d.eligible & d.has_history).sum()),comparable_copy_primary=int((d.eligible & d.copy_available).sum()),all_target_franchises_excluded_from_learned_training=True,all_training_seasons_strictly_earlier=True,all_training_seasons_immediately_previous=True,player_names_ids_not_features=True,source_files_unchanged=True,source_sha256=sha(__file__),checks='Synthetic correction/fallback/permutation checks; all-target chronology/franchise assertions; hub/self-pass removal and conserved pass mass; frozen source hashes')
    (out/'validation.json').write_text(json.dumps(validation,indent=2))
    method='''# Previous-season-only cross-franchise role-transfer experiment

Goal: learn general replacement patterns from earlier seasons without reusing the target franchise's historical networks. For each target team-season, training examples are primary-eligible hub-absence comparisons from the immediately preceding season ONLY, excluding the target franchise. The calendar-adjacent season is required; older available seasons are never substituted. For 2025-26 the only historical training season is 2024-25. Features are first20 passing shares, forecast and extra minutes, broad prior positions, position match, active roster size and hub involvement. No player/team IDs or names are model features. Players can still recur after changing franchises; this is not leave-player-out validation. Target roster IDs only align matrices.

Two weighted ridge regressions predict log multiplicative corrections to each nonhub player's outgoing and incoming passing share relative to the current separate-role forecast. Historical observed shares include all later passes in the denominator, while fitted player rows are first20 current-forecast active candidates. A 0.05 uniform pseudomass across candidates stabilizes zero shares. Each historical team-season has total training weight one. Features are standardized within the allowed training set; ridge=10 and multiplier cap=4 are fixed before this benchmark and not selected against target outcomes. Current edge weights are multiplied by the learned sender and receiver factors and renormalized to original pass mass. Hub/self-passes stay zero. Existing positive connections may strengthen/weaken, but previously zero edges and later-arriving players cannot be created. Forecast minutes themselves are unchanged. These are empirical role multipliers, not a causal model or a model of game-specific availability.

Every target is frozen before its own season's later absence matrices are read for evaluation. The immediately previous season's later games provide historical labels for the next target season. No older season's absence outcomes enter the fit. Historical training features retain their already frozen previous-roster position metadata; this window restriction concerns the historical network examples, not a recomputation of those past feature records. No outcome from the target season enters its prediction. 2020-21 has no earlier saved season and falls back exactly to current. 2025-26 is not used to choose settings, but all seasons have been examined in earlier research, so this is retrospective temporal/franchise-separated testing rather than fresh independent validation.

The separate historical-copy baseline deliberately uses the same franchise's immediately preceding season, only when its selected hub is the same player and its absence comparison meets primary exposure criteria. Map that earlier absence network to the target's active first20 forecast roster, discard out-of-roster edges, renormalize retained edges, and assign zero to unseen connections. Retained previous pass mass is reported. Unavailable copies are NaN, never silently replaced with current, and comparisons against copying use identical available observations. This is a baseline only; the learned model does not use these same-franchise outcomes.

Current separate-role and simple-deletion baselines are assessed on the same samples. The five-game whole-network blend remains a separate saved experiment and is not used here. Observed centrality regressions are unchanged. Entropy uses the inherited retrospective union denominator; later roster information affects its evaluation reference, not forecast edges. Missing target comparisons stay undefined. Evaluation requires ten or more tracked later hub-absent games and a nonempty observed network, without an early-present count threshold. Historical training still uses the frozen fifteen-early-present and ten-later-absent eligibility. No statistical significance or external-validation claims are made.

'''+summary.to_csv(index=False)
    (out/'METHOD_AND_RESULTS.md').write_text(method);print(summary.to_string(index=False),flush=True);print(validation,flush=True);return 0

if __name__=='__main__':raise SystemExit(main())
