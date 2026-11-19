"""Sampling-aware entropy analysis of actual tracked five-player lineup networks.

Main case is chosen by repeated-lineup sample size, never by entropy or fit.
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd
from scipy.special import digamma
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def H(A):
    A=np.asarray(A,float)
    if A.shape[-2:]!=(5,5) or not np.isfinite(A).all() or (A<0).any() or np.any(np.diagonal(A,axis1=-2,axis2=-1)!=0):raise ValueError('Need finite five-player counts with zero diagonal')
    total=A.sum(axis=(-2,-1));out=A.sum(axis=-1,keepdims=True);p=np.divide(A,out,out=np.zeros_like(A),where=out>0);bits=-np.sum(A*np.log2(np.maximum(p,1e-300)),axis=(-2,-1));return np.divide(bits,total,out=np.full_like(np.asarray(total,dtype=float),np.nan),where=total>0)/2


def posterior_mean_entropy(A,alpha):
    """Dirichlet posterior expected receiver entropy, fixed empirical source weights."""
    n=A.sum(1);N=A.sum();value=0.
    for i in range(5):
        a=np.delete(A[i],i)+alpha;row=(digamma(a.sum()+1)-np.sum(a/a.sum()*digamma(a+1)))/np.log(2)
        value+=(n[i]/N)*row
    return float(value/2)


def smooth_row_model(A,alpha):
    P=A+alpha;np.fill_diagonal(P,0);return P/P.sum(1,keepdims=True)


def marginal_model(A,alpha):
    incoming=A.sum(0)+alpha;P=np.tile(incoming,(5,1));np.fill_diagonal(P,0);return P/P.sum(1,keepdims=True)


def loss(A,P):return float(-np.sum(A*np.log2(np.maximum(P,1e-300)))/A.sum())


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--lineup-entropy-analysis',action='store_true');parser.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'nba_passing_output');parser.add_argument('--destination',type=Path);parser.add_argument('--draws',type=int,default=2000);args=parser.parse_args(argv)
    if args.draws<100:raise ValueError('At least 100 diagnostic draws required')
    root=args.output_dir.expanduser();base=root/'lineup_role_networks';out=args.destination or base/'entropy_analysis';out.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(20261115)
    summary=pd.read_csv(base/'lineup_summary.csv',dtype={'game_id':str});summary=summary[summary.inferred_passes.gt(0)].copy();assigned=pd.read_csv(base/'lineup_role_assignments.csv');passes=pd.read_csv(base/'pass_role_assignments.csv',dtype={'game_id':str});rows=[];matrices={};sim_pooled={};sim_within={};source=[]
    for z in summary.itertuples():
        key=(z.game_id,z.team,z.lineup);file=base/'lineups'/z.game_id/z.team/z.lineup/'player_pass_counts.csv';frame=pd.read_csv(file,index_col=0);A=frame.to_numpy(float);assert np.isclose(A.sum(),z.inferred_passes);matrices[key]=(A,frame.index.astype(int).tolist());N=A.sum();n=A.sum(1);h=float(H(A));assert np.isclose(h,z.conditional_entropy_fixed_five)
        # Independent-uniform diagnostic conditions on each observed passer count.
        draws=np.zeros((args.draws,5,5))
        for i in range(5):draws[:,i,np.arange(5)!=i]=rng.multinomial(int(n[i]),np.full(4,.25),size=args.draws)
        sim=H(draws);row=dict(game_id=z.game_id,team=z.team,lineup=z.lineup,passes=int(N),observed_sources=int(sum(n>0)),minimum_outgoing_count=int(n.min()),raw_conditional_entropy=h,uniform_reference_mean=float(sim.mean()),uniform_reference_q025=float(np.quantile(sim,.025)),uniform_reference_q975=float(np.quantile(sim,.975)),uniform_reference_bias=1-float(sim.mean()),observed_minus_uniform_reference_mean=h-float(sim.mean()),observed_source_receiver_pairs=int((A>0).sum()))
        correction=sum(max(int((A[i]>0).sum())-1,0) for i in range(5))/(2*N*np.log(2))/2;row['miller_madow_observed_support']=h+correction;row['miller_madow_outside_0_1']=not (0<=h+correction<=1)
        for alpha in [.1,.5,1.]:row[f'dirichlet_mean_alpha_{alpha:g}']=posterior_mean_entropy(A,alpha)
        rows.append(row);a=assigned[(assigned.team==z.team)&assigned.lineup.eq(z.lineup)].set_index('player_id').loc[frame.index.astype(int)];role=np.array([['PG','SG','SF','PF','C'].index(r) for r in a.inferred_role]);inv=np.argsort(role);rdraw=draws[:,inv,:][:,:,inv]
        if z.team not in sim_pooled:sim_pooled[z.team]=np.zeros_like(draws);sim_within[z.team]=np.zeros(args.draws)
        sim_pooled[z.team]+=rdraw;sim_within[z.team]+=N*sim;source.append(dict(path=str(file),sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
    audit=pd.DataFrame(rows);audit.to_csv(out/'sample_size_and_entropy_audit.csv',index=False)
    pool=pd.read_csv(base/'pooled_entropy_and_assignment_sensitivity.csv');ref=[]
    for z in pool.itertuples():
        pooled=H(sim_pooled[z.team]);within=sim_within[z.team]/z.inferred_passes;gap=pooled-within
        ref.append(dict(team=z.team,observed_pooled_entropy=z.canonical_pooled_entropy,observed_within_entropy=z.weighted_within_lineup_entropy,observed_gap=z.observed_mixing_gap,uniform_reference_pooled_mean=pooled.mean(),uniform_reference_within_mean=within.mean(),uniform_reference_gap_mean=gap.mean(),uniform_reference_gap_q025=np.quantile(gap,.025),uniform_reference_gap_q975=np.quantile(gap,.975),is_realistic_basketball_null=False,is_confidence_interval=False))
    ref=pd.DataFrame(ref);ref.to_csv(out/'pooling_and_finite_sampling_reference.csv',index=False)
    # Select the most sampled recurring five-player lineup, without inspecting outcomes/entropy.
    repeated=summary.groupby(['team','lineup']).agg(games=('game_id','nunique'),passes=('inferred_passes','sum'));repeated=repeated[repeated.games.ge(2)].sort_values(['passes','games'],ascending=False)
    if repeated.empty:raise ValueError('No recurring lineup available for a matched-lineup case study')
    team,lineup=repeated.index[0];case=summary[(summary.team==team)&summary.lineup.eq(lineup)].sort_values('game_id');caseA=np.stack([matrices[(z.game_id,team,lineup)][0] for z in case.itertuples()]);ids=matrices[(case.iloc[0].game_id,team,lineup)][1];names=assigned[(assigned.team==team)&assigned.lineup.eq(lineup)].set_index('player_id').loc[ids]['name'].tolist();min_total=int(caseA.sum((1,2)).min());row_min=caseA.sum(2).min(0).astype(int);details=[];rarefactions=[];robust_matrices=[]
    if row_min.sum()==0:raise ValueError('No common passer support in recurring case')
    for z,A in zip(case.itertuples(),caseA):
        total=rng.multivariate_hypergeometric(A.reshape(-1).astype(int),min_total,size=args.draws).reshape(args.draws,5,5);matched=np.zeros_like(total)
        for i in range(5):matched[:,i,:]=rng.multivariate_hypergeometric(A[i].astype(int),int(row_min[i]),size=args.draws)
        ht=H(total);hr=H(matched);g=passes[(passes.game_id==z.game_id)&(passes.team==team)&passes.lineup.astype(str).eq(lineup)];robust=g[g.robust_across_all_settings.eq(True)];R=np.zeros((5,5))
        for p in robust.itertuples():R[ids.index(p.source),ids.index(p.target)]+=1
        robust_matrices.append(R)
        same_count=rng.multivariate_hypergeometric(A.reshape(-1).astype(int),int(R.sum()),size=args.draws).reshape(args.draws,5,5);same_h=H(same_count)
        row=dict(game_id=z.game_id,team=team,lineup=lineup,full_at_robust_count_mean=float(np.nanmean(same_h)),full_at_robust_count_q025=float(np.nanquantile(same_h,.025)),full_at_robust_count_q975=float(np.nanquantile(same_h,.975)),passes=int(A.sum()),raw_entropy=float(H(A)),matched_total_passes=min_total,matched_source_passes=int(row_min.sum()),equal_total_mean=float(ht.mean()),equal_total_q025=float(np.quantile(ht,.025)),equal_total_q975=float(np.quantile(ht,.975)),equal_passer_mean=float(hr.mean()),equal_passer_q025=float(np.quantile(hr,.025)),equal_passer_q975=float(np.quantile(hr,.975)),robust_setting_passes=int(R.sum()),robust_setting_entropy=float(H(R)),dirichlet_alpha_point1=posterior_mean_entropy(A,.1),dirichlet_alpha_point5=posterior_mean_entropy(A,.5),dirichlet_alpha1=posterior_mean_entropy(A,1.),subsampling_range_is_confidence=False)
        details.append(row)
        for j in range(args.draws):rarefactions.append(dict(game_id=z.game_id,draw=j,equal_total_entropy=ht[j],equal_passer_entropy=hr[j]))
    details=pd.DataFrame(details);details.to_csv(out/'recurring_lineup_case.csv',index=False);pd.DataFrame(rarefactions).to_csv(out/'recurring_lineup_subsampling_draws.csv',index=False);pd.DataFrame(dict(player_id=ids,name=names,common_outgoing_sample=row_min)).to_csv(out/'recurring_lineup_players.csv',index=False)
    # Genuine held-game scoring; smoothing/choices fixed, no tuning on held-out counts.
    scores=[];predictions=[]
    for held,(z,test) in enumerate(zip(case.itertuples(),caseA)):
        train=caseA[np.arange(len(caseA))!=held].sum(0)
        for alpha in [.1,.5,1.]:
            models={'uniform_four_receivers':(np.ones((5,5))-np.eye(5))/4,'receiver_popularity':marginal_model(train,alpha),'passer_specific_receivers':smooth_row_model(train,alpha)}
            for name,P in models.items():
                assert np.allclose(P.sum(1),1) and np.trace(P)==0
                scores.append(dict(held_game=z.game_id,model=name,alpha=alpha,test_passes=int(test.sum()),training_passes=int(train.sum()),held_logloss_bits_per_pass=loss(test,P),improvement_over_uniform_bits=2-loss(test,P)))
                if alpha==.5:
                    for i in range(5):
                        for j in range(5):
                            if i!=j:predictions.append(dict(held_game=z.game_id,model=name,source_id=ids[i],target_id=ids[j],probability=P[i,j],held_observed_count=int(test[i,j])))
    scores=pd.DataFrame(scores);scores.to_csv(out/'leave_one_game_out_prediction.csv',index=False);pd.DataFrame(predictions).to_csv(out/'held_game_probability_predictions.csv',index=False);aggregate=scores.groupby(['model','alpha']).apply(lambda g:np.average(g.held_logloss_bits_per_pass,weights=g.test_passes),include_groups=False).rename('pooled_held_logloss_bits').reset_index();aggregate['improvement_over_uniform_bits']=2-aggregate.pooled_held_logloss_bits;aggregate.to_csv(out/'pooled_held_game_prediction.csv',index=False)
    robust_scores=[]
    for held,z in enumerate(case.itertuples()):
        test=robust_matrices[held]
        if test.sum()==0:continue
        train=sum((A for i,A in enumerate(robust_matrices) if i!=held),np.zeros((5,5)))
        for alpha in [.1,.5,1.]:
            for name,P in [('uniform_four_receivers',(np.ones((5,5))-np.eye(5))/4),('receiver_popularity',marginal_model(train,alpha)),('passer_specific_receivers',smooth_row_model(train,alpha))]:
                robust_scores.append(dict(held_game=z.game_id,sample='threshold_agreement_subset',alpha=alpha,model=name,test_passes=int(test.sum()),logloss_bits=loss(test,P)))
    pd.DataFrame(robust_scores).to_csv(out/'threshold_subset_prediction.csv',index=False)
    # Factual context for the three test games from the original metadata.
    labels={}
    for z in case.itertuples():
        f=root/'sportvu_passing_sequences/frame_cache'/(z.game_id+'_metadata.json');m=json.loads(f.read_text());opp=[v['abbreviation'] for v in m['teams'].values() if v['abbreviation']!=team][0];labels[z.game_id]=m['game_date']+' vs '+opp
    with PdfPages(out/'lineup_entropy_analysis.pdf') as pdf:
        fig,axes=plt.subplots(1,2,figsize=(15,6));axes[0].hist(audit.passes,bins=np.arange(0,65,5),color='#347d96',edgecolor='white');axes[0].set_xlabel('Retained inferred passes per lineup-game group');axes[0].set_ylabel('Groups');axes[0].set_title(f'79 passing-positive groups | median {audit.passes.median():g} passes')
        axes[1].scatter(audit.passes,audit.raw_conditional_entropy,label='Observed empirical entropy',color='#347d96',s=32);axes[1].scatter(audit.passes,audit.uniform_reference_mean,label='Finite-sample uniform reference mean',marker='x',color='#c27b3a');axes[1].set_xlabel('Retained inferred passes');axes[1].set_ylabel('Conditional entropy / log2(4)');axes[1].set_ylim(0,1.03);axes[1].legend(fontsize=8);axes[1].set_title('Small samples underestimate even a uniform process');fig.text(.5,.015,'Reference independently samples four equally likely receivers, holding observed passer totals fixed. It is a sampling diagnostic, not a realistic tactical null or significance test.',ha='center',fontsize=9);fig.tight_layout(rect=[0,.055,1,1]);pdf.savefig(fig);fig.savefig(out/'sample_size_diagnostic.png',dpi=150);plt.close(fig)
        fig,ax=plt.subplots(figsize=(12,7));xx=np.arange(len(ref));width=.19;ax.bar(xx-1.5*width,ref.observed_within_entropy,width,label='Observed within-lineup',color='#347d96');ax.bar(xx-.5*width,ref.observed_pooled_entropy,width,label='Observed pooled roles',color='#c27b3a');ax.bar(xx+.5*width,ref.uniform_reference_within_mean,width,label='Uniform reference: within',color='#92b5c0');ax.bar(xx+1.5*width,ref.uniform_reference_pooled_mean,width,label='Uniform reference: pooled',color='#e0c3a2');ax.set_xticks(xx,ref.team);ax.set_ylim(0,1.05);ax.set_ylabel('Conditional entropy / log2(4)');ax.set_title('Pooling gap includes finite-sample effects, even under a uniform process');ax.legend(fontsize=9);fig.text(.5,.015,'An observed gap is not pure evidence of tactical lineup dependence. Original role-assignment sensitivity remains separate.',ha='center',fontsize=9);fig.tight_layout(rect=[0,.05,1,1]);pdf.savefig(fig);fig.savefig(out/'pooling_sampling_diagnostic.png',dpi=150);plt.close(fig)
        fig,axes=plt.subplots(1,len(case),figsize=(17,6));vmax=max((A/np.maximum(A.sum(1,keepdims=True),1)).max() for A in caseA)
        for ax,z,A in zip(axes,case.itertuples(),caseA):
            P=np.divide(A,A.sum(1,keepdims=True),out=np.zeros_like(A),where=A.sum(1,keepdims=True)>0);im=ax.imshow(P,vmin=0,vmax=vmax,cmap='Blues');ax.set_xticks(range(5),[name.split(' ')[-1] for name in names],rotation=50,ha='right');ax.set_yticks(range(5),names);ax.set_xlabel('Receiver');ax.set_ylabel('Passer');ax.set_title(f'{labels[z.game_id]}\nn={int(A.sum())}; entropy={float(H(A)):.3f}')
            for i in range(5):
                for j in range(5):ax.text(j,i,'—' if i==j else str(int(A[i,j])),ha='center',va='center',color='white' if P[i,j]>.5*vmax else '#222',fontsize=9)
        fig.suptitle('Same five players in all three games | cells: observed counts; shading: row probabilities');fig.tight_layout(rect=[0,0,1,.93]);pdf.savefig(fig);fig.savefig(out/'same_lineup_receiver_matrices.png',dpi=150);plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(15,6));xx=np.arange(len(details));axes[0].plot(xx,details.raw_entropy,'o-',color='#347d96',label='Original observed counts');lo=details.equal_passer_mean-details.equal_passer_q025;hi=details.equal_passer_q975-details.equal_passer_mean;axes[0].errorbar(xx,details.equal_passer_mean,yerr=np.maximum(np.array([lo,hi]),0.),fmt='o',color='#c27b3a',capsize=5,label='Same passer counts: subsampling mean/range');axes[0].set_xticks(xx,[labels[g] for g in details.game_id],rotation=15);axes[0].set_ylabel('Conditional entropy / log2(4)');axes[0].set_ylim(0,1);axes[0].legend(fontsize=8);axes[0].set_title(f'Matched source totals: {int(row_min.sum())} passes per game')
        for col,label,color in [('dirichlet_alpha_point1','Prior alpha=0.1','#777777'),('dirichlet_alpha_point5','Prior alpha=0.5','#347d96'),('dirichlet_alpha1','Prior alpha=1.0','#c27b3a')]:axes[1].plot(xx,details[col],'o-',label=label,color=color)
        axes[1].set_xticks(xx,[labels[g] for g in details.game_id],rotation=15);axes[1].set_ylabel('Posterior expected entropy (conditional model)');axes[1].set_ylim(0,1);axes[1].legend(fontsize=8);axes[1].set_title('Small-sample prior sensitivity is retained, not hidden');fig.text(.5,.015,'Subsampling ranges are not confidence intervals. A zero-width range when all passes are retained means no subsampling variation, not zero estimation uncertainty.\nDirichlet estimates assume independent receiver choices given passer and keep empirical source weights; none validates complete NBA passing coverage.',ha='center',fontsize=9);fig.tight_layout(rect=[0,.085,1,1]);pdf.savefig(fig);fig.savefig(out/'same_lineup_sample_matching.png',dpi=150);plt.close(fig)
        fig,ax=plt.subplots(figsize=(12,7));main=scores[scores.alpha.eq(.5)];width=.24
        for j,(model,label,color) in enumerate([('uniform_four_receivers','Uniform four-receiver baseline','#999'),('receiver_popularity','Receiver-popularity model','#c27b3a'),('passer_specific_receivers','Passer-specific model','#347d96')]):
            y=main[main.model.eq(model)].set_index('held_game').loc[details.game_id].held_logloss_bits_per_pass;ax.bar(xx+(j-1)*width,y,width,label=label,color=color)
        ax.axhline(2,color='gray',ls='--');ax.set_xticks(xx,[labels[g] for g in details.game_id]);ax.set_ylabel('Held-game predictive log loss, bits per retained pass (lower is better)');ax.set_title('Train on two games; score the same five-player lineup in the third');ax.legend(fontsize=9);fig.text(.5,.015,'Fixed smoothing alpha=0.5; alpha=0.1 and 1 sensitivity is saved. Log loss is a predictive score and can exceed 2 bits; it is not normalized entropy.\nOnly three held games from one lineup; no population confidence interval, significance test, or causal claim.',ha='center',fontsize=9);fig.tight_layout(rect=[0,.08,1,1]);pdf.savefig(fig);fig.savefig(out/'held_game_prediction.png',dpi=150);plt.close(fig)
    settings=dict(primary_question='How much passing diversity persists with the same five players, and do conditional receiver patterns predict another game?',case_selection='largest retained-pass total among five-player groups observed in at least two games; no entropy/outcome selection',team=team,lineup=lineup,names=names,passes=int(caseA.sum()),games=len(case),draws=args.draws,seed=20261115,common_passer_counts=dict(zip(map(str,ids),map(int,row_min))),uniform_reference='independent receiver choices, fixed observed passer totals; diagnostic only',subsampling_ranges_are_confidence=False,role_assignment_used_for_main_case=False)
    (out/'analysis_settings.json').write_text(json.dumps(settings,indent=2));source += [dict(path=str(base/f),sha256=hashlib.sha256((base/f).read_bytes()).hexdigest()) for f in ['lineup_summary.csv','lineup_role_assignments.csv','pass_role_assignments.csv']];pd.DataFrame(source).to_csv(out/'source_manifest.csv',index=False)
    report='''# Sampling-aware entropy analysis of tracked lineups

The primary analysis uses ACTUAL PLAYER IDENTITIES inside the tracked five-player groups. Inferred PG/SG/SF/PF/C labels are unnecessary for within-lineup entropy because role relabeling is a permutation. The original role networks remain a visualization and the role-assignment sensitivity stays separate. Entropy is calculated from unscaled passing counts only. The event-diagram edges are not row-normalized into an entropy model: shot weights were rescaled for display, event layers overlap, and their coverage differs from inferred passes. No arbitrary starness index or regression across tiny lineup samples is added.

## Why sample size comes first

Most graphs are based on very few retained inferred passes. An empirical entropy estimate can look low simply because some receivers have not yet appeared. The four-receiver independent-uniform diagnostic holds each lineup's observed passer totals fixed and simulates a maximally diverse process. Its empirical entropy is still below 1 at these sample sizes. This is a sampling illustration, not a realistic basketball null, randomization test or confidence interval. It ignores pass/chain dependence and tracking-selection effects.

sample_size_and_entropy_audit.csv retains raw entropy, observed-support Miller–Madow values (unclipped, with out-of-bounds flags), and posterior expected entropy under symmetric Dirichlet alpha 0.1, 0.5 and 1 priors on each passer's four possible receivers. Source weights remain the empirical outgoing shares. These are model/prior sensitivities, not automatic corrections that establish true entropy. No statistical rank ordering of the 79 lineups is claimed.

## Same-five-player case

The largest repeated lineup is selected by pass volume alone. Comparisons hold five actual player identities fixed. Both total-count and source-count-matched sampling without replacement are saved. Their ranges describe subsets of these already observed passes, not independent possessions or confidence about population entropy. When a game's entire sample is retained, the subsampling range can have zero width despite genuine estimation uncertainty. Opponents and other context still differ across games, so no opponent effect or tactical intervention is identified.

## Predictive check

For each held game, train only on the other games of the same lineup. Compare a fixed uniform-four-receiver baseline (2 bits per pass), a smoothed receiver-popularity model (incoming frequency, renormalized to exclude the current passer), and a smoothed passer-specific receiver model. Default alpha=0.5 is fixed; alpha=0.1 and 1 are sensitivity checks, not tuned by held-game fit. Loss is mean negative log2 assigned probability: lower is better, and scores can exceed 2 bits. It is a predictive cross-entropy, not a normalized entropy estimate.

This is three held games from one lineup, not external validation of NBA offense. Passes remain threshold-based tracking inferences with incomplete coverage and within-chain dependence. No independent-pass p-value or population confidence interval is produced. Better prediction than uniform suggests repeatable patterns in this retained sample; improvement over receiver popularity would more specifically support passer-dependent partner information. A failure to improve also has scientific value.

## Pooling gap

The observed pooled-role entropy minus pass-weighted within-lineup entropy is nonnegative. The independent-uniform diagnostic produces a positive gap too because tiny within-lineup samples depress empirical estimates. Therefore the observed gap cannot be attributed wholly to lineup dependence or tactical mixing. The reference does not account for realistic possession paths, co-playing opportunity beyond the known five, or inference bias.

## Tracking-settings check

For the selected lineup, robust_setting_entropy uses only passes retained under strict, primary and relaxed settings in the earlier extractor. Agreement is sensitivity to inference thresholds, not a probability of pass correctness. Count differences also affect empirical entropy, so raw robust-subset entropy is not a clean accuracy comparison. The full sample is downsampled to each threshold-agreement subset count as an additional diagnostic, and threshold_subset_prediction.csv repeats the held-game score check on agreement-only passes. Agreement does not establish correctness.

## Dataset and files

Three 2015–16 historical games; no new games downloaded and no 180-season results are inferred from this pilot. Start with lineup_entropy_analysis.pdf (five pages). recurring_lineup_case.csv and recurring_lineup_players.csv identify the case and matched counts. leave_one_game_out_prediction.csv and pooled_held_game_prediction.csv retain every prediction result, including prior sensitivity. Matrix/lineage hashes are saved. The graphs preserve original player/pass data and earlier analyses.

## Main case: observed and matched entropy

'''+details.to_string(index=False)+'\n\n## Held-game score summary\n\n'+aggregate.to_string(index=False)+'\n\n## Pooling/sampling diagnostic\n\n'+ref.to_string(index=False)+'\n\nMethodological reference: [Paninski (2003), Estimation of Entropy and Mutual Information](https://www.cns.nyu.edu/pub/lcv/paninski-infoEst-2003.pdf). Small-sample information estimates need explicit attention to bias and uncertainty. The finite-sample benchmark and matched-lineup study here are diagnostic applications, not claims of a validated correction or calibrated confidence intervals for basketball tracking.\n'
    (out/'START_HERE.md').write_text(report);print('PRIMARY CASE:',team,lineup,names);print(details.to_string(index=False));print('PREDICTION:');print(aggregate.to_string(index=False));print('SAMPLING REFERENCE:');print(ref.to_string(index=False));print('Saved',out);return 0

if __name__=='__main__':raise SystemExit(main())
