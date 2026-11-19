"""Conditional passing entropy versus offensive structure, split by leader alignment."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import linregress,t

Y=[('passing_ec_concentration','Passing-only EC concentration'),('offensive_action_hhi','Offensive action-share HHI')]
COLORS={'Shared leader':'#247c9c','Separate leaders':'#c56a37'}


def conditional_entropy(A):
    A=np.asarray(A,float)
    if A.ndim!=2 or A.shape[0]!=A.shape[1] or not np.isfinite(A).all() or (A<0).any() or np.diag(A).any():raise ValueError('Invalid passing matrix')
    active=np.flatnonzero(A.sum(0)+A.sum(1)>0);A=A[np.ix_(active,active)]
    if len(A)<3 or not np.isfinite(A).all() or (A<0).any() or np.diag(A).any() or A.sum()<=0:raise ValueError('Invalid passing matrix')
    w=A.sum(1);P=np.divide(A,w[:,None],out=np.zeros_like(A),where=w[:,None]>0)
    bits=float(-np.sum(A*np.log2(np.maximum(P,1e-300)))/A.sum())
    return bits/np.log2(len(A)-1),bits,len(A),A.sum()


def regress(d,y,x='conditional_entropy',interaction=None,season_fe=False):
    h=d[x].to_numpy(float);hc=h-h.mean();columns=[np.ones(len(d)),hc];terms=['intercept','entropy']
    if interaction:
        z=(d.leader_alignment.eq('Shared leader').to_numpy(float) if interaction=='alignment' else d.passing_leader_gap.to_numpy(float)-d.passing_leader_gap.mean())
        columns += [z,hc*z];terms += ['shared_leader' if interaction=='alignment' else 'centered_passing_gap','entropy_interaction']
    if season_fe:
        dummy=pd.get_dummies(d.season,drop_first=True,dtype=float)
        columns += [dummy[c].to_numpy() for c in dummy];terms += ['season_'+c for c in dummy]
    X=np.column_stack(columns);v=d[y].to_numpy(float)
    if np.linalg.matrix_rank(X)<X.shape[1]:raise ValueError('Rank-deficient model')
    b=np.linalg.lstsq(X,v,rcond=None)[0];res=v-X@b;bread=np.linalg.inv(X.T@X)
    if d.season.nunique()>1:
        groups=d.team.unique();G=len(groups);meat=np.zeros_like(bread)
        for g in groups:
            k=d.team.eq(g).to_numpy();u=X[k].T@res[k];meat+=np.outer(u,u)
        cov=bread@meat@bread*G/(G-1)*(len(d)-1)/(len(d)-X.shape[1]);df=G-1;kind='franchise-clustered'
    else:
        hat=np.sum(X*(X@bread),axis=1);u=res/(1-hat);cov=bread@(X.T@(X*u[:,None]**2))@bread;df=len(d)-X.shape[1];kind='HC3';G=d.team.nunique()
    q=linregress(h,v);j=terms.index('entropy');se=np.sqrt(max(cov[j,j],0));c=t.ppf(.975,df)
    result=dict(n=len(d),franchises=G,r=q.rvalue,r_squared=q.rvalue**2,slope=b[j],slope_per_0_1_entropy=b[j]/10,
        robust_se=se,robust_p=2*t.sf(abs(b[j]/se),df),slope_ci_low=b[j]-c*se,slope_ci_high=b[j]+c*se,
        standard_errors=kind,df=df,season_fixed_effects=season_fe,entropy_mean=float(h.mean()),
        fitted_intercept_at_entropy_mean=b[0],outcome=y,x=x,interaction=interaction or 'none',
        full_model_r_squared=1-np.sum(res**2)/np.sum((v-v.mean())**2))
    if interaction:
        j=terms.index('entropy_interaction');se=np.sqrt(max(cov[j,j],0));result.update(interaction_slope=b[j],interaction_se=se,interaction_p=2*t.sf(abs(b[j]/se),df),interaction_ci_low=b[j]-c*se,interaction_ci_high=b[j]+c*se)
    return result,b,cov,terms


def holm(frame,source,target,mask):
    ix=frame.loc[mask,source].sort_values().index
    frame.loc[ix,target]=np.minimum(1,np.maximum.accumulate([(len(ix)-j)*frame.loc[i,source] for j,i in enumerate(ix)]))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--entropy-structure-profile',action='store_true');p.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'nba_passing_output');p.add_argument('--destination',type=Path)
    args=p.parse_args(argv);root=args.output_dir.expanduser();out=args.destination or root/'offensive_structure_profiles/entropy_analysis';out.mkdir(parents=True,exist_ok=True)
    profile_path=root/'offensive_structure_profiles/team_season_profiles.csv';d=pd.read_csv(profile_path)
    sourcehashes=pd.read_csv(root/'offensive_structure_profiles/source_manifest.csv');sourcehashes=dict(zip(sourcehashes.path,sourcehashes.sha256));rows=[];sources=[dict(path=str(profile_path),sha256=hashlib.sha256(profile_path.read_bytes()).hexdigest())]
    for z in d.itertuples():
        f=root/'full_roster_entropy_centrality/historical_full_rosters'/z.season/z.team/'passing_matrix.csv';sha=hashlib.sha256(f.read_bytes()).hexdigest()
        if str(f) not in sourcehashes or sourcehashes[str(f)]!=sha:raise ValueError('Passing source changed since profile: '+str(f))
        A=pd.read_csv(f,index_col=0);assert A.index.astype(int).tolist()==A.columns.astype(int).tolist();h,bits,n,total=conditional_entropy(A.to_numpy(float));assert n==z.passing_players and np.isclose(total,z.recorded_passes)
        rows.append(dict(team=z.team,season=z.season,conditional_entropy=h,conditional_entropy_bits=bits));sources.append(dict(path=str(f),sha256=sha))
    d=d.merge(pd.DataFrame(rows),on=['team','season'],validate='one_to_one');assert len(d)==180 and d.groupby('season').size().eq(30).all();d.to_csv(out/'entropy_and_structure.csv',index=False);pd.DataFrame(sources).to_csv(out/'source_manifest.csv',index=False)
    models=[];cache={};sens=[]
    for y,label in Y:
        for season,g in [('ALL',d),*list(d.groupby('season'))]:
            for group in ['All teams','Shared leader','Separate leaders']:
                a=g if group=='All teams' else g[g.leader_alignment.eq(group)]
                if len(a)<5 or a.conditional_entropy.nunique()<2:continue
                r,b,cov,terms=regress(a,y);r.update(season=season,group=group);models.append(r);cache[y,season,group]=(r,b,cov)
        for group in ['All teams','Shared leader','Separate leaders']:
            a=d if group=='All teams' else d[d.leader_alignment.eq(group)]
            for x,fe in [('conditional_entropy_bits',False),('conditional_entropy',True)]:
                r,_,_,_=regress(a,y,x=x,season_fe=fe);r.update(season='ALL',group=group);sens.append(r)
    models=pd.DataFrame(models);models['holm_p']=np.nan
    holm(models,'robust_p','holm_p',models.season.eq('ALL')&models.group.ne('All teams'))
    holm(models,'robust_p','holm_p',models.season.ne('ALL')&models.group.ne('All teams'))
    models.to_csv(out/'group_regressions.csv',index=False);pd.DataFrame(sens).to_csv(out/'bits_and_season_effect_sensitivity.csv',index=False)
    interactions=[];gap_results=[];gap_cache={}
    for y,label in Y:
        for fe in [False,True]:
            r,_,_,_=regress(d,y,interaction='alignment',season_fe=fe);r.update(season='ALL');interactions.append(r)
            g,b,cov,terms=regress(d,y,interaction='gap',season_fe=fe);g.update(season='ALL');gap_results.append(g)
            if not fe:gap_cache[y]=(g,b,cov,terms)
        for season,g in d.groupby('season'):
            r,_,_,_=regress(g,y,interaction='alignment');r.update(season=season);interactions.append(r)
    interactions=pd.DataFrame(interactions);interactions['holm_interaction_p']=np.nan
    holm(interactions,'interaction_p','holm_interaction_p',interactions.season.eq('ALL')&~interactions.season_fixed_effects)
    holm(interactions,'interaction_p','holm_interaction_p',interactions.season.ne('ALL'))
    interactions.to_csv(out/'shared_vs_separate_slope_tests.csv',index=False);gap_results=pd.DataFrame(gap_results);gap_results['holm_interaction_p']=np.nan;holm(gap_results,'interaction_p','holm_interaction_p',~gap_results.season_fixed_effects);gap_results.to_csv(out/'continuous_leader_gap_interactions.csv',index=False)
    def draw(ax,g,y,label,season,labels=False):
        for group,color in COLORS.items():
            a=g[g.leader_alignment.eq(group)];key=(y,season,group)
            if key not in cache:continue
            r,b,cov=cache[key];xx=np.linspace(a.conditional_entropy.min(),a.conditional_entropy.max(),130);X=np.column_stack([np.ones(len(xx)),xx-r['entropy_mean']]);pred=X@b;band=t.ppf(.975,r['df'])*np.sqrt(np.maximum(np.sum(X*(X@cov),axis=1),0));ax.fill_between(xx,pred-band,pred+band,color=color,alpha=.13);ax.plot(xx,pred,color=color,lw=2);ax.scatter(a.conditional_entropy,a[y],color=color,s=28,alpha=.78,label=f'{group}: n={len(a)}, r={r["r"]:.3f}, R²={r["r_squared"]:.1%}')
            if labels:
                for z in a.itertuples():ax.annotate(z.team,(z.conditional_entropy,getattr(z,y)),xytext=(3,3),textcoords='offset points',fontsize=6)
        ax.set_xlabel('Normalized conditional passing entropy H(receiver | passer)');ax.set_ylabel(label);ax.set_title(season);ax.grid(alpha=.14);ax.legend(fontsize=7)
    with PdfPages(out/'entropy_structure_regressions.pdf') as pdf:
        fig,axes=plt.subplots(1,2,figsize=(16,6))
        for ax,(y,label) in zip(axes,Y):draw(ax,d,y,label,'ALL')
        fig.suptitle('Conditional passing entropy and offensive structure | 180 team-seasons',fontsize=15);fig.text(.5,.015,'Separate descriptive regression lines; 95% fitted-mean bands cluster repeated seasons by franchise. Basket excluded. No primary covariates.\nGroup differences require an interaction test; significance within one group alone does not establish different slopes.',ha='center',fontsize=9);fig.tight_layout(rect=[0,.075,1,.93]);pdf.savefig(fig);fig.savefig(out/'pooled_shared_vs_separate.png',dpi=160);plt.close(fig)
        for y,label in Y:
            fig,axes=plt.subplots(2,3,figsize=(17,11))
            for ax,(season,g) in zip(axes.flat,d.groupby('season')):draw(ax,g,y,label,season,labels=True)
            fig.suptitle(label+' versus conditional passing entropy | separate seasons',fontsize=16);fig.text(.5,.01,'HC3 fitted-mean uncertainty; exact subgroup sample sizes shown. Leader identities and concentration are derived from the same season data.',ha='center',fontsize=9);fig.tight_layout(rect=[0,.035,1,.94]);pdf.savefig(fig);fig.savefig(out/(y+'_by_season.png'),dpi=150);plt.close(fig)
        for season,g in d.groupby('season'):
            fig,axes=plt.subplots(1,2,figsize=(16,6))
            for ax,(y,label) in zip(axes,Y):draw(ax,g,y,label,season,labels=True)
            fig.suptitle(season+' | passing entropy and offensive structure');fig.tight_layout(rect=[0,0,1,.93]);pdf.savefig(fig);fig.savefig(out/(season+'_entropy_profiles.png'),dpi=150);plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(15,6));gg=np.linspace(d.passing_leader_gap.min(),d.passing_leader_gap.max(),150)
        for ax,(y,label) in zip(axes,Y):
            r,b,cov,terms=gap_cache[y];L=np.zeros((len(gg),len(b)));L[:,terms.index('entropy')]=1;L[:,terms.index('entropy_interaction')]=gg-d.passing_leader_gap.mean();slopes=L@b;band=t.ppf(.975,r['df'])*np.sqrt(np.maximum(np.sum(L*(L@cov),axis=1),0));ax.fill_between(gg,(slopes-band)/10,(slopes+band)/10,alpha=.2,color='#247c9c');ax.plot(gg,slopes/10,color='#247c9c');ax.axhline(0,color='gray',ls='--');ax.set_xlabel('Passing EC first–second leader gap');ax.set_ylabel(label+' change per +0.1 entropy');ax.set_title(f'Exploratory continuous interaction\nHolm p={float(gap_results.loc[(gap_results.outcome==y)&~gap_results.season_fixed_effects,"holm_interaction_p"].iloc[0]):.3g}');ax.grid(alpha=.15)
        fig.suptitle('Does passing-leader dominance change the entropy association?');fig.text(.5,.015,'Gap is continuous; no one-star/two-star cutoff. Bands are franchise-clustered. This does not identify a causal effect of leadership.',ha='center',fontsize=9);fig.tight_layout(rect=[0,.055,1,.93]);pdf.savefig(fig);fig.savefig(out/'leader_gap_slope_sensitivity.png',dpi=150);plt.close(fig)
    report='''# Entropy within the revised offensive-structure profile

Main question: How does conditional passing entropy relate to offensive concentration, and does the association differ when the passing and offensive-action leaders are the same player?

The X-axis is original player-to-player conditional passing entropy, not basket-augmented entropy. For each passer, calculate receiver-share Shannon entropy, average using the passer's share of all passes, and divide by log2(N−1) where N is the number of passing-active players. Raw bits are retained as sensitivity. Matrix hashes are checked against the revised profile provenance before joining. No minutes cutoff or primary covariates; basket edges are excluded throughout.

The two Y-axes are (1) passing-only normalized EC concentration on A+A.T and (2) season offensive action-share HHI, based on FGA+0.44*FTA+TOV. These are concentration measures, not official USG%, skill or validated dependence. Colors identify Shared leader versus Separate leaders using the player IDs in the revised profile. Existing top-one/top-two shares and gaps remain in entropy_and_structure.csv; they complement rather than replace entropy.

Pooled group fits and 95% fitted-mean bands use franchise-clustered standard errors and t(G−1), with exact numbers of franchises and observations saved. Per-season group fits use HC3 and t(n−2). Lines are drawn only across each group's observed entropy range. Different Pearson r or a significant slope in only one group does not prove different slopes. The separate interaction model y~centered_entropy+shared+centered_entropy*shared directly tests the slope difference. Positive interaction coefficient means a more positive/less negative slope for Shared than Separate leaders. Holm adjustment covers the two primary pooled interaction tests; season interactions form a separate 12-test family. Four pooled subgroup slope tests and 24 season subgroup slope tests have separate Holm families. Unadjusted baseline and sensitivity results are labeled.

The exploratory leader-gap model uses y~centered_entropy+centered_gap+entropy*gap, with the passing EC first–second gap as a continuous modifier. It avoids inventing a cutoff for one-star versus two-star teams. Two pooled gap interactions receive their own Holm correction; these are exploratory, not a second confirmatory discovery family. Seasonal fixed effects and raw entropy bits are supplementary sensitivities. In fixed-effects/interaction rows, r and r_squared are unadjusted bivariate descriptors; full_model_r_squared is saved separately. The entropy main slope in a gap model is evaluated at mean gap, and in an alignment model it is the Separate-leader slope.

Interpretation limits: season entropy mixes passing choices, playing-time allocation and lineup changes. H(receiver|passer) is empirical target diversity, not a proven measure of in-lineup offensive unpredictability or Markov entropy rate. EC and entropy share the same pass matrix, so their association is partly structural; action HHI is influenced by minutes. Shared/separate labels are also determined from the same season, so conditioning on them can introduce selection effects. Groups are descriptive, not randomized. These are exploratory comparisons following earlier analyses; correction within the current test families does not account for every analysis tried across the project. Strong regressions do not demonstrate better offense, causal effects or validated star dependence. Earlier absence checks remain exploratory and are not re-estimated here. All NBA inputs are saved data; no fresh API request was made.

## Pooled subgroup regressions

'''+models[(models.season=='ALL')&(models.group!='All teams')][['outcome','group','n','franchises','r','r_squared','slope','robust_p','holm_p']].to_string(index=False)+'\n\n## Direct pooled slope-difference tests\n\n'+interactions[(interactions.season=='ALL')&~interactions.season_fixed_effects][['outcome','interaction_slope','interaction_p','holm_interaction_p']].to_string(index=False)+'\n\n## Continuous passing-leader-gap tests (exploratory)\n\n'+gap_results[~gap_results.season_fixed_effects][['outcome','interaction_slope','interaction_p','holm_interaction_p']].to_string(index=False)+'\n\nRun python3 teamgraphs.py --entropy-structure-profile, or the standalone module with --output-dir /path/to/nba_passing_output. Outputs go into offensive_structure_profiles/entropy_analysis; earlier profiles and analyses are preserved.\n'
    (out/'START_HERE.md').write_text(report);print(models[(models.season=='ALL')&(models.group!='All teams')][['outcome','group','n','franchises','r','r_squared','slope','robust_p']].to_string(index=False));print('INTERACTIONS:');print(interactions[(interactions.season=='ALL')&~interactions.season_fixed_effects][['outcome','interaction_slope','interaction_p','holm_interaction_p']].to_string(index=False));print('GAP:');print(gap_results[~gap_results.season_fixed_effects][['outcome','interaction_slope','interaction_p','holm_interaction_p']].to_string(index=False));print('Saved',out);return 0

if __name__=='__main__':raise SystemExit(main())
