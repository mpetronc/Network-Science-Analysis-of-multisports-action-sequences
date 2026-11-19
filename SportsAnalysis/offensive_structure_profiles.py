"""Observed offensive structure: concentration, leader dominance and role alignment.

Standalone or teamgraphs.py --offensive-structure-profile. Saved NBA API data only.
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from scipy.stats import rankdata,pearsonr,linregress,t
from scipy.sparse.csgraph import connected_components

COLORS={'Higher in both':'#b94d3c','Lower in both':'#2b887c','Higher passing only':'#8173af','Higher offensive actions only':'#c49225'}


def profile(A,roster):
    """Return exact player populations, shares and deterministic leader definitions."""
    ids=roster.PLAYER_ID.astype(int).to_numpy();A=np.asarray(A,float)
    if len(set(ids))!=len(ids) or A.shape!=(len(ids),len(ids)):raise ValueError('Duplicate/mismatched player IDs')
    v=roster[['FGA','FTA','TOV','MIN']].apply(pd.to_numeric,errors='raise').to_numpy(float)
    if not np.isfinite(v).all() or (v<0).any() or not np.isfinite(A).all() or (A<0).any() or np.diag(A).any():raise ValueError('Invalid counts or missing inputs')
    active=np.flatnonzero(A.sum(0)+A.sum(1)>0);B=A[np.ix_(active,active)];n=len(B)
    if n<2 or A.sum()<=0 or connected_components((B+B.T)>0,directed=False)[0]!=1:raise ValueError('Need a connected active passing network')
    vals,V=np.linalg.eigh(B+B.T);c=np.abs(V[:,-1]);c/=c.sum();assert np.allclose((B+B.T)@c,vals[-1]*c,rtol=1e-7,atol=1e-6)
    fullc=np.zeros(len(ids));fullc[active]=c;actions=v[:,0]+.44*v[:,1]+v[:,2]
    if actions.sum()<=0 or v[:,3].sum()<=0:raise ValueError('No actions or minutes')
    u=actions/actions.sum();m=v[:,3]/v[:,3].sum();ma=m[active]/m[active].sum()
    ci=sorted(active,key=lambda i:(-fullc[i],ids[i]));ui=sorted(range(len(ids)),key=lambda i:(-u[i],ids[i]))
    ch=float(c@c);uh=float(u@u);mh=float(m@m);cn=(ch-1/n)/(1-1/n)
    def gap(a,b):return float((a-b)/(a+b))
    row=dict(passing_players=n,roster_players=len(ids),recorded_passes=float(A.sum()),
        passing_ec_concentration=float(np.clip(cn,0,1)),passing_ec_hhi=ch,offensive_action_hhi=uh,effective_action_players=1/uh,
        passing_top1_share=float(fullc[ci[0]]),passing_top2_share=float(fullc[ci[:2]].sum()),passing_leader_gap=gap(fullc[ci[0]],fullc[ci[1]]),
        action_top1_share=float(u[ui[0]]),action_top2_share=float(u[ui[:2]].sum()),action_leader_gap=gap(u[ui[0]],u[ui[1]]),
        passing_leader_id=int(ids[ci[0]]),passing_leader_name=str(roster.iloc[ci[0]].PLAYER_NAME),passing_runner_up_name=str(roster.iloc[ci[1]].PLAYER_NAME),
        action_leader_id=int(ids[ui[0]]),action_leader_name=str(roster.iloc[ui[0]].PLAYER_NAME),action_runner_up_name=str(roster.iloc[ui[1]].PLAYER_NAME),
        passing_leader_action_share=float(u[ci[0]]),action_leader_ec_share=float(fullc[ui[0]]),
        leader_alignment='Shared leader' if ci[0]==ui[0] else 'Separate leaders',
        passing_leader_tie=bool(np.isclose(fullc[ci[0]],fullc[ci[1]],rtol=1e-6,atol=1e-10)),
        action_leader_tie=bool(np.isclose(u[ui[0]],u[ui[1]],rtol=1e-6,atol=1e-10)),
        minutes_hhi=mh,active_passing_minutes_hhi=float(ma@ma),
        passing_concentration_vs_minutes=ch/float(ma@ma),action_concentration_vs_minutes=uh/mh,
        dependence_validated=False)
    players=pd.DataFrame(dict(player_id=ids,name=roster.PLAYER_NAME.to_numpy(),minutes=v[:,3],minutes_share=m,
        passing_ec_share=fullc,offensive_actions=actions,action_share=u,passing_active=np.isin(np.arange(len(ids)),active)))
    return row,players


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--offensive-structure-profile',action='store_true');p.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'nba_passing_output');p.add_argument('--destination',type=Path)
    args=p.parse_args(argv);root=args.output_dir.expanduser();out=args.destination or root/'offensive_structure_profiles';out.mkdir(parents=True,exist_ok=True)
    rows=[];allplayers=[];sources=[]
    for f in sorted((root/'full_roster_entropy_centrality/historical_full_rosters').glob('20*/*/passing_matrix.csv')):
        frame=pd.read_csv(f,index_col=0);ids=frame.index.astype(int).tolist()
        if frame.columns.astype(int).tolist()!=ids:raise ValueError('Matrix row/column IDs disagree')
        rosterpath=f.with_name('roster.csv');roster=pd.read_csv(rosterpath)
        if roster.PLAYER_ID.duplicated().any() or set(roster.PLAYER_ID)!=set(ids):raise ValueError('Roster mismatch')
        roster=roster.set_index('PLAYER_ID').loc[ids].reset_index();r,players=profile(frame.to_numpy(float),roster);r.update(team=f.parent.name,season=f.parent.parent.name);rows.append(r);players.insert(0,'team',r['team']);players.insert(0,'season',r['season']);allplayers.append(players)
        for file in [f,rosterpath]:sources.append(dict(team=r['team'],season=r['season'],path=str(file),sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
    d=pd.DataFrame(rows);assert len(d)==180 and d.groupby('season').size().eq(30).all()
    x='passing_ec_concentration';y='offensive_action_hhi';xc=float(d[x].median());yc=float(d[y].median())
    d['concentration_profile']=np.select([(d[x]>=xc)&(d[y]>=yc),(d[x]<xc)&(d[y]<yc),d[x]>=xc],['Higher in both','Lower in both','Higher passing only'],default='Higher offensive actions only')
    d['passing_concentration_percentile']=100*(rankdata(d[x],method='average')-1)/(len(d)-1);d['action_concentration_percentile']=100*(rankdata(d[y],method='average')-1)/(len(d)-1)
    d['supplementary_combined_concentration']=(d.passing_concentration_percentile+d.action_concentration_percentile)/2
    players=pd.concat(allplayers,ignore_index=True);d.to_csv(out/'team_season_profiles.csv',index=False);players.to_csv(out/'player_shares.csv',index=False);pd.DataFrame(sources).to_csv(out/'source_manifest.csv',index=False)
    summary=d.groupby(['concentration_profile','leader_alignment']).size().unstack(fill_value=0);summary.to_csv(out/'profile_counts.csv');d[d.season.eq('2025-26')].to_csv(out/'2025-26_profiles.csv',index=False)
    pd.DataFrame([dict(season=s,n=len(g),shared=int(g.leader_alignment.eq('Shared leader').sum()),separate=int(g.leader_alignment.eq('Separate leaders').sum())) for s,g in d.groupby('season')]).to_csv(out/'leader_alignment_by_season.csv',index=False)
    correlations=[]
    for scope,g in [('ALL',d),*list(d.groupby('season'))]:
        for a,b in [(x,y),('passing_leader_gap','action_leader_gap')]:
            r=pearsonr(g[a],g[b]).statistic;correlations.append(dict(scope=scope,x=a,y=b,n=len(g),r=r,r_squared=r*r,status='descriptive; repeated franchises pooled'))
    pd.DataFrame(correlations).to_csv(out/'descriptive_correlations.csv',index=False)
    (out/'reference_rules.json').write_text(json.dumps(dict(reference='180 saved team-seasons 2020-21 to 2025-26',passing_median=xc,action_median=yc,leader_tie_rule='smallest player ID breaks exact ties; near-tie flags retained',primary='continuous concentration + leader dominance + leader identity',supplementary='equal mean of pooled concentration percentile ranks; not a validated starness probability',opportunity='HHI / minutes HHI for the matching population; inequality comparison, not causal adjustment',cutoffs='pooled medians are reference lines, not discovered classes'),indent=2))
    def scatter(ax,g,a,b,title,labels=False,mode='raw'):
        for alignment,marker in [('Shared leader','o'),('Separate leaders','D')]:
            z=g[g.leader_alignment.eq(alignment)]
            ax.scatter(z[a],z[b],c=[COLORS[q] for q in z.concentration_profile],marker=marker,s=36 if labels else 26,alpha=.8,edgecolors='white',linewidths=.35)
        if labels:
            for z in g.itertuples():ax.annotate(z.team,(getattr(z,a),getattr(z,b)),xytext=(3,3),textcoords='offset points',fontsize=7)
        if mode=='raw':ax.axvline(xc,c='gray',ls='--',lw=1);ax.axhline(yc,c='gray',ls='--',lw=1)
        elif mode=='opportunity':ax.axvline(1,c='gray',ls='--',lw=1);ax.axhline(1,c='gray',ls='--',lw=1)
        ax.set_title(title,fontsize=12);ax.grid(alpha=.15)
    def legend(ax):
        handles=[Line2D([0],[0],marker='o',color='none',markerfacecolor=c,markeredgecolor=c,label=q) for q,c in COLORS.items()]+[Line2D([0],[0],marker='o',color='gray',ls='',label='Shared leader'),Line2D([0],[0],marker='D',color='gray',ls='',label='Separate leaders')]
        ax.legend(handles=handles,fontsize=8,loc='best')
    with PdfPages(out/'offensive_structure_profiles.pdf') as pdf:
        fig,ax=plt.subplots(figsize=(11,8));scatter(ax,d,x,y,'Observed offensive structure | 180 team-seasons');legend(ax);ax.set_xlabel('Passing-only normalized EC concentration');ax.set_ylabel('Offensive action-share HHI');fig.text(.5,.015,'Dashed lines: fixed pooled medians. Colors: relative concentration. Shapes: whether the same player leads both roles.\nNo basket edges, minutes cutoff or regression controls. This is a descriptive season profile, not validated star dependence.',ha='center',fontsize=9);fig.tight_layout(rect=[0,.06,1,1]);pdf.savefig(fig);fig.savefig(out/'all_seasons_concentration.png',dpi=150);plt.close(fig)
        for season,g in d.groupby('season'):
            fig,ax=plt.subplots(figsize=(11,8));scatter(ax,g,x,y,season+' | observed offensive structure',labels=True);legend(ax);ax.set_xlabel('Passing-only normalized EC concentration');ax.set_ylabel('Offensive action-share HHI');ax.set_xlim(d[x].min()-.005,d[x].max()+.005);ax.set_ylim(d[y].min()-.005,d[y].max()+.005);fig.tight_layout();pdf.savefig(fig);fig.savefig(out/(season+'_concentration.png'),dpi=150);plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(15,6))
        for ax,g,title in [(axes[0],d,'All 180 team-seasons'),(axes[1],d[d.season.eq('2025-26')],'2025–26')]:
            scatter(ax,g,'passing_leader_gap','action_leader_gap',title,labels=len(g)==30,mode='gaps');ax.set_xlabel('Passing EC first–second gap');ax.set_ylabel('Offensive-action first–second gap')
        fig.suptitle('Leader dominance: a large gap indicates a clearer leader over the runner-up');fig.text(.5,.015,'Gap = (share1 − share2) / (share1 + share2). A small gap does not prove broad balance; check top-two shares. Colors retain raw concentration profiles.',ha='center',fontsize=9);fig.tight_layout(rect=[0,.05,1,.94]);pdf.savefig(fig);fig.savefig(out/'leader_dominance.png',dpi=150);plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(15,6));latest=d[d.season.eq('2025-26')]
        scatter(axes[0],latest,x,y,'Observed season concentration',labels=True);axes[0].set_xlabel('Passing-only normalized EC concentration');axes[0].set_ylabel('Offensive-action HHI')
        scatter(axes[1],latest,'passing_concentration_vs_minutes','action_concentration_vs_minutes','Comparison relative to minutes inequality',labels=True,mode='opportunity');axes[1].set_xlabel('Passing EC HHI / active-player minutes HHI');axes[1].set_ylabel('Action HHI / roster minutes HHI')
        fig.suptitle('2025–26 | raw and opportunity-relative views answer different questions');fig.text(.5,.015,'Ratio 1 means equal inequality of the two share distributions, not identical roles or an exact shared-court-time baseline. Colors retain raw profiles.',ha='center',fontsize=9);fig.tight_layout(rect=[0,.05,1,.94]);pdf.savefig(fig);fig.savefig(out/'2025-26_minutes_comparison.png',dpi=150);plt.close(fig)
        # Deterministic illustrations selected only from current-season structure; no performance selection.
        choices=[];used=set()
        candidates=[('Higher concentration, shared leader',latest[(latest.concentration_profile=='Higher in both')&(latest.leader_alignment=='Shared leader')].sort_values('supplementary_combined_concentration',ascending=False)),('Higher concentration, separate leaders',latest[(latest.concentration_profile=='Higher in both')&(latest.leader_alignment=='Separate leaders')].sort_values('supplementary_combined_concentration',ascending=False)),('Lower season concentration',latest.sort_values('supplementary_combined_concentration')),('Two comparable passing leaders',latest[latest.passing_top2_share>=latest.passing_top2_share.median()].sort_values('passing_leader_gap'))]
        for label,candidate in candidates:
            candidate=candidate[~candidate.team.isin(used)]
            if candidate.empty:continue
            z=candidate.iloc[0];choices.append(dict(selection=label,team=z.team,season=z.season));used.add(z.team);g=players[(players.team==z.team)&(players.season==z.season)].copy();g=g.sort_values('passing_ec_share',ascending=False).head(10).iloc[::-1]
            fig,ax=plt.subplots(figsize=(12,7));yy=np.arange(len(g));ax.barh(yy+.18,g.passing_ec_share,height=.34,label='Passing EC share',color='#336eab');ax.barh(yy-.18,g.action_share,height=.34,label='Offensive-action share',color='#dc8d39');ax.set_yticks(yy,g.name);ax.set_xlabel('Share within team-season');ax.legend();ax.set_title(f'{z.team} {z.season} | {label}\nPassing leader: {z.passing_leader_name}; action leader: {z.action_leader_name}');fig.text(.5,.01,f'Passing top two: {z.passing_top2_share:.1%}; gap: {z.passing_leader_gap:.3f} | Actions top two: {z.action_top2_share:.1%}; gap: {z.action_leader_gap:.3f}\nShowing the ten highest passing-EC players; shares retain full-team denominators.',ha='center',fontsize=9);fig.tight_layout(rect=[0,.055,1,1]);pdf.savefig(fig);fig.savefig(out/('example_'+z.team+'.png'),dpi=150);plt.close(fig)
    pd.DataFrame(choices).to_csv(out/'illustration_selection.csv',index=False)
    # Existing absence results are an exploratory check only; no new data or causal validation.
    absent=root/'2025-26/team_season_topics/four_tactical_questions/hub_absence_matched_metrics.csv';elig=root/'2025-26/team_season_topics/hub_absence_resilience/eligibility.csv';validation_note='No compatible saved absence table found; dependence is not validated.'
    if absent.exists() and elig.exists():
        a=pd.read_csv(absent);e=pd.read_csv(elig);joined=latest.merge(e[['team','hub_id']],on='team',how='left',validate='one_to_one').merge(a[['team','hub_name','hub_present_games','hub_absent_games','adjusted_offensive_penalty','entropy_loss']],on='team',how='left',validate='one_to_one');joined['matching_passing_leader']=joined.hub_id.eq(joined.passing_leader_id);joined['eligible_for_exploratory_check']=joined.matching_passing_leader&joined.adjusted_offensive_penalty.notna();joined.to_csv(out/'absence_validation_coverage.csv',index=False);valid=joined[joined.eligible_for_exploratory_check].copy();vr=[]
        for a in ['passing_ec_concentration','offensive_action_hhi','supplementary_combined_concentration','passing_leader_gap','action_leader_gap']:
            if len(valid)>3:
                q=linregress(valid[a],valid.adjusted_offensive_penalty);X=np.column_stack([np.ones(len(valid)),valid[a]]);bread=np.linalg.inv(X.T@X);b=np.linalg.lstsq(X,valid.adjusted_offensive_penalty,rcond=None)[0];res=valid.adjusted_offensive_penalty.to_numpy()-X@b;hat=np.sum(X*(X@bread),axis=1);u=res/(1-hat);cov=bread@(X.T@(X*u[:,None]**2))@bread;se=np.sqrt(cov[1,1]);vr.append(dict(metric=a,n=len(valid),r=q.rvalue,r_squared=q.rvalue**2,slope=q.slope,hc3_p=2*t.sf(abs(q.slope/se),len(valid)-2),status='exploratory, secondary unadjusted p; no prospective or causal validation'))
        pd.DataFrame(vr).to_csv(out/'exploratory_absence_check.csv',index=False)
        if len(valid)>3:
            fig,axes=plt.subplots(1,2,figsize=(14,6))
            for ax,a,title in [(axes[0],x,'Passing concentration'),(axes[1],'supplementary_combined_concentration','Supplementary combined concentration')]:
                ax.scatter(valid[a],valid.adjusted_offensive_penalty,c=[COLORS[q] for q in valid.concentration_profile],s=40);coef=np.polyfit(valid[a],valid.adjusted_offensive_penalty,1);xx=np.linspace(valid[a].min(),valid[a].max(),100);ax.plot(xx,np.polyval(coef,xx),c='gray');ax.axhline(0,c='gray',ls='--',lw=1);ax.set_xlabel(title);ax.set_ylabel('Saved adjusted offensive absence penalty');ax.set_title(f'Exploratory absence check | n={len(valid)}')
                for z in valid.itertuples():ax.annotate(z.team,(getattr(z,a),z.adjusted_offensive_penalty),xytext=(3,3),textcoords='offset points',fontsize=7)
            fig.text(.5,.015,'Existing selected absence sample; full-season predictors are retrospective. Estimated penalty precision varies. No causal or external validation claim.',ha='center',fontsize=9);fig.tight_layout(rect=[0,.055,1,1]);fig.savefig(out/'exploratory_absence_check.png',dpi=150);fig.savefig(out/'exploratory_absence_check.pdf');plt.close(fig)
        validation_note=f'Existing absence data: {len(valid)}/30 current-season teams have an adjusted penalty and the same identified passing hub. Results are exploratory, not a validation of dependence. Season predictors include the absence periods; sample selection, exposure and outcome estimation matter. No fresh downloads or absence re-estimation were performed.'
        for file in [absent,elig]:sources.append(dict(team='ALL',season='2025-26',path=str(file),sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
        pd.DataFrame(sources).to_csv(out/'source_manifest.csv',index=False)
    report='''# Revised offensive-structure profile

The primary result is a three-part profile: concentration, leader dominance and leader identity. Start with offensive_structure_profiles.pdf or the current-season concentration PNG. All 180 saved NBA API full-roster season networks are included. No basket edges, minute cutoffs, statistical controls or new downloads enter the primary profile.

## Definitions

Passing concentration: EC shares from original player-to-player A+A.T, excluding passing isolates, then (HHI−1/N)/(1−1/N). It is overall network prominence, not specifically outgoing creation or possession time. Offensive-action concentration: HHI of full-roster shares of FGA+0.44*FTA+TOV; it is season workload, not official USG% or efficiency. Input missingness and invalid values are rejected, not silently zero-filled. A zero-pass player remains in action shares when they have actions. Full player shares are saved.

Dominance: top-one share, combined top-two share and (share1−share2)/(share1+share2), calculated separately for passing EC and offensive actions. A small gap suggests comparable leaders; it does not prove broad balance. A large gap suggests a clearer leader; no universal one-star/two-star cutoff is imposed. Exact ties use player ID for reproducibility and near-ties are flagged.

Alignment: the same versus different player IDs leading passing EC and action totals. This is not necessarily the highest official usage-rate player. Secondary fields show the action share of the passing leader and EC share of the action leader.

## Reference lines and supplementary score

Quadrants use fixed pooled medians across the 180 team-seasons, with labels higher/lower concentration rather than star/balanced classes. They are descriptive sample references, not discovered populations. All seasons use the same reference lines. The supplementary_combined_concentration score is the equal mean of two percentile ranks on this reference, on a 0–100 scale. It is not the primary result, a probability, or a validated star-dependence measure. Frozen rules are saved. A future scoring implementation should use this reference rather than silently re-ranking a new dataset.

## Opportunity view

The minutes comparison is separate from the primary observed-season view. EC HHI is divided by minutes HHI among the same passing-active players; action HHI is divided by full-roster minutes HHI. Ratio 1 means equal inequality, not equal player shares or correct shared-court exposure. This is not a causal adjustment or a formal co-play null. Heavy minutes can legitimately contribute to season dependence; the two views answer different questions.

## Reading leader shares

Illustrations are deterministically selected from 2025–26 using only concentration, alignment and leader-gap definitions. They are illustrations, not evidence validating a class. Ten highest passing-EC players are drawn, but every share keeps its full-team denominator. The complete player table contains everyone.

## Absence validation status

'''+validation_note+'''

The optional saved absence check uses team-specific pre-existing adjusted offense penalties and only matching hub IDs. Positive penalties mean worse offense without the hub. It does not reconstruct the matching model, propagate all first-stage penalty uncertainty, use an independent prospective predictor, or establish how the team would have performed if the player were present. Multiple p-values are exploratory and uncorrected. It is evidence to examine, not a completed validation. Claims of dependence need a pre-specified prospective game-level design.

## Results

'''+summary.to_string()+f'\n\nShared leader: {int(d.leader_alignment.eq("Shared leader").sum())}/180. Near-tied passing leaders: {int(d.passing_leader_tie.sum())}; near-tied action leaders: {int(d.action_leader_tie.sum())}.\n\nRecommended question: Do teams concentrate passing and possession-ending actions in the same player, or divide those roles between a distributor and a finisher?\n\nRun: python3 teamgraphs.py --offensive-structure-profile. Standalone: python3 offensive_structure_profiles.py --output-dir /path/to/nba_passing_output. Source hashes, raw shares and exact definitions are retained; previous analyses are preserved.\n'
    (out/'START_HERE.md').write_text(report);print(summary.to_string());print(validation_note);print('Saved',out);return 0

if __name__=='__main__':raise SystemExit(main())
