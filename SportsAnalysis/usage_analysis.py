"""Usage/entropy regressions and a passing-only EC × usage starness quadrant."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import t,rankdata,spearmanr
from basket_regressions import fit

MODELS=[('passing_usage','passing_entropy','usage_hhi','Passing entropy vs usage concentration'),
        ('basket_usage','basket_entropy','usage_hhi','Pass-or-shot entropy vs usage concentration'),
        ('passing_leader_rate','passing_entropy','leader_est_usg_pct','Passing entropy vs leading player estimated USG%'),
        ('basket_leader_rate','basket_entropy','leader_est_usg_pct','Pass-or-shot entropy vs leading player estimated USG%')]


def usage_metrics(roster):
    required=['FGA','FTA','TOV','MIN']
    values=roster[required].apply(pd.to_numeric,errors='raise').to_numpy(float)
    if not np.isfinite(values).all() or (values<0).any():raise ValueError('Missing or invalid usage inputs')
    actions=values[:,0]+.44*values[:,1]+values[:,2];total=actions.sum()
    if total<=0:raise ValueError('Zero usage action total')
    shares=actions/total;hhi=float(shares@shares);n=len(shares)
    # Common box-score approximation, not NBA official on-court USG%.
    team_minutes=values[:,3].sum()/5
    rates=np.divide(100*actions*team_minutes,total*values[:,3],out=np.full(n,np.nan),where=values[:,3]>0)
    leader=int(np.argmax(actions))
    return dict(usage_hhi=hhi,usage_concentration_normalized=(hhi-1/n)/(1-1/n),
                effective_usage_players=1/hhi,total_usage_actions=total,
                leader_est_usg_pct=float(rates[leader]),leader_action_share=float(shares[leader]),
                usage_leader_id=int(roster.iloc[leader].PLAYER_ID),usage_leader_name=str(roster.iloc[leader].PLAYER_NAME)),actions,shares,rates


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--usage-starness',action='store_true');p.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'nba_passing_output');p.add_argument('--destination',type=Path)
    args=p.parse_args(argv);root=args.output_dir.expanduser();out=args.destination or root/'basket_networks/usage_starness';out.mkdir(parents=True,exist_ok=True)
    metrics=pd.read_csv(root/'basket_networks/regressions/network_metrics_and_ratio_sensitivity.csv');d=metrics[metrics.basket_ratio.eq(1)].copy();rows=[];players=[];manifest=[]
    for z in d.itertuples():
        folder=root/'full_roster_entropy_centrality/historical_full_rosters'/z.season/z.team
        roster=pd.read_csv(folder/'roster.csv').sort_values('PLAYER_ID').reset_index(drop=True);frame=pd.read_csv(folder/'passing_matrix.csv',index_col=0);ids=frame.index.astype(int).tolist();roster=roster.set_index('PLAYER_ID').loc[ids].reset_index();A=frame.to_numpy(float)
        u,actions,shares,rates=usage_metrics(roster)
        active=np.flatnonzero(A.sum(0)+A.sum(1)>0);S=A[np.ix_(active,active)];_,vectors=np.linalg.eigh(S+S.T);v=np.abs(vectors[:,-1]);v/=v.sum();ec_shares=np.zeros(len(ids));ec_shares[active]=v;hub=int(np.argmax(ec_shares));hhi=float(v@v)
        assert np.isclose((hhi-1/len(active))/(1-1/len(active)),z.passing_ec,atol=1e-10)
        rows.append(dict(team=z.team,season=z.season,**u,passing_hub_id=ids[hub],passing_hub_name=str(roster.iloc[hub].PLAYER_NAME),passing_hub_share=float(ec_shares[hub]),same_passing_usage_leader=ids[hub]==u['usage_leader_id']))
        for i,r in enumerate(roster.itertuples()):players.append(dict(team=z.team,season=z.season,player_id=r.PLAYER_ID,name=r.PLAYER_NAME,usage_actions=actions[i],usage_action_share=shares[i],estimated_usg_pct=rates[i],passing_ec_share=ec_shares[i],minutes=r.MIN))
        for f in [folder/'roster.csv',folder/'passing_matrix.csv']:manifest.append(dict(team=z.team,season=z.season,path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
    d=d.merge(pd.DataFrame(rows),on=['team','season'],validate='one_to_one');assert len(d)==180 and d.groupby('season').size().eq(30).all()
    # Fixed pooled empirical reference: average percentile rank, mapped to [0,100].
    d['passing_percentile']=100*(rankdata(d.passing_ec,method='average')-1)/(len(d)-1)
    d['usage_percentile']=100*(rankdata(d.usage_hhi,method='average')-1)/(len(d)-1)
    d['starness_index']=.5*d.passing_percentile+.5*d.usage_percentile
    d['component_gap']=d.passing_percentile-d.usage_percentile
    xcut=float(d.passing_ec.median());ycut=float(d.usage_hhi.median())
    d['quadrant']=np.select([(d.passing_ec>=xcut)&(d.usage_hhi>=ycut),(d.passing_ec<xcut)&(d.usage_hhi<ycut),d.passing_ec>=xcut],['Concentrated in both','Distributed in both','Passing concentrated only'],default='Usage concentrated only')
    for w in [.25,.75]:d[f'starness_passing_weight_{w:g}']=w*d.passing_percentile+(1-w)*d.usage_percentile
    d.to_csv(out/'team_season_starness_and_usage.csv',index=False);pd.DataFrame(players).to_csv(out/'player_usage_and_passing_shares.csv',index=False);pd.DataFrame(manifest).to_csv(out/'source_manifest.csv',index=False)
    rules=dict(reference='180 full-roster team-seasons, 2020-21 through 2025-26',passing_metric='normalized player EC concentration on passing-only A+A.T',usage_metric='HHI of season FGA + 0.44 FTA + TOV shares',passing_quadrant_median=xcut,usage_quadrant_median=ycut,percentile_formula='100*(average_rank-1)/(180-1)',starness_formula='0.5*passing_percentile + 0.5*usage_percentile',status='Descriptive relative index, not a validated superstar classifier; no fitted outcome weights')
    (out/'starness_reference_rules.json').write_text(json.dumps(rules,indent=2))
    models=[];cache={}
    for name,x,y,title in MODELS:
        for season,g in [('ALL',d),*list(d.groupby('season'))]:
            r,cov,df=fit(g,x,y);r.update(model=name,season=season,x=x,y=y);models.append(r);cache[name,season]=(r,cov,df)
        r,_,_=fit(d,x,y,season_fe=True);r.update(model=name,season='ALL_SEASON_FE',x=x,y=y);models.append(r)
    regressions=pd.DataFrame(models);regressions['holm_p']=np.nan
    for mask in [regressions.season.eq('ALL'),~regressions.season.isin(['ALL','ALL_SEASON_FE'])]:
        order=regressions.loc[mask].robust_p.sort_values().index;adjusted=np.maximum.accumulate([(len(order)-j)*regressions.loc[i,'robust_p'] for j,i in enumerate(order)]);regressions.loc[order,'holm_p']=np.minimum(1,adjusted)
    regressions.to_csv(out/'usage_regressions.csv',index=False)
    secondary=[]
    for x in ['passing_entropy','basket_entropy']:
        r,_,_=fit(d,x,'usage_concentration_normalized');r.update(x=x,y='normalized_usage_hhi');secondary.append(r)
    # Quadrant axes may correlate; report this rather than treating them as independent evidence.
    r,_,_=fit(d,'passing_ec','usage_hhi');r.update(x='passing_ec',y='usage_hhi');secondary.append(r)
    pd.DataFrame(secondary).to_csv(out/'quadrant_and_normalization_sensitivity.csv',index=False)
    colors={'Concentrated in both':'#ba4a35','Distributed in both':'#2c8b7a','Passing concentrated only':'#8272b5','Usage concentrated only':'#d59b24'}
    def quadrant(ax,g,title,labels=False):
        ax.axvline(xcut,color='gray',ls='--',lw=1);ax.axhline(ycut,color='gray',ls='--',lw=1)
        for q,group in g.groupby('quadrant'):ax.scatter(group.passing_ec,group.usage_hhi,label=q,color=colors[q],s=42,alpha=.82)
        if labels:
            for z in g.itertuples():ax.annotate(z.team,(z.passing_ec,z.usage_hhi),xytext=(3,3),textcoords='offset points',fontsize=7)
        ax.set_xlabel('Passing-only normalized EC concentration');ax.set_ylabel('Usage action-share HHI');ax.set_title(title);ax.grid(alpha=.13)
        dx=d.passing_ec.max()-d.passing_ec.min();dy=d.usage_hhi.max()-d.usage_hhi.min();ax.set_xlim(d.passing_ec.min()-.1*dx,d.passing_ec.max()+.1*dx);ax.set_ylim(d.usage_hhi.min()-.1*dy,d.usage_hhi.max()+.1*dy)
    with PdfPages(out/'starness_quadrants.pdf') as pdf:
        fig,ax=plt.subplots(figsize=(11,8));quadrant(ax,d,'Passing structure × usage concentration | 180 team-seasons');ax.legend(fontsize=9);fig.text(.5,.02,'Fixed pooled medians define relative quadrants. High-high concentration can involve different players. Basket edges excluded from EC.',ha='center',fontsize=10);fig.tight_layout(rect=[0,.05,1,1]);pdf.savefig(fig);fig.savefig(out/'all_seasons_starness_quadrant.png',dpi=160);plt.close(fig)
        fig,axes=plt.subplots(2,3,figsize=(16,10))
        for ax,(season,g) in zip(axes.flat,d.groupby('season')):quadrant(ax,g,season,labels=True)
        fig.suptitle('One quadrant per season | same pooled median cutoffs and axes',fontsize=16);fig.tight_layout(rect=[0,0,1,.95]);pdf.savefig(fig);fig.savefig(out/'season_starness_quadrants.png',dpi=160);plt.close(fig)
        for season,g in d.groupby('season'):
            fig,ax=plt.subplots(figsize=(11,8));quadrant(ax,g,season+' | passing-only EC × usage',labels=True);ax.legend(fontsize=9);fig.tight_layout();pdf.savefig(fig);fig.savefig(out/(season+'_starness_quadrant.png'),dpi=150);plt.close(fig)
    def draw(ax,g,name,x,y,title,season,labels=False):
        r,cov,df=cache[name,season];grid=np.linspace(g[x].min(),g[x].max(),100);X=np.column_stack([np.ones(len(grid)),grid]);pred=r['intercept']+r['slope']*grid;band=t.ppf(.975,df)*np.sqrt(np.maximum(np.sum(X*(X@cov),axis=1),0));ax.fill_between(grid,pred-band,pred+band,color='#b95c32',alpha=.18);ax.plot(grid,pred,color='#b95c32');ax.scatter(g[x],g[y],c=[colors[q] for q in g.quadrant],s=30,alpha=.8)
        if labels:
            for z in g.itertuples():ax.annotate(z.team,(getattr(z,x),getattr(z,y)),xytext=(3,3),textcoords='offset points',fontsize=6)
        ax.set_title(f'{title}\n{season}: r={r["r"]:.3f}; R²={r["r_squared"]:.1%}; robust p={r["robust_p"]:.3g}',fontsize=11);ax.set_xlabel('Conditional passing entropy' if x=='passing_entropy' else 'Conditional pass-or-shot entropy');ax.set_ylabel('Usage action-share HHI' if y=='usage_hhi' else 'Leading action player estimated USG%');ax.grid(alpha=.15)
    with PdfPages(out/'usage_regressions.pdf') as pdf:
        for selection,stem in [(MODELS[:2],'usage_concentration'),(MODELS[2:],'leader_estimated_usage')]:
            fig,axes=plt.subplots(1,2,figsize=(15,6))
            for ax,(name,x,y,title) in zip(axes,selection):draw(ax,d,name,x,y,title,'ALL')
            fig.suptitle('180 team-seasons | pooled uncertainty clustered by franchise');fig.text(.5,.015,'Colors show passing-only EC × usage quadrant. Usage HHI and estimated player USG% are different quantities.',ha='center',fontsize=10);fig.tight_layout(rect=[0,.05,1,.94]);pdf.savefig(fig);fig.savefig(out/(stem+'_pooled.png'),dpi=160);plt.close(fig)
        for name,x,y,title in MODELS:
            fig,axes=plt.subplots(2,3,figsize=(16,10))
            for ax,(season,g) in zip(axes.flat,d.groupby('season')):draw(ax,g,name,x,y,title,season,labels=True)
            fig.suptitle(title+' | separate seasons');fig.tight_layout(rect=[0,0,1,.95]);pdf.savefig(fig);fig.savefig(out/(name+'_by_season.png'),dpi=150);plt.close(fig)
    pooled=regressions[regressions.season.eq('ALL')];agreement=int(d.same_passing_usage_leader.sum());sens=[]
    for w in [.25,.75]:sens.append(dict(passing_weight=w,spearman_vs_equal_weight=spearmanr(d.starness_index,d[f'starness_passing_weight_{w:g}']).statistic))
    pd.DataFrame(sens).to_csv(out/'index_weight_sensitivity.csv',index=False)
    report='''# Usage regressions and passing/usage starness index

The quadrant X-axis is normalized eigenvector concentration on ONLY the original passing network (A+A.T). Basket edges never enter this axis. The Y-axis is usage action-share HHI: for each player, a=FGA+0.44*FTA+TOV, s=a/sum(a), HHI=sum(s²). Larger means a smaller set of players accounts for more season offensive actions. It reflects playing time as well as on-court role. It is NOT official NBA individual USG%. All full-roster season totals are retained; no minutes cutoff, covariates, or minutes-concentration/roster-count regression controls. Normalized usage HHI is saved as sensitivity, not a control.

For the separate player-rate regressions, estimated_USG%=100*(player actions/player minutes)*(sum roster minutes/5)/sum roster actions. This is the common season box-score approximation, NOT downloaded official on-court NBA USG%. Select the leader by largest season action total, rather than the highest rate among tiny-minute reserves; the choice is made without consulting entropy. The full estimated-rate table is retained. Missing inputs are rejected, not zero-filled. Official player USG% was unavailable in these saved inputs. Zero-minute rates remain undefined.

The new index equals 0.5*passing concentration percentile + 0.5*usage concentration percentile, relative to these 180 team-seasons. Percentiles use 100*(average rank−1)/(180−1). It is a descriptive relative score on a 0–100 scale, not a probability or validated star classifier. The observed maximum need not reach 100. Reference values and median quadrant cutoffs are frozen in starness_reference_rules.json. Adding future networks must not silently re-rank this reference. Equal weights are a choice, not learned from outcomes; .25/.75 component-weight sensitivity is saved.

Median quadrants describe concentrated in both, distributed in both, passing-concentrated only, or usage-concentrated only. A high-high team can have different passing and finishing stars; same_passing_usage_leader records whether the leaders match. An average index can hide disagreement, so retain both components, quadrant, and component_gap. The components can correlate; this is not two independent pieces of evidence of stardom.

Regressions compare both original passing entropy and synthetic basket-augmented pass-or-shot entropy against usage HHI, and separately against the leading action player's estimated USG%. The augmented distribution uses scaled FGA and is not an observed sequence probability. It shares FGA inputs with usage, so its association can partly be mechanical. Original entropy uses only passes but can still co-vary with roster/playing time. No causal or performance inference follows. Pooled p-values and 95% fitted-mean bands cluster 180 observations into 30 franchises; per-season models use HC3 with t28. Holm correction spans four pooled models, and separately all 24 season tests. Season fixed effects are an additional pooled sensitivity; r/R² in those rows remain unadjusted bivariate descriptors.

Inputs reuse saved NBA API totals, with no fresh download. No basket-network or earlier results are overwritten.

## Pooled regressions

'''+pooled[['model','r','r_squared','robust_p','holm_p']].to_string(index=False)+f'\n\nPassing hub and usage leader match in {agreement}/180 team-seasons.\n\n## Quadrant counts\n\n'+d.quadrant.value_counts().to_string()+'\n\n## Component relationship\n\n'+pd.DataFrame(secondary).tail(1).to_string(index=False)
    (out/'METHOD_AND_RESULTS.md').write_text(report);print(pooled[['model','r','r_squared','robust_p','holm_p']].to_string(index=False));print(d.quadrant.value_counts().to_string());print('Same leader:',agreement,'/180');print(pd.DataFrame(secondary).tail(1)[['r','r_squared','robust_p']].to_string(index=False));print('Saved',out);return 0

if __name__=='__main__':raise SystemExit(main())
