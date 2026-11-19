"""Regress conditional entropy against player EC concentration, with/without basket."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import linregress,t
from basket_networks import build_networks


def entropy(A):
    A=np.asarray(A,float);n=len(A);total=A.sum();out=A.sum(axis=1)
    bits=0.
    for row,w in zip(A,out):
        if w>0:
            p=row[row>0]/w;bits+=(w/total)*(-np.sum(p*np.log2(p)))
    return float(bits/np.log2(n-1)),float(bits)


def ec(A,player_count):
    S=A+A.T
    values,vectors=np.linalg.eigh(S)
    v=np.abs(vectors[:,-1]);assert np.allclose(S@v,values[-1]*v,rtol=1e-7,atol=1e-6)
    shares=v[:player_count]/v[:player_count].sum();hhi=float(shares@shares)
    return (hhi-1/player_count)/(1-1/player_count),hhi,float(v[-1]/v.sum())


def fit(d,x,y,season_fe=False):
    xx=d[x].to_numpy();yy=d[y].to_numpy();X=np.column_stack([np.ones(len(d)),xx])
    if season_fe:X=np.column_stack([X,pd.get_dummies(d.season,drop_first=True,dtype=float).to_numpy()])
    b=np.linalg.lstsq(X,yy,rcond=None)[0];e=yy-X@b;bread=np.linalg.inv(X.T@X)
    if d.season.nunique()>1:
        groups=d.team.unique();meat=np.zeros((X.shape[1],X.shape[1]))
        for team in groups:
            keep=d.team.eq(team).to_numpy();u=X[keep].T@e[keep];meat+=np.outer(u,u)
        cov=bread@meat@bread*len(groups)/(len(groups)-1)*(len(d)-1)/(len(d)-X.shape[1]);df=len(groups)-1;kind='team-clustered'
    else:
        hat=np.sum(X*(X@bread),axis=1);u=e/(1-hat);cov=bread@(X.T@(X*u[:,None]**2))@bread;df=len(d)-X.shape[1];kind='HC3'
    se=np.sqrt(cov[1,1]);q=linregress(xx,yy)
    return dict(n=len(d),r=q.rvalue,r_squared=q.rvalue**2,slope=b[1],intercept=b[0],
        ols_p=q.pvalue,robust_p=2*t.sf(abs(b[1]/se),df),slope_se=se,
        ci_low=b[1]-t.ppf(.975,df)*se,ci_high=b[1]+t.ppf(.975,df)*se,
        standard_errors=kind,season_fixed_effects=season_fe),cov,df


MODELS=[('passing_only','passing_entropy','passing_ec','Original passing network'),
        ('passing_entropy_basket_ec','passing_entropy','basket_player_ec','Passing entropy vs basket-informed player EC'),
        ('basket_both','basket_entropy','basket_player_ec','Augmented entropy vs basket-informed player EC')]


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--basket-regressions',action='store_true');p.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'nba_passing_output');p.add_argument('--destination',type=Path)
    args=p.parse_args(argv);root=args.output_dir.expanduser();out=args.destination or root/'basket_networks/regressions';out.mkdir(parents=True,exist_ok=True)
    rows=[];manifest=[]
    for file in sorted((root/'full_roster_entropy_centrality/historical_full_rosters').glob('20*/*/passing_matrix.csv')):
        roster_path=file.with_name('roster.csv');frame=pd.read_csv(file,index_col=0);roster=pd.read_csv(roster_path)
        G,H,audit=build_networks(frame,roster,1.)
        A=frame.to_numpy(float);active=np.flatnonzero(A.sum(0)+A.sum(1)>0);B=A[np.ix_(active,active)]
        h,bits=entropy(B);c,hhi,_=ec(B,len(B))
        for ratio in [.5,1.,2.]:
            w=audit.basket_weight.to_numpy()*ratio;keep=np.flatnonzero(A.sum(0)+A.sum(1)+w>0);C=np.zeros((len(keep)+1,len(keep)+1));C[:-1,:-1]=A[np.ix_(keep,keep)];C[:-1,-1]=w[keep]
            hc,bc=entropy(C);cc,ch,bs=ec(C,len(keep))
            rows.append(dict(team=file.parent.name,season=file.parent.parent.name,basket_ratio=ratio,
                passing_entropy=h,passing_entropy_bits=bits,passing_ec=c,passing_hhi=hhi,
                basket_entropy=hc,basket_entropy_bits=bc,basket_player_ec=cc,basket_player_hhi=ch,
                basket_share_all_node_ec=bs,passing_players=len(B),basket_players=len(keep),
                recorded_passes=A.sum(),total_fga=audit.fga.sum()))
        for source in [file,roster_path]:manifest.append(dict(path=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
    data=pd.DataFrame(rows);primary=data[data.basket_ratio.eq(1.)].copy();assert len(primary)==180 and primary.groupby('season').size().eq(30).all()
    data.to_csv(out/'network_metrics_and_ratio_sensitivity.csv',index=False);pd.DataFrame(manifest).to_csv(out/'source_manifest.csv',index=False)
    models=[];cache={}
    for name,x,y,title in MODELS:
        for season,d in [('ALL',primary),*list(primary.groupby('season'))]:
            result,cov,df=fit(d,x,y);result.update(model=name,season=season,x=x,y=y);models.append(result);cache[name,season]=(result,cov,df)
        result,_,_=fit(primary,x,y,season_fe=True);result.update(model=name,season='ALL_SEASON_FE',x=x,y=y);models.append(result)
    results=pd.DataFrame(models);results['holm_p']=np.nan
    for mask in [results.season.eq('ALL'),~results.season.isin(['ALL','ALL_SEASON_FE'])]:
        order=results.loc[mask].robust_p.sort_values().index;adj=np.maximum.accumulate([(len(order)-j)*results.loc[i,'robust_p'] for j,i in enumerate(order)]);results.loc[order,'holm_p']=np.minimum(1,adj)
    results.to_csv(out/'regressions.csv',index=False)
    sensitivity=[]
    for ratio,d in data.groupby('basket_ratio'):
        for name,x,y,title in MODELS[1:]:
            r,_,_=fit(d,x,y);r.update(basket_ratio=ratio,model=name);sensitivity.append(r)
    pd.DataFrame(sensitivity).to_csv(out/'ratio_sensitivity_regressions.csv',index=False)
    colors=dict(zip(sorted(primary.season.unique()),plt.get_cmap('tab10').colors))
    def draw(ax,d,name,x,y,title,season,labels=False):
        r,cov,df=cache[name,season];grid=np.linspace(d[x].min(),d[x].max(),150);X=np.column_stack([np.ones(len(grid)),grid]);band=t.ppf(.975,df)*np.sqrt(np.maximum(np.sum(X*(X@cov),axis=1),0));pred=r['intercept']+r['slope']*grid
        ax.fill_between(grid,pred-band,pred+band,color='#c25c2a',alpha=.15);ax.plot(grid,pred,color='#c25c2a')
        for s,g in d.groupby('season'):ax.scatter(g[x],g[y],color=colors[s],s=26,alpha=.8,label=s)
        if labels:
            for z in d.itertuples():ax.annotate(z.team,(getattr(z,x),getattr(z,y)),xytext=(3,3),textcoords='offset points',fontsize=6)
        ax.set_title(f'{title}\n{season}: n={len(d)}; r={r["r"]:.3f}; R²={r["r_squared"]:.1%}; robust p={r["robust_p"]:.3g}',fontsize=11)
        ax.set_xlabel('Conditional passing entropy' if x=='passing_entropy' else 'Conditional pass-or-shot entropy');ax.set_ylabel('Normalized player EC concentration');ax.grid(alpha=.15)
    with PdfPages(out/'basket_regressions.pdf') as pdf:
        fig,axes=plt.subplots(1,3,figsize=(20,6))
        for ax,(name,x,y,title) in zip(axes,MODELS):draw(ax,primary,name,x,y,title,'ALL')
        axes[0].legend(fontsize=8);fig.suptitle('180 team-seasons | original versus basket-informed networks | basket ratio = 1',fontsize=15);fig.text(.5,.012,'Pooled p-values and 95% fitted-mean bands cluster repeated seasons by franchise. Structural association; no causal or performance claim.',ha='center',fontsize=10);fig.tight_layout(rect=[0,.045,1,.92]);pdf.savefig(fig);fig.savefig(out/'pooled_comparison.png',dpi=150);plt.close(fig)
        for name,x,y,title in MODELS:
            fig,axes=plt.subplots(2,3,figsize=(16,10))
            for ax,(season,d) in zip(axes.flat,primary.groupby('season')):draw(ax,d,name,x,y,title,season,labels=True)
            fig.suptitle(title+' — separate season regressions',fontsize=16);fig.text(.5,.015,'30 teams per season; HC3 uncertainty. Corrected p-values for the 18 season tests are in regressions.csv.',ha='center');fig.tight_layout(rect=[0,.05,1,.94]);pdf.savefig(fig);fig.savefig(out/(name+'_by_season.png'),dpi=150);plt.close(fig)
    readme='''# Basket-node regressions

These analyses reuse all 180 saved NBA API full-roster regular-season networks. No live NBA data were requested. The three primary plots compare: (1) original conditional passing entropy versus original player EC concentration; (2) the same passing entropy versus basket-informed player EC concentration; (3) conditional pass-or-shot entropy versus basket-informed player EC concentration.

EC follows the earlier combined-involvement definition: dominant eigenvector of A+A.T, L1-normalized to player shares, concentration=(HHI−1/N)/(1−1/N). For the augmented graph, compute the eigenvector WITH the basket, then REMOVE the basket entry and renormalize player shares before calculating player concentration. Thus basket connections affect player scores without treating the common basket as a star player. Symmetrization is for the EC metric only; the exported graph retains directed player→basket edges. These scores describe combined passing/shooting prominence rather than scoring skill. Original EC excludes passing isolates; augmented EC includes any player active in passing or FGA; cohort sizes are saved.

Entropy: H(destination | source), averaging each row entropy using its share of all edge weight. Original probabilities use pass counts; augmented probabilities use pass counts plus scaled FGA, so they are a synthetic pass-or-shot distribution, NOT observed possession transition probabilities. Basket has no outgoing weight, contributes zero source weight, and has no fabricated absorbing self-loop. Normalize by log2(N−1) for N graph nodes, following the earlier convention; raw bits are also saved. Original and augmented normalized entropies have different supports. Basket ratio 1 assigns half the global weight to shots; .5 and 2 sensitivity results are included. Changing this chosen ratio can mechanically change the association.

Pooled regressions report Pearson r and descriptive R², OLS slopes, and franchise-clustered standard errors (30 groups, t29). A season-fixed-effects pooled slope is an additional sensitivity check; r/R² in those rows remain the unadjusted bivariate descriptors. Separate seasons use HC3 with t28. Holm correction covers the three pooled tests as one family, and all 18 season tests as another. Ratios .5 and 2 are exploratory sensitivity checks. No minutes, roster-count, or outcome controls. Confidence bands describe fitted means rather than individual-team prediction.

Both metrics are derived from the same network, so strong relationships can be partly mathematical. Neither a strong regression nor a small p-value proves that balanced teams play better, or validates star labels. These are observational team-season comparisons, not 180 independent franchises. Source hashes, every metric, regression, and ratio sensitivity are retained.
'''
    table=results[results.season.eq('ALL')][['model','r','r_squared','slope','robust_p','holm_p']]
    readme+='\n## Pooled results\n\n'+table.to_string(index=False)+'\n\n## Ratio sensitivity\n\n'+pd.DataFrame(sensitivity)[['model','basket_ratio','r','r_squared','robust_p']].to_string(index=False)+'\n'
    (out/'METHOD_AND_RESULTS.md').write_text(readme);print(table.to_string(index=False));print('Saved '+str(out));return 0

if __name__=='__main__':raise SystemExit(main())
