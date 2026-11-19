"""Plots and methodology for a retrospective split-role benchmark."""
from pathlib import Path
import inspect,json
import numpy as np,pandas as pd,networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def report(root,baseline,out,data,summary,skip_network_plots=False):
    root,baseline,out=map(Path,[root,baseline,out])
    ledger=pd.read_csv(out/'minutes_and_passing_role_weights.csv')
    names={}
    for season in sorted(data.season.unique()):
        f=pd.read_csv(root/'star_dependency_study/game_cache'/(season+'_player_base.csv.gz'))
        names[season]=f.drop_duplicates('PLAYER_ID').set_index('PLAYER_ID').PLAYER_NAME.to_dict()
    q=data[data.eligible]
    with PdfPages(out/'passing_role_model_comparison.pdf') as pdf:
        fig,axes=plt.subplots(1,2,figsize=(14,6))
        for ax,metric,title in [(axes[0],'edge','Passing-pattern error (total variation)'),(axes[1],'entropy','Absolute normalized entropy error')]:
            x=q['%s_error_old'%metric];y=q['%s_error_new'%metric];limit=max(x.max(),y.max())*1.06
            ax.scatter(x,y,s=30,alpha=.7);ax.plot([0,limit],[0,limit],c='firebrick',label='Equal error')
            ax.set(xlim=(0,limit),ylim=(0,limit),xlabel='Minutes-only forecast error',ylabel='Separate passing-role forecast error',title=title)
            ax.legend()
        fig.suptitle(f'{len(q)} eligible team-seasons | below the diagonal means the new model is closer',fontsize=14)
        fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(out/'old_vs_new_forecast_errors.png',dpi=150);pdf.savefig(fig);plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(15,6));s=summary[summary['sample'].eq('eligible')&summary.season.ne('ALL')];x=np.arange(len(s))
        for ax,metric,title in [(axes[0],'edge','Median passing-pattern error'),(axes[1],'entropy','Median absolute entropy error')]:
            ax.bar(x-.18,s['median_%s_old'%metric],.36,label='Minutes-only')
            ax.bar(x+.18,s['median_%s_new'%metric],.36,label='Separate passing roles')
            ax.set_xticks(x,s.season,rotation=30);ax.set_title(title);ax.set_ylabel('Lower is closer to the observed network');ax.legend()
        fig.tight_layout();fig.savefig(out/'forecast_errors_by_season.png',dpi=150);pdf.savefig(fig);plt.close(fig)
        examples=[('2025-26','DEN'),('2021-22','GSW'),('2024-25','NYK')]
        fig,axes=plt.subplots(3,1,figsize=(15,13))
        for ax,(season,team) in zip(axes,examples):
            w=ledger[ledger.season.eq(season)&ledger.team.eq(team)&~ledger.is_hub].copy()
            w['display_score']=w.minute_replacement_share+w.sender_replacement_share+w.receiver_replacement_share
            w=w.nlargest(7,'display_score');x=np.arange(len(w))
            for offset,col,label in [(-.25,'minute_replacement_share','Share of extra hub minutes'),(0,'sender_replacement_share','Share of hub outgoing passes'),(.25,'receiver_replacement_share','Share of passes previously received by hub')]:
                ax.bar(x+offset,w[col],.25,label=label)
            ax.set_xticks(x,[names[season].get(pid,str(pid)) for pid in w.player_id],rotation=12);ax.set_ylabel('Pre-route replacement share');ax.set_title(season+' '+team);ax.legend(fontsize=8)
        fig.suptitle('Replacing court time and replacing passing responsibility are separate decisions\nTop seven players by combined displayed shares; exclusions to prevent self-passes are applied afterward',fontsize=13)
        fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(out/'minutes_vs_passing_roles.png',dpi=150);pdf.savefig(fig);plt.close(fig)
    if not skip_network_plots:
        with PdfPages(out/'all_180_split_role_networks.pdf') as pdf:
            for z in data.itertuples():
                src=baseline/z.season/z.team;dest=out/z.season/z.team
                frames=[pd.read_csv(src/'original_evaluation.csv',index_col=0),pd.read_csv(dest/'predicted_evaluation.csv',index_col=0),pd.read_csv(src/'actual_evaluation.csv',index_col=0)]
                ids=frames[0].index.astype(int).tolist();matrices=[f.to_numpy(float) for f in frames]
                U=sum(B/B.sum() for B in matrices if B.sum()>0);G=nx.from_numpy_array(U+U.T);G=G.subgraph(np.flatnonzero(U.sum(0)+U.sum(1)>0)).copy()
                opts=dict(seed=42,weight=None,iterations=300,k=1.5/np.sqrt(len(G)))
                if 'method' in inspect.signature(nx.spring_layout).parameters:opts['method']='force'
                pos=nx.spring_layout(G,**opts);coords=np.array(list(pos.values()));lo=coords.min(0)-.25;hi=coords.max(0)+.25
                scale=max((1000*B/B.sum()).max() for B in matrices if B.sum()>0)
                fig,axes=plt.subplots(1,3,figsize=(24,9))
                titles=['Original: early hub-present games','Predicted: separate minutes and passing roles',f'Actual: {z.actual_absent_games} later tracked hub-absent games']
                for ax,B,title,h in zip(axes,matrices,titles,[z.entropy_original,z.entropy_new,z.entropy_actual]):
                    if B.sum()==0:ax.text(.5,.5,'No observed later absence comparison',ha='center',va='center',transform=ax.transAxes);ax.set_title(title);ax.axis('off');continue
                    active=np.flatnonzero(B.sum(0)+B.sum(1)>0);H=nx.DiGraph();H.add_nodes_from(active);edges=[(i,j) for i in active for j in active if B[i,j]>0];H.add_edges_from(edges)
                    sizes=[16000*(B.sum(0)[i]+B.sum(1)[i])/(2*B.sum()) for i in H]
                    nx.draw_networkx_edges(H,pos,width=[.1+5*1000*B[i,j]/B.sum()/scale for i,j in edges],alpha=.27,edge_color='#3a7b99',arrows=True,arrowsize=8,node_size=sizes,connectionstyle='arc3,rad=.07',ax=ax)
                    nx.draw_networkx_nodes(H,pos,node_size=sizes,node_color=['#e59d47' if ids[i]==z.hub_id else '#dcebef' for i in H],edgecolors='#536b76',ax=ax)
                    nx.draw_networkx_labels(H,pos,labels={i:str(names[z.season].get(ids[i],ids[i])).replace(' ','\n',1) for i in H},font_size=6.5,ax=ax)
                    ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[1],hi[1]);ax.axis('off');ax.set_title(title+f'\nOutgoing conditional entropy {h:.4f}',fontsize=12)
                fig.suptitle(f'{z.season} {z.team} | passing hub: {z.hub_name}\nMinutes forecast unchanged; sender and receiver responsibilities estimated separately',fontsize=16)
                fig.text(.5,.02,'All role inputs precede the first-20 cutoff. Existing nonhub edges stay fixed; hub edges are redistributed without self-passes.\nNode areas use each graph’s own involvement shares; fixed Fruchterman–Reingold positions are for display only.\nMissing tracking exclusions and regression eligibility are inherited from the baseline study.',ha='center',fontsize=10)
                fig.tight_layout(rect=[0,.1,1,.91]);fig.savefig(dest/(z.team+'_three_networks.png'),dpi=120);pdf.savefig(fig);plt.close(fig)
                print('Plotted',z.season,z.team,flush=True)
    primary=summary[summary['sample'].eq('eligible')&summary.season.eq('ALL')].iloc[0].to_dict()
    (out/'summary.json').write_text(json.dumps(primary,indent=2))
    text=['# Separate minutes and passing roles: six-season benchmark',
          'All 180 forecasts retain the previously frozen minutes allocation, hub identity and EC index. Extra minutes still use early rotations, previous-season broad position labels and earlier replacement evidence. Passing responsibility is now separate from those extra-minute shares.',
          'For each nonhub player with positive early hub-present minutes and forecast minutes, estimate outgoing passes per minute and incoming passes per minute separately. Shrink each rate toward its nonhub team peer rate using 48 prior minutes. Forecast total minutes × each shrunk rate gives separate sender and receiver weights. Total forecast minutes, rather than extra minutes, allow an existing starter to inherit playmaking without receiving much additional court time.',
          'If early hub-absent games exist, shrink each player’s absence rate toward their own shrunk present rate using 48 prior minutes. Positive rate gains × forecast minutes form additional role evidence. Blend with base role weights using n_early_abs/(n_early_abs+5), separately for sending and receiving. If there is no positive gain evidence, use base weights. The 48-minute and five-game constants were fixed before this benchmark and were not optimized against its errors.',
          'Keep nonhub-to-nonhub edges fixed. Distribute each hub-to-j edge among replacement senders using sender weights, excluding j. Distribute each i-to-hub edge among replacement receivers using receiver weights, excluding i. Renormalize each route. Pass mass is preserved and the hub row, column and diagonal are zero.',
          'Touches and potential assists are NOT used: complete first-20 tracking for those measures is not saved across all six seasons. Passes per minute measure recorded passing activity, not the quality of playmaking. Position is not a hard restriction on the passing-role step.',
          'All new matrices and input hashes are frozen before evaluation matrices are read. Original and actual networks are unchanged. The prior 96-team-season exposure sample is the primary benchmark; all 171 observed comparisons are also exported. Missing-data exclusions, entropy denominators and candidates are inherited. Original observed EC–entropy regressions therefore remain unchanged.',
          'This is retrospective model development using previously examined absence outcomes, not fresh external validation. A later untouched season is needed to validate a final chosen rule. No weights were tuned to maximize improvement. Low-minute rates, static minutes, later injuries and roster moves remain limitations. Early-only replacement candidates are unchanged to isolate the passing-role change.',
          f"Primary benchmark: median entropy error changes from {primary['median_entropy_old']:.4f} to {primary['median_entropy_new']:.4f}, with {int(primary['teams_entropy_improved'])}/{int(primary['n'])} improving. Median edge-share error changes from {primary['median_edge_old']:.4f} to {primary['median_edge_new']:.4f}, with {int(primary['teams_edge_improved'])}/{int(primary['n'])} improving and {int(primary['teams_edge_worsened'])}/{int(primary['n'])} worsening. This version is closer on entropy but is not an overall improvement in predicting who passes to whom.",
          json.dumps(primary,indent=2),
          'Negative new-minus-old error means improvement. Total-variation error compares normalized edge shares, with 0 identical and 1 disjoint. Entropy error is absolute difference under the same inherited within-team reference; different networks can have similar entropy, so both errors matter.']
    (out/'METHOD_AND_RESULTS.md').write_text('\n\n'.join(text))
