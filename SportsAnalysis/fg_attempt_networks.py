"""Whole-season original player passing graphs with three FGA sinks.
Run: python3 teamgraphs.py --fg-attempt-networks [--season 2021-22 --team GSW]
"""
from pathlib import Path
import argparse,hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
import pandas as pd
from nba_api.stats.static import teams
from basket_networks import build_networks,player_positions,BASKET
CATEGORIES={'OTHER_2PT':('Mid-range /\nother 2PT','#d97920'), 'THREE_PT':('3PT','#925ab2'), 'LAYUP_DUNK':('Layup / dunk','#26967b')}

def classify(shots):
    if shots[['SHOT_TYPE','ACTION_TYPE']].isna().any().any():
        raise ValueError('Missing shot classification; download complete tracking data')
    if not shots.SHOT_TYPE.isin(['2PT Field Goal','3PT Field Goal']).all():
        raise ValueError('Unknown shot type')
    if not shots.SHOT_ATTEMPTED_FLAG.eq(1).all():
        raise ValueError('Shot file must contain attempts, including misses')
    if shots.duplicated(['GAME_ID','GAME_EVENT_ID']).any():
        raise ValueError('Duplicate shot events')
    return np.where(shots.SHOT_TYPE.eq('3PT Field Goal'),'THREE_PT',
        np.where(shots.ACTION_TYPE.str.contains('layup|dunk',case=False,regex=True),'LAYUP_DUNK','OTHER_2PT'))

def build(matrix,roster,shots):
    original,augmented,audit=build_networks(matrix,roster)
    augmented.remove_node(BASKET)
    shots=shots.copy();shots['category']=classify(shots)
    counts=shots.groupby(['PLAYER_ID','category']).size().unstack(fill_value=0).reindex(columns=CATEGORIES,fill_value=0)
    ids=audit.player_id.tolist()
    if not set(counts.index)<=set(ids):
        raise ValueError('Shot participants missing from original roster')
    counts=counts.reindex(ids,fill_value=0)
    r=roster.set_index('PLAYER_ID').loc[ids]
    if not np.array_equal(counts.sum(axis=1).to_numpy(),r.FGA.to_numpy()):
        raise ValueError('Player FGA mismatch between season roster and shot chart')
    if not np.array_equal(counts.THREE_PT.to_numpy(),r.FG3A.to_numpy()):
        raise ValueError('Player 3PA mismatch')
    for field,mask in [('FGM',shots.SHOT_MADE_FLAG.eq(1)),('FG3M',shots.SHOT_MADE_FLAG.eq(1)&shots.SHOT_TYPE.eq('3PT Field Goal'))]:
        actual=shots.loc[mask].groupby('PLAYER_ID').size().reindex(ids,fill_value=0)
        if not np.array_equal(actual.to_numpy(),r[field].to_numpy()):
            raise ValueError(f'Player {field} mismatch')
    scale=augmented.graph['basket_scale']
    audit=audit.drop(columns=['basket_weight'])
    for category,(label,color) in CATEGORIES.items():
        augmented.add_node(category,name=label.replace('\n',' '),node_type='shot_category')
        audit[category+'_attempts']=counts[category].to_numpy()
        for pid,n in counts[category].items():
            if n>0: augmented.add_edge(pid,category,weight=float(n*scale),raw_count=int(n),edge_type='shot_attempt')
    audit['strict_midrange_attempts']=shots.loc[shots.SHOT_ZONE_BASIC.eq('Mid-Range')].groupby('PLAYER_ID').size().reindex(ids,fill_value=0).to_numpy()
    assert all(augmented[u][v]==a for u,v,a in original.edges(data=True))
    assert np.isclose(sum(a['weight'] for u,v,a in augmented.edges(data=True) if a['edge_type']=='shot_attempt'),audit.outgoing_passes.sum())
    assert all(augmented.out_degree(n)==0 for n in CATEGORIES)
    return original,augmented,audit

def figure(original,graph,audit,team,season):
    pos=player_positions(original,graph);pos.pop(BASKET,None)
    for cat,y in zip(CATEGORIES,[.85,0,-.85]): pos[cat]=np.array([1.8,y])
    fig,ax=plt.subplots(figsize=(16,11))
    players=[n for n in original if graph.degree(n)>0]
    activity=audit.set_index('player_id').eval('incoming_passes + outgoing_passes')
    sizes={n:250+1100*activity.loc[n]/max(activity.max(),1) for n in players}
    sizes.update({n:3000 for n in CATEGORIES})
    H=graph.subgraph(players+list(CATEGORIES));maximum=max(a['weight'] for _,_,a in H.edges(data=True))
    for category,color in [(None,'#3971a5')]+[(c,v[1]) for c,v in CATEGORIES.items()]:
        edges=[(u,v) for u,v,a in H.edges(data=True) if (a['edge_type']=='pass' if category is None else v==category)]
        nx.draw_networkx_edges(H,pos,edgelist=edges,edge_color=color,alpha=.28 if category is None else .7,
            width=[.2+5*H[u][v]['weight']/maximum for u,v in edges],arrowsize=14,
            node_size=[sizes[n] for n in H],connectionstyle='arc3,rad=0.09',ax=ax)
    nx.draw_networkx_nodes(H,pos,nodelist=players,node_size=[sizes[n] for n in players],node_color='#f3cf65',edgecolors='#716036',ax=ax)
    nx.draw_networkx_labels(H,pos,labels={n:H.nodes[n]['name'].replace(' ','\n',1) for n in players},font_size=8,ax=ax)
    for cat,(label,color) in CATEGORIES.items():
        ax.text(*pos[cat],label,ha='center',va='center',fontsize=12,bbox=dict(boxstyle='round,pad=.65',fc='white',ec=color,lw=2),zorder=10)
    ax.set_title(f'{team} | {season} regular season\nOriginal player passing network + field-goal attempt end nodes',fontsize=19,pad=20)
    ax.set_xlim(-1.35,2.35);ax.set_ylim(-1.4,1.35);ax.axis('off')
    fig.legend(handles=[Line2D([0],[0],color='#3971a5',lw=3,label='Original passes')]+[Line2D([0],[0],color=color,lw=3,label=label.replace('\n',' ')) for label,color in CATEGORIES.values()],loc='lower center',bbox_to_anchor=(.5,.075),ncol=4,frameon=False)
    fig.text(.5,.025,f"{audit.outgoing_passes.sum():,.0f} tracked passes | {audit.fga.sum():,.0f} FG attempts (makes + misses)\nShot-edge weight = total passes × player/category attempts ÷ team FGA. Combined shot weight equals total passes.\nOther 2PT includes floaters, hooks and tip-ins. Season aggregates; edges do not establish observed pass-to-shot sequences.",ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.12,1,1]);return fig

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fg-attempt-networks',action='store_true')
    p.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'nba_passing_output')
    p.add_argument('--shot-cache',type=Path)
    p.add_argument('--destination',type=Path)
    p.add_argument('--season',default='ALL');p.add_argument('--team',default='ALL')
    a=p.parse_args(argv);root=a.output_dir.expanduser();out=a.destination or root/'fg_attempt_networks';cache=a.shot_cache or root/'shot_attempt_cache'
    paths=sorted((root/'full_roster_entropy_centrality/historical_full_rosters').glob('20*/*/passing_matrix.csv'))
    paths=[x for x in paths if (a.season=='ALL' or x.parent.parent.name==a.season) and (a.team.upper()=='ALL' or x.parent.name==a.team.upper())]
    if not paths: raise FileNotFoundError('No matching original full-roster passing networks')
    out.mkdir(parents=True,exist_ok=True);mapping={t['abbreviation']:t['id'] for t in teams.get_teams()};rows=[];manifest=[];loaded=None
    with PdfPages(out/'all_fg_attempt_networks.pdf') as pdf:
        for k,path in enumerate(paths,1):
            season,team=path.parent.parent.name,path.parent.name
            file=cache/f'{season}_shot_attempts.csv.gz'
            if loaded!=season:
                allshots=pd.read_csv(file);classify(allshots);loaded=season
            shots=allshots.loc[allshots.TEAM_ID.eq(mapping[team])]
            original,graph,audit=build(pd.read_csv(path,index_col=0),pd.read_csv(path.with_name('roster.csv')),shots)
            folder=out/season/team;folder.mkdir(parents=True,exist_ok=True)
            audit.to_csv(folder/'player_shot_counts.csv',index=False)
            pd.DataFrame([dict(source=u,target=v,**d) for u,v,d in graph.edges(data=True)]).to_csv(folder/'edges.csv',index=False)
            nx.write_graphml(graph,folder/'passing_and_shot_types.graphml')
            fig=figure(original,graph,audit,team,season);fig.savefig(folder/'network.png',dpi=125);fig.savefig(folder/'network.pdf');pdf.savefig(fig);plt.close(fig)
            rows.append(dict(season=season,team=team,passes=audit.outgoing_passes.sum(),fga=audit.fga.sum(),**{cat:int(audit[cat+'_attempts'].sum()) for cat in CATEGORIES},player_FGA_3PA_FGM_3PM_verified=True,original_passes_unchanged=True,scaled_shot_sum_equals_pass_sum=True))
            for source in [path,path.with_name('roster.csv'),file]: manifest.append(dict(season=season,team=team,path=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
            print(f'{k}/{len(paths)} {season} {team}: verified {len(shots)} FGA',flush=True)
    pd.DataFrame(rows).to_csv(out/'validation.csv',index=False);pd.DataFrame(manifest).to_csv(out/'source_manifest.csv',index=False)
    (out/'README.md').write_text(__doc__+'\n\nThree exhaustive FGA categories: 3PT; two-point layups/dunks identified by ACTION_TYPE; all remaining two-pointers (mid-range / other 2PT). Attempts include misses. Strict NBA Mid-Range counts are exported separately. Original player-to-player passes remain unchanged; Fruchterman–Reingold uses only passing edges for placement. All shot edges together are normalized to total passes. Raw counts remain in edges.csv. True isolates are omitted from drawings. No minutes cutoff. Missing shot data and mismatched FGA/3PA/FGM/3PM are errors, never replaced with zeros. Original NBA API ShotChartDetail, ContextMeasure=FGA, Regular Season; team-filtered totals verified per player. These are season aggregates, not recovered possession sequences. Prior commands and outputs are preserved.\n')
    return 0
if __name__=='__main__': raise SystemExit(main())
