"""All-team observed hub-present/absent passing networks with fixed Fruchterman–Reingold layouts."""
from pathlib import Path
import argparse,json,inspect,hashlib
import numpy as np,pandas as pd,networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--league-network-changes',action='store_true');p.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'nba_passing_output');p.add_argument('--data-dir',type=Path);p.add_argument('--destination',type=Path);a=p.parse_args(argv);root=a.output_dir;data=a.data_dir or root/'league_star_presence_entropy';out=a.destination or data/'network_changes';out.mkdir(parents=True,exist_ok=True);table=pd.read_csv(data/'all_30_team_entropy.csv').sort_values('ec_rank');assert len(table)==30
 namesframe=pd.read_csv(root/'2025-26/team_season_topics/player_games.csv');names=namesframe.drop_duplicates('PLAYER_ID').set_index('PLAYER_ID').PLAYER_NAME.to_dict();edges=[];nodes=[];manifest=[];summary=[]
 def read(f):
  manifest.append(dict(path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest()));return pd.read_csv(f,index_col=0)
 def shares(M):
  active=M.sum(0)+M.sum(1)>0;v=np.zeros(len(M))
  if active.sum()>1:
   _,eig=np.linalg.eigh((M+M.T)[np.ix_(active,active)]);e=np.abs(eig[:,-1]);v[active]=e/e.sum()
  return v
 with PdfPages(out/'all_30_team_network_changes.pdf') as pdf:
  for z in table.itertuples():
   full=read(root/'2025-26/team_season_topics/networks'/z.team/'passing_matrix.csv');ids=full.index.astype(int).tolist();assert ids==full.columns.astype(int).tolist();T=full.to_numpy(float);hub=ids.index(int(z.hub_id));available=z.absent_games>0
   if z.status=='complete_cached':frame=read(root/'2025-26/team_season_topics/hub_absence_resilience/present_networks'/z.team/'passing_matrix.csv');assert frame.index.astype(int).tolist()==ids;P=frame.to_numpy(float);A=T-P
   elif z.status=='complete_new':
    frame=read(data/'new_networks'/z.team/'present_passing_matrix.csv');other=read(data/'new_networks'/z.team/'absent_passing_matrix.csv');assert frame.index.astype(int).tolist()==ids and other.index.astype(int).tolist()==ids;P=frame.to_numpy(float);A=other.to_numpy(float)
   elif z.status=='no_observed_absence':P=T.copy();A=np.zeros_like(T)
   else:raise ValueError('Unverified team '+z.team)
   assert (P>=0).all() and A.min()>=-1e-8 and np.allclose(P+A,T);A=np.maximum(A,0);assert A[hub].sum()+A[:,hub].sum()==0
   U=1000*P/P.sum();V=1000*A/A.sum() if available else None;change=V-U if available else None
   folder=out/z.team;folder.mkdir(exist_ok=True);pd.DataFrame(U,index=ids,columns=ids).to_csv(folder/'present_per_1000_passes.csv')
   if available:pd.DataFrame(V,index=ids,columns=ids).to_csv(folder/'absent_per_1000_passes.csv');pd.DataFrame(change,index=ids,columns=ids).to_csv(folder/'change_per_1000_passes.csv')
   # One unweighted pooled-topology layout and one pairwise visual scale for both contexts.
   layoutM=(U+V)/2 if available else U;G=nx.from_numpy_array(layoutM+layoutM.T);active=np.flatnonzero(T.sum(0)+T.sum(1)>0);G=G.subgraph(active).copy();opts=dict(seed=42,weight=None,iterations=500,k=1.5/np.sqrt(len(G)))
   if 'method' in inspect.signature(nx.spring_layout).parameters:opts['method']='force'
   pos=nx.spring_layout(G,**opts);pd.DataFrame([dict(player_id=ids[i],player_name=names.get(ids[i],str(ids[i])),x=float(pos[i][0]),y=float(pos[i][1])) for i in G]).to_csv(folder/'fixed_layout.csv',index=False)
   scale=max(U.max(),V.max() if available else 0);fig,axes=plt.subplots(1,3,figsize=(24,9));limits=np.array(list(pos.values()));xmin,ymin=limits.min(0)-.3;xmax,ymax=limits.max(0)+.3
   def draw(ax,M,title,difference=False):
    if M is None:ax.text(.5,.5,'No observed hub-absence games\nComparison unavailable',ha='center',va='center',transform=ax.transAxes,fontsize=16);ax.set_title(title);ax.axis('off');return
    present=np.flatnonzero(M.sum(0)+M.sum(1)>0) if not difference else np.flatnonzero(T.sum(0)+T.sum(1)>0);H=nx.DiGraph();H.add_nodes_from(present)
    pair=[(i,j) for i in range(len(ids)) for j in range(len(ids)) if i!=j and (abs(M[i,j])>1e-10 if difference else M[i,j]>0)]
    if difference:pair=sorted(pair,key=lambda ij:abs(M[ij]),reverse=True)[:40]
    for i,j in pair:H.add_edge(i,j)
    involvement=(U.sum(0)+U.sum(1)+(V.sum(0)+V.sum(1) if available else U.sum(0)+U.sum(1)))/2;sizes=[160+1400*involvement[i]/max(involvement.max(),1) for i in H];den=max(abs(M[i,j]) for i,j in pair) if difference and pair else scale
    nx.draw_networkx_edges(H,pos,edgelist=pair,width=[.1+5*abs(M[i,j])/den for i,j in pair],edge_color=[('#24865c' if M[i,j]>0 else '#be4650') if difference else '#447f9d' for i,j in pair],alpha=.65 if difference else .25,arrows=True,arrowsize=9,node_size=sizes,connectionstyle='arc3,rad=.08',ax=ax)
    nx.draw_networkx_nodes(H,pos,node_size=sizes,node_color=['#e79546' if i==hub else '#d8e8ed' for i in H],edgecolors='#536b76',ax=ax);nx.draw_networkx_labels(H,pos,labels={i:str(names.get(ids[i],ids[i])).replace(' ','\n',1) for i in H},font_size=6.5,ax=ax);ax.set_xlim(xmin,xmax);ax.set_ylim(ymin,ymax);ax.set_title(title,fontsize=13);ax.axis('off')
   draw(axes[0],U,f'Hub present: {z.present_games} games\nEntropy {z.with_hub_entropy:.4f}');draw(axes[1],V,f'Hub absent: {z.absent_games} games'+(f'\nEntropy {z.without_hub_entropy:.4f}' if available else ''));draw(axes[2],change,'Largest 40 directed edge-share changes\nGreen: increase | Red: decrease',True)
   flag=' | Fewer than 10 absence games: excluded from regressions' if 0<z.absent_games<10 else '';fig.suptitle(f'#{z.ec_rank} {z.team} | Hub: {z.hub_name} | EC concentration index {z.passing_ec_concentration:.4f}\n2025–26 observed passing networks'+flag,fontsize=17);fig.text(.5,.025,'Widths: passes per 1,000 team passes; present/absent panels share a scale. Difference panel has its own scale. Orange node = hub.\nFixed Fruchterman–Reingold positions; node sizes held constant within team; context isolates omitted. Layout distances have no court meaning.\nDifference graph includes mechanical loss of hub edges. Full directed changes exported; these are pooled game contexts, not matched lineups or causal effects.',ha='center',fontsize=11);fig.tight_layout(rect=[0,.12,1,.9]);fig.savefig(folder/f'{z.team}_network_changes.png',dpi=150);pdf.savefig(fig);plt.close(fig)
   ep=shares(P);ea=shares(A) if available else np.full(len(ids),np.nan)
   for i,pid in enumerate(ids):nodes.append(dict(team=z.team,player_id=pid,player_name=names.get(pid,str(pid)),is_hub=i==hub,present_ec_share=ep[i],absent_ec_share=ea[i],change_ec_share=ea[i]-ep[i]))
   if available:
    for i in range(len(ids)):
     for j in range(len(ids)):
      if U[i,j]>0 or V[i,j]>0:edges.append(dict(team=z.team,passer_id=ids[i],passer=names.get(ids[i],str(ids[i])),receiver_id=ids[j],receiver=names.get(ids[j],str(ids[j])),touches_hub=i==hub or j==hub,present_per_1000=U[i,j],absent_per_1000=V[i,j],change_per_1000=change[i,j]))
   summary.append(dict(team=z.team,ec_rank=z.ec_rank,comparison_available=available,present_games=z.present_games,absent_games=z.absent_games));print('Saved',z.team,flush=True)
 pd.DataFrame(edges).to_csv(out/'all_directed_edge_changes.csv',index=False);pd.DataFrame(nodes).to_csv(out/'player_ec_changes.csv',index=False);pd.DataFrame(summary).to_csv(out/'coverage.csv',index=False);pd.DataFrame(manifest).to_csv(out/'source_manifest.csv',index=False)
 (out/'README.md').write_text('All 30 teams, 2025–26, ordered by full-season passing-only EC concentration index. Each page shows full observed hub-present, actual hub-absent, and the largest 40 absolute directed edge-share changes. The absent graph is not obtained by simply deleting the hub: it uses recorded games without that player.\n\nAll positive passing edges appear in context panels; isolates are omitted separately in each panel. No minutes cutoff. Weights are per 1,000 team passes, so unequal sample totals do not directly set widths; this does not control player availability, minutes, opponents or sampling noise. Node size is pooled in/out pass involvement, held constant within each team; it is not EC. Orange node identifies hub. All EC calculations use passing only; the player table includes per-context L1 EC shares, not the team concentration index.\n\nPositions use the same unweighted pooled-topology Fruchterman–Reingold layout (unweighted placement improves label separation; widths still encode pass shares) per team across all three panels. Present and absent edge widths share one scale; difference widths use a separate scale. Green means an edge gained share, red means lost share. Hub-edge losses are included and are mechanically expected, not evidence of tactical change. Raw normalized entropy is displayed; equal-pass-count sampling is used in the separate regression study.\n\n27 actual comparisons, with all 30 listed. LAC/NOP/WAS have no actual absence games and are marked unavailable. BOS/HOU/MIL/MIN/NYK/PHX remain illustrated but are excluded from the 21-team regressions. Full tables and matrix exports retain all changes, including those outside the 40 largest displayed edges.\n')
 return 0
if __name__=='__main__':raise SystemExit(main())
