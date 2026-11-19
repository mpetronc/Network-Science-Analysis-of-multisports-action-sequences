"""All-team prior-on/off-selected player played/missed passing networks with fixed Fruchterman–Reingold layouts."""
from pathlib import Path
import argparse,json,inspect,hashlib
import numpy as np,pandas as pd,networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--onoff-star-networks',action='store_true');p.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'nba_passing_output');p.add_argument('--data-dir',type=Path);p.add_argument('--destination',type=Path);a=p.parse_args(argv);root=a.output_dir;data=a.data_dir or root/'onoff_dependency';out=a.destination or data/'network_changes';out.mkdir(parents=True,exist_ok=True);table=pd.read_csv(data/'network_metrics.csv').sort_values('ec_rank');assert len(table)==30
 namesframe=pd.read_csv(root/'2025-26/team_season_topics/player_games.csv');names=namesframe.drop_duplicates('PLAYER_ID').set_index('PLAYER_ID').PLAYER_NAME.to_dict();edges=[];nodes=[];manifest=[];summary=[]
 def read(f):
  manifest.append(dict(path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest()));return pd.read_csv(f,index_col=0)
 def shares(M):
  active=M.sum(0)+M.sum(1)>0;v=np.zeros(len(M))
  if active.sum()>1:
   _,eig=np.linalg.eigh((M+M.T)[np.ix_(active,active)]);e=np.abs(eig[:,-1]);v[active]=e/e.sum()
  return v
 with PdfPages(out/'onoff_selected_star_networks.pdf') as pdf:
  for z in table.itertuples():
   full=read(data/'networks'/z.team/'present_passing_matrix.csv');other=read(data/'networks'/z.team/'absent_passing_matrix.csv');ids=full.index.astype(int).tolist();assert ids==full.columns.astype(int).tolist() and other.index.astype(int).tolist()==ids;P=full.to_numpy(float);A=other.to_numpy(float);T=P+A;hub=ids.index(int(z.hub_id));available=z.absent_games>0
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
    if M is None:ax.text(.5,.5,'No observed selected-player absence games\nComparison unavailable',ha='center',va='center',transform=ax.transAxes,fontsize=16);ax.set_title(title);ax.axis('off');return
    present=np.flatnonzero(M.sum(0)+M.sum(1)>0) if not difference else np.flatnonzero(T.sum(0)+T.sum(1)>0);H=nx.DiGraph();H.add_nodes_from(present)
    pair=[(i,j) for i in range(len(ids)) for j in range(len(ids)) if i!=j and (abs(M[i,j])>1e-10 if difference else M[i,j]>0)]
    if difference:pair=sorted(pair,key=lambda ij:abs(M[ij]),reverse=True)[:40]
    for i,j in pair:H.add_edge(i,j)
    involvement=(U.sum(0)+U.sum(1)+(V.sum(0)+V.sum(1) if available else U.sum(0)+U.sum(1)))/2;sizes=[160+1400*involvement[i]/max(involvement.max(),1) for i in H];den=max(abs(M[i,j]) for i,j in pair) if difference and pair else scale
    nx.draw_networkx_edges(H,pos,edgelist=pair,width=[.1+5*abs(M[i,j])/den for i,j in pair],edge_color=[('#24865c' if M[i,j]>0 else '#be4650') if difference else '#447f9d' for i,j in pair],alpha=.65 if difference else .25,arrows=True,arrowsize=9,node_size=sizes,connectionstyle='arc3,rad=.08',ax=ax)
    nx.draw_networkx_nodes(H,pos,node_size=sizes,node_color=['#e79546' if i==hub else '#d8e8ed' for i in H],edgecolors='#536b76',ax=ax);nx.draw_networkx_labels(H,pos,labels={i:str(names.get(ids[i],ids[i])).replace(' ','\n',1) for i in H},font_size=6.5,ax=ax);ax.set_xlim(xmin,xmax);ax.set_ylim(ymin,ymax);ax.set_title(title,fontsize=13);ax.axis('off')
   draw(axes[0],U,f'Star played: {z.present_games} games\nEntropy {z.with_hub_entropy:.4f}');draw(axes[1],V,f'Star missed: {z.absent_games} games'+(f'\nEntropy {z.without_hub_entropy:.4f}' if available else ''));draw(axes[2],change,'Largest 40 directed edge-share changes\nGreen: increase | Red: decrease',True)
   flag=' | Not eligible for regression (need ≥15 played and ≥10 missed games)' if z.present_games<15 or z.absent_games<10 else '';fig.suptitle(f'#{z.ec_rank} {z.team} | Selected player: {z.hub_name} | Prior on/off ORtg gap {z.prior_on_off_gap:+.1f}\n2025–26 observed passing networks | prior-season on/off selection'+flag,fontsize=17);fig.text(.5,.025,'Widths: passes per 1,000 team passes; present/absent panels share a scale. Difference panel has its own scale. Orange node = selected player.\nFixed Fruchterman–Reingold positions; node sizes held constant within team; context isolates omitted. Layout distances have no court meaning.\nDifference graph includes mechanical loss of selected-player edges. Full directed changes exported; these are pooled game contexts, not matched lineups or causal effects.',ha='center',fontsize=11);fig.tight_layout(rect=[0,.12,1,.9]);fig.savefig(folder/f'{z.team}_network_changes.png',dpi=150);pdf.savefig(fig);plt.close(fig)
   ep=shares(P);ea=shares(A) if available else np.full(len(ids),np.nan)
   for i,pid in enumerate(ids):nodes.append(dict(team=z.team,player_id=pid,player_name=names.get(pid,str(pid)),is_hub=i==hub,present_ec_share=ep[i],absent_ec_share=ea[i],change_ec_share=ea[i]-ep[i]))
   if available:
    for i in range(len(ids)):
     for j in range(len(ids)):
      if U[i,j]>0 or V[i,j]>0:edges.append(dict(team=z.team,passer_id=ids[i],passer=names.get(ids[i],str(ids[i])),receiver_id=ids[j],receiver=names.get(ids[j],str(ids[j])),touches_hub=i==hub or j==hub,present_per_1000=U[i,j],absent_per_1000=V[i,j],change_per_1000=change[i,j]))
   summary.append(dict(team=z.team,ec_rank=z.ec_rank,comparison_available=available,present_games=z.present_games,absent_games=z.absent_games));print('Saved',z.team,flush=True)
 pd.DataFrame(edges).to_csv(out/'all_directed_edge_changes.csv',index=False);pd.DataFrame(nodes).to_csv(out/'player_ec_changes.csv',index=False);pd.DataFrame(summary).to_csv(out/'coverage.csv',index=False);pd.DataFrame(manifest).to_csv(out/'source_manifest.csv',index=False)
 (out/'README.md').write_text('Three panels: actual games played, actual games missed, and largest 40 directed passing-share changes. Selected players are previous-season on/off offensive-rating leaders among eligible retained players. All 30 teams appear in dependency-rank order. Per-team fixed Fruchterman–Reingold layouts; positive context edges shown, isolates omitted. Edge widths are passes per 1,000 team passes. Difference panel uses its own scale. Node size is pooled passing involvement, not EC; orange identifies the selected player. Positive share changes green, negative red. Mechanical losses of star edges are included. Full edge changes exported. Trade-censored teams use a shorter observation window. These are full-game played/missed networks, not passing during on/off-court stints. See the parent METHOD_AND_RESULTS.md for definitions, qualification and limitations.\n')
 return 0
if __name__=='__main__':raise SystemExit(main())
