"""Plot all 180 original / previous-season-only predicted / observed networks."""
from pathlib import Path
import argparse,json,inspect,hashlib
import numpy as np,pandas as pd,networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

CASES=[('2022-23','DAL'),('2020-21','WAS'),('2022-23','NYK'),('2025-26','DEN'),('2021-22','GSW'),('2025-26','TOR')]


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'nba_passing_output');p.add_argument('--destination',type=Path);p.add_argument('--model-dir',type=Path);a=p.parse_args();root=a.output_dir;model=a.model_dir or root.parent.parent/'prediction_model';out=a.destination or model/'all_networks';out.mkdir(parents=True,exist_ok=True);baseline=root/'all_season_hub_redistribution';data=pd.read_csv(model/'team_season_comparison.csv');meta=pd.read_csv(baseline/'all_team_season_entropy_and_ec.csv');rows=[]
 pdf=PdfPages(out/'all_180_team_networks.pdf');season_pdfs={}
 for season,team in sorted(zip(data.season,data.team)):
  z=data[data.season.eq(season)&data.team.eq(team)].iloc[0];m=meta[meta.season.eq(season)&meta.team.eq(team)].iloc[0];lookup=pd.read_csv(root/'player_names.csv');names=lookup[lookup.season.eq(season)].set_index('PLAYER_ID').PLAYER_NAME.to_dict();src=baseline/season/team;paths=[src/'original_evaluation.csv',model/season/team/'predicted_evaluation.csv',src/'actual_evaluation.csv'];frames=[pd.read_csv(f,index_col=0) for f in paths];ids=frames[0].index.astype(int).tolist()
  assert all(f.index.astype(int).tolist()==ids and f.columns.astype(int).tolist()==ids for f in frames)
  matrices=[f.to_numpy(float) for f in frames];P,B,A=matrices;den=m.normalization_log2
  def entropy(M):
   if M.sum()<=0:return np.nan
   prob=np.divide(M,M.sum(1)[:,None],out=np.zeros_like(M),where=M.sum(1)[:,None]>0);mask=M>0;return float(-(M[mask]/M.sum()*np.log2(prob[mask])).sum()/den)
  values=[entropy(M) for M in matrices];assert np.isclose(values[2],z.entropy_actual,equal_nan=True)
  U=sum(M/M.sum() for M in matrices if M.sum()>0);G=nx.from_numpy_array(U+U.T);G=G.subgraph(np.flatnonzero(U.sum(0)+U.sum(1)>0)).copy();opts=dict(seed=42,weight=None,iterations=300,k=1.5/np.sqrt(len(G)))
  if 'method' in inspect.signature(nx.spring_layout).parameters:opts['method']='force'
  pos=nx.spring_layout(G,**opts);coords=np.array(list(pos.values()));lo=coords.min(0)-.22;hi=coords.max(0)+.22;edge_scale=max((M/M.sum()).max() for M in matrices if M.sum()>0)
  fig,axes=plt.subplots(1,3,figsize=(25,10));middle='Predicted: previous-season role adjustments' if z.has_history else 'Predicted: original model fallback\nNo earlier saved season for training'
  titles=['Original: early hub-present games',middle,f'Actual: {int(m.later_hub_absent_games)} later hub-absent games']
  for ax,M,title,h in zip(axes,matrices,titles,values):
   if M.sum()<=0:
    ax.text(.5,.5,'No observed later absence comparison',ha='center',va='center',transform=ax.transAxes,fontsize=13);ax.set_title(title,fontsize=12);ax.axis('off');continue
   active=np.flatnonzero(M.sum(0)+M.sum(1)>0);H=nx.DiGraph();H.add_nodes_from(active);edges=[(i,j) for i in active for j in active if M[i,j]>0];H.add_edges_from(edges);shares=(M.sum(0)+M.sum(1))/(2*M.sum());sizes=[16000*shares[i] for i in H]
   nx.draw_networkx_edges(H,pos,width=[.1+5*(M[i,j]/M.sum())/edge_scale for i,j in edges],alpha=.3,edge_color='#37768e',arrows=True,arrowsize=9,node_size=sizes,connectionstyle='arc3,rad=.07',ax=ax)
   nx.draw_networkx_nodes(H,pos,node_size=sizes,node_color=['#e59d47' if ids[i]==m.hub_id else '#dcebef' for i in H],edgecolors='#536b76',ax=ax)
   nx.draw_networkx_labels(H,pos,labels={i:str(names.get(ids[i],ids[i])).replace(' ','\n',1) for i in H},font_size=7.5,ax=ax)
   ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[1],hi[1]);ax.axis('off');ax.set_title(title+f'\nNormalized outgoing entropy {h:.4f}',fontsize=12)
  fmt=lambda v:f'{v:.3f}' if np.isfinite(v) else 'unavailable'
  fig.suptitle(f'{season} {team} | passing hub: {m.hub_name}\nConnection error: original separate-role {fmt(z.edge_current)} | previous-season model {fmt(z.edge_learned)}',fontsize=17)
  note='Prediction learns from '+str(int(z.historical_team_seasons))+' comparisons from the immediately previous season, excluding this franchise.' if z.has_history else 'No earlier saved season: the middle panel retains the original separate-role prediction.'
  fig.text(.5,.035,note+'\nNode areas use each panel\'s own passing involvement. Fixed Fruchterman-Reingold positions aid comparison.\nOriginal and actual matrices are unchanged; entropy levels use the existing within-team evaluation reference.',ha='center',fontsize=11)
  fig.tight_layout(rect=[0,.12,1,.91]);file=out/season/f'{team}_three_networks.png';file.parent.mkdir(parents=True,exist_ok=True);fig.savefig(file,dpi=130);pdf.savefig(fig)
  if season not in season_pdfs:season_pdfs[season]=PdfPages(out/(season+'_all_30_teams.pdf'))
  season_pdfs[season].savefig(fig);plt.close(fig)
  rows.append(dict(season=season,team=team,hub_name=m.hub_name,has_history=bool(z.has_history),training_team_seasons=int(z.historical_team_seasons),entropy_original=values[0],entropy_prediction=values[1],entropy_actual=values[2],edge_current=z.edge_current,edge_learned=z.edge_learned,eligible=bool(z.eligible),observed=bool(z.observed),image=str(file.relative_to(out)),source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in paths}));print('Plotted',season,team,flush=True)
 pdf.close()
 for season_pdf in season_pdfs.values():season_pdf.close()
 pd.DataFrame([{k:v for k,v in r.items() if k!='source_sha256'} for r in rows]).to_csv(out/'all_team_network_summary.csv',index=False);(out/'plot_provenance.json').write_text(json.dumps(rows,indent=2));assert len(list(out.glob('*/*_three_networks.png')))==180

if __name__=='__main__':main()
