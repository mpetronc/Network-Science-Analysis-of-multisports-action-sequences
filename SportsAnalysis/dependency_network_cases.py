"""Outcome-selected network case illustrations, with fixed panels and explicit uncertainty."""
from pathlib import Path
import argparse,hashlib,json,inspect
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import networkx as nx
from scipy.stats import t


def entropy(A):
 r=A.sum(1);P=np.divide(A,r[:,None],out=np.zeros_like(A),where=r[:,None]>0);return float(-np.sum(A*np.log2(np.maximum(P,1e-300)))/A.sum()/np.log2(len(A)-1))
def shares(A):
 _,V=np.linalg.eigh(A+A.T);v=np.abs(V[:,-1]);return v/v.sum()
def holm(df,p='p',output='holm_p'):
 order=df[p].sort_values().index;df.loc[order,output]=np.minimum(1,np.maximum.accumulate([(len(order)-j)*df.loc[i,p] for j,i in enumerate(order)]))

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dependency-network-cases',action='store_true');p.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'nba_passing_output');p.add_argument('--destination',type=Path);a=p.parse_args(argv);root=a.output_dir;study=root/'star_dependency_study';out=a.destination or study/'network_case_studies';out.mkdir(parents=True,exist_ok=True)
 cases=pd.read_csv(study/'adaptation/limited_passing_adaptation.csv');perf=pd.read_csv(study/'new_team_season_data.csv');perf=perf[perf.policy.eq('passing_hub')].copy();cases=cases.merge(perf,on=['team','season','star_id','star_name'],validate='one_to_one');assert len(cases)==9
 for frame in [cases,perf]:frame['offense_change']=-frame.penalty;frame['change_ci_low']=-frame.ci_high;frame['change_ci_high']=-frame.ci_low
 # Selection is based on point-estimated performance ONLY, before comparing network changes.
 chosen=[];used=set()
 for label,subset in [('Largest estimated drops',cases.sort_values('offense_change').head(2)),('Largest estimated rises',cases[cases.offense_change>0].sort_values('offense_change',ascending=False).head(2))]:
  for team in subset.team:chosen.append((team,label));used.add(team)
 for team in cases[~cases.team.isin(used)].assign(distance=lambda d:d.offense_change.abs()).sort_values('distance').head(2).team:chosen.append((team,'Closest-to-zero estimates'))
 category=dict(chosen);cases['case_group']=cases.team.map(category).fillna('Other available network case');cases['core_case']=cases.team.isin([cases.sort_values('offense_change').iloc[0].team,cases.sort_values('offense_change').iloc[-1].team,cases.assign(distance=cases.offense_change.abs()).sort_values('distance').iloc[0].team])
 rankings=perf.sort_values('offense_change');rankings.to_csv(out/'all_92_performance_ranking.csv',index=False);stats=[];manifest=[];cache=study/'game_cache';players=pd.read_csv(cache/'2025-26_player_base.csv.gz',dtype={'GAME_ID':str});players.GAME_ID=players.GAME_ID.str.zfill(10);names=players.drop_duplicates('PLAYER_ID').set_index('PLAYER_ID').PLAYER_NAME.to_dict();data={}
 for z in cases.itertuples():
  folder=root/'2025-26/team_season_topics';fullfile=folder/'networks'/z.team/'passing_matrix.csv';presentfile=folder/'hub_absence_resilience/present_networks'/z.team/'passing_matrix.csv';rosterfile=root/'full_roster_entropy_centrality/historical_full_rosters/2025-26'/z.team/'roster.csv';full=pd.read_csv(fullfile,index_col=0);pres=pd.read_csv(presentfile,index_col=0);assert full.index.astype(str).tolist()==pres.index.astype(str).tolist() and full.columns.tolist()==pres.columns.tolist();ids=full.index.astype(int).tolist();P=pres.to_numpy(float);A=full.to_numpy(float)-P;assert A.min()>=-1e-8;A=np.maximum(A,0);star=ids.index(int(z.star_id));assert A[star].sum()+A[:,star].sum()==0;take=[i for i in range(len(ids)) if i!=star]
  while True:
   Pn=P[np.ix_(take,take)];An=A[np.ix_(take,take)];keep=(Pn.sum(1)>0)&(An.sum(1)>0)
   if keep.all():break
   take=[i for i,k in zip(take,keep) if k]
  samples=np.minimum(Pn.sum(1),An.sum(1));M=Pn*samples[:,None]/Pn.sum(1)[:,None];N=An*samples[:,None]/An.sum(1)[:,None];remaining=[ids[i] for i in take];assert len(remaining)==z.common_nonstar_players and np.allclose(M.sum(1),N.sum(1))
  pc,ac=shares(M),shares(N);incoming=(N.sum(0)-M.sum(0))/samples.sum();ecdelta=ac-pc
  roster=pd.read_csv(rosterfile);tid=int(roster.TEAM_ID.mode().iloc[0]);pt=players[players.TEAM_ABBREVIATION.eq(z.team)];presentgames=set(pt[pt.PLAYER_ID.eq(z.star_id)].GAME_ID);pg=pt[pt.GAME_ID.isin(presentgames)].groupby('PLAYER_ID').FGA.sum().reindex(remaining,fill_value=0);ag=pt[~pt.GAME_ID.isin(presentgames)].groupby('PLAYER_ID').FGA.sum().reindex(remaining,fill_value=0);fgadelta=ag/ag.sum()-pg/pg.sum();pm=pt[pt.GAME_ID.isin(presentgames)].groupby('PLAYER_ID').MIN.sum().reindex(remaining,fill_value=0);am=pt[~pt.GAME_ID.isin(presentgames)].groupby('PLAYER_ID').MIN.sum().reindex(remaining,fill_value=0);minutes_delta=am/am.sum()-pm/pm.sum();rate_pre=np.divide(36*pg.to_numpy(),pm.to_numpy(),out=np.full(len(pm),np.nan),where=pm.to_numpy()>0);rate_abs=np.divide(36*ag.to_numpy(),am.to_numpy(),out=np.full(len(am),np.nan),where=am.to_numpy()>0)
  # Formal individual estimates use their own within-unit design df, not 92 independent cases.
  gamedata=pd.read_csv(study/'passing_hub_primary_games.csv.gz',dtype={'GAME_ID':str});g=gamedata[(gamedata.season==z.season)&gamedata.team.eq(z.team)];X=np.column_stack([np.ones(len(g)),g.absent,g.game_number/82,g.home,g.b2b,g.rest,g.opp_b2b,(g.opp_def_pregame-110)/10,g.other_missing,g.opponent_missing]);df=len(g)-np.linalg.matrix_rank(X);pvalue=2*t.sf(abs(z.penalty/z.se),df)
  # Practical equivalence illustration, not a declaration that nonsignificance means stability.
  bound=2.;tost=max(t.sf((z.offense_change+bound)/z.se,df),t.cdf((z.offense_change-bound)/z.se,df))
  stats.append(dict(team=z.team,p_individual=pvalue,tost_p_within_2_points=tost,df=df,expected_matched_entropy_change=entropy(N)-entropy(M),expected_matched_EC_change=((ac@ac)-(pc@pc))/(1-1/len(pc)),present_nonstar_passing_leader=names.get(remaining[np.argmax(pc)],str(remaining[np.argmax(pc)])),absent_passing_leader=names.get(remaining[np.argmax(ac)],str(remaining[np.argmax(ac)])),present_nonstar_top_EC_share=float(pc.max()),absent_top_EC_share=float(ac.max()),absent_top2_EC_gap=float(np.sort(ac)[-1]-np.sort(ac)[-2]),present_core_shooting_HHI=float(np.sum((pg/pg.sum())**2)),absent_core_shooting_HHI=float(np.sum((ag/ag.sum())**2)),nonstar_FGA_change=float((ag/ag.sum()).max()-(pg/pg.sum()).max())))
  data[z.team]=(z,ids,P,A,remaining,M,N,pc,ac,incoming,fgadelta,minutes_delta)
  target=out/z.team;target.mkdir(exist_ok=True);pd.DataFrame(M,index=remaining,columns=remaining).to_csv(target/'present_matched_expected_passes.csv');pd.DataFrame(N,index=remaining,columns=remaining).to_csv(target/'absent_matched_expected_passes.csv');pd.DataFrame(dict(player_id=remaining,name=[names.get(i,str(i)) for i in remaining],present_EC_share=pc,absent_EC_share=ac,incoming_pass_share_change=incoming,shooting_share_change=fgadelta.to_numpy(),minutes_share_change=minutes_delta.to_numpy(),present_FGA_per36=rate_pre,absent_FGA_per36=rate_abs)).to_csv(target/'player_role_changes.csv',index=False)
  for file in [fullfile,presentfile,rosterfile]:manifest.append(dict(team=z.team,path=str(file),sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
 stats=pd.DataFrame(stats);holm(stats,'p_individual','holm_p_nine');holm(stats,'tost_p_within_2_points','holm_equivalence_p_nine');cases=cases.merge(stats,on='team',validate='one_to_one');cases.to_csv(out/'nine_network_case_metrics.csv',index=False);cases[cases.team.isin(category)].to_csv(out/'selected_six_cases.csv',index=False)
 with PdfPages(out/'network_case_comparisons.pdf') as pdf:
  fig,ax=plt.subplots(figsize=(11,7));ranked=cases.sort_values('offense_change')
  for i,z in enumerate(ranked.itertuples()):ax.errorbar(z.offense_change,i,xerr=[[z.offense_change-z.change_ci_low],[z.change_ci_high-z.offense_change]],fmt='o',color='#ad4935' if z.offense_change<0 else '#2c8691',capsize=3)
  ax.set_yticks(range(len(ranked)),[f'{z.team}: {z.star_name}' for z in ranked.itertuples()]);ax.axvline(0,c='grey',ls='--');ax.axvspan(-2,2,color='grey',alpha=.08,label='Illustrative ±2-point equivalence region');ax.set_xlabel('Estimated offense change when hub absent: points per 100 possessions');ax.set_title('Nine available paired-network cases | 2025–26\nNominal 95% individual intervals; rankings are uncertain');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/'nine_case_performance.png',dpi=150);pdf.savefig(fig);plt.close(fig)
  illustrations=chosen+[(team,'Other available network case') for team in cases.team if team not in category]
  for team,label in illustrations:
   z,ids,P,A,remaining,M,N,pc,ac,incoming,fgadelta,minutes_delta=data[team];row=cases[cases.team.eq(team)].iloc[0]
   # Shared positions from ALL original passing edges, never shooting edges.
   G=nx.from_numpy_array(P+A+(P+A).T);maximum=max((d['weight'] for _,_,d in G.edges(data=True)),default=1.)
   for _,_,d in G.edges(data=True):d['weight']=np.sqrt(d['weight']/maximum)
   opts=dict(seed=42,iterations=300,weight='weight',k=1.5/np.sqrt(len(ids)))
   if 'method' in inspect.signature(nx.spring_layout).parameters:opts['method']='force'
   positions=nx.spring_layout(G,**opts);pos={pid:positions[i] for i,pid in enumerate(ids)};fig=plt.figure(figsize=(26,15));grid=fig.add_gridspec(2,6,height_ratios=[1.5,1]);axes=[fig.add_subplot(grid[0,0:2]),fig.add_subplot(grid[0,2:4]),fig.add_subplot(grid[0,4:6])];commonmax=max(M.max(),N.max());involvement=(M.sum(0)+M.sum(1)+N.sum(0)+N.sum(1))/2;size={pid:220+1100*involvement[i]/max(involvement.max(),1) for i,pid in enumerate(remaining)};size[int(z.star_id)]=1600
   for ax,mat,nodes,title,scale in [(axes[0],P,ids,'Hub-present network | full roster',P.max()),(axes[1],M,remaining,'Present games | common nonhub panel, matched outgoing counts',commonmax),(axes[2],N,remaining,'Absent games | same common panel and outgoing counts',commonmax)]:
    H=nx.DiGraph();H.add_nodes_from(nodes)
    for i,j in zip(*np.nonzero(mat)):H.add_edge(nodes[i],nodes[j],weight=mat[i,j])
    visible=[pid for pid in nodes if H.degree(pid)>0];H=H.subgraph(visible);sizes=[size.get(pid,180) for pid in H];edges=list(H.edges)
    nx.draw_networkx_edges(H,pos,edgelist=edges,width=[.15+4*H[u][v]['weight']/max(scale,1) for u,v in edges],alpha=.33,arrowsize=10,edge_color='#397496',connectionstyle='arc3,rad=.08',node_size=sizes,ax=ax)
    nx.draw_networkx_nodes(H,pos,nodelist=list(H),node_size=sizes,node_color=['#d65a45' if pid==z.star_id else '#e8ce74' for pid in H],edgecolors='#756a4b',ax=ax)
    nx.draw_networkx_labels(H,pos,labels={pid:names.get(pid,str(pid)).replace(' ','\n',1) for pid in H},font_size=7,ax=ax);ax.set_title(title,fontsize=11);ax.set_xlim(-1.2,1.2);ax.set_ylim(-1.2,1.2);ax.axis('off')
   b1=fig.add_subplot(grid[1,0:3]);b2=fig.add_subplot(grid[1,3:6]);order=np.argsort(ac)[::-1][:12];yy=np.arange(len(order));labs=[names.get(remaining[i],str(remaining[i])) for i in order]
   b1.barh(yy+.18,pc[order]*100,height=.35,label='Present, hub removed',color='#3b789f');b1.barh(yy-.18,ac[order]*100,height=.35,label='Absent',color='#c58335');b1.set_yticks(yy,labs);b1.invert_yaxis();b1.set_xlabel('Share of nonhub passing eigenvector centrality (%)');b1.set_title('Passing influence after volume matching (top 12)');b1.legend(fontsize=8)
   diffs=fgadelta.to_numpy();order=np.argsort(np.abs(diffs))[::-1][:12];yy=np.arange(len(order));b2.barh(yy-.18,diffs[order]*100,height=.34,color=['#288b7a' if v>0 else '#c66a43' for v in diffs[order]],label='FGA share change');b2.barh(yy+.18,minutes_delta.to_numpy()[order]*100,height=.3,color='#b6b6b6',label='Minutes share change');b2.set_yticks(yy,[names.get(remaining[i],str(remaining[i])) for i in order]);b2.invert_yaxis();b2.axvline(0,c='grey',lw=.8);b2.set_xlabel('Change in remaining-player FGA or minutes share (percentage points)');b2.set_title('Shot-share changes alongside playing opportunities');b2.legend(fontsize=8)
   fig.suptitle(f'{team} | {label} | {z.star_name} absent\nEstimated offense change: {z.offense_change:+.2f} points/100 (95% CI {z.change_ci_low:+.2f} to {z.change_ci_high:+.2f})',fontsize=19)
   fig.text(.5,.025,f'Count-matched sample mean Δpassing entropy: {z.entropy_change:+.4f}; ΔEC concentration: {z.EC_concentration_change:+.4f}. Present/absent performance games: {z.present}/{z.absent}.\nPanels 2–3 use equal outgoing counts per remaining player and a shared edge-width scale; panel 1 has its own width scale. Player positions shared; EC excludes shooting.\nNetworks and shooting shares summarize full-season presence/absence contexts; performance estimates use the stricter study window after game 10 and trade censor.\nCases selected by performance estimates for illustration. Intervals are wide; these are not causal mechanisms, reliable rankings, or proof of equivalence. Red node marks the hub; node sizes fixed in matched panels. Bars show top 12 with full-panel denominators; true isolates hidden.',ha='center',fontsize=10)
   fig.tight_layout(rect=[0,.1,1,.92]);fig.savefig(out/team/'network_case.png',dpi=130);fig.savefig(out/team/'network_case.pdf');pdf.savefig(fig);plt.close(fig)
  fig,axes=plt.subplots(1,2,figsize=(14,6))
  for ax,x,title in [(axes[0],'entropy_change','Change in count-matched outgoing passing entropy'),(axes[1],'EC_concentration_change','Change in count-matched nonhub EC concentration')]:
   ax.errorbar(cases[x],cases.offense_change,yerr=[cases.offense_change-cases.change_ci_low,cases.change_ci_high-cases.offense_change],fmt='o',alpha=.6,capsize=2)
   for z in cases.itertuples():ax.annotate(z.team,(getattr(z,x),z.offense_change),fontsize=8,xytext=(3,3),textcoords='offset points')
   ax.axhline(0,c='grey',ls='--');ax.set_xlabel(title);ax.set_ylabel('Estimated offense change absent minus present');ax.set_title('All nine available cases | exploratory comparison')
  fig.suptitle('Do network changes reliably separate declines, rises and near-zero estimates?');fig.text(.5,.015,'Vertical intervals are uncertainty in the individual performance estimates. Nine available cases are not a representative or independent validation set.\nNo regression on the six outcome-selected illustrations is treated as evidence; network contexts and performance windows differ.',ha='center',fontsize=9);fig.tight_layout(rect=[0,.09,1,.93]);fig.savefig(out/'all_nine_changes.png',dpi=150);pdf.savefig(fig);plt.close(fig)
 for f in [study/'new_team_season_data.csv',study/'adaptation/limited_passing_adaptation.csv',cache/'2025-26_player_base.csv.gz']:manifest.append(dict(team='ALL',path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
 pd.DataFrame(manifest).to_csv(out/'source_manifest.csv',index=False)
 from dependency_network_cases_report import write_report
 write_report(out)
 print(cases[['team','case_group','offense_change','change_ci_low','change_ci_high','entropy_change','EC_concentration_change','present_nonstar_passing_leader','absent_passing_leader','holm_p_nine','holm_equivalence_p_nine']].sort_values('offense_change').to_string(index=False));return 0
if __name__=='__main__':raise SystemExit(main())
