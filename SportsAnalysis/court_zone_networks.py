"""Original whole-season passing networks with NBA court-zone shot sinks."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Circle,Rectangle,Arc
from nba_api.stats.static import teams
import networkx as nx
from basket_networks import build_networks,player_positions,BASKET

ZONES={
 'RA':('Restricted area',(0,0)),
 'PAINT_L':('Paint · left',(-55,60)), 'PAINT_C':('Paint · center',(0,115)), 'PAINT_R':('Paint · right',(55,60)),
 'MID_L':('Mid-range · left',(-157,45)), 'MID_LC':('Mid-range · left center',(-130,155)), 'MID_C':('Mid-range · center',(0,205)), 'MID_RC':('Mid-range · right center',(130,155)), 'MID_R':('Mid-range · right',(157,45)),
 'CORNER_L':('Left corner 3',(-223,-18)), 'ARC_L':('Above-break 3 · left',(-190,245)), 'ARC_C':('Above-break 3 · center',(0,320)), 'ARC_R':('Above-break 3 · right',(190,245)), 'CORNER_R':('Right corner 3',(223,-18)),
 'BACKCOURT':('Backcourt / long heaves',(0,410))}
AREA={'Left Side(L)':'L','Left Side Center(LC)':'LC','Center(C)':'C','Right Side Center(RC)':'RC','Right Side(R)':'R'}

def classify_zones(shots):
 required=['SHOT_ZONE_BASIC','SHOT_ZONE_AREA','SHOT_ATTEMPTED_FLAG','SHOT_MADE_FLAG','SHOT_TYPE']
 if shots[required].isna().any().any() or not shots.SHOT_ATTEMPTED_FLAG.eq(1).all():raise ValueError('Missing shot zones or non-attempt events')
 if shots.duplicated(['GAME_ID','GAME_EVENT_ID']).any():raise ValueError('Duplicate shot events')
 mapping={('Restricted Area','Center(C)'):'RA',('Left Corner 3','Left Side(L)'):'CORNER_L',('Right Corner 3','Right Side(R)'):'CORNER_R',('Backcourt','Back Court(BC)'):'BACKCOURT',('Above the Break 3','Back Court(BC)'):'BACKCOURT'}
 for area,code in AREA.items():
  mapping[('Mid-Range',area)]='MID_'+code
 for area,code in [('Left Side(L)','L'),('Center(C)','C'),('Right Side(R)','R')]:mapping[('In The Paint (Non-RA)',area)]='PAINT_'+code
 for area,code in [('Left Side Center(LC)','L'),('Center(C)','C'),('Right Side Center(RC)','R')]:mapping[('Above the Break 3',area)]='ARC_'+code
 keys=list(zip(shots.SHOT_ZONE_BASIC,shots.SHOT_ZONE_AREA));unknown=set(keys)-set(mapping)
 if unknown:raise ValueError(f'Unknown official shot-zone pair: {unknown}')
 return pd.Series([mapping[k] for k in keys],index=shots.index)

def build(matrix,roster,shots):
 original,_,audit=build_networks(matrix,roster);graph=original.copy();shots=shots.copy();shots['zone']=classify_zones(shots)
 ids=list(original);r=roster.set_index('PLAYER_ID').loc[ids]
 if not set(shots.PLAYER_ID)<=set(ids):raise ValueError('Unknown shooter')
 counts=shots.groupby(['PLAYER_ID','zone']).size().unstack(fill_value=0).reindex(index=ids,columns=ZONES,fill_value=0).fillna(0)
 makes=shots.loc[shots.SHOT_MADE_FLAG.eq(1)].groupby(['PLAYER_ID','zone']).size().unstack(fill_value=0).reindex(index=ids,columns=ZONES,fill_value=0).fillna(0)
 for field,mask in [('FGA',np.ones(len(shots),dtype=bool)),('FGM',shots.SHOT_MADE_FLAG.eq(1)),('FG3A',shots.SHOT_TYPE.eq('3PT Field Goal')),('FG3M',shots.SHOT_TYPE.eq('3PT Field Goal')&shots.SHOT_MADE_FLAG.eq(1))]:
  value=shots.loc[mask].groupby('PLAYER_ID').size().reindex(ids,fill_value=0)
  if not np.array_equal(value.to_numpy(),r[field].to_numpy()):raise ValueError(f'Player {field} mismatch')
 for zone,(name,_) in ZONES.items():
  graph.add_node(zone,name=name,node_type='shot_zone')
  for pid,n in counts[zone].items():
   if n:graph.add_edge(pid,zone,weight=float(n),raw_count=int(n),made=int(makes.loc[pid,zone]),edge_type='shot_attempt')
 graph.graph.update(total_passes=float(matrix.to_numpy().sum()),total_fga=float(counts.to_numpy().sum()),shot_weighting='Raw FGA counts, no pass-to-shot rescaling',ec_source='Original passing edges only')
 assert all(graph[u][v]==a for u,v,a in original.edges(data=True))
 assert all(graph.out_degree(z)==0 for z in ZONES)
 assert sum(a['raw_count'] for _,_,a in graph.edges(data=True) if a['edge_type']=='shot_attempt')==r.FGA.sum()
 return original,graph,counts,makes,audit

def court(ax,counts,makes):
 ax.set_aspect('equal');ax.set_xlim(-275,275);ax.set_ylim(-70,450);ax.set_facecolor('#fcfaf3')
 for patch in [Rectangle((-250,-52.5),500,470,fill=False),Rectangle((-80,-52.5),160,190,fill=False),Circle((0,137.5),60,fill=False),Circle((0,0),7.5,fill=False),Arc((0,0),80,80,theta1=0,theta2=180),Arc((0,0),475,475,theta1=22,theta2=158)]:
  patch.set_edgecolor('#a7a7a7');patch.set_linewidth(1);ax.add_patch(patch)
 ax.plot([-220,-220],[-52.5,92.5],c='#aaa',lw=1);ax.plot([220,220],[-52.5,92.5],c='#aaa',lw=1);ax.plot([-30,30],[-12.5,-12.5],c='#888',lw=2)
 for z,(name,pos) in ZONES.items():
  f=int(counts[z].sum());m=int(makes[z].sum());pct=m/f if f else np.nan;color=plt.cm.RdYlGn(np.clip((pct-.25)/.45,0,1)) if f else '#ddd'
  short=name.replace('Above-break 3 · ','3 · ').replace('Mid-range · ','Mid · ').replace('Restricted area','At rim').replace('Backcourt / long heaves','Backcourt')
  ax.text(*pos,f'{short}\n{m:,}/{f:,}\n{pct:.1%}' if f else short+'\n0 attempts',ha='center',va='center',fontsize=7,bbox=dict(boxstyle='round,pad=.3',fc=color,ec='white',alpha=.93))
 ax.set_title('Court-zone attempts and FG%\nOfficial NBA categories; schematic label positions',fontsize=13,pad=15);ax.axis('off')

def figure(original,graph,counts,makes,audit,team,season):
 fig,(ax,zoneax)=plt.subplots(1,2,figsize=(24,14),gridspec_kw={'width_ratios':[2.3,1]})
 pos=player_positions(original,graph);pos.pop(BASKET,None);activezones=[z for z in ZONES if graph.in_degree(z)>0]
 for z,y in zip(activezones,np.linspace(1.2,-1.2,len(activezones))):pos[z]=np.array([1.8,y])
 players=[n for n in original if graph.degree(n)>0];H=graph.subgraph(players+activezones);activity=audit.set_index('player_id').eval('incoming_passes+outgoing_passes');peak=max(activity.max(),1);sizes={n:250+1300*activity.loc[n]/peak for n in players};sizes.update({z:500 for z in activezones});maximum=max(a['weight'] for _,_,a in H.edges(data=True))
 colors={z:plt.cm.tab20(i/len(ZONES)) for i,z in enumerate(ZONES)}
 for kind in ['pass']+activezones:
  edges=[(u,v) for u,v,a in H.edges(data=True) if (a['edge_type']=='pass' if kind=='pass' else v==kind)]
  if not edges:continue
  nx.draw_networkx_edges(H,pos,edgelist=edges,edge_color='#356c9b' if kind=='pass' else [colors[kind]],alpha=.25 if kind=='pass' else .62,width=[.18+4.5*H[u][v]['weight']/maximum for u,v in edges],node_size=[sizes[n] for n in H],arrowsize=10,connectionstyle='arc3,rad=.06',ax=ax)
 nx.draw_networkx_nodes(H,pos,nodelist=players,node_size=[sizes[n] for n in players],node_color='#edcf76',edgecolors='#756332',ax=ax)
 nx.draw_networkx_labels(H,pos,labels={n:H.nodes[n]['name'].replace(' ','\n',1) for n in players},font_size=8,ax=ax)
 for z in activezones:ax.text(*pos[z],ZONES[z][0],fontsize=9,va='center',ha='left',bbox=dict(boxstyle='round,pad=.3',fc='white',ec=colors[z]),zorder=10)
 ax.set_xlim(-1.3,2.75);ax.set_ylim(-1.4,1.4);ax.set_title('Original player passing network + court-zone shot end nodes\nFruchterman–Reingold layout from passing edges only',fontsize=15,pad=20);ax.axis('off');court(zoneax,counts,makes)
 fig.suptitle(f'{team} | {season} whole regular season | 14 half-court zones + backcourt',fontsize=21)
 P=graph.graph['total_passes'];F=graph.graph['total_fga'];fig.text(.5,.025,f'{P:,.0f} passes | {F:,.0f} field-goal attempts (makes + misses) | One raw count per pass or FGA; no 50/50 shooting inflation.\nBlue edges: original passes. Colored edges: player → official shot zone. Widths share one count scale; player size uses passing involvement.\nEC uses player-to-player passing edges ONLY. True isolates hidden; all counts retained. Season aggregates do not reconstruct pass-to-shot sequences.\nCourt colors show raw FG% (red ≈25%, green ≈70%); zone labels mark categories, not exact boundaries. Backcourt label is schematic, not a shot-coordinate location.',ha='center',fontsize=10)
 fig.tight_layout(rect=[0,.1,1,.95]);return fig

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--court-zone-networks',action='store_true');p.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'nba_passing_output');p.add_argument('--destination',type=Path);p.add_argument('--season',default='ALL');p.add_argument('--team',default='ALL');a=p.parse_args(argv);root=a.output_dir;out=a.destination or root/'court_zone_networks';out.mkdir(parents=True,exist_ok=True)
 files=sorted((root/'full_roster_entropy_centrality/historical_full_rosters').glob('20*/*/passing_matrix.csv'));files=[f for f in files if (a.season=='ALL' or f.parent.parent.name==a.season) and (a.team.upper()=='ALL' or f.parent.name==a.team.upper())]
 if not files:raise FileNotFoundError('No matching full-season networks')
 mapping={t['abbreviation']:t['id'] for t in teams.get_teams()};loaded=None;rows=[];manifest=[]
 with PdfPages(out/'all_court_zone_networks.pdf') as pdf:
  for k,f in enumerate(files,1):
   season,team=f.parent.parent.name,f.parent.name;cache=root/'shot_attempt_cache'/f'{season}_shot_attempts.csv.gz'
   if loaded!=season:shots=pd.read_csv(cache);classify_zones(shots);loaded=season
   teamshots=shots.loc[shots.TEAM_ID.eq(mapping[team])];original,graph,counts,makes,audit=build(pd.read_csv(f,index_col=0),pd.read_csv(f.with_name('roster.csv')),teamshots);folder=out/season/team;folder.mkdir(parents=True,exist_ok=True)
   counts.to_csv(folder/'player_zone_attempts.csv');makes.to_csv(folder/'player_zone_makes.csv');pd.DataFrame([dict(zone=z,name=ZONES[z][0],fga=int(counts[z].sum()),fgm=int(makes[z].sum())) for z in ZONES]).to_csv(folder/'zone_totals.csv',index=False)
   pd.DataFrame([dict(source=u,target=v,**d) for u,v,d in graph.edges(data=True)]).to_csv(folder/'edges.csv',index=False);nx.write_graphml(graph,folder/'network.graphml')
   fig=figure(original,graph,counts,makes,audit,team,season);fig.savefig(folder/'network.png',dpi=120);fig.savefig(folder/'network.pdf');pdf.savefig(fig);plt.close(fig)
   mapfig,mapax=plt.subplots(figsize=(9,10));court(mapax,counts,makes);mapfig.suptitle(f'{team} | {season} regular season',fontsize=17);mapfig.text(.5,.025,'Makes / attempts and FG% by official NBA zone. Schematic label positions; colors show raw FG%.',ha='center',fontsize=8);mapfig.tight_layout(rect=[0,.05,1,.95]);mapfig.savefig(folder/'court_zone_map.png',dpi=160);mapfig.savefig(folder/'court_zone_map.pdf');plt.close(mapfig)
   rows.append(dict(team=team,season=season,passes=graph.graph['total_passes'],fga=graph.graph['total_fga'],active_zones=int((counts.sum(axis=0)>0).sum()),passing_edges_unchanged=True,player_FGA_3PA_FGM_3PM_match=True,shot_edges_are_raw_counts=True))
   for file in [f,f.with_name('roster.csv'),cache]:manifest.append(dict(team=team,season=season,path=str(file),sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
   print(f'{k}/{len(files)} {season} {team}: {graph.graph["total_fga"]:,.0f} FGA',flush=True)
 pd.DataFrame(rows).to_csv(out/'validation.csv',index=False);pd.DataFrame(manifest).to_csv(out/'source_manifest.csv',index=False);(out/'zone_definitions.json').write_text(json.dumps({z:dict(label=name,schematic_label_coordinates=pos) for z,(name,pos) in ZONES.items()},indent=2))
 (out/'README.md').write_text('''Court-zone sinks replace broad shot-type endpoints in a separate output folder, preserving prior outputs. Uses official NBA ShotChartDetail SHOT_ZONE_BASIC + SHOT_ZONE_AREA, not inferred action descriptions. 14 half-court zones: restricted area; paint left/center/right; five mid-range areas; two corners and three above-break areas. An extra backcourt/heave category retains all attempts, including Above the Break 3 records with Back Court area. Court-label positions are schematic; boundaries are not asserted to match the reference image exactly. Exact LOC_X/LOC_Y are available in source caches but no custom boundaries are inferred. No inbound/rebound/turnover/foul nodes. Raw attempt counts include misses; raw edge weights equal counts and shooting is NOT normalized to total passes. Every player FGA/FGM/3PA/3PM checked against original roster totals. Original passes preserved; FR positions and any EC use only original passing edges. Sinks have no outgoing edges or self-loops. Isolates hidden only in drawings. Court panel shows raw made/attempted and FG%, not league-relative efficiency. Counts have different uncertainty. Season aggregates do not establish observed pass-to-shot sequences.\n''');return 0
if __name__=='__main__':raise SystemExit(main())
