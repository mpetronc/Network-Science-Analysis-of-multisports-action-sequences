"""Infer conservative passing sequences from SportVU tracking, not annotated pass events.
Data source: sealneaward/nba-movement-data (original data credited to neilmj).
Frames are deduplicated by period and tracking timestamp before analysis.
Run --help for reusable game-directory inputs. No repository setup script is executed.
"""
from pathlib import Path
import json,numpy as np,hashlib,collections,time

def prepare(path,out):
 stamp=hashlib.sha256(path.read_bytes()).hexdigest();cache=out/(path.stem+'_frames.npz');meta=out/(path.stem+'_metadata.json')
 if cache.exists() and meta.exists() and json.loads(meta.read_text()).get('source_sha256')==stamp:
  info=json.loads(meta.read_text());info['source']=str(path.resolve());meta.write_text(json.dumps(info,indent=2));return
 d=json.loads(path.read_text());rosters={};teams={};unique={};raw=bad=duplicates=conflicts=0;event_ids={};conflict_keys=set()
 for ev in d['events']:
  for side in ['home','visitor']:
   team=ev[side];tid=int(team['teamid']);teams[tid]={k:v for k,v in team.items() if k!='players'}
   for p in team['players']:rosters[int(p['playerid'])]=dict(p,team_id=tid,name=p['firstname']+' '+p['lastname'])
  for m in ev['moments']:
   raw+=1
   try:
    q,ts,clock,shot=int(m[0]),int(m[1]),float(m[2]),m[3];ball=[x for x in m[5] if int(x[1])==-1];players=sorted([x for x in m[5] if int(x[1])!=-1],key=lambda x:int(x[1]))
    if len(ball)!=1 or len(players)!=10 or len(set(int(x[1]) for x in players))!=10:bad+=1;continue
    tids=[int(x[0]) for x in players];cts=collections.Counter(tids)
    if len(cts)!=2 or sorted(cts.values())!=[5,5]:bad+=1;continue
    a=np.array(ball[0][2:5]+[float(v) for p in players for v in p[2:4]],float)
    if not np.isfinite(a).all() or q<1 or clock<0 or clock>(720 if q<=4 else 300):bad+=1;continue
    key=(q,ts);rec=(clock,float(shot) if shot is not None else np.nan,tuple(int(x[1]) for x in players),tuple(tids),a)
    if key in unique:
     duplicates+=1;old=unique[key]
     if old[2:4]!=rec[2:4] or abs(old[0]-clock)>.005 or not np.allclose(old[4],a,atol=1e-6,rtol=0):conflicts+=1;conflict_keys.add(key)
    else:unique[key]=rec
   except (ValueError,IndexError,TypeError):bad+=1
 # Conflicting duplicate timestamps are omitted, never resolved by blending positions.
 keys=sorted(k for k in unique if k not in conflict_keys);records=[unique[k] for k in keys];values=np.stack([r[4] for r in records]);out.mkdir(exist_ok=True,parents=True)
 np.savez_compressed(cache,quarter=np.array([k[0] for k in keys],np.int8),timestamp_ms=np.array([k[1] for k in keys],np.int64),clock=np.array([r[0] for r in records]),shotclock=np.array([r[1] for r in records]),player_ids=np.array([r[2] for r in records],np.int32),team_ids=np.array([r[3] for r in records],np.int64),ball=values[:,:3].astype(np.float32),xy=values[:,3:].reshape(-1,10,2).astype(np.float32))
 report=dict(game_id=str(d['gameid']),game_date=d['gamedate'],source=str(path.resolve()),source_sha256=stamp,raw_moments=raw,duplicate_occurrences=duplicates,invalid_moments=bad,conflicting_duplicate_occurrences=conflicts,conflicting_timestamps_removed=len(conflict_keys),unique_frames=len(keys),teams=teams,roster=rosters,events=len(d['events']));meta.write_text(json.dumps(report,indent=2));print(path.stem,report['raw_moments'],report['unique_frames'],'duplicates',duplicates,'conflicts',len(conflict_keys),flush=True)

import argparse
from dataclasses import dataclass,asdict
from collections import Counter,defaultdict
import pandas as pd

@dataclass(frozen=True)
class Settings:
    name:str='primary'
    radius:float=3.0
    margin:float=.75
    dwell:float=.12
    minimum_distance:float=4.0
    maximum_flight:float=1.2
    maximum_height:float=8.5
    same_owner_bridge:float=.28
    event_guard:float=.25

SETTINGS=[Settings('strict',2.5,1.0,.16,5.0,1.0,8.5),Settings(),Settings('relaxed',3.5,.5,.08,3.0,1.4,9.0)]

def clock_seconds(value):
    m,s=str(value).split(':');return 60*float(m)+float(s)

def elapsed(q,c):
    return ((q-1)*720 if q<=4 else 2880+(q-5)*300)+(720 if q<=4 else 300)-c

def context_arrays(frames,pbp,event_guard=.25):
    q=frames['quarter'];c=frames['clock'];t=frames['timestamp_ms']/1000
    dt=np.diff(t);dc=c[:-1]-c[1:];same=(q[1:]==q[:-1])&np.all(frames['player_ids'][1:]==frames['player_ids'][:-1],axis=1)&np.all(frames['team_ids'][1:]==frames['team_ids'][:-1],axis=1)
    edge=same&(dt>=.015)&(dt<=.085)&(dc>=.015)&(dc<=.085)&(abs(dt-dc)<=.04)
    live=np.r_[False,edge]|np.r_[edge,False];guard=np.zeros(len(t),bool)
    for r in pbp.itertuples():
        if int(r.EVENTMSGTYPE) in set(range(1,14)):
            guard|=(q==int(r.PERIOD))&(np.abs(c-clock_seconds(r.PCTIMESTRING))<=event_guard)
    ball=frames['ball'];inside=(ball[:,0]>=0)&(ball[:,0]<=94)&(ball[:,1]>=0)&(ball[:,1]<=50)&(ball[:,2]>=0)
    eligible=live&(~guard)&inside
    connected=np.r_[False,edge]&eligible&np.r_[False,eligible[:-1]]
    starts=eligible&(~connected);segments=np.cumsum(starts)-1;segments[~eligible]=-1
    dist=np.linalg.norm(frames['xy']-ball[:,None,:2],axis=2);rank=np.argsort(dist,axis=1);n1=rank[:,0];n2=rank[:,1];d1=dist[np.arange(len(t)),n1];d2=dist[np.arange(len(t)),n2]
    return dict(time=t,live=live,edge=edge,guard=guard,eligible=eligible,segment=segments,nearest_index=n1,nearest_distance=d1,second_distance=d2)


def infer(frames,pbp,settings=Settings()):
    ctx=context_arrays(frames,pbp,settings.event_guard);t=ctx['time'];n=len(t);ix=np.arange(n);nearest=frames['player_ids'][ix,ctx['nearest_index']]
    good=ctx['eligible']&(ctx['nearest_distance']<=settings.radius)&((ctx['second_distance']-ctx['nearest_distance'])>=settings.margin)&(frames['ball'][:,2]<=settings.maximum_height)
    candidate=np.where(good,nearest,-1);stable=np.full(n,-1,np.int64);bouts=[]
    changes=np.r_[0,np.flatnonzero((candidate[1:]!=candidate[:-1])|(ctx['segment'][1:]!=ctx['segment'][:-1]))+1,n]
    for a,b in zip(changes[:-1],changes[1:]):
        if candidate[a]<0 or ctx['segment'][a]<0:continue
        duration=t[b-1]-t[a]+.04
        if duration+1e-6<settings.dwell:continue
        pid=int(candidate[a]);j=int(np.flatnonzero(frames['player_ids'][a]==pid)[0]);team=int(frames['team_ids'][a,j]);stable[a:b]=pid
        if bouts and bouts[-1]['player_id']==pid and bouts[-1]['segment']==int(ctx['segment'][a]) and t[a]-t[bouts[-1]['end_frame']]<=settings.same_owner_bridge and frames['ball'][bouts[-1]['end_frame']:a+1,2].max()<=settings.maximum_height:
            bouts[-1]['end_frame']=b-1;bouts[-1]['support_frames']+=b-a
        else:bouts.append(dict(player_id=pid,team_id=team,segment=int(ctx['segment'][a]),start_frame=int(a),end_frame=int(b-1),support_frames=int(b-a)))
    inferred=stable.copy()
    for b in bouts:inferred[b['start_frame']:b['end_frame']+1]=b['player_id']
    transfers=[];sequences=[];active_chain=None;chain_counter=0
    def finish():
        nonlocal active_chain
        if active_chain is not None and len(active_chain['players'])>=2:sequences.append(active_chain)
        active_chain=None
    for previous,current in zip(bouts,bouts[1:]):
        a=previous['end_frame'];b=current['start_frame'];gap=t[b]-t[a];same_segment=previous['segment']==current['segment']
        if previous['player_id']==current['player_id']:
            finish();continue
        reasons=[]
        if not same_segment:reasons.append('tracking_lineup_live_or_event_boundary')
        if previous['team_id']!=current['team_id']:reasons.append('opponent_control')
        if gap<=0 or gap>settings.maximum_flight:reasons.append('flight_gap')
        displacement=float(np.linalg.norm(frames['ball'][b,:2]-frames['ball'][a,:2]));maxheight=float(frames['ball'][a:b+1,2].max())
        if displacement<settings.minimum_distance:reasons.append('close_transfer_or_handoff')
        if maxheight>settings.maximum_height:reasons.append('high_ball_shot_or_lob')
        if gap>0 and displacement/gap>100:reasons.append('implausible_speed')
        free=int(np.sum(ctx['nearest_distance'][a+1:b]>settings.radius)) if b>a+1 else 0
        if free<1:reasons.append('no_free_flight_evidence')
        team=previous['team_id'];lineup=tuple(sorted(int(pid) for pid,tid in zip(frames['player_ids'][a],frames['team_ids'][a]) if tid==team));opponents=tuple(sorted(int(pid) for pid,tid in zip(frames['player_ids'][a],frames['team_ids'][a]) if tid!=team))
        record=dict(setting=settings.name,quarter=int(frames['quarter'][a]),source=previous['player_id'],target=current['player_id'],source_team=team,target_team=current['team_id'],release_frame=int(a),receive_frame=int(b),release_timestamp_ms=int(frames['timestamp_ms'][a]),receive_timestamp_ms=int(frames['timestamp_ms'][b]),release_clock=float(frames['clock'][a]),receive_clock=float(frames['clock'][b]),flight_seconds=float(gap),ball_displacement_ft=displacement,maximum_ball_height_ft=maxheight,free_flight_frames=free,lineup='-'.join(map(str,lineup)),opponent_lineup='-'.join(map(str,opponents)),accepted=not reasons,reason=';'.join(reasons),chain_id='')
        if not reasons:
            if active_chain is None or active_chain['players'][-1]!=previous['player_id']:
                finish();chain_counter+=1;active_chain=dict(chain_id=chain_counter,team_id=team,quarter=int(frames['quarter'][a]),segment=previous['segment'],lineup=record['lineup'],players=[previous['player_id']],release_frames=[],receive_frames=[])
            active_chain['players'].append(current['player_id']);active_chain['release_frames'].append(int(a));active_chain['receive_frames'].append(int(b));record['chain_id']=active_chain['chain_id']
        else:finish()
        transfers.append(record)
    finish();ctx.update(candidate=candidate,stable_owner=stable,inferred_owner=inferred)
    columns=['setting','quarter','source','target','source_team','target_team','release_frame','receive_frame','release_timestamp_ms','receive_timestamp_ms','release_clock','receive_clock','flight_seconds','ball_displacement_ft','maximum_ball_height_ft','free_flight_frames','lineup','opponent_lineup','accepted','reason','chain_id']
    return pd.DataFrame(transfers,columns=columns),sequences,ctx,pd.DataFrame(bouts)


def network_tables(passes,sequences,names):
    edges=Counter();triples=Counter();states=Counter()
    for r in passes.itertuples():edges[(int(r.source),int(r.target))]+=1
    for s in sequences:
        path=s['players']
        for a,b in zip(path,path[1:]):states[(a,b)]+=1
        for a,b,c in zip(path,path[1:],path[2:]):triples[(a,b,c)]+=1
    active=sorted(set(pid for pair in edges for pid in pair));A=pd.DataFrame(0,index=active,columns=active,dtype=int)
    for (a,b),n in edges.items():A.loc[a,b]=n
    totals=A.sum(axis=1);P=A.div(totals.replace(0,np.nan),axis=0)
    state_list=sorted(states);labels=[str(a)+'>'+str(b) for a,b in state_list];M=pd.DataFrame(0,index=labels,columns=labels,dtype=int)
    for (a,b,c),n in triples.items():M.loc[str(a)+'>'+str(b),str(b)+'>'+str(c)]=n
    Q=M.div(M.sum(axis=1).replace(0,np.nan),axis=0)
    et=pd.DataFrame([dict(source=a,target=b,source_name=names.get(a,str(a)),target_name=names.get(b,str(b)),inferred_passes=n) for (a,b),n in edges.items()],columns=['source','target','source_name','target_name','inferred_passes'])
    tt=pd.DataFrame([dict(previous=a,current=b,next=c,previous_name=names.get(a,str(a)),current_name=names.get(b,str(b)),next_name=names.get(c,str(c)),count=n) for (a,b,c),n in triples.items()],columns=['previous','current','next','previous_name','current_name','next_name','count'])
    def entropy(counter,key):
        groups=defaultdict(Counter)
        for (a,b,c),n in counter.items():groups[(b,) if key==1 else (a,b)][c]+=n
        N=sum(sum(g.values()) for g in groups.values())
        return sum(sum(g.values())/N*(-sum((n/sum(g.values()))*np.log2(n/sum(g.values())) for n in g.values())) for g in groups.values()) if N else np.nan
    h1=entropy(triples,1);h2=entropy(triples,2)
    summary=dict(inferred_passes=int(A.to_numpy().sum()),active_players=len(active),sequences=len(sequences),multi_pass_sequences=sum(len(s['players'])>=3 for s in sequences),triples=sum(triples.values()),distinct_pair_states=len(states),h_next_given_current_bits=h1,h_next_given_previous_current_bits=h2,training_entropy_difference_bits=h1-h2)
    return A,P,M,Q,et,tt,summary


def export_network(folder,passes,sequences,names):
    folder.mkdir(exist_ok=True,parents=True);A,P,M,Q,edges,triples,summary=network_tables(passes,sequences,names)
    for name,d in [('pass_counts',A),('transition_probabilities',P),('pair_state_counts',M),('pair_state_probabilities',Q)]:d.to_csv(folder/(name+'.csv'),index=True)
    edges.to_csv(folder/'edges.csv',index=False);triples.to_csv(folder/'triples.csv',index=False);(folder/'metrics.json').write_text(json.dumps({k:(None if isinstance(v,(float,np.floating)) and not np.isfinite(v) else v) for k,v in summary.items()},indent=2,allow_nan=False))
    return A,M,summary


def infer_sensitivity(primary,other):
    flags=[];used=set()
    for r in primary.itertuples():
        q=other[(other.source==r.source)&(other.target==r.target)&(other.quarter==r.quarter)];best=None
        for i,z in q.iterrows():
            score=abs(z.release_timestamp_ms-r.release_timestamp_ms)+abs(z.receive_timestamp_ms-r.receive_timestamp_ms)
            if i not in used and abs(z.release_timestamp_ms-r.release_timestamp_ms)<=250 and abs(z.receive_timestamp_ms-r.receive_timestamp_ms)<=250 and (best is None or score<best[0]):best=(score,i)
        flags.append(best is not None)
        if best:used.add(best[1])
    return flags


def draw_network(ax,A,names,title,second=False):
    import networkx as nx
    if A.empty or not A.to_numpy().any():ax.text(.5,.5,'No eligible inferred connections',ha='center');ax.axis('off');return
    G=nx.from_pandas_adjacency(A,create_using=nx.DiGraph);G.remove_nodes_from(list(nx.isolates(G)))
    if second and len(G)>14:
        rank=sorted(G,key=lambda n:-(G.in_degree(n,weight='weight')+G.out_degree(n,weight='weight')));G=G.subgraph(rank[:14]).copy();G.remove_nodes_from(list(nx.isolates(G)))
    H=nx.Graph();H.add_nodes_from(G)
    for a,b,d in G.edges(data=True):H.add_edge(a,b,weight=H.get_edge_data(a,b,{}).get('weight',0)+d['weight'])
    pos=nx.spring_layout(H,weight='weight',seed=42);strength=dict(G.degree(weight='weight'));mx=max(strength.values());sizes=[180+1200*strength[n]/mx for n in G];maximum=max(d['weight'] for a,b,d in G.edges(data=True));nx.draw_networkx_edges(G,pos,ax=ax,node_size=sizes,width=[.4+3*d['weight']/maximum for a,b,d in G.edges(data=True)],alpha=.4,edge_color='#3d789d',arrowsize=9,connectionstyle='arc3,rad=.12');nx.draw_networkx_nodes(G,pos,ax=ax,node_size=sizes,node_color='#d4e6f1',edgecolors='#34769e')
    def label(n):
        if second:return ' → '.join(names.get(int(x),x).split()[-1] for x in str(n).split('>'))
        return names.get(int(n),str(n))
    top=sorted(G,key=lambda n:-strength[n])[:min(8,len(G))]
    for yy,n in zip(np.linspace(.92,.08,len(top)),sorted(top,key=lambda n:-pos[n][1])):ax.annotate(label(n),xy=pos[n],xytext=(.01,yy),textcoords='axes fraction',fontsize=8,ha='left',bbox=dict(fc='white',ec='none',alpha=.9),arrowprops=dict(arrowstyle='-',color='#536879',lw=.6))
    ax.set_title(title,fontsize=12);ax.axis('off');ax.margins(.2)


def audit_contact_sheet(frames,row,names,path):
    import matplotlib.pyplot as plt
    a=int(row.release_frame);b=int(row.receive_frame);indices=np.linspace(max(0,a-3),min(len(frames['clock'])-1,b+4),6).astype(int);fig,axes=plt.subplots(2,3,figsize=(15,8))
    for ax,k in zip(axes.flat,indices):
        ax.set(xlim=(-2,96),ylim=(-2,52),aspect='equal');ax.plot([0,94,94,0,0],[0,0,50,50,0],color='gray');ax.plot([47,47],[0,50],color='gray',lw=.5)
        for team in np.unique(frames['team_ids'][k]):
            sel=frames['team_ids'][k]==team;ax.scatter(frames['xy'][k,sel,0],frames['xy'][k,sel,1],s=45,c='#2473b8' if team==row.source_team else '#d8913b')
        for pid in [int(row.source),int(row.target)]:
            j=np.flatnonzero(frames['player_ids'][k]==pid)
            if len(j):x,y=frames['xy'][k,j[0]];ax.annotate(names.get(pid,str(pid)).split()[-1],(x,y),xytext=(3,3),textcoords='offset points',fontsize=8)
        ax.scatter(frames['ball'][k,0],frames['ball'][k,1],s=35,color='red',zorder=5);ax.set_title(f'Q{frames["quarter"][k]} clock={frames["clock"][k]:.2f}; ball z={frames["ball"][k,2]:.1f}ft',fontsize=9)
    fig.suptitle('Inferred transfer audit: '+names[int(row.source)]+' → '+names[int(row.target)]+'\nRed=ball; snapshots do not establish annotated ground truth.');fig.tight_layout(rect=[0,0,1,.92]);fig.savefig(path,dpi=140);plt.close(fig)


def main(argv=None):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--sportvu-passing-sequences',action='store_true');parser.add_argument('--tracking-dir',type=Path,default=Path(__file__).resolve().parent/'sportvu_sequence_data/raw');parser.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'nba_passing_output/sportvu_passing_sequences');parser.add_argument('--games',nargs='*');args=parser.parse_args(argv);out=args.output_dir.expanduser();out.mkdir(exist_ok=True,parents=True);raw=args.tracking_dir.expanduser();paths=[p for p in sorted(raw.glob('*.json')) if p.stem.isdigit() and (not args.games or p.stem in args.games)]
    if not paths:raise ValueError('No raw SportVU game JSON files found')
    allpasses=[];allseq=[];allsensitivity=[];game_stats=[];assist_checks=[];metadata=[];allnames={};teamnames={};matrices=[];source_manifest=[]
    with PdfPages(out/'passing_sequence_networks.pdf') as pdf:
        def save(fig,name):fig.tight_layout(rect=[0,.07,1,.94]);pdf.savefig(fig);fig.savefig(out/(name+'.pdf'));fig.savefig(out/(name+'.png'),dpi=145);plt.close(fig)
        for path in paths:
            prepare(path,out/'frame_cache');gid=path.stem;meta=json.loads((out/'frame_cache'/(gid+'_metadata.json')).read_text());metadata.append(meta);frames=dict(np.load(out/'frame_cache'/(gid+'_frames.npz')));pbppath=raw/(gid+'_events.csv')
            if not pbppath.exists():raise ValueError('Need repository play-by-play CSV: '+str(pbppath))
            pbp=pd.read_csv(pbppath,dtype={'GAME_ID':str});names={int(k):v['name'] for k,v in meta['roster'].items()};allnames.update(names);teamnames.update({int(k):v['abbreviation'] for k,v in meta['teams'].items()});source_manifest.extend([dict(game_id=gid,path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in [path,pbppath]])
            results={}
            for setting in SETTINGS:
                transfers,sequences,ctx,bouts=infer(frames,pbp,setting);transfers.insert(0,'game_id',gid);accepted=transfers[transfers.accepted].copy();results[setting.name]=(accepted,sequences,ctx,transfers,bouts);folder=out/'games'/gid/setting.name;folder.mkdir(exist_ok=True,parents=True);transfers.to_csv(folder/'all_candidate_transfers.csv',index=False);pd.DataFrame([{**s,'game_id':gid,'setting':setting.name,'players':'>'.join(map(str,s['players'])),'release_frames':';'.join(map(str,s['release_frames'])),'receive_frames':';'.join(map(str,s['receive_frames']))} for s in sequences]).to_csv(folder/'sequences.csv',index=False)
                for team in sorted(map(int,meta['teams'])):
                    own=accepted[accepted.source_team.eq(team)];seq=[s for s in sequences if s['team_id']==team];A,M,summary=export_network(folder/teamnames[team],own,seq,names);summary.update(game_id=gid,date=meta['game_date'],team=teamnames[team],setting=setting.name);allsensitivity.append(summary)
            accepted,sequences,ctx,transfers,bouts=results['primary'];accepted=accepted.copy();accepted['also_strict']=infer_sensitivity(accepted,results['strict'][0]);accepted['also_relaxed']=infer_sensitivity(accepted,results['relaxed'][0]);accepted['robust_across_all_settings']=accepted.also_strict&accepted.also_relaxed;accepted['source_name']=accepted.source.map(names);accepted['target_name']=accepted.target.map(names);accepted.to_csv(out/'games'/gid/'inferred_passes.csv',index=False);pd.DataFrame(list(meta['roster'].values())).to_csv(out/'games'/gid/'roster.csv',index=False);allpasses.append(accepted)
            for s in sequences:allseq.append(dict(game_id=gid,**s))
            pd.DataFrame(dict(quarter=frames['quarter'],timestamp_ms=frames['timestamp_ms'],clock=frames['clock'],segment=ctx['segment'],eligible=ctx['eligible'],pbp_guard=ctx['guard'],nearest_player=frames['player_ids'][np.arange(len(ctx['time'])),ctx['nearest_index']],nearest_distance=ctx['nearest_distance'],second_distance=ctx['second_distance'],candidate=ctx['candidate'],stable_owner=ctx['stable_owner'],inferred_owner=ctx['inferred_owner'])).to_csv(out/'games'/gid/'frame_owner_audit.csv',index=False)
            live_seconds=float(np.sum((frames['clock'][:-1]-frames['clock'][1:])[ctx['edge']]));eligible_seconds=float(np.sum((frames['clock'][:-1]-frames['clock'][1:])[ctx['edge']&ctx['eligible'][:-1]&ctx['eligible'][1:]]));game_stats.append(dict(game_id=gid,date=meta['game_date'],raw_moments=meta['raw_moments'],unique_frames=meta['unique_frames'],duplicate_occurrences=meta['duplicate_occurrences'],conflicting_timestamps_removed=meta['conflicting_timestamps_removed'],live_clock_seconds=live_seconds,eligible_clock_seconds=eligible_seconds,stable_owned_frames=int((ctx['stable_owner']>=0).sum()),eligible_frames=int(ctx['eligible'].sum()),inferred_passes=len(accepted),robust_passes=int(accepted.robust_across_all_settings.sum()),candidate_transfers=len(transfers)))
            # A partial assist-pair detection check, not precision of every inferred pass.
            for r in pbp.itertuples():
                if int(r.EVENTMSGTYPE)!=1 or pd.isna(r.PLAYER2_ID) or int(r.PLAYER2_ID)<=0:continue
                shooter,assister=int(r.PLAYER1_ID),int(r.PLAYER2_ID);clock=clock_seconds(r.PCTIMESTRING);near=(frames['quarter']==int(r.PERIOD))&(frames['clock']>=clock+.3)&(frames['clock']<=clock+4);usable=int((ctx['live']&near).sum())>=20;match=accepted[(accepted.quarter==int(r.PERIOD))&(accepted.source==assister)&(accepted.target==shooter)&((accepted.receive_clock-clock)>=-.5)&((accepted.receive_clock-clock)<=4)]
                assist_checks.append(dict(game_id=gid,event=int(r.EVENTNUM),quarter=int(r.PERIOD),clock=clock,assister=assister,shooter=shooter,assister_name=names.get(assister,str(assister)),shooter_name=names.get(shooter,str(shooter)),usable_pre_shot_tracking=usable,pair_detected=bool(len(match))))
            for team in sorted(map(int,meta['teams'])):
                own=accepted[accepted.source_team.eq(team)];seq=[s for s in sequences if s['team_id']==team];A,M,summary=export_network(out/'games'/gid/'primary'/teamnames[team],own,seq,names);matrices.append(dict(game_id=gid,team=teamnames[team],**summary));fig,axes=plt.subplots(1,2,figsize=(16,9));draw_network(axes[0],A,names,teamnames[team]+' · first-order inferred passes');draw_network(axes[1],M,names,'Second-order: previous→current pair states\nTop14 active states shown; full matrices saved',True);fig.suptitle(meta['game_date']+' · inferred tracking sequences · '+gid);fig.text(.5,.02,'Conservative inferred passes, not annotated events. Sequences stop at uncertain transfers, lineups, gaps and PBP boundaries.\nNode size: pass/transition involvement. FR layout is aesthetic; missing outgoing probability rows remain undefined.',ha='center',fontsize=9);save(fig,gid+'_'+teamnames[team]+'_sequence_networks')
                # Exact observed lineup identity at both inferred pass endpoints.
                lineups=[]
                for lineup,g in own.groupby('lineup'):
                    selected=list(map(int,lineup.split('-')));A5=pd.DataFrame(0,index=selected,columns=selected,dtype=int)
                    for z in g.itertuples():A5.loc[int(z.source),int(z.target)]+=1
                    folder=out/'games'/gid/'lineup_networks'/teamnames[team]/lineup;folder.mkdir(exist_ok=True,parents=True);A5.to_csv(folder/'pass_counts.csv');lineups.append(dict(game_id=gid,team=teamnames[team],lineup=lineup,players='; '.join(names[x] for x in selected),inferred_passes=len(g)))
                table=pd.DataFrame(lineups,columns=['game_id','team','lineup','players','inferred_passes']);table.to_csv(out/'games'/gid/'primary'/teamnames[team]/'lineup_summary.csv',index=False)
                if len(table):
                    best=table.sort_values(['inferred_passes','lineup'],ascending=[False,True]).iloc[0];A5=pd.read_csv(out/'games'/gid/'lineup_networks'/teamnames[team]/best.lineup/'pass_counts.csv',index_col=0);A5.index=A5.index.astype(int);A5.columns=A5.columns.astype(int);fig,ax=plt.subplots(figsize=(12,9));draw_network(ax,A5,names,teamnames[team]+' · most-observed inferred-pass five-player lineup');fig.suptitle(meta['game_date']+f' · {best.inferred_passes} inferred passes');fig.text(.5,.02,'Same five teammates are tracked on court at both endpoints of every accepted pass; no substitution is crossed.\nCounts cover eligible recorded live-clock periods only, not complete official lineup totals.',ha='center',fontsize=9);save(fig,gid+'_'+teamnames[team]+'_lineup_network')
            example=accepted[(accepted.source_team==1610612744)&accepted.robust_across_all_settings].copy()
            if len(example):
                example=example.sort_values('ball_displacement_ft');r=example.iloc[len(example)//2];audit_contact_sheet(frames,r,names,out/(gid+'_transfer_contact_sheet.png'));pd.DataFrame([r]).to_csv(out/(gid+'_transfer_contact_sheet_pass.csv'),index=False)
            print('Processed',gid,meta['game_date'],'inferred passes',len(accepted),'robust',int(accepted.robust_across_all_settings.sum()),flush=True)
        passes=pd.concat(allpasses,ignore_index=True);passes.to_csv(out/'all_inferred_passes.csv',index=False);pd.DataFrame(allsensitivity).to_csv(out/'settings_and_network_metrics.csv',index=False);pd.DataFrame(game_stats).to_csv(out/'game_coverage_audit.csv',index=False);pd.DataFrame(assist_checks).to_csv(out/'assist_pair_check.csv',index=False);pd.DataFrame(source_manifest).to_csv(out/'source_manifest.csv',index=False);pd.DataFrame(matrices).to_csv(out/'primary_network_metrics.csv',index=False)
        sequence_rows=[dict(game_id=s['game_id'],chain_id=s['chain_id'],team_id=s['team_id'],quarter=s['quarter'],lineup=s['lineup'],players='>'.join(map(str,s['players'])),player_names=' → '.join(allnames[p] for p in s['players']),passes=len(s['players'])-1) for s in allseq];pd.DataFrame(sequence_rows).to_csv(out/'all_sequences.csv',index=False)
        for team in sorted(passes.source_team.unique()):
            own=passes[passes.source_team.eq(team)];seq=[s for s in allseq if s['team_id']==team];A,M,summary=export_network(out/'pooled'/teamnames[int(team)],own,seq,allnames);fig,axes=plt.subplots(1,2,figsize=(16,9));draw_network(axes[0],A,allnames,teamnames[int(team)]+' · pooled inferred passes');draw_network(axes[1],M,allnames,'Ordered-pair sequence network',True);fig.suptitle('Historical sample: inferred sequence networks, not a complete season');save(fig,'pooled_'+teamnames[int(team)]+'_sequence_networks')
    checks=pd.DataFrame(assist_checks);covered=checks[checks.usable_pre_shot_tracking] if len(checks) else checks
    report=['# Inferred passing-sequence networks from SportVU tracking',
        f'Processed {len(paths)} historical games from the requested sealneaward/nba-movement-data repository. These are movement-inferred conservative transfers, NOT an annotated or validated complete list of NBA passes. The sample is January2016, not the previous2021–22 or2020s datasets.',
        '## What is included',
        'Player-to-player count/probability matrices; actual ordered inferred-player sequences; second-order states (previous,current) with transitions (A,B)→(B,C) only for consecutive accepted passes within a continuous chain; exact tracked five-player lineup matrices; frame ownership and all rejected-candidate audits; three threshold settings; source hashes and tracking coverage; original matching play-by-play used as conservative boundaries.',
        '## Inference and boundaries',
        'Deduplicate raw moments by period and tracking timestamp; remove conflicting duplicate frames rather than blending them. Retain exactly ten unique players, five per team, and one ball. Running-clock adjacent frames require15–85ms tracking and game-clock steps, agreement within40ms, and unchanged ten-player lineup. Out-of-court ball frames, gaps, lineup changes and ±0.25s around PBP shots, rebounds, turnovers, fouls, free throws, substitutions, timeouts, jumps and period events break continuity. This omits inbound/stopped-clock passes and near-event transfers.',
        'Infer a candidate owner only when the nearest player is within3ft, at least0.75ft closer than the second player, and the ball is below8.5ft; require0.12s of stable support. Brief same-owner unknown gaps up to0.28s can bridge dribbles. A pass requires a different same-team stable receiver, same continuous segment, at most1.2s flight, at least4ft ball displacement, at least one free-flight frame beyond3ft from every player, ball height≤8.5ft, and displacement speed≤100ft/s. Short transfers/handoffs, high-ball shots OR lobs, ambiguous ownership, opponent control and gaps are rejected separately. This intentionally misses close passes and lobs. Thresholds are heuristics, not learned pass probabilities.',
        '## Threshold sensitivity',str([asdict(s) for s in SETTINGS]),
        f'Primary inferred passes: {len(passes)}; matching strict AND relaxed settings within250ms at both endpoints: {int(passes.robust_across_all_settings.sum())}. Agreement is a sensitivity flag, not a probability a pass is correct. Settings are not guaranteed nested.',
        '## Sequence interpretation',
        'Unknown or rejected transfers break chains; they are not silently skipped to create A→C or A→B→C paths. Second-order matrices retain full state counts while diagrams show only the14 most involved pair states. Probability rows without outgoing transitions are undefined (NaN), not forced to an absorbing state. The listed first-/second-order entropy comparison uses the SAME inferred triples; conditioning on more context reduces training entropy by construction. The difference is not evidence of a true Markov memory effect without an appropriate held-out/null test. No stationarity or Markov property is claimed.',
        '## Five-player lineups',
        'Every accepted pass belongs to the five same-team player IDs tracked on court at both endpoints, with the whole ten-player lineup unchanged throughout its continuous segment. Per-lineup matrices sum back to team primary counts. These are genuine tracked lineup identities for inferred passes, unlike simply trimming season networks to five starters. They remain incomplete conservative tracking subsets, not official full lineup passing totals.',
        '## Limited assist-pair sanity check',
        f'Official assisted made shots in supplied PBP: {len(checks)}; usable preceding tracking windows: {len(covered)}; inferred passer→shooter pair detected within the stated−0.5 to4s receive-time window: {int(covered.pair_detected.sum()) if len(covered) else 0}. This is a partial detection check, NOT pass precision or comprehensive recall. Assists are selected final passes, event clocks are rounded, some tracking windows are missing, and the event guard deliberately removes near-shot frames. All ordinary passes are unlabeled, so no actual-data pass accuracy percentage is claimed.',
        '## Limitations',
        'Nearest-player proximity does not guarantee ownership. Screens, dribbles, contested catches, tips, rebounds, short touch passes and tracking noise can be misclassified. High-ball rejection removes legitimate lobs too. PBP boundaries are approximate and can split real sequences. Unknown intervals are never filled as zero passes or fabricated links. Lineups and sequence timing are observed tracking context, but transfer identity is inferred. Contact-sheet examples support visual review, not full ground-truth validation.',
        '## Reproducibility',
        'Original dataset: https://github.com/sealneaward/nba-movement-data ; pinned sample commit5c9ee6e1387e052c19ab2d22e12120c2210cbe54. Frame conversion follows the repository schema, with no execution of its setup scripts. Raw JSON/PBP SHA256s and deduplication audit are saved. Standalone sportvu_passing_sequences.py supports another directory of raw game JSON plus GAMEID_events.csv. Previous project analyses and CLI remain separate.'
    ];(out/'README_RESULTS.md').write_text('\n\n'.join(report));(out/'inference_settings.json').write_text(json.dumps([asdict(s) for s in SETTINGS],indent=2));(out/'run_manifest.json').write_text(json.dumps(dict(games=[p.stem for p in paths],inferred_passes=len(passes),robust_across_settings=int(passes.robust_across_all_settings.sum()),repository='https://github.com/sealneaward/nba-movement-data',sample_commit='5c9ee6e1387e052c19ab2d22e12120c2210cbe54',status='exploratory movement-inferred sequences; not complete annotated passes',settings=[asdict(s) for s in SETTINGS]),indent=2));print(pd.DataFrame(game_stats).to_string(index=False),flush=True);return 0

if __name__=='__main__':raise SystemExit(main())
