"""Actual tracked five-player lineups with explicit, ambiguous role-slot inference.

This is a lineup-constrained prior model, not recorded tactical role ground truth.
"""
from pathlib import Path
import argparse,itertools,json,hashlib,sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import networkx as nx
POS=['PG','SG','SF','PF','C'];PERMS=np.array(list(itertools.permutations(range(5))),int)


def allowed(tag):
    values=set()
    for t in str(tag).upper().split('-'):
        if t=='G':values.update([0,1])
        elif t=='F':values.update([2,3])
        elif t=='C':values.add(4)
        elif t in POS:values.add(POS.index(t))
    return values or set(range(5))


def candidates(ids,primary,broad,penalty=4.,distance='squared'):
    """Enumerate all 120 bijections; priors never use passing or outcome counts."""
    ids=tuple(sorted(map(int,ids)))
    if len(ids)!=5 or len(set(ids))!=5:raise ValueError('Five distinct tracked players required')
    p=np.array([POS.index(primary[i]) for i in ids]);cost=np.zeros((5,5))
    for i,pid in enumerate(ids):
        delta=np.abs(np.arange(5)-p[i]);cost[i]=delta**2 if distance=='squared' else delta
        cost[i]+=penalty*np.array([r not in allowed(broad[pid]) for r in range(5)])
    scores=cost[np.arange(5)[None,:],PERMS].sum(1);best=float(scores.min());opt=PERMS[np.isclose(scores,best,atol=1e-12)].copy();higher=scores[scores>best+1e-12]
    return ids,opt,best,float(higher.min()-best) if len(higher) else np.nan


def transform(A,assignment):
    A=np.asarray(A,float);assignment=np.asarray(assignment,int)
    if A.shape!=(5,5) or sorted(assignment.tolist())!=list(range(5)) or not np.isfinite(A).all() or (A<0).any() or np.diag(A).any():raise ValueError('Invalid five-player matrix or role bijection')
    T=np.eye(5)[assignment];B=T.T@A@T
    assert np.array_equal(B.sum(),A.sum()) and np.trace(B)==0
    return B


def entropy(A):
    total=A.sum()
    if total<=0:return np.nan
    w=A.sum(1);p=np.divide(A,w[:,None],out=np.zeros_like(A,dtype=float),where=w[:,None]>0)
    return float(-np.sum(A*np.log2(np.maximum(p,1e-300)))/total/np.log2(4))


def clock(s):
    m,sec=str(s).split(':');return 60*int(m)+float(sec)


def draw_network(ax,A,labels,title,positions=None):
    G=nx.from_numpy_array(A,create_using=nx.DiGraph);active=[i for i in G if G.degree(i)>0];G=G.subgraph(active).copy();layout=nx.Graph();layout.add_nodes_from(G)
    for u,v,a in G.edges(data=True):layout.add_edge(u,v,weight=layout.get_edge_data(u,v,{}).get('weight',0)+a['weight'])
    opts=dict(seed=42,iterations=200,weight='weight')
    import inspect
    if 'method' in inspect.signature(nx.spring_layout).parameters:opts['method']='force'
    positions=nx.spring_layout(layout,**opts) if positions is None else positions;sizes=[1000+1700*(G.in_degree(i,weight='weight')+G.out_degree(i,weight='weight'))/max(A.sum(),1) for i in G];maxw=max((a['weight'] for _,_,a in G.edges(data=True)),default=1)
    if len(G):
        nx.draw_networkx_nodes(G,positions,node_size=sizes,node_color='#f1d367',edgecolors='#555',ax=ax);nx.draw_networkx_edges(G,positions,width=[.5+4*a['weight']/maxw for _,_,a in G.edges(data=True)],node_size=sizes,edge_color='#3a7399',arrows=True,arrowsize=15,connectionstyle='arc3,rad=.12',ax=ax);nx.draw_networkx_labels(G,positions,labels={i:labels[i] for i in G},font_size=8.5,ax=ax)
    ax.set_title(title,fontsize=12);ax.axis('off');ax.margins(.16)
    return positions


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--lineup-role-networks',action='store_true');p.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'nba_passing_output');p.add_argument('--destination',type=Path);p.add_argument('--metadata-dir',type=Path,default=Path(__file__).resolve().with_name('position_network_metadata'));p.add_argument('--sensitivity-draws',type=int,default=500);args=p.parse_args(argv)
    if args.sensitivity_draws<1:raise ValueError('At least one sensitivity draw required')
    root=args.output_dir.expanduser();base=root/'sportvu_passing_sequences';out=args.destination or root/'lineup_role_networks';out.mkdir(parents=True,exist_ok=True)
    mappings=pd.read_csv(args.metadata_dir/'historical_positions.csv',dtype={'game_id':str});passes=pd.read_csv(base/'all_inferred_passes.csv',dtype={'game_id':str});assert passes.accepted.eq(True).all();sources=[];configs={};stints=[];passrows=[];events=[];eventaudit=[];rosterinfo={};context={};info={}
    for game,g in mappings.groupby('game_id'):
        f=base/'frame_cache'/(game+'_frames.npz');data=np.load(f);frame={k:data[k] for k in data.files};meta=json.loads((base/'frame_cache'/(game+'_metadata.json')).read_text());names={int(i):r['name'] for i,r in meta['roster'].items()};broad={int(i):r['position'] for i,r in meta['roster'].items()};primary=dict(zip(g.player_id,g.position));teamnames={int(i):r['abbreviation'] for i,r in meta['teams'].items()};N=len(frame['clock']);q=frame['quarter'];ts=frame['timestamp_ms']/1000;dt=np.diff(ts);dc=frame['clock'][:-1]-frame['clock'][1:]
        fullsame=np.r_[False,np.all(frame['player_ids'][1:]==frame['player_ids'][:-1],axis=1)&np.all(frame['team_ids'][1:]==frame['team_ids'][:-1],axis=1)&(q[1:]==q[:-1])];live=np.r_[False,(dt>=.015)&(dt<=.085)&(dc>=.015)&(dc<=.085)&(np.abs(dt-dc)<=.04)]&fullsame
        for tid,team in teamnames.items():
            ownmask=frame['team_ids']==tid;valid=ownmask.sum(1)==5;valid&=np.array([len(set(r))==10 for r in frame['player_ids']]);own=np.sort(np.where(ownmask,frame['player_ids'],np.iinfo(np.int32).max),axis=1)[:,:5]
            if not np.isin(own[valid],g[g.team==team].player_id).all():raise ValueError('Tracked player outside known team roster')
            unique,inverse=np.unique(own[valid],axis=0,return_inverse=True);codes=np.full(N,-1,int);codes[valid]=inverse
            breaks=np.r_[True,(~fullsame[1:])|(dt<=0)|(dt>.085)|(~valid[1:])|(~valid[:-1])];sid=np.cumsum(breaks)-1
            for local,ids_array in enumerate(unique):
                ids=tuple(map(int,ids_array));lineup='-'.join(map(str,ids));key=(team,lineup);k=game+'_'+team+'_'+lineup;keep=valid&(codes==local);observed_seconds=float(np.sum(np.r_[0,dc][keep&live]));framecount=int(keep.sum())
                if key not in configs:
                    ids,opt,best,nextgap=candidates(ids,primary,broad);configs[key]=dict(ids=ids,optimal=opt,best=best,nextgap=nextgap,primary={i:primary[i] for i in ids},broad={i:broad[i] for i in ids},names={i:names[i] for i in ids})
                else:assert all(configs[key]['primary'][i]==primary[i] and configs[key]['broad'][i]==broad[i] for i in ids)
                info[k]=dict(game_id=game,team=team,team_id=tid,lineup=lineup,key=key,ids=ids,tracked_frames=framecount,observed_running_clock_seconds=observed_seconds,tracked_segments=int(np.unique(sid[keep]).size),A=np.zeros((5,5)),events=[])
                rosterinfo[k]=(game,team,lineup);context[game,team,local]=k
            # Observed fragments, not official substitution intervals.
            starts=np.r_[0,np.flatnonzero(breaks[1:])+1];ends=np.r_[starts[1:]-1,N-1]
            for a,b in zip(starts,ends):
                if not valid[a]:continue
                k=context[game,team,int(codes[a])];stints.append(dict(game_id=game,team=team,segment_id=int(sid[a]),lineup=info[k]['lineup'],quarter=int(q[a]),start_frame=int(a),end_frame=int(b),start_clock=float(frame['clock'][a]),end_clock=float(frame['clock'][b]),tracked_frames=int(b-a+1),observed_running_clock_seconds=float(np.sum(np.r_[0,dc][a:b+1]*live[a:b+1]))))
            subset=passes[passes.game_id.eq(game)&passes.source_team.eq(tid)]
            for z in subset.itertuples():
                a,b=int(z.release_frame),int(z.receive_frame)
                assert int(z.release_timestamp_ms)==int(frame['timestamp_ms'][a]) and int(z.receive_timestamp_ms)==int(frame['timestamp_ms'][b]) and int(z.quarter)==int(q[a])==int(q[b])
                if a>b or not valid[a:b+1].all() or sid[a]!=sid[b] or not np.all(codes[a:b+1]==codes[a]):raise AssertionError('Pass crosses a tracked lineup/gap boundary')
                k=context[game,team,int(codes[a])];item=info[k];ids=item['ids'];assert z.source in ids and z.target in ids and str(z.lineup)==item['lineup'];item['A'][ids.index(z.source),ids.index(z.target)]+=1
                assignment=configs[item['key']]['optimal'][0];source_role=POS[assignment[ids.index(z.source)]];target_role=POS[assignment[ids.index(z.target)]];assert source_role!=target_role
                r=z._asdict();r.pop('Index',None);r.update(team=team,role_source=source_role,role_target=target_role,tracked_segment_id=int(sid[a]),optimal_assignment_count=len(configs[item['key']]['optimal']));passrows.append(r)
            pbppath=root.parent/'sportvu_sequence_data/raw'/(game+'_events.csv');pbp=pd.read_csv(pbppath)
            for z in pbp.itertuples():
                typ=int(z.EVENTMSGTYPE);desc=' '.join(str(getattr(z,c,'')) for c in ['HOMEDESCRIPTION','VISITORDESCRIPTION','NEUTRALDESCRIPTION']);annotations=[]
                if typ in [1,2]:annotations=[(int(z.PLAYER1_ID),None,('3PT' if '3PT' in desc.upper() else '2PT')+(' made' if typ==1 else ' missed'))]
                elif typ==3:annotations=[(int(z.PLAYER1_ID),None,'FT missed' if 'MISS' in desc.upper() else 'FT made')]
                elif typ==4:annotations=[(int(z.PLAYER1_ID),'Rebound',None)]
                elif typ==5:
                    annotations=[(int(z.PLAYER1_ID),None,'Turnover')]
                    if 'STEAL' in desc.upper():annotations.append((int(z.PLAYER2_ID),'Steal',None))
                elif typ==6:
                    action=int(z.EVENTMSGACTIONTYPE)
                    if action==2:annotations=[(int(z.PLAYER2_ID),None,'Shooting foul drawn')]
                    elif action in [1,3,11,28]:annotations=[(int(z.PLAYER2_ID),None,'Other foul drawn')]
                    elif action==4:annotations=[(int(z.PLAYER1_ID),None,'Offensive foul')]
                for actor,source,target in annotations:
                    if actor not in set(g[g.team==team].player_id):continue
                    near=valid&(q==int(z.PERIOD))&(np.abs(frame['clock']-clock(z.PCTIMESTRING))<=.25);local_codes=np.unique(codes[near]);reason='mapped'
                    if len(local_codes)==0:reason='no tracking within 0.25 clock seconds'
                    elif len(local_codes)>1:reason='multiple own lineups near event clock'
                    else:
                        k=context[game,team,int(local_codes[0])];item=info[k]
                        if actor not in item['ids']:reason='recorded actor not in uniquely tracked lineup'
                        else:
                            cfg=configs[item['key']];role=POS[cfg['optimal'][0][item['ids'].index(actor)]];item['events'].append(dict(source=source or role,target=target or role,raw_count=1.,kind='clock_matched_pbp_event'))
                    eventaudit.append(dict(game_id=game,team=team,eventnum=int(z.EVENTNUM),event_type=typ,actor_id=actor,source_event=source or 'position',target_event=target or 'position',status=reason,lineup=info[k]['lineup'] if reason=='mapped' else '',clock_tolerance_seconds=.25))
        for source in [f,base/'frame_cache'/(game+'_metadata.json'),pbppath]:sources.append(dict(path=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
        print('Tracked lineups and event alignment:',game,flush=True)
    mappingrows=[];optionrows=[];summaries=[]
    for (team,lineup),cfg in configs.items():
        for option,a in enumerate(cfg['optimal']):
            for pid,role in zip(cfg['ids'],a):optionrows.append(dict(team=team,lineup=lineup,assignment_id=option,player_id=pid,role=POS[role],minimum_cost=cfg['best']))
        a=cfg['optimal'][0]
        for pid,role in zip(cfg['ids'],a):mappingrows.append(dict(team=team,lineup=lineup,player_id=pid,name=cfg['names'][pid],season_primary_position=cfg['primary'][pid],tracking_broad_position=cfg['broad'][pid],inferred_role=POS[role],changed_from_primary=POS[role]!=cfg['primary'][pid],outside_tracking_broad_class=role not in allowed(cfg['broad'][pid]),optimal_assignment_count=len(cfg['optimal']),assignment_status='unique minimum-cost' if len(cfg['optimal'])==1 else 'ambiguous minimum-cost',minimum_cost=cfg['best'],next_cost_gap=cfg['nextgap']))
    pd.DataFrame(mappingrows).to_csv(out/'lineup_role_assignments.csv',index=False);pd.DataFrame(optionrows).to_csv(out/'all_tied_role_assignments.csv',index=False);pd.DataFrame(stints).to_csv(out/'observed_tracking_segments.csv',index=False);pd.DataFrame(eventaudit).to_csv(out/'event_lineup_alignment_audit.csv',index=False);pd.DataFrame(passrows).to_csv(out/'pass_role_assignments.csv',index=False)
    positive={k:item for k,item in info.items() if item['A'].sum()>0};assert len(passrows)==len(passes) and sum(i['A'].sum() for i in info.values())==len(passes)
    for k,item in info.items():
        cfg=configs[item['key']];B=transform(item['A'],cfg['optimal'][0]);h=entropy(B);assert (np.isnan(h) and np.isnan(entropy(item['A']))) or np.isclose(h,entropy(item['A']))
        folder=out/'lineups'/item['game_id']/item['team']/item['lineup'];folder.mkdir(parents=True,exist_ok=True);pd.DataFrame(item['A'],index=item['ids'],columns=item['ids']).to_csv(folder/'player_pass_counts.csv');pd.DataFrame(B,index=POS,columns=POS).to_csv(folder/'inferred_role_pass_counts.csv');w=B.sum(1);P=np.divide(B,w[:,None],out=np.full_like(B,np.nan),where=w[:,None]>0);pd.DataFrame(P,index=POS,columns=POS).to_csv(folder/'inferred_role_pass_probabilities.csv')
        summaries.append({**{x:item[x] for x in ['game_id','team','lineup','tracked_frames','observed_running_clock_seconds','tracked_segments']},'inferred_passes':int(B.sum()),'passing_players_observed':int(((item['A'].sum(0)+item['A'].sum(1))>0).sum()),'conditional_entropy_fixed_five':h,'optimal_assignment_count':len(cfg['optimal']),'minimum_cost':cfg['best'],'next_cost_gap':cfg['nextgap'],'matched_pbp_annotations':len(item['events']),'has_complete_official_lineup_totals':False})
    summary=pd.DataFrame(summaries);summary.to_csv(out/'lineup_summary.csv',index=False)
    from position_networks import draw,event_edges
    with PdfPages(out/'all_observed_lineup_networks.pdf') as pdf, PdfPages(out/'all_lineup_role_event_diagrams.pdf') as eventpdf:
        for count,(k,item) in enumerate(sorted(positive.items(),key=lambda x:(x[1]['game_id'],x[1]['team'],-x[1]['A'].sum())),1):
            cfg=configs[item['key']];a=cfg['optimal'][0];B=transform(item['A'],a);folder=out/'lineups'/item['game_id']/item['team']/item['lineup'];fig,axes=plt.subplots(1,2,figsize=(16,7));pp=draw_network(axes[0],item['A'],[cfg['names'][pid].replace(' ','\n',1) for pid in item['ids']],'Actual tracked five-player group');rp={int(a[i]):v for i,v in pp.items()};draw_network(axes[1],B,POS,'Inferred one-to-one role slots',positions=rp)
            fig.suptitle(f'{item["team"]} | {item["game_id"]} | {int(B.sum())} retained inferred passes',fontsize=15);text='; '.join(f'{POS[role]}: {cfg["names"][pid]}' for pid,role in sorted(zip(cfg['ids'],a),key=lambda v:v[1]));fig.text(.5,.055,text,ha='center',fontsize=9);fig.text(.5,.015,f'{len(cfg["optimal"])} equally minimum-cost assignments | Shared player locations | Entropy unchanged by relabeling | No role self-loops\nObserved passing isolates omitted from drawing; all five players remain in matrices and assignment tables. Incomplete tracking-inferred passes, not official lineup totals.',ha='center',fontsize=9);fig.tight_layout(rect=[0,.1,1,.93]);pdf.savefig(fig);fig.savefig(folder/'player_vs_role_network.png',dpi=130);plt.close(fig)
            ev=pd.DataFrame(item['events']);ev=ev.groupby(['source','target','kind'],as_index=False).raw_count.sum().to_dict('records') if len(ev) else [];E=event_edges(B,ev);F=E.loc[E.target.isin(['2PT made','2PT missed','3PT made','3PT missed']),'raw_count'].sum();fig,E,_=draw(E,f'{item["team"]} | {item["game_id"]} | tracked five-player lineup | {int(B.sum())} inferred passes',mode='equal_fg' if F>0 else 'observed',historical=True);fig.text(.5,.90,text+f' | {len(cfg["optimal"])} tied role assignments',ha='center',fontsize=8);eventpdf.savefig(fig);fig.savefig(folder/'lineup_role_events.png',dpi=130);plt.close(fig);E.to_csv(folder/'event_edges.csv',index=False)
            if count%20==0:print(count,'lineup diagrams saved',flush=True)
    # Role assignment sensitivity: the same five-player group keeps one assignment across its repeated appearances.
    rng=np.random.default_rng(42);sensitivity=[];poolrows=[];teams=sorted({i['team'] for i in positive.values()})
    for team in teams:
        items=[i for i in positive.values() if i['team']==team];total=sum(i['A'].sum() for i in items);within=sum(i['A'].sum()*entropy(i['A']) for i in items)/total;canonical=sum((transform(i['A'],configs[i['key']]['optimal'][0]) for i in items),np.zeros((5,5)));folder=out/'pooled'/team;folder.mkdir(parents=True,exist_ok=True);pd.DataFrame(canonical,index=POS,columns=POS).to_csv(folder/'canonical_role_pass_counts.csv')
        values=[]
        for draw_id in range(args.sensitivity_draws):
            choices={key:int(rng.integers(len(configs[key]['optimal']))) for key in sorted({i['key'] for i in items})};B=sum((transform(i['A'],configs[i['key']]['optimal'][choices[i['key']]]) for i in items),np.zeros((5,5)));h=entropy(B);values.append(h);sensitivity.append(dict(team=team,draw=draw_id,pooled_entropy=h,weighted_within_lineup_entropy=within,mixing_gap=h-within,total_passes=int(total)));assert np.trace(B)==0 and np.isclose(B.sum(),total) and h>=within-1e-10
        poolrows.append(dict(team=team,games=len({i['game_id'] for i in items}),lineup_game_groups=len(items),inferred_passes=int(total),canonical_pooled_entropy=entropy(canonical),weighted_within_lineup_entropy=within,observed_mixing_gap=entropy(canonical)-within,assignment_sensitivity_q025=np.quantile(values,.025),assignment_sensitivity_q975=np.quantile(values,.975),assignment_sensitivity_min=min(values),assignment_sensitivity_max=max(values),interval_is_statistical_confidence=False))
    pools=pd.DataFrame(poolrows);pools.to_csv(out/'pooled_entropy_and_assignment_sensitivity.csv',index=False);pd.DataFrame(sensitivity).to_csv(out/'assignment_sensitivity_draws.csv',index=False)
    write_model_sensitivity(out,configs,positive)
    overview(out,positive,configs,pools,base)
    for source in [base/'all_inferred_passes.csv',args.metadata_dir/'historical_positions.csv']:sources.append(dict(path=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
    pd.DataFrame(sources).to_csv(out/'source_manifest.csv',index=False);(out/'role_inference_rules.json').write_text(json.dumps(dict(roles=POS,primary_cost='squared ordinal distance from listed position',broad_mismatch_penalty=4,cost_uses_passing_or_outcomes=False,role_assignment_scope='unique team plus five player IDs, held fixed across repeated appearances',canonical_tie_break='lexicographic player-ID/role permutation, not extra evidence',sensitivity_draws=args.sensitivity_draws,sensitivity_seed=42,status='inferred role slots, not verified tactical positions',event_clock_tolerance_seconds=.25,ambiguity_rule='reject PBP annotation if multiple own lineups occur within clock window',stint_max_tracking_gap_seconds=.085,official_lineup_minutes=False),indent=2))
    chosen=summary[summary.inferred_passes.gt(0)];unique_positive={i['key'] for i in positive.values()};ambiguous=sum(len(configs[k]['optimal'])>1 for k in unique_positive);pass_ambiguous=sum(i['A'].sum() for i in positive.values() if len(configs[i['key']]['optimal'])>1)
    report=f'''# Actual tracked lineups with inferred role slots

Three historical 2015–16 games from sealneaward/nba-movement-data, not the 180 later NBA API season networks. The raw tracking provides who is on court; it does not label dynamic PG/SG/SF/PF/C roles. This pipeline assigns those slots using an explicit prior model and retains ambiguity.

Observed lineup-game groups: {len(summary)}; groups with at least one retained inferred pass: {len(chosen)}. Distinct passing-positive team/five-player groups: {len(unique_positive)}; {ambiguous} have more than one minimum-cost role assignment. {int(pass_ambiguous)}/{len(passes)} retained passes occur in those ambiguous groups. All {len(passes)} previously retained passes remain; there is no new pass-accuracy validation. The source extractor is incomplete and threshold-based.

## How roles were inferred

Each five-player group receives one of all 120 one-to-one assignments. Cost is squared ordinal distance from the player's 2015–16 listed primary position, plus 4 when the assigned role falls outside the raw tracking's broad G/F/C category. This is a transparent provisional convention, not a trained/validated position detector. No pass, shot, entropy or outcome statistic chooses roles. All equally minimum-cost assignments are saved. The lexicographic canonical tie-break is only for reproducible drawings. A unique minimum does not prove the role is correct. Choices of ordinal cost and broad-category weight remain assumptions; actual roles can vary within a possession and this model holds them fixed for the same five-player group.

## Actual lineups and timing

Five distinct teammates are required in every retained frame and at every point of an accepted pass. Both teams' on-court configuration, quarter and frame continuity are checked; no pass crosses a substitution/configuration or tracking-gap boundary. Observed tracking segments split on full on-court configuration changes, period changes and timestamp gaps above 85ms. These are observed fragments, not reconstructed official substitution stints. Running-clock seconds sum qualifying adjacent clock decrements; they are incomplete observed exposure, not official lineup minutes or a basis for claiming complete rates. Zero-pass observed groups remain in the summary; networks are drawn only for passing-positive groups. Passing isolates are omitted from graph drawings but all five IDs remain in exported matrices.

## Why the loops disappear and entropy does not change within a lineup

Each player in the five-player group has a different inferred role slot, so a genuine teammate pass cannot become PG→PG. This absence of loops is a consequence of the one-to-one model, not proof the tactical role labels are true. Relabeling a single five-player matrix is a permutation, so its conditional entropy is exactly unchanged. Entropy is averaged over observed passing source weights and normalized by log2(4), retaining the actual five-player support. Unknown zero-outgoing rows are NaN in exported probabilities.

Pooling lineups merges different players into roles, so the pooled role network and entropy do depend on the assignments. We sample uniformly from the tied minimum-cost assignments 500 times (or the requested draw count), holding the same five-player group's selection consistent across games. The resulting ranges are role-model sensitivity intervals, not statistical confidence intervals or probabilities of role correctness. They cover tied solutions under this cost model, not every plausible role model or within-possession switching. role_model_sensitivity.csv additionally compares primary-only, stronger broad-position penalties and absolute-distance role costs; these are model checks, not independent validation.

The pooled entropy equals weighted within-lineup entropy plus an observed mixing gap. The gap is nonnegative and describes how pooling contexts adds destination diversity. Small within-lineup pass samples bias empirical entropy downward, so this gap cannot be interpreted as pure tactical lineup dependence or a causal effect. Many groups have very few passes; no regression across these small lineup samples is claimed.

## PBP event diagrams and court review

Recorded shots, rebounds, steals and fouls are attached only when there is a unique own-team lineup among valid tracking frames within ±0.25 game-clock seconds in the same quarter, with the named actor present. Rounded clocks and tracking gaps prevent some matches; all decisions and rejection reasons are exported. This is a clock-based aggregate event match, not a pass→shot sequence reconstruction. Inbound receivers and unsupported event combinations remain undrawn. Left event arrows identify the recorded rebounder or steal winner, not the next receiver. Shot/foul/FT layers are not mutually exclusive possession outcomes. PBP labels inherit role ambiguity. The field-goal display scale can equalize FG and pass weight, but never enters passing entropy.

Court snapshots use observed coordinates for human review, not to assert that court location alone determines position. They show one moment, not a validation of a tactical assignment. The mapping is stable per five-player group; a player can be assigned a different slot in another lineup.

## Files

- overview_and_role_review.pdf: best-observed lineups, an ambiguous example and court snapshots.
- all_observed_lineup_networks.pdf: player and inferred-role networks for every passing-positive lineup-game group.
- all_lineup_role_event_diagrams.pdf: reference-style diagrams with strictly clock-matched PBP annotations.
- lineup_role_assignments.csv and all_tied_role_assignments.csv: role priors, canonical assignments, costs and all tied solutions.
- observed_tracking_segments.csv, lineup_summary.csv and pass_role_assignments.csv: exact tracking/pass lineage.
- event_lineup_alignment_audit.csv: matched and rejected PBP annotations.
- pooled_entropy_and_assignment_sensitivity.csv: pooled and within-lineup entropy plus assignment sensitivity.

No new games or raw data were downloaded. Original player, season-position and regression outputs are preserved. Run python3 teamgraphs.py --lineup-role-networks or the standalone module with --output-dir and --metadata-dir pointing at the existing project data.

## Pooled descriptive results

'''+pools.to_string(index=False)
    (out/'START_HERE.md').write_text(report);print('Passing-positive groups:',len(chosen),'unique groups:',len(unique_positive),'ambiguous:',ambiguous,'passes in ambiguous groups:',int(pass_ambiguous));print(pools.to_string(index=False));print('PBP matching statuses:');print(pd.DataFrame(eventaudit).status.value_counts().to_string());print('Saved',out);return 0


def write_model_sensitivity(out,configs,positive):
    rows=[];assignments=[]
    policies=[('primary_squared_broad4',4.,'squared'),('primary_squared_only',0.,'squared'),('primary_squared_broad8',8.,'squared'),('primary_absolute_broad4',4.,'absolute')]
    for name,penalty,distance in policies:
        chosen={};counts={}
        for key,cfg in configs.items():
            ids,opt,best,gap=candidates(cfg['ids'],cfg['primary'],cfg['broad'],penalty,distance);chosen[key]=opt[0];counts[key]=len(opt)
            for pid,role in zip(ids,opt[0]):assignments.append(dict(policy=name,team=key[0],lineup=key[1],player_id=pid,role=POS[role],optimal_assignment_count=len(opt),minimum_cost=best))
        for team in sorted({i['team'] for i in positive.values()}):
            items=[i for i in positive.values() if i['team']==team];B=np.zeros((5,5));changed=0.;ambiguous=0.
            for item in items:
                key=item['key'];a=chosen[key];base=configs[key]['optimal'][0];B+=transform(item['A'],a);mask=(a[:,None]!=base[:,None])|(a[None,:]!=base[None,:]);changed+=item['A'][mask].sum();ambiguous+=item['A'].sum() if counts[key]>1 else 0
            rows.append(dict(policy=name,team=team,total_passes=int(B.sum()),pooled_entropy=entropy(B),passes_with_different_role_pair=int(changed),passes_in_tied_assignments=int(ambiguous),is_ground_truth_validation=False))
    pd.DataFrame(rows).to_csv(out/'role_model_sensitivity.csv',index=False);pd.DataFrame(assignments).to_csv(out/'role_model_assignments.csv',index=False)


def overview(out,positive,configs,pools,base):
    picks=[]
    for game,team in sorted({(i['game_id'],i['team']) for i in positive.values()}):picks.append(max((i for i in positive.values() if i['game_id']==game and i['team']==team),key=lambda i:i['A'].sum()))
    ambiguous=[i for i in positive.values() if len(configs[i['key']]['optimal'])>1]
    if ambiguous:picks.append(max(ambiguous,key=lambda i:i['A'].sum()))
    with PdfPages(out/'overview_and_role_review.pdf') as pdf:
        fig,ax=plt.subplots(figsize=(11,6));xx=np.arange(len(pools));ax.bar(xx-.18,pools.weighted_within_lineup_entropy,.35,label='Pass-weighted within-lineup entropy',color='#3b7e9a');ax.bar(xx+.18,pools.canonical_pooled_entropy,.35,label='Pooled inferred-role entropy',color='#cf9653');ax.set_xticks(xx,pools.team);ax.set_ylabel('Conditional entropy / log2(4)');ax.set_ylim(0,1);ax.legend();ax.set_title('Historical pilot: pooling lineups adds apparent passing diversity');fig.text(.5,.015,'Small within-lineup samples and uncertain role assignments limit interpretation; bars are descriptive, not estimates of causal effects.',ha='center',fontsize=9);fig.tight_layout(rect=[0,.05,1,1]);pdf.savefig(fig);fig.savefig(out/'pooled_vs_within_lineup_entropy.png',dpi=150);plt.close(fig)
        for ix,item in enumerate(picks):
            cfg=configs[item['key']];A=item['A'];a=cfg['optimal'][0];B=transform(A,a);fig,axes=plt.subplots(1,2,figsize=(15,7));labels=[POS[r]+'\n'+cfg['names'][pid].replace(' ','\n',1) for pid,r in sorted(zip(cfg['ids'],a),key=lambda v:v[1])];draw_network(axes[0],B,labels,f'{int(A.sum())} inferred passes | {len(cfg["optimal"])} tied assignments')
            frame=np.load(base/'frame_cache'/(item['game_id']+'_frames.npz'));p=pd.read_csv(out/'pass_role_assignments.csv',dtype={'game_id':str});p=p[(p.game_id==item['game_id'])&(p.team==item['team'])&(p.lineup.astype(str)==item['lineup'])];idx=int(p.iloc[0].release_frame);ax=axes[1];ax.set_xlim(0,94);ax.set_ylim(0,50);ax.set_aspect('equal');ax.axvline(47,c='gray',lw=.8);ax.scatter([5.25,88.75],[25,25],marker='s',c='gray',s=30)
            for col,pid in enumerate(frame['player_ids'][idx]):
                xy=frame['xy'][idx,col];own=int(pid) in item['ids'];ax.scatter(xy[0],xy[1],s=100 if own else 35,c='#26798f' if own else '#c7c7c7')
                if own:
                    role=POS[a[item['ids'].index(int(pid))]];ax.annotate(role+' '+cfg['names'][int(pid)],xy,xytext=(3,4),textcoords='offset points',fontsize=7)
            ball=frame['ball'][idx];ax.scatter(ball[0],ball[1],c='orange',s=45);ax.set_title(f'Observed snapshot | Q{int(frame["quarter"][idx])}, clock {int(frame["clock"][idx])//60:02d}:{frame["clock"][idx]%60:05.2f}');ax.set_xlabel('Feet');ax.set_ylabel('Feet');fig.suptitle(f'{item["team"]} | {item["game_id"]} | inferred roles need review');fig.text(.5,.015,'Blue: this tracked five-player group; gray: opponents; orange: ball. Positions are observed, labels are inferred. A single snapshot does not establish tactical roles.',ha='center',fontsize=9);fig.tight_layout(rect=[0,.06,1,.94]);pdf.savefig(fig);fig.savefig(out/(f'review_{ix+1}_{item["team"]}_{item["game_id"]}.png'),dpi=140);plt.close(fig)
        if ambiguous:
            item=max(ambiguous,key=lambda i:i['A'].sum());cfg=configs[item['key']];fig,axes=plt.subplots(1,2,figsize=(15,7))
            canonical_positions=None
            for ax,a,title in zip(axes,cfg['optimal'][:2],['Canonical tied assignment','Equally minimum-cost alternative']):
                role_positions=None if canonical_positions is None else {int(a[i]):canonical_positions[int(cfg['optimal'][0][i])] for i in range(5) if int(cfg['optimal'][0][i]) in canonical_positions}
                labels=[POS[r]+'\n'+cfg['names'][pid].replace(' ','\n',1) for pid,r in sorted(zip(cfg['ids'],a),key=lambda v:v[1])];current_positions=draw_network(ax,transform(item['A'],a),labels,title,positions=role_positions)
                if canonical_positions is None:canonical_positions=current_positions
            fig.suptitle('Ambiguity is retained, not resolved by passing outcomes');fig.text(.5,.015,f'{item["team"]}, {item["game_id"]}: {len(cfg["optimal"])} optimal assignments; both networks preserve all passes and the same within-lineup entropy.',ha='center');fig.tight_layout(rect=[0,.05,1,.94]);pdf.savefig(fig);fig.savefig(out/'ambiguous_assignment_example.png',dpi=150);plt.close(fig)

if __name__=='__main__':raise SystemExit(main())
