"""Five listed-position passing networks and reference-style event diagrams.

Counts remain NBA API totals, or a separately labeled historical GitHub pilot.
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse,FancyBboxPatch,FancyArrowPatch
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import linregress,t
POS=['PG','SG','SF','PF','C']
SHOT=['2PT made','2PT missed','3PT made','3PT missed']
BASE_OUT=SHOT+['FT made','FT missed','Turnover']
ROLE_XY={'PG':(.32,.50),'SG':(.45,.80),'SF':(.61,.31),'PF':(.46,.18),'C':(.61,.68)}


def aggregate(A,roles):
    if len(roles)!=len(A) or not set(roles)<=set(POS):raise ValueError('Missing five-position mapping')
    A=np.asarray(A,float)
    if A.shape!=(len(roles),len(roles)) or not np.isfinite(A).all() or (A<0).any() or np.diag(A).any():raise ValueError('Invalid player pass counts')
    membership=np.zeros((len(A),5))
    for i,r in enumerate(roles):membership[i,POS.index(r)]=1
    B=membership.T@A@membership
    if not np.isclose(B.sum(),A.sum()):raise AssertionError('Lost passes')
    return B,membership


def role_metrics(B,actions):
    B=np.asarray(B,float);actions=np.asarray(actions,float)
    if B.shape!=(5,5) or actions.shape!=(5,) or not np.isfinite(B).all() or not np.isfinite(actions).all() or (B<0).any() or (actions<0).any() or B.sum()<=0 or actions.sum()<=0:raise ValueError('Invalid position counts or action totals')
    total=B.sum();w=B.sum(1);prob=np.divide(B,w[:,None],out=np.zeros_like(B),where=w[:,None]>0)
    bits=float(-np.sum(B*np.log2(np.maximum(prob,1e-300)))/total)
    _,V=np.linalg.eigh(B+B.T);c=np.abs(V[:,-1]);c/=c.sum();a=actions/actions.sum();ci=np.argsort(-c,kind='stable');ui=np.argsort(-a,kind='stable');cross=B.copy();np.fill_diagonal(cross,0);cw=cross.sum(1);cp=np.divide(cross,cw[:,None],out=np.zeros_like(cross),where=cw[:,None]>0);cbits=-np.sum(cross*np.log2(np.maximum(cp,1e-300)))/cross.sum() if cross.sum()>0 else np.nan
    return dict(position_entropy=bits/np.log2(5),position_entropy_bits=bits,position_ec_concentration=float((c@c-.2)/.8),position_action_hhi=float(a@a),same_leading_position=POS[ci[0]]==POS[ui[0]],passing_leading_position=POS[ci[0]],action_leading_position=POS[ui[0]],same_position_passes=float(np.trace(B)),same_position_pass_share=float(np.trace(B)/total),cross_position_entropy=float(cbits/np.log2(4)),passes=float(total)),c,a,prob


def event_edges(B,events):
    rows=[]
    for i,p in enumerate(POS):
        for j,q in enumerate(POS):
            if B[i,j]>0:rows.append(dict(source=p,target=q,raw_count=float(B[i,j]),kind='pass'))
    for r in events:
        if r['raw_count']>0:rows.append(r)
    return pd.DataFrame(rows)


def draw(edges,title,mode='equal_fg',historical=False):
    """Manual reference layout, with no invented inbound or sequencing arrows."""
    E=edges.copy();P=E.loc[E.kind.eq('pass'),'raw_count'].sum();F=E.loc[E.target.isin(SHOT),'raw_count'].sum();scale=P/F if F>0 and mode=='equal_fg' else 1.
    E['display_weight']=E.raw_count;E.loc[E.target.isin(SHOT),'display_weight']*=scale
    ordered=E.sort_values('display_weight',ascending=False);cut=np.searchsorted(ordered.display_weight.cumsum().to_numpy(),.6*E.display_weight.sum())+1;red=set(ordered.index[:cut]);E['highlight']=E.index.isin(red)
    fig,ax=plt.subplots(figsize=(19,10));ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off');patches={};xy=dict(ROLE_XY)
    for p,(x,y) in ROLE_XY.items():
        if not (E.source.eq(p)|E.target.eq(p)).any():continue
        obj=Ellipse((x,y),.063,.112,facecolor='white',edgecolor='#222222',lw=1.6,zorder=5);ax.add_patch(obj);patches[p]=obj;ax.text(x,y,p,ha='center',va='center',fontsize=18,zorder=6)
    def box(name,label,center,width=.16,height=.049,unknown=False):
        x,y=center;obj=FancyBboxPatch((x-width/2,y-height/2),width,height,boxstyle='square,pad=0.004',fc='white' if not unknown else '#f3f3f3',ec='#222222' if not unknown else '#999999',lw=1.5 if not unknown else 1.1,linestyle='-' if not unknown else '--',zorder=5);ax.add_patch(obj);patches[name]=obj;xy[name]=center;ax.text(x,y,label,ha='center',va='center',fontsize=12 if not unknown else 10,color='#222222' if not unknown else '#777777',zorder=6)
    box('Inbound','Inbound\nreceiver unavailable',(.095,.87),.15,.065,True);box('Rebound','Rebounder',(.095,.49),.14,.055);box('Steal','Steal winner',(.095,.14),.14,.055)
    outcomes=BASE_OUT+[p for p in ['Shooting foul drawn','Other foul drawn','Offensive foul'] if p in set(E.target)]
    for name,y in zip(outcomes,np.linspace(.91,.22,len(outcomes))):box(name,name,(.875,float(y)),.19,.044)
    box('unknown','Not reconstructed: inbound receiver,\nshot/foul combinations, complete\npossession paths',(.86,.08),.24,.09,True)
    maxw=E.display_weight.max();samecounts={}
    for highlight in [False,True]:
        for idx,row in E[E.highlight.eq(highlight)].iterrows():
            s,q=row.source,row.target
            if s not in xy or q not in xy:raise ValueError('Unexpected node '+str((s,q)))
            color='#ef3b22' if highlight else '#bdc2c6';alpha=.86 if highlight else .43;lw=.25+8*np.sqrt(row.display_weight/maxw)
            if s==q:
                x,y=xy[s];arrow=FancyArrowPatch((x-.019,y+.043),(x+.020,y+.043),connectionstyle='arc3,rad=-2.1',arrowstyle='-|>',mutation_scale=11+lw,linewidth=lw,color=color,alpha=alpha,zorder=2 if highlight else 1);ax.add_patch(arrow)
            else:
                rad=.12 if s in POS and q in POS else .025
                arrow=FancyArrowPatch(xy[s],xy[q],patchA=patches[s],patchB=patches[q],connectionstyle=f'arc3,rad={rad}',arrowstyle='-|>',mutation_scale=12+lw,linewidth=lw,color=color,alpha=alpha,zorder=2 if highlight else 1);ax.add_patch(arrow)
    fig.suptitle(title,fontsize=17,y=.98)
    subtitle=('Historical pilot: incomplete tracking-inferred passes + recorded play-by-play events' if historical else 'Original NBA API season passing counts + recorded season event totals')
    fig.text(.5,.935,subtitle,ha='center',fontsize=12)
    text=(f'FG display scale: {scale:.3f} × counts; FG-outcome edges together = total passes' if mode=='equal_fg' else 'Observed counts; FG edges are not rescaled')
    fig.text(.5,.025,text+'\nRed: largest edges covering 60% of display weight. Width: square root of display weight. Role loops are passes between different players with the same listed position.\nLeft arrows identify recorded rebounders/steal winners, not their next receiver. Event layers have different coverage and are not a complete Markov transition model.',ha='center',fontsize=9)
    fig.subplots_adjust(left=.01,right=.99,top=.91,bottom=.10)
    return fig,E,scale


def season_events(roster,roles):
    rows=[]
    for p in POS:
        r=roster[np.asarray(roles)==p]
        fg2m=float((r.FGM-r.FG3M).sum());fg2a=float((r.FGA-r.FG3A).sum());fg3m=float(r.FG3M.sum());fg3a=float(r.FG3A.sum())
        values={'2PT made':fg2m,'2PT missed':fg2a-fg2m,'3PT made':fg3m,'3PT missed':fg3a-fg3m,'FT made':float(r.FTM.sum()),'FT missed':float((r.FTA-r.FTM).sum()),'Turnover':float(r.TOV.sum())}
        if min(values.values())<0:raise ValueError('Invalid shooting splits')
        for target,count in values.items():rows.append(dict(source=p,target=target,raw_count=count,kind='recorded_event'))
        rows.append(dict(source='Rebound',target=p,raw_count=float(r.REB.sum()),kind='recorded_event'));rows.append(dict(source='Steal',target=p,raw_count=float(r.STL.sum()),kind='recorded_event'))
    return rows


def fit(d,x,y):
    X=np.column_stack([np.ones(len(d)),d[x]]);v=d[y].to_numpy();b=np.linalg.lstsq(X,v,rcond=None)[0];e=v-X@b;bread=np.linalg.inv(X.T@X)
    if d.season.nunique()>1:
        G=d.team.nunique();meat=np.zeros((2,2))
        for team in d.team.unique():
            k=d.team.eq(team).to_numpy();u=X[k].T@e[k];meat+=np.outer(u,u)
        cov=bread@meat@bread*G/(G-1)*(len(d)-1)/(len(d)-2);df=G-1;kind='franchise-clustered'
    else:
        h=np.sum(X*(X@bread),axis=1);u=e/(1-h);cov=bread@(X.T@(X*u[:,None]**2))@bread;df=len(d)-2;kind='HC3'
    q=linregress(d[x],v);se=np.sqrt(cov[1,1]);return dict(n=len(d),r=q.rvalue,r_squared=q.rvalue**2,slope=b[1],intercept=b[0],p=2*t.sf(abs(b[1]/se),df),standard_errors=kind),cov,df


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--position-networks',action='store_true');p.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'nba_passing_output');p.add_argument('--destination',type=Path);p.add_argument('--metadata-dir',type=Path,default=Path(__file__).resolve().with_name('position_network_metadata'));p.add_argument('--team',default='ALL');p.add_argument('--season',default='ALL');p.add_argument('--no-historical',action='store_true');args=p.parse_args(argv)
    root=args.output_dir.expanduser();out=args.destination or root/'position_networks';out.mkdir(parents=True,exist_ok=True);metadata=args.metadata_dir.expanduser();mapping=pd.read_csv(metadata/'player_positions.csv');rows=[];sources=[];assignments=[]
    files=sorted((root/'full_roster_entropy_centrality/historical_full_rosters').glob('20*/*/passing_matrix.csv'));files=[f for f in files if (args.team=='ALL' or f.parent.name==args.team.upper()) and (args.season=='ALL' or f.parent.parent.name==args.season)]
    if not files:raise ValueError('No matching NBA API season matrices')
    with PdfPages(out/'position_event_networks.pdf') as pdf:
        for k,f in enumerate(files,1):
            season,team=f.parent.parent.name,f.parent.name;frame=pd.read_csv(f,index_col=0);ids=frame.index.astype(int).tolist();assert frame.columns.astype(int).tolist()==ids
            roster=pd.read_csv(f.with_name('roster.csv')).set_index('PLAYER_ID').loc[ids].reset_index();roles=mapping[(mapping.season==season)&(mapping.team==team)].set_index('player_id').loc[ids];assert len(roles)==len(ids) and roles.position.isin(POS).all();B,T=aggregate(frame.to_numpy(float),roles.position.tolist());actions=T.T@(roster.FGA+.44*roster.FTA+roster.TOV).to_numpy(float);metrics,c,u,prob=role_metrics(B,actions);metrics.update(team=team,season=season,players=len(ids),represented_positions=len(set(roles.position)),absent_listed_positions='|'.join(p for p in POS if p not in set(roles.position)));rows.append(metrics)
            folder=out/'nba_api'/season/team;folder.mkdir(parents=True,exist_ok=True);pd.DataFrame(B,index=POS,columns=POS).to_csv(folder/'position_pass_counts.csv');prob_export=prob.copy();prob_export[B.sum(1)==0]=np.nan;pd.DataFrame(prob_export,index=POS,columns=POS).to_csv(folder/'position_pass_probabilities.csv');pd.DataFrame(dict(position=POS,ec_share=c,offensive_actions=actions,action_share=u,minutes=T.T@roster.MIN.to_numpy(),fga=T.T@roster.FGA.to_numpy())).to_csv(folder/'position_shares.csv',index=False)
            E=event_edges(B,season_events(roster,roles.position.tolist()));fig,E,scale=draw(E,f'{team} | {season} regular season | listed-position network');pdf.savefig(fig);fig.savefig(folder/'position_event_network.png',dpi=140);fig.savefig(folder/'position_event_network.pdf');plt.close(fig);E.to_csv(folder/'event_edges.csv',index=False)
            a=roles.reset_index();a['season']=season;a['team']=team;assignments.append(a)
            for file in [f,f.with_name('roster.csv')]:sources.append(dict(path=str(file),sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
            if k%10==0 or k==1:print(f'{k}/{len(files)} NBA API role networks saved',flush=True)
    d=pd.DataFrame(rows);d.to_csv(out/'position_network_metrics.csv',index=False);pd.concat(assignments).to_csv(out/'used_player_position_assignments.csv',index=False)
    regressions=[]
    with PdfPages(out/'position_entropy_regressions.pdf') as pdf:
        for scope,pop in [('ALL',d),*list(d.groupby('season'))]:
            if len(pop)<5:continue
            fig,axes=plt.subplots(1,2,figsize=(14,6))
            for ax,y,label in [(axes[0],'position_ec_concentration','Position EC concentration'),(axes[1],'position_action_hhi','Offensive-action concentration across positions')]:
                r,cov,df=fit(pop,'position_entropy',y);r.update(season=scope,outcome=y);regressions.append(r);xx=np.linspace(pop.position_entropy.min(),pop.position_entropy.max(),100);X=np.column_stack([np.ones(len(xx)),xx]);pred=r['intercept']+r['slope']*xx;band=t.ppf(.975,df)*np.sqrt(np.maximum(np.sum(X*(X@cov),axis=1),0));ax.fill_between(xx,pred-band,pred+band,color='#c46e35',alpha=.15);ax.plot(xx,pred,c='#c46e35');ax.scatter(pop.position_entropy,pop[y],c=np.where(pop.same_leading_position,'#278699','#b37146'),s=32,alpha=.8)
                if scope!='ALL':
                    for z in pop.itertuples():ax.annotate(z.team,(z.position_entropy,getattr(z,y)),xytext=(3,3),textcoords='offset points',fontsize=6)
                ax.set_title(f'{label}\nn={len(pop)}; r={r["r"]:.3f}; R²={r["r_squared"]:.1%}; robust p={r["p"]:.3g}');ax.set_xlabel('Position conditional passing entropy / log2(5)');ax.set_ylabel(label);ax.grid(alpha=.15)
            fig.suptitle(f'{scope} | five-position passing structure');fig.text(.5,.015,'PG, SG, SF, PF, C are season-listed labels, not simultaneous lineup slots. Within-position passes retained.\nBlue: same leading position for EC/actions; brown: different. Structural exploratory regressions, not player-star dependence.',ha='center',fontsize=9);fig.tight_layout(rect=[0,.07,1,.92]);pdf.savefig(fig);fig.savefig(out/(str(scope)+'_position_entropy_regressions.png'),dpi=150);plt.close(fig)
    R=pd.DataFrame(regressions)
    if R.empty:R=pd.DataFrame(columns=['season','outcome','p'])
    R['holm_p']=np.nan
    for flag in [R.season.eq('ALL'),R.season.ne('ALL')]:
        ix=R.loc[flag,'p'].sort_values().index
        if len(ix):R.loc[ix,'holm_p']=np.minimum(1,np.maximum.accumulate([(len(ix)-j)*R.loc[i,'p'] for j,i in enumerate(ix)]))
    R.to_csv(out/'position_entropy_regressions.csv',index=False)
    coverage_models=[]
    for scope,population in [('ALL',d),('All five labels represented',d[d.represented_positions.eq(5)])]:
        if len(population)<5:continue
        for outcome in ['position_ec_concentration','position_action_hhi']:
            model,_,_=fit(population,'position_entropy',outcome);model.update(scope=scope,outcome=outcome);coverage_models.append(model)
    pd.DataFrame(coverage_models).to_csv(out/'position_coverage_sensitivity.csv',index=False)
    if not args.no_historical:historical(root,out,metadata,sources)
    for f in [*metadata.glob('*.csv'),*metadata.glob('*.json')]:sources.append(dict(path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
    pd.DataFrame(sources).to_csv(out/'source_manifest.csv',index=False)
    (out/'START_HERE.md').write_text('''# Five-position networks

Open position_event_networks.pdf for the NBA API season diagrams. historical_github/historical_position_event_networks.pdf separately shows the three-game 2015–16 tracking/PBP pilot. position_entropy_regressions.pdf uses the season passing networks only. The two data populations are never pooled.

Player positions are season/team-specific listed labels from Basketball-Reference per-game tables. Original NBA API passing and box-score counts are unchanged. All roster assignments are exported. One case, MEM 2024–25, has no SF label in the source; this reflects the labeling scheme, not proof the team played without a small forward. Isolated positions are omitted from the drawing, while the fixed five-category matrix retains a zero category. A complete-five-label regression sensitivity is saved separately. Seven explicit name aliases were reconciled using season/team records. Games-played agree for all 3,922 player-team-season assignments. Listed position is not the actual role occupied at each moment, and no one-player-per-position constraint is imposed. Multiple listed PGs can play together. No minutes cutoff.

Each role matrix is T.T @ A @ T. Total passes are preserved exactly. Diagonal role loops are real passes between DIFFERENT players assigned the SAME position. They are not player self-passes. Role entropy includes these loops and is divided by log2(5), since all five destination categories are possible. A cross-position-only entropy sensitivity excludes these passes and uses log2(4); it concerns a different conditional sample and is saved in metrics. Role EC uses B+B.T including within-position activity, L1 player-role shares, and (HHI−1/5)/(1−1/5). Offensive-action HHI uses positional shares of FGA+0.44*FTA+TOV. These are measures of positional involvement, not player starness or efficiency.

The diagram follows the requested layout. Internal arrows count passes. Rebound/steal arrows identify the listed position of the recorded rebounder/steal winner, NOT the next receiver. Right arrows summarize observed shot/FT/turnover events. Inbounds and unobserved event combinations are marked unavailable and receive no fabricated arrows. Unassigned team rebounds/turnovers are not attributed to a position; their omitted counts are recorded in the historical coverage table. Historical PBP adds recorded shooting-foul/other-foul-drawn events; these are additional event layers, not mutually exclusive possession endings. They can accompany shots/FTs. No complete possession path is claimed.

For readability and your earlier basket-edge request, the four FG outcome categories together have display weight equal to all passing edges: display_weight = raw_count * total_passes / total_FGA for FG outcomes. Other event counts are unchanged. event_edges.csv retains BOTH raw_count and display_weight. Red denotes the largest edges cumulatively covering 60% of display weight; widths use a square-root display transform. Scaling/highlighting never affect entropy, passing EC or regression inputs. The entire diagram is not a Markov transition matrix: event layers differ in coverage, especially the incomplete historical inferred passes. Only position_pass_probabilities.csv is a conditional passing probability matrix; categories with no outgoing passes have undefined (blank/NaN) rows, not fabricated outgoing transitions.

Season regression uncertainty clusters repeated seasons by franchise; individual seasons use HC3. Holm correction covers two pooled tests and separately all season tests. These are exploratory and share a pass matrix between entropy and EC. Five fixed categories reduce dependence on roster count, but positional exposure, label choices, season aggregation and within-position pooling still matter. Season labels do not reconstruct simultaneous lineups. Passing diversity between listed positions is a different outcome from diversity between individual players.

Run python3 teamgraphs.py --position-networks. The standalone module needs --output-dir pointing at the existing SportsAnalysis output folder and --metadata-dir pointing at the included position metadata. Optional --team and --season restrict season diagrams. --no-historical skips the separate historical pilot. No downloads are performed by this analysis module.
''')
    print(R[R.season.eq('ALL')].to_string(index=False));print('Saved',out);return 0


def historical(root,out,metadata,sources):
    mapping=pd.read_csv(metadata/'historical_positions.csv',dtype={'game_id':str});base=root/'sportvu_passing_sequences';passes=pd.read_csv(base/'all_inferred_passes.csv',dtype={'game_id':str});folder=out/'historical_github';folder.mkdir(parents=True,exist_ok=True);eventrows=[];coverage=[]
    raw=root.parent/'sportvu_sequence_data/raw';verified=json.loads((raw/'verified_sources.json').read_text());checks={f['file']:f for f in verified['files']}
    with PdfPages(folder/'historical_position_event_networks.pdf') as pdf:
        for game,gamepos in mapping.groupby('game_id'):
            f=raw/(game+'_events.csv');data=f.read_bytes()
            if hashlib.sha256(data).hexdigest()!=checks[f.name]['sha256']:raise ValueError('PBP hash mismatch')
            pbp=pd.read_csv(f);sources.append(dict(path=str(f),sha256=hashlib.sha256(data).hexdigest()));position=dict(zip(gamepos.player_id,gamepos.position));teams=dict(zip(gamepos.player_id,gamepos.team));teamids=dict(zip(gamepos.player_id,gamepos.team_id))
            for team,g in gamepos.groupby('team'):
                p=passes[passes.game_id.eq(game)&passes.source.isin(g.player_id)];B=np.zeros((5,5))
                for z in p.itertuples():B[POS.index(position[z.source]),POS.index(position[z.target])]+=1
                events=[]
                for z in pbp.itertuples():
                    desc=' '.join(str(getattr(z,c,'')) for c in ['HOMEDESCRIPTION','VISITORDESCRIPTION','NEUTRALDESCRIPTION']);kind=int(z.EVENTMSGTYPE);actor=int(z.PLAYER1_ID);recipient=int(z.PLAYER2_ID)
                    def add(src,target,player,eventkind='recorded_pbp'):
                        if player not in position or teams.get(player)!=team:return
                        events.append(dict(source=src if src else position[player],target=target if target else position[player],raw_count=1.,kind=eventkind));eventrows.append(dict(game_id=game,team=team,eventnum=z.EVENTNUM,event_type=kind,actor_player_id=player,source=src if src else position[player],target=target if target else position[player],description=desc))
                    if kind in [1,2] and teams.get(actor)==team:add(None,('3PT' if '3PT' in desc.upper() else '2PT')+(' made' if kind==1 else ' missed'),actor)
                    elif kind==3 and teams.get(actor)==team:add(None,'FT missed' if 'MISS' in desc.upper() else 'FT made',actor)
                    elif kind==4:add('Rebound',None,actor)
                    elif kind==5:
                        add(None,'Turnover',actor)
                        if 'STEAL' in desc.upper():add('Steal',None,recipient)
                    elif kind==6:
                        # Foul drawn is assigned to the explicitly recorded recipient; no inferred foul/shot success combinations.
                        if int(z.EVENTMSGACTIONTYPE)==2:add(None,'Shooting foul drawn',recipient)
                        elif int(z.EVENTMSGACTIONTYPE) in [1,3,11,28]:add(None,'Other foul drawn',recipient)
                        elif int(z.EVENTMSGACTIONTYPE)==4:add(None,'Offensive foul',actor)
                ev=pd.DataFrame(events)
                if not ev.empty:ev=ev.groupby(['source','target','kind'],as_index=False).raw_count.sum().to_dict('records')
                else:ev=[]
                E=event_edges(B,ev);fig,E,scale=draw(E,f'{team} | game {game} | 2015–16 historical example',historical=True);pdf.savefig(fig);fig.savefig(folder/(game+'_'+team+'.png'),dpi=150);fig.savefig(folder/(game+'_'+team+'.pdf'));plt.close(fig);E.to_csv(folder/(game+'_'+team+'_edges.csv'),index=False);pd.DataFrame(B,index=POS,columns=POS).to_csv(folder/(game+'_'+team+'_position_pass_counts.csv'))
                tid=int(g.team_id.iloc[0]);team_events=pbp[pbp.PLAYER1_TEAM_ID.eq(tid)|pbp.PLAYER1_ID.eq(tid)];expected_fg=int(team_events.EVENTMSGTYPE.isin([1,2]).sum());mapped_fg=int(E.loc[E.target.isin(SHOT),'raw_count'].sum());assert expected_fg==mapped_fg
                coverage.append(dict(game_id=game,team=team,inferred_passes=len(p),pbp_events=len(pbp),fg_events=expected_fg,mapped_fg_events=mapped_fg,team_turnover_events=int(team_events.EVENTMSGTYPE.eq(5).sum()),mapped_player_turnovers=int(E.loc[E.target.eq('Turnover'),'raw_count'].sum()),unassigned_team_turnovers=int(team_events.EVENTMSGTYPE.eq(5).sum())-int(E.loc[E.target.eq('Turnover'),'raw_count'].sum()),unassigned_team_rebounds=int((team_events.EVENTMSGTYPE.eq(4)&~team_events.PLAYER1_ID.isin(gamepos.player_id)).sum()),inbound_receiver_links=0,complete_possession_sequences=False))
    pd.DataFrame(eventrows).to_csv(folder/'pbp_event_edge_audit.csv',index=False);pd.DataFrame(coverage).to_csv(folder/'coverage.csv',index=False);sources.append(dict(path=str(base/'all_inferred_passes.csv'),sha256=hashlib.sha256((base/'all_inferred_passes.csv').read_bytes()).hexdigest()))

if __name__=='__main__':raise SystemExit(main())
