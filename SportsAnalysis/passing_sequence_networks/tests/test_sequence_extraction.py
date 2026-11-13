from pathlib import Path
import runpy,numpy as np,pandas as pd,tempfile,json
m=runpy.run_path(str(Path(__file__).resolve().parents[1]/'sportvu_passing_sequences.py'));Settings=m['Settings'];infer=m['infer'];blank=pd.DataFrame(columns=['EVENTMSGTYPE','PERIOD','PCTIMESTRING'])
xy=np.array([[20,25],[35,25],[50,25],[65,25],[5,45],[80,5],[80,45],[90,10],[90,40],[10,5]],float)
def make(route,heights=None):
 b=np.array(route,float);N=len(b);return dict(quarter=np.ones(N,dtype=int),timestamp_ms=np.arange(N)*40+100000,clock=719-np.arange(N)*.04,player_ids=np.tile(np.arange(1,11),(N,1)),team_ids=np.tile([1]*5+[2]*5,(N,1)),xy=np.tile(xy,(N,1,1)),ball=np.column_stack([b,np.full(N,3.) if heights is None else heights]))
def route(nodes):
 pts=[np.tile(xy[n],(8,1)) for n in nodes];r=[pts[0]]
 for a,b,p in zip(nodes,nodes[1:],pts[1:]):r.extend([np.linspace(xy[a],xy[b],12)[1:-1],p])
 return np.concatenate(r)
def accepted(f,pbp=blank):
 t,s,_,_=infer(f,pbp,Settings());return t[t.accepted],s,t
f=make(route([0,1,2]));a,s,t=accepted(f);assert list(zip(a.source,a.target))==[(1,2),(2,3)] and s[0]['players']==[1,2,3]
A,P,M,Q,e,triples,summary=m['network_tables'](a,s,{1:'A',2:'B',3:'C'});assert A.to_numpy().sum()==2 and M.to_numpy().sum()==1 and triples.iloc[0]['count']==1
f=make(np.tile(xy[0],(40,1)));assert len(accepted(f)[0])==0
f=make(route([0,1]));f['ball'][11:16,2]=14;assert len(accepted(f)[0])==0 and accepted(f)[2].reason.str.contains('high_ball').any()
f=make(route([0,1]));f['team_ids'][:,1]=2;f['team_ids'][:,5]=1;assert len(accepted(f)[0])==0 and accepted(f)[2].reason.str.contains('opponent').any()
f=make(route([0,1]));f['timestamp_ms'][13:]+=1000;assert len(accepted(f)[0])==0
f=make(route([0,1]));f['player_ids'][13:,9]=99;assert len(accepted(f)[0])==0
f=make(route([0,1]));pbp=pd.DataFrame([dict(EVENTMSGTYPE=2,PERIOD=1,PCTIMESTRING='11:'+str(f['clock'][13]-660))]);assert len(accepted(f,pbp)[0])==0
f=make(route([0,1,2,3]));f['ball'][29:34,2]=14;a,s,t=accepted(f);assert [x['players'] for x in s]==[[1,2],[3,4]];assert not m['network_tables'](a,s,{})[5].shape[0]
f=make(route([0,1]));f['xy'][:,1]=[24,25];f['ball'][:,0]=np.clip(f['ball'][:,0],20,24);assert len(accepted(f)[0])==0
# Overlapping event windows must not duplicate the same timestamp.
with tempfile.TemporaryDirectory() as td:
 p=Path(td);f=make(route([0,1]));mom=[]
 for k in range(len(f['clock'])):mom.append([1,int(f['timestamp_ms'][k]),float(f['clock'][k]),24,None,[[-1,-1,*f['ball'][k].tolist()]]+[[int(tid),int(pid),*pos.tolist(),0] for tid,pid,pos in zip(f['team_ids'][k],f['player_ids'][k],f['xy'][k])]])
 home=dict(name='One',teamid=1,abbreviation='ONE',players=[dict(playerid=i,firstname='P',lastname=str(i)) for i in range(1,6)]);away=dict(name='Two',teamid=2,abbreviation='TWO',players=[dict(playerid=i,firstname='P',lastname=str(i)) for i in range(6,11)]);d=dict(gameid='synthetic',gamedate='2000-01-01',events=[dict(eventId=1,home=home,visitor=away,moments=mom),dict(eventId=2,home=home,visitor=away,moments=mom[5:])]);path=p/'synthetic.json';path.write_text(json.dumps(d));m['prepare'](path,p/'cache');meta=json.loads((p/'cache/synthetic_metadata.json').read_text());assert meta['unique_frames']==len(mom) and meta['duplicate_occurrences']==len(mom)-5
 d=json.loads(json.dumps(d));d['events'][1]['moments'][0][5][0][2]+=1;path.write_text(json.dumps(d));m['prepare'](path,p/'cache');meta=json.loads((p/'cache/synthetic_metadata.json').read_text());assert meta['conflicting_timestamps_removed']==1 and meta['unique_frames']==len(mom)-1
print('PASS: true-pass chronology, second-order triples, dribbles, shots, opponent control, tracking gaps, substitutions, PBP boundaries, rejected-transfer chain breaks, close handoffs, and event-window deduplication.')
