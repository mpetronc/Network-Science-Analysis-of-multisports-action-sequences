"""Keep only official teammate recipients within each requested date window."""
import pandas as pd

def add_profile(data,M,ix,r,frame):
 if not hasattr(data,'_scope_players'):data._scope_players={};data._invalid_edges=[];data._invalid_edge_keys=set()
 key=(r['team_id'],r['start'],r['end'])
 if key not in data._scope_players:data._scope_players[key]=set(data.players.loc[data.players.TEAM_ID.eq(r['team_id'])&data.players.GAME_DATE.between(r['start'],r['end']),'PLAYER_ID'])
 known=data._scope_players[key];assert known<=set(ix) and r['player_id'] in known;changed=False
 for e in frame.itertuples():
  if e.PASS<=0:continue
  receiver=int(e.PASS_TEAMMATE_PLAYER_ID)
  if receiver not in known or receiver not in ix:
   edgekey=(r['team'],r['kind'],r['start'],r['end'],r['player_id'],receiver,float(e.PASS))
   if edgekey not in data._invalid_edge_keys:
    data._invalid_edge_keys.add(edgekey);data._invalid_edges.append(dict(team=r['team'],kind=r['kind'],start=r['start'],end=r['end'],sender_id=r['player_id'],receiver_id=receiver,passes=float(e.PASS),reason='Recipient is not an official participant for this team in the requested date window'));changed=True
   continue
  M[ix[r['player_id']],ix[receiver]]+=e.PASS
 if changed:pd.DataFrame(data._invalid_edges).to_csv(data.out/'excluded_invalid_recipient_edges.csv',index=False)
