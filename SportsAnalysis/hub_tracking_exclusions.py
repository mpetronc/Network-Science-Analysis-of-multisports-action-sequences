"""Exclude unverified absence windows as whole windows, never zero-fill unknown rows."""
from pathlib import Path
import json
import pandas as pd

def prepare(data,meta):
 # Split ambiguous multi-game windows around appearances of players whose outgoing profile is unverified.
 superseded=set();original=list(data.plan)
 keys=sorted({(r['team'],r['start'],r['end']) for r in original if r['kind']=='later_absence'})
 for team,start,end in keys:
  records=[r for r in original if r['team']==team and r['kind']=='later_absence' and r['start']==start and r['end']==end];bad=[]
  for r in records:
   a=json.loads(data.file(r).with_suffix('.json').read_text())
   if not a['complete'] and a['error']=='Empty profile without independently verified zero':bad.append(r['player_id'])
  if not bad:continue
  tid=records[0]['team_id'];g=data.games[data.games.TEAM_ID.eq(tid)&data.games.GAME_DATE.between(start,end)].sort_values(['GAME_DATE','GAME_ID'])
  if len(g)<=1:continue
  problem=set(data.players.loc[data.players.TEAM_ID.eq(tid)&data.players.PLAYER_ID.isin(bad)&data.players.GAME_DATE.between(start,end),'GAME_ID']);parts=[];buffer=[]
  for i,row in g.iterrows():
   if row.GAME_ID in problem:
    if buffer:parts.append(g.loc[buffer]);buffer=[]
    parts.append(g.loc[[i]])
   else:buffer.append(i)
  if buffer:parts.append(g.loc[buffer])
  if len(parts)<=1:continue
  superseded.add((team,'later_absence',start,end))
  for part in parts:data.add_window(team,tid,'later_absence',part)
 if superseded:data.download({'later_absence'})
 groups={}
 for r in data.plan:
  if r['kind'] not in ['later_absence','eval_total']:continue
  key=(r['team'],r['kind'],r['start'],r['end']);groups.setdefault(key,[]).append(r)
 excluded=[dict(team=team,kind=kind,start=start,end=end,reason='Ambiguous multi-game window replaced by smaller date windows',treatment='Superseded source window; no games excluded solely for this record') for team,kind,start,end in sorted(superseded)];unavailable_reference=set();data.excluded_windows=set(superseded)
 for key,records in groups.items():
  if key in superseded:continue
  bad=[(r,json.loads(data.file(r).with_suffix('.json').read_text())) for r in records if not json.loads(data.file(r).with_suffix('.json').read_text())['complete']]
  if not bad:continue
  # Connection failures remain retryable errors. Only an unresolved empty tracking response is treated as data unavailability.
  if any(a['error']!='Empty profile without independently verified zero' for _,a in bad):continue
  team,kind,start,end=key;unavailable_reference.add(team);data.excluded_windows.add(key)
  excluded.append(dict(team=team,kind=kind,start=start,end=end,unverified_player_ids=[r['player_id'] for r,_ in bad],reason='Unresolved empty NBA tracking profile after receiver-side and independent daily zero checks',treatment='Whole absence window excluded from outcome; never zero-filled' if kind=='later_absence' else 'Full-period response unused; later-present reference unavailable'))
 for z in meta:
  z['official_later_hub_absent_games']=z['later_hub_absent_games'];dropped=set()
  for team,kind,start,end in data.excluded_windows-superseded:
   if team==z['team'] and kind=='later_absence':
    g=data.games[data.games.TEAM_ID.eq(z['team_id'])&data.games.GAME_DATE.between(start,end)];dropped.update(g.GAME_ID.tolist())
  z['excluded_later_absence_game_ids']=sorted(set(z['later_absent_game_ids'])&dropped);z['later_absent_game_ids']=[gid for gid in z['later_absent_game_ids'] if gid not in dropped];z['later_hub_absent_games']=len(z['later_absent_game_ids']);z['excluded_later_absence_games']=len(z['excluded_later_absence_game_ids']);z['later_present_reference_available']=z['team'] not in unavailable_reference
  (data.out/z['team']/'evaluation_tracking_scope.json').write_text(json.dumps({k:v for k,v in z.items() if k in ['official_later_hub_absent_games','later_hub_absent_games','excluded_later_absence_game_ids','excluded_later_absence_games','later_present_reference_available']},indent=2))
 (data.out/'tracking_exclusions.json').write_text(json.dumps(excluded,indent=2));print('Tracking window exclusions',excluded,flush=True)
 return unavailable_reference
