"""Reuse unfiltered raw NBA full-season payloads only for an exact full-season window."""
from pathlib import Path
import json,hashlib,shutil
import pandas as pd
from teamgraphs import validate_passing

def seed(data):
 if not any(r['kind']=='eval_total' for r in data.plan):return
 cache=Path.home()/'nba_passing_cache';lookup={}
 for source in cache.glob('playerdashptpass_*.json'):
  d=json.loads(source.read_text());p=d.get('parameters',{})
  if p.get('PerMode')!='Totals' or p.get('SeasonType')!='Regular Season' or p.get('DateFrom') or p.get('DateTo'):continue
  if any(p.get(k) not in ['',None,0,'0'] for k in ['LastNGames','Month','OpponentTeamID','Location','Outcome','SeasonSegment','VsConference','VsDivision','ISTRound']):continue
  if p.get('LeagueID') not in ['00',None]:continue
  lookup[(p.get('Season'),int(p.get('TeamID',0)),int(p.get('PlayerID',0)))]=(source,d)
 n=0
 for r in data.plan:
  if r['kind']!='eval_total':continue
  file=data.file(r);audit=file.with_suffix('.json')
  if file.exists() and audit.exists():continue
  g=data.games[data.games.TEAM_ID.eq(r['team_id'])]
  if r['start']!=g.GAME_DATE.min() or r['end']!=g.GAME_DATE.max():continue
  first=int(g.GAME_DATE.min()[:4]);month=int(g.GAME_DATE.min()[5:7]);year=first if month>=7 else first-1;key=(f'{year}-{(year+1)%100:02d}',r['team_id'],r['player_id'])
  if key not in lookup:continue
  source,d=lookup[key]
  try:f=validate_passing(d['payload'],r['player_id'],r['team_id'])
  except Exception:continue
  if f.empty or f.G.gt(r['expected_games']).any():continue
  known=set(data.players.loc[data.players.TEAM_ID.eq(r['team_id']),'PLAYER_ID'])
  if not set(f.loc[f.PASS.gt(0),'PASS_TEAMMATE_PLAYER_ID'].astype(int))<=known:continue
  rawfolder=data.out/'unfiltered_full_season_cache';rawfolder.mkdir(exist_ok=True);copy=rawfolder/source.name
  if not copy.exists():shutil.copy2(source,copy)
  source=copy.resolve()
  file.parent.mkdir(parents=True,exist_ok=True);tmp=file.with_suffix('.seed.tmp.csv');f.to_csv(tmp,index=False);tmp.replace(file);info=dict(r,complete=True,error='',profile_passes=float(f.PASS.sum()),source_raw_cache=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),original_parameters=d['parameters'],scope_equivalence='Unfiltered regular-season Totals; target dates equal first and last official team games; no minutes filter applied');tmp=audit.with_suffix('.seed.tmp.json');tmp.write_text(json.dumps(info,indent=2));tmp.replace(audit);n+=1
 print('Reused exact full-season raw profiles',n,flush=True)
