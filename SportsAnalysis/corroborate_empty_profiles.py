from pathlib import Path
import argparse,json
import pandas as pd
from nba_api.stats.endpoints import playerdashptpass
from teamgraphs import NBAClient,Config,extract_team_metrics

def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,required=True);a=p.parse_args(argv);out=a.data_dir;plan=pd.read_csv(out/'network_download_plan.csv');client=NBAClient(Config(season='2025-26',cache_dir=out/'passing_api_cache'));confirmed=[]
 for file in (out/'passing_profiles').glob('*/*/*.json'):
  info=json.loads(file.read_text())
  if info.get('complete') or info.get('error')!='Empty profile without independently verified zero':continue
  others=plan[plan.team.eq(info['team'])&plan.start.eq(info['start'])&plan.end.eq(info['end'])&plan.player_id.ne(info['player_id'])];total=0;complete=len(others)>0;proof=[]
  for z in others.itertuples():
   audit=out/'passing_profiles'/z.team/(z.start+'_'+z.end)/(str(z.player_id)+'.json')
   if not audit.exists() or not json.loads(audit.read_text()).get('complete'):complete=False;break
   params=dict(player_id=z.player_id,team_id=z.team_id,season='2025-26',season_type_all_star='Regular Season',per_mode_simple='Totals',date_from_nullable=pd.Timestamp(z.start).strftime('%m/%d/%Y'),date_to_nullable=pd.Timestamp(z.end).strftime('%m/%d/%Y'));query=playerdashptpass.PlayerDashPtPass(**params,get_request=False);snap=client.cache.read(query.endpoint,dict(query.parameters))
   if snap is None:complete=False;break
   received=extract_team_metrics(snap['payload'],'PassesReceived',{'PLAYER_ID','TEAM_ID','PASS_TEAMMATE_PLAYER_ID','PASS','G'})
   if len(received):
    assert received.PLAYER_ID.eq(z.player_id).all() and received.TEAM_ID.eq(z.team_id).all() and received.G.le(z.expected_games).all();total+=float(received.loc[received.PASS_TEAMMATE_PLAYER_ID.eq(info['player_id']),'PASS'].sum())
   proof.append(dict(receiver_id=z.player_id,source=str(client.cache.path_for(query.endpoint,dict(query.parameters)))))
  if complete and total==0:
   info.update(complete=True,error='',zero_confirmation='All other participating players receiver-side tracking profiles report zero passes from this sender; no self-pass is possible',receiver_profiles=proof);file.write_text(json.dumps(info,indent=2));confirmed.append((info['team'],info['player_id'],info['start']))
 client.close();print('Receiver-side zero confirmations',confirmed)
if __name__=='__main__':main()
