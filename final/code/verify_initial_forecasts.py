"""Rebuild the frozen minute and separate passing-role rules from packaged first20 inputs."""
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd
from hub_redistribution_model import redistribute
from split_role_hub_model import forecast


def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--package-root',type=Path,default=Path(__file__).resolve().parent.parent);p.add_argument('--destination',type=Path);a=p.parse_args(argv);root=a.package_root/'data/nba_passing_output';base=root/'all_season_hub_redistribution';split=root/'split_role_hub_redistribution';ledger=pd.read_csv(split/'minutes_and_passing_role_weights.csv');checked=[]
 for specfile in sorted(base.glob('*/*/frozen_specification.json')):
  season,team=specfile.relative_to(base).parts[:2];spec=json.loads(specfile.read_text());s=spec['selection'];ids=s['training_player_ids'];h=ids.index(s['hub_id']);P=pd.read_csv(specfile.parent/'hub_present_training.csv',index_col=0).to_numpy(float);F=pd.read_csv(specfile.parent/'first20_all_passing.csv',index_col=0).to_numpy(float);E=F-P;mp=np.array(spec['training_present_minutes']);ma=np.array(spec['training_absent_minutes']);q=ledger[ledger.season.eq(season)&ledger.team.eq(team)];positions={int(z.player_id):str(z.prior_position).split('/') if str(z.prior_position)!='Unknown' else [] for z in q.itertuples()};historical=dict(per_game=ma/max(1,s['first20_hub_absent_games']),present_per_game=mp/s['first20_hub_present_games']);_,minutes=redistribute(P,mp,historical,s['first20_hub_absent_games'],h,ids,positions);assert np.allclose(minutes['forecast_minutes'],spec['forecast_minutes'])
  M,_=forecast(P,E,mp,ma,np.array(spec['forecast_minutes']),h,s['first20_hub_absent_games']);saved=pd.read_csv(split/season/team/'frozen_passing_role_prediction.csv',index_col=0).to_numpy(float);assert np.allclose(M,saved);assert np.isclose(M.sum(),P.sum()) and not np.diag(M).any() and M[h].sum()+M[:,h].sum()==0;checked.append(dict(season=season,team=team,minute_reconstruction=True,passing_reconstruction=True))
 assert len(checked)==180;result=dict(passed=True,team_seasons=180,minutes_and_initial_passing_forecasts_reconstructed=True)
 if a.destination:a.destination.mkdir(parents=True,exist_ok=True);(a.destination/'initial_reconstruction.json').write_text(json.dumps(result,indent=2))
 print(result);return 0

if __name__=='__main__':main()
