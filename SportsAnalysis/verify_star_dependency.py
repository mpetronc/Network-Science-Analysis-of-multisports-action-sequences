"""Audit chronology, absence labels, lagged predictors and the independent full panel fit."""
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('study_dir',type=Path);p.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'nba_passing_output');p.add_argument('--cache',type=Path);a=p.parse_args(argv);out=a.study_dir;root=a.output_dir;cache=a.cache or out/'game_cache';allplayers=[]
 for season in ['2020-21','2021-22','2022-23','2023-24','2024-25','2025-26']:
  f=cache/f'{season}_player_base.csv.gz';P=pd.read_csv(f,dtype={'GAME_ID':str});P['season']=season;P.GAME_ID=P.GAME_ID.str.zfill(10);allplayers.append(P)
 players=pd.concat(allplayers,ignore_index=True);appearance=players.groupby(['season','TEAM_ID','GAME_ID']).PLAYER_ID.agg(set).to_dict();games=pd.read_csv(out/'pregame_team_data.csv.gz',dtype={'GAME_ID':str});games['date']=pd.to_datetime(games.date);prior_metrics=pd.read_csv(root/'court_zone_networks/regressions/four_metrics.csv');fits=pd.read_csv(out/'primary_panel_results.csv');elig=pd.read_csv(out/'eligibility.csv');coverage={};history=players.groupby(['season','PLAYER_ID']).MIN.sum()
 # Verify pregame defense by explicit date filtering (independent of cumsum implementation).
 for z in games[games.season.ne('2020-21')].sample(150,random_state=11).itertuples():
  same=games[(games.season==z.season)&games.TEAM_ID.eq(z.opponent_id)];previous=same[same.date<z.date];prevyear=str(int(z.season[:4])-1)+'-'+z.season[2:4];before=games[(games.season==prevyear)&games.TEAM_ID.eq(z.opponent_id)];rate=100*before.opponent_points.sum()/before.opponent_possessions.sum();expected=100*(previous.opponent_points.sum()+1000*rate/100)/(previous.opponent_possessions.sum()+1000);assert np.isclose(expected,z.opp_def_pregame)
 for policy in ['passing_hub','leading_shooter']:
  d=pd.read_csv(out/(policy+'_primary_games.csv.gz'),dtype={'GAME_ID':str});d.date=pd.to_datetime(d.date);d.landmark_end_date=pd.to_datetime(d.landmark_end_date);assert np.isfinite(d[['off_rating','pre_ec','pre_shooting_hhi','other_missing','opponent_missing','opp_def_pregame']]).all().all()
  assert not d.duplicated(['season','team','GAME_ID']).any() and d.game_number.gt(10).all() and d.opponent_game_number.gt(10).all() and (d.date>d.landmark_end_date).all()
  for z in d.itertuples():assert z.absent==int(z.star_id not in appearance[(z.season,z.TEAM_ID,z.GAME_ID)])
  for unit,g in d.groupby('unit'):
   z=g.iloc[0];assert z.absent==int(z.star_id not in appearance[(z.season,z.TEAM_ID,z.GAME_ID)])
   prior=str(int(z.season[:4])-1)+'-'+z.season[2:4];assert z.prior_season==prior
   original=prior_metrics[(prior_metrics.season==prior)&prior_metrics.team.eq(z.team)].iloc[0];assert int(z.star_id)==int(original.passing_leader_id if policy=='passing_hub' else original.shooting_leader_id)
   folder=root/'full_roster_entropy_centrality/historical_full_rosters'/prior/z.team;R=pd.read_csv(folder/'roster.csv').set_index('PLAYER_ID');A=pd.read_csv(folder/'passing_matrix.csv',index_col=0);act=A.to_numpy().sum(0)+A.to_numpy().sum(1)>0;B=A.to_numpy()[np.ix_(act,act)];_,V=np.linalg.eigh(B+B.T);v=np.abs(V[:,-1]);v/=v.sum();ec=(v@v-1/len(v))/(1-1/len(v));assert np.isclose(ec,z.pre_ec,atol=1e-8);q=R.FGA/R.FGA.sum();assert np.isclose(q@q,z.pre_shooting_hhi)
   early=games[(games.season==z.season)&games.TEAM_ID.eq(z.TEAM_ID)&games.game_number.le(10)];roster=set().union(*[appearance[(z.season,z.TEAM_ID,game)] for game in early.GAME_ID]);assert int(z.star_id) in roster
   key=history.loc[prior].reindex(sorted(roster-{int(z.star_id)})).dropna().sort_values(ascending=False,kind='stable').head(3);weights=(key/key.sum()).to_dict()
   for r in g.sample(min(3,len(g)),random_state=3).itertuples():assert np.isclose(r.other_missing,sum(w for p,w in weights.items() if p not in appearance[(r.season,r.TEAM_ID,r.GAME_ID)]))
   own=players[(players.season==z.season)&players.PLAYER_ID.eq(z.star_id)&players.TEAM_ID.ne(z.TEAM_ID)]
   if len(own):assert g.date.max()<pd.to_datetime(own.GAME_DATE).min().normalize()-pd.Timedelta(days=14)
   assert g.absent.eq(0).sum()>=15 and g.absent.eq(1).sum()>=5
  # Independently fit the FULL design, rather than the implementation's partialled-out model.
  cat=pd.Categorical(d.unit);D=np.eye(len(cat.categories))[cat.codes];trend=(d.game_number.to_numpy()-41)/41;controls=d[['home','b2b','rest','opp_b2b','opp_def_pregame','other_missing','opponent_missing']].to_numpy(float);controls[:,4]=(controls[:,4]-110)/10
  Z=np.column_stack([D,D*trend[:,None],controls]);X=np.column_stack([d.absent]+[d.absent*d[c+'_z'] for c in ['pre_ec','pre_shooting_hhi','shared_leader']]);coef=np.linalg.lstsq(np.column_stack([Z,X]),d.off_rating,rcond=None)[0][-4:];record=fits[fits.policy==policy];assert np.allclose(coef,record.coefficient_absence_effect,atol=1e-9)
  coverage[policy]=dict(games=len(d),team_seasons=d.unit.nunique(),franchises=d.team.nunique(),absence_games=int(d.absent.sum()))
 episodes=pd.read_csv(out/'absence_episodes.csv',dtype={'pre_games':str,'absence_game_ids':str})
 for policy,group in episodes.groupby('policy'):
  panel=pd.read_csv(out/(policy+'_primary_games.csv.gz'),dtype={'GAME_ID':str}).set_index(['unit','GAME_ID'])
  for z in group.itertuples():
   pre=panel.loc[[(z.unit,g) for g in z.pre_games.split(';')]];absent=panel.loc[[(z.unit,g) for g in z.absence_game_ids.split(';')]];assert len(pre)==5 and pre.absent.sum()<=1 and absent.absent.eq(1).all() and len(absent)>=2;assert pre.game_number.max()<absent.game_number.min()
 assert len(fits)==8 and fits.holm_p.between(0,1).all();audit=dict(coverage=coverage,chronology_validated=True,prior_only_EC_and_FGA_features=True,EC_independently_recomputed_without_shooting=True,nonappearance_labels_match_complete_player_logs=True,pregame_defense_excludes_current_and_future_games=True,trade_censor_checked=True,newcomer_key_availability_uses_prior_NBA_minutes=True,full_design_OLS_reproduces_primary_coefficients=True,episode_windows_validated=True)
 (out/'verification.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
