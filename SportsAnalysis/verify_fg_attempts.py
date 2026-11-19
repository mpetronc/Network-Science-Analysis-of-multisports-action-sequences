"""Independent integrity checks for generated whole-season graphs."""
from pathlib import Path
import sys
import pandas as pd
import numpy as np
from fg_attempt_networks import classify,CATEGORIES
s=pd.DataFrame(dict(GAME_ID=['a']*5,GAME_EVENT_ID=range(5),SHOT_TYPE=['2PT Field Goal']*4+['3PT Field Goal'],ACTION_TYPE=['Driving Layup Shot','Dunk Shot','Hook Shot','Floating Jump Shot','Jump Shot'],SHOT_ATTEMPTED_FLAG=[1]*5))
assert list(classify(s))==['LAYUP_DUNK','LAYUP_DUNK','OTHER_2PT','OTHER_2PT','THREE_PT']
for bad in [pd.concat([s,s.iloc[[0]]]),s.assign(ACTION_TYPE=None),s.assign(SHOT_TYPE='unknown')]:
 try: classify(bad)
 except ValueError: pass
 else: raise AssertionError('Invalid data accepted')
root=Path(sys.argv[1]);audit=pd.read_csv(root/'validation.csv');assert len(audit)==180
for row in audit.itertuples():
 e=pd.read_csv(root/row.season/row.team/'edges.csv')
 shots=e[e.edge_type=='shot_attempt'];passes=e[e.edge_type=='pass']
 assert set(shots.target)<=set(CATEGORIES)
 assert not e.source.astype(str).isin(CATEGORIES).any()
 assert shots.raw_count.sum()==row.fga
 assert np.isclose(shots.weight.sum(),passes.weight.sum())
 assert np.array_equal(passes.weight,passes.raw_count)
 assert not (e.source.astype(str)==e.target.astype(str)).any()
print('Verified 180 team-seasons, shot partition/scaling, sinks, no self-loops, pass counts, classifier and invalid inputs.')
