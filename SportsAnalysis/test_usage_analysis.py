import numpy as np
import pandas as pd
from usage_analysis import usage_metrics
r = pd.DataFrame({'PLAYER_ID':[1,2], 'PLAYER_NAME':['One','Two'], 'FGA':[10,20], 'FTA':[0,0], 'TOV':[0,0], 'MIN':[100,200]})
u, actions, shares, rates = usage_metrics(r)
assert np.allclose(shares, [1/3,2/3])
assert np.isclose(u['usage_hhi'], 5/9)
assert np.allclose(rates, [20,20])
try:
    usage_metrics(r.assign(FTA=[np.nan,0]))
except ValueError:
    pass
else:
    raise AssertionError('Missing input accepted')
print('PASS: known usage shares, HHI, estimated rate and missing data handling')
