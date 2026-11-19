"""Checks the basket normalization independently of NBA availability."""
import numpy as np
import pandas as pd
from basket_networks import BASKET, build_networks

A = pd.DataFrame([[0, 30, 0], [10, 0, 0], [0, 0, 0]],
                 index=[1, 2, 3], columns=[1, 2, 3])
roster = pd.DataFrame({'PLAYER_ID': [1, 2, 3],
                       'PLAYER_NAME': ['One', 'Two', 'Three'], 'FGA': [5, 10, 5]})
for ratio in [.5, 1, 2]:
    original, augmented, audit = build_networks(A, roster, ratio)
    assert np.allclose(audit.basket_weight, np.array([10, 20, 10])*ratio)
    assert augmented[1][2]['weight'] == 30
    assert augmented[2][1]['weight'] == 10
    assert augmented.out_degree(BASKET) == 0
    assert original.degree(3) == 0 and augmented.degree(3) == 1
for invalid in [roster.assign(FGA=[np.nan, 10, 5]),
                roster.assign(FGA=[-1, 10, 5]),
                roster.assign(FGA=[0, 0, 0]), roster.iloc[:2]]:
    try:
        build_networks(A, invalid)
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid FGA or mismatched roster accepted')
print('PASS: basket scaling, original passes, sink, isolates and missing-data handling')
