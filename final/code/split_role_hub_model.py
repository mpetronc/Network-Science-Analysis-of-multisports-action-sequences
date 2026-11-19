"""Separate extra playing time from outgoing and incoming passing responsibility."""
import numpy as np

RATE_PRIOR_MINUTES = 48.0
HISTORICAL_PRIOR_GAMES = 5.0


def _normalize(values, fallback):
    values = np.asarray(values, float)
    if values.sum() <= 0:
        values = np.asarray(fallback, float)
    assert np.isfinite(values).all() and (values >= 0).all() and values.sum() > 0
    return values / values.sum()


def forecast(P, early_absent, present_minutes, absent_minutes, forecast_minutes,
             hub, early_absent_games):
    """Inputs are training counts/minutes and a training-only minutes forecast."""
    P = np.asarray(P, float)
    early_absent = np.asarray(early_absent, float)
    mp = np.asarray(present_minutes, float)
    ma = np.asarray(absent_minutes, float)
    fm = np.asarray(forecast_minutes, float)
    assert P.shape == early_absent.shape == (len(mp), len(mp))
    assert np.isfinite(P).all() and (P >= 0).all()
    assert np.isfinite(early_absent).all() and (early_absent >= 0).all()
    assert all(np.isfinite(v).all() and (v >= 0).all() for v in [mp, ma, fm])
    assert mp[hub] > 0 and fm[hub] == 0
    assert np.isclose(fm.sum(), 240) and fm.max() <= 48 + 1e-7
    eligible = (mp > 0) & (fm > 0)
    eligible[hub] = False
    capacity = fm * eligible
    details = {}
    weights = []
    for name, axis in [('sender', 1), ('receiver', 0)]:
        counts = P.sum(axis=axis)
        absent_counts = early_absent.sum(axis=axis)
        peer_rate = counts[eligible].sum() / mp[eligible].sum()
        rate = (counts + RATE_PRIOR_MINUTES * peer_rate) / (mp + RATE_PRIOR_MINUTES)
        rate[~eligible] = 0
        base_weight = _normalize(fm * rate * eligible, capacity)
        absence_rate = (absent_counts + RATE_PRIOR_MINUTES * rate) / (ma + RATE_PRIOR_MINUTES)
        gain = fm * np.maximum(absence_rate - rate, 0) * eligible
        alpha = (early_absent_games / (early_absent_games + HISTORICAL_PRIOR_GAMES)
                 if early_absent_games > 0 and gain.sum() > 0 else 0.0)
        gain_weight = _normalize(gain, base_weight)
        weight = (1 - alpha) * base_weight + alpha * gain_weight
        weight[~eligible] = 0
        weight /= weight.sum()
        weights.append(weight)
        details[name] = dict(peer_passes_per_minute=float(peer_rate),
                             shrunk_present_passes_per_minute=rate.tolist(),
                             shrunk_absent_passes_per_minute=absence_rate.tolist(),
                             forecast_activity_weights=base_weight.tolist(),
                             historical_gain_weights=gain_weight.tolist(),
                             historical_blend=float(alpha), weights=weight.tolist())
    D = P.copy()
    D[hub, :] = 0
    D[:, hub] = 0
    M = D.copy()
    route_fallbacks = 0

    def route(weight, exclude):
        nonlocal route_fallbacks
        result = weight.copy()
        result[exclude] = 0
        result[hub] = 0
        if result.sum() <= 0:
            result = capacity.copy()
            result[exclude] = 0
            route_fallbacks += 1
        return _normalize(result, result)

    for j in range(len(mp)):
        if j != hub and P[hub, j] > 0:
            M[:, j] += P[hub, j] * route(weights[0], j)
    for i in range(len(mp)):
        if i != hub and P[i, hub] > 0:
            M[i, :] += P[i, hub] * route(weights[1], i)
    assert np.isfinite(M).all() and (M >= 0).all()
    assert np.isclose(M.sum(), P.sum())
    assert not np.diag(M).any() and M[hub].sum() + M[:, hub].sum() == 0
    details.update(rate_prior_minutes=RATE_PRIOR_MINUTES,
                   historical_prior_games=HISTORICAL_PRIOR_GAMES,
                   eligible=eligible.tolist(), forecast_minutes=fm.tolist(),
                   route_fallbacks=route_fallbacks)
    return M, details
