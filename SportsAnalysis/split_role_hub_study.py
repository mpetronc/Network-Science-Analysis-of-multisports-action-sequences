"""Benchmark separate minutes and passing roles against the frozen six-season model."""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import pandas as pd
from split_role_hub_model import forecast


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def entropy(M, denominator):
    if M.sum() <= 0:
        return np.nan
    nz = M > 0
    transition = M / np.maximum(M.sum(1)[:, None], 1)
    return float(-(M[nz] / M.sum() * np.log2(transition[nz])).sum() / denominator)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split-role-hub-study', action='store_true')
    p.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent/'nba_passing_output')
    p.add_argument('--baseline', type=Path)
    p.add_argument('--destination', type=Path)
    p.add_argument('--skip-network-plots', action='store_true')
    a = p.parse_args(argv)
    baseline = a.baseline or a.output_dir/'all_season_hub_redistribution'
    out = a.destination or a.output_dir/'split_role_hub_redistribution'
    out.mkdir(parents=True, exist_ok=True)
    ledger, frozen = [], []
    # Freeze every forecast before loading any actual evaluation matrix.
    selections = sorted(baseline.glob('*/*/frozen_specification.json'))
    if len(selections) != 180:
        raise ValueError('Expected 180 saved training networks. Run --all-season-hub-study first or set --baseline to its completed output.')
    for specfile in selections:
        season, team = specfile.relative_to(baseline).parts[:2]
        source = specfile.parent
        destination = out/season/team
        destination.mkdir(parents=True, exist_ok=True)
        spec = json.loads(specfile.read_text())
        selection = spec['selection']
        ids = selection['training_player_ids']
        Pframe = pd.read_csv(source/'hub_present_training.csv', index_col=0)
        P = Pframe.to_numpy(float)
        A20 = pd.read_csv(source/'first20_all_passing.csv', index_col=0).to_numpy(float)
        assert Pframe.index.astype(int).tolist() == ids
        early = A20-P
        assert early.min() >= 0
        hub = ids.index(selection['hub_id'])
        assert early[hub].sum()+early[:, hub].sum() == 0
        M, rules = forecast(P, early, spec['training_present_minutes'],
                            spec['training_absent_minutes'], spec['forecast_minutes'],
                            hub, selection['first20_hub_absent_games'])
        pd.DataFrame(M, index=ids, columns=ids).to_csv(destination/'frozen_passing_role_prediction.csv')
        rules.update(season=season, team=team, hub_id=selection['hub_id'],
                     hub_name=selection['hub_name'], training_player_ids=ids,
                     first20_last_date=selection['first20_last_date'],
                     minute_weights=spec['replacement_weights'],
                     training_present_minutes=spec['training_present_minutes'],
                     training_absent_minutes=spec['training_absent_minutes'],
                     first20_hub_absent_games=selection['first20_hub_absent_games'],
                     source_hashes={name:sha(source/name) for name in
                        ['hub_present_training.csv','first20_all_passing.csv','frozen_specification.json']},
                     prediction_sha256=sha(destination/'frozen_passing_role_prediction.csv'),
                     freeze_stage='Before reading evaluation matrices; model settings fixed before this benchmark')
        (destination/'frozen_specification.json').write_text(json.dumps(rules, indent=2))
        frozen.append(dict(season=season, team=team, sha256=rules['prediction_sha256']))
        for i, pid in enumerate(ids):
            ledger.append(dict(season=season, team=team, player_id=pid, is_hub=i==hub,
                               forecast_minutes=spec['forecast_minutes'][i],
                               extra_minutes=spec['extra_minutes'][i],
                               minute_replacement_share=spec['replacement_weights'][i],
                               sender_replacement_share=rules['sender']['weights'][i],
                               receiver_replacement_share=rules['receiver']['weights'][i],
                               sender_rate=rules['sender']['shrunk_present_passes_per_minute'][i],
                               receiver_rate=rules['receiver']['shrunk_present_passes_per_minute'][i],
                               sender_historical_blend=rules['sender']['historical_blend'],
                               receiver_historical_blend=rules['receiver']['historical_blend']))
    (out/'freeze_manifest.json').write_text(json.dumps(frozen, indent=2))
    labels = pd.concat([pd.read_csv(baseline/season/'replacement_minutes_and_weights.csv').assign(season=season)
                        for season in sorted({r['season'] for r in ledger})], ignore_index=True)
    display_columns = ['season','team','player_id','player_name','prior_position','position_match','depth_rank']
    pd.DataFrame(ledger).merge(labels[display_columns],on=['season','team','player_id'],how='left',validate='one_to_one').to_csv(out/'minutes_and_passing_role_weights.csv',index=False)
    print('Frozen', len(frozen), 'passing-role predictions', flush=True)
    baseline_data = pd.read_csv(baseline/'all_team_season_entropy_and_ec.csv')
    results = []
    for z in baseline_data.itertuples():
        source = baseline/z.season/z.team
        destination = out/z.season/z.team
        original = pd.read_csv(source/'original_evaluation.csv', index_col=0)
        ids = original.index.astype(int).tolist()
        P = original.to_numpy(float)
        old = pd.read_csv(source/'predicted_evaluation.csv', index_col=0).to_numpy(float)
        A = pd.read_csv(source/'actual_evaluation.csv', index_col=0).to_numpy(float)
        path = destination/'frozen_passing_role_prediction.csv'
        assert sha(path) == next(f['sha256'] for f in frozen if f['season']==z.season and f['team']==z.team)
        M = pd.read_csv(path, index_col=0).reindex(index=ids, columns=list(map(str, ids)), fill_value=0).to_numpy(float)
        pd.DataFrame(M, index=ids, columns=ids).to_csv(destination/'predicted_evaluation.csv')
        D = P.copy();hub = ids.index(z.hub_id);D[hub]=0;D[:,hub]=0
        row = dict(season=z.season, team=z.team, hub_id=z.hub_id, hub_name=z.hub_name,
                   eligible=bool(z.primary_eligible), actual_absent_games=z.later_hub_absent_games,
                   entropy_original=entropy(P,z.normalization_log2),
                   entropy_old=entropy(old,z.normalization_log2),
                   entropy_new=entropy(M,z.normalization_log2),
                   entropy_actual=entropy(A,z.normalization_log2), denominator=z.normalization_log2,
                   original_passes=float(P.sum()), observed=bool(A.sum()>0))
        for name,B in [('old',old),('new',M),('deletion',D)]:
            row['edge_error_'+name]=float(abs(B/B.sum()-A/A.sum()).sum()/2) if A.sum()>0 else np.nan
            row['entropy_error_'+name]=abs(entropy(B,z.normalization_log2)-row['entropy_actual'])
        assert np.isclose(row['entropy_original'],z.entropy_original)
        if A.sum()>0:assert np.isclose(row['entropy_actual'],z.entropy_actual)
        results.append(row)
    data = pd.DataFrame(results)
    data['new_minus_old_edge_error']=data.edge_error_new-data.edge_error_old
    data['new_minus_old_entropy_error']=data.entropy_error_new-data.entropy_error_old
    data.to_csv(out/'prediction_comparison.csv',index=False)
    summaries=[]
    for sample,q in [('eligible',data[data.eligible]),('all_observed',data[data.observed])]:
        for season,g in [('ALL',q),*list(q.groupby('season'))]:
            summaries.append(dict(sample=sample,season=season,n=len(g),
                                  median_edge_old=g.edge_error_old.median(),
                                  median_edge_new=g.edge_error_new.median(),
                                  median_paired_edge_change=g.new_minus_old_edge_error.median(),
                                  teams_edge_improved=int((g.new_minus_old_edge_error < -1e-8).sum()),
                                  teams_edge_worsened=int((g.new_minus_old_edge_error > 1e-8).sum()),
                                  median_entropy_old=g.entropy_error_old.median(),
                                  median_entropy_new=g.entropy_error_new.median(),
                                  median_paired_entropy_change=g.new_minus_old_entropy_error.median(),
                                  teams_entropy_improved=int((g.new_minus_old_entropy_error < -1e-8).sum())))
    summary=pd.DataFrame(summaries)
    summary.to_csv(out/'performance_summary.csv',index=False)
    from split_role_hub_report import report
    report(a.output_dir,baseline,out,data,summary,a.skip_network_plots)
    print(summary[summary.season.eq('ALL')].to_string(index=False),flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
