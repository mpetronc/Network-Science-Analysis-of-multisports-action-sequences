"""Add an FGA-weighted basket node to saved, full-roster NBA API networks.

Run standalone, or through teamgraphs.py --basket-networks. No downloads.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
import pandas as pd

BASKET = 'BASKET'


def build_networks(matrix, roster, basket_ratio=1.0):
    """Keep every pass unchanged; basket weights sum to ratio * total passes.

    The roster must be the same team-season Totals response as the matrix.
    No missing FGA is silently converted to zero. Zero-outdegree basket stays
    a sink: we do not invent a self-loop or an outgoing transition.
    """
    matrix = matrix.copy()
    matrix.index = matrix.index.astype(int)
    matrix.columns = matrix.columns.astype(int)
    ids = matrix.index.tolist()
    if matrix.index.has_duplicates or ids != matrix.columns.tolist():
        raise ValueError('Passing matrix must have identical, unique row/column IDs')
    A = matrix.to_numpy(float)
    if (not np.isfinite(A).all() or (A < 0).any() or np.diag(A).any()
            or not np.allclose(A, np.round(A), rtol=0, atol=1e-8)):
        raise ValueError('Passing matrix must contain finite nonnegative counts, without self-passes')
    if not np.isfinite(basket_ratio) or basket_ratio <= 0:
        raise ValueError('Basket ratio must be positive and finite')
    roster = roster.copy()
    roster['PLAYER_ID'] = pd.to_numeric(roster.PLAYER_ID, errors='raise').astype(int)
    if roster.PLAYER_ID.duplicated().any() or set(ids) != set(roster.PLAYER_ID):
        raise ValueError('Full passing-matrix roster and shooting roster must match exactly')
    roster = roster.set_index('PLAYER_ID').loc[ids]
    fga = pd.to_numeric(roster.FGA, errors='raise').to_numpy(float)
    if (not np.isfinite(fga).all() or (fga < 0).any()
            or not np.allclose(fga, np.round(fga), rtol=0, atol=1e-8)):
        raise ValueError('FGA must be known, nonnegative season totals')
    if roster.PLAYER_NAME.isna().any():
        raise ValueError('Missing player name')
    total_passes, total_fga = float(A.sum()), float(fga.sum())
    if total_passes <= 0 or total_fga <= 0:
        raise ValueError('Positive passing and FGA totals required for normalization')
    scale = basket_ratio * total_passes / total_fga
    original = nx.DiGraph(edge_weight_description='NBA API tracked pass counts')
    for i, pid in enumerate(ids):
        original.add_node(pid, name=str(roster.loc[pid, 'PLAYER_NAME']),
                          node_type='player', field_goal_attempts=float(fga[i]))
    for i, j in zip(*np.nonzero(A)):
        original.add_edge(ids[i], ids[j], weight=float(A[i, j]),
                          raw_count=float(A[i, j]), edge_type='pass')
    augmented = original.copy()
    augmented.graph.update(basket_ratio=float(basket_ratio), basket_scale=scale,
                           total_passes=total_passes, total_fga=total_fga,
                           basket_edge_description='FGA share times basket_ratio times total passes')
    augmented.add_node(BASKET, name='Basket', node_type='basket')
    for i, pid in enumerate(ids):
        if fga[i] > 0:
            augmented.add_edge(pid, BASKET, weight=float(fga[i] * scale),
                               raw_count=float(fga[i]), edge_type='shot_attempt')
    audit = pd.DataFrame(dict(player_id=ids, player_name=roster.PLAYER_NAME.tolist(),
        fga=fga, fga_share=fga/total_fga, basket_weight=fga*scale,
        incoming_passes=A.sum(axis=0), outgoing_passes=A.sum(axis=1)))
    assert np.isclose(sum(augmented[p][BASKET]['weight'] for p in ids if augmented.has_edge(p, BASKET)), basket_ratio*total_passes)
    assert all(augmented[u][v]['weight'] == a['weight'] for u, v, a in original.edges(data=True))
    return original, augmented, audit


def player_positions(original, augmented):
    """Fruchterman–Reingold positions shared by both panels.

    Lay out original passing counts only; add basket at the right. This keeps
    player movement from obscuring the effect of the added shooting edges.
    """
    layout = nx.Graph()
    layout.add_nodes_from(n for n in original if augmented.degree(n) > 0)
    for u, v, a in original.edges(data=True):
        layout.add_edge(u, v, weight=layout.get_edge_data(u, v, {}).get('weight', 0.)+a['weight'])
    maximum = max((a['weight'] for _, _, a in layout.edges(data=True)), default=1.)
    for _, _, a in layout.edges(data=True):
        a['weight'] = np.sqrt(a['weight'] / maximum)
    options = dict(seed=42, iterations=300, threshold=1e-5, weight='weight', k=1.5 / np.sqrt(max(len(layout), 1)))
    if 'method' in inspect.signature(nx.spring_layout).parameters:
        options['method'] = 'force'
    positions = nx.spring_layout(layout, **options)
    positions[BASKET] = np.array([1.7, 0.])
    return positions


def comparison_figure(original, augmented, audit, team, season):
    positions = player_positions(original, augmented)
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    maximum = max(a['weight'] for _, _, a in augmented.edges(data=True))
    activity = audit.set_index('player_id').eval('incoming_passes + outgoing_passes')
    peak = max(float(activity.max()), 1.)
    limits = np.array(list(positions.values()))
    for ax, G, title in zip(axes, [original, augmented],
            ['Original passing network', 'Passing network + scaled shot attempts']):
        players = [n for n in original if G.degree(n) > 0]
        visible = players + ([BASKET] if BASKET in G else [])
        sizes = {n: 180+1050*float(activity.loc[n])/peak for n in players}
        sizes[BASKET] = 1800
        H = G.subgraph(visible)
        nx.draw_networkx_nodes(H, positions, nodelist=players,
            node_size=[sizes[n] for n in players], node_color='#f4cd55',
            edgecolors='#62572c', ax=ax)
        if BASKET in H:
            nx.draw_networkx_nodes(H, positions, nodelist=[BASKET], node_size=1800,
                node_shape='s', node_color='#f29b52', edgecolors='#8c4208', ax=ax)
        for edge_type, color, alpha in [('pass', '#3166a1', .62), ('shot_attempt', '#d45d0c', .82)]:
            edges = [(u, v) for u, v, a in H.edges(data=True) if a['edge_type'] == edge_type]
            if edges:
                nx.draw_networkx_edges(H, positions, edgelist=edges,
                    width=[.25+5.25*H[u][v]['weight']/maximum for u, v in edges],
                    node_size=[sizes[n] for n in H], arrows=True, arrowsize=13,
                    edge_color=color, alpha=alpha, connectionstyle='arc3,rad=0.10', ax=ax)
        labels = {n: H.nodes[n]['name'].replace(' ', '\n', 1) for n in players}
        if BASKET in H:
            labels[BASKET] = 'BASKET\n(FGA)'
        nx.draw_networkx_labels(H, positions, labels=labels, font_size=8, ax=ax)
        ax.set_title(title, fontsize=16)
        ax.set_xlim(limits[:, 0].min()-.35, limits[:, 0].max()+.4)
        ax.set_ylim(limits[:, 1].min()-.3, limits[:, 1].max()+.3)
        ax.axis('off')
    P, F = augmented.graph['total_passes'], augmented.graph['total_fga']
    ratio = augmented.graph['basket_ratio']
    fig.suptitle(f'{team} | {season} Regular Season | Full roster', fontsize=19)
    fig.legend(handles=[Line2D([0],[0],color='#3166a1',lw=3,label='Tracked passes'),
                        Line2D([0],[0],color='#d45d0c',lw=3,label='Scaled FGA')],
               loc='lower center', bbox_to_anchor=(.5,.075), ncol=2, frameon=False)
    fig.text(.5,.025,
        f'Basket weight = {ratio:g} × total passes × player FGA / team FGA | Passes: {P:,.0f}; FGA: {F:,.0f}; basket weight: {ratio*P:,.0f}\n'
        'Shared edge-width scale (0.25 pt baseline) and player positions; node size uses passing involvement. Isolates omitted from each drawing.\n'
        'NBA API saved season totals; no minutes cutoff. Attempts include misses. Scaled edges are not observed pass→shot sequences.',
        ha='center', fontsize=10)
    fig.tight_layout(rect=[0,.12,1,.94])
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--basket-networks', action='store_true')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent/'nba_passing_output',
                        help='Existing SportsAnalysis output root containing full-roster matrices')
    parser.add_argument('--destination', type=Path, help='Default: OUTPUT_DIR/basket_networks')
    parser.add_argument('--season', default='ALL')
    parser.add_argument('--team', default='ALL')
    parser.add_argument('--basket-ratio', type=float, default=1., help='Total basket weight / total passes (default 1)')
    args = parser.parse_args(argv)
    root = args.output_dir.expanduser()
    source = root/'full_roster_entropy_centrality/historical_full_rosters'
    out = args.destination.expanduser() if args.destination else root/'basket_networks'
    sources = sorted(source.glob('20*/*/passing_matrix.csv'))
    sources = [p for p in sources if (args.season == 'ALL' or p.parent.parent.name == args.season)
               and (args.team.upper() == 'ALL' or p.parent.name == args.team.upper())]
    if not sources:
        raise FileNotFoundError(f'No matching full-roster matrices in {source}')
    out.mkdir(parents=True, exist_ok=True)
    rows, manifest = [], []
    with PdfPages(out/'all_basket_networks.pdf') as combined:
        for k, path in enumerate(sources, 1):
            team, season = path.parent.name, path.parent.parent.name
            roster_path = path.with_name('roster.csv')
            matrix, roster = pd.read_csv(path, index_col=0), pd.read_csv(roster_path)
            original, augmented, audit = build_networks(matrix, roster, args.basket_ratio)
            folder = out/season/team
            folder.mkdir(parents=True, exist_ok=True)
            for graph, filename in [(original, 'original_passing'), (augmented, 'passing_with_basket')]:
                graph.graph.update(team=team, season=season, season_type='Regular Season', source='NBA.com via nba_api; saved Totals data')
                nx.write_graphml(graph, folder/(filename+'.graphml'))
            audit.to_csv(folder/'player_basket_weights.csv', index=False)
            nodes = list(augmented.nodes)
            nx.to_pandas_adjacency(augmented, nodelist=nodes, weight='weight').to_csv(folder/'augmented_weight_matrix.csv')
            pd.DataFrame([dict(source=u, target=v, **a) for u,v,a in augmented.edges(data=True)]).to_csv(folder/'edges.csv', index=False)
            fig = comparison_figure(original, augmented, audit, team, season)
            fig.savefig(folder/'network_comparison.png', dpi=130)
            fig.savefig(folder/'network_comparison.pdf')
            combined.savefig(fig)
            plt.close(fig)
            P, F = augmented.graph['total_passes'], augmented.graph['total_fga']
            shot_sum = audit.basket_weight.sum()
            original_matrix = nx.to_numpy_array(original, nodelist=list(original), weight='weight')
            augmented_players = nx.to_numpy_array(augmented, nodelist=list(original), weight='weight')
            unchanged = bool(np.array_equal(original_matrix, augmented_players))
            normalized = bool(np.isclose(shot_sum, args.basket_ratio*P))
            if not unchanged or not normalized:
                raise AssertionError('Passing preservation or basket scaling failed')
            rows.append(dict(team=team, season=season, roster_players=len(audit),
                original_visible_players=len(original)-len(list(nx.isolates(original))),
                augmented_visible_players=len(augmented)-1-len(list(nx.isolates(augmented))),
                total_passes=P, total_fga=F, basket_scale=augmented.graph['basket_scale'],
                basket_ratio=args.basket_ratio, total_basket_weight=shot_sum,
                basket_fraction_of_all_weight=shot_sum/(shot_sum+P),
                passing_edges_unchanged=unchanged, normalization_verified=normalized))
            for kind, file in [('passing_matrix',path),('roster_totals',roster_path)]:
                manifest.append(dict(team=team, season=season, kind=kind, path=str(file),
                                     sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
            print(f'{k}/{len(sources)} {season} {team}: passes={P:,.0f}, FGA={F:,.0f}, basket weight={shot_sum:,.0f}', flush=True)
    pd.DataFrame(rows).to_csv(out/'normalization_audit.csv', index=False)
    pd.DataFrame(manifest).to_csv(out/'source_manifest.csv', index=False)
    (out/'settings.json').write_text(json.dumps(vars(args), default=str, indent=2))
    (out/'README.md').write_text('''# NBA API passing networks with a basket node

Each page pairs the original full-roster passing network with a second graph adding player → BASKET edges. Open all_basket_networks.pdf, or a season/team network_comparison.png. GraphML, augmented matrices, raw FGA and scaled basket weights are saved per team-season.

Formula: basket_weight_i = basket_ratio × sum(all player-to-player passes) × FGA_i / sum(all roster FGA). The default basket_ratio=1 gives shooting edges the same combined weight as passing edges (50% of augmented edge weight). All original directed passing edges stay exactly unchanged. This is a transparent modeling choice; try --basket-ratio 0.5 or 2 with a separate --destination to assess sensitivity.

Inputs are the previously saved team-filtered NBA API PlayerDashPtPass Totals and LeagueDashPlayerStats Totals networks, covering matching regular-season windows. Full rosters include traded participants with their team-filtered statistics; no minutes cutoff. Roster IDs must match exactly; invalid/missing FGA is rejected. These outputs reuse saved NBA data, with no fresh API request. Source hashes and normalization checks are saved. No movement-inferred SportVU passes are used.

Basket edges count attempts, including misses, rather than made shots, assists, or observed sequences after a pass. They can include unassisted shots. Scaling does not measure efficiency or establish a pass-to-shot relationship. The basket is a directed sink, with no fabricated outgoing/self-loop edges. Avoid interpreting incoming eigenvector centrality on this sink-augmented graph as the original passing centrality. Existing entropy and centrality analyses are untouched.

The player layout is Fruchterman–Reingold using square-root-compressed original passing weights for readable spacing. This compression affects placement only; exported weights stay unchanged. Both panels use identical player positions, node sizes based on pass involvement, and one common edge-width scale (0.25 pt plus a linear weight term); the basket sits at the right for readability. These are not court positions. True isolates are omitted from each drawing but retained in exported data; a player with no passes and positive FGA appears in the augmented panel.

Run from SportsAnalysis: python3 teamgraphs.py --basket-networks [--season 2021-22] [--team GSW]. Standalone: python3 basket_networks.py --output-dir /path/to/nba_passing_output. Optional --destination controls the new results folder. Earlier commands and results are preserved.
''')
    print('Saved '+str(out), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
