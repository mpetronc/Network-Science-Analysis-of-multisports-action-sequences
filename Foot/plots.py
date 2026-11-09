"""Figures for the season comparison, passing maps and prediction results."""

import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
import networkx as nx
import numpy as np
import pandas as pd

BLUE = "#186789"
ORANGE = "#c15e2b"
DARK = "#172c38"


def comparison(teams, output):
    figure, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(13, 6), layout="constrained")
    x, y = "conditional_receiver_entropy", "pagerank_concentration"
    for ax, (league, data) in zip(axes, teams.groupby("sport", sort=False)):
        complete = data.loc[data["coverage_complete"]]
        partial = data.loc[~data["coverage_complete"]]
        ax.scatter(complete[x], complete[y], s=50, color=BLUE, label="Complete profiles")
        if not partial.empty:
            ax.scatter(partial[x], partial[y], s=48, facecolors="none", edgecolors=ORANGE,
                       label="Partial or prior results")
        if len(complete) >= 5:
            ax.axvline(complete[x].median(), color="#aabac2", linewidth=1, linestyle=":")
            ax.axhline(complete[y].median(), color="#aabac2", linewidth=1, linestyle=":")
        values = data.dropna(subset=[x, y])
        if not values.empty:
            points = {values[x].idxmin(), values[x].idxmax(), values[y].idxmin(), values[y].idxmax()}
            if league == "NBA":
                points.update(complete.index)
            else:
                points.update(values.index[values["team"].eq("Leicester City")])
            for i in sorted(points):
                row = values.loc[i]
                ax.annotate(row["team"], (row[x], row[y]), xytext=(5, 5),
                            textcoords="offset points", fontsize=8, color=DARK)
        ax.set_title(f"{league} · {data['season'].iloc[0]} · {len(complete)}/{len(data)} complete", loc="left")
        ax.set_xlabel("Conditional receiver entropy")
        ax.grid(alpha=0.15)
        ax.legend(loc="upper right", frameon=False, fontsize=8)
        ax.margins(x=0.14, y=0.15)
    axes[0].set_ylabel("PageRank concentration (adjusted for roster size)")
    figure.suptitle("How varied is passing when play revolves around a few players?", fontsize=16, color=DARK)
    figure.savefig(output / "nba_football_comparison.png", dpi=165)
    plt.close(figure)


def _pitch(ax):
    color = "#87a697"
    ax.set_facecolor("#edf4f0")
    ax.add_patch(Rectangle((0, 0), 100, 100, fill=False, color=color))
    ax.plot([50, 50], [0, 100], color=color, linewidth=0.8)
    for x in (0, 84):
        ax.add_patch(Rectangle((x, 20), 16, 60, fill=False, color=color, linewidth=0.8))
    for x in (0, 95):
        ax.add_patch(Rectangle((x, 37), 5, 26, fill=False, color=color, linewidth=0.8))
    ax.add_patch(Circle((50, 50), 9, fill=False, color=color, linewidth=0.8))
    ax.set(xlim=(-4, 104), ylim=(-5, 105), aspect=0.67)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


def _graph(ids, matrix, players):
    graph = nx.DiGraph()
    for i, player in enumerate(players.itertuples(index=False)):
        graph.add_node(str(ids[i]), label=str(player.player), pagerank=float(player.pagerank),
                       passes_sent=int(player.passes_sent), passes_received=int(player.passes_received))
    for row, col in zip(*np.where(matrix > 0)):
        graph.add_edge(str(ids[row]), str(ids[col]), weight=int(matrix[row, col]))
    return graph


def _top_routes(graph, positions):
    routes = [(a, b, w) for a, b, w in graph.edges(data="weight") if a in positions and b in positions]
    return sorted(routes, key=lambda edge: edge[2], reverse=True)[:30]


def _draw_routes(ax, routes, positions, node_sizes=None):
    max_weight = max((weight for _, _, weight in routes), default=1)
    for a, b, weight in routes:
        shrink_a = np.sqrt(node_sizes[a]) / 2 + 1 if node_sizes else 8
        shrink_b = np.sqrt(node_sizes[b]) / 2 + 1 if node_sizes else 8
        ax.add_patch(FancyArrowPatch(
            positions[a], positions[b], connectionstyle="arc3,rad=0.10",
            arrowstyle="-|>", shrinkA=shrink_a, shrinkB=shrink_b, mutation_scale=9,
            color=BLUE, alpha=0.37, linewidth=0.45 + 2.3 * weight / max_weight, zorder=2,
        ))


def _space_pitch_nodes(locations, sizes):
    """Place larger markers first and move crowded markers to the nearest free spot."""
    aspect = 0.67
    scale = np.array([1, aspect])
    offsets = np.array([(x, y) for x in range(-22, 23) for y in range(-22, 23)])
    offsets = offsets[np.argsort((offsets ** 2).sum(axis=1), kind="stable")]
    placed, radii = {}, []
    for player in sorted(locations, key=sizes.__getitem__, reverse=True):
        radius = np.sqrt(sizes[player]) / 11
        candidates = np.asarray(locations[player]) * scale + offsets
        candidates = candidates[((candidates >= radius) & (candidates <= 100 * scale - radius)).all(axis=1)]
        if placed:
            distances = np.linalg.norm(candidates[:, None] - np.array(list(placed.values())), axis=2)
            clearance = (distances - np.array(radii) - radius).min(axis=1)
            free = np.flatnonzero(clearance >= 0.8)
            candidate = candidates[free[0] if len(free) else clearance.argmax()]
        else:
            candidate = candidates[0]
        placed[player] = candidate
        radii.append(radius)
    return {player: position / scale for player, position in placed.items()}


def draw_football_network(name, season, ids, matrix, players, path, *,
                          source="StatsBomb", receiver_source="recorded", cohort=None, match_label=None):
    graph = _graph(ids, matrix, players)
    path.parent.mkdir(parents=True, exist_ok=True)
    locations = {str(p.player_id): (p.x, p.y) for p in players.itertuples(index=False)
                 if pd.notna(p.x) and pd.notna(p.y)}
    ranks = {p: float(np.nan_to_num(graph.nodes[p]["pagerank"], nan=0.0)) for p in graph}
    order = sorted(graph, key=ranks.__getitem__, reverse=True)
    numbers = {player: i for i, player in enumerate(order, start=1)}
    sizes = {p: 120 + 4100 * ranks[p] for p in graph}
    positions = _space_pitch_nodes(locations, sizes)

    figure, (pitch, roster) = plt.subplots(1, 2, figsize=(13.7, 8.0), width_ratios=(3, 1),
                                           layout="constrained")
    _pitch(pitch)
    title = f"{name} · {season}"
    if match_label:
        title += f" · {match_label}"
    if cohort:
        title += f" · {cohort.replace('_', ' ').title()}"
    figure.suptitle(title, x=0.02, ha="left", fontsize=17, color=DARK)
    roster.set_axis_off()
    roster.set(xlim=(0, 1), ylim=(len(order), -1.3))
    roster.text(0.03, -0.8, "Player", weight="bold", color=DARK)
    roster.text(0.98, -0.8, "PageRank", weight="bold", color=DARK, ha="right")

    for p, original in locations.items():
        if not np.allclose(original, positions[p]):
            x, y = zip(original, positions[p])
            pitch.plot(x, y, color="#5f766d", alpha=0.55, linewidth=0.6, zorder=1)

    _draw_routes(pitch, _top_routes(graph, locations), positions, sizes)
    unlocated = [p for p in order if p not in locations]
    if unlocated:
        pitch.text(1, -8, "No pass location", va="center", fontsize=9, color=DARK)
        for i, p in enumerate(unlocated):
            positions[p] = (33 + 8 * (i % 8), -8 - 7 * (i // 8))
        pitch.set_ylim(positions[unlocated[-1]][1] - 5, 105)

    if order:
        xs, ys = zip(*(positions[p] for p in order))
        pitch.scatter(xs, ys, s=[sizes[p] for p in order],
                      color=[BLUE if p in locations else "#dae7df" for p in order],
                      edgecolor="white", linewidth=0.9, zorder=3)

    for p in order:
        number = numbers[p]
        color = "white" if p in locations else DARK
        pitch.text(*positions[p], str(number), ha="center", va="center", color=color, fontsize=7,
                   weight="bold", zorder=4)
        row = number - 0.5
        roster.text(0.03, row, f"{number:>2}", color=BLUE, fontsize=8, weight="bold", va="center")
        roster.text(0.16, row, graph.nodes[p]["label"], color=DARK, fontsize=8, va="center")
        roster.text(0.98, row, f"{ranks[p]:.1%}", color=DARK, fontsize=8, va="center", ha="right")

    figure.supxlabel(
        f"{len(order)} players · Node size shows PageRank · Up to 30 busiest routes"
        f" · {source}, {receiver_source} receivers",
        fontsize=9, color=DARK,
    )
    figure.savefig(path.with_suffix(".png"), dpi=155)
    plt.close(figure)
    for player, (x, y) in locations.items():
        graph.nodes[player].update(x=float(x), y=float(y))
    graph.graph.update(team=name, season=season, source=source, receiver_source=receiver_source)
    if cohort:
        graph.graph["cohort"] = cohort
    nx.write_graphml(graph, path.with_suffix(".graphml"))


def draw_network(name, season, ids, matrix, players, folder, *, football=False, complete=True):
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    if football:
        draw_football_network(name, season, ids, matrix, players, folder / slug)
        return

    graph = _graph(ids, matrix, players)
    if not graph.number_of_edges():
        return
    positions = nx.spring_layout(graph, seed=42, weight="weight")
    shown = _top_routes(graph, positions)
    active = list(dict.fromkeys(p for a, b, _ in shown for p in (a, b)))
    figure, ax = plt.subplots(figsize=(9, 6.4), layout="constrained")
    ax.set_axis_off()
    ax.margins(0.15)
    _draw_routes(ax, shown, positions)

    xs, ys = zip(*(positions[p] for p in active))
    ax.scatter(xs, ys, s=[55 + 1250 * graph.nodes[p]["pagerank"] for p in active],
               color=BLUE, edgecolor="white", linewidth=1, zorder=3)
    for p, x, y in zip(active, xs, ys):
        parts = graph.nodes[p]["label"].split()
        label = f"{parts[0][0]}. {parts[-1]}" if len(parts) > 1 else parts[0] if parts else p
        ax.annotate(label, (x, y), xytext=(5, 6), textcoords="offset points", fontsize=7,
                    color=DARK, zorder=4)
    note = "Nodes sized by PageRank · 30 most frequent routes at most"
    if not complete:
        note += " · partial tracking profiles"
    ax.set_title(f"{name} · {season}", loc="left", fontsize=15, color=DARK, pad=18)
    figure.text(0.02, 0.015, f"{note} · Source: NBA.com", fontsize=8, color=DARK)
    folder.mkdir(parents=True, exist_ok=True)
    figure.savefig(folder / f"{slug}.png", dpi=155)
    plt.close(figure)


def openplay_prediction(scores, output):
    figure, (left, right) = plt.subplots(1, 2, figsize=(13, 5.4), layout="constrained")
    seasons, base, context = [], [], []
    for season, group in scores.groupby("season", sort=True):
        seasons.append(season)
        base.append(np.average(group["baseline_bits"], weights=group["entries"]))
        context.append(np.average(group["context_bits"], weights=group["entries"]))
    left.plot(seasons, base, color=ORANGE, marker="o", label="Team + pass origin")
    left.plot(seasons, context, color=BLUE, marker="o", label="Also uses prior action")
    left.tick_params(axis="x", labelrotation=50)
    left.legend(frameon=False)
    left.set_title("Prediction error by held-out season", loc="left")
    left.set_ylabel("Log loss (bits per entry, lower is better)")
    valid = scores.dropna(subset=["shot_hhi", "context_bits"])
    shade = pd.Categorical(valid["season"], categories=seasons).codes
    points = right.scatter(valid["shot_hhi"], valid["context_bits"], c=shade, cmap="viridis",
                           s=25, alpha=0.7)
    right.set_title("Shot concentration and predictability", loc="left")
    right.set_xlabel("Non-penalty shot concentration in earlier matches (HHI)")
    right.set_ylabel("Held-out log loss (bits per entry)")
    legend = figure.colorbar(points, ax=right, ticks=[0, len(seasons) - 1], shrink=0.8)
    legend.ax.set_yticklabels([seasons[0], seasons[-1]])
    for ax in (left, right):
        ax.grid(alpha=0.18)
    figure.suptitle("Where Premier League passes enter the final third · Source: OpenPlay (Noah Bair)",
                    fontsize=14, color=DARK)
    figure.savefig(output / "openplay_tactical_prediction.png", dpi=165)
    plt.close(figure)


def possession_patterns(patterns, output):
    top = patterns.groupby("sequence")["possessions"].sum().nlargest(10).sort_values()
    figure, ax = plt.subplots(figsize=(9, 5.4), layout="constrained")
    ax.barh(top.index, top, color=BLUE)
    ax.set_title("The four actions leading to a possession’s first shot", loc="left", color=DARK)
    ax.set_xlabel("Possessions · Premier League 2015–16 · Source: StatsBomb")
    ax.grid(axis="x", alpha=0.16)
    figure.savefig(output / "statsbomb_possession_patterns.png", dpi=160)
    plt.close(figure)
