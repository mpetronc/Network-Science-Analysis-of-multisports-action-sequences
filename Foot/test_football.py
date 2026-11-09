from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import networkx as nx
import numpy as np
import pandas as pd

from benchmark import match_passes
from followup import LineupShuffle, predict_late_entropy
from metrics import network_metrics
from openplay import EntryModel, receiver_candidates
from plots import draw_network
from regression import fit_regression
from season_exports import export_network, player_export
from season_reports import inferred_club_networks, js_divergence, statsbomb_minutes, summarize


class FootballTests(unittest.TestCase):
    def test_entropy_and_pagerank(self):
        metrics, players = network_metrics([[0, 2, 2], [1, 0, 0], [1, 0, 0]])
        self.assertAlmostEqual(metrics["conditional_receiver_entropy"], 4 / 6)
        self.assertAlmostEqual(players["pagerank"].sum(), 1)
        self.assertEqual(players["pagerank"].idxmax(), 0)

    def test_mutual_information_uses_raw_entropy_before_normalization(self):
        metrics, _ = network_metrics([[0, 2, 2], [1, 0, 0], [1, 0, 0]])
        expected = np.log(3) - 2 / 3 * np.log(2)
        self.assertAlmostEqual(metrics["passing_mutual_information"], expected)
        self.assertAlmostEqual(metrics["normalized_passing_mutual_information"], expected / np.log(3))
        self.assertAlmostEqual(metrics["effective_receivers"], 3)

    def test_minutes_filter_applies_to_both_ends_and_shots(self):
        teams = summarize({(1, 2): 6, (3, 2): 3, (2, 3): 4}, {1: "A", 2: "B", 3: "C"},
                          {1: 2, 3: 3}, {1: 90, 2: 90, 3: 15}, 18.75)
        full = teams["all_players"].metrics
        filtered = teams["minutes_filter"].metrics
        self.assertEqual(full["recorded_passes"], 13)
        self.assertEqual(filtered["recorded_passes"], 6)
        self.assertEqual(filtered["non_penalty_shots"], 2)

    def test_statsbomb_minutes_carry_first_half_stoppage_into_second_half(self):
        events = [{"type": {"name": "Half End"}, "period": i, "timestamp": f"00:{minute}:00"}
                  for i, minute in [(1, 47), (2, 49)]]
        lineups = [{"team_id": 10, "lineup": [
            {"player_id": 1, "positions": [{"from": "00:00", "from_period": 1,
                                             "to": "70:00", "to_period": 2}]},
            {"player_id": 2, "positions": [{"from": "70:00", "from_period": 2,
                                             "to": None, "to_period": None}]},
        ]}]
        self.assertEqual(statsbomb_minutes(events, lineups), {(10, 1): 72.0, (10, 2): 24.0})
        self.assertAlmostEqual(js_divergence([2, 0], [0, 5]), 1.0)
        self.assertAlmostEqual(js_divergence([2, 3], [4, 6]), 0.0)

    def test_missing_profile_is_distinct_from_known_zero(self):
        metrics, players = network_metrics([[0, 3, 0], [np.nan, 0, np.nan], [0, 1, 0]])
        self.assertEqual(metrics["recorded_passes"], 4)
        self.assertEqual(players["known_outgoing_profile"].tolist(), [True, False, True])

    def test_football_map_keeps_players_without_a_displayed_route(self):
        matrix = np.array([[0, 10, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        _, players = network_metrics(matrix)
        players["player_id"] = [1, 2, 3, 4]
        players["player"] = ["Defender", "Passing hub", "Goalkeeper", "No passes"]
        players["x"] = [52, 52, 12, np.nan]
        players["y"] = [50, 50, 50, np.nan]

        def verify_figure(figure, *_args, **_kwargs):
            pitch, roster = figure.axes
            markers, = pitch.collections
            self.assertEqual(len(markers.get_offsets()), 4)
            self.assertEqual(len({tuple(xy) for xy in markers.get_offsets()}), 4)
            self.assertGreater(markers.get_sizes()[0], markers.get_sizes()[-1])
            self.assertLess(markers.get_offsets()[-1][1], 0)
            self.assertIn("Goalkeeper", [text.get_text() for text in roster.texts])

        with TemporaryDirectory() as directory, patch(
            "matplotlib.figure.Figure.savefig", autospec=True, side_effect=verify_figure
        ):
            draw_network("Club", "2015-16", [1, 2, 3, 4], matrix, players, Path(directory), football=True)

    def test_guess_stops_at_fixture_or_opposition_touch(self):
        columns = ["match_id", "period", "team_id", "player_id", "clock",
                   "type", "outcome", "x", "y", "end_x", "end_y"]
        events = pd.DataFrame([
            (1, "FirstHalf", 1, 7, 1, "Pass", "Successful", 0, 0, 12, 12),
            (1, "FirstHalf", 1, 8, 2, "Pass", "Successful", 12, 12, 15, 15),
            (1, "FirstHalf", 2, 9, 3, "BallTouch", "Successful", 15, 15, None, None),
            (2, "FirstHalf", 1, 7, 1, "Pass", "Successful", 0, 0, 12, 12),
            (3, "FirstHalf", 1, 8, 2, "BallTouch", "Successful", 12, 12, None, None),
        ], columns=columns)
        guesses = receiver_candidates(events)["guess"]
        self.assertEqual(guesses.loc[0], 8)
        self.assertTrue(pd.isna(guesses.loc[1]))
        self.assertTrue(pd.isna(guesses.loc[3]))

    def test_alignment_ignores_receiver(self):
        recorded = pd.DataFrame({"period": [1, 1], "passer_id": [1, 1], "clock": [7, 40],
                                 "x": [10, 10], "y": [10, 10], "end_x": [40, 40],
                                 "end_y": [30, 30], "receiver_id": [5, 6]})
        other = recorded.assign(receiver_id=[9, 9], clock=[8, 70])
        self.assertEqual(match_passes(recorded, other), [(0, 0)])

    def test_inferred_export_keeps_weighted_edges_and_unresolved_pass_counts(self):
        passes = pd.DataFrame({
            "match_id": [1, 1, 1], "player_id": [10, 10, 10], "guess": [20, 20, np.nan],
            "x": [20, 20, 30], "y": [30, 31, 32], "end_x": [50, 50, 60], "end_y": [51, 50, 40],
        })
        squad = pd.DataFrame({
            "match_id": [1, 1, 1], "player_id": [10, 20, 30],
            "player": ["A", "B", "C"], "minutes_played": [90, 90, 5],
        })
        coverage, networks, appearances = inferred_club_networks(
            passes, squad, {20: 2}, 18.75, include_players=True
        )
        network = networks["all_players"]
        players = player_export(network, passes, appearances, {20: 2})
        keys = {"team": "Club", "season": "2020-21", "cohort": "all_players"}
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            metrics, edges = export_network(folder, keys, coverage, network, players)
            graph = nx.read_graphml(folder / "passing_network.graphml", node_type=int)
            self.assertTrue((folder / "passing_network.png").is_file())
            self.assertEqual(set(graph), {10, 20, 30})
            self.assertEqual(graph.graph["receiver_source"], "inferred")
            self.assertEqual(graph[10][20]["weight"], 2)
            self.assertEqual(edges["pass_count"].sum(), metrics["inferred_network_passes"])
            self.assertEqual(players.loc[players["player_id"].eq(10), "unresolved_receivers"].item(), 1)
            self.assertEqual(metrics["successful_passes"], 3)

    def test_regression_recovers_within_club_slope_with_season_effects(self):
        rows = []
        for team in range(5):
            for season in range(4):
                concentration = (team + season + (team * season) % 3) * 0.01
                entropy = 0.9 + 0.02 * team + 0.05 * season - 0.04 * concentration / 0.01
                rows.append((team, season, concentration, entropy))
        frame = pd.DataFrame(rows, columns=["team", "season", "pagerank_concentration",
                                           "conditional_receiver_entropy"])
        fit = fit_regression(frame, controls=("team", "season"), draws=200)
        self.assertAlmostEqual(fit.slope, -0.04)
        self.assertAlmostEqual(fit.interval[0], -0.04)
        self.assertAlmostEqual(fit.interval[1], -0.04)
        self.assertAlmostEqual(fit.r_squared, 1.0)

    def test_lineup_shuffle_keeps_receiver_margins_and_excludes_self_passes(self):
        senders = np.tile([0, 1, 2, 3], 8)
        receivers = np.tile([1, 2, 3, 0], 8)
        groups = np.repeat([0, 1], 16)
        shuffle = LineupShuffle(senders, receivers, groups)
        rng = np.random.default_rng(7)
        for _ in range(20):
            shuffle.sweep(rng)
        self.assertFalse(np.any(senders == shuffle.receivers))
        self.assertFalse(np.array_equal(receivers, shuffle.receivers))
        for group in np.unique(groups):
            chosen = groups == group
            np.testing.assert_array_equal(np.sort(receivers[chosen]), np.sort(shuffle.receivers[chosen]))

    def test_future_entropy_does_not_train_its_own_forecast(self):
        rows = []
        for season in ["2014-15", "2015-16", "2016-17"]:
            for i, (early, concentration) in enumerate([(0.72, 0.02), (0.78, 0.01), (0.76, 0.03), (0.82, 0.04)]):
                rows.append((season, i, early, concentration, 0.3 + 0.6 * early - concentration))
        halves = pd.DataFrame(rows, columns=[
            "season", "team_id", "conditional_receiver_entropy_early", "pagerank_concentration_early",
            "conditional_receiver_entropy_late",
        ])
        first = predict_late_entropy(halves)
        np.testing.assert_allclose(first["entropy_and_hubs"], first["conditional_receiver_entropy_late"])
        halves.loc[halves["season"].eq("2016-17"), "conditional_receiver_entropy_late"] = 0.5
        second = predict_late_entropy(halves)
        np.testing.assert_allclose(first["entropy_and_hubs"], second["entropy_and_hubs"])

    def test_tactical_model_uses_only_fitted_outcomes(self):
        entries = pd.DataFrame({
            "team_id": [1, 1], "team": ["Club", "Club"], "depth": [1, 1], "origin": [0, 0],
            "previous": ["Pass", "Recovery"], "destination": [0, 2],
        })
        model = EntryModel()
        model.fit(pd.concat([entries] * 20))
        score = model.evaluate(entries).iloc[0]
        self.assertGreater(score["context_gain_bits"], 0)
        self.assertEqual(model.evaluate(entries).iloc[0]["context_bits"], score["context_bits"])


if __name__ == "__main__":
    unittest.main()
