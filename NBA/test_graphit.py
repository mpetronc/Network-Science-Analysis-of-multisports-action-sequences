"""Offline regression checks: no requests are sent to NBA.com."""

import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "nba-passing-mpl"))

import numpy as np
import pandas as pd
import requests

SPEC = importlib.util.spec_from_file_location("graphit_gpt", Path(__file__).with_name("graphit.GPT.py"))
app = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = app
SPEC.loader.exec_module(app)

TEAM = {"id": 1610612744, "abbreviation": "GSW"}


def roster():
    return pd.DataFrame({"PLAYER_ID": [1, 2, 3],
                         "PLAYER_NAME": ["Jay Williams", "Joe Williams", "Third Player"]})


def profile(player_id, rows):
    return pd.DataFrame([
        {"PLAYER_ID": player_id, "TEAM_ID": TEAM["id"],
         "PASS_TEAMMATE_PLAYER_ID": receiver, "PASS": count}
        for receiver, count in rows
    ], columns=["PLAYER_ID", "TEAM_ID", "PASS_TEAMMATE_PLAYER_ID", "PASS"])


def payload(name, columns, rows):
    return {"resultSets": [{"name": name, "headers": columns, "rowSet": rows}]}


def http_error(status, retry_after=None):
    response = requests.Response()
    response.status_code = status
    response._content_consumed = True
    if retry_after is not None:
        response.headers["Retry-After"] = retry_after
    return requests.HTTPError(str(status), response=response)


class EntropyTests(unittest.TestCase):
    def test_uniform_networks(self):
        for n in (3, 5, 15):
            with self.subTest(n=n):
                receiver, conditional = app.compute_team_entropies(np.ones((n, n)) - np.eye(n))
                self.assertAlmostEqual(receiver, 1.0)
                self.assertAlmostEqual(conditional, 1.0)

    def test_two_players(self):
        self.assertEqual(app.compute_team_entropies([[0, 10], [10, 0]]), (1.0, 0.0))

    def test_deterministic_receivers(self):
        receiver, conditional = app.compute_team_entropies([[0, 10, 0], [10, 0, 0], [0, 10, 0]])
        expected = -(math.log(1 / 3) / 3 + 2 * math.log(2 / 3) / 3) / math.log(3)
        self.assertAlmostEqual(receiver, expected)
        self.assertAlmostEqual(conditional, 0)

    def test_sender_weights_are_pass_shares(self):
        _, conditional = app.compute_team_entropies([[0, 10, 10], [80, 0, 0], [0, 0, 0]])
        self.assertAlmostEqual(conditional, 0.2)

    def test_no_passes_is_undefined(self):
        for matrix in (np.zeros((0, 0)), [[0]], np.zeros((3, 3))):
            self.assertEqual(app.compute_team_entropies(matrix), (None, None))

    def test_invalid_matrices(self):
        for matrix in ([[1, 0], [0, 0]], [[0, -1], [1, 0]], [[0, np.nan], [1, 0]], [[0, 1, 2]]):
            with self.subTest(matrix=matrix), self.assertRaises(ValueError):
                app.compute_team_entropies(matrix)


class MatrixTests(unittest.TestCase):
    def make_data(self, profiles, failures=None):
        return app.build_team_data(TEAM, roster(), profiles, failures or {}, [], app.Config())

    def test_missing_player_keeps_known_incoming_passes(self):
        data = self.make_data({1: profile(1, [(2, 10), (3, 90)]), 2: profile(2, [(1, 10)])},
                              {3: "timeout"})
        self.assertEqual(data.player_ids, [1, 2, 3])
        self.assertEqual(np.nansum(data.matrix), 110)
        self.assertEqual(data.matrix[0, 2], 90)
        self.assertTrue(np.isnan(data.matrix[2, 0]))
        self.assertIsNone(data.metrics()["receiver_entropy"])
        self.assertIsNotNone(data.metrics()["observed_only_receiver_entropy"])
        self.assertFalse(data.metrics()["player_profiles_complete"])

    def test_additional_receiver_is_retained_and_flagged(self):
        data = self.make_data({1: profile(1, [(99, 7)])})
        self.assertIn(99, data.player_ids)
        self.assertIn(99, data.missing)
        self.assertEqual(data.metrics()["additional_receiver_ids"], [99])
        self.assertEqual(np.nansum(data.matrix), 7)

    def test_duplicate_pair_rows_are_accumulated(self):
        data = self.make_data({1: profile(1, [(2, 4), (2, 6)])})
        self.assertEqual(data.matrix[0, 1], 10)

    def test_empty_profile_remains_unknown(self):
        data = self.make_data({1: profile(1, [])})
        self.assertIn(1, data.missing)
        self.assertTrue(np.isnan(data.matrix[0, 1]))
        self.assertIsNone(data.metrics()["observed_only_receiver_entropy"])

    def test_graph_ids_prevent_label_collisions_and_keep_isolates(self):
        data = self.make_data({1: profile(1, [(2, 4)])})
        graph = app.build_graph(data)
        self.assertEqual(set(graph.nodes), {1, 2, 3})
        self.assertTrue(graph.has_edge(1, 2))
        self.assertNotEqual(app.short_labels(data)[1], app.short_labels(data)[2])

    def test_saved_graph_round_trip_retains_direction_weights_and_missing_nodes(self):
        import networkx as nx

        data = self.make_data({1: profile(1, [(2, 4)])}, {2: "timeout", 3: "not available"})
        data.names[3] = "Name & accented é <test>"
        with tempfile.TemporaryDirectory() as temporary:
            app.save_team_data(data, Path(temporary))
            graph = nx.read_graphml(Path(temporary) / "passing_network.graphml", node_type=int)
        self.assertTrue(graph.is_directed())
        self.assertEqual(set(graph.nodes), {1, 2, 3})
        self.assertEqual(graph[1][2]["weight"], 4)
        self.assertFalse(graph.has_edge(2, 1))
        self.assertEqual(graph.nodes[3]["name"], data.names[3])
        self.assertEqual(graph.nodes[2]["missing_reason"], "timeout")
        self.assertFalse(graph.nodes[2]["outgoing_profile_available"])
        self.assertTrue(graph.nodes[1]["outgoing_profile_available"])
        self.assertEqual(graph.nodes[2]["known_incoming_passes"], 4)
        self.assertFalse(graph.graph["player_profiles_complete"])
        self.assertEqual(graph.graph["team"], "GSW")
        self.assertIn("NBA.com", graph.graph["source"])

    def test_graph_export_keeps_players_even_without_any_edges(self):
        import networkx as nx

        data = self.make_data({})
        with tempfile.TemporaryDirectory() as temporary:
            app.save_graph_data(data, Path(temporary))
            graph = nx.read_graphml(Path(temporary) / "passing_network.graphml", node_type=int)
        self.assertEqual(set(graph.nodes), {1, 2, 3})
        self.assertEqual(graph.number_of_edges(), 0)

    def test_outputs_preserve_unknown_values_and_plot(self):
        data = self.make_data({1: profile(1, [(2, 4)])})
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            app.save_team_data(data, directory)
            app.plot_team_network(data, directory / "graph.png")
            self.assertGreater((directory / "graph.png").stat().st_size, 1000)
            self.assertIn("NA", (directory / "passing_matrix.csv").read_text())
            self.assertIsNone(json.loads((directory / "metrics.json").read_text())["receiver_entropy"])
            pairs = pd.read_csv(directory / "passes.csv")
            self.assertEqual(len(pairs), 6)
            self.assertEqual(pairs["pass_count"].sum(), 4)
            self.assertTrue(pairs.loc[pairs["passer_id"] == 2, "pass_count"].isna().all())
            self.assertEqual(pd.read_csv(directory / "metrics.csv").iloc[0]["status"], "partial")


class ValidationTests(unittest.TestCase):
    def test_named_result_sets_ignore_order(self):
        raw = payload("PassesMade", ["PLAYER_ID", "TEAM_ID", "PASS_TEAMMATE_PLAYER_ID", "PASS"],
                      [[1, TEAM["id"], 2, 10]])
        raw["resultSets"].insert(0, {"name": "PassesReceived", "headers": [], "rowSet": []})
        frame = app.validate_passing(raw, 1, TEAM["id"])
        self.assertEqual(frame["PASS"].sum(), 10)

    def test_missing_columns_and_wrong_team_are_errors(self):
        for raw in (
            payload("PassesMade", ["PASS"], [[4]]),
            payload("PassesMade", ["PLAYER_ID", "TEAM_ID", "PASS_TEAMMATE_PLAYER_ID", "PASS"],
                    [[1, 99, 2, 4]]),
        ):
            with self.assertRaises(ValueError):
                app.validate_passing(raw, 1, TEAM["id"])

    def test_passing_counts_and_self_passes_are_validated(self):
        columns = ["PLAYER_ID", "TEAM_ID", "PASS_TEAMMATE_PLAYER_ID", "PASS"]
        for receiver, count in ((2, -1), (2, 0.5), (2, None), (1, 5)):
            with self.subTest(receiver=receiver, count=count), self.assertRaises(ValueError):
                app.validate_passing(payload("PassesMade", columns, [[1, TEAM["id"], receiver, count]]),
                                     1, TEAM["id"])

    def test_team_and_activity_filter_are_both_applied(self):
        columns = ["PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION", "GP", "MIN"]
        raw = payload("LeagueDashPlayerStats", columns, [
            [1, "A", TEAM["id"], "GSW", 1, 10], [2, "B", 99, "BOS", 2, 10],
            [3, "C", TEAM["id"], "GSW", 1, 0],
        ])
        self.assertEqual(app.validate_roster(raw, TEAM["id"], "GSW")["PLAYER_ID"].tolist(), [1])
        raw["resultSets"][0]["headers"].pop()
        with self.assertRaises(ValueError):
            app.validate_roster(raw, TEAM["id"], "GSW")


class CacheTests(unittest.TestCase):
    def test_team_and_other_query_parameters_have_different_keys(self):
        cache = app.DiskCache(app.Config())
        first = {"player_id": 1, "team_id": 2, "season": "2025-26", "season_type": "Regular Season"}
        for key, value in (("team_id", 3), ("season", "2024-25"), ("season_type", "Playoffs")):
            self.assertNotEqual(cache.path_for("passing", first),
                                cache.path_for("passing", {**first, key: value}))

    def test_round_trip_corruption_expiry_and_refresh(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = app.Config(cache_dir=Path(temporary))
            cache = app.DiskCache(config)
            snapshot = {"version": app.CACHE_VERSION, "endpoint": "passing", "parameters": {"team": 1},
                        "fetched_at": app.utc_now(), "payload": {"resultSets": []}}
            cache.write(snapshot)
            self.assertEqual(cache.read("passing", {"team": 1}), snapshot)
            self.assertIsNone(app.DiskCache(app.Config(cache_dir=Path(temporary), refresh=True)).read(
                "passing", {"team": 1}))
            self.assertIsNone(app.DiskCache(app.Config(cache_dir=Path(temporary), max_cache_age_hours=0)).read(
                "passing", {"team": 1}))
            cache.path_for("passing", {"team": 1}).write_text("broken json")
            self.assertIsNone(cache.read("passing", {"team": 1}))


class RequestTests(unittest.TestCase):
    def setUp(self):
        self.client = app.NBAClient(app.Config(min_delay=0, max_delay=0, retries=2, backoff_factor=0))
        self.sleep = patch.object(app.time, "sleep")
        self.mock_sleep = self.sleep.start()

    def tearDown(self):
        self.sleep.stop()
        self.client.close()

    def test_403_stops_without_retry(self):
        factory = Mock(side_effect=http_error(403))
        with self.assertRaises(app.DownloadsStopped):
            self.client.request(factory)
        self.assertEqual(factory.call_count, 1)
        self.assertIsNotNone(self.client.stopped_reason)

    def test_429_respects_retry_after_then_succeeds(self):
        factory = Mock(side_effect=[http_error(429, "7"), {"ok": True}])
        self.assertEqual(self.client.request(factory), {"ok": True})
        self.assertIn(unittest.mock.call(7.0), self.mock_sleep.call_args_list)

    def test_persistent_429_stops_run(self):
        with self.assertRaises(app.DownloadsStopped):
            self.client.request(Mock(side_effect=http_error(429)))

    def test_long_retry_after_stops_instead_of_retrying_early(self):
        factory = Mock(side_effect=http_error(429, "3600"))
        with self.assertRaises(app.DownloadsStopped):
            self.client.request(factory)
        self.assertEqual(factory.call_count, 1)

    def test_repeated_timeout_failures_eventually_stop(self):
        factory = Mock(side_effect=requests.Timeout("offline"))
        for _ in range(2):
            with self.assertRaises(app.DataUnavailable):
                self.client.request(factory)
        with self.assertRaises(app.DownloadsStopped):
            self.client.request(factory)
        self.assertEqual(factory.call_count, 6)

    def test_raised_http_status_keeps_response_details(self):
        error = http_error(429, "9")
        with self.assertRaises(requests.HTTPError) as raised:
            self.client._check_status(error.response)
        self.assertEqual(app.retry_after_seconds(raised.exception.response), 9)

    def test_retry_after_accepts_http_dates(self):
        from datetime import datetime, timedelta, timezone
        from email.utils import format_datetime
        response = requests.Response()
        response.headers["Retry-After"] = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=30))
        self.assertGreater(app.retry_after_seconds(response), 28)
        self.assertLessEqual(app.retry_after_seconds(response), 30)


class APIIntegrationTests(unittest.TestCase):
    """Exercise real nba_api endpoint parsing through a mocked HTTP adapter."""

    @staticmethod
    def response(request, raw, status=200):
        result = requests.Response()
        result.status_code = status
        result._content = json.dumps(raw).encode()
        result._content_consumed = True
        result.url = request.url
        result.request = request
        return result

    def test_real_endpoint_adapters_validation_and_cache(self):
        columns = ["PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION", "GP", "MIN"]
        roster_raw = payload("LeagueDashPlayerStats", columns, [[1, "Player A", TEAM["id"], "GSW", 3, 10]])
        pass_raw = payload("PassesMade", ["PLAYER_ID", "TEAM_ID", "PASS_TEAMMATE_PLAYER_ID", "PASS"],
                           [[1, TEAM["id"], 2, 10]])
        pass_raw["resultSets"].append({"name": "PassesReceived", "headers": [], "rowSet": []})
        with tempfile.TemporaryDirectory() as temporary:
            client = app.NBAClient(app.Config(cache_dir=Path(temporary), min_delay=0, max_delay=0))
            try:
                def send(request, **kwargs):
                    return self.response(request, roster_raw if "leaguedashplayerstats" in request.url else pass_raw)

                with patch("requests.adapters.HTTPAdapter.send", side_effect=send) as adapter:
                    players, snapshot = client.roster(TEAM)
                    passing, _ = client.passing(1, TEAM["id"])
                    self.assertEqual(players["PLAYER_ID"].tolist(), [1])
                    self.assertEqual(passing["PASS"].sum(), 10)
                    self.assertEqual(snapshot["parameters"]["TeamID"], TEAM["id"])
                    self.assertEqual(snapshot["parameters"]["PerMode"], "Totals")
                    self.assertIn("LastNGames", snapshot["parameters"])
                    self.assertIn("library_version", snapshot)
                    client.passing(1, TEAM["id"])
                    self.assertEqual(adapter.call_count, 2)
            finally:
                client.close()

    def test_real_endpoint_403_is_not_hidden_by_nba_api(self):
        client = app.NBAClient(app.Config(min_delay=0, max_delay=0))
        try:
            with patch("requests.adapters.HTTPAdapter.send",
                       side_effect=lambda request, **kwargs: self.response(request, {}, status=403)) as adapter:
                with self.assertRaises(app.DownloadsStopped):
                    client.passing(1, TEAM["id"])
                self.assertEqual(adapter.call_count, 1)
        finally:
            client.close()


class PipelineTests(unittest.TestCase):
    @staticmethod
    def complete_data(team="BOS", multiplier=1):
        from dataclasses import replace
        data = app.build_team_data(TEAM, roster(), {
            1: profile(1, [(2, 3)]), 2: profile(2, [(3, 3)]), 3: profile(3, [(1, 3)])
        }, {}, [], app.Config())
        return replace(data, team=team, matrix=data.matrix * multiplier)

    def test_access_denial_marks_remaining_players_without_more_downloads(self):
        client = Mock(config=app.Config(), stopped_reason=None)
        client.roster.return_value = (roster(), {})

        def passing(pid, team_id):
            if pid == 1:
                return profile(1, [(3, 10)]), {}
            client.stopped_reason = "HTTP 403"
            raise app.DownloadsStopped("HTTP 403")

        client.passing.side_effect = passing
        data = app.run_single_team_pipeline(TEAM, client)
        self.assertEqual(client.passing.call_count, 2)
        self.assertEqual(set(data.missing), {2, 3})
        self.assertEqual(np.nansum(data.matrix), 10)

    def test_all_team_mode_and_saved_run_manifest(self):
        team_b = {"id": 1610612738, "abbreviation": "BOS"}
        complete = app.build_team_data(TEAM, roster(), {
            1: profile(1, [(2, 3)]), 2: profile(2, [(3, 3)]), 3: profile(3, [(1, 3)])
        }, {}, [], app.Config())

        def run(team, client):
            from dataclasses import replace
            return replace(complete, team=team["abbreviation"])

        with tempfile.TemporaryDirectory() as temporary, \
             patch.object(app.static_teams, "get_teams", return_value=[TEAM, team_b]), \
             patch.object(app, "NBAClient") as client_class, \
             patch.object(app, "run_single_team_pipeline", side_effect=run) as pipeline, \
             patch.object(app, "plot_team_network"):
            client_class.return_value.stopped_reason = None
            self.assertEqual(app.main(["--team", "ALL", "--no-show", "--output-dir", temporary]), 0)
            self.assertEqual(pipeline.call_count, 2)
            manifest = json.loads((Path(temporary) / "2025-26" / "Regular_Season" / "run_summary.json").read_text())
            self.assertEqual({item["team"] for item in manifest["teams"]}, {"GSW", "BOS"})
            self.assertEqual(manifest["not_processed"], [])

    def test_default_all_teams_checkpoints_before_next_team_and_plot(self):
        teams = [TEAM, {"id": 1610612738, "abbreviation": "BOS"}]
        seen = []
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "2025-26" / "Regular_Season"

            def run(team, client):
                if seen:
                    saved = pd.read_csv(output / "team_metrics.csv")
                    self.assertEqual(saved["team"].tolist(), ["BOS"])
                    self.assertEqual(pd.read_csv(output / "all_team_passes.csv")["team"].unique().tolist(), ["BOS"])
                    checkpoint = json.loads((output / "run_summary.json").read_text())
                    self.assertEqual(checkpoint["not_processed"], ["GSW"])
                seen.append(team["abbreviation"])
                return self.complete_data(team["abbreviation"])

            def plot(data, path, **kwargs):
                self.assertIn(data.team, pd.read_csv(output / "team_metrics.csv")["team"].tolist())
                self.assertTrue((output / data.team / "passing_network.graphml").exists())

            with patch.object(app.static_teams, "get_teams", return_value=teams), \
                 patch.object(app, "NBAClient") as client_class, \
                 patch.object(app, "run_single_team_pipeline", side_effect=run), \
                 patch.object(app, "plot_team_network", side_effect=plot):
                client_class.return_value.stopped_reason = None
                self.assertEqual(app.main(["--no-show", "--output-dir", temporary]), 0)
            self.assertEqual(seen, ["BOS", "GSW"])
            self.assertEqual(len(pd.read_csv(output / "team_metrics.csv")), 2)
            self.assertEqual(len(pd.read_csv(output / "all_team_passes.csv")), 12)

    def test_finished_team_csv_survives_later_failure_and_interruption(self):
        teams = [TEAM, {"id": 1610612738, "abbreviation": "BOS"}]
        for failure in (app.DataUnavailable("offline"), KeyboardInterrupt()):
            with self.subTest(failure=type(failure).__name__), tempfile.TemporaryDirectory() as temporary, \
                 patch.object(app.static_teams, "get_teams", return_value=teams), \
                 patch.object(app, "NBAClient") as client_class, \
                 patch.object(app, "run_single_team_pipeline", side_effect=[self.complete_data(), failure]), \
                 patch.object(app, "plot_team_network"):
                client_class.return_value.stopped_reason = None
                if isinstance(failure, KeyboardInterrupt):
                    with self.assertRaises(KeyboardInterrupt):
                        app.main(["--no-show", "--output-dir", temporary])
                else:
                    self.assertEqual(app.main(["--no-show", "--output-dir", temporary]), 1)
                output = Path(temporary) / "2025-26" / "Regular_Season"
                metrics = pd.read_csv(output / "team_metrics.csv")
                self.assertEqual(metrics.loc[metrics["team"] == "BOS", "status"].iloc[0], "complete")
                passes = pd.read_csv(output / "all_team_passes.csv")
                self.assertEqual(passes["team"].unique().tolist(), ["BOS"])
                self.assertEqual(passes["pass_count"].sum(), 9)
                self.assertTrue((output / "BOS" / "passing_matrix.csv").exists())
                if isinstance(failure, app.DataUnavailable):
                    failed = metrics.loc[metrics["team"] == "GSW"].iloc[0]
                    self.assertEqual(failed["status"], "failed")
                    self.assertTrue(pd.isna(failed["recorded_passes_in_available_profiles"]))

    def test_rerun_updates_team_without_duplicates_and_retains_other_teams(self):
        teams = [TEAM, {"id": 1610612738, "abbreviation": "BOS"}]
        with tempfile.TemporaryDirectory() as temporary, \
             patch.object(app.static_teams, "get_teams", return_value=teams), \
             patch.object(app, "NBAClient") as client_class, \
             patch.object(app, "run_single_team_pipeline") as pipeline, \
             patch.object(app, "plot_team_network"):
            client_class.return_value.stopped_reason = None
            pipeline.side_effect = lambda team, client: self.complete_data(team["abbreviation"])
            self.assertEqual(app.main(["--no-show", "--output-dir", temporary]), 0)
            pipeline.side_effect = lambda team, client: self.complete_data(team["abbreviation"], multiplier=2)
            self.assertEqual(app.main(["--team", "BOS", "--no-show", "--output-dir", temporary]), 0)
            output = Path(temporary) / "2025-26" / "Regular_Season"
            metrics = pd.read_csv(output / "team_metrics.csv").set_index("team")
            self.assertEqual(len(metrics), 2)
            self.assertEqual(metrics.loc["BOS", "recorded_passes_in_available_profiles"], 18)
            self.assertEqual(metrics.loc["GSW", "recorded_passes_in_available_profiles"], 9)
            pairs = pd.read_csv(output / "all_team_passes.csv")
            self.assertEqual(len(pairs), 12)
            self.assertEqual(pairs.loc[pairs["team"] == "BOS", "pass_count"].sum(), 18)

    def test_plot_failure_does_not_stop_team_csv_exports(self):
        teams = [TEAM, {"id": 1610612738, "abbreviation": "BOS"}]
        with tempfile.TemporaryDirectory() as temporary, \
             patch.object(app.static_teams, "get_teams", return_value=teams), \
             patch.object(app, "NBAClient") as client_class, \
             patch.object(app, "run_single_team_pipeline",
                          side_effect=lambda team, client: self.complete_data(team["abbreviation"])), \
             patch.object(app, "plot_team_network", side_effect=[RuntimeError("backend failure"), None]):
            client_class.return_value.stopped_reason = None
            self.assertEqual(app.main(["--no-show", "--output-dir", temporary]), 1)
            output = Path(temporary) / "2025-26" / "Regular_Season"
            self.assertEqual(len(pd.read_csv(output / "team_metrics.csv")), 2)
            self.assertEqual(json.loads((output / "run_summary.json").read_text())["plot_errors"],
                             {"BOS": "backend failure"})


class AtomicSaveTests(unittest.TestCase):
    def test_interrupted_write_preserves_previous_csv(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "saved.csv"
            path.write_text("team,passes\nBOS,9\n")

            def interrupted(handle):
                handle.write("unfinished replacement")
                raise RuntimeError("simulated interruption")

            with self.assertRaises(RuntimeError):
                app.atomic_write(path, interrupted)
            self.assertEqual(path.read_text(), "team,passes\nBOS,9\n")
            self.assertEqual(list(Path(temporary).glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
