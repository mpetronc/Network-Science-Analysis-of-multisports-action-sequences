import os
import pickle
import random
import time
import matplotlib.pyplot as plt
import networkx as np_nx
import numpy as np
import pandas as pd
from nba_api.stats.endpoints import leaguedashplayerstats, playerdashptpass
from nba_api.stats.library import http as nba_http
from nba_api.stats.static import players as static_players
from nba_api.stats.static import teams as static_teams

# Configuration: Use user home directory for safe cache permissions
SEASON = "2025-26"
TARGET_TEAM = "GSW"  # Change to any team abbreviation, e.g., "LAL", "BOS", "MIL"
CACHE_DIR = os.path.expanduser("~/nba_passing_cache")
if not os.path.exists(CACHE_DIR):
  os.makedirs(CACHE_DIR, exist_ok=True)


def get_team_players(team_abbr, retries=4, backoff_factor=3.0):
  """Fetches league stats and filters players strictly for the target team."""
  print(f"Fetching player list for {team_abbr}...")

  for attempt in range(retries):
    try:
      stats = leaguedashplayerstats.LeagueDashPlayerStats(
          season=SEASON,
          measure_type_detailed_defense="Advanced",
          timeout=60,
          headers={
              "User-Agent": (
                  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
                  " (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
              ),
              "Accept": "application/json, text/plain, */*",
              "Accept-Language": "en-US,en;q=0.9",
              "Referer": "https://www.nba.com/",
              "Origin": "https://www.nba.com",
          },
      )
      df = stats.get_data_frames()[0]

      # Filter by team and active minutes/games
      if "GP" in df.columns and "MIN" in df.columns:
        df = df[
            (df["TEAM_ABBREVIATION"] == team_abbr)
            & (df["GP"] > 0)
            & (df["MIN"] > 0)
        ]

      return df[["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION"]].to_dict(
          "records"
      )
    except Exception as e:
      print(f"Attempt {attempt + 1} failed to fetch player list: {e}")
      try:
        nba_http.NBAStatsHTTP.clear_session()
      except Exception:
        pass

      if attempt < retries - 1:
        sleep_time = backoff_factor * (attempt + 1)
        print(f"Retrying in {sleep_time} seconds...")
        time.sleep(sleep_time)
      else:
        raise RuntimeError("Failed to fetch player stats after maximum retries.")


def fetch_player_passing(player_id, team_id, retries=4, backoff_factor=4.0):
  """Fetches passing data using browser headers, caching, and socket clearing on timeout."""
  cache_file = os.path.join(CACHE_DIR, f"player_{player_id}_{SEASON}.pkl")

  if os.path.exists(cache_file):
    with open(cache_file, "rb") as f:
      return pickle.load(f)

  for attempt in range(retries):
    try:
      time.sleep(random.uniform(2.0, 3.5))

      pass_endpoint = playerdashptpass.PlayerDashPtPass(
          player_id=player_id,
          team_id=team_id,
          season=SEASON,
          season_type_all_star="Regular Season",
          headers={
              "User-Agent": (
                  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
                  " (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
              ),
              "Accept": "application/json, text/plain, */*",
              "Accept-Language": "en-US,en;q=0.9",
              "Referer": "https://www.nba.com/",
              "Origin": "https://www.nba.com",
          },
          timeout=60,
      )
      dfs = pass_endpoint.get_data_frames()

      passes_made = dfs[0] if len(dfs) > 0 else pd.DataFrame()
      passes_received = dfs[1] if len(dfs) > 1 else pd.DataFrame()

      data = {"made": passes_made, "received": passes_received}

      with open(cache_file, "wb") as f:
        pickle.dump(data, f)

      return data

    except Exception as e:
      print(
          f"Block/Timeout on player {player_id} (Attempt"
          f" {attempt + 1}/{retries}): {e}"
      )
      try:
        nba_http.NBAStatsHTTP.clear_session()
      except Exception:
        pass

      if attempt < retries - 1:
        sleep_time = backoff_factor * (2**attempt)
        print(f"Cooling down for {sleep_time:.1f} seconds to reset socket...")
        time.sleep(sleep_time)
      else:
        print(f"Failed to fetch player {player_id} after max retries. Skipping.")
        return None


def compute_team_entropies(matrix):
  """Computes normalized receiver entropy and conditional entropy for a team matrix."""
  total_passes = matrix.sum()
  n = len(matrix)
  if total_passes == 0 or n <= 1:
    return 0.0, 0.0

  max_entropy = np.log(n - 1) if n > 1 else 1.0

  incoming_passes = matrix.sum(axis=0)
  p_receiver = incoming_passes / total_passes
  p_receiver = p_receiver[p_receiver > 0]
  receiver_entropy = (
      -np.sum(p_receiver * np.log(p_receiver)) / max_entropy
      if max_entropy > 0
      else 0.0
  )

  outgoing_passes = matrix.sum(axis=1)
  team_pass_share = outgoing_passes / total_passes

  conditional_entropy = 0.0
  for i in range(n):
    if outgoing_passes[i] > 0:
      p_choice = matrix[i, :] / outgoing_passes[i]
      p_choice = p_choice[p_choice > 0]
      h_i = -np.sum(p_choice * np.log(p_choice))
      norm_h_i = h_i / max_entropy if max_entropy > 0 else 0.0
      conditional_entropy += team_pass_share[i] * norm_h_i

  return float(receiver_entropy), float(conditional_entropy)


def plot_team_network(team_abbr, team_data):
  """Plots the network graph for a specific team using NetworkX and Matplotlib."""
  matrix = team_data["matrix"]
  player_ids = team_data["player_ids"]

  id_to_name = {}
  for pid in player_ids:
    p_info = static_players.find_player_by_id(pid)
    if p_info:
      name_parts = p_info["full_name"].split(" ")
      short_name = (
          f"{name_parts[0][0]}. {name_parts[-1]}"
          if len(name_parts) > 1
          else p_info["full_name"]
      )
      id_to_name[pid] = short_name
    else:
      id_to_name[pid] = str(pid)

  G = np_nx.DiGraph()
  n = len(matrix)
  for i in range(n):
    for j in range(n):
      weight = matrix[i, j]
      if weight > 0:
        G.add_edge(
            id_to_name[player_ids[i]], id_to_name[player_ids[j]], weight=weight
        )

  if len(G.edges) == 0:
    print(f"No passing edges found to plot for {team_abbr}.")
    return

  plt.figure(figsize=(10, 10))
  plt.title(
      f"{team_abbr} Passing Network Graph ({SEASON} Regular Season)", fontsize=14
  )

  pos = np_nx.spring_layout(G, k=0.2, seed=42)

  node_sizes = [
      (G.in_degree(node, weight="weight") + G.out_degree(node, weight="weight"))
      * 1.2
      for node in G.nodes()
  ]

  edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
  max_w = max(edge_weights) if edge_weights else 1
  edge_widths = [0.5 + 4.0 * (w / max_w) for w in edge_weights]

  np_nx.draw_networkx_nodes(
      G, pos, node_size=node_sizes, node_color="gold", alpha=0.9
  )
  np_nx.draw_networkx_edges(
      G,
      pos,
      width=edge_widths,
      edge_color="royal blue",
      alpha=0.6,
      arrows=True,
      arrowsize=12,
      connectionstyle="arc3,rad=0.1",
  )
  np_nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")

  plt.axis("off")
  plt.tight_layout()
  plt.show()


def run_single_team_pipeline(team_abbr):
  team_players = get_team_players(team_abbr)
  print(
      f"Processing target team: {team_abbr} ({len(team_players)} active"
      " players)..."
  )

  team_info = [
      t for t in static_teams.get_teams() if t["abbreviation"] == team_abbr
  ]
  if not team_info:
    print(f"Team {team_abbr} not found in static teams.")
    return None

  team_id = team_info[0]["id"]
  team_passing_data = {}

  for p in team_players:
    p_id = p["PLAYER_ID"]
    data = fetch_player_passing(p_id, team_id)
    if data and not data["made"].empty:
      team_passing_data[p_id] = data["made"]

  if not team_passing_data:
    print("No passing data collected.")
    return None

  active_p_ids = list(team_passing_data.keys())
  n = len(active_p_ids)
  id_to_idx = {pid: idx for idx, pid in enumerate(active_p_ids)}

  adjacency_matrix = np.zeros((n, n))

  for passer_id, df_made in team_passing_data.items():
    if passer_id not in id_to_idx:
      continue
    i = id_to_idx[passer_id]

    target_id_col = (
        "PASS_TEAMMATE_PLAYER_ID"
        if "PASS_TEAMMATE_PLAYER_ID" in df_made.columns
        else None
    )
    pass_val_col = "PASS" if "PASS" in df_made.columns else None

    if not target_id_col or not pass_val_col:
      continue

    for _, row in df_made.iterrows():
      receiver_id = row[target_id_col]
      passes_count = row[pass_val_col]

      if receiver_id in id_to_idx:
        j = id_to_idx[receiver_id]
        adjacency_matrix[i, j] = passes_count

  rec_ent, cond_ent = compute_team_entropies(adjacency_matrix)

  print("\n" + "=" * 50)
  print(f" TEAM PASSING METRICS: {team_abbr}")
  print("=" * 50)
  print(f"Active Players Analyzed: {n}")
  print(f"Total Team Passes: {int(adjacency_matrix.sum())}")
  print(f"Receiver Entropy: {rec_ent:.4f}")
  print(f"Conditional Entropy: {cond_ent:.4f}")

  team_data = {
      "matrix": adjacency_matrix,
      "player_ids": active_p_ids,
      "id_to_idx": id_to_idx,
  }
  return team_data


if __name__ == "__main__":
  gsw_data = run_single_team_pipeline(TARGET_TEAM)

  if gsw_data:
    print(f"\nGenerating network plot for {TARGET_TEAM}...")
    plot_team_network(TARGET_TEAM, gsw_data)
