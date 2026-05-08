from __future__ import annotations

from pathlib import Path
import json
import re

import pandas as pd
import requests


HISTORICAL_BASE = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
FPL_API_BASE = "https://fantasy.premierleague.com/api"
UNDERSTAT_LEAGUE_URL = "https://understat.com/league/EPL/{season_start}"

DEFAULT_SEASONS = [
    "2016-17",
    "2017-18",
    "2018-19",
    "2019-20",
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
]

NUMERIC_COLUMNS = [
    "total_points",
    "minutes",
    "goals_scored",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "expected_goals_conceded",
    "xP",
    "value",
    "selected",
    "transfers_in",
    "transfers_out",
    "transfers_balance",
]

ROLLING_SOURCE_COLUMNS = [
    "total_points",
    "minutes",
    "goals_scored",
    "assists",
    "expected_goals",
    "expected_assists",
]

def get_json_from_url(url, timeout=30):
    # Get JSON data from a URL.
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.json()


def get_text_from_url(url, timeout=30):
    # Get raw text (HTML) from a URL.
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.text


def season_start_year(season):
    # Turn "2023-24" into 2023.
    return int(season.split("-")[0])


def make_name_key(value):
    # Normalize a player name so merges are easier.
    cleaned = re.sub(r"[^a-z0-9]+", "", str(value).lower())
    return cleaned


def load_historical_fpl(seasons):
    # Download all requested seasons and combine into one table.
    all_frames = []
    for season in seasons:
        url = f"{HISTORICAL_BASE}/{season}/gws/merged_gw.csv"
        try:
            frame = pd.read_csv(url)
        except UnicodeDecodeError:
            frame = pd.read_csv(url, encoding="latin-1")
        frame["season"] = season
        all_frames.append(frame)

    data = pd.concat(all_frames, ignore_index=True)
    data = data.rename(columns={"round": "gw"})
    data["kickoff_time"] = pd.to_datetime(data["kickoff_time"], errors="coerce", utc=True)

    for col in NUMERIC_COLUMNS:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    for col in ["team", "opponent_team", "gw", "element"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    data["season_start"] = data["season"].map(season_start_year)
    data["name_key"] = data["name"].map(make_name_key)
    data = data.sort_values(["season", "element", "gw"]).reset_index(drop=True)
    return data


def add_leakage_safe_features(data):
    # Add rolling features that only use previous gameweeks.
    frame = data.copy()
    group_keys = ["season", "element"]

    frame["rest_days"] = (
        frame.groupby(group_keys, sort=False)["kickoff_time"]
        .diff()
        .dt.total_seconds()
        .div(86400)
    )

    for col in ROLLING_SOURCE_COLUMNS:
        for window in (3, 5):
            feature_name = f"{col}_roll{window}"
            frame[feature_name] = frame.groupby(group_keys, sort=False)[col].transform(
                lambda s: s.shift(1).rolling(window, min_periods=1).mean()
            )

    team_week = (
        frame.groupby(["season", "team", "gw"], as_index=False)
        .agg(
            team_goals_scored_gw=("goals_scored", "sum"),
            team_goals_conceded_gw=("goals_conceded", "sum"),
            team_points_gw=("total_points", "sum"),
        )
        .sort_values(["season", "team", "gw"])
    )

    for metric in ["team_goals_scored_gw", "team_goals_conceded_gw", "team_points_gw"]:
        team_week[f"{metric}_roll5"] = team_week.groupby(["season", "team"], sort=False)[metric].transform(
            lambda s: s.shift(1).rolling(5, min_periods=1).mean()
        )

    frame = frame.merge(
        team_week[
            [
                "season",
                "team",
                "gw",
                "team_goals_scored_gw_roll5",
                "team_goals_conceded_gw_roll5",
                "team_points_gw_roll5",
            ]
        ],
        on=["season", "team", "gw"],
        how="left",
    )

    opponent_strength = team_week[
        ["season", "team", "gw", "team_points_gw_roll5", "team_goals_conceded_gw_roll5"]
    ].rename(
        columns={
            "team": "opponent_team",
            "team_points_gw_roll5": "opponent_team_points_roll5",
            "team_goals_conceded_gw_roll5": "opponent_team_gc_roll5",
        }
    )

    frame = frame.merge(opponent_strength, on=["season", "opponent_team", "gw"], how="left")
    return frame


def load_understat_player_season(season_start):
    # Pull Understat season stats and keep useful columns.
    url = UNDERSTAT_LEAGUE_URL.format(season_start=season_start)
    html = get_text_from_url(url)
    match = re.search(r"playersData\s*=\s*JSON\.parse\('(.*?)'\)", html)
    if not match:
        return pd.DataFrame()

    payload = bytes(match.group(1), "utf-8").decode("unicode_escape")
    raw = json.loads(payload)
    frame = pd.DataFrame(raw)
    if frame.empty:
        return frame

    keep_cols = [
        "player_name",
        "team_title",
        "games",
        "time",
        "goals",
        "xG",
        "assists",
        "xA",
        "shots",
        "key_passes",
        "npg",
        "npxG",
    ]
    available_cols = [c for c in keep_cols if c in frame.columns]
    frame = frame[available_cols].copy()
    frame["season_start"] = season_start
    frame["name_key"] = frame["player_name"].map(make_name_key)

    for col in ["games", "time", "goals", "xG", "assists", "xA", "shots", "key_passes", "npg", "npxG"]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

    return frame


def merge_understat(data, seasons):
    # Merge Understat data into the historical FPL dataset.
    season_starts = sorted({season_start_year(season) for season in seasons})
    understat_frames = []

    for season_start in season_starts:
        try:
            season_df = load_understat_player_season(season_start)
            if not season_df.empty:
                understat_frames.append(season_df)
        except Exception:
            # Keep pipeline robust even if Understat temporarily blocks requests.
            continue

    if not understat_frames:
        return data

    understat = pd.concat(understat_frames, ignore_index=True)
    merge_cols = [c for c in understat.columns if c not in {"player_name", "team_title"}]
    merged = data.merge(understat[merge_cols], on=["season_start", "name_key"], how="left")
    return merged


def load_live_player_pool():
    # Get current-season players from the official FPL API.
    bootstrap = get_json_from_url(f"{FPL_API_BASE}/bootstrap-static/")
    teams = pd.DataFrame(bootstrap["teams"])[["id", "name", "short_name", "strength"]]
    teams = teams.rename(columns={"id": "team"})

    elements = pd.DataFrame(bootstrap["elements"])
    keep_cols = [
        "id",
        "web_name",
        "team",
        "element_type",
        "now_cost",
        "selected_by_percent",
        "form",
        "points_per_game",
        "minutes",
        "goals_scored",
        "assists",
        "clean_sheets",
        "expected_goals",
        "expected_assists",
        "expected_goal_involvements",
        "expected_goals_conceded",
        "transfers_in_event",
        "transfers_out_event",
        "status",
        "news",
    ]
    elements = elements[keep_cols].rename(columns={"id": "element"})
    elements["season"] = "2025-26"
    elements["season_start"] = 2025
    elements["name_key"] = elements["web_name"].map(make_name_key)
    elements = elements.merge(teams, on="team", how="left")
    return elements


def build_dataset(root=".", seasons=DEFAULT_SEASONS):
    # Run the full pipeline and save both output CSV files.
    root_path = Path(root).resolve()
    processed_dir = root_path / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    historical = load_historical_fpl(seasons)
    historical = add_leakage_safe_features(historical)
    historical = merge_understat(historical, seasons)

    live_pool = load_live_player_pool()

    historical_output = processed_dir / "fpl_model_dataset.csv"
    live_output = processed_dir / "live_player_pool.csv"
    historical.to_csv(historical_output, index=False)
    live_pool.to_csv(live_output, index=False)

    return {"historical_dataset": historical, "live_player_pool": live_pool}


if __name__ == "__main__":
    # Run this file directly to build datasets quickly.
    result = build_dataset(root=Path(__file__).resolve().parents[1])
    print(f"Historical dataset shape: {result['historical_dataset'].shape}")
    print(f"Live player pool shape: {result['live_player_pool'].shape}")
    print("Saved to data/processed/")
