from __future__ import annotations

from pathlib import Path
import json
import re

import pandas as pd
import requests


HISTORICAL_BASE = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
MASTER_TEAM_LIST_URL = f"{HISTORICAL_BASE}/master_team_list.csv"
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


def make_player_id(value):
    # Canonical cross-season player ID. Strips vaastav's numeric suffixes
    # (e.g. "Mohamed_Salah_233" -> "mohamedsalah") so historical and live rows match.
    cleaned = re.sub(r"[^a-z]+", "", str(value).lower())
    return cleaned


def load_master_team_list(extra_seasons=None):
    # vaastav publishes a season-by-season team_id -> team_name table, but it
    # lags behind by ~1 season. Fall back to each season's own teams.csv for
    # anything master_team_list doesn't cover yet.
    # Columns: season, team (id within that season), team_name.
    base = pd.read_csv(MASTER_TEAM_LIST_URL)
    base = base[["season", "team", "team_name"]]
    covered = set(base["season"].unique())

    if extra_seasons is None:
        extra_seasons = []

    extra_frames = []
    for season in extra_seasons:
        if season in covered:
            continue
        try:
            per_season = pd.read_csv(f"{HISTORICAL_BASE}/{season}/teams.csv")[["id", "name"]]
        except Exception:
            continue
        per_season = per_season.rename(columns={"id": "team", "name": "team_name"})
        per_season["season"] = season
        extra_frames.append(per_season[["season", "team", "team_name"]])

    if extra_frames:
        base = pd.concat([base, *extra_frames], ignore_index=True)
    return base


def _resolve_team_name_for_season(frame, season, team_map):
    # Normalize each season's table to a `team_name` string column.
    # 2020-21+ merged_gw.csv has team as a club name string already.
    # 2016-17 to 2019-20 has no team column at all, so backfill via
    # players_raw.csv (element -> numeric team id) + master_team_list (id -> name).
    if "team" in frame.columns and frame["team"].dtype == object:
        return frame.rename(columns={"team": "team_name"})

    try:
        roster = pd.read_csv(f"{HISTORICAL_BASE}/{season}/players_raw.csv")[["id", "team"]]
    except Exception:
        frame["team_name"] = pd.NA
        return frame

    season_map = team_map.loc[team_map["season"] == season, ["team", "team_name"]]
    roster = roster.merge(season_map, on="team", how="left").drop(columns=["team"])
    roster = roster.rename(columns={"id": "element"})
    if "team" in frame.columns:
        frame = frame.drop(columns=["team"])
    return frame.merge(roster, on="element", how="left")


def load_historical_fpl(seasons, team_map=None):
    # Download all requested seasons and combine into one table.
    if team_map is None:
        team_map = load_master_team_list()

    all_frames = []
    for season in seasons:
        url = f"{HISTORICAL_BASE}/{season}/gws/merged_gw.csv"
        try:
            frame = pd.read_csv(url)
        except UnicodeDecodeError:
            frame = pd.read_csv(url, encoding="latin-1")
        frame["season"] = season
        frame = _resolve_team_name_for_season(frame, season, team_map)
        all_frames.append(frame)

    data = pd.concat(all_frames, ignore_index=True)
    data = data.rename(columns={"round": "gw"})
    data["kickoff_time"] = pd.to_datetime(data["kickoff_time"], errors="coerce", utc=True)

    for col in NUMERIC_COLUMNS:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    for col in ["opponent_team", "gw", "element"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    data["season_start"] = data["season"].map(season_start_year)
    data["name_key"] = data["name"].map(make_name_key)
    data["player_id"] = data["name"].map(make_player_id)
    data = data.sort_values(["season", "element", "gw"]).reset_index(drop=True)
    return data


def attach_opponent_team_name(data, team_map):
    # opponent_team is always an integer ID; resolve to a canonical club name
    # so opponent-strength features survive across seasons.
    opp_lookup = team_map.rename(
        columns={"team": "_team_id", "team_name": "opponent_team_name"}
    )
    data = data.merge(
        opp_lookup,
        left_on=["season", "opponent_team"],
        right_on=["season", "_team_id"],
        how="left",
    ).drop(columns=["_team_id"])
    return data


def build_season_anchor_table(data):
    # One row per (season_start, player_id) summarizing the prior season.
    # ppg uses appearances (minutes > 0) not 38, so bench players aren't penalised.
    # minutes_share normalises against a full season (38 GWs * 90 min).
    # The season_start column in the output is the season the anchor APPLIES to
    # (i.e. already shifted forward by 1), so a left-merge "just works".
    appeared = (data["minutes"].fillna(0) > 0).astype(int)
    season_agg = (
        data.assign(_appeared=appeared)
        .groupby(["season_start", "player_id"], as_index=False)
        .agg(
            _season_points=("total_points", "sum"),
            _season_minutes=("minutes", "sum"),
            _season_apps=("_appeared", "sum"),
        )
    )
    season_agg["last_season_ppg"] = (
        season_agg["_season_points"] / season_agg["_season_apps"].replace(0, pd.NA)
    )
    season_agg["last_season_minutes_share"] = season_agg["_season_minutes"] / (38 * 90)
    season_agg["season_start"] = season_agg["season_start"] + 1
    return season_agg[
        ["season_start", "player_id", "last_season_ppg", "last_season_minutes_share"]
    ]


def add_cross_season_anchors(data, anchor_table):
    # Join prior-season anchors. NaN for first-PL-season players is expected
    # and meaningful — combined with is_promoted_team, it tells the model
    # "no history, use the cold-start prior."
    return data.merge(anchor_table, on=["season_start", "player_id"], how="left")


def add_promoted_flag(data):
    # Mark (season, team) as promoted if the team wasn't in the previous season.
    # Earliest season is set to NA because we have no prior to compare against.
    season_teams = data[["season_start", "team_name"]].drop_duplicates().dropna()
    prev = season_teams.copy()
    prev["season_start"] = prev["season_start"] + 1
    prev["_was_in_pl_last"] = True

    earliest = int(season_teams["season_start"].min())
    data = data.merge(prev, on=["season_start", "team_name"], how="left")
    data["is_promoted_team"] = data["_was_in_pl_last"].isna().astype(object)
    data.loc[data["season_start"] == earliest, "is_promoted_team"] = pd.NA
    return data.drop(columns=["_was_in_pl_last"])


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
        frame.groupby(["season", "team_name", "gw"], as_index=False)
        .agg(
            team_goals_scored_gw=("goals_scored", "sum"),
            team_goals_conceded_gw=("goals_conceded", "sum"),
            team_points_gw=("total_points", "sum"),
        )
        .sort_values(["season", "team_name", "gw"])
    )

    for metric in ["team_goals_scored_gw", "team_goals_conceded_gw", "team_points_gw"]:
        team_week[f"{metric}_roll5"] = team_week.groupby(["season", "team_name"], sort=False)[metric].transform(
            lambda s: s.shift(1).rolling(5, min_periods=1).mean()
        )

    frame = frame.merge(
        team_week[
            [
                "season",
                "team_name",
                "gw",
                "team_goals_scored_gw_roll5",
                "team_goals_conceded_gw_roll5",
                "team_points_gw_roll5",
            ]
        ],
        on=["season", "team_name", "gw"],
        how="left",
    )

    opponent_strength = team_week[
        ["season", "team_name", "gw", "team_points_gw_roll5", "team_goals_conceded_gw_roll5"]
    ].rename(
        columns={
            "team_name": "opponent_team_name",
            "team_points_gw_roll5": "opponent_team_points_roll5",
            "team_goals_conceded_gw_roll5": "opponent_team_gc_roll5",
        }
    )

    frame = frame.merge(
        opponent_strength,
        on=["season", "opponent_team_name", "gw"],
        how="left",
    )
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


def load_live_player_pool(team_map=None, anchor_table=None):
    # Get current-season players from the official FPL API.
    bootstrap = get_json_from_url(f"{FPL_API_BASE}/bootstrap-static/")
    teams = pd.DataFrame(bootstrap["teams"])[["id", "name", "short_name", "strength"]]
    teams = teams.rename(columns={"id": "team", "name": "team_name"})

    elements = pd.DataFrame(bootstrap["elements"])
    keep_cols = [
        "id",
        "first_name",
        "second_name",
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

    # Match vaastav's "First_Second" name format so player_id joins historical rows.
    elements["full_name"] = (
        elements["first_name"].astype(str) + "_" + elements["second_name"].astype(str)
    )
    elements["player_id"] = elements["full_name"].map(make_player_id)
    elements["name_key"] = elements["web_name"].map(make_name_key)
    elements = elements.merge(teams, on="team", how="left")

    if anchor_table is not None:
        elements = elements.merge(
            anchor_table, on=["season_start", "player_id"], how="left"
        )
    else:
        elements["last_season_ppg"] = pd.NA
        elements["last_season_minutes_share"] = pd.NA

    # Promoted = club not present in 2024-25 master team list.
    if team_map is not None:
        prev_teams = set(team_map.loc[team_map["season"] == "2024-25", "team_name"])
        if prev_teams:
            elements["is_promoted_team"] = ~elements["team_name"].isin(prev_teams)
        else:
            elements["is_promoted_team"] = pd.NA
    else:
        elements["is_promoted_team"] = pd.NA

    return elements


def build_dataset(root=".", seasons=DEFAULT_SEASONS):
    # Run the full pipeline and save both output CSV files.
    root_path = Path(root).resolve()
    processed_dir = root_path / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    team_map = load_master_team_list(extra_seasons=seasons)

    historical = load_historical_fpl(seasons, team_map=team_map)
    historical = attach_opponent_team_name(historical, team_map)
    historical = add_promoted_flag(historical)
    anchor_table = build_season_anchor_table(historical)
    historical = add_cross_season_anchors(historical, anchor_table)
    historical = add_leakage_safe_features(historical)
    historical = merge_understat(historical, seasons)

    live_pool = load_live_player_pool(team_map=team_map, anchor_table=anchor_table)

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
