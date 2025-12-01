# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO

st.set_page_config(page_title="FootLens Lite – Injuries & Impact", layout="wide")

# -------------------
# USER CONFIG: map your CSV columns here if names differ
# -------------------
COLMAP = {
    "player": ["Name", "player", "Player", "name"],
    "team": ["Team Name", "team", "Club", "club"],
    "match_date": ["match_date", "Match Date", "date", "Match_Date"],
    "injury_start": ["injury_start", "Injury Start", "injury_from"],
    "injury_end": ["injury_end", "Injury End", "injury_to"],
    "before_rating": ["Match1_before_injury_Player_rating", "rating_before", "before_rating"],
    "after_rating": ["Match1_after_injury_Player_rating", "rating_after", "after_rating"],
    "injury_date": ["Date of Injury", "date_of_injury", "injury_date"],
    "age": ["Age", "age"]
}

# Helper to find first matching column from possible names
def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

@st.cache_data
def load_and_clean(path="player_injuries_impact.csv"):
    df = pd.read_csv(path)
    # normalize column names (keep original but create canonical ones)
    canonical = {}
    for key, poss in COLMAP.items():
        found = find_col(df, poss)
        canonical[key] = found

    # create canonical columns (if found) to be used in the app
    for k, col in canonical.items():
        if col is not None:
            df[k] = df[col]
        else:
            df[k] = np.nan  # column missing -> fill with NaN

    # parse dates
    for dcol in ["match_date", "injury_start", "injury_end", "injury_date"]:
        if dcol in df.columns:
            df[dcol] = pd.to_datetime(df[dcol], errors="coerce")

    # numeric ratings
    for r in ["before_rating", "after_rating", "age"]:
        df[r] = pd.to_numeric(df[r], errors="coerce")

    # performance drop and phase labeling
    df["performance_drop"] = df["before_rating"] - df["after_rating"]

    def phase_label(row):
        s = row["injury_start"]
        e = row["injury_end"]
        m = row["match_date"]
        if pd.isna(s) or pd.isna(e) or pd.isna(m):
            return "Unknown / Not enough data"
        if m < s:
            return "Before injury"
        if s <= m <= e:
            return "During absence"
        return "After return"

    df["phase"] = df.apply(phase_label, axis=1)

    # month for heatmap
    df["injury_month"] = df["injury_date"].dt.month_name().str.slice(stop=3)

    return df, canonical

df, canonical = load_and_clean()

# -------------------
# SIDEBAR
# -------------------
st.sidebar.title("Filters & Controls")
teams = ["All"] + sorted(df["team"].dropna().unique().tolist())
players = ["All"] + sorted(df["player"].dropna().unique().tolist())

selected_team = st.sidebar.selectbox("Team", options=teams)
selected_player = st.sidebar.selectbox("Player (highlight)", options=players)
min_date = df["match_date"].min()
max_date = df["match_date"].max()
date_range = st.sidebar.date_input("Match date range", value=(min_date, max_date) if not pd.isna(min_date) else None)

show_only_significant = st.sidebar.checkbox("Only show significant drops (>= 2.0)", value=False)
download_button = st.sidebar.checkbox("Show export button", value=True)

# Apply filters
fdf = df.copy()
if selected_team != "All":
    fdf = fdf[fdf["team"] == selected_team]
if selected_player != "All":
    fdf = fdf[fdf["player"] == selected_player]
# date range filter
if isinstance(date_range, (list, tuple)) and len(date_range) == 2 and fdf["match_date"].notna().any():
    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    fdf = fdf[(fdf["match_date"] >= start) & (fdf["match_date"] <= end)]

if show_only_significant:
    fdf = fdf[fdf["performance_drop"] >= 2.0]

# -------------------
# TOP KPI ROW
# -------------------
st.title("⚽ FootLens Lite — Injuries & Performance Impact")
k1, k2, k3, k4 = st.columns(4)

with k1:
    st.metric("Records (filtered)", len(fdf))

with k2:
    total_injuries = fdf["injury_date"].notna().sum()
    st.metric("Injury
