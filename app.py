# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO

st.set_page_config(page_title="FootLens (matched to CSV) – Injuries & Impact", layout="wide")

# -----------------------
# Helper: expected column names (from your CSV)
# -----------------------
EXP_COLS = {
    "player": "Name",
    "team": "Team Name",
    "age": "Age",
    "injury_date": "Date of Injury",
    "return_date": "Date of return",
    # before ratings (Match1..3)
    "before_ratings": [
        "Match1_before_injury_Player_rating",
        "Match2_before_injury_Player_rating",
        "Match3_before_injury_Player_rating",
    ],
    # after ratings (Match1..3)
    "after_ratings": [
        "Match1_after_injury_Player_rating",
        "Match2_after_injury_Player_rating",
        "Match3_after_injury_Player_rating",
    ],
    # before goal diffs (optional)
    "before_gd": [
        "Match1_before_injury_GD",
        "Match2_before_injury_GD",
        "Match3_before_injury_GD",
    ],
    "after_gd": [
        "Match1_after_injury_GD",
        "Match2_after_injury_GD",
        "Match3_after_injury_GD",
    ],
    # season (optional)
    "season": "Season"
}

# -----------------------
# Load CSV (file in same folder)
# -----------------------
@st.cache_data
def load_df(path="player_injuries_impact.csv"):
    df = pd.read_csv(path, dtype=str)  # load as str first to clean
    # Replace "N.A." (literal in your CSV) and empty strings with NaN
    df = df.replace({"N.A.": np.nan, "NA": np.nan, "": np.nan})
    # Trim column names (in case of stray spaces)
    df.columns = [c.strip() for c in df.columns]
    return df

try:
    df_raw = load_df()
except FileNotFoundError:
    st.error("Could not find 'player_injuries_impact.csv' in the app folder. Upload it or adjust the path in load_df().")
    st.stop()
except Exception as e:
    st.error(f"Error loading CSV: {e}")
    st.stop()

# -----------------------
# Verify required columns exist and list missing
# -----------------------
present = list(df_raw.columns)
required_simple = [EXP_COLS["player"], EXP_COLS["team"], EXP_COLS["injury_date"]]
missing = [c for c in required_simple if c not in present]

# check at least one before and after rating exists
before_exist = [c for c in EXP_COLS["before_ratings"] if c in present]
after_exist = [c for c in EXP_COLS["after_ratings"] if c in present]

if missing:
    st.error(f"Missing required columns: {missing}. Please ensure your CSV contains these exact headers.")
    st.stop()

if not before_exist or not after_exist:
    st.error("Your CSV must contain at least one 'MatchX_before_injury_Player_rating' and one 'MatchX_after_injury_Player_rating' column.")
    st.stop()

# -----------------------
# Data cleaning & type conversion
# -----------------------
df = df_raw.copy()

# Convert dates
df[EXP_COLS["injury_date"]] = pd.to_datetime(df[EXP_COLS["injury_date"]], errors="coerce")
if EXP_COLS["return_date"] in df.columns:
    df[EXP_COLS["return_date"]] = pd.to_datetime(df[EXP_COLS["return_date"]], errors="coerce")

# Convert age to numeric if present
if EXP_COLS["age"] in df.columns:
    df[EXP_COLS["age"]] = pd.to_numeric(df[EXP_COLS["age"]], errors="coerce")
else:
    df[EXP_COLS["age"]] = np.nan

# Helper to parse numeric rating columns (some may be absent)
def col_to_numeric(df_, cols):
    found = []
    for c in cols:
        if c in df_.columns:
            df_[c] = pd.to_numeric(df_[c], errors="coerce")
            found.append(c)
    return found

before_cols = col_to_numeric(df, EXP_COLS["before_ratings"])
after_cols = col_to_numeric(df, EXP_COLS["after_ratings"])
before_gd_cols = col_to_numeric(df, EXP_COLS["before_gd"])
after_gd_cols = col_to_numeric(df, EXP_COLS["after_gd"])

# Compute average before and after ratings across available matches
df["avg_before_rating"] = df[before_cols].mean(axis=1, skipna=True)
df["avg_after_rating"] = df[after_cols].mean(axis=1, skipna=True)

# Compute average team goal-diff before/after if available (optional)
if before_gd_cols and after_gd_cols:
    df["avg_before_gd"] = df[before_gd_cols].mean(axis=1, skipna=True)
    df["avg_after_gd"] = df[after_gd_cols].mean(axis=1, skipna=True)
else:
    df["avg_before_gd"] = np.nan
    df["avg_after_gd"] = np.nan

# Performance drop: rating-based (primary)
df["performance_drop"] = df["avg_before_rating"] - df["avg_after_rating"]

# Phase: Before / During / After based on injury and return dates (if match date exists we can't derive exactly; we use presence)
def label_phase(row):
    start = row[EXP_COLS["injury_date"]]
    ret = row[EXP_COLS["return_date"]] if EXP_COLS["return_date"] in df.columns else pd.NaT
    if pd.isna(start):
        return "No recorded injury"
    if pd.notna(ret) and pd.notna(start):
        # If return date exists, we can classify as "During absence" for matches between
        # But since CSV rows are per-match, we'll approximate:
        return "Injury window"
    return "Injury recorded"

df["phase"] = df.apply(label_phase, axis=1)

# Injury month for heatmap
df["injury_month"] = df[EXP_COLS["injury_date"]].dt.month_name().str.slice(0,3)

# Season column safe
season_col = EXP_COLS["season"] if EXP_COLS["season"] in df.columns else None

# -----------------------
# Sidebar filters
# -----------------------
st.sidebar.header("Filters")

teams = sorted(df[EXP_COLS["team"]].dropna().unique().tolist())
players = sorted(df[EXP_COLS["player"]].dropna().unique().tolist())
seasons = sorted(df[season_col].dropna().unique().tolist()) if season_col else []

selected_team = st.sidebar.selectbox("Select team", options=["All"] + teams)
selected_player = st.sidebar.selectbox("Select player (optional)", options=["All"] + players)
selected_season = st.sidebar.selectbox("Select season (optional)", options=["All"] + seasons) if seasons else "All"

date_min = df[EXP_COLS["injury_date"]].min()
date_max = df[EXP_COLS["injury_date"]].max()
date_range = st.sidebar.date_input("Injury date range", value=(date_min, date_max) if pd.notna(date_min) else None)

significant_only = st.sidebar.checkbox("Only show significant drops (>= 2.0)", value=False)
export_csv = st.sidebar.checkbox("Show export button", value=True)

# Subset dataframe according to filters
fdf = df.copy()
if selected_team != "All":
    fdf = fdf[fdf[EXP_COLS["team"]] == selected_team]
if selected_player != "All":
    fdf = fdf[fdf[EXP_COLS["player"]] == selected_player]
if selected_season != "All" and season_col:
    fdf = fdf[fdf[season_col] == selected_season]
# date range filter if valid
if isinstance(date_range, (list, tuple)) and len(date_range) == 2 and fdf[EXP_COLS["injury_date"]].notna().any():
    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    fdf = fdf[(fdf[EXP_COLS["injury_date"]] >= start) & (fdf[EXP_COLS["injury_date"]] <= end)]

if significant_only:
    fdf = fdf[fdf["performance_drop"] >= 2.0]

# -----------------------
# KPIs
# -----------------------
st.title("⚽ FootLens – Player Injuries & Team Performance (matched to your CSV)")
k1, k2, k3, k4 = st.columns(4)

with k1:
    st.metric("Filtered records", len(fdf))

with k2:
    st.metric("Injury records", int(fdf[EXP_COLS["injury_date"]].notna().sum()))

with k3:
    avg_drop = fdf["performance_drop"].dropna().mean()
    st.metric("Avg performance drop", f"{avg_drop:.2f}" if not np.isnan(avg_drop) else "N/A")

with k4:
    before_m = fdf["avg_before_rating"].dropna().mean()
    after_m = fdf["avg_after_rating"].dropna().mean()
    st.metric("Avg before → after", f"{before_m:.2f} → {after_m:.2f}" if not (np.isnan(before_m) or np.isnan(after_m)) else "N/A")

st.markdown("---")

# -----------------------
# Visual 1: Top 10 injuries with highest performance drop (Bar)
# -----------------------
st.subheader("1) Top 10 injuries with highest performance drop")

perf_df = fdf[[EXP_COLS["player"], EXP_COLS["team"], "avg_before_rating", "avg_after_rating", "performance_drop"]].dropna(subset=["performance_drop"])
if perf_df.empty:
    st.info("No performance drop data available for the selected filters.")
else:
    top_drops = perf_df.sort_values("performance_drop", ascending=False).head(10)
    fig1 = px.bar(top_drops, x=EXP_COLS["player"], y="performance_drop", color=EXP_COLS["team"],
                  title="Top 10 players with highest rating drop (before → after)")
    st.plotly_chart(fig1, use_container_width=True)

# -----------------------
# Visual 2: Player performance timeline (Line)
# -----------------------
st.subheader("2) Player performance timeline — Before vs After")

if selected_player != "All":
    player_rows = df[df[EXP_COLS["player"]] == selected_player].sort_values(EXP_COLS["injury_date"])
    if player_rows[["avg_before_rating", "avg_after_rating"]].dropna(how="all").empty:
        st.info("Not enough rating data for this player.")
    else:
        timeline = player_rows[[EXP_COLS["injury_date"], "avg_before_rating", "avg_after_rating"]].melt(id_vars=EXP_COLS["injury_date"],
                                                                                                       var_name="Phase", value_name="Rating")
        fig2 = px.line(timeline, x=EXP_COLS["injury_date"], y="Rating", color="Phase", markers=True,
                       title=f"{selected_player} — Avg rating before vs after injury")
        st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("Select a player from the sidebar to view their timeline (or use the filter).")

# -----------------------
# Visual 3: Heatmap (Injuries by month × team)
# -----------------------
st.subheader("3) Heatmap — Injury frequency by month and club")

heat = fdf.groupby([EXP_COLS["team"], "injury_month"]).size().reset_index(name="count")
if heat.empty:
    st.info("No injury month data for heatmap.")
else:
    # pivot and keep months in calendar order if present
    pivot = heat.pivot(index=EXP_COLS["team"], columns="injury_month", values="count").fillna(0)
    month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    present = [m for m in month_order if m in pivot.columns]
    if not present:
        # fallback: use pivot.columns order
        present = list(pivot.columns)
    fig3 = px.imshow(pivot[present].values if len(present)>0 else pivot.values,
                     x=present if present else list(pivot.columns),
                     y=pivot.index,
                     labels=dict(x="Month", y="Team", color="Injury count"),
                     title="Injury frequency (Team × Month)")
    st.plotly_chart(fig3, use_container_width=True)

# -----------------------
# Visual 4: Scatter — Age vs performance drop
# -----------------------
st.subheader("4) Age vs Performance Drop")

scatter_df = fdf.dropna(subset=[EXP_COLS["age"], "performance_drop"])
if scatter_df.empty:
    st.info("Insufficient age & rating data for scatter plot.")
else:
    fig4 = px.scatter(scatter_df, x=EXP_COLS["age"], y="performance_drop", color=EXP_COLS["team"], hover_data=[EXP_COLS["player"]],
                      title="Player age vs performance drop after injury")
    # add aggregated trendline (avg by age)
    age_agg = scatter_df.groupby(EXP_COLS["age"], as_index=False)["performance_drop"].mean().sort_values(EXP_COLS["age"])
    if not age_agg.empty:
        line = px.line(age_agg, x=EXP_COLS["age"], y="performance_drop")
        fig4.add_traces(line.data)
    st.plotly_chart(fig4, use_container_width=True)

# -----------------------
# Visual 5: Leaderboard — Comeback players by rating improvement
# -----------------------
st.subheader("5) Comeback leaderboard — Rating improvement after injury")

leader = fdf.copy()
leader["improvement"] = leader["avg_after_rating"] - leader["avg_before_rating"]
leaderboard = leader.sort_values("improvement", ascending=False)[[EXP_COLS["player"], EXP_COLS["team"], "improvement"]].dropna().head(10)

if leaderboard.empty:
    st.info("No comeback/improvement data available.")
else:
    st.table(leaderboard.rename(columns={EXP_COLS["player"]: "Player", EXP_COLS["team"]: "Team", "improvement": "Improvement"}).style.format({"Improvement": "{:.2f}"}))

# -----------------------
# Data preview & export
# -----------------------
st.markdown("---")
st.subheader("Filtered data preview (first 200 rows)")
st.dataframe(fdf.head(200))

if export_csv:
    buf = BytesIO()
    fdf.to_csv(buf, index=False)
    buf.seek(0)
    st.download_button("Download filtered CSV", data=buf, file_name="footlens_filtered.csv", mime="text/csv")

st.markdown("**Notes:** This app is matched to the column names present in your CSV. If you rename columns in the CSV, update the `EXP_COLS` mapping at the top of this file. Make sure `requirements.txt` (with `plotly`) is present next to this `app.py` when deploying to Streamlit Cloud.")
