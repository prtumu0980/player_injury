# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO

st.set_page_config(page_title="FootLens – Injuries & Performance", layout="wide")

# -----------------------
# Expected CSV Columns (matched exactly to your dataset)
# -----------------------
EXP_COLS = {
    "player": "Name",
    "team": "Team Name",
    "age": "Age",
    "injury_date": "Date of Injury",
    "return_date": "Date of return",
    "before_ratings": [
        "Match1_before_injury_Player_rating",
        "Match2_before_injury_Player_rating",
        "Match3_before_injury_Player_rating",
    ],
    "after_ratings": [
        "Match1_after_injury_Player_rating",
        "Match2_after_injury_Player_rating",
        "Match3_after_injury_Player_rating",
    ],
    "season": "Season"
}

# -----------------------
# Load CSV
# -----------------------
@st.cache_data
def load_df(path="player_injuries_impact.csv"):
    df = pd.read_csv(path, dtype=str)
    df = df.replace({"N.A.": np.nan, "NA": np.nan, "": np.nan})
    df.columns = [c.strip() for c in df.columns]
    return df

try:
    df_raw = load_df()
except Exception as e:
    st.error(f"Error loading CSV: {e}")
    st.stop()

# -----------------------
# Clean + Convert Types
# -----------------------
df = df_raw.copy()

# Convert dates
df[EXP_COLS["injury_date"]] = pd.to_datetime(df[EXP_COLS["injury_date"]], errors="coerce")
df[EXP_COLS["return_date"]] = pd.to_datetime(df[EXP_COLS["return_date"]], errors="coerce")

# Numeric age
df[EXP_COLS["age"]] = pd.to_numeric(df[EXP_COLS["age"]], errors="coerce")

# Convert rating columns to numeric
def make_numeric(df_, cols):
    found = []
    for c in cols:
        if c in df_.columns:
            df_[c] = pd.to_numeric(df_[c], errors="coerce")
            found.append(c)
    return found

before_cols = make_numeric(df, EXP_COLS["before_ratings"])
after_cols  = make_numeric(df, EXP_COLS["after_ratings"])

# Compute averages
df["avg_before_rating"] = df[before_cols].mean(axis=1, skipna=True)
df["avg_after_rating"]  = df[after_cols].mean(axis=1, skipna=True)

# Performance drop
df["performance_drop"] = df["avg_before_rating"] - df["avg_after_rating"]

# Injury month for heatmap
df["injury_month"] = df[EXP_COLS["injury_date"]].dt.month_name().str.slice(0, 3)

# -----------------------
# Sidebar Filters
# -----------------------
st.sidebar.header("Filters")

teams = sorted(df[EXP_COLS["team"]].dropna().unique().tolist())
players = sorted(df[EXP_COLS["player"]].dropna().unique().tolist())

selected_team = st.sidebar.selectbox("Select team", ["All"] + teams)
selected_player = st.sidebar.selectbox("Select player (optional)", ["All"] + players)

date_min = df[EXP_COLS["injury_date"]].min()
date_max = df[EXP_COLS["injury_date"]].max()

date_range = st.sidebar.date_input(
    "Injury date range",
    value=(date_min, date_max) if pd.notna(date_min) else None
)

significant_only = st.sidebar.checkbox("Only show significant drops (>=2.0)", False)
export_csv = st.sidebar.checkbox("Enable CSV download", True)

# -----------------------
# Apply Filters
# -----------------------
fdf = df.copy()

if selected_team != "All":
    fdf = fdf[fdf[EXP_COLS["team"]] == selected_team]

if selected_player != "All":
    fdf = fdf[fdf[EXP_COLS["player"]] == selected_player]

if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    fdf = fdf[(fdf[EXP_COLS["injury_date"]] >= start) & (fdf[EXP_COLS["injury_date"]] <= end)]

if significant_only:
    fdf = fdf[fdf["performance_drop"] >= 2.0]

# -----------------------
# KPIs
# -----------------------
st.title("⚽ FootLens – Player Injury Impact Dashboard")

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.metric("Filtered Records", len(fdf))

with k2:
    st.metric("Injury Records", int(fdf[EXP_COLS["injury_date"]].notna().sum()))

with k3:
    avg_drop = fdf["performance_drop"].mean()
    st.metric("Avg Drop", f"{avg_drop:.2f}" if pd.notna(avg_drop) else "N/A")

with k4:
    b = fdf["avg_before_rating"].mean()
    a = fdf["avg_after_rating"].mean()
    st.metric("Avg Before → After", f"{b:.2f} → {a:.2f}" if pd.notna(b) and pd.notna(a) else "N/A")

st.markdown("---")

# -----------------------
# Visual 1 – Top 10 performance drops
# -----------------------
st.subheader("1) Top 10 Performance Drops (Bar Chart)")

perf = fdf[[EXP_COLS["player"], EXP_COLS["team"], "performance_drop"]].dropna()
top10 = perf.sort_values("performance_drop", ascending=False).head(10)

if not top10.empty:
    fig1 = px.bar(top10, x=EXP_COLS["player"], y="performance_drop",
                  color=EXP_COLS["team"],
                  title="Top 10 Players with Biggest Drop")
    st.plotly_chart(fig1, use_container_width=True)
else:
    st.info("No data available.")

# -----------------------
# Visual 2 – Player Timeline (Before vs After)
# -----------------------
st.subheader("2) Player Performance Timeline — Before vs After")

# IF NO PLAYER SELECTED → SHOW ALL PLAYERS
if selected_player == "All":
    timeline_df = df[[EXP_COLS["player"], EXP_COLS["injury_date"],
                      "avg_before_rating", "avg_after_rating"]].dropna(
        subset=["avg_before_rating", "avg_after_rating"]
    )

    if timeline_df.empty:
        st.info("No timeline data available.")
    else:
        melted = timeline_df.melt(
            id_vars=[EXP_COLS["player"], EXP_COLS["injury_date"]],
            value_vars=["avg_before_rating", "avg_after_rating"],
            var_name="Phase",
            value_name="Rating"
        )

        fig2 = px.line(
            melted,
            x=EXP_COLS["injury_date"],
            y="Rating",
            color=EXP_COLS["player"],
            line_group=EXP_COLS["player"],
            title="All Players – Before vs After Injury Timeline",
            hover_data=[EXP_COLS["player"]]
        )
        st.plotly_chart(fig2, use_container_width=True)

# IF PLAYER SELECTED → SHOW ONLY THEIR TIMELINE
else:
    p_df = df[df[EXP_COLS["player"]] == selected_player].sort_values(EXP_COLS["injury_date"])

    if p_df[["avg_before_rating", "avg_after_rating"]].dropna(how="all").empty:
        st.info("Not enough rating data for this player.")
    else:
        melted = p_df.melt(
            id_vars=EXP_COLS["injury_date"],
            value_vars=["avg_before_rating", "avg_after_rating"],
            var_name="Phase",
            value_name="Rating"
        )

        fig2 = px.line(
            melted,
            x=EXP_COLS["injury_date"],
            y="Rating",
            color="Phase",
            markers=True,
            title=f"{selected_player} – Before vs After Injury"
        )
        st.plotly_chart(fig2, use_container_width=True)

# -----------------------
# Visual 3 – Heatmap
# -----------------------
st.subheader("3) Heatmap – Injury Frequency by Month & Club")

heat = fdf.groupby([EXP_COLS["team"], "injury_month"]).size().reset_index(name="count")

if not heat.empty:
    pivot = heat.pivot(index=EXP_COLS["team"], columns="injury_month", values="count").fillna(0)
    fig3 = px.imshow(
        pivot.values,
        x=pivot.columns,
        y=pivot.index,
        labels={"x": "Month", "y": "Team", "color": "Count"},
        title="Injury Frequency Heatmap"
    )
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.info("No heatmap data available.")

# -----------------------
# Visual 4 – Age vs Performance Drop
# -----------------------
st.subheader("4) Age vs Performance Drop (Scatter Plot)")

scatter = fdf.dropna(subset=[EXP_COLS["age"], "performance_drop"])

if not scatter.empty:
    fig4 = px.scatter(
        scatter,
        x=EXP_COLS["age"],
        y="performance_drop",
        color=EXP_COLS["team"],
        hover_data=[EXP_COLS["player"]],
        title="Age vs Performance Drop"
    )
    st.plotly_chart(fig4, use_container_width=True)
else:
    st.info("No age data available.")

# -----------------------
# Visual 5 – Comeback Leaderboard
# -----------------------
st.subheader("5) Comeback Leaderboard — Rating Improvement")

leader = fdf.copy()
leader["improvement"] = leader["avg_after_rating"] - leader["avg_before_rating"]
table = leader[[EXP_COLS["player"], EXP_COLS["team"], "improvement"]].dropna().sort_values("improvement", ascending=False).head(10)

st.table(table.style.format({"improvement": "{:.2f}"}))

# -----------------------
# Data + CSV Export
# -----------------------
st.subheader("Filtered Data Preview")
st.dataframe(fdf.head(200))

if export_csv:
    buf = BytesIO()
    fdf.to_csv(buf, index=False)
    buf.seek(0)
    st.download_button("Download Filtered CSV",
                       data=buf,
                       file_name="footlens_filtered.csv",
                       mime="text/csv")
