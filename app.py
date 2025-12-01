# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO

st.set_page_config(page_title="FootLens Lite – Injuries & Impact", layout="wide")

# -------------------
# USER CONFIG: map CSV columns here
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

def find_col(df, candidates):
    """Find and return the first matching column from candidates."""
    for c in candidates:
        if c in df.columns:
            return c
    return None

@st.cache_data
def load_and_clean(path="player_injuries_impact.csv"):
    df = pd.read_csv(path)

    # Map canonical names
    canonical = {}
    for key, poss in COLMAP.items():
        canonical[key] = find_col(df, poss)

    # Create canonical columns
    for k, col in canonical.items():
        if col is not None:
            df[k] = df[col]
        else:
            df[k] = np.nan

    # Parse dates
    for dcol in ["match_date", "injury_start", "injury_end", "injury_date"]:
        df[dcol] = pd.to_datetime(df[dcol], errors="coerce")

    # Numeric cleaning
    for r in ["before_rating", "after_rating", "age"]:
        df[r] = pd.to_numeric(df[r], errors="coerce")

    # Performance drop
    df["performance_drop"] = df["before_rating"] - df["after_rating"]

    # Phase label
    def phase_label(row):
        s, e, m = row["injury_start"], row["injury_end"], row["match_date"]
        if pd.isna(s) or pd.isna(e) or pd.isna(m):
            return "Unknown / Not enough data"
        if m < s:
            return "Before injury"
        if s <= m <= e:
            return "During absence"
        return "After return"

    df["phase"] = df.apply(phase_label, axis=1)

    # Injury month
    df["injury_month"] = df["injury_date"].dt.month_name().str[:3] if df["injury_date"].notna().any() else np.nan

    return df

# Load data
try:
    df = load_and_clean()
except Exception as e:
    st.error(f"Error loading CSV file: {e}")
    st.stop()

# -------------------
# SIDEBAR FILTERS
# -------------------
st.sidebar.title("Filters & Controls")

teams = ["All"] + sorted(df["team"].dropna().unique().tolist())
players = ["All"] + sorted(df["player"].dropna().unique().tolist())

selected_team = st.sidebar.selectbox("Team", teams)
selected_player = st.sidebar.selectbox("Player (highlight)", players)

min_date, max_date = df["match_date"].min(), df["match_date"].max()
date_range = st.sidebar.date_input(
    "Match date range",
    value=(min_date, max_date) if not pd.isna(min_date) else None
)

significant_only = st.sidebar.checkbox("Only show significant drops (>= 2.0)", False)
allow_download = st.sidebar.checkbox("Enable CSV download", True)

# -------------------
# APPLY FILTERS
# -------------------
fdf = df.copy()

if selected_team != "All":
    fdf = fdf[fdf["team"] == selected_team]

if selected_player != "All":
    fdf = fdf[fdf["player"] == selected_player]

if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    fdf = fdf[(fdf["match_date"] >= start) & (fdf["match_date"] <= end)]

if significant_only:
    fdf = fdf[fdf["performance_drop"] >= 2.0]

# -------------------
# TOP KPI ROW
# -------------------
st.title("⚽ FootLens Lite — Injuries & Performance Impact")

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.metric("Filtered Records", len(fdf))

with k2:
    st.metric("Injury Records", int(fdf["injury_date"].notna().sum()))

with k3:
    avg_drop = fdf["performance_drop"].mean()
    st.metric("Avg Performance Drop", f"{avg_drop:.2f}" if pd.notna(avg_drop) else "N/A")

with k4:
    before = fdf["before_rating"].mean()
    after = fdf["after_rating"].mean()
    st.metric("Avg Before → After", f"{before:.2f} → {after:.2f}" if pd.notna(before) and pd.notna(after) else "N/A")

st.markdown("---")

# -------------------
# VISUAL LAYOUT
# -------------------
left, right = st.columns([3, 1])

# LEFT SIDE — BIG CHARTS
with left:
    # Team drop chart
    st.subheader("Average Performance Drop by Team")
    team_drop = fdf.groupby("team", as_index=False)["performance_drop"].mean().dropna()
    if not team_drop.empty:
        fig1 = px.bar(team_drop, x="team", y="performance_drop", title="Avg Performance Drop per Team")
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.info("No team-level rating data found.")

    # Player timeline chart
    st.subheader("Player Timeline — Before vs After")
    if selected_player != "All":
        pdf = df[df["player"] == selected_player].sort_values("match_date")
        if not pdf.empty:
            melted = pdf.melt(id_vars="match_date", value_vars=["before_rating", "after_rating"], var_name="Phase", value_name="Rating")
            fig2 = px.line(melted, x="match_date", y="Rating", color="Phase", title=f"Timeline: {selected_player}")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No rating data for this player.")
    else:
        st.info("Select a player to view timeline.")

    # Top drops & improvements
    st.subheader("Top Drops & Top Comebacks")

    drops = fdf.sort_values("performance_drop", ascending=False).head(10)
    improve = fdf.sort_values("performance_drop", ascending=True).head(10)

    st.write("### Biggest Drops")
    st.table(drops[["player", "team", "performance_drop"]])

    st.write("### Biggest Improvements")
    st.table(improve[["player", "team", "performance_drop"]])

# RIGHT SIDE — HEATMAP & SCATTER
with right:
    st.subheader("Injuries Heatmap (Team × Month)")
    heat = fdf.groupby(["team", "injury_month"]).size().reset_index(name="count")
    if not heat.empty:
        pivot = heat.pivot(index="team", columns="injury_month", values="count").fillna(0)
        fig3 = px.imshow(
            pivot.values,
            x=pivot.columns,
            y=pivot.index,
            color_continuous_scale="Reds",
            labels={"x": "Month", "y": "Team", "color": "Count"},
            title="Injury Count by Month"
        )
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("No injury-month data available.")

    st.subheader("Age vs Performance Drop")
    scatter_df = fdf.dropna(subset=["age", "performance_drop"])
    if not scatter_df.empty:
        fig4 = px.scatter(
            scatter_df,
            x="age",
            y="performance_drop",
            color="team",
            hover_data=["player"],
            title="Age vs Performance Drop"
        )
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("No age or drop data.")

# -------------------
# DATA TABLE + DOWNLOAD
# -------------------
st.markdown("---")
st.subheader("Filtered Data Preview")
st.dataframe(fdf.head(200))

if allow_download:
    buf = BytesIO()
    fdf.to_csv(buf, index=False)
    buf.seek(0)
    st.download_button("Download Filtered CSV", data=buf, file_name="footlens_filtered.csv", mime="text/csv")
