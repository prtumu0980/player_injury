# New logic: If no player selected → show all players
if selected_player == "All":
    timeline_df = df[[EXP_COLS["player"], EXP_COLS["injury_date"], "avg_before_rating", "avg_after_rating"]].dropna(subset=["avg_before_rating", "avg_after_rating"])
    
    if timeline_df.empty:
        st.info("No rating data available for any players.")
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
            color=EXP_COLS["player"],   # Different players = different colors
            line_group=EXP_COLS["player"],
            hover_data=[EXP_COLS["player"]],
            title="All Players — Before vs After Injury Timeline"
        )
        st.plotly_chart(fig2, use_container_width=True)

else:
    # Single player timeline
    player_rows = df[df[EXP_COLS["player"]] == selected_player].sort_values(EXP_COLS["injury_date"])
    if player_rows[["avg_before_rating", "avg_after_rating"]].dropna(how="all").empty:
        st.info("Not enough rating data for this player.")
    else:
        melted = player_rows.melt(
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
            title=f"{selected_player} — Before vs After Injury"
        )
        st.plotly_chart(fig2, use_container_width=True)
