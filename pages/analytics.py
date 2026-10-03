"""Interactive, descriptive overview of India's synthetic youth career data."""

from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from utils.analytics import (
    NOT_REPORTED,
    build_analytics_figures,
    prepare_analytics_data,
)
from utils.preprocessing import DATA_PATH, SKILL_COLUMNS

st.set_page_config(page_title="India's Youth Career Landscape", page_icon="\U0001F4CA", layout="wide")

FILTER_COLUMNS = [
    ("Gender", "Gender"),
    ("Region", "Region"),
    ("City_Tier", "City Tier"),
    ("Stream_12th", "12th Stream"),
    ("Current_Degree", "Current Degree"),
    ("Primary_Interest", "Primary Interest"),
    ("Influencing_Factor", "Influencing Factor"),
    ("Job_Preference", "Job Preference"),
    ("Preferred_Career", "Preferred Career"),
]


@st.cache_data
def load_data() -> pd.DataFrame:
    return prepare_analytics_data(pd.read_csv(DATA_PATH))


def reset_filters(age_bounds: tuple[int, int], options: dict[str, list[str]]) -> None:
    st.session_state["analytics_age"] = age_bounds
    for column, _ in FILTER_COLUMNS:
        st.session_state[f"analytics_{column}"] = options[column]


def _mode(series: pd.Series, fallback: str = "Not reported") -> str:
    modes = series.dropna().mode()
    return str(modes.iloc[0]) if not modes.empty else fallback


def main() -> None:
    data = load_data()
    total_students = len(data)
    age_min = int(data["Age"].min())
    age_max = int(data["Age"].max())
    age_bounds = (age_min, age_max)
    filter_options = {
        column: sorted(data[column].dropna().astype(str).unique().tolist())
        for column, _ in FILTER_COLUMNS
    }

    st.title("India's Youth Career Landscape")
    st.caption("Explore student-reported career preferences in the supplied synthetic dataset.")

    st.sidebar.header("Filters")
    if "analytics_age" not in st.session_state:
        st.session_state["analytics_age"] = age_bounds
    for column, _ in FILTER_COLUMNS:
        key = f"analytics_{column}"
        if key not in st.session_state:
            st.session_state[key] = filter_options[column]
    age_range = st.sidebar.slider(
        "Age", min_value=age_min, max_value=age_max, key="analytics_age",
    )
    selected_filters: dict[str, list[str]] = {}
    for column, label in FILTER_COLUMNS:
        selected_filters[column] = st.sidebar.multiselect(
            label, options=filter_options[column], key=f"analytics_{column}",
        )
    st.sidebar.button(
        "Reset", use_container_width=True, key="analytics_reset",
        on_click=reset_filters, args=(age_bounds, filter_options),
    )

    filtered = data.loc[data["Age"].between(age_range[0], age_range[1])].copy()
    for column, _ in FILTER_COLUMNS:
        choices = selected_filters[column]
        # An empty multiselect is treated as no restriction; Reset restores all values.
        if choices:
            filtered = filtered.loc[filtered[column].astype(str).isin(choices)]

    st.markdown(f"**Showing {len(filtered):,} of {total_students:,} students**")
    if filtered.empty:
        st.warning("No students match these filters. Adjust one or more filters to see the charts.")
        st.stop()

    most_preferred = _mode(filtered["Preferred_Career"])
    career_share = float(filtered["Preferred_Career"].eq(most_preferred).mean())
    reported_motivation = filtered.loc[filtered["Influencing_Factor"].ne(NOT_REPORTED), "Influencing_Factor"]
    most_common_motivation = _mode(reported_motivation)
    most_common_interest = _mode(filtered["Primary_Interest"])
    average_academic = filtered["Academic_Percentage"].mean()
    average_skill = filtered[SKILL_COLUMNS].mean(axis=1).mean()

    st.subheader("At a glance")
    kpi_rows = [st.columns(3), st.columns(3)]
    kpis = [
        ("Students Analyzed", f"{len(filtered):,}"),
        ("Most Preferred Career", f"{most_preferred} ({career_share:.1%})"),
        ("Most Common Motivation", most_common_motivation),
        ("Most Common Interest", most_common_interest),
        ("Average Academic Percentage", f"{average_academic:.1f}%" if pd.notna(average_academic) else "Not reported"),
        ("Average Skill Score", f"{average_skill:.1f} / 10" if pd.notna(average_skill) else "Not reported"),
    ]
    for index, (label, value) in enumerate(kpis):
        with kpi_rows[index // 3][index % 3]:
            st.metric(label, value)

    st.subheader("Career patterns")
    figures = build_analytics_figures(filtered)
    first_left, first_right = st.columns(2)
    with first_left:
        st.plotly_chart(figures["careers"], width="stretch", key="chart_careers")
    with first_right:
        st.plotly_chart(figures["motivation"], width="stretch", key="chart_motivation")
        excluded = int(filtered["Influencing_Factor"].eq(NOT_REPORTED).sum())
        st.caption(f"Motivation chart excludes {excluded:,} students with missing or unreported motivation.")

    second_left, second_right = st.columns(2)
    with second_left:
        st.plotly_chart(figures["skills"], width="stretch", key="chart_skills")
    with second_right:
        st.plotly_chart(figures["stream_career"], width="stretch", key="chart_stream_career")

    third_left, third_right = st.columns(2)
    with third_left:
        st.plotly_chart(figures["tier_career"], width="stretch", key="chart_tier_career")
    with third_right:
        st.plotly_chart(figures["interest_career"], width="stretch", key="chart_interest_career")

    st.plotly_chart(figures["region_career"], width="stretch", key="chart_region_career")
    st.caption("Career, skill, stream, city tier, interest, and region patterns are descriptive associations in synthetic data; they do not establish causation and may not represent real students or job markets.")


main()
