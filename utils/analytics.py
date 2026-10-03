"""Reusable descriptive analytics for the synthetic youth career dataset."""

from __future__ import annotations

import pandas as pd

from utils.preprocessing import SKILL_COLUMNS

CAREER_COLUMN = "Preferred_Career"
MOTIVATION_COLUMN = "Influencing_Factor"
NOT_REPORTED = "Not reported"


def prepare_analytics_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with missing motivation responses labelled consistently."""
    prepared = df.copy()
    if MOTIVATION_COLUMN in prepared.columns:
        prepared[MOTIVATION_COLUMN] = prepared[MOTIVATION_COLUMN].fillna(NOT_REPORTED)
    return prepared


def career_skill_benchmarks(df: pd.DataFrame) -> pd.DataFrame:
    """Mean of each available skill column per career, indexed by career label."""
    if CAREER_COLUMN not in df.columns:
        raise KeyError(f"Required column {CAREER_COLUMN!r} is missing.")
    columns = [column for column in SKILL_COLUMNS if column in df.columns]
    if not columns:
        return pd.DataFrame(columns=SKILL_COLUMNS).rename_axis(CAREER_COLUMN)
    if df.empty:
        return pd.DataFrame(columns=columns, index=pd.Index([], name=CAREER_COLUMN), dtype=float)
    return df.groupby(CAREER_COLUMN, dropna=False)[columns].mean().sort_index()


def motivation_share_by_career(df: pd.DataFrame) -> pd.DataFrame:
    """Share of reported motivation responses within each career.

    Missing/"Not reported" answers are excluded from both the numerator and
    denominator. Rows are careers and columns are motivation labels.
    """
    if CAREER_COLUMN not in df.columns or MOTIVATION_COLUMN not in df.columns:
        missing = [column for column in (CAREER_COLUMN, MOTIVATION_COLUMN) if column not in df.columns]
        raise KeyError(f"Required column(s) missing: {', '.join(missing)}")
    data = prepare_analytics_data(df)
    data = data.loc[data[MOTIVATION_COLUMN].ne(NOT_REPORTED)]
    if data.empty:
        return pd.DataFrame(index=pd.Index([], name=CAREER_COLUMN))
    return pd.crosstab(data[CAREER_COLUMN], data[MOTIVATION_COLUMN], normalize="index").sort_index()


def build_analytics_figures(df: pd.DataFrame) -> dict[str, object]:
    """Create all dashboard Plotly figures from already-filtered rows.

    Each figure handles small and empty intermediate categories, including a
    filtered sample with a single student or only unreported motivations.
    """
    import plotly.express as px
    import plotly.graph_objects as go

    data = prepare_analytics_data(df)
    figures: dict[str, object] = {}

    if data.empty:
        return figures

    career_counts = data[CAREER_COLUMN].value_counts().rename_axis("Career").reset_index(name="Count")
    career_counts["Share"] = career_counts["Count"] / len(data)
    career_counts["Label"] = career_counts.apply(lambda row: f"{row['Count']} ({row['Share']:.1%})", axis=1)
    figures["careers"] = px.bar(
        career_counts.sort_values("Count"), x="Count", y="Career", orientation="h",
        text="Label", title="Preferred careers", labels={"Count": "Students", "Career": "Career"},
    ).update_traces(textposition="outside", cliponaxis=False)

    motivation = data.loc[data[MOTIVATION_COLUMN].ne(NOT_REPORTED), MOTIVATION_COLUMN].value_counts().rename_axis("Motivation").reset_index(name="Students")
    donut = go.Figure()
    if not motivation.empty:
        donut.add_trace(go.Pie(labels=motivation["Motivation"], values=motivation["Students"], hole=0.58, textinfo="label+percent"))
    else:
        donut.add_annotation(text="No reported motivation in this selection", x=0.5, y=0.5, showarrow=False)
    donut.update_layout(title="Reported motivations")
    figures["motivation"] = donut

    skills = career_skill_benchmarks(data)
    figures["skills"] = px.imshow(
        skills, text_auto=".1f", aspect="auto", color_continuous_scale="Blues",
        labels={"x": "Skill", "y": "Preferred career", "color": "Mean score"},
        title="Mean skill ratings by career",
    )

    stream_career = pd.crosstab(data["Stream_12th"], data[CAREER_COLUMN], normalize="index")
    figures["stream_career"] = px.bar(
        stream_career, x=stream_career.index, y=stream_career.columns, barmode="relative",
        labels={"x": "12th stream", "value": "Share within stream", "variable": "Career"},
        title="Career mix within 12th stream",
    ).update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 1])

    tier_career = pd.crosstab(data["City_Tier"], data[CAREER_COLUMN], normalize="index")
    figures["tier_career"] = px.bar(
        tier_career, x=tier_career.index, y=tier_career.columns, barmode="group",
        labels={"x": "City tier", "value": "Share within tier", "variable": "Career"},
        title="Career mix by city tier",
    ).update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 1])

    interest_counts = data.groupby(["Primary_Interest", CAREER_COLUMN], observed=True).size().reset_index(name="Students")
    sources = sorted(interest_counts["Primary_Interest"].dropna().unique().tolist())
    targets = sorted(interest_counts[CAREER_COLUMN].dropna().unique().tolist())
    labels = [*sources, *targets]
    source_index = {label: index for index, label in enumerate(sources)}
    target_index = {label: len(sources) + index for index, label in enumerate(targets)}
    figures["interest_career"] = go.Figure(data=[go.Sankey(
        node={"label": labels, "pad": 14, "thickness": 18},
        link={"source": interest_counts["Primary_Interest"].map(source_index),
              "target": interest_counts[CAREER_COLUMN].map(target_index),
              "value": interest_counts["Students"]},
    )])
    figures["interest_career"].update_layout(title="Primary interest to preferred career")

    region_career = pd.crosstab(data["Region"], data[CAREER_COLUMN], normalize="index")
    figures["region_career"] = px.imshow(
        region_career, text_auto=".0%", aspect="auto", color_continuous_scale="Teal",
        zmin=0, zmax=1,
        labels={"x": "Preferred career", "y": "Region", "color": "Share within region"},
        title="Career share within each region",
    )
    return figures
