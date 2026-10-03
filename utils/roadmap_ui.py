"""Structured Streamlit components for a six-month career roadmap."""

from __future__ import annotations

from typing import Any

import streamlit as st

SKILL_LABELS = {
    "Programming_Skill_1_10": "Programming",
    "Communication_Skill_1_10": "Communication",
    "Analytical_Skill_1_10": "Analytical",
    "Creative_Design_Skill_1_10": "Creative / design",
}


def _items(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _show_list(value: Any) -> None:
    items = _items(value)
    if not items:
        st.caption("No details provided.")
        return
    for item in items:
        if isinstance(item, dict):
            st.markdown("- " + " - ".join(f"**{key.replace('_', ' ').title()}:** {val}" for key, val in item.items()))
        else:
            st.markdown(f"- {item}")


def render_roadmap(
    roadmap: dict[str, Any],
    student_profile: dict[str, Any] | None = None,
    career_averages: dict[str, Any] | None = None,
    predicted_career: str | None = None,
) -> None:
    """Render roadmap content in clear sections and compact cards."""
    st.markdown("## Your 6-Month Career Roadmap")
    if predicted_career:
        st.caption(f"Built around the ML-predicted career: **{predicted_career}**. The roadmap supports this match and does not replace it.")

    st.markdown("### Why This Career")
    st.info(roadmap.get("why_this_career", ""))

    left, right = st.columns(2)
    with left:
        st.markdown("### Strengths")
        _show_list(roadmap.get("strengths"))
    with right:
        st.markdown("### Skills You Need to Build")
        profile = student_profile or {}
        averages = career_averages or {}
        for column, label in SKILL_LABELS.items():
            rating = profile.get(column)
            average = averages.get(column)
            if rating is None:
                continue
            rating_float = min(10.0, max(0.0, float(rating)))
            filled = int(round(rating_float))
            bar = "\u2588" * filled + "\u2591" * (10 - filled)
            st.markdown(f"`{label} {bar} {rating_float:g}/10`")
            if average is not None:
                gap = rating_float - float(average)
                direction = "above" if gap >= 0 else "below"
                st.caption(f"Dataset average for {predicted_career or 'this career'}: {float(average):.1f}/10; your rating is {abs(gap):.1f} points {direction} that descriptive benchmark.")

        st.markdown("**Recommended gap areas**")
        _show_list(roadmap.get("skill_gaps"))

    st.markdown("### Technologies")
    technologies = [str(item) for item in _items(roadmap.get("technologies")) if str(item).strip()]
    st.markdown("  **->**  ".join(technologies) if technologies else "No technologies provided.")

    st.markdown("### Recommended Learning Sequence")
    _show_list(roadmap.get("learning_sequence"))

    st.markdown("### Six-Month Action Plan")
    months = _items(roadmap.get("months"))
    for row_start in range(0, len(months), 3):
        columns = st.columns(3)
        for column, month in zip(columns, months[row_start : row_start + 3]):
            with column:
                with st.container(border=True):
                    st.markdown(f"#### Month {month.get('month', row_start + 1)}: {month.get('title', 'Action plan')}")
                    st.markdown("**Learning goals**")
                    _show_list(month.get("learning_goals"))
                    st.markdown("**Technologies**")
                    st.write(" -> ".join(str(value) for value in _items(month.get("technologies"))) or "As listed in the learning plan")
                    st.markdown("**Practical tasks**")
                    _show_list(month.get("practical_tasks"))
                    st.markdown("**Deliverable**")
                    st.write(month.get("deliverable", "Not specified"))

    st.markdown("### Portfolio Projects")
    projects = _items(roadmap.get("projects"))
    for row_start in range(0, len(projects), 3):
        columns = st.columns(3)
        for column, project in zip(columns, projects[row_start : row_start + 3]):
            with column:
                with st.container(border=True):
                    if isinstance(project, dict):
                        st.markdown(f"#### {project.get('name', 'Project idea')}")
                        if project.get("difficulty"):
                            st.caption(f"Difficulty: {project['difficulty']}")
                        st.markdown("**Technologies**")
                        st.write(", ".join(str(value) for value in _items(project.get("technologies"))) or "To be selected")
                        st.markdown("**What to build**")
                        st.write(project.get("build", ""))
                        st.markdown("**Why it helps**")
                        st.write(project.get("why_it_helps", ""))
                    else:
                        st.markdown(f"#### {project}")

    internship_tab, interview_tab = st.tabs(["Internship Preparation", "Interview Preparation"])
    with internship_tab:
        st.markdown("### Internship Preparation")
        _show_list(roadmap.get("internship_prep"))
    with interview_tab:
        st.markdown("### Interview Preparation")
        _show_list(roadmap.get("interview_prep"))

    st.markdown("### Your Open-Source Path")
    open_source = roadmap.get("open_source", {})
    open_source_columns = st.columns(2)
    for column, key, label in zip(
        open_source_columns * 3,
        ["topics", "repository_types", "issue_categories", "search_keywords", "contribution_ideas"],
        ["Topics", "Repository types", "Issue categories", "Search keywords", "Contribution ideas"],
    ):
        with column:
            st.markdown(f"**{label}**")
            _show_list(open_source.get(key, []))

    st.markdown("---")
    st.info("### From Data -> Prediction -> Action\nThe supplied synthetic student data informs the ML career prediction. This roadmap turns that predicted profile into practical next steps; it does not replace the model's prediction.")
