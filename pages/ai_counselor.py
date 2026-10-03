"""Career roadmap and verified GitHub coaching in one counselor page."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd
import streamlit as st

from utils.gemma import (
    explain_issues,
    generate_roadmap,
    generate_search_plan,
    is_configured,
)
from utils.github_search import search_hacktoberfest_issues
from utils.preprocessing import DATA_PATH, SKILL_COLUMNS
from utils.roadmap_ui import render_roadmap

st.set_page_config(page_title="AI Career Counselor", page_icon="\U0001F9ED", layout="wide")

TRACKER_ITEMS = [
    "PR 1 - Week 1 (Oct 3-9): setup and a documentation fix",
    "PR 2 - Weeks 2-3 (Oct 10-23): a focused bug fix",
    "PR 3 - Weeks 2-3 (Oct 10-23): add or improve a test",
    "PR 4 - Week 4 (Oct 24-31): a larger issue, with buffer time",
]


@st.cache_data
def load_career_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def _show_search_plan(plan: dict[str, list[str]]) -> None:
    left, right = st.columns(2)
    with left:
        st.markdown("**Languages**")
        st.write(", ".join(plan.get("languages", [])) or "No language returned")
        st.markdown("**Topics and keywords**")
        st.write(", ".join(plan.get("topics", []) + plan.get("keywords", [])) or "No search terms returned")
    with right:
        st.markdown("**Issue categories**")
        st.write(", ".join(plan.get("labels", [])) or "No labels returned")
        st.markdown("**Contribution types matched to your ratings**")
        st.write(", ".join(plan.get("contribution_types", [])) or "No contribution types returned")


def _fallback_search_terms(plan: dict[str, list[str]]) -> None:
    st.markdown("**Search plan to use without links**")
    st.markdown("- **Keywords:** " + (", ".join(plan.get("keywords", [])) or "None returned"))
    st.markdown("- **Issue categories:** " + (", ".join(plan.get("labels", [])) or "None returned"))
    st.markdown("- **Contribution types:** " + (", ".join(plan.get("contribution_types", [])) or "None returned"))


def _render_issue_cards(
    issues: list[dict[str, Any]],
    explanations: list[dict[str, Any]],
    searched_at: str,
) -> None:
    try:
        timestamp = datetime.fromisoformat(searched_at.replace("Z", "+00:00")).astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    except (ValueError, AttributeError):
        timestamp = searched_at
    st.caption(f"Found via GitHub search on {timestamp}")
    explanations_by_index = {item["id"]: item for item in explanations}
    for index, issue in enumerate(issues):
        with st.container(border=True):
            st.markdown(f"#### {issue['title']}")
            st.caption(f"{issue['repo_full_name']} | Updated {issue['updated_at']} | {issue['comments']} comments")
            labels = issue.get("labels", [])
            if labels:
                st.write("Labels: " + ", ".join(labels))
            explanation = explanations_by_index.get(index)
            if explanation:
                st.write(explanation["why_this_fits"])
                st.caption(f"Gemma's difficulty estimate: {explanation['difficulty']}")
            st.link_button("Open verified GitHub issue", issue["html_url"], width="stretch")


def _render_hacktoberfest_tab(
    profile: dict[str, Any] | None,
    prediction: Any | None,
    roadmap: dict[str, Any] | None,
) -> None:
    st.subheader("Hacktoberfest Mode")
    st.markdown("Career + skill gaps -> Gemma search plan -> live GitHub verification -> Gemma explanations for verified issues.")
    st.info("Check the current rules and register at hacktoberfest.com")

    career = str(prediction.index[0]) if prediction is not None else None
    if profile is not None and career and roadmap:
        skill_gaps = roadmap.get("skill_gaps", [])
        if st.button("Build My GitHub Search Plan", type="primary", key="hack_build_plan"):
            if not is_configured():
                st.error("Gemini API key is missing. Set GEMINI_API_KEY in the environment or Streamlit secrets to build a search plan.")
            else:
                with st.spinner("Matching contribution types to your profile..."):
                    try:
                        plan = generate_search_plan(profile, career, skill_gaps)
                    except RuntimeError as exc:
                        st.error(str(exc))
                    else:
                        st.session_state["hacktoberfest_search_plan"] = plan
                        st.session_state.pop("hacktoberfest_search_result", None)
                        st.session_state.pop("hacktoberfest_explanations", None)
                        st.success("Your search plan is ready.")

        plan = st.session_state.get("hacktoberfest_search_plan")
        if plan:
            _show_search_plan(plan)
            if st.button("Search GitHub for verified issues", key="hack_search_issues"):
                with st.spinner("Checking current repository and issue data on GitHub..."):
                    result = search_hacktoberfest_issues(plan.get("languages", []))
                st.session_state["hacktoberfest_search_result"] = result
                st.session_state.pop("hacktoberfest_explanations", None)

            result = st.session_state.get("hacktoberfest_search_result")
            if result is not None:
                issues = result.get("issues", [])
                if result.get("message"):
                    st.info(result["message"])
                if issues:
                    if st.button("Explain these verified issues with Gemma", key="hack_explain_issues"):
                        if not is_configured():
                            st.error("Gemini API key is missing. Set GEMINI_API_KEY in the environment or Streamlit secrets to explain issues.")
                        else:
                            with st.spinner("Explaining only the verified GitHub results..."):
                                try:
                                    explanations = explain_issues(profile, career, issues)
                                except RuntimeError as exc:
                                    st.error(str(exc))
                                else:
                                    st.session_state["hacktoberfest_explanations"] = explanations
                    _render_issue_cards(
                        issues,
                        st.session_state.get("hacktoberfest_explanations", []),
                        result.get("searched_at", ""),
                    )
                else:
                    if result.get("searched_at"):
                        st.caption(f"GitHub search checked on {result['searched_at']}.")
                    _fallback_search_terms(plan)
    elif profile is None or career is None:
        st.info("Complete your student profile and ML career prediction first. Hacktoberfest Mode keeps the model's career match fixed.")
    else:
        st.info("Generate your six-month career roadmap first. Hacktoberfest Mode uses its skill-gap section to shape your search plan.")

    st.markdown("### PR Readiness Checklist")
    st.warning("Before opening a PR, read the repository's CONTRIBUTING.md and any project AI-use policy. Spammy or AI-generated PRs can make a contributor ineligible. Gemma coaches you; it never writes PRs.")
    st.checkbox("I read this repository's CONTRIBUTING.md", key="hacktoberfest_ready_contributing")
    st.checkbox("I read the project's AI-use policy", key="hacktoberfest_ready_ai_policy")
    st.checkbox("My change is focused, tested, and ready for maintainer review", key="hacktoberfest_ready_review")

    st.markdown("### Four-PR Personal Tracker (Oct 3-31)")
    for index, label in enumerate(TRACKER_ITEMS, 1):
        st.checkbox(label, key=f"hacktoberfest_pr_{index}")
    completed = sum(bool(st.session_state.get(f"hacktoberfest_pr_{index}")) for index in range(1, 5))
    st.progress(completed / 4, text=f"Personal tracker: {completed} of 4 PR milestones checked (Oct 3-31).")


def main() -> None:
    st.title("AI Career Counselor")
    st.markdown("Turn your ML career match into a practical six-month action plan.")
    roadmap_tab, hacktoberfest_tab = st.tabs(["AI Career Roadmap", "Hacktoberfest Mode"])

    profile = st.session_state.get("career_profile")
    prediction = st.session_state.get("career_prediction")
    roadmap = st.session_state.get("career_roadmap")

    with roadmap_tab:
        if profile is None or prediction is None:
            st.info("Complete your student profile and predict a career match in Career Predictor first. The counselor builds a roadmap around that ML prediction.")
            st.info("### From Data -> Prediction -> Action\nStudent profile data informs the ML prediction; the counselor then creates practical next steps without replacing that prediction.")
        else:
            predicted_career = str(prediction.index[0])
            st.subheader(f"ML-predicted career: {predicted_career}")
            career_data = load_career_data()
            career_averages = (
                career_data.loc[career_data["Preferred_Career"].eq(predicted_career), SKILL_COLUMNS]
                .mean()
                .to_dict()
            )
            button_label = "Regenerate Roadmap" if roadmap else "Generate My 6-Month Career Roadmap"
            if st.button(button_label, type="primary", key="counselor_generate"):
                if not is_configured():
                    st.error("Gemini API key is missing. Set GEMINI_API_KEY in the environment or Streamlit secrets to generate a roadmap.")
                else:
                    with st.spinner("Building your career roadmap..."):
                        try:
                            roadmap = generate_roadmap(profile, predicted_career, career_averages)
                        except RuntimeError as exc:
                            st.error(str(exc))
                        else:
                            st.session_state["career_roadmap"] = roadmap
                            st.session_state["career_roadmap_career"] = predicted_career
                            st.success("Your roadmap has been generated.")
            roadmap = st.session_state.get("career_roadmap")
            if roadmap:
                render_roadmap(roadmap, profile, career_averages, predicted_career)
            st.warning("Career predictions are data-driven matches based on the model and information provided. They are not guarantees of career success.")

    with hacktoberfest_tab:
        _render_hacktoberfest_tab(profile, prediction, roadmap)


main()
