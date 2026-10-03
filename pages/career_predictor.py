"""Student profile form and ML career match explorer."""

from __future__ import annotations

import streamlit as st
import pandas as pd

from utils.preprocessing import DATA_PATH, SKILL_COLUMNS, cgpa_to_percentage, load_artifacts, predict_profile
from utils.career_profiles import CAREER_PROFILES, CAREER_PROFILE_NOTE
from utils.gemma import generate_roadmap, is_configured
from utils.roadmap_ui import render_roadmap

st.set_page_config(page_title="Find Your Career Match", page_icon="\U0001F9ED", layout="wide")

SKILL_LABELS = {
    "Programming_Skill_1_10": "Programming",
    "Communication_Skill_1_10": "Communication",
    "Analytical_Skill_1_10": "Analytical",
    "Creative_Design_Skill_1_10": "Creative / design",
}
PROFILE_KEYS = [
    "profile_gender", "profile_region", "profile_tier", "profile_stream",
    "profile_degree", "profile_interest", "profile_influence", "profile_score_type",
    "profile_age", "profile_score", *[f"profile_{column}" for column in SKILL_COLUMNS],
]
DISCLAIMER = "Career predictions are data-driven matches based on the model and information provided. They are not guarantees of career success."


@st.cache_resource
def cached_artifacts():
    """Load saved inference artifacts once per app process; never fit a model here."""
    return load_artifacts()


@st.cache_data
def dataset() -> pd.DataFrame:
    """Read the source table for real selectbox options and descriptive benchmarks."""
    return pd.read_csv(DATA_PATH)


def render_career_reference() -> None:
    st.subheader("Career profile reference")
    st.caption(CAREER_PROFILE_NOTE)
    tabs = st.tabs(list(CAREER_PROFILES))
    for tab, (career, details) in zip(tabs, CAREER_PROFILES.items()):
        with tab:
            st.markdown(f"### {details['icon']} {career}")
            st.write(details["overview"])
            left, right = st.columns(2)
            with left:
                st.markdown("**Skills to develop**")
                st.write("   ".join(details["skills"]))
                st.markdown("**Common entry roles**")
                st.write("   ".join(details["entry_roles"]))
                st.markdown("**Industries**")
                st.write("   ".join(details["industries"]))
            with right:
                st.markdown("**Technologies**")
                st.write("   ".join(details["technologies"]))
                st.markdown("**Learning areas**")
                st.write("   ".join(details["learning_areas"]))
                st.markdown("**A possible technology learning path**")
                st.write(" ? ".join(details["tech_path"]))
            st.markdown("**Project ideas**")
            for project in details["projects"]:
                st.markdown(f"- {project}")


def render_priorities(data: pd.DataFrame) -> None:
    st.subheader("Explore Career Priorities")
    st.info("Salary and job-satisfaction data are not currently available in this dataset.")
    focus = st.radio("Show student-reported focus", ["Salary Focus", "Satisfaction Focus"], horizontal=True, key="priority_focus")
    factor = "Salary/Package" if focus == "Salary Focus" else "Passion/Interest"
    share = data.assign(_cited=data["Influencing_Factor"].eq(factor)).groupby("Preferred_Career")["_cited"].mean().sort_values()
    chart = share.rename("Share of students").to_frame()
    st.bar_chart(chart, x_label="Share of students", y_label="Career", horizontal=True)
    st.caption("Method: share of students in each career group who self-reported this motivation as their influencing factor. This is self-reported motivation, not pay or satisfaction, and does not establish causation.")


def render_comparison(data: pd.DataFrame) -> None:
    st.subheader("Compare Careers")
    career = st.selectbox("Choose a career to compare", list(CAREER_PROFILES), key="compare_career")
    reference = CAREER_PROFILES[career]
    averages = data.loc[data["Preferred_Career"].eq(career), SKILL_COLUMNS].mean()
    comparison = pd.DataFrame({
        "Area": ["Required skills", "Programming average", "Communication average", "Analytical average", "Creative / design average", "Technologies", "Project ideas"],
        "Details": [
            ", ".join(reference["skills"]),
            f"{averages[SKILL_COLUMNS[0]]:.1f} / 10",
            f"{averages[SKILL_COLUMNS[1]]:.1f} / 10",
            f"{averages[SKILL_COLUMNS[2]]:.1f} / 10",
            f"{averages[SKILL_COLUMNS[3]]:.1f} / 10",
            ", ".join(reference["technologies"]),
            "; ".join(reference["projects"]),
        ],
    })
    st.table(comparison)
    st.caption("Skill averages are calculated from students in this synthetic dataset who selected this career.")


def render_model_details(metadata: dict) -> None:
    with st.expander("About this model & its limitations"):
        metrics = metadata.get("metrics", {})
        st.write(f"Saved model: **{metadata.get('model_name', 'Unknown')}**. It uses the student profile you provide to estimate probabilities across the nine career labels.")
        st.markdown("**Held-out evaluation (synthetic data)**")
        st.write(f"Accuracy: {metrics.get('accuracy', float('nan')):.1%}   Top-3 accuracy: {metrics.get('top_3_accuracy', float('nan')):.1%}   Macro F1: {metrics.get('macro_f1', float('nan')):.3f}")
        st.write(f"Majority-class baseline: {metrics.get('majority_class_baseline', float('nan')):.1%} ({metrics.get('majority_class', 'unknown')}).")
        st.markdown("**Per-class recall on held-out data**")
        recalls = metadata.get("per_class_recall", {})
        st.table(pd.DataFrame({"Career": list(recalls), "Recall": [f"{value:.1%}" for value in recalls.values()]}))
        st.caption(metadata.get("data_note", "The dataset is synthetic; metrics describe this synthetic dataset only."))
        st.write("The dataset is synthetic and may not represent real students or job markets. A probability is the model's relative estimate for the provided profile, not a guarantee or a measure of aptitude. The model can reflect dataset limitations and cannot account for every personal, educational, or local factor.")


def render_result(data: pd.DataFrame, metadata: dict) -> None:
    result = st.session_state.get("career_prediction")
    profile = st.session_state.get("career_profile")
    if result is None or profile is None:
        return

    career = result.index[0]
    confidence = float(result.iloc[0])
    st.markdown(
        f"""<div style="background:linear-gradient(115deg,#234e70,#4776a8,#7656a6);padding:1.5rem 1.8rem;border-radius:16px;color:white;margin:1rem 0 1.25rem 0">
        <div style="font-size:.78rem;letter-spacing:.12em;font-weight:700;opacity:.9">YOUR CAREER MATCH</div>
        <div style="font-size:1.05rem;margin-top:.65rem">Predicted Career</div>
        <div style="font-size:2rem;font-weight:750">{CAREER_PROFILES.get(career, {}).get('icon', '?')} {career}</div>
        <div style="font-size:1.05rem">{confidence:.1%} model probability</div></div>""",
        unsafe_allow_html=True,
    )
    metrics = metadata.get("metrics", {})
    st.caption(f"Held-out accuracy: {metrics.get('accuracy', float('nan')):.1%}   Held-out top-3 accuracy: {metrics.get('top_3_accuracy', float('nan')):.1%}. These metrics are measured on the synthetic dataset.")

    st.markdown("#### Why this match?")
    career_rows = data[data["Preferred_Career"].eq(career)]
    benchmark = career_rows[SKILL_COLUMNS].mean()
    bullets = []
    for column in SKILL_COLUMNS:
        actual = float(profile[column])
        average = float(benchmark[column])
        direction = "above" if actual >= average else "below"
        bullets.append(f"Your {SKILL_LABELS[column].lower()} rating is **{actual:.0f}/10**, {direction} the **{average:.1f}/10** average reported by students in this career group in the dataset.")
    for field, label in [("Primary_Interest", "primary interest"), ("Influencing_Factor", "motivation")]:
        chosen = profile[field]
        share = float(career_rows[field].eq(chosen).mean()) if len(career_rows) else 0
        bullets.append(f"**{chosen}** was your {label}; **{share:.0%}** of students labelled {career} in the dataset reported the same response. This is a descriptive comparison, not evidence that the response causes this career choice.")
    for bullet in bullets:
        st.markdown(f"- {bullet}")

    st.markdown("#### Other Possible Matches")
    for alternative, probability in result.iloc[1:3].items():
        st.markdown(f"**{alternative}**   {float(probability):.1%}")
        st.progress(min(max(float(probability), 0.0), 1.0))
    if len(result) > 1 and float(result.iloc[0] - result.iloc[1]) <= 0.10:
        st.caption("The top two model probabilities are within 10 percentage points, so the leading match is relatively close.")

    st.markdown("### Next step: build your roadmap")
    if st.button("Generate My 6-Month Career Roadmap", type="primary", key="predictor_generate_roadmap"):
        if not is_configured():
            st.error("Gemini API key is missing. Set GEMINI_API_KEY in the environment or Streamlit secrets to generate a roadmap.")
        else:
            with st.spinner("Building your career roadmap..."):
                try:
                    roadmap = generate_roadmap(profile, career, benchmark.to_dict())
                except RuntimeError as exc:
                    st.error(str(exc))
                else:
                    st.session_state["career_roadmap"] = roadmap
                    st.session_state["career_roadmap_career"] = career
                    st.success("Your roadmap has been generated.")

    saved_roadmap = st.session_state.get("career_roadmap")
    if saved_roadmap:
        render_roadmap(saved_roadmap, profile, benchmark.to_dict(), career)


def main() -> None:
    data = dataset()
    preprocessor, model, metadata = cached_artifacts()

    st.title("Find Your Career Match")
    st.markdown('> "Tell us about your interests, strengths and goals. Our machine-learning model will identify the career profile that most closely matches your responses."')

    button_a, button_b, _ = st.columns([1.4, 1.2, 3])
    with button_a:
        example_clicked = st.button("Fill with an example student", use_container_width=True)
    with button_b:
        another_clicked = st.button("Try Another Profile", use_container_width=True)
    if example_clicked:
        example = {
            "profile_age": 21,
            "profile_gender": "Female",
            "profile_region": "South",
            "profile_tier": "Tier 1",
            "profile_stream": "Science",
            "profile_degree": "B.Sc.",
            "profile_score_type": "Percentage",
            "profile_score": 80.0,
            "profile_interest": "Business & Finance",
            "profile_Programming_Skill_1_10": 8,
            "profile_Communication_Skill_1_10": 6,
            "profile_Analytical_Skill_1_10": 8,
            "profile_Creative_Design_Skill_1_10": 4,
            "profile_influence": "Salary/Package",
        }
        st.session_state.update(example)
        st.session_state.pop("career_prediction", None)
        st.session_state.pop("career_profile", None)
        st.session_state.pop("career_roadmap", None)
        st.session_state.pop("career_roadmap_career", None)
        st.rerun()
    if another_clicked:
        st.session_state.pop("career_prediction", None)
        st.session_state.pop("career_profile", None)
        st.session_state.pop("career_roadmap", None)
        st.session_state.pop("career_roadmap_career", None)
        st.rerun()

    options = {column: sorted(data[column].dropna().astype(str).unique().tolist()) for column in ["Gender", "Region", "City_Tier", "Stream_12th", "Current_Degree", "Primary_Interest", "Influencing_Factor"]}
    st.markdown("### Step 1: Tell us about yourself")
    with st.form("career_profile_form"):
        with st.container(border=True):
            st.markdown("#### About you")
            age = st.number_input("Age", min_value=13, max_value=100, value=None, step=1, key="profile_age", placeholder="Enter age")
            c1, c2, c3 = st.columns(3)
            with c1:
                gender = st.selectbox("Gender", options["Gender"], index=None, placeholder="Choose ", key="profile_gender")
            with c2:
                region = st.selectbox("Region", options["Region"], index=None, placeholder="Choose ", key="profile_region")
            with c3:
                tier = st.selectbox("City tier", options["City_Tier"], index=None, placeholder="Choose ", key="profile_tier")

        with st.container(border=True):
            st.markdown("#### Academics")
            c1, c2 = st.columns(2)
            with c1:
                stream = st.selectbox("12th stream", options["Stream_12th"], index=None, placeholder="Choose ", key="profile_stream")
            with c2:
                degree = st.selectbox("Current degree", options["Current_Degree"], index=None, placeholder="Choose ", key="profile_degree")
            score_type = st.radio("Score type", ["Percentage", "CGPA"], horizontal=True, key="profile_score_type")
            max_score = 100.0 if score_type == "Percentage" else 10.0
            score_label = "Academic score (%)" if score_type == "Percentage" else "Current CGPA (out of 10)"
            score = st.number_input(score_label, min_value=0.0, max_value=max_score, value=None, step=0.1, key="profile_score", placeholder="Enter your score")
            if score is not None and score_type == "CGPA":
                st.caption(f"Converted using the saved training-data fit: {cgpa_to_percentage(score, metadata['cgpa_fit']):.1f}%")

        with st.container(border=True):
            st.markdown("#### Interests, skills & motivation")
            interest = st.selectbox("Primary interest", options["Primary_Interest"], index=None, placeholder="Choose ", key="profile_interest")
            st.markdown("Rate each skill from 1 (still developing) to 10 (strong).")
            skill_values = {}
            for column, label in SKILL_LABELS.items():
                skill_values[column] = st.slider(label, min_value=1, max_value=10, key=f"profile_{column}")
            influence = st.radio("What matters most to you?", options["Influencing_Factor"], key="profile_influence")

        submitted = st.form_submit_button("Predict My Career", type="primary", use_container_width=True)

    if submitted:
        required = {
            "Age": age, "Gender": gender, "Region": region, "City tier": tier,
            "12th stream": stream, "Current degree": degree, "Academic score": score,
            "Primary interest": interest, "Motivation": influence,
        }
        missing = [label for label, value in required.items() if value is None or value == ""]
        if missing:
            st.error("Please complete the required fields: " + ", ".join(missing) + ".")
        else:
            score_values = {"Academic_Percentage": float(score)} if score_type == "Percentage" else {"Current_CGPA": float(score)}
            profile = {
                "Age": int(age), "Gender": gender, "Region": region, "City_Tier": tier,
                "Stream_12th": stream, "Current_Degree": degree, **score_values,
                "Primary_Interest": interest, "Influencing_Factor": influence, **skill_values,
            }
            st.session_state.pop("career_roadmap", None)
            st.session_state.pop("career_roadmap_career", None)
            st.session_state["career_prediction"] = predict_profile(profile, preprocessor, model, metadata)
            st.session_state["career_profile"] = profile

    render_result(data, metadata)
    st.markdown("---")
    st.markdown("### Step 2: Explore career paths")
    render_career_reference()
    st.markdown("---")
    render_priorities(data)
    st.markdown("---")
    render_comparison(data)
    render_model_details(metadata)
    st.warning(DISCLAIMER)


main()
