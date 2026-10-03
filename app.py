"""Application entry point and top-level Streamlit navigation."""

import streamlit as st


def home() -> None:
    """Render the empty home page scaffold."""
    st.title("India Youth Career")


st.set_page_config(page_title="India Youth Career", page_icon="🎓", layout="wide")
pages = [
    st.Page(home, title="Home", default=True),
    st.Page("pages/analytics.py", title="Analytics"),
    st.Page("pages/career_predictor.py", title="Career Predictor"),
    st.Page("pages/ai_counselor.py", title="AI Career Counselor"),
]
st.navigation(pages).run()
