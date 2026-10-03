"""Offline tests for verified Hacktoberfest search and counselor UI."""

from __future__ import annotations

import json
from types import SimpleNamespace
from pathlib import Path

import pytest
import requests
import pandas as pd


from utils.gemma import explain_issues, generate_roadmap, generate_search_plan
from utils.github_search import SEARCH_ISSUES, SEARCH_REPOSITORIES, search_hacktoberfest_issues


PROFILE = {
    "Age": 20,
    "Stream_12th": "Science",
    "Current_Degree": "B.Sc.",
    "Programming_Skill_1_10": 8,
    "Communication_Skill_1_10": 4,
    "Analytical_Skill_1_10": 7,
    "Creative_Design_Skill_1_10": 8,
    "Primary_Interest": "Technology",
    "Influencing_Factor": "Passion/Interest",
}


class FakeResponse:
    def __init__(self, payload=None, status_code=200):
        self.payload = payload or {"items": []}
        self.status_code = status_code

    def json(self):
        return self.payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.responses.pop(0)


class FakeModels:
    def __init__(self, text):
        self.text = text
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(text=self.text)


def test_github_rate_limit_falls_back_without_links():
    session = FakeSession([FakeResponse(status_code=429)])
    result = search_hacktoberfest_issues(["Python"], token="test-token", session=session)
    assert result["issues"] == []
    assert "rate-limited" in result["message"]
    assert "No issue links" in result["message"]
    assert session.calls[0][0] == SEARCH_REPOSITORIES
    assert session.calls[0][1]["headers"]["Authorization"] == "Bearer test-token"


@pytest.mark.parametrize("status", [403, 422, 429])
def test_github_blocked_statuses_fall_back_without_links(status):
    session = FakeSession([FakeResponse(status_code=status)])
    result = search_hacktoberfest_issues(["Python"], token="", session=session)
    assert result["issues"] == []
    assert "No issue links" in result["message"]


def test_github_network_error_falls_back_without_links():
    class BrokenSession:
        def get(self, *_args, **_kwargs):
            raise requests.Timeout("mock timeout")

    result = search_hacktoberfest_issues(["Python"], token="", session=BrokenSession())
    assert result["issues"] == []
    assert "could not connect" in result["message"]


def test_empty_github_results_have_no_issue_links():
    session = FakeSession([FakeResponse({"items": []})])
    result = search_hacktoberfest_issues(["Python"], token="", session=session)
    assert result["issues"] == []
    assert "No matching active" in result["message"]
    assert len(session.calls) == 1


def test_two_step_search_returns_api_issue_fields_and_tries_labels():
    repository = {"full_name": "verified/project", "updated_at": "2026-10-01T00:00:00Z"}
    issue = {
        "html_url": "https://github.com/verified/project/issues/7",
        "title": "Improve input validation",
        "labels": [{"name": "good first issue"}],
        "comments": 2,
        "updated_at": "2026-10-02T00:00:00Z",
    }
    session = FakeSession([
        FakeResponse({"items": [repository]}),
        FakeResponse({"items": [issue]}),
        FakeResponse({"items": [issue]}),
        FakeResponse({"items": []}),
    ])
    result = search_hacktoberfest_issues(["Python"], token="", session=session)
    assert len(result["issues"]) == 1
    assert result["issues"][0] == {
        "repo_full_name": "verified/project",
        "html_url": issue["html_url"],
        "title": issue["title"],
        "labels": ["good first issue"],
        "comments": 2,
        "updated_at": issue["updated_at"],
    }
    assert [call[0] for call in session.calls] == [SEARCH_REPOSITORIES] + [SEARCH_ISSUES] * 3
    assert all("topic:" not in call[1]["params"]["q"] for call in session.calls[1:])
    assert 'label:"help wanted"' in session.calls[2][1]["params"]["q"]


def test_gemma_plan_maps_contribution_types_to_student_ratings():
    models = FakeModels(json.dumps({
        "languages": ["Python"], "topics": ["data"], "labels": ["good first issue"],
        "keywords": ["beginner"], "contribution_types": ["documentation"],
    }))
    plan = generate_search_plan(PROFILE, "Data Scientist", ["communication"], client=SimpleNamespace(models=models))
    assert "bug fixes" in plan["contribution_types"]
    assert "UI and design issues" in plan["contribution_types"]
    assert "documentation improvements" in plan["contribution_types"]
    assert "tests and reproducible examples" in plan["contribution_types"]
    assert "ML-predicted career: Data Scientist" in models.calls[0]["contents"]


def test_explain_issues_drops_invented_indices():
    issues = [{
        "repo_full_name": "verified/project", "html_url": "https://github.com/verified/project/issues/3",
        "title": "Add validation", "labels": ["help wanted"], "comments": 0,
        "updated_at": "2026-10-01T00:00:00Z",
    }]
    models = FakeModels(json.dumps([
        {"id": 0, "why_this_fits": "A useful scoped task.", "difficulty": "Beginner"},
        {"id": 99, "why_this_fits": "Invented issue.", "difficulty": "Easy"},
    ]))
    explanation = explain_issues(PROFILE, "Data Scientist", issues, client=SimpleNamespace(models=models))
    assert explanation == [{"id": 0, "why_this_fits": "A useful scoped task.", "difficulty": "Beginner"}]
    prompt = models.calls[0]["contents"]
    assert "verified/project" in prompt and "Invented issue" not in prompt
    assert "https://github.com/verified/project/issues/3" not in prompt


def test_fenced_roadmap_response_strips_open_source_links():
    titles = ["Foundations", "Core Skills", "Applied Projects", "Advanced Skills", "Portfolio + Open Source", "Internship + Interview Preparation"]
    payload = {
        "why_this_career": "It matches the profile.", "strengths": ["Analysis"],
        "skill_gaps": [{"skill": "Communication", "note": "Practice concise summaries."}],
        "technologies": ["Python"], "learning_sequence": ["Python basics"],
        "months": [{"month": i, "title": title, "learning_goals": [], "technologies": [], "practical_tasks": [], "deliverable": "One artifact"} for i, title in enumerate(titles, 1)],
        "projects": [{"name": f"Project {i}", "difficulty": "Beginner", "technologies": ["Python"], "build": "Build it", "why_it_helps": "Practice."} for i in range(3)],
        "internship_prep": [], "interview_prep": [],
        "open_source": {
            "topics": ["Keep this", "Remove https://example.org/path", "Remove github.com/fake/repo"],
            "repository_types": ["documentation"], "issue_categories": ["tests"],
            "search_keywords": ["beginner python"], "contribution_ideas": ["Improve examples"],
        },
    }
    models = FakeModels("```json\n" + json.dumps(payload) + "\n```")
    roadmap = generate_roadmap(
        PROFILE,
        "Data Scientist",
        {"Programming_Skill_1_10": 6.4, "Communication_Skill_1_10": 6.1, "Analytical_Skill_1_10": 6.8, "Creative_Design_Skill_1_10": 5.4},
        client=SimpleNamespace(models=models),
    )
    assert roadmap["open_source"]["topics"] == ["Keep this"]
    assert "github.com" not in json.dumps(roadmap["open_source"]).lower()
    assert len(roadmap["months"]) == 6


def test_counselor_apptest_shows_hacktoberfest_tab_and_tracker():
    from streamlit.testing.v1 import AppTest

    page = Path(__file__).resolve().parents[1] / "pages" / "ai_counselor.py"
    app = AppTest.from_file(str(page)).run(timeout=30)
    assert not app.exception, [error.message for error in app.exception]
    assert [tab.label for tab in app.tabs] == ["AI Career Roadmap", "Hacktoberfest Mode"]
    assert len(app.checkbox) == 7
    assert any("Check the current rules and register at hacktoberfest.com" in item.value for item in app.info)
    assert "2025 rule" not in str(app.markdown).lower()
    app.checkbox[-4].check().run(timeout=30)
    assert app.session_state["hacktoberfest_pr_1"] is True


def test_hacktoberfest_apptest_uses_only_verified_issue_links(monkeypatch):
    from streamlit.testing.v1 import AppTest

    import utils.gemma as gemma
    import utils.github_search as github_search

    verified_issue = {
        "repo_full_name": "verified/project",
        "html_url": "https://github.com/verified/project/issues/7",
        "title": "Improve input validation",
        "labels": ["good first issue"],
        "comments": 0,
        "updated_at": "2026-10-02T00:00:00Z",
    }
    monkeypatch.setattr(
        github_search,
        "search_hacktoberfest_issues",
        lambda _languages: {"issues": [verified_issue], "message": "", "searched_at": "2026-10-03T10:00:00+00:00"},
    )
    monkeypatch.setattr(
        gemma,
        "explain_issues",
        lambda _profile, _career, issues: [{"id": 0, "why_this_fits": "A scoped validation task.", "difficulty": "Beginner"}] if issues == [verified_issue] else [],
    )
    monkeypatch.setattr(gemma, "is_configured", lambda: True)

    page = Path(__file__).resolve().parents[1] / "pages" / "ai_counselor.py"
    app = AppTest.from_file(str(page))
    app.session_state["career_profile"] = PROFILE
    app.session_state["career_prediction"] = pd.Series({"Data Scientist": 0.7})
    app.session_state["career_roadmap"] = {"skill_gaps": [{"skill": "Communication", "note": "Practice."}]}
    app.session_state["hacktoberfest_search_plan"] = {
        "languages": ["Python"], "topics": ["data"], "labels": ["good first issue"],
        "keywords": ["beginner"], "contribution_types": ["documentation improvements"],
    }
    app.run(timeout=30)
    assert not app.exception, [error.message for error in app.exception]
    app.button(key="hack_search_issues").click().run(timeout=30)
    app.button(key="hack_explain_issues").click().run(timeout=30)
    assert any("Found via GitHub search on" in item.value for item in app.caption)
    assert any(verified_issue["title"] in item.value for item in app.markdown)
    assert any(verified_issue["repo_full_name"] in item.value for item in app.caption)
    assert any("A scoped validation task." in item.value for item in app.markdown)
    assert not app.exception, [error.message for error in app.exception]
