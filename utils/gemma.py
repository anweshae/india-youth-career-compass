"""Google GenAI integration for personalized, ML-anchored career roadmaps."""

from __future__ import annotations

import json
import os
import re
from typing import Any

import streamlit as st

DEFAULT_MODEL = "gemma-4-26b-a4b-it"
SYSTEM_PROMPT = """You are an AI career counselor. The machine-learning model has identified a career profile that matches this student's responses. Do not override or replace the ML prediction. Instead, analyze the student's strengths and gaps and create a realistic personalized development roadmap. Your response must include: 1. Why this career matches the student's profile 2. Current strengths 3. Skill gaps 4. Technologies to learn 5. Recommended learning sequence 6. Portfolio projects 7. Internship preparation 8. Interview preparation 9. Open-source contribution opportunities 10. A six-month action plan. Do not guarantee employment or salary outcomes. Give practical, actionable and realistic recommendations.

Hard rules: never invent repository names, URLs or issue numbers; do not include salary figures; base the plan on the student's given 1-10 skill ratings; respond with ONE JSON object only."""
REQUIRED_KEYS = {
    "why_this_career", "strengths", "skill_gaps", "technologies",
    "learning_sequence", "months", "projects", "internship_prep",
    "interview_prep", "open_source",
}
MONTH_TITLES = [
    "Foundations", "Core Skills", "Applied Projects", "Advanced Skills",
    "Portfolio + Open Source", "Internship + Interview Preparation",
]


def _api_key() -> str | None:
    key = os.getenv("GEMINI_API_KEY")
    if key and key.strip():
        return key.strip()
    try:
        key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        key = None
    return str(key).strip() if key else None


def _model_name() -> str:
    model_name = os.getenv("GEMMA_MODEL")
    if not model_name:
        try:
            model_name = st.secrets.get("GEMMA_MODEL")
        except Exception:
            model_name = None
    return str(model_name).strip() if model_name and str(model_name).strip() else DEFAULT_MODEL

def is_configured() -> bool:
    """Return whether an API key is available from the environment or secrets."""
    return bool(_api_key())


def _student_prompt(
    student_profile: dict[str, Any],
    predicted_career: str,
    career_averages: dict[str, Any],
) -> str:
    skill_columns = {
        "Programming_Skill_1_10": "Programming",
        "Communication_Skill_1_10": "Communication",
        "Analytical_Skill_1_10": "Analytical",
        "Creative_Design_Skill_1_10": "Creative / design",
    }
    profile = {
        "age": student_profile.get("Age"),
        "12th_stream": student_profile.get("Stream_12th"),
        "current_degree": student_profile.get("Current_Degree"),
        "skills_1_to_10": {
            label: student_profile.get(column)
            for column, label in skill_columns.items()
        },
        "primary_interest": student_profile.get("Primary_Interest"),
        "motivation": student_profile.get("Influencing_Factor"),
    }
    benchmarks = {
        label: career_averages.get(column)
        for column, label in skill_columns.items()
    }
    schema = {
        "why_this_career": "string",
        "strengths": ["string"],
        "skill_gaps": [{"skill": "string", "note": "string"}],
        "technologies": ["string"],
        "learning_sequence": ["string"],
        "months": [
            {
                "month": 1,
                "title": "Foundations",
                "learning_goals": ["string"],
                "technologies": ["string"],
                "practical_tasks": ["string"],
                "deliverable": "string",
            }
        ],
        "projects": [
            {
                "name": "string",
                "difficulty": "string",
                "technologies": ["string"],
                "build": "string",
                "why_it_helps": "string",
            }
        ],
        "internship_prep": ["string"],
        "interview_prep": ["string"],
        "open_source": {
            "topics": ["string"],
            "repository_types": ["string"],
            "issue_categories": ["string"],
            "search_keywords": ["string"],
            "contribution_ideas": ["string"],
        },
    }
    month_titles = "; ".join(f"Month {i + 1}: {title}" for i, title in enumerate(MONTH_TITLES))
    return (
        "Create a personalized development plan from the following student data. "
        "Treat the ML Predicted Career as fixed: do not change or recommend replacing it. "
        "Skill averages are descriptive benchmarks from the synthetic dataset, not causal evidence. "
        "Do not claim the data represents real students. Do not invent a repository, URL, or issue number. "
        "Return exactly one JSON object matching this schema; no markdown fences or commentary. "
        "The months array must have exactly six entries with these titles in order: " + month_titles + ". "
        "Include 3 to 5 projects. Use these exact top-level keys and field shapes: "
        + json.dumps(schema, ensure_ascii=False)
        + "\n\nStudent profile: " + json.dumps(profile, ensure_ascii=False)
        + "\nML Predicted Career: " + predicted_career
        + "\nDataset average skills for this career (out of 10): "
        + json.dumps(benchmarks, ensure_ascii=False)
    )


def _remove_link_strings(value: Any) -> Any:
    """Remove any open-source string that contains a URL or github.com."""
    if isinstance(value, str):
        has_link = re.search(
            r"(?i)(github\.com|\bhttps?://|\bwww\.|\b[a-z0-9-]+(?:\.[a-z0-9-]+)*\.[a-z]{2,}(?:/\S*)?)",
            value,
        )
        return None if has_link else value
    if isinstance(value, list):
        return [cleaned for item in value if (cleaned := _remove_link_strings(item)) is not None]
    if isinstance(value, dict):
        cleaned_dict = {}
        for key, item in value.items():
            if isinstance(key, str) and re.search(r"(?i)(github\.com|\bhttps?://|\bwww\.|\b[a-z0-9-]+(?:\.[a-z0-9-]+)*\.[a-z]{2,}(?:/\S*)?)", key):
                continue
            cleaned = _remove_link_strings(item)
            if cleaned is not None:
                cleaned_dict[key] = cleaned
        return cleaned_dict
    return value


def _parse_response(response_text: str) -> dict[str, Any]:
    text = (response_text or "").strip()
    text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise RuntimeError("Gemma returned no JSON object. Please try generating the roadmap again.")
    try:
        payload = json.loads(text[start : end + 1])
    except (json.JSONDecodeError, TypeError) as exc:
        raise RuntimeError("Gemma returned invalid JSON. Please try generating the roadmap again.") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Gemma's response must be one JSON object. Please try again.")
    missing = REQUIRED_KEYS.difference(payload)
    if missing:
        raise RuntimeError("Gemma's JSON is missing required sections: " + ", ".join(sorted(missing)) + ". Please try again.")
    for key in ("strengths", "skill_gaps", "technologies", "learning_sequence", "projects", "internship_prep", "interview_prep"):
        if not isinstance(payload[key], list):
            raise RuntimeError(f"Gemma's {key.replace('_', ' ')} section has an invalid format. Please try again.")
    if not isinstance(payload["why_this_career"], str):
        raise RuntimeError("Gemma's career explanation has an invalid format. Please try again.")
    if not 3 <= len(payload["projects"]) <= 5 or any(not isinstance(project, dict) for project in payload["projects"]):
        raise RuntimeError("Gemma's response must contain three to five structured portfolio projects. Please try again.")
    for project in payload["projects"]:
        if not {"name", "difficulty", "technologies", "build", "why_it_helps"}.issubset(project):
            raise RuntimeError("A portfolio project is missing required details. Please try again.")
    if any(not isinstance(gap, dict) or not {"skill", "note"}.issubset(gap) for gap in payload["skill_gaps"]):
        raise RuntimeError("Gemma's skill-gap section has an invalid format. Please try again.")
    if not isinstance(payload["months"], list) or len(payload["months"]) != 6:
        raise RuntimeError("Gemma's roadmap must contain six monthly plan cards. Please try again.")
    if any(not isinstance(month, dict) for month in payload["months"]):
        raise RuntimeError("Gemma's monthly plan has an invalid format. Please try again.")
    for index, (month, title) in enumerate(zip(payload["months"], MONTH_TITLES), 1):
        if month.get("month") != index or month.get("title") != title:
            raise RuntimeError("Gemma's monthly plan is missing the required six-month sequence. Please try again.")
        if not {"learning_goals", "technologies", "practical_tasks", "deliverable"}.issubset(month):
            raise RuntimeError("A monthly plan card is missing required details. Please try again.")
    if not isinstance(payload["open_source"], dict):
        raise RuntimeError("Gemma's open-source section has an invalid format. Please try again.")
    required_open_source = {"topics", "repository_types", "issue_categories", "search_keywords", "contribution_ideas"}
    if not required_open_source.issubset(payload["open_source"]):
        raise RuntimeError("Gemma's open-source section is missing required details. Please try again.")
    if any(not isinstance(payload["open_source"][key], list) for key in required_open_source):
        raise RuntimeError("Gemma open-source details must be lists. Please try again.")
    payload["open_source"] = _remove_link_strings(payload["open_source"])
    return payload


def generate_roadmap(
    student_profile: dict[str, Any],
    predicted_career: str,
    career_averages: dict[str, Any],
    *,
    client: Any | None = None,
) -> dict[str, Any]:
    """Request and parse one structured roadmap; client injection supports tests."""
    if client is None:
        api_key = _api_key()
        if not api_key:
            raise RuntimeError("Gemini API key is missing. Set GEMINI_API_KEY in the environment or Streamlit secrets.")
        try:
            from google import genai

            client = genai.Client(api_key=api_key)
        except Exception as exc:
            raise RuntimeError(f"Could not initialize the Gemini client: {exc}") from exc

    prompt = _student_prompt(student_profile, predicted_career, career_averages)
    try:
        response = client.models.generate_content(
            model=_model_name(),
            contents=prompt,
            config={"system_instruction": SYSTEM_PROMPT, "response_mime_type": "application/json"},
        )
    except Exception as exc:
        raise RuntimeError(f"Gemma API request failed: {exc}") from exc

    response_text = getattr(response, "text", None)
    if not response_text:
        raise RuntimeError("Gemma returned an empty response. Please try generating the roadmap again.")
    return _parse_response(response_text)

SEARCH_PLAN_SYSTEM_PROMPT = """You are a career learning coach. Use the student's real 1-10 ratings and skill gaps to create a focused GitHub search plan. Do not invent repositories, issues, URLs, or contribution opportunities. Return exactly one JSON object with string arrays named languages, topics, labels, keywords, and contribution_types. The contribution types must fit the student's demonstrated skill levels."""
ISSUE_EXPLANATION_SYSTEM_PROMPT = """You explain only issues present in the supplied GitHub API results. Do not add, replace, or invent issues, repository names, URLs, facts, or issue numbers. Reference each result by its provided zero-based index. Return one JSON array of objects with id, why_this_fits, and difficulty. The explanation is coaching only; it must not write a pull request."""
SEARCH_PLAN_KEYS = {"languages", "topics", "labels", "keywords", "contribution_types"}


def _new_genai_client(client: Any | None) -> Any:
    if client is not None:
        return client
    api_key = _api_key()
    if not api_key:
        raise RuntimeError("Gemini API key is missing. Set GEMINI_API_KEY in the environment or Streamlit secrets.")
    try:
        from google import genai

        return genai.Client(api_key=api_key)
    except Exception as exc:
        raise RuntimeError(f"Could not initialize the Gemini client: {exc}") from exc


def _model_json(prompt: str, system_prompt: str, client: Any | None = None) -> Any:
    client = _new_genai_client(client)
    try:
        response = client.models.generate_content(
            model=_model_name(),
            contents=prompt,
            config={"system_instruction": system_prompt, "response_mime_type": "application/json"},
        )
    except Exception as exc:
        raise RuntimeError(f"Gemma API request failed: {exc}") from exc
    response_text = getattr(response, "text", None)
    if not response_text:
        raise RuntimeError("Gemma returned an empty response. Please try again.")
    text = re.sub(r"^\s*```(?:json)?\s*", "", response_text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    candidates = [(text.find("{"), "}"), (text.find("["), "]")]
    candidates = [(start, closer) for start, closer in candidates if start >= 0]
    if not candidates:
        raise RuntimeError("Gemma returned no JSON. Please try again.")
    start, closer = min(candidates, key=lambda candidate: candidate[0])
    end = text.rfind(closer)
    try:
        return json.loads(text[start : end + 1])
    except (json.JSONDecodeError, TypeError) as exc:
        raise RuntimeError("Gemma returned invalid JSON. Please try again.") from exc


def _profile_ratings(profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "age": profile.get("Age"),
        "stream": profile.get("Stream_12th"),
        "degree": profile.get("Current_Degree"),
        "primary_interest": profile.get("Primary_Interest"),
        "motivation": profile.get("Influencing_Factor"),
        "skills_1_to_10": {
            "programming": profile.get("Programming_Skill_1_10"),
            "communication": profile.get("Communication_Skill_1_10"),
            "analytical": profile.get("Analytical_Skill_1_10"),
            "creative_design": profile.get("Creative_Design_Skill_1_10"),
        },
    }


def _skill_matched_contributions(profile: dict[str, Any]) -> list[str]:
    ratings = {
        "programming": profile.get("Programming_Skill_1_10"),
        "communication": profile.get("Communication_Skill_1_10"),
        "creative_design": profile.get("Creative_Design_Skill_1_10"),
        "analytical": profile.get("Analytical_Skill_1_10"),
    }
    contributions = []
    if ratings["communication"] is not None and float(ratings["communication"]) <= 5:
        contributions.extend(["documentation improvements", "tests and reproducible examples"])
    if ratings["programming"] is not None and float(ratings["programming"]) >= 7:
        contributions.append("bug fixes")
    if ratings["creative_design"] is not None and float(ratings["creative_design"]) >= 7:
        contributions.append("UI and design issues")
    if ratings["analytical"] is not None and float(ratings["analytical"]) >= 7:
        contributions.append("data validation and analysis tasks")
    return contributions


def generate_search_plan(
    profile: dict[str, Any],
    career: str,
    skill_gaps: list[dict[str, Any]] | list[str],
    *,
    client: Any | None = None,
) -> dict[str, list[str]]:
    """Ask Gemma for search terms, then anchor contribution types to ratings."""
    prompt = (
        "Create a concise search plan for verified GitHub issues that could help this student grow toward the ML-predicted career. "
        "Use only the provided student profile and skill gaps. Do not suggest repository names or URLs. "
        "Return exactly one JSON object with these keys, each containing a list of strings: "
        + json.dumps({key: ["string"] for key in sorted(SEARCH_PLAN_KEYS)}, ensure_ascii=False)
        + "\nStudent profile and 1-10 ratings: " + json.dumps(_profile_ratings(profile), ensure_ascii=False)
        + "\nML-predicted career: " + career
        + "\nSkill gaps: " + json.dumps(skill_gaps, ensure_ascii=False)
    )
    payload = _model_json(prompt, SEARCH_PLAN_SYSTEM_PROMPT, client)
    if not isinstance(payload, dict) or not SEARCH_PLAN_KEYS.issubset(payload):
        raise RuntimeError("Gemma's search plan is missing required fields. Please try again.")
    result: dict[str, list[str]] = {}
    for key in SEARCH_PLAN_KEYS:
        value = payload[key]
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise RuntimeError(f"Gemma's search plan field '{key}' must be a list of strings. Please try again.")
        result[key] = value
    matched = _skill_matched_contributions(profile)
    result["contribution_types"] = list(dict.fromkeys(result["contribution_types"] + matched))
    cleaned = _remove_link_strings(result)
    return {key: cleaned[key] for key in SEARCH_PLAN_KEYS}


def explain_issues(
    profile: dict[str, Any],
    career: str,
    issues: list[dict[str, Any]],
    *,
    client: Any | None = None,
) -> list[dict[str, Any]]:
    """Explain only API-returned issues and discard non-existent Gemma indices."""
    if not issues:
        return []
    verified_issue_data = []
    for index, issue in enumerate(issues):
        verified_issue_data.append(
            {
                "id": index,
                "repo_full_name": issue.get("repo_full_name"),
                "title": issue.get("title"),
                "labels": issue.get("labels", []),
                "comments": issue.get("comments"),
                "updated_at": issue.get("updated_at"),
            }
        )
    prompt = (
        "Explain which of these already-verified GitHub issues may fit the student's learning goals. "
        "The issue list below is the complete set of results; do not suggest any other issue. "
        "Only use the API-provided titles, repository names, labels, and activity fields as issue facts. "
        "Return a JSON array with objects containing integer id (the supplied zero-based index), why_this_fits, and difficulty. "
        "Do not include links.\nStudent profile: " + json.dumps(_profile_ratings(profile), ensure_ascii=False)
        + "\nML-predicted career: " + career
        + "\nVerified GitHub API results: " + json.dumps(verified_issue_data, ensure_ascii=False)
    )
    payload = _model_json(prompt, ISSUE_EXPLANATION_SYSTEM_PROMPT, client)
    if isinstance(payload, dict):
        if "explanations" in payload and isinstance(payload["explanations"], list):
            payload = payload["explanations"]
        elif {"id", "why_this_fits", "difficulty"}.issubset(payload):
            payload = [payload]
        else:
            raise RuntimeError("Gemma's issue explanations have an invalid format. Please try again.")
    if not isinstance(payload, list):
        raise RuntimeError("Gemma's issue explanations have an invalid format. Please try again.")
    explanations = []
    verified_repo_names = {issue.get("repo_full_name") for issue in issues}
    seen: set[int] = set()
    for item in payload:
        if not isinstance(item, dict):
            continue
        identifier = item.get("id")
        if isinstance(identifier, bool):
            continue
        if isinstance(identifier, int):
            index = identifier
        elif isinstance(identifier, str) and identifier.isdecimal():
            index = int(identifier)
        else:
            continue
        if index < 0 or index >= len(issues) or index in seen:
            continue
        why = item.get("why_this_fits")
        difficulty = item.get("difficulty")
        if not isinstance(why, str) or not isinstance(difficulty, str):
            continue
        if _remove_link_strings(why) is None:
            continue
        repo_mentions = re.findall(r"(?i)\b[a-z0-9_.-]+/[a-z0-9_.-]+\b", why)
        if any(repo not in verified_repo_names for repo in repo_mentions):
            continue
        seen.add(index)
        explanations.append({"id": index, "why_this_fits": why, "difficulty": difficulty})
    return explanations
