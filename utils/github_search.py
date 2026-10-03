"""Verified GitHub issue search for Hacktoberfest-mode coaching."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import os
import re
from typing import Any, Callable

import requests
import streamlit as st

API_ROOT = "https://api.github.com"
SEARCH_REPOSITORIES = f"{API_ROOT}/search/repositories"
SEARCH_ISSUES = f"{API_ROOT}/search/issues"
ISSUE_LABELS = ("good first issue", "help wanted", "hacktoberfest")
SHORT_TIMEOUT_SECONDS = 4


def _github_token() -> str | None:
    token = os.getenv("GITHUB_TOKEN")
    if token:
        return token.strip() or None
    try:
        token = st.secrets.get("GITHUB_TOKEN")
    except Exception:
        token = None
    return str(token).strip() if token else None


def _headers(token: str | None) -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _failure_message(status: int | None, response: Any | None = None) -> str:
    if status in (403, 429):
        return "GitHub search is currently rate-limited. No issue links are shown; use the search keywords and issue categories below and try again later."
    if status == 422:
        return "GitHub could not process this search query. No issue links are shown; use the search keywords and issue categories below."
    if status is not None:
        return f"GitHub search returned HTTP {status}. No issue links are shown; use the search keywords and issue categories below."
    return "GitHub search could not connect. No issue links are shown; use the search keywords and issue categories below."


def _result(issues: list[dict[str, Any]], message: str, searched_at: str) -> dict[str, Any]:
    return {"issues": issues, "message": message, "searched_at": searched_at}


def _search_response(get: Callable[..., Any], url: str, params: dict[str, str], headers: dict[str, str], timeout: float) -> dict[str, Any]:
    response = get(url, params=params, headers=headers, timeout=timeout)
    status = getattr(response, "status_code", 200)
    if status in (403, 422, 429):
        raise _GitHubSearchError(_failure_message(status))
    if status >= 400:
        raise _GitHubSearchError(_failure_message(status))
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise _GitHubSearchError("GitHub returned an unexpected search response. No issue links are shown; use the search plan below.")
    return payload


class _GitHubSearchError(RuntimeError):
    pass


def _valid_language(value: Any) -> str | None:
    language = str(value).strip()
    if not re.fullmatch(r"[A-Za-z0-9+#.-]{1,30}", language):
        return None
    return language


def _search_hacktoberfest_issues(
    languages: tuple[str, ...],
    token: str | None,
    timeout: float,
    max_repositories: int,
    *,
    get: Callable[..., Any],
    today: datetime | None = None,
) -> dict[str, Any]:
    searched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if not languages:
        return _result([], "No programming languages were available in the search plan. Use the search keywords and issue categories below.", searched_at)

    now = today or datetime.now(timezone.utc)
    cutoff = (now.astimezone(timezone.utc).date() - timedelta(days=60)).isoformat()
    headers = _headers(token)
    repositories: dict[str, dict[str, Any]] = {}
    try:
        for language in languages[:3]:
            query = f"topic:hacktoberfest language:{language} good-first-issues:>0 pushed:>={cutoff} archived:false"
            payload = _search_response(
                get,
                SEARCH_REPOSITORIES,
                {"q": query, "sort": "updated", "order": "desc", "per_page": "10"},
                headers,
                timeout,
            )
            for repository in payload.get("items", []):
                full_name = repository.get("full_name")
                if isinstance(full_name, str) and full_name:
                    repositories[full_name] = repository

        top_repositories = sorted(
            repositories.values(),
            key=lambda repository: str(repository.get("updated_at", "")),
            reverse=True,
        )[:max_repositories]
        if not top_repositories:
            return _result([], "No matching active Hacktoberfest repositories were found. Use the search keywords and issue categories below.", searched_at)

        verified_issues: list[dict[str, Any]] = []
        seen_links: set[str] = set()
        for repository in top_repositories:
            full_name = repository["full_name"]
            for label in ISSUE_LABELS:
                query = f'repo:{full_name} is:issue is:open no:assignee label:"{label}"'
                payload = _search_response(
                    get,
                    SEARCH_ISSUES,
                    {"q": query, "sort": "updated", "order": "desc", "per_page": "10"},
                    headers,
                    timeout,
                )
                for item in payload.get("items", []):
                    html_url = item.get("html_url")
                    title = item.get("title")
                    if not isinstance(html_url, str) or not isinstance(title, str) or html_url in seen_links:
                        continue
                    seen_links.add(html_url)
                    verified_issues.append(
                        {
                            "repo_full_name": full_name,
                            "html_url": html_url,
                            "title": title,
                            "labels": [label_item.get("name", "") for label_item in item.get("labels", [])],
                            "comments": item.get("comments", 0),
                            "updated_at": item.get("updated_at", ""),
                        }
                    )
        return _result(verified_issues, "" if verified_issues else "GitHub search completed but found no open, unassigned issues with the searched labels. Use the search keywords and issue categories below.", searched_at)
    except _GitHubSearchError as exc:
        return _result([], str(exc), searched_at)
    except requests.RequestException:
        return _result([], _failure_message(None), searched_at)
    except (ValueError, TypeError, KeyError):
        return _result([], "GitHub returned an unexpected response. No issue links are shown; use the search keywords and issue categories below.", searched_at)


@st.cache_data(ttl=900, show_spinner=False)
def _cached_search(languages: tuple[str, ...], token: str | None, timeout: float, max_repositories: int) -> dict[str, Any]:
    return _search_hacktoberfest_issues(
        languages,
        token,
        timeout,
        max_repositories,
        get=requests.get,
    )


def search_hacktoberfest_issues(
    languages: list[str] | tuple[str, ...],
    *,
    token: str | None = None,
    timeout: float = SHORT_TIMEOUT_SECONDS,
    max_repositories: int = 2,
    session: Any | None = None,
    today: datetime | None = None,
) -> dict[str, Any]:
    """Find recent repository-backed open issues; failures return no links.

    Regular app requests use a 15-minute Streamlit cache. Tests may inject a
    requests-compatible session and clock to exercise API responses directly.
    """
    safe_languages = tuple(dict.fromkeys(language for value in languages if (language := _valid_language(value))))[:3]
    auth_token = token if token is not None else _github_token()
    if session is None and today is None:
        return _cached_search(safe_languages, auth_token, timeout, max_repositories)
    requester = session.get if hasattr(session, "get") else requests.get
    return _search_hacktoberfest_issues(
        safe_languages,
        auth_token,
        timeout,
        max_repositories,
        get=requester,
        today=today,
    )
