# India Youth Career

A Streamlit app for exploring synthetic student career preferences, predicting a data-driven career profile, and planning practical next steps with Gemma and verified GitHub issue data.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

## API configuration

Set the Gemini API key in the environment or in `.streamlit/secrets.toml`. The counselor checks `GEMINI_API_KEY`; the model defaults to `gemma-4-26b-a4b-it` and can be changed with `GEMMA_MODEL`.

```powershell
$env:GEMINI_API_KEY = "your-key"
$env:GEMMA_MODEL = "gemma-4-26b-a4b-it"
# Optional. Authenticated search can provide a higher GitHub API rate limit.
$env:GITHUB_TOKEN = "your-token"
```

A local secrets file can contain:

```toml
GEMINI_API_KEY = "your-key"
GEMMA_MODEL = "gemma-4-26b-a4b-it"
GITHUB_TOKEN = "your-optional-token"
```

Keep secrets out of source control. The `.streamlit/secrets.toml` path is ignored by Git. `GITHUB_TOKEN` is optional; search failures and rate limits show the Gemma search terms and issue categories without links.

## Architecture

- `app.py` provides Home, Analytics, Career Predictor, and AI Career Counselor navigation.
- `pages/analytics.py` applies interactive filters to descriptive charts based on the synthetic dataset.
- `pages/career_predictor.py` validates a student profile and uses cached, already-trained artifacts from `models/`. Model training remains separate in `train_model.py`.
- `pages/ai_counselor.py` contains two tabs: the six-month Gemma roadmap and Hacktoberfest Mode.
- `utils/gemma.py` makes structured Google GenAI requests. The ML prediction stays authoritative; Gemma adds guidance and never replaces it.
- `utils/github_search.py` searches active repositories, then open unassigned issues using GitHub's search API. Issue titles, repository names, labels, dates, and links come from API results. Gemma explanations reference returned results by index; links in the UI always come from the GitHub API record.
- `utils/roadmap_ui.py` renders the roadmap as skill comparisons, monthly cards, project cards, and preparation sections.

## Integrity notes

The supplied career-choice data is synthetic and has no salary or job-satisfaction fields. Dashboard patterns are descriptive associations and do not establish causation. Model metrics describe held-out data from this synthetic dataset, and prediction probabilities are model estimates rather than guarantees.

The app does not train a model while serving pages, fabricate confidence values, or invent repository names, issue numbers, or GitHub links. Hacktoberfest Mode searches GitHub live and only renders issue links returned by the API. If GitHub search is unavailable, it falls back to search keywords and issue categories without links. Gemma coaches contributors but does not write pull requests. Always check the current rules and register at hacktoberfest.com.

Career predictions are data-driven matches based on the model and information provided. They are not guarantees of career success.

## Tests

Run the offline mocked API and Streamlit tests with:

```powershell
python -m pytest
```

The suite covers GitHub rate limiting and empty results, API-backed issue fields, Gemma's rating-aware search plan, dropping invented explanation indexes, URL stripping, and the counselor's Hacktoberfest tab and tracker. It does not require live API calls.

## 2-3 minute demo script

1. **Analytics (~20 seconds):** show the synthetic-data note, filter controls, and a chart update.
2. **Predictor (~30 seconds):** fill the example profile, show the ML result and alternatives, and note that the model's match is preserved.
3. **Gemma roadmap (~50 seconds):** generate the six-month roadmap and scan the skill gaps, deliverables, and project cards.
4. **Hacktoberfest tab (~40 seconds):** build a search plan, search GitHub, compare API-backed issue cards with Gemma's explanations, and show the PR-readiness checklist and personal tracker. Use a configured Gemini key and network access for the live portions.
5. **Close (~10 seconds):** recap **From Data -> Prediction -> Action**.
