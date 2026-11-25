# Engineering & Project Conventions

This document defines the shared engineering standards used across `ai-engineering-labs`.

---

## 1. Repository & Experiment Structure

- **No empty directories:** Folders (`01-...` through `06-...`, `architecture/`, `assets/`) are created only when active work begins.
- **Per-experiment isolation:** Each Python experiment has its own `pyproject.toml`, `uv.lock`, `.env.example`, `README.md`, `src/`, and `tests/`.
- **Naming pattern:** `<NN>-<track>/<exp|poc|ref>-<NN>-<kebab-case-name>` (e.g., `01-python-ai-fundamentals/exp-01-async-api-and-structured-data`).

Standard Python experiment layout:

```text
exp-01-example/
├── README.md          # Follows docs/templates/project-readme-template.md
├── pyproject.toml     # Minimal pinned dependencies for this experiment
├── .env.example       # Documented environment variables (no real secrets)
├── src/
│   └── <package_name>/
│       ├── __init__.py
│       ├── config.py  # Typed settings loaded from environment
│       ├── models.py  # Pydantic request/response/domain contracts
│       └── ...
└── tests/
    └── test_*.py      # Fast offline/mocked unit tests (+ optional live markers)
```

---

## 2. Python Engineering Standards

- **Runtime:** Python `3.13+` managed with `uv`.
- **Typing:** All functions, methods, and public interfaces must include explicit type hints.
- **Data contracts:** Prefer `pydantic` models (or `dataclasses` when zero external dependencies are desired) for structured inputs, outputs, and configuration.
- **Architecture boundaries:** Separate transport/SDK clients from domain logic and orchestration so external APIs can be mocked cleanly in tests.
- **Resilience & observability:**
  - Always configure explicit network timeouts.
  - Handle expected HTTP/API errors gracefully with structured error types or clear messages.
  - Use Python's `logging` module rather than ad-hoc `print()` in library/service code; never log API keys, tokens, or sensitive payloads.
- **Linting & Formatting:** Follow [ruff.toml](../ruff.toml) (`line-length = 100`, `py313`).

---

## 3. Testing Standards

- Every experiment must include `pytest` tests in `tests/`.
- **Offline-first unit tests:** Core logic, schema validation, error handling, and mocked API behavior must pass without network calls or paid API keys.
- **Optional live tests:** If live provider integration tests exist, gate them behind an environment check or `@pytest.mark.live` so default `uv run pytest` runs reliably offline.
- Report exact test commands and observed scope in each experiment's `README.md`.

---

## 4. Security & Privacy Hygiene

- **Never commit:** `.env`, API keys, tokens, credentials, client code, client datasets, proprietary architecture, or personal data.
- **Free-tier LLM awareness:** Cloud free tiers (such as Google AI Studio's Gemini API free tier) may use submitted prompts/outputs to improve provider models—use synthetic or public sample data only.

---

## 5. Git & Documentation Discipline

- Initialize commits with clear, concise imperative messages (e.g., `docs: initialize repository structure and roadmap`).
- Document major technical choices using [adr-template.md](./templates/adr-template.md).
- Never claim unverified benchmarks, production usage, or unobserved test results.
