# Contributing to BankMind

Thank you for your interest in contributing. This document describes the process for contributing code, documentation, and ideas to the project.

---

## Development Setup

### Prerequisites

- Python 3.11+
- Git
- Docker (optional)

### Setup

```bash
# 1. Fork and clone the repository
git clone https://github.com/yourusername/bank-ai-assistant.git
cd bank-ai-assistant

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. Install dependencies (including dev tools)
pip install -r requirements.txt

# 4. Copy and configure environment
cp .env.example .env
# Edit .env with your OpenAI API key

# 5. Verify setup — tests should pass without an API key
pytest tests/ -v
```

---

## Branching Strategy

- `main` — stable, production-ready code only
- `develop` — integration branch for features
- `feature/your-feature-name` — individual feature branches

Branch from `develop`, open PRs targeting `develop`. PRs to `main` come only from `develop` via a release process.

---

## Commit Message Convention

Follow the [Conventional Commits](https://www.conventionalcommits.org/) format:

```
type(scope): short description

[optional body]
[optional footer]
```

**Types:**
- `feat` — New feature
- `fix` — Bug fix
- `docs` — Documentation changes only
- `refactor` — Code restructuring, no feature change
- `test` — Adding or fixing tests
- `chore` — Build process, dependency updates

**Examples:**
```
feat(agents): add confidence score to DocQAAgent responses
fix(compliance): correct BNB Ordinance 38 threshold to BGN 30,000
docs(use-cases): add DORA compliance use case
test(compliance): add edge case tests for empty document input
```

---

## Code Standards

### Style

- Follow PEP 8 with a 100-character line limit
- Run `flake8 src/ config/ tests/` before committing — CI will enforce this
- Use `black` for auto-formatting (optional but recommended): `black src/ config/ tests/`
- Sort imports with `isort`: `isort src/ config/ tests/`

### Docstrings

All public classes and methods must have docstrings explaining:
1. **What** the method does (one-sentence summary)
2. **Why** — the reasoning behind design choices (especially in agents)
3. **Args** and **Returns** with types

```python
def query(self, question: str) -> tuple[str, list[dict]]:
    """
    Answer a question using the indexed document corpus.

    Uses semantic similarity search to retrieve the most relevant document
    chunks, then synthesises a grounded answer via the LLM. Temperature is
    set to 0 to ensure deterministic, reproducible answers — critical for
    banking Q&A where employees need to rely on consistency.

    Args:
        question: The employee's natural language question.

    Returns:
        A tuple of (answer_text, sources_list).
    """
```

### Type Hints

All function signatures must have type hints. Use Python 3.11+ syntax (`list[dict]` not `List[Dict]`).

---

## Testing Requirements

- All new features must include unit tests in `tests/`
- Tests must not make real LLM API calls — use `unittest.mock.patch`
- Coverage must not decrease below the current threshold (70%)
- Run the full test suite before opening a PR: `pytest tests/ --cov=src --cov-fail-under=70`

### Writing Good Tests

```python
def test_your_feature_does_expected_thing(self):
    """Tests should have descriptive names that explain what is being tested."""
    with patch("src.agents.your_agent.LANGCHAIN_AVAILABLE", False):
        from src.agents.your_agent import YourAgent
        agent = YourAgent(settings=make_mock_settings())
        result = agent.your_method("input")
        assert isinstance(result, expected_type)
        assert "expected_content" in result
```

---

## Banking Domain Notes

This project operates in a regulated environment. When contributing, be aware:

1. **Accuracy over creativity** — Any changes to agent prompts should maintain strict grounding (no hallucination)
2. **Regulatory references** — Check that any BNB/ECB/GDPR references cited in the codebase or documentation are accurate before adding them
3. **PII sensitivity** — Never add real customer data to tests, examples, or documentation
4. **Audit trails** — Any change that affects what gets logged should be reviewed carefully

---

## Submitting a Pull Request

1. Ensure your branch is up to date with `develop`
2. Run `pytest tests/ -v` — all tests must pass
3. Run `flake8 src/ config/ tests/` — no linting errors
4. Open a PR with:
   - A clear title following the commit convention
   - A description of what changed and why
   - Any relevant test output
5. PRs require one approval from a reviewer before merging

---

## Reporting Issues

Open a GitHub Issue with:
- A clear description of the problem
- Steps to reproduce
- Expected vs. actual behaviour
- Python version and OS

For security vulnerabilities, do not open a public issue — contact the maintainer directly.

---

## License

By contributing, you agree that your contributions will be licensed under the project's MIT License.
