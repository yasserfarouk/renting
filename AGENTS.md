# AGENTS.md

## Build & Test
- **Python:** 3.10 required (`requires-python = "~=3.10"`)
- **Install:** `pip install -e .` or `uv sync`
- **Run all tests:** `pytest`
- **Run single test file:** `pytest tests/test_negotiation.py`
- **Run single test:** `pytest tests/test_negotiation.py::test_smoke`

## Code Style
- **Formatting:** No enforced formatter (black/ruff not configured)
- **Imports:** stdlib first, then third-party (numpy, torch), then local (`from environment.X`)
- **Naming:** PascalCase for classes, snake_case for functions/variables, UPPER_SNAKE_CASE for constants
- **Private:** Leading underscore (`_method`, `_attribute`)
- **Types:** Use modern Python 3.10+ hints (`str | None`, `list[int]`, `dict[str, X]`)

## Error Handling
- Use `assert` for invariants: `assert len(items) > 0`
- Use `ValueError` for invalid inputs with descriptive messages
- Use context managers for resource cleanup

## Key Entry Points
- `ppo.py` - Training script
- `evaluate.py` - Evaluation script
- `environment/negotiation.py` - Core negotiation environment
