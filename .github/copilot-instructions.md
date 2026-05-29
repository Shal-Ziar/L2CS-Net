# HMI-Flow Project Guidelines

## Code Style

- **Keep it simple**: Prefer straightforward solutions over clever ones
- **Write atomic code**: Each function should do one thing well
- **Use expressive variable names**: `user_session_data` not `usd`, `calculate_attention_score` not `calc_att`
- **low cyclomatic complexity**: Avoid deeply nested code, break complex logic into smaller functions
- **Always use type hints**: For function parameters, return values, and class attributes

```python
# Good
def process_eeg_signal(raw_data: np.ndarray, sample_rate: int) -> dict[str, float]:
    processed_signal = apply_filter(raw_data)
    return {"mean": processed_signal.mean(), "std": processed_signal.std()}

# Avoid
def proc(d, sr):
    ps = flt(d)
    return {"m": ps.mean(), "s": ps.std()}
```

## Documentation

- Use concise language—focus on what and why, not obvious how
- Prefer inline comments for complex logic, docstrings for public APIs
- **Do not create README.md files** unless explicitly requested

## Conventions

- Follow existing patterns in the codebase (see [analysis/](../analysis/), [collector/](../collector/), [hmi_flow/](../hmi_flow/))
- Use `black` for formatting (configured in pyproject.toml)
- Organize imports with `isort`
