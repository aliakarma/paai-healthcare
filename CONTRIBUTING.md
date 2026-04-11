# Contributing to AgHealth+ (PAAI)

Thank you for your interest in contributing to the PAAI framework! This document provides guidelines for contributing to this research codebase.

## Code of Conduct

All contributors are expected to uphold a professional and respectful tone in discussions and code review.

## Contribution Types

We welcome contributions in the following areas:

### 1. Bug Reports & Fixes
- **How to Report**: Open a GitHub issue with:
  - Reproducible example / steps
  - Expected vs. actual behavior
  - Python version, OS, and dependency versions
  - Error traceback (if applicable)

- **Process**:
  - Create a branch: `git checkout -b fix/issue-name`
  - Add a test that reproduces the bug
  - Fix the bug and ensure tests pass
  - Submit a PR with reference to the issue

### 2. Feature Requests
- Open a GitHub discussion or issue describing the feature
- Explain the use case and expected impact
- Wait for maintainer feedback before implementing

### 3. Documentation Improvements
- Typos and clarity improvements are always welcome
- Submit a PR with changes to `.md` files
- Large documentation restructuring should be discussed first

### 4. Research Extensions
For extensions to the RL training, agent behavior, or evaluation methodology:
- Open an issue describing the extension
- Include performance implications and validation approach
- Coordinate with maintainers to ensure alignment with paper methodology

## Development Setup

### Prerequisites
- Python 3.10 or higher
- Git

### Local Setup
```bash
# Clone repository
git clone https://github.com/aliakarma/paai-healthcare.git
cd paai-healthcare

# Install in development mode
pip install -r requirements.txt
pip install -e .

# Run tests to verify setup
python -m pytest tests/ -v
```

### Development Workflow

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/descriptive-name
   ```

2. **Make changes** with proper documentation:
   - Add docstrings following the existing style (Google-style docstrings)
   - Include type hints for function arguments and returns
   - Add tests for new functionality

3. **Run tests locally**:
   ```bash
   python -m pytest tests/ -v --cov=.  # With coverage
   ```

4. **Commit with clear messages**:
   ```bash
   git commit -m "component: Brief description of change"
   # Examples:
   # - "agents: Fix medicine agent sodium cap enforcement"
   # - "orchestrator: Add error handling for missing feature store"
   # - "docs: Update MIMIC setup guide with new credentials URL"
   ```

5. **Push and create a pull request**:
   ```bash
   git push origin feature/descriptive-name
   ```

## Pull Request Guidelines

### Before Submitting
- [ ] Code follows existing style conventions
- [ ] All functions and classes have docstrings
- [ ] New code includes type hints
- [ ] All tests pass locally: `python -m pytest tests/ -v`
- [ ] No unnecessary dependencies added
- [ ] Changes don't break reproducibility conditions (`--seed 42`)

### PR Description
Include:
- **What**: Brief description of changes
- **Why**: Motivation or issue being fixed
- **Testing**: How the change was tested
- **Impact**: Any changes to reproducibility or results

### Review Process
- Maintainers will review within 1-2 weeks
- Feedback will be given inline; please respond to all comments
- Re-request review after making requested changes

## Testing Requirements

All contributions should maintain or improve test coverage:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_agents.py -v

# Run with coverage report
python -m pytest tests/ --cov=. --cov-report=html
```

### Tests Locations
- `tests/` — main test suite
- `test_bugfixes.py` — regression tests (root directory)
- Each major module should have corresponding test file

## Documentation Standards

- **Module docstrings**: Describe module purpose and main classes/functions
- **Function docstrings**: Include purpose, arguments (with types), return value, and examples
- **Comments**: Explain *why*, not *what* (code should be self-documenting)
- **MarkDown files**: Use clear headers, code blocks with language tags, and tables where appropriate

Example Python docstring:
```python
def check_medication_interaction(drug_a: str, drug_b: str) -> bool:
    """Check for known drug-drug interactions.
    
    Args:
        drug_a: Name of first drug
        drug_b: Name of second drug
        
    Returns:
        True if interaction exists, False otherwise
        
    Example:
        >>> check_medication_interaction("metformin", "lisinopril")
        False
    """
```

## Reproducibility & Paper Alignment

**Important**: This codebase implements a published research methodology. When contributing:

1. **Do not change default hyperparameters** in `configs/` without discussion
2. **Preserve `--seed 42` behavior** — results must be reproducible
3. **Document any algorithmic changes** clearly
4. **Test against baseline results** in Table 2 of the paper

If you believe a change is necessary for correctness:
- Open an issue with reproduction steps showing the incorrect behavior
- Wait for maintainer acknowledgement
- Make changes only after consensus

## Code Style

### Python Style
- Follow PEP 8 guidelines
- Use type hints: `def func(x: int) -> str:`
- Line length: aim for < 100 characters
- Use meaningful variable names

### Imports
- Group: standard library, third-party, local
- Use `from module import name` for clarity
- Avoid wildcard imports (`from X import *`)

### Example
```python
"""module_name.py — brief description."""

import os
from pathlib import Path

import numpy as np
import yaml

from agents.base_agent import BaseAgent
from knowledge.knowledge_graph import KnowledgeGraph


class ExampleClass:
    """Class description."""
    
    def __init__(self, config_path: str) -> None:
        """Initialize from config file."""
        self.config = yaml.safe_load(open(config_path))
```

## Common Contribution Scenarios

### Adding a New Evaluation Metric
1. Add metric function to `evaluation/metrics.py` with full docstring
2. Add test case to `tests/test_metrics.py`
3. Update evaluation pipeline in `evaluation/run_evaluation.py`
4. Document in `docs/evaluation.md` if creating new doc

### Fixing an Agent Bug
1. Add test to `tests/test_agents.py` that reproduces the bug
2. Fix the bug in `agents/<agent_name>.py`
3. Ensure test now passes
4. Update agent docstring if behavior changed

### Adding a New Configuration Option
1. Add parameter to relevant YAML file in `configs/`
2. Document the parameter with description and units
3. Add loading/validation in the corresponding `.py` module
4. Add test case validating the parameter

## Getting Help

- **Technical questions**: Open a GitHub discussion
- **Bug reports**: Open an issue with reproducible example
- **Documentation confusion**: Request clarification in a discussion

## License

All contributions are licensed under Apache 2.0. By submitting a contribution, you agree to this licensing.

---

Thank you for contributing to AgHealth+ PAAI! 🚀
