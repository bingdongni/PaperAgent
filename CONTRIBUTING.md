# Contributing to PaperAgent

First off, thank you for considering contributing to PaperAgent! It's people like you that make PaperAgent such a great tool for the research community.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Commit Guidelines](#commit-guidelines)
- [Pull Request Process](#pull-request-process)

## Code of Conduct

This project and everyone participating in it is governed by our commitment to fostering an open and welcoming environment. Please be respectful and constructive in all interactions.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/paperagent.git`
3. Add upstream remote: `git remote add upstream https://github.com/original/paperagent.git`
4. Create a branch: `git checkout -b feature/your-feature-name`

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues. When creating a bug report, include:

- **Clear title and description**
- **Steps to reproduce**
- **Expected vs actual behavior**
- **Environment details** (OS, Python version, etc.)
- **Screenshots** (if applicable)

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, include:

- **Clear title and description**
- **Use case and motivation**
- **Proposed solution**
- **Alternative solutions considered**

### Code Contributions

We welcome code contributions! Here are areas where help is especially appreciated:

- **New agents**: Implement new specialized agents
- **Tool integrations**: Add support for more literature databases, citation managers, etc.
- **Journal templates**: Contribute formatting templates for academic journals
- **Documentation**: Improve guides, add examples, fix typos
- **Tests**: Increase test coverage
- **Bug fixes**: Fix reported issues

## Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/paperagent.git
cd paperagent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies including dev tools
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Set up pre-commit hooks
pre-commit install

# Run tests
pytest tests/

# Start development server
uvicorn paperagent.api.main:app --reload
```

## Coding Standards

### Python Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- Use [Black](https://github.com/psf/black) for code formatting
- Use [isort](https://pycqa.github.io/isort/) for import sorting
- Use type hints where possible

```python
# Good
def analyze_paper(paper_id: int, detailed: bool = False) -> Dict[str, Any]:
    """
    Analyze a research paper.

    Args:
        paper_id: Paper database ID
        detailed: Whether to perform detailed analysis

    Returns:
        Analysis results dictionary
    """
    pass

# Bad
def analyze_paper(paper_id, detailed=False):
    pass
```

### Documentation

- All functions should have docstrings
- Use Google-style docstrings
- Keep README.md up to date
- Add examples for new features

### Testing

- Write tests for new features
- Maintain or improve code coverage
- Use pytest for testing
- Mock external API calls

```python
def test_literature_search():
    agent = LiteratureAgent()
    result = agent.execute({
        'action': 'search_literature',
        'query': 'test query',
        'project_id': 1
    })
    assert 'total_papers' in result
    assert result['total_papers'] >= 0
```

## Commit Guidelines

### Commit Messages

Use conventional commit messages:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**

```
feat(literature): add PubMed integration

Implement PubMed API integration for medical literature search.
Includes rate limiting and error handling.

Closes #123
```

```
fix(writing): resolve abstract word count bug

Fixed issue where abstract word count included LaTeX commands.

Fixes #456
```

### Branch Naming

- `feature/description`: New features
- `fix/description`: Bug fixes
- `docs/description`: Documentation updates
- `refactor/description`: Code refactoring

## Pull Request Process

1. **Update your fork**
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Make your changes**
   - Write code
   - Add tests
   - Update documentation

3. **Run checks locally**
   ```bash
   # Format code
   black paperagent/
   isort paperagent/

   # Run linters
   flake8 paperagent/
   mypy paperagent/

   # Run tests
   pytest tests/ --cov=paperagent
   ```

4. **Commit and push**
   ```bash
   git add .
   git commit -m "feat(agent): add new feature"
   git push origin feature/your-feature-name
   ```

5. **Create Pull Request**
   - Go to GitHub and create a PR
   - Fill in the PR template
   - Link related issues
   - Wait for review

### PR Checklist

- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] All tests pass
- [ ] No new warnings
- [ ] Linked to related issues

### Review Process

- Maintainers will review your PR
- Address feedback and requested changes
- Once approved, your PR will be merged
- Your contribution will be credited in release notes

## Development Tips

### Running Specific Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_literature_agent.py

# Run specific test
pytest tests/test_literature_agent.py::test_search_papers

# Run with coverage
pytest --cov=paperagent --cov-report=html
```

### Debugging

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run with debugger
python -m pdb -m pytest tests/test_file.py
```

### Database Migrations

When modifying database models:

```bash
# Generate migration
alembic revision --autogenerate -m "description"

# Apply migration
alembic upgrade head
```

## Need Help?

- 📖 Check the [documentation](README.md)
- 💬 Ask questions in [Discussions](https://github.com/yourusername/paperagent/discussions)
- 🐛 Report bugs in [Issues](https://github.com/yourusername/paperagent/issues)

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Given credit in documentation

Thank you for contributing to PaperAgent! 🎉
