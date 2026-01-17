# Testing Guide

This document provides comprehensive information about the PaperAgent testing infrastructure.

## Table of Contents

- [Test Structure](#test-structure)
- [Running Tests](#running-tests)
- [Test Coverage](#test-coverage)
- [Writing Tests](#writing-tests)
- [Continuous Integration](#continuous-integration)

## Test Structure

The test suite is organized into three main categories:

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── pytest.ini               # Pytest configuration
│
├── unit/                    # Unit tests
│   ├── test_core.py         # Core configuration and LLM tests
│   ├── test_agents.py       # Agent functionality tests
│   ├── test_tools.py        # Tool module tests
│   └── test_database.py     # Database model tests
│
└── integration/             # Integration tests
    ├── test_workflows.py    # End-to-end workflow tests
    └── test_api.py          # API endpoint tests
```

### Unit Tests

Unit tests focus on individual components in isolation:

- **test_core.py**: Tests for Settings, LLMManager, and Prompts
- **test_agents.py**: Tests for all agent classes (Boss, Literature, Experiment, Writing)
- **test_tools.py**: Tests for tools (ArxivCollector, LaTeXProcessor, AdvancedStatistics, etc.)
- **test_database.py**: Tests for database models and queries

### Integration Tests

Integration tests verify that components work together correctly:

- **test_workflows.py**: End-to-end research workflows
- **test_api.py**: FastAPI endpoint integration tests

## Running Tests

### Run All Tests

```bash
# Run entire test suite
pytest

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=paperagent --cov-report=html
```

### Run Specific Test Categories

```bash
# Run only unit tests
pytest tests/unit/

# Run only integration tests
pytest tests/integration/

# Run specific test file
pytest tests/unit/test_core.py

# Run specific test class
pytest tests/unit/test_agents.py::TestLiteratureAgent

# Run specific test function
pytest tests/unit/test_agents.py::TestLiteratureAgent::test_search_literature
```

### Run Tests with Markers

```bash
# Run only unit tests (marked with @pytest.mark.unit)
pytest -m unit

# Run only integration tests
pytest -m integration

# Skip slow tests
pytest -m "not slow"

# Run slow tests only
pytest -m slow
```

### Parallel Test Execution

```bash
# Run tests in parallel using pytest-xdist
pytest -n auto  # Uses all available CPUs

pytest -n 4     # Uses 4 CPUs
```

## Test Coverage

### Generate Coverage Report

```bash
# Generate HTML coverage report
pytest --cov=paperagent --cov-report=html

# Open report
# Windows
start htmlcov/index.html

# Linux/Mac
open htmlcov/index.html
```

### Coverage Requirements

The project enforces a minimum coverage threshold of **70%**. This is configured in `pytest.ini`:

```ini
[pytest]
addopts = --cov-fail-under=70
```

### View Coverage in Terminal

```bash
# Terminal coverage report
pytest --cov=paperagent --cov-report=term
```

## Writing Tests

### Test Fixtures

Common fixtures are defined in `conftest.py`:

```python
# Database fixtures
db_engine      # SQLAlchemy engine (in-memory SQLite)
db_session     # Database session

# Mock fixtures
mock_llm       # Mock LLM manager
mock_arxiv_response
mock_scholar_response

# Sample data
sample_project_data
sample_literature_data
sample_dataframe
sample_code
```

### Using Fixtures in Tests

```python
def test_create_project(db_session, sample_project_data):
    """Test project creation"""
    project = Project(**sample_project_data)
    db_session.add(project)
    db_session.commit()

    assert project.id is not None
```

### Mocking External APIs

```python
from unittest.mock import patch

@patch('paperagent.tools.literature_collector.ArxivCollector')
def test_search_papers(mock_arxiv, mock_llm, db_session):
    """Test paper search with mocked API"""
    mock_arxiv.return_value.search.return_value = [
        {'title': 'Test Paper', 'authors': ['Author A']}
    ]

    agent = LiteratureAgent(llm=mock_llm, db_session=db_session)
    result = agent.execute({'action': 'search_literature', 'query': 'ML'})

    assert 'total_papers' in result
```

### Testing Async Code

```python
import pytest

@pytest.mark.asyncio
async def test_async_function():
    """Test async functionality"""
    result = await some_async_function()
    assert result is not None
```

### Testing Exceptions

```python
def test_invalid_provider(mock_llm, db_session):
    """Test error handling for invalid provider"""
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        LLMManager(provider="invalid_provider")
```

### Parametrized Tests

```python
@pytest.mark.parametrize("input,expected", [
    ("hello", "HELLO"),
    ("world", "WORLD"),
    ("test", "TEST")
])
def test_uppercase(input, expected):
    assert input.upper() == expected
```

## Test Best Practices

### 1. Test Naming Convention

- Test files: `test_*.py`
- Test functions: `test_*`
- Test classes: `Test*`

```python
class TestLiteratureAgent:
    def test_search_literature(self):
        pass

    def test_analyze_gap(self):
        pass
```

### 2. Arrange-Act-Assert Pattern

```python
def test_create_project(db_session):
    # Arrange
    project_data = {'name': 'Test', 'research_field': 'CS'}

    # Act
    project = Project(**project_data)
    db_session.add(project)
    db_session.commit()

    # Assert
    assert project.id is not None
    assert project.name == 'Test'
```

### 3. Mock External Dependencies

Always mock external API calls, file I/O, and database operations in unit tests:

```python
@patch('paperagent.tools.literature_collector.requests.get')
def test_api_call(mock_get):
    mock_get.return_value.json.return_value = {'data': 'test'}
    # Test code here
```

### 4. Use Descriptive Test Names

```python
# Good
def test_literature_search_returns_correct_number_of_papers():
    pass

# Bad
def test_search():
    pass
```

### 5. Keep Tests Independent

Each test should be able to run independently and in any order.

### 6. Test Edge Cases

```python
def test_empty_input():
    """Test handling of empty input"""
    pass

def test_null_input():
    """Test handling of null input"""
    pass

def test_large_input():
    """Test handling of large dataset"""
    pass
```

## Conditional Test Execution

Some tests require specific conditions to run (API keys, databases, etc.):

```python
# In conftest.py
skip_if_no_openai = pytest.mark.skipif(
    not os.getenv('OPENAI_API_KEY'),
    reason="OpenAI API key not available"
)

# In test file
@skip_if_no_openai
def test_openai_integration():
    pass
```

## Continuous Integration

### GitHub Actions Example

Create `.github/workflows/tests.yml`:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v2

    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt

    - name: Run tests
      run: |
        pytest --cov=paperagent --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v2
      with:
        files: ./coverage.xml
```

## Debugging Tests

### Run Tests with Debugger

```bash
# Run with Python debugger
python -m pdb -m pytest tests/test_file.py

# Run with ipdb (install: pip install ipdb)
pytest --pdb  # Drop into debugger on failure
```

### Show Print Statements

```bash
# Show print output even for passing tests
pytest -s

# Show output for failed tests only
pytest --tb=short
```

### Verbose Output

```bash
# Very verbose
pytest -vv

# Show test durations
pytest --durations=10  # Show 10 slowest tests
```

## Performance Testing

### Benchmarking with pytest-benchmark

```bash
pip install pytest-benchmark
```

```python
def test_performance(benchmark):
    """Test function performance"""
    result = benchmark(expensive_function, arg1, arg2)
    assert result is not None
```

## Test Data Management

### Using Fixtures for Test Data

Store test data in `tests/fixtures/`:

```python
# In conftest.py
@pytest.fixture
def sample_pdf():
    return Path(__file__).parent / "fixtures" / "sample.pdf"
```

### Temporary Files

```python
def test_with_temp_file(tmp_path):
    """Test using temporary directory"""
    temp_file = tmp_path / "test.txt"
    temp_file.write_text("test content")

    # Test code using temp_file
```

## Common Issues and Solutions

### Issue: Tests Pass Individually but Fail Together

**Solution**: Tests may be sharing state. Check for:
- Database transactions not being rolled back
- Global variables being modified
- Mock objects not being reset

### Issue: Slow Test Suite

**Solutions**:
- Use `pytest-xdist` for parallel execution
- Mark slow tests with `@pytest.mark.slow`
- Mock expensive operations
- Use in-memory databases for tests

### Issue: Flaky Tests

**Solutions**:
- Avoid timing dependencies
- Mock random number generators
- Ensure proper test isolation
- Use `pytest-repeat` to identify flaky tests:
  ```bash
  pytest --count=100 tests/test_flaky.py
  ```

## Additional Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Python Testing with pytest Book](https://pragprog.com/titles/bopytest/)
- [Test-Driven Development](https://en.wikipedia.org/wiki/Test-driven_development)

## Support

For questions about testing:
- Check existing test examples in the codebase
- Refer to pytest documentation
- Open an issue on GitHub

---

**Happy Testing! 🧪**
