"""
Pytest configuration and fixtures
"""

import pytest
import os
import tempfile
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from unittest.mock import Mock, MagicMock

from paperagent.database.models import Base
from paperagent.database.database import get_db
from paperagent.core.config import Settings


@pytest.fixture(scope="session")
def test_settings():
    """Test settings override"""
    return Settings(
        database_url="sqlite:///:memory:",
        redis_url="redis://localhost:6379/15",  # Use test database
        default_llm_provider="ollama",
        debug=True
    )


@pytest.fixture(scope="function")
def db_engine():
    """Create test database engine"""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)
    engine.dispose()


@pytest.fixture(scope="function")
def db_session(db_engine):
    """Create test database session"""
    Session = sessionmaker(bind=db_engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture
def mock_llm():
    """Mock LLM for testing"""
    mock = MagicMock()
    mock.generate.return_value = "Test response from LLM"
    mock.chat.return_value = "Test chat response"
    return mock


@pytest.fixture
def sample_project_data():
    """Sample project data for testing"""
    return {
        "name": "Test Research Project",
        "description": "A test project for unit testing",
        "research_field": "Computer Science",
        "keywords": ["testing", "pytest", "automation"]
    }


@pytest.fixture
def sample_literature_data():
    """Sample literature data for testing"""
    return {
        "title": "Test Paper on Machine Learning",
        "authors": ["John Doe", "Jane Smith"],
        "abstract": "This is a test abstract for a machine learning paper.",
        "year": 2024,
        "doi": "10.1000/test.doi",
        "source": "arxiv"
    }


@pytest.fixture
def sample_paper_data():
    """Sample paper data for testing"""
    return {
        "title": "A Comprehensive Study of Testing Frameworks",
        "abstract": "This paper presents a comprehensive study of modern testing frameworks.",
        "introduction": "Testing is crucial for software quality...",
        "methodology": "We conducted experiments using pytest...",
        "results": "The results show improved code quality...",
        "conclusion": "In conclusion, automated testing is essential..."
    }


@pytest.fixture
def temp_dir():
    """Create temporary directory for file tests"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_pdf_path(temp_dir):
    """Create a sample PDF for testing"""
    # This would create an actual PDF, but for testing we'll use a mock path
    pdf_path = temp_dir / "sample.pdf"
    pdf_path.touch()
    return str(pdf_path)


@pytest.fixture
def sample_latex_content():
    """Sample LaTeX content for testing"""
    return r"""
\documentclass{article}
\usepackage{amsmath}

\title{Test Document}
\author{Test Author}

\begin{document}
\maketitle

\section{Introduction}
This is a test document.

\end{document}
"""


@pytest.fixture
def sample_bibtex():
    """Sample BibTeX entry for testing"""
    return """
@article{test2024,
  title={Test Article},
  author={Doe, John and Smith, Jane},
  journal={Test Journal},
  year={2024},
  volume={1},
  pages={1--10}
}
"""


@pytest.fixture
def sample_dataframe():
    """Sample pandas DataFrame for testing"""
    import pandas as pd
    import numpy as np

    np.random.seed(42)
    return pd.DataFrame({
        'group': ['A'] * 50 + ['B'] * 50,
        'value': np.random.randn(100),
        'category': np.random.choice(['X', 'Y', 'Z'], 100),
        'score': np.random.uniform(0, 100, 100)
    })


@pytest.fixture
def sample_code():
    """Sample Python code for testing"""
    return '''
def fibonacci(n):
    """Calculate Fibonacci number"""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

class Calculator:
    def add(self, a, b):
        return a + b

    def subtract(self, a, b):
        return a - b
'''


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singleton instances between tests"""
    yield
    # Add cleanup code if needed


@pytest.fixture
def mock_arxiv_response():
    """Mock arXiv API response"""
    return [
        {
            "title": "Deep Learning for NLP",
            "authors": ["Author One", "Author Two"],
            "abstract": "This paper presents deep learning methods for NLP.",
            "arxiv_id": "2024.12345",
            "published": "2024-01-01",
            "pdf_url": "https://arxiv.org/pdf/2024.12345"
        }
    ]


@pytest.fixture
def mock_scholar_response():
    """Mock Google Scholar response"""
    return [
        {
            "title": "Machine Learning Survey",
            "authors": ["Researcher A", "Researcher B"],
            "abstract": "A comprehensive survey of machine learning.",
            "year": "2024",
            "citation_count": 100,
            "url": "https://example.com/paper"
        }
    ]


# Markers for conditional test execution
def pytest_configure(config):
    """Configure custom markers"""
    config.addinivalue_line(
        "markers", "requires_api: mark test as requiring API credentials"
    )
    config.addinivalue_line(
        "markers", "requires_database: mark test as requiring database"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


def pytest_collection_modifyitems(config, items):
    """Skip tests based on markers and environment"""
    skip_api = pytest.mark.skip(reason="API credentials not available")
    skip_db = pytest.mark.skip(reason="Database not available")

    for item in items:
        if "requires_api" in item.keywords:
            # Check if API keys are set
            if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
                item.add_marker(skip_api)

        if "requires_database" in item.keywords:
            # Check if database is available
            if not os.getenv("DATABASE_URL"):
                item.add_marker(skip_db)
