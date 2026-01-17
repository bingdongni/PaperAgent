"""
Integration tests for FastAPI endpoints
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import json


@pytest.fixture
def client(db_session):
    """Create test client"""
    from paperagent.api.main import app

    # Override database dependency
    from paperagent.api.main import get_db

    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db

    return TestClient(app)


@pytest.mark.integration
class TestProjectAPI:
    """Test project management API endpoints"""

    def test_create_project(self, client):
        """Test POST /api/projects/"""
        response = client.post(
            "/api/projects/",
            json={
                "name": "Test Project",
                "research_field": "Computer Science",
                "keywords": ["AI", "ML"],
                "description": "Test project description"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert data["name"] == "Test Project"

    def test_get_project(self, client):
        """Test GET /api/projects/{id}"""
        # Create project first
        create_response = client.post(
            "/api/projects/",
            json={
                "name": "Get Test Project",
                "research_field": "AI"
            }
        )
        project_id = create_response.json()["id"]

        # Get project
        response = client.get(f"/api/projects/{project_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == project_id
        assert data["name"] == "Get Test Project"

    def test_list_projects(self, client):
        """Test GET /api/projects/"""
        # Create multiple projects
        for i in range(3):
            client.post(
                "/api/projects/",
                json={
                    "name": f"Project {i}",
                    "research_field": "CS"
                }
            )

        # List projects
        response = client.get("/api/projects/")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 3

    def test_update_project(self, client):
        """Test PUT /api/projects/{id}"""
        # Create project
        create_response = client.post(
            "/api/projects/",
            json={"name": "Original Name", "research_field": "AI"}
        )
        project_id = create_response.json()["id"]

        # Update project
        response = client.put(
            f"/api/projects/{project_id}",
            json={
                "name": "Updated Name",
                "status": "in_progress"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Updated Name"

    def test_delete_project(self, client):
        """Test DELETE /api/projects/{id}"""
        # Create project
        create_response = client.post(
            "/api/projects/",
            json={"name": "To Delete", "research_field": "AI"}
        )
        project_id = create_response.json()["id"]

        # Delete project
        response = client.delete(f"/api/projects/{project_id}")

        assert response.status_code == 200

        # Verify deletion
        get_response = client.get(f"/api/projects/{project_id}")
        assert get_response.status_code == 404


@pytest.mark.integration
class TestLiteratureAPI:
    """Test literature management API endpoints"""

    @patch('paperagent.tools.literature_collector.ArxivCollector')
    def test_search_literature(self, mock_arxiv, client):
        """Test POST /api/literature/search"""
        # Mock arXiv response
        mock_arxiv.return_value.search.return_value = [
            {
                'title': 'Test Paper',
                'authors': ['Author A'],
                'abstract': 'Abstract',
                'arxiv_id': '2024.0001',
                'published': '2024-01-01',
                'url': 'https://arxiv.org/abs/2024.0001'
            }
        ]

        # Create project
        project_response = client.post(
            "/api/projects/",
            json={"name": "Lit Test", "research_field": "CS"}
        )
        project_id = project_response.json()["id"]

        # Search literature
        response = client.post(
            "/api/literature/search",
            json={
                "query": "machine learning",
                "project_id": project_id,
                "source": "arxiv",
                "max_results": 10
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert "total_papers" in data or "papers" in data

    def test_get_literature(self, client):
        """Test GET /api/literature/{id}"""
        # Create project and literature
        project_response = client.post(
            "/api/projects/",
            json={"name": "Test", "research_field": "CS"}
        )
        project_id = project_response.json()["id"]

        lit_response = client.post(
            "/api/literature/",
            json={
                "project_id": project_id,
                "title": "Test Paper",
                "authors": "Author A",
                "source": "arxiv"
            }
        )
        lit_id = lit_response.json()["id"]

        # Get literature
        response = client.get(f"/api/literature/{lit_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["title"] == "Test Paper"

    def test_analyze_literature_gap(self, client, mock_llm):
        """Test POST /api/literature/analyze-gap"""
        # Create project
        project_response = client.post(
            "/api/projects/",
            json={"name": "Gap Analysis", "research_field": "AI"}
        )
        project_id = project_response.json()["id"]

        # Mock LLM response
        mock_llm.generate.return_value = """
        Research Gaps:
        1. Limited work on X
        2. Need for more Y
        """

        # Analyze gap
        response = client.post(
            "/api/literature/analyze-gap",
            json={"project_id": project_id}
        )

        assert response.status_code == 200
        data = response.json()
        assert "gaps" in data or "analysis" in data


@pytest.mark.integration
class TestExperimentAPI:
    """Test experiment management API endpoints"""

    def test_create_experiment(self, client):
        """Test POST /api/experiments/"""
        # Create project
        project_response = client.post(
            "/api/projects/",
            json={"name": "Exp Test", "research_field": "CS"}
        )
        project_id = project_response.json()["id"]

        # Create experiment
        response = client.post(
            "/api/experiments/",
            json={
                "project_id": project_id,
                "name": "Baseline Experiment",
                "description": "Test baseline performance",
                "design": {"method": "A/B testing"}
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Baseline Experiment"

    def test_design_experiment(self, client, mock_llm):
        """Test POST /api/experiments/design"""
        # Create project
        project_response = client.post(
            "/api/projects/",
            json={"name": "Design Test", "research_field": "AI"}
        )
        project_id = project_response.json()["id"]

        # Mock LLM
        mock_llm.generate.return_value = """
        Experiment Design:
        Hypothesis: X improves Y
        Methodology: Controlled experiment
        """

        # Design experiment
        response = client.post(
            "/api/experiments/design",
            json={
                "project_id": project_id,
                "research_question": "Does X improve Y?"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert "design" in data or "methodology" in data

    def test_analyze_data(self, client, sample_dataframe):
        """Test POST /api/experiments/{id}/analyze"""
        # Create project and experiment
        project_response = client.post(
            "/api/projects/",
            json={"name": "Analysis Test", "research_field": "CS"}
        )
        project_id = project_response.json()["id"]

        exp_response = client.post(
            "/api/experiments/",
            json={
                "project_id": project_id,
                "name": "Test Experiment"
            }
        )
        exp_id = exp_response.json()["id"]

        # Analyze data
        response = client.post(
            f"/api/experiments/{exp_id}/analyze",
            json={
                "data": sample_dataframe.to_dict(),
                "analysis_type": "descriptive"
            }
        )

        assert response.status_code == 200

    def test_get_experiment_results(self, client):
        """Test GET /api/experiments/{id}/results"""
        # Create project and experiment with results
        project_response = client.post(
            "/api/projects/",
            json={"name": "Results Test", "research_field": "AI"}
        )
        project_id = project_response.json()["id"]

        exp_response = client.post(
            "/api/experiments/",
            json={
                "project_id": project_id,
                "name": "Results Experiment",
                "results": {"accuracy": 0.95}
            }
        )
        exp_id = exp_response.json()["id"]

        # Get results
        response = client.get(f"/api/experiments/{exp_id}/results")

        assert response.status_code == 200
        data = response.json()
        assert "accuracy" in data or "results" in data


@pytest.mark.integration
class TestPaperAPI:
    """Test paper writing API endpoints"""

    def test_create_paper(self, client):
        """Test POST /api/papers/"""
        # Create project
        project_response = client.post(
            "/api/projects/",
            json={"name": "Paper Test", "research_field": "CS"}
        )
        project_id = project_response.json()["id"]

        # Create paper
        response = client.post(
            "/api/papers/",
            json={
                "project_id": project_id,
                "title": "Research Paper on ML",
                "abstract": "This paper presents..."
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["title"] == "Research Paper on ML"

    def test_generate_structure(self, client, mock_llm):
        """Test POST /api/papers/generate-structure"""
        # Create project
        project_response = client.post(
            "/api/projects/",
            json={"name": "Structure Test", "research_field": "AI"}
        )
        project_id = project_response.json()["id"]

        # Mock LLM
        mock_llm.generate.return_value = """
        Paper Structure:
        1. Abstract
        2. Introduction
        3. Methodology
        4. Results
        5. Conclusion
        """

        # Generate structure
        response = client.post(
            "/api/papers/generate-structure",
            json={
                "project_id": project_id,
                "paper_type": "conference"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert "structure" in data

    def test_write_section(self, client, mock_llm):
        """Test POST /api/papers/{id}/sections"""
        # Create project and paper
        project_response = client.post(
            "/api/projects/",
            json={"name": "Section Test", "research_field": "CS"}
        )
        project_id = project_response.json()["id"]

        paper_response = client.post(
            "/api/papers/",
            json={
                "project_id": project_id,
                "title": "Test Paper"
            }
        )
        paper_id = paper_response.json()["id"]

        # Mock LLM
        mock_llm.generate.return_value = "Introduction: Machine learning has..."

        # Write section
        response = client.post(
            f"/api/papers/{paper_id}/sections",
            json={
                "section": "introduction",
                "context": {}
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert "content" in data

    def test_polish_paper(self, client, mock_llm):
        """Test POST /api/papers/{id}/polish"""
        # Create paper
        project_response = client.post(
            "/api/projects/",
            json={"name": "Polish Test", "research_field": "AI"}
        )
        project_id = project_response.json()["id"]

        paper_response = client.post(
            "/api/papers/",
            json={
                "project_id": project_id,
                "title": "Test Paper",
                "sections": {"introduction": "Draft intro"}
            }
        )
        paper_id = paper_response.json()["id"]

        # Mock LLM
        mock_llm.generate.return_value = "Polished introduction text..."

        # Polish paper
        response = client.post(
            f"/api/papers/{paper_id}/polish",
            json={"style": "academic"}
        )

        assert response.status_code == 200


@pytest.mark.integration
class TestTaskAPI:
    """Test task management API endpoints"""

    def test_create_task(self, client):
        """Test POST /api/tasks/"""
        # Create project
        project_response = client.post(
            "/api/projects/",
            json={"name": "Task Test", "research_field": "CS"}
        )
        project_id = project_response.json()["id"]

        # Create task
        response = client.post(
            "/api/tasks/",
            json={
                "project_id": project_id,
                "title": "Literature Review",
                "description": "Review existing papers",
                "priority": "high"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["title"] == "Literature Review"

    def test_update_task_status(self, client):
        """Test PUT /api/tasks/{id}/status"""
        # Create project and task
        project_response = client.post(
            "/api/projects/",
            json={"name": "Status Test", "research_field": "AI"}
        )
        project_id = project_response.json()["id"]

        task_response = client.post(
            "/api/tasks/",
            json={
                "project_id": project_id,
                "title": "Test Task",
                "status": "pending"
            }
        )
        task_id = task_response.json()["id"]

        # Update status
        response = client.put(
            f"/api/tasks/{task_id}/status",
            json={"status": "in_progress"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "in_progress"

    def test_get_project_tasks(self, client):
        """Test GET /api/projects/{id}/tasks"""
        # Create project with tasks
        project_response = client.post(
            "/api/projects/",
            json={"name": "Tasks Test", "research_field": "CS"}
        )
        project_id = project_response.json()["id"]

        for i in range(3):
            client.post(
                "/api/tasks/",
                json={
                    "project_id": project_id,
                    "title": f"Task {i}"
                }
            )

        # Get project tasks
        response = client.get(f"/api/projects/{project_id}/tasks")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 3


@pytest.mark.integration
class TestAPIErrorHandling:
    """Test API error handling"""

    def test_not_found_error(self, client):
        """Test 404 error handling"""
        response = client.get("/api/projects/99999")
        assert response.status_code == 404

    def test_validation_error(self, client):
        """Test 422 validation error"""
        response = client.post(
            "/api/projects/",
            json={"name": ""}  # Empty name should fail validation
        )
        assert response.status_code == 422

    def test_invalid_json(self, client):
        """Test handling of invalid JSON"""
        response = client.post(
            "/api/projects/",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code in [400, 422]

    def test_missing_required_field(self, client):
        """Test missing required field"""
        response = client.post(
            "/api/projects/",
            json={"research_field": "CS"}  # Missing name
        )
        assert response.status_code == 422


@pytest.mark.integration
class TestAPIAuthentication:
    """Test API authentication (if implemented)"""

    def test_unauthenticated_access(self, client):
        """Test access without authentication"""
        # This test depends on authentication implementation
        pass

    def test_invalid_token(self, client):
        """Test with invalid authentication token"""
        # This test depends on authentication implementation
        pass


@pytest.mark.integration
class TestAPIPagination:
    """Test API pagination"""

    def test_paginated_projects(self, client):
        """Test pagination on project listing"""
        # Create many projects
        for i in range(25):
            client.post(
                "/api/projects/",
                json={"name": f"Project {i}", "research_field": "CS"}
            )

        # Get first page
        response = client.get("/api/projects/?page=1&page_size=10")

        assert response.status_code == 200
        data = response.json()
        assert len(data) <= 10

    def test_paginated_literature(self, client):
        """Test pagination on literature listing"""
        # Create project
        project_response = client.post(
            "/api/projects/",
            json={"name": "Pagination Test", "research_field": "CS"}
        )
        project_id = project_response.json()["id"]

        # Create many literature entries
        for i in range(30):
            client.post(
                "/api/literature/",
                json={
                    "project_id": project_id,
                    "title": f"Paper {i}",
                    "authors": f"Author {i}",
                    "source": "arxiv"
                }
            )

        # Get paginated results
        response = client.get(
            f"/api/projects/{project_id}/literature?page=1&page_size=10"
        )

        assert response.status_code == 200


@pytest.mark.integration
class TestAPIFiltering:
    """Test API filtering capabilities"""

    def test_filter_projects_by_field(self, client):
        """Test filtering projects by research field"""
        # Create projects in different fields
        client.post("/api/projects/", json={"name": "AI Project", "research_field": "AI"})
        client.post("/api/projects/", json={"name": "CS Project", "research_field": "CS"})
        client.post("/api/projects/", json={"name": "Bio Project", "research_field": "Biology"})

        # Filter by field
        response = client.get("/api/projects/?research_field=AI")

        assert response.status_code == 200
        data = response.json()
        assert all(p["research_field"] == "AI" for p in data)

    def test_filter_tasks_by_status(self, client):
        """Test filtering tasks by status"""
        # Create project with tasks
        project_response = client.post(
            "/api/projects/",
            json={"name": "Filter Test", "research_field": "CS"}
        )
        project_id = project_response.json()["id"]

        client.post("/api/tasks/", json={"project_id": project_id, "title": "Task 1", "status": "pending"})
        client.post("/api/tasks/", json={"project_id": project_id, "title": "Task 2", "status": "completed"})
        client.post("/api/tasks/", json={"project_id": project_id, "title": "Task 3", "status": "pending"})

        # Filter by status
        response = client.get(f"/api/projects/{project_id}/tasks?status=pending")

        assert response.status_code == 200
        data = response.json()
        assert all(t["status"] == "pending" for t in data)
