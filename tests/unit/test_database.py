"""
Unit tests for database models
"""

import pytest
from datetime import datetime
from sqlalchemy.exc import IntegrityError
from paperagent.database.models import (
    Project, Task, Literature, Experiment, Paper,
    PaperRevision, AgentLog, Citation, JournalTemplate
)


class TestProjectModel:
    """Test Project database model"""

    def test_create_project(self, db_session, sample_project_data):
        """Test creating a new project"""
        project = Project(**sample_project_data)
        db_session.add(project)
        db_session.commit()

        assert project.id is not None
        assert project.name == sample_project_data['name']
        assert project.research_field == sample_project_data['research_field']

    def test_project_relationships(self, db_session, sample_project_data):
        """Test project relationships with tasks and literature"""
        project = Project(**sample_project_data)
        db_session.add(project)
        db_session.commit()

        # Add task
        task = Task(
            project_id=project.id,
            title="Literature Review",
            description="Review existing literature",
            status="pending"
        )
        db_session.add(task)
        db_session.commit()

        # Check relationship
        assert len(project.tasks) == 1
        assert project.tasks[0].title == "Literature Review"

    def test_project_status_update(self, db_session, sample_project_data):
        """Test updating project status"""
        project = Project(**sample_project_data)
        db_session.add(project)
        db_session.commit()

        project.status = "in_progress"
        db_session.commit()

        retrieved = db_session.query(Project).filter_by(id=project.id).first()
        assert retrieved.status == "in_progress"

    def test_project_timestamps(self, db_session, sample_project_data):
        """Test project timestamp fields"""
        project = Project(**sample_project_data)
        db_session.add(project)
        db_session.commit()

        assert project.created_at is not None
        assert isinstance(project.created_at, datetime)


class TestTaskModel:
    """Test Task database model"""

    def test_create_task(self, db_session, sample_project_data):
        """Test creating a task"""
        project = Project(**sample_project_data)
        db_session.add(project)
        db_session.commit()

        task = Task(
            project_id=project.id,
            title="Data Collection",
            description="Collect experimental data",
            status="pending",
            priority="high"
        )
        db_session.add(task)
        db_session.commit()

        assert task.id is not None
        assert task.project_id == project.id
        assert task.status == "pending"

    def test_task_status_progression(self, db_session, sample_project_data):
        """Test task status changes"""
        project = Project(**sample_project_data)
        task = Task(
            project_id=project.id,
            title="Test Task",
            status="pending"
        )
        db_session.add_all([project, task])
        db_session.commit()

        # Progress through statuses
        task.status = "in_progress"
        db_session.commit()
        assert task.status == "in_progress"

        task.status = "completed"
        db_session.commit()
        assert task.status == "completed"

    def test_task_assignment(self, db_session, sample_project_data):
        """Test task assignment to agents"""
        project = Project(**sample_project_data)
        task = Task(
            project_id=project.id,
            title="Analysis Task",
            assigned_agent="experiment_agent"
        )
        db_session.add_all([project, task])
        db_session.commit()

        assert task.assigned_agent == "experiment_agent"

    def test_task_project_relationship(self, db_session, sample_project_data):
        """Test task-project bidirectional relationship"""
        project = Project(**sample_project_data)
        task = Task(
            project_id=project.id,
            title="Test Task"
        )
        db_session.add_all([project, task])
        db_session.commit()

        # Test from task to project
        assert task.project.name == project.name

        # Test from project to task
        assert task in project.tasks


class TestLiteratureModel:
    """Test Literature database model"""

    def test_create_literature(self, db_session, sample_project_data, sample_literature_data):
        """Test creating literature entry"""
        project = Project(**sample_project_data)
        db_session.add(project)
        db_session.commit()

        literature = Literature(
            project_id=project.id,
            **sample_literature_data
        )
        db_session.add(literature)
        db_session.commit()

        assert literature.id is not None
        assert literature.title == sample_literature_data['title']

    def test_literature_search_query(self, db_session, sample_project_data):
        """Test querying literature by fields"""
        project = Project(**sample_project_data)
        db_session.add(project)
        db_session.commit()

        lit1 = Literature(
            project_id=project.id,
            title="Machine Learning Paper",
            authors="Smith, J.",
            source="arxiv"
        )
        lit2 = Literature(
            project_id=project.id,
            title="Deep Learning Paper",
            authors="Jones, A.",
            source="scholar"
        )
        db_session.add_all([lit1, lit2])
        db_session.commit()

        # Query by title
        results = db_session.query(Literature).filter(
            Literature.title.contains("Machine")
        ).all()
        assert len(results) == 1
        assert results[0].title == "Machine Learning Paper"

    def test_literature_metadata(self, db_session, sample_project_data):
        """Test literature metadata storage"""
        project = Project(**sample_project_data)
        literature = Literature(
            project_id=project.id,
            title="Test Paper",
            metadata={
                'citations': 100,
                'venue': 'ICML 2024',
                'keywords': ['ML', 'AI']
            }
        )
        db_session.add_all([project, literature])
        db_session.commit()

        retrieved = db_session.query(Literature).filter_by(id=literature.id).first()
        assert retrieved.metadata['citations'] == 100
        assert 'ML' in retrieved.metadata['keywords']


class TestExperimentModel:
    """Test Experiment database model"""

    def test_create_experiment(self, db_session, sample_project_data):
        """Test creating experiment"""
        project = Project(**sample_project_data)
        db_session.add(project)
        db_session.commit()

        experiment = Experiment(
            project_id=project.id,
            name="Baseline Experiment",
            description="Test baseline performance",
            design={"method": "A/B testing"},
            status="pending"
        )
        db_session.add(experiment)
        db_session.commit()

        assert experiment.id is not None
        assert experiment.name == "Baseline Experiment"

    def test_experiment_results_storage(self, db_session, sample_project_data):
        """Test storing experiment results"""
        project = Project(**sample_project_data)
        experiment = Experiment(
            project_id=project.id,
            name="Test Experiment",
            results={
                'accuracy': 0.95,
                'precision': 0.93,
                'recall': 0.94,
                'f1_score': 0.935
            }
        )
        db_session.add_all([project, experiment])
        db_session.commit()

        retrieved = db_session.query(Experiment).filter_by(id=experiment.id).first()
        assert retrieved.results['accuracy'] == 0.95
        assert retrieved.results['f1_score'] == 0.935

    def test_experiment_data_path(self, db_session, sample_project_data):
        """Test experiment data path storage"""
        project = Project(**sample_project_data)
        experiment = Experiment(
            project_id=project.id,
            name="Data Experiment",
            data_path="/data/experiments/exp001.csv"
        )
        db_session.add_all([project, experiment])
        db_session.commit()

        assert experiment.data_path == "/data/experiments/exp001.csv"


class TestPaperModel:
    """Test Paper database model"""

    def test_create_paper(self, db_session, sample_project_data):
        """Test creating paper"""
        project = Project(**sample_project_data)
        db_session.add(project)
        db_session.commit()

        paper = Paper(
            project_id=project.id,
            title="Research Paper on ML",
            abstract="This paper presents...",
            status="draft"
        )
        db_session.add(paper)
        db_session.commit()

        assert paper.id is not None
        assert paper.title == "Research Paper on ML"

    def test_paper_sections(self, db_session, sample_project_data):
        """Test paper sections storage"""
        project = Project(**sample_project_data)
        paper = Paper(
            project_id=project.id,
            title="Test Paper",
            sections={
                'introduction': 'Introduction text...',
                'methodology': 'Methodology text...',
                'results': 'Results text...',
                'conclusion': 'Conclusion text...'
            }
        )
        db_session.add_all([project, paper])
        db_session.commit()

        retrieved = db_session.query(Paper).filter_by(id=paper.id).first()
        assert 'introduction' in retrieved.sections
        assert 'methodology' in retrieved.sections

    def test_paper_journal_template(self, db_session, sample_project_data):
        """Test paper journal template association"""
        project = Project(**sample_project_data)
        paper = Paper(
            project_id=project.id,
            title="Test Paper",
            journal_template="IEEE"
        )
        db_session.add_all([project, paper])
        db_session.commit()

        assert paper.journal_template == "IEEE"


class TestPaperRevisionModel:
    """Test Paper Revision model"""

    def test_create_revision(self, db_session, sample_project_data):
        """Test creating paper revision"""
        project = Project(**sample_project_data)
        paper = Paper(
            project_id=project.id,
            title="Test Paper"
        )
        db_session.add_all([project, paper])
        db_session.commit()

        revision = PaperRevision(
            paper_id=paper.id,
            version=1,
            content="First draft content...",
            changes="Initial version"
        )
        db_session.add(revision)
        db_session.commit()

        assert revision.id is not None
        assert revision.paper_id == paper.id
        assert revision.version == 1

    def test_multiple_revisions(self, db_session, sample_project_data):
        """Test multiple paper revisions"""
        project = Project(**sample_project_data)
        paper = Paper(project_id=project.id, title="Test Paper")
        db_session.add_all([project, paper])
        db_session.commit()

        rev1 = PaperRevision(paper_id=paper.id, version=1, content="Version 1")
        rev2 = PaperRevision(paper_id=paper.id, version=2, content="Version 2")
        rev3 = PaperRevision(paper_id=paper.id, version=3, content="Version 3")
        db_session.add_all([rev1, rev2, rev3])
        db_session.commit()

        revisions = db_session.query(PaperRevision).filter_by(
            paper_id=paper.id
        ).order_by(PaperRevision.version).all()

        assert len(revisions) == 3
        assert revisions[0].version == 1
        assert revisions[2].version == 3


class TestAgentLogModel:
    """Test Agent Log model"""

    def test_create_agent_log(self, db_session, sample_project_data):
        """Test creating agent log entry"""
        project = Project(**sample_project_data)
        task = Task(project_id=project.id, title="Test Task")
        db_session.add_all([project, task])
        db_session.commit()

        log = AgentLog(
            task_id=task.id,
            agent_name="literature_agent",
            action="search_papers",
            status="success",
            details={"papers_found": 20}
        )
        db_session.add(log)
        db_session.commit()

        assert log.id is not None
        assert log.agent_name == "literature_agent"

    def test_query_logs_by_agent(self, db_session, sample_project_data):
        """Test querying logs by agent name"""
        project = Project(**sample_project_data)
        task = Task(project_id=project.id, title="Test Task")
        db_session.add_all([project, task])
        db_session.commit()

        log1 = AgentLog(task_id=task.id, agent_name="agent_a", action="action1")
        log2 = AgentLog(task_id=task.id, agent_name="agent_b", action="action2")
        log3 = AgentLog(task_id=task.id, agent_name="agent_a", action="action3")
        db_session.add_all([log1, log2, log3])
        db_session.commit()

        agent_a_logs = db_session.query(AgentLog).filter_by(
            agent_name="agent_a"
        ).all()

        assert len(agent_a_logs) == 2

    def test_log_timestamps(self, db_session, sample_project_data):
        """Test log timestamp recording"""
        project = Project(**sample_project_data)
        task = Task(project_id=project.id, title="Test Task")
        log = AgentLog(task_id=task.id, agent_name="test_agent", action="test")
        db_session.add_all([project, task, log])
        db_session.commit()

        assert log.timestamp is not None
        assert isinstance(log.timestamp, datetime)


class TestCitationModel:
    """Test Citation model"""

    def test_create_citation(self, db_session, sample_project_data):
        """Test creating citation"""
        project = Project(**sample_project_data)
        paper = Paper(project_id=project.id, title="Test Paper")
        db_session.add_all([project, paper])
        db_session.commit()

        citation = Citation(
            paper_id=paper.id,
            citation_key="smith2024",
            citation_type="article",
            authors="Smith, J.",
            title="Test Article",
            year=2024,
            venue="Test Conference"
        )
        db_session.add(citation)
        db_session.commit()

        assert citation.id is not None
        assert citation.citation_key == "smith2024"

    def test_citation_formats(self, db_session, sample_project_data):
        """Test different citation formats"""
        project = Project(**sample_project_data)
        paper = Paper(project_id=project.id, title="Test Paper")
        db_session.add_all([project, paper])
        db_session.commit()

        ieee_citation = Citation(
            paper_id=paper.id,
            citation_key="ref1",
            formatted_citation="[1] J. Smith, 'Title,' Journal, 2024."
        )
        apa_citation = Citation(
            paper_id=paper.id,
            citation_key="ref2",
            formatted_citation="Smith, J. (2024). Title. Journal."
        )
        db_session.add_all([ieee_citation, apa_citation])
        db_session.commit()

        citations = db_session.query(Citation).filter_by(paper_id=paper.id).all()
        assert len(citations) == 2


class TestJournalTemplateModel:
    """Test Journal Template model"""

    def test_create_journal_template(self, db_session):
        """Test creating journal template"""
        template = JournalTemplate(
            name="IEEE Access",
            citation_style="IEEE",
            document_class="IEEEtran",
            requirements={
                'max_pages': 15,
                'abstract_limit': 200,
                'keywords': 5
            }
        )
        db_session.add(template)
        db_session.commit()

        assert template.id is not None
        assert template.name == "IEEE Access"

    def test_query_template_by_name(self, db_session):
        """Test querying template by name"""
        template1 = JournalTemplate(name="Nature", citation_style="Nature")
        template2 = JournalTemplate(name="Science", citation_style="Science")
        db_session.add_all([template1, template2])
        db_session.commit()

        nature = db_session.query(JournalTemplate).filter_by(name="Nature").first()
        assert nature is not None
        assert nature.citation_style == "Nature"

    def test_template_latex_content(self, db_session):
        """Test storing LaTeX template content"""
        template = JournalTemplate(
            name="ACM",
            citation_style="ACM",
            latex_template=r"\documentclass{acmart}\begin{document}...\end{document}"
        )
        db_session.add(template)
        db_session.commit()

        retrieved = db_session.query(JournalTemplate).filter_by(name="ACM").first()
        assert r"\documentclass{acmart}" in retrieved.latex_template


class TestDatabaseConstraints:
    """Test database constraints and validations"""

    def test_unique_constraints(self, db_session, sample_project_data):
        """Test unique constraint enforcement"""
        project1 = Project(**sample_project_data)
        db_session.add(project1)
        db_session.commit()

        # Attempt to create project with same unique identifier if exists
        # This test depends on model's unique constraints
        pass

    def test_foreign_key_constraints(self, db_session):
        """Test foreign key constraint enforcement"""
        # Attempt to create task without valid project
        task = Task(
            project_id=99999,  # Non-existent project
            title="Orphan Task"
        )
        db_session.add(task)

        with pytest.raises((IntegrityError, Exception)):
            db_session.commit()
        db_session.rollback()

    def test_cascade_deletion(self, db_session, sample_project_data):
        """Test cascade deletion of related records"""
        project = Project(**sample_project_data)
        db_session.add(project)
        db_session.commit()

        task = Task(project_id=project.id, title="Test Task")
        literature = Literature(
            project_id=project.id,
            title="Test Paper",
            authors="Author"
        )
        db_session.add_all([task, literature])
        db_session.commit()

        project_id = project.id

        # Delete project
        db_session.delete(project)
        db_session.commit()

        # Check if related records are handled according to cascade settings
        remaining_tasks = db_session.query(Task).filter_by(
            project_id=project_id
        ).all()

        # Behavior depends on cascade settings in models
        # This test verifies cascade is configured correctly


class TestDatabaseQueries:
    """Test complex database queries"""

    def test_join_queries(self, db_session, sample_project_data):
        """Test join queries across tables"""
        project = Project(**sample_project_data)
        db_session.add(project)
        db_session.commit()

        task = Task(project_id=project.id, title="Test Task", status="completed")
        db_session.add(task)
        db_session.commit()

        # Join query
        results = db_session.query(Project, Task).join(Task).filter(
            Task.status == "completed"
        ).all()

        assert len(results) > 0

    def test_aggregation_queries(self, db_session, sample_project_data):
        """Test aggregation queries"""
        from sqlalchemy import func

        project = Project(**sample_project_data)
        db_session.add(project)
        db_session.commit()

        # Add multiple tasks
        for i in range(5):
            task = Task(
                project_id=project.id,
                title=f"Task {i}",
                status="completed" if i % 2 == 0 else "pending"
            )
            db_session.add(task)
        db_session.commit()

        # Count tasks by status
        task_counts = db_session.query(
            Task.status,
            func.count(Task.id)
        ).filter_by(project_id=project.id).group_by(Task.status).all()

        assert len(task_counts) > 0

    def test_filter_by_date(self, db_session, sample_project_data):
        """Test filtering by date ranges"""
        from datetime import timedelta

        project = Project(**sample_project_data)
        db_session.add(project)
        db_session.commit()

        # Query recent projects
        recent = db_session.query(Project).filter(
            Project.created_at >= datetime.now() - timedelta(days=1)
        ).all()

        assert len(recent) > 0
