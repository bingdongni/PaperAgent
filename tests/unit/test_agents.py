"""
Unit tests for agent modules
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from paperagent.agents.boss_agent import BossAgent
from paperagent.agents.literature_agent import LiteratureAgent
from paperagent.agents.experiment_agent import ExperimentAgent
from paperagent.agents.writing_agent import WritingAgent


class TestLiteratureAgent:
    """Test Literature Agent"""

    def test_initialization(self, mock_llm, db_session):
        """Test literature agent initialization"""
        agent = LiteratureAgent(llm=mock_llm, db_session=db_session)
        assert agent.llm is not None
        assert agent.db_session is not None

    @patch('paperagent.tools.literature_collector.ArxivCollector')
    def test_search_literature(self, mock_arxiv, mock_llm, db_session, sample_project_data):
        """Test literature search functionality"""
        # Mock arxiv response
        mock_arxiv.return_value.search.return_value = [
            {
                'title': 'Test Paper 1',
                'authors': ['Author 1'],
                'abstract': 'Test abstract 1',
                'arxiv_id': 'arxiv:2024.0001',
                'published': '2024-01-01',
                'url': 'https://arxiv.org/abs/2024.0001'
            }
        ]

        agent = LiteratureAgent(llm=mock_llm, db_session=db_session)
        result = agent.execute({
            'action': 'search_literature',
            'query': 'machine learning',
            'project_id': 1,
            'max_results': 10
        })

        assert 'total_papers' in result
        assert result['total_papers'] >= 0
        assert 'papers' in result

    def test_analyze_literature_gap(self, mock_llm, db_session, sample_literature_data):
        """Test literature gap analysis"""
        mock_llm.generate.return_value = """
        Based on the literature review, the following gaps are identified:
        1. Limited research on real-time systems
        2. Lack of scalability studies
        3. Need for more empirical validation
        """

        agent = LiteratureAgent(llm=mock_llm, db_session=db_session)
        result = agent.execute({
            'action': 'analyze_gap',
            'project_id': 1,
            'literature_ids': [1, 2, 3]
        })

        assert 'gaps' in result
        assert len(result['gaps']) > 0

    def test_cluster_literature(self, mock_llm, db_session, sample_literature_data):
        """Test literature clustering"""
        agent = LiteratureAgent(llm=mock_llm, db_session=db_session)
        result = agent.execute({
            'action': 'cluster_literature',
            'project_id': 1,
            'num_clusters': 3
        })

        assert 'clusters' in result
        assert isinstance(result['clusters'], list)

    def test_summarize_literature(self, mock_llm, db_session, sample_literature_data):
        """Test literature summarization"""
        mock_llm.generate.return_value = """
        Summary of literature:
        The reviewed papers focus on machine learning applications in various domains.
        Key findings include improved accuracy through ensemble methods and
        the importance of feature engineering.
        """

        agent = LiteratureAgent(llm=mock_llm, db_session=db_session)
        result = agent.execute({
            'action': 'summarize',
            'literature_id': 1
        })

        assert 'summary' in result
        assert len(result['summary']) > 0


class TestExperimentAgent:
    """Test Experiment Agent"""

    def test_initialization(self, mock_llm, db_session):
        """Test experiment agent initialization"""
        agent = ExperimentAgent(llm=mock_llm, db_session=db_session)
        assert agent.llm is not None
        assert agent.db_session is not None

    def test_design_experiment(self, mock_llm, db_session, sample_project_data):
        """Test experiment design"""
        mock_llm.generate.return_value = """
        Experiment Design:
        1. Hypothesis: Method A performs better than Method B
        2. Variables: Algorithm type (independent), Accuracy (dependent)
        3. Methodology: Controlled experiment with 5-fold cross-validation
        4. Expected outcomes: 5% improvement in accuracy
        """

        agent = ExperimentAgent(llm=mock_llm, db_session=db_session)
        result = agent.execute({
            'action': 'design_experiment',
            'project_id': 1,
            'research_question': 'Does method A improve accuracy?',
            'constraints': {'budget': 'low', 'time': '2 weeks'}
        })

        assert 'design' in result
        assert 'methodology' in result

    def test_analyze_data(self, mock_llm, db_session, sample_dataframe):
        """Test data analysis"""
        agent = ExperimentAgent(llm=mock_llm, db_session=db_session)
        result = agent.execute({
            'action': 'analyze_data',
            'experiment_id': 1,
            'data': sample_dataframe,
            'analysis_type': 'descriptive'
        })

        assert 'statistics' in result
        assert 'analysis' in result

    def test_statistical_test(self, mock_llm, db_session, sample_dataframe):
        """Test statistical testing"""
        agent = ExperimentAgent(llm=mock_llm, db_session=db_session)
        result = agent.execute({
            'action': 'statistical_test',
            'test_type': 'ttest',
            'data': sample_dataframe,
            'group_column': 'group',
            'value_column': 'value'
        })

        assert 'test_result' in result
        assert 'p_value' in result or 'statistic' in result

    def test_visualize_results(self, mock_llm, db_session, sample_dataframe):
        """Test results visualization"""
        agent = ExperimentAgent(llm=mock_llm, db_session=db_session)
        result = agent.execute({
            'action': 'visualize',
            'experiment_id': 1,
            'data': sample_dataframe,
            'plot_type': 'bar'
        })

        assert 'visualization' in result or 'plot_path' in result


class TestWritingAgent:
    """Test Writing Agent"""

    def test_initialization(self, mock_llm, db_session):
        """Test writing agent initialization"""
        agent = WritingAgent(llm=mock_llm, db_session=db_session)
        assert agent.llm is not None
        assert agent.db_session is not None

    def test_generate_structure(self, mock_llm, db_session, sample_project_data):
        """Test paper structure generation"""
        mock_llm.generate.return_value = """
        Paper Structure:
        1. Title: A Novel Approach to Machine Learning
        2. Abstract
        3. Introduction
        4. Related Work
        5. Methodology
        6. Experiments
        7. Results
        8. Discussion
        9. Conclusion
        10. References
        """

        agent = WritingAgent(llm=mock_llm, db_session=db_session)
        result = agent.execute({
            'action': 'generate_structure',
            'project_id': 1,
            'paper_type': 'conference'
        })

        assert 'structure' in result
        assert isinstance(result['structure'], (list, dict))

    def test_write_section(self, mock_llm, db_session, sample_project_data):
        """Test section writing"""
        mock_llm.generate.return_value = """
        # Introduction

        Machine learning has revolutionized various domains of computer science.
        In this paper, we propose a novel approach that addresses the limitations
        of existing methods. Our contributions are threefold: (1) we introduce
        a new algorithm, (2) we provide theoretical analysis, and (3) we
        demonstrate empirical improvements on benchmark datasets.
        """

        agent = WritingAgent(llm=mock_llm, db_session=db_session)
        result = agent.execute({
            'action': 'write_section',
            'paper_id': 1,
            'section': 'introduction',
            'context': {'research_gap': 'Limited work on X', 'contributions': ['A', 'B', 'C']}
        })

        assert 'content' in result
        assert len(result['content']) > 0

    def test_polish_text(self, mock_llm, db_session):
        """Test text polishing"""
        original_text = "This paper present a new method for machine learning."
        polished_text = "This paper presents a novel method for machine learning applications."

        mock_llm.generate.return_value = polished_text

        agent = WritingAgent(llm=mock_llm, db_session=db_session)
        result = agent.execute({
            'action': 'polish',
            'text': original_text,
            'style': 'academic'
        })

        assert 'polished_text' in result
        assert result['polished_text'] != original_text

    def test_check_grammar(self, mock_llm, db_session):
        """Test grammar checking"""
        agent = WritingAgent(llm=mock_llm, db_session=db_session)
        result = agent.execute({
            'action': 'check_grammar',
            'text': 'This are a test sentence with error.'
        })

        assert 'errors' in result or 'suggestions' in result

    def test_generate_abstract(self, mock_llm, db_session, sample_project_data):
        """Test abstract generation"""
        mock_llm.generate.return_value = """
        This paper presents a novel machine learning approach that improves
        accuracy by 15% over state-of-the-art methods. We introduce a new
        algorithm based on ensemble learning and demonstrate its effectiveness
        on multiple benchmark datasets. Our results show significant improvements
        in both accuracy and computational efficiency.
        """

        agent = WritingAgent(llm=mock_llm, db_session=db_session)
        result = agent.execute({
            'action': 'generate_abstract',
            'paper_id': 1,
            'max_words': 200
        })

        assert 'abstract' in result
        assert len(result['abstract'].split()) <= 250


class TestBossAgent:
    """Test Boss Agent"""

    def test_initialization(self, mock_llm, db_session):
        """Test boss agent initialization"""
        agent = BossAgent(llm=mock_llm, db_session=db_session)
        assert agent.llm is not None
        assert agent.db_session is not None

    def test_create_project(self, mock_llm, db_session):
        """Test project creation"""
        agent = BossAgent(llm=mock_llm, db_session=db_session)
        result = agent.execute({
            'action': 'create_project',
            'name': 'Test Research Project',
            'research_field': 'Computer Science',
            'keywords': ['machine learning', 'AI']
        })

        assert 'project_id' in result
        assert 'status' in result

    def test_decompose_task(self, mock_llm, db_session, sample_project_data):
        """Test task decomposition"""
        mock_llm.generate.return_value = """
        Task Decomposition:
        1. Literature Review (Priority: High, Duration: 2 weeks)
        2. Experiment Design (Priority: High, Duration: 1 week)
        3. Data Collection (Priority: Medium, Duration: 1 week)
        4. Analysis (Priority: High, Duration: 2 weeks)
        5. Paper Writing (Priority: High, Duration: 3 weeks)
        """

        agent = BossAgent(llm=mock_llm, db_session=db_session)
        result = agent.execute({
            'action': 'decompose_task',
            'project_id': 1,
            'main_goal': 'Complete research paper on ML'
        })

        assert 'tasks' in result
        assert len(result['tasks']) > 0

    def test_assign_task(self, mock_llm, db_session, sample_project_data):
        """Test task assignment"""
        agent = BossAgent(llm=mock_llm, db_session=db_session)
        result = agent.execute({
            'action': 'assign_task',
            'task_id': 1,
            'agent_type': 'literature'
        })

        assert 'assigned' in result or 'status' in result

    def test_monitor_progress(self, mock_llm, db_session, sample_project_data):
        """Test progress monitoring"""
        agent = BossAgent(llm=mock_llm, db_session=db_session)
        result = agent.execute({
            'action': 'monitor_progress',
            'project_id': 1
        })

        assert 'progress' in result
        assert 'tasks' in result or 'completion_rate' in result

    def test_quality_check(self, mock_llm, db_session, sample_project_data):
        """Test quality checking"""
        mock_llm.generate.return_value = """
        Quality Check Results:
        - Completeness: 90%
        - Accuracy: Good
        - Issues: Minor formatting inconsistencies
        - Recommendations: Review section 3.2
        """

        agent = BossAgent(llm=mock_llm, db_session=db_session)
        result = agent.execute({
            'action': 'quality_check',
            'project_id': 1,
            'check_type': 'comprehensive'
        })

        assert 'quality_score' in result or 'issues' in result

    @patch('paperagent.agents.literature_agent.LiteratureAgent')
    @patch('paperagent.agents.experiment_agent.ExperimentAgent')
    @patch('paperagent.agents.writing_agent.WritingAgent')
    def test_execute_workflow(self, mock_writing, mock_experiment, mock_literature,
                             mock_llm, db_session, sample_project_data):
        """Test complete workflow execution"""
        # Mock sub-agent responses
        mock_literature.return_value.execute.return_value = {
            'status': 'completed',
            'total_papers': 20
        }
        mock_experiment.return_value.execute.return_value = {
            'status': 'completed',
            'experiment_id': 1
        }
        mock_writing.return_value.execute.return_value = {
            'status': 'completed',
            'paper_id': 1
        }

        agent = BossAgent(llm=mock_llm, db_session=db_session)
        result = agent.execute({
            'action': 'execute_workflow',
            'project_id': 1,
            'workflow_type': 'full'
        })

        assert 'status' in result
        assert 'completed_tasks' in result or 'progress' in result


class TestAgentIntegration:
    """Test agent integration and communication"""

    def test_literature_to_experiment_flow(self, mock_llm, db_session,
                                          sample_project_data, sample_literature_data):
        """Test data flow from literature agent to experiment agent"""
        lit_agent = LiteratureAgent(llm=mock_llm, db_session=db_session)
        exp_agent = ExperimentAgent(llm=mock_llm, db_session=db_session)

        # Literature agent finds gaps
        lit_result = lit_agent.execute({
            'action': 'analyze_gap',
            'project_id': 1,
            'literature_ids': [1, 2]
        })

        # Experiment agent uses gaps to design experiment
        exp_result = exp_agent.execute({
            'action': 'design_experiment',
            'project_id': 1,
            'research_gaps': lit_result.get('gaps', [])
        })

        assert exp_result is not None

    def test_experiment_to_writing_flow(self, mock_llm, db_session,
                                       sample_project_data, sample_dataframe):
        """Test data flow from experiment agent to writing agent"""
        exp_agent = ExperimentAgent(llm=mock_llm, db_session=db_session)
        write_agent = WritingAgent(llm=mock_llm, db_session=db_session)

        # Experiment agent analyzes data
        exp_result = exp_agent.execute({
            'action': 'analyze_data',
            'experiment_id': 1,
            'data': sample_dataframe,
            'analysis_type': 'descriptive'
        })

        # Writing agent uses results to write section
        write_result = write_agent.execute({
            'action': 'write_section',
            'paper_id': 1,
            'section': 'results',
            'experiment_results': exp_result
        })

        assert write_result is not None
        assert 'content' in write_result


class TestAgentErrorHandling:
    """Test agent error handling"""

    def test_invalid_action(self, mock_llm, db_session):
        """Test handling of invalid action"""
        agent = LiteratureAgent(llm=mock_llm, db_session=db_session)

        with pytest.raises((ValueError, KeyError)):
            agent.execute({'action': 'invalid_action'})

    def test_missing_parameters(self, mock_llm, db_session):
        """Test handling of missing required parameters"""
        agent = LiteratureAgent(llm=mock_llm, db_session=db_session)

        with pytest.raises((ValueError, KeyError, TypeError)):
            agent.execute({'action': 'search_literature'})  # Missing query

    def test_llm_failure_handling(self, db_session):
        """Test handling of LLM API failures"""
        mock_failing_llm = Mock()
        mock_failing_llm.generate.side_effect = Exception("API Error")

        agent = WritingAgent(llm=mock_failing_llm, db_session=db_session)

        with pytest.raises(Exception):
            agent.execute({
                'action': 'write_section',
                'paper_id': 1,
                'section': 'introduction'
            })

    def test_database_error_handling(self, mock_llm):
        """Test handling of database errors"""
        mock_failing_db = Mock()
        mock_failing_db.query.side_effect = Exception("Database Error")

        agent = BossAgent(llm=mock_llm, db_session=mock_failing_db)

        with pytest.raises(Exception):
            agent.execute({
                'action': 'create_project',
                'name': 'Test Project'
            })
