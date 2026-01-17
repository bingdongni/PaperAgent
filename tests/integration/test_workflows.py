"""
Integration tests for end-to-end workflows
"""

import pytest
from unittest.mock import Mock, patch
from paperagent.agents.boss_agent import BossAgent
from paperagent.agents.literature_agent import LiteratureAgent
from paperagent.agents.experiment_agent import ExperimentAgent
from paperagent.agents.writing_agent import WritingAgent


@pytest.mark.integration
class TestCompleteResearchWorkflow:
    """Test complete research workflow from start to finish"""

    @patch('paperagent.tools.literature_collector.ArxivCollector')
    def test_full_workflow(self, mock_arxiv, mock_llm, db_session):
        """Test complete workflow: project creation -> literature -> experiment -> writing"""

        # Mock external API calls
        mock_arxiv.return_value.search.return_value = [
            {
                'title': 'Related Paper 1',
                'authors': ['Author A'],
                'abstract': 'Research on topic X',
                'arxiv_id': 'arxiv:2024.0001',
                'published': '2024-01-01',
                'url': 'https://arxiv.org/abs/2024.0001'
            }
        ]

        # Step 1: Create project
        boss = BossAgent(llm=mock_llm, db_session=db_session)
        project_result = boss.execute({
            'action': 'create_project',
            'name': 'ML Research Project',
            'research_field': 'Computer Science',
            'keywords': ['machine learning', 'neural networks']
        })

        assert 'project_id' in project_result
        project_id = project_result['project_id']

        # Step 2: Literature review
        lit_agent = LiteratureAgent(llm=mock_llm, db_session=db_session)

        # Search literature
        search_result = lit_agent.execute({
            'action': 'search_literature',
            'query': 'machine learning neural networks',
            'project_id': project_id,
            'max_results': 20
        })
        assert 'total_papers' in search_result

        # Analyze gaps
        mock_llm.generate.return_value = """
        Research gaps identified:
        1. Limited work on real-time processing
        2. Need for better scalability
        3. Lack of empirical validation
        """

        gap_result = lit_agent.execute({
            'action': 'analyze_gap',
            'project_id': project_id
        })
        assert 'gaps' in gap_result

        # Step 3: Design experiment
        exp_agent = ExperimentAgent(llm=mock_llm, db_session=db_session)

        mock_llm.generate.return_value = """
        Experiment Design:
        Hypothesis: New method improves performance
        Variables: Algorithm type, Dataset size
        Methodology: 5-fold cross-validation
        """

        exp_design = exp_agent.execute({
            'action': 'design_experiment',
            'project_id': project_id,
            'research_question': 'Does method A improve accuracy?'
        })
        assert 'design' in exp_design or 'methodology' in exp_design

        # Step 4: Write paper
        write_agent = WritingAgent(llm=mock_llm, db_session=db_session)

        # Generate structure
        mock_llm.generate.return_value = """
        Paper Structure:
        1. Abstract
        2. Introduction
        3. Related Work
        4. Methodology
        5. Experiments
        6. Results
        7. Conclusion
        """

        structure = write_agent.execute({
            'action': 'generate_structure',
            'project_id': project_id,
            'paper_type': 'conference'
        })
        assert 'structure' in structure

        # Write introduction
        mock_llm.generate.return_value = "Introduction: Machine learning has..."

        intro = write_agent.execute({
            'action': 'write_section',
            'paper_id': 1,
            'section': 'introduction',
            'context': gap_result
        })
        assert 'content' in intro

    def test_iterative_refinement_workflow(self, mock_llm, db_session):
        """Test workflow with iterative refinement and quality checks"""

        boss = BossAgent(llm=mock_llm, db_session=db_session)

        # Create project
        project = boss.execute({
            'action': 'create_project',
            'name': 'Iterative Research',
            'research_field': 'AI'
        })
        project_id = project['project_id']

        # Initial quality check
        mock_llm.generate.return_value = """
        Quality Check:
        - Completeness: 60%
        - Issues: Missing methodology details
        - Recommendations: Add more experimental details
        """

        initial_check = boss.execute({
            'action': 'quality_check',
            'project_id': project_id
        })

        assert 'quality_score' in initial_check or 'issues' in initial_check

        # Refinement iteration
        write_agent = WritingAgent(llm=mock_llm, db_session=db_session)

        mock_llm.generate.return_value = "Enhanced methodology section..."

        refinement = write_agent.execute({
            'action': 'write_section',
            'paper_id': 1,
            'section': 'methodology',
            'context': initial_check
        })

        assert 'content' in refinement

        # Final quality check
        mock_llm.generate.return_value = """
        Quality Check:
        - Completeness: 95%
        - Issues: None
        - Status: Ready for submission
        """

        final_check = boss.execute({
            'action': 'quality_check',
            'project_id': project_id
        })


@pytest.mark.integration
class TestLiteratureWorkflow:
    """Test literature review workflow"""

    @patch('paperagent.tools.literature_collector.ArxivCollector')
    @patch('paperagent.tools.literature_collector.ScholarCollector')
    def test_multi_source_literature_collection(self, mock_scholar, mock_arxiv,
                                               mock_llm, db_session):
        """Test collecting literature from multiple sources"""

        # Mock responses
        mock_arxiv.return_value.search.return_value = [
            {'title': 'ArXiv Paper', 'authors': ['A'], 'abstract': 'Abstract A',
             'arxiv_id': '2024.0001', 'published': '2024-01-01',
             'url': 'https://arxiv.org/abs/2024.0001'}
        ]

        mock_scholar.return_value.search.return_value = [
            {'title': 'Scholar Paper', 'authors': ['B'], 'abstract': 'Abstract B',
             'year': '2024', 'citations': 10}
        ]

        boss = BossAgent(llm=mock_llm, db_session=db_session)
        project = boss.execute({
            'action': 'create_project',
            'name': 'Literature Test',
            'research_field': 'CS'
        })

        lit_agent = LiteratureAgent(llm=mock_llm, db_session=db_session)

        # Collect from arXiv
        arxiv_results = lit_agent.execute({
            'action': 'search_literature',
            'query': 'machine learning',
            'project_id': project['project_id'],
            'source': 'arxiv',
            'max_results': 10
        })

        # Collect from Scholar
        scholar_results = lit_agent.execute({
            'action': 'search_literature',
            'query': 'machine learning',
            'project_id': project['project_id'],
            'source': 'scholar',
            'max_results': 10
        })

        assert arxiv_results['total_papers'] >= 0
        assert scholar_results['total_papers'] >= 0

    def test_literature_analysis_pipeline(self, mock_llm, db_session, sample_project_data):
        """Test literature analysis pipeline: collect -> cluster -> summarize -> gap analysis"""

        from paperagent.database.models import Project, Literature

        # Setup project with literature
        project = Project(**sample_project_data)
        db_session.add(project)
        db_session.commit()

        # Add sample literature
        papers = [
            Literature(
                project_id=project.id,
                title=f"Paper {i}",
                authors=f"Author {i}",
                abstract=f"Abstract about topic {i % 3}",
                source="arxiv"
            )
            for i in range(10)
        ]
        db_session.add_all(papers)
        db_session.commit()

        lit_agent = LiteratureAgent(llm=mock_llm, db_session=db_session)

        # Cluster literature
        cluster_result = lit_agent.execute({
            'action': 'cluster_literature',
            'project_id': project.id,
            'num_clusters': 3
        })
        assert 'clusters' in cluster_result

        # Summarize clusters
        mock_llm.generate.return_value = "Summary of cluster 1: Papers focus on..."

        summary = lit_agent.execute({
            'action': 'summarize',
            'project_id': project.id
        })
        assert 'summary' in summary

        # Gap analysis
        mock_llm.generate.return_value = """
        Gaps identified:
        1. Limited empirical studies
        2. Need for theoretical foundations
        """

        gaps = lit_agent.execute({
            'action': 'analyze_gap',
            'project_id': project.id
        })
        assert 'gaps' in gaps


@pytest.mark.integration
class TestExperimentWorkflow:
    """Test experiment workflow"""

    def test_experiment_design_to_analysis(self, mock_llm, db_session,
                                          sample_project_data, sample_dataframe):
        """Test workflow from experiment design to data analysis"""

        from paperagent.database.models import Project, Experiment

        project = Project(**sample_project_data)
        db_session.add(project)
        db_session.commit()

        exp_agent = ExperimentAgent(llm=mock_llm, db_session=db_session)

        # Design experiment
        mock_llm.generate.return_value = """
        Design:
        Hypothesis: Method A > Method B
        Variables: Algorithm (independent), Accuracy (dependent)
        Methodology: A/B testing with 5-fold CV
        """

        design = exp_agent.execute({
            'action': 'design_experiment',
            'project_id': project.id,
            'research_question': 'Which method is better?'
        })

        # Create experiment record
        experiment = Experiment(
            project_id=project.id,
            name="Comparison Experiment",
            design=design.get('design', {}),
            status="running"
        )
        db_session.add(experiment)
        db_session.commit()

        # Analyze data
        analysis = exp_agent.execute({
            'action': 'analyze_data',
            'experiment_id': experiment.id,
            'data': sample_dataframe,
            'analysis_type': 'descriptive'
        })
        assert 'statistics' in analysis or 'analysis' in analysis

        # Statistical testing
        stat_test = exp_agent.execute({
            'action': 'statistical_test',
            'experiment_id': experiment.id,
            'test_type': 'ttest',
            'data': sample_dataframe
        })
        assert 'test_result' in stat_test or 'p_value' in stat_test

        # Visualize results
        viz = exp_agent.execute({
            'action': 'visualize',
            'experiment_id': experiment.id,
            'data': sample_dataframe,
            'plot_type': 'bar'
        })

    def test_multiple_experiments_comparison(self, mock_llm, db_session, sample_project_data):
        """Test comparing results from multiple experiments"""

        from paperagent.database.models import Project, Experiment

        project = Project(**sample_project_data)
        db_session.add(project)
        db_session.commit()

        # Create multiple experiments
        exp1 = Experiment(
            project_id=project.id,
            name="Experiment 1",
            results={'accuracy': 0.85}
        )
        exp2 = Experiment(
            project_id=project.id,
            name="Experiment 2",
            results={'accuracy': 0.90}
        )
        exp3 = Experiment(
            project_id=project.id,
            name="Experiment 3",
            results={'accuracy': 0.88}
        )
        db_session.add_all([exp1, exp2, exp3])
        db_session.commit()

        exp_agent = ExperimentAgent(llm=mock_llm, db_session=db_session)

        # Compare experiments
        comparison = exp_agent.execute({
            'action': 'compare_experiments',
            'experiment_ids': [exp1.id, exp2.id, exp3.id]
        })


@pytest.mark.integration
class TestWritingWorkflow:
    """Test paper writing workflow"""

    def test_complete_paper_generation(self, mock_llm, db_session, sample_project_data):
        """Test generating complete paper from structure to final draft"""

        from paperagent.database.models import Project, Paper

        project = Project(**sample_project_data)
        db_session.add(project)
        db_session.commit()

        write_agent = WritingAgent(llm=mock_llm, db_session=db_session)

        # Generate structure
        mock_llm.generate.return_value = """
        Structure:
        1. Abstract
        2. Introduction
        3. Related Work
        4. Methodology
        5. Experiments
        6. Results
        7. Discussion
        8. Conclusion
        """

        structure = write_agent.execute({
            'action': 'generate_structure',
            'project_id': project.id,
            'paper_type': 'conference'
        })

        # Create paper
        paper = Paper(
            project_id=project.id,
            title="Research Paper",
            sections={}
        )
        db_session.add(paper)
        db_session.commit()

        # Write each section
        sections = ['abstract', 'introduction', 'methodology', 'results', 'conclusion']

        for section in sections:
            mock_llm.generate.return_value = f"Content for {section} section..."

            section_content = write_agent.execute({
                'action': 'write_section',
                'paper_id': paper.id,
                'section': section
            })

            assert 'content' in section_content

            # Update paper
            paper.sections[section] = section_content['content']

        db_session.commit()

        # Polish entire paper
        mock_llm.generate.return_value = "Polished version of the paper..."

        polished = write_agent.execute({
            'action': 'polish',
            'paper_id': paper.id,
            'style': 'academic'
        })

    def test_revision_workflow(self, mock_llm, db_session, sample_project_data):
        """Test paper revision workflow with multiple versions"""

        from paperagent.database.models import Project, Paper, PaperRevision

        project = Project(**sample_project_data)
        paper = Paper(
            project_id=project.id,
            title="Test Paper",
            sections={'introduction': 'Initial intro'}
        )
        db_session.add_all([project, paper])
        db_session.commit()

        write_agent = WritingAgent(llm=mock_llm, db_session=db_session)

        # Create revisions
        for version in range(1, 4):
            mock_llm.generate.return_value = f"Improved content version {version}"

            revised = write_agent.execute({
                'action': 'revise_section',
                'paper_id': paper.id,
                'section': 'introduction',
                'feedback': 'Make it more concise'
            })

            # Save revision
            revision = PaperRevision(
                paper_id=paper.id,
                version=version,
                content=revised.get('content', ''),
                changes=f"Revision {version}"
            )
            db_session.add(revision)

        db_session.commit()

        # Verify revisions
        revisions = db_session.query(PaperRevision).filter_by(
            paper_id=paper.id
        ).count()
        assert revisions == 3


@pytest.mark.integration
class TestCrossAgentCommunication:
    """Test communication and data flow between agents"""

    def test_boss_coordinates_agents(self, mock_llm, db_session):
        """Test boss agent coordinating other agents"""

        boss = BossAgent(llm=mock_llm, db_session=db_session)

        # Create project
        project = boss.execute({
            'action': 'create_project',
            'name': 'Coordinated Research',
            'research_field': 'AI'
        })

        # Decompose tasks
        mock_llm.generate.return_value = """
        Tasks:
        1. Literature Review - Priority: High - Agent: Literature
        2. Experiment Design - Priority: High - Agent: Experiment
        3. Paper Writing - Priority: Medium - Agent: Writing
        """

        tasks = boss.execute({
            'action': 'decompose_task',
            'project_id': project['project_id'],
            'main_goal': 'Complete research paper'
        })

        assert 'tasks' in tasks

        # Monitor progress
        progress = boss.execute({
            'action': 'monitor_progress',
            'project_id': project['project_id']
        })

        assert 'progress' in progress or 'completion_rate' in progress

    @patch('paperagent.agents.literature_agent.LiteratureAgent')
    @patch('paperagent.agents.experiment_agent.ExperimentAgent')
    @patch('paperagent.agents.writing_agent.WritingAgent')
    def test_data_flow_between_agents(self, mock_write, mock_exp, mock_lit,
                                     mock_llm, db_session):
        """Test data flowing between literature -> experiment -> writing agents"""

        # Mock agent responses
        lit_result = {
            'gaps': ['Gap 1', 'Gap 2'],
            'key_findings': ['Finding 1', 'Finding 2']
        }
        exp_result = {
            'design': {'methodology': 'A/B testing'},
            'results': {'accuracy': 0.95}
        }
        write_result = {
            'content': 'Paper content...'
        }

        mock_lit.return_value.execute.return_value = lit_result
        mock_exp.return_value.execute.return_value = exp_result
        mock_write.return_value.execute.return_value = write_result

        boss = BossAgent(llm=mock_llm, db_session=db_session)

        # Execute coordinated workflow
        result = boss.execute({
            'action': 'execute_workflow',
            'project_id': 1,
            'workflow_type': 'full'
        })

        # Verify agents were called
        mock_lit.return_value.execute.assert_called()
        mock_exp.return_value.execute.assert_called()
        mock_write.return_value.execute.assert_called()


@pytest.mark.integration
@pytest.mark.slow
class TestLongRunningWorkflows:
    """Test long-running workflows with multiple stages"""

    def test_multi_iteration_research_cycle(self, mock_llm, db_session):
        """Test multiple iterations of research cycle"""

        boss = BossAgent(llm=mock_llm, db_session=db_session)

        project = boss.execute({
            'action': 'create_project',
            'name': 'Iterative Research',
            'research_field': 'Machine Learning'
        })

        # Multiple research iterations
        for iteration in range(3):
            # Literature update
            lit_agent = LiteratureAgent(llm=mock_llm, db_session=db_session)
            lit_agent.execute({
                'action': 'search_literature',
                'query': f'machine learning iteration {iteration}',
                'project_id': project['project_id']
            })

            # Experiment refinement
            exp_agent = ExperimentAgent(llm=mock_llm, db_session=db_session)
            exp_agent.execute({
                'action': 'design_experiment',
                'project_id': project['project_id'],
                'iteration': iteration
            })

            # Progress check
            progress = boss.execute({
                'action': 'monitor_progress',
                'project_id': project['project_id']
            })

    def test_large_scale_literature_processing(self, mock_llm, db_session):
        """Test processing large number of papers"""

        from paperagent.database.models import Project, Literature

        project = Project(
            name="Large Scale Review",
            research_field="AI"
        )
        db_session.add(project)
        db_session.commit()

        # Add many papers
        papers = [
            Literature(
                project_id=project.id,
                title=f"Paper {i}",
                authors=f"Authors {i}",
                abstract=f"Abstract {i}",
                source="arxiv"
            )
            for i in range(100)
        ]
        db_session.add_all(papers)
        db_session.commit()

        lit_agent = LiteratureAgent(llm=mock_llm, db_session=db_session)

        # Process in batches
        batch_size = 20
        for batch_start in range(0, 100, batch_size):
            batch_ids = [p.id for p in papers[batch_start:batch_start + batch_size]]

            result = lit_agent.execute({
                'action': 'analyze_batch',
                'literature_ids': batch_ids
            })
