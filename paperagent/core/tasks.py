"""
Celery tasks for background processing

This module defines asynchronous tasks that can be executed in the background
using Celery workers. These tasks include literature search, data analysis,
paper generation, and other time-consuming operations.
"""

from celery import Celery
from celery.utils.log import get_task_logger
from typing import Dict, Any, Optional
import os

# Initialize logger
logger = get_task_logger(__name__)

# Celery configuration
CELERY_BROKER_URL = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0')
CELERY_RESULT_BACKEND = os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')

# Initialize Celery app
celery_app = Celery(
    'paperagent',
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND
)

# Configure Celery
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes max
    task_soft_time_limit=25 * 60,  # 25 minutes soft limit
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_reject_on_worker_lost=True,
)


@celery_app.task(name='paperagent.tasks.search_literature', bind=True)
def search_literature_task(self, project_id: int, query: str, max_results: int = 50) -> Dict[str, Any]:
    """
    Background task for literature search

    Args:
        project_id: Project ID
        query: Search query
        max_results: Maximum number of results

    Returns:
        Search results dictionary
    """
    try:
        logger.info(f"Starting literature search for project {project_id}, query: {query}")

        from paperagent.agents.literature_agent import LiteratureAgent
        from paperagent.database.database import get_db

        # Get database session
        db = next(get_db())

        try:
            # Create agent and execute search
            agent = LiteratureAgent(db_session=db)
            result = agent.execute({
                'action': 'search_literature',
                'project_id': project_id,
                'query': query,
                'max_results': max_results
            })

            logger.info(f"Literature search completed for project {project_id}")
            return {
                'status': 'success',
                'result': result
            }

        finally:
            db.close()

    except Exception as e:
        logger.error(f"Literature search failed for project {project_id}: {str(e)}")
        self.retry(exc=e, countdown=60, max_retries=3)


@celery_app.task(name='paperagent.tasks.analyze_data', bind=True)
def analyze_data_task(self, experiment_id: int, analysis_type: str = 'descriptive') -> Dict[str, Any]:
    """
    Background task for experiment data analysis

    Args:
        experiment_id: Experiment ID
        analysis_type: Type of analysis to perform

    Returns:
        Analysis results dictionary
    """
    try:
        logger.info(f"Starting data analysis for experiment {experiment_id}")

        from paperagent.agents.experiment_agent import ExperimentAgent
        from paperagent.database.database import get_db

        db = next(get_db())

        try:
            agent = ExperimentAgent(db_session=db)
            result = agent.execute({
                'action': 'analyze_data',
                'experiment_id': experiment_id,
                'analysis_type': analysis_type
            })

            logger.info(f"Data analysis completed for experiment {experiment_id}")
            return {
                'status': 'success',
                'result': result
            }

        finally:
            db.close()

    except Exception as e:
        logger.error(f"Data analysis failed for experiment {experiment_id}: {str(e)}")
        self.retry(exc=e, countdown=60, max_retries=3)


@celery_app.task(name='paperagent.tasks.generate_paper_section', bind=True)
def generate_paper_section_task(
    self,
    paper_id: int,
    section: str,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Background task for generating a paper section

    Args:
        paper_id: Paper ID
        section: Section name (introduction, methodology, etc.)
        context: Optional context for generation

    Returns:
        Generated section content
    """
    try:
        logger.info(f"Starting section generation for paper {paper_id}, section: {section}")

        from paperagent.agents.writing_agent import WritingAgent
        from paperagent.database.database import get_db

        db = next(get_db())

        try:
            agent = WritingAgent(db_session=db)
            result = agent.execute({
                'action': 'write_section',
                'paper_id': paper_id,
                'section': section,
                'context': context or {}
            })

            logger.info(f"Section generation completed for paper {paper_id}")
            return {
                'status': 'success',
                'result': result
            }

        finally:
            db.close()

    except Exception as e:
        logger.error(f"Section generation failed for paper {paper_id}: {str(e)}")
        self.retry(exc=e, countdown=60, max_retries=3)


@celery_app.task(name='paperagent.tasks.generate_full_paper', bind=True)
def generate_full_paper_task(self, paper_id: int) -> Dict[str, Any]:
    """
    Background task for generating a complete paper

    Args:
        paper_id: Paper ID

    Returns:
        Complete paper generation result
    """
    try:
        logger.info(f"Starting full paper generation for paper {paper_id}")

        from paperagent.agents.writing_agent import WritingAgent
        from paperagent.database.database import get_db

        db = next(get_db())

        try:
            agent = WritingAgent(db_session=db)

            # Generate all sections
            sections = ['abstract', 'introduction', 'related_work',
                       'methodology', 'experiments', 'results', 'conclusion']

            results = {}
            for section in sections:
                self.update_state(state='PROGRESS', meta={'current': section})

                section_result = agent.execute({
                    'action': 'write_section',
                    'paper_id': paper_id,
                    'section': section
                })

                results[section] = section_result

            logger.info(f"Full paper generation completed for paper {paper_id}")
            return {
                'status': 'success',
                'results': results
            }

        finally:
            db.close()

    except Exception as e:
        logger.error(f"Full paper generation failed for paper {paper_id}: {str(e)}")
        self.retry(exc=e, countdown=120, max_retries=2)


@celery_app.task(name='paperagent.tasks.format_paper', bind=True)
def format_paper_task(
    self,
    paper_id: int,
    template: str = 'ieee'
) -> Dict[str, Any]:
    """
    Background task for paper formatting

    Args:
        paper_id: Paper ID
        template: Journal template name

    Returns:
        Formatted paper result
    """
    try:
        logger.info(f"Starting paper formatting for paper {paper_id}, template: {template}")

        from paperagent.agents.formatting_agent import FormattingAgent
        from paperagent.database.database import get_db

        db = next(get_db())

        try:
            agent = FormattingAgent(db_session=db)
            result = agent.execute({
                'action': 'format_paper',
                'paper_id': paper_id,
                'template': template
            })

            logger.info(f"Paper formatting completed for paper {paper_id}")
            return {
                'status': 'success',
                'result': result
            }

        finally:
            db.close()

    except Exception as e:
        logger.error(f"Paper formatting failed for paper {paper_id}: {str(e)}")
        self.retry(exc=e, countdown=60, max_retries=3)


@celery_app.task(name='paperagent.tasks.extract_pdf_content', bind=True)
def extract_pdf_content_task(self, file_path: str) -> Dict[str, Any]:
    """
    Background task for extracting content from PDF

    Args:
        file_path: Path to PDF file

    Returns:
        Extracted content dictionary
    """
    try:
        logger.info(f"Starting PDF content extraction for {file_path}")

        from paperagent.tools.multimodal_analyzer import PDFDeepAnalyzer

        analyzer = PDFDeepAnalyzer()
        result = analyzer.analyze_pdf(file_path)

        logger.info(f"PDF content extraction completed for {file_path}")
        return {
            'status': 'success',
            'result': result
        }

    except Exception as e:
        logger.error(f"PDF extraction failed for {file_path}: {str(e)}")
        self.retry(exc=e, countdown=30, max_retries=3)


@celery_app.task(name='paperagent.tasks.cluster_literature', bind=True)
def cluster_literature_task(
    self,
    project_id: int,
    num_clusters: int = 5
) -> Dict[str, Any]:
    """
    Background task for clustering literature

    Args:
        project_id: Project ID
        num_clusters: Number of clusters

    Returns:
        Clustering results
    """
    try:
        logger.info(f"Starting literature clustering for project {project_id}")

        from paperagent.agents.literature_agent import LiteratureAgent
        from paperagent.database.database import get_db

        db = next(get_db())

        try:
            agent = LiteratureAgent(db_session=db)
            result = agent.execute({
                'action': 'cluster_literature',
                'project_id': project_id,
                'num_clusters': num_clusters
            })

            logger.info(f"Literature clustering completed for project {project_id}")
            return {
                'status': 'success',
                'result': result
            }

        finally:
            db.close()

    except Exception as e:
        logger.error(f"Literature clustering failed for project {project_id}: {str(e)}")
        self.retry(exc=e, countdown=60, max_retries=3)


# Task routing and monitoring
@celery_app.task(name='paperagent.tasks.health_check')
def health_check_task() -> Dict[str, str]:
    """
    Simple health check task

    Returns:
        Health status dictionary
    """
    return {
        'status': 'healthy',
        'service': 'celery',
        'timestamp': str(celery_app.now())
    }


# Export celery app
__all__ = ['celery_app']
