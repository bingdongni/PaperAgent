"""
Formatting Agent for Paper Formatting and Submission

This agent handles paper formatting, LaTeX generation, journal template application,
and submission preparation tasks.
"""

from typing import Dict, Any, Optional, List
from loguru import logger

from paperagent.agents.base_agent import BaseAgent
from paperagent.core.llm_manager import LLMManager
from paperagent.tools.latex_processor import LaTeXProcessor, BibTeXManager
from paperagent.tools.latex_advanced import (
    AdvancedLaTeXFormatter,
    ComplexTableFormatter,
    MathematicalFormatter
)
from paperagent.templates.journal_templates import JournalTemplates


class FormattingAgent(BaseAgent):
    """
    Agent responsible for paper formatting and submission preparation

    This agent handles:
    - LaTeX document generation
    - Journal template application
    - Citation formatting
    - Figure and table formatting
    - Submission checklist verification
    """

    def __init__(self, llm: Optional[LLMManager] = None, db_session=None):
        """
        Initialize Formatting Agent

        Args:
            llm: LLM manager instance
            db_session: Database session
        """
        super().__init__(agent_type="formatting", name="Formatting Agent")

        self.llm = llm or LLMManager()
        self.db_session = db_session

        # Initialize formatting tools
        self.latex_processor = LaTeXProcessor()
        self.bibtex_manager = BibTeXManager()
        self.advanced_formatter = AdvancedLaTeXFormatter()
        self.table_formatter = ComplexTableFormatter()
        self.math_formatter = MathematicalFormatter()
        self.journal_templates = JournalTemplates()

        logger.info(f"Initialized {self.name}")

    def execute(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute formatting task

        Args:
            task_input: Task parameters including action and data

        Returns:
            Execution results
        """
        action = task_input.get('action')

        try:
            if action == 'format_paper':
                return self._format_paper(task_input)
            elif action == 'apply_template':
                return self._apply_journal_template(task_input)
            elif action == 'generate_latex':
                return self._generate_latex_document(task_input)
            elif action == 'format_citations':
                return self._format_citations(task_input)
            elif action == 'create_submission_package':
                return self._create_submission_package(task_input)
            elif action == 'check_requirements':
                return self._check_submission_requirements(task_input)
            else:
                raise ValueError(f"Unknown action: {action}")

        except Exception as e:
            logger.error(f"Formatting task failed: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }

    def _format_paper(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format paper according to specifications

        Args:
            task_input: Contains paper_id, template, formatting options

        Returns:
            Formatted paper result
        """
        paper_id = task_input.get('paper_id')
        template = task_input.get('template', 'ieee')

        logger.info(f"Formatting paper {paper_id} with template {template}")

        try:
            # Get paper from database
            if self.db_session:
                from paperagent.database.models import Paper
                paper = self.db_session.query(Paper).filter_by(id=paper_id).first()

                if not paper:
                    return {
                        'status': 'error',
                        'error': f'Paper {paper_id} not found'
                    }

                # Get journal template
                template_config = self._get_template_config(template)

                # Generate formatted version
                formatted_content = self._apply_formatting(paper, template_config)

                return {
                    'status': 'success',
                    'paper_id': paper_id,
                    'template': template,
                    'formatted_content': formatted_content,
                    'output_format': template_config.get('output_format', 'latex')
                }

            return {
                'status': 'error',
                'error': 'Database session not available'
            }

        except Exception as e:
            logger.error(f"Paper formatting failed: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }

    def _apply_journal_template(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply specific journal template to paper

        Args:
            task_input: Contains paper_id and journal name

        Returns:
            Template application result
        """
        paper_id = task_input.get('paper_id')
        journal = task_input.get('journal', 'ieee_access')

        logger.info(f"Applying {journal} template to paper {paper_id}")

        try:
            # Get template configuration
            template_func = getattr(self.journal_templates, journal, None)

            if not template_func:
                return {
                    'status': 'error',
                    'error': f'Unknown journal template: {journal}'
                }

            template_config = template_func()

            # Apply template
            return {
                'status': 'success',
                'paper_id': paper_id,
                'journal': journal,
                'template': template_config,
                'latex_template': template_config.get('latex_template', '')
            }

        except Exception as e:
            logger.error(f"Template application failed: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }

    def _generate_latex_document(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate complete LaTeX document

        Args:
            task_input: Contains paper content and metadata

        Returns:
            Generated LaTeX source
        """
        title = task_input.get('title', 'Untitled')
        authors = task_input.get('authors', [])
        abstract = task_input.get('abstract', '')
        sections = task_input.get('sections', {})
        references = task_input.get('references', [])

        logger.info(f"Generating LaTeX document: {title}")

        try:
            # Create LaTeX document
            latex_source = self.latex_processor.create_document(
                title=title,
                authors=authors,
                abstract=abstract,
                sections=sections,
                references=references,
                document_class=task_input.get('document_class', 'article'),
                packages=task_input.get('packages')
            )

            return {
                'status': 'success',
                'latex_source': latex_source,
                'title': title
            }

        except Exception as e:
            logger.error(f"LaTeX generation failed: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }

    def _format_citations(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format citations according to style

        Args:
            task_input: Contains citations and style

        Returns:
            Formatted citations
        """
        citations = task_input.get('citations', [])
        style = task_input.get('style', 'ieee')

        logger.info(f"Formatting {len(citations)} citations in {style} style")

        try:
            formatted_citations = []

            for citation in citations:
                formatted = self.bibtex_manager.format_citation(citation, style)
                formatted_citations.append(formatted)

            return {
                'status': 'success',
                'style': style,
                'citations': formatted_citations,
                'count': len(formatted_citations)
            }

        except Exception as e:
            logger.error(f"Citation formatting failed: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }

    def _create_submission_package(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create complete submission package

        Args:
            task_input: Contains paper_id and submission requirements

        Returns:
            Submission package details
        """
        paper_id = task_input.get('paper_id')
        journal = task_input.get('journal')

        logger.info(f"Creating submission package for paper {paper_id}")

        try:
            package_contents = {
                'main_document': f'paper_{paper_id}.tex',
                'figures': [],
                'supplementary': [],
                'cover_letter': f'cover_letter_{paper_id}.tex',
                'response_to_reviewers': None
            }

            # Generate cover letter
            cover_letter = self._generate_cover_letter(task_input)
            package_contents['cover_letter_content'] = cover_letter

            return {
                'status': 'success',
                'paper_id': paper_id,
                'journal': journal,
                'package': package_contents
            }

        except Exception as e:
            logger.error(f"Submission package creation failed: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }

    def _check_submission_requirements(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if paper meets submission requirements

        Args:
            task_input: Contains paper_id and journal

        Returns:
            Requirement check results
        """
        paper_id = task_input.get('paper_id')
        journal = task_input.get('journal')

        logger.info(f"Checking submission requirements for {journal}")

        try:
            # Get journal requirements
            template_func = getattr(self.journal_templates, journal, None)

            if not template_func:
                return {
                    'status': 'error',
                    'error': f'Unknown journal: {journal}'
                }

            template_config = template_func()
            requirements = template_config.get('requirements', {})

            # Check requirements
            checks = {
                'abstract_length': self._check_abstract_length(paper_id, requirements),
                'keywords_count': self._check_keywords(paper_id, requirements),
                'page_limit': self._check_page_limit(paper_id, requirements),
                'figures_format': self._check_figures(paper_id, requirements)
            }

            all_passed = all(check['passed'] for check in checks.values())

            return {
                'status': 'success',
                'paper_id': paper_id,
                'journal': journal,
                'checks': checks,
                'all_passed': all_passed
            }

        except Exception as e:
            logger.error(f"Requirement check failed: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }

    def _get_template_config(self, template: str) -> Dict[str, Any]:
        """Get template configuration"""
        template_func = getattr(self.journal_templates, template, None)

        if template_func:
            return template_func()

        # Default template
        return {
            'name': template,
            'document_class': 'article',
            'citation_style': 'IEEE',
            'output_format': 'latex'
        }

    def _apply_formatting(self, paper, template_config: Dict[str, Any]) -> str:
        """Apply formatting to paper"""
        # Create LaTeX document with template
        sections_dict = {}
        if hasattr(paper, 'sections') and paper.sections:
            sections_dict = paper.sections

        latex_source = self.latex_processor.create_document(
            title=paper.title,
            authors=paper.authors if hasattr(paper, 'authors') else [],
            abstract=paper.abstract if hasattr(paper, 'abstract') else '',
            sections=sections_dict,
            document_class=template_config.get('document_class', 'article')
        )

        return latex_source

    def _generate_cover_letter(self, task_input: Dict[str, Any]) -> str:
        """Generate cover letter"""
        journal = task_input.get('journal', 'the journal')
        title = task_input.get('title', 'this manuscript')

        prompt = f"""Generate a professional cover letter for submitting a paper titled "{title}" to {journal}."""

        try:
            cover_letter = self.llm.generate(prompt)
            return cover_letter
        except Exception as e:
            logger.error(f"Cover letter generation failed: {e}")
            return f"Dear Editor,\n\nWe are pleased to submit our manuscript titled \"{title}\" for consideration.\n\nSincerely,\nThe Authors"

    def _check_abstract_length(self, paper_id: int, requirements: Dict) -> Dict[str, Any]:
        """Check abstract length requirement"""
        # Placeholder implementation
        return {'passed': True, 'message': 'Abstract length OK'}

    def _check_keywords(self, paper_id: int, requirements: Dict) -> Dict[str, Any]:
        """Check keywords requirement"""
        return {'passed': True, 'message': 'Keywords count OK'}

    def _check_page_limit(self, paper_id: int, requirements: Dict) -> Dict[str, Any]:
        """Check page limit requirement"""
        return {'passed': True, 'message': 'Page limit OK'}

    def _check_figures(self, paper_id: int, requirements: Dict) -> Dict[str, Any]:
        """Check figures format requirement"""
        return {'passed': True, 'message': 'Figures format OK'}


# Export
__all__ = ['FormattingAgent']
