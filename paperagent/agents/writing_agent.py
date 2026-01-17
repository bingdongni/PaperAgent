"""
Writing Agent - Handles paper writing, polishing, and academic content generation
"""

from typing import Dict, Any, List, Optional
import json
from loguru import logger

from paperagent.agents.base_agent import BaseAgent
from paperagent.database.models import AgentType, Paper, PaperRevision, TaskStatus
from paperagent.database.database import get_db_context
from paperagent.core.prompts import WritingPrompts
from paperagent.tools.document_processor import TextProcessor


class WritingAgent(BaseAgent):
    """
    Writing Agent for academic paper composition

    Capabilities:
    - Paper structure generation
    - Section writing
    - Abstract generation
    - Academic polishing
    - Citation formatting
    - Multi-language support (EN/CN)
    """

    def __init__(self):
        super().__init__(AgentType.WRITING, "Writing Agent")
        self.prompts = WritingPrompts()
        self.text_processor = TextProcessor()

    def execute(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute writing-related tasks

        Args:
            task_input: Must contain 'action' key with value:
                - 'create_structure': Create paper outline
                - 'write_section': Write a specific section
                - 'write_abstract': Generate abstract
                - 'polish_text': Polish and improve text
                - 'generate_draft': Generate complete draft

        Returns:
            Task results
        """
        action = task_input.get('action')

        if action == 'create_structure':
            return self.create_paper_structure(task_input)
        elif action == 'write_section':
            return self.write_section(task_input)
        elif action == 'write_abstract':
            return self.write_abstract(task_input)
        elif action == 'polish_text':
            return self.polish_text(task_input)
        elif action == 'generate_draft':
            return self.generate_complete_draft(task_input)
        else:
            raise ValueError(f"Unknown action: {action}")

    def create_paper_structure(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create paper structure/outline

        Args:
            task_input: Contains 'title', 'objective', 'findings', 'journal', 'project_id'

        Returns:
            Paper structure with sections and key points
        """
        self.validate_input(task_input, ['title', 'objective', 'project_id'])

        title = task_input['title']
        objective = task_input['objective']
        findings = task_input.get('findings', [])
        journal = task_input.get('journal', 'General Academic Journal')
        field = task_input.get('field', 'Computer Science')
        project_id = task_input['project_id']

        logger.info(f"Creating paper structure: {title}")

        # Generate structure prompt
        prompt = self.prompts.PAPER_STRUCTURE.format(
            title=title,
            objective=objective,
            findings=json.dumps(findings) if isinstance(findings, list) else findings,
            journal=journal,
            field=field
        )

        # Get LLM response
        response = self.generate_text(prompt, max_tokens=2000, temperature=0.6)

        try:
            structure = self.parse_json_response(response)

            # Create paper in database
            with get_db_context() as db:
                paper = Paper(
                    project_id=project_id,
                    title=structure.get('title', title),
                    target_journal=journal,
                    status=TaskStatus.IN_PROGRESS
                )
                db.add(paper)
                db.commit()
                db.refresh(paper)

                structure['paper_id'] = paper.id

            self.log_action("create_structure", {"title": title, "paper_id": structure.get('paper_id')})

            return structure

        except Exception as e:
            logger.error(f"Failed to parse paper structure: {e}")
            return {"error": str(e)}

    def write_section(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Write a specific section of the paper

        Args:
            task_input: Contains 'section', 'context', 'key_points', 'paper_id'

        Returns:
            Written section content
        """
        self.validate_input(task_input, ['section', 'context'])

        section = task_input['section']
        context = task_input['context']
        key_points = task_input.get('key_points', [])
        requirements = task_input.get('requirements', '')
        citation_style = task_input.get('citation_style', 'IEEE')
        target_audience = task_input.get('target_audience', 'Academic researchers')
        paper_id = task_input.get('paper_id')

        logger.info(f"Writing section: {section}")

        # Generate section writing prompt
        prompt = self.prompts.SECTION_WRITING.format(
            section=section,
            context=context,
            requirements=requirements,
            key_points=json.dumps(key_points) if isinstance(key_points, list) else key_points,
            citation_style=citation_style,
            target_audience=target_audience
        )

        # Get LLM response
        content = self.generate_text(prompt, max_tokens=2000, temperature=0.7)

        # Update paper in database
        if paper_id:
            with get_db_context() as db:
                paper = db.query(Paper).filter(Paper.id == paper_id).first()
                if paper:
                    # Map section name to database field
                    section_field_map = {
                        'introduction': 'introduction',
                        'literature review': 'literature_review',
                        'methodology': 'methodology',
                        'method': 'methodology',
                        'results': 'results',
                        'discussion': 'discussion',
                        'conclusion': 'conclusion'
                    }

                    field_name = section_field_map.get(section.lower())
                    if field_name:
                        setattr(paper, field_name, content)
                        db.commit()

        # Calculate metrics
        word_count = self.text_processor.count_words(content)
        readability = self.text_processor.calculate_readability(content)

        self.log_action("write_section", {"section": section, "word_count": word_count})

        return {
            "section": section,
            "content": content,
            "word_count": word_count,
            "readability": readability,
            "paper_id": paper_id
        }

    def write_abstract(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate paper abstract

        Args:
            task_input: Contains 'paper_id' or paper content sections

        Returns:
            Generated abstract
        """
        paper_id = task_input.get('paper_id')
        word_limit = task_input.get('word_limit', 250)

        # Load paper content
        with get_db_context() as db:
            paper = db.query(Paper).filter(Paper.id == paper_id).first()
            if not paper:
                raise ValueError(f"Paper {paper_id} not found")

            # Generate abstract prompt
            prompt = self.prompts.ABSTRACT_GENERATION.format(
                title=paper.title,
                introduction=paper.introduction or task_input.get('introduction', ''),
                methodology=paper.methodology or task_input.get('methodology', ''),
                results=paper.results or task_input.get('results', ''),
                conclusion=paper.conclusion or task_input.get('conclusion', ''),
                word_limit=word_limit
            )

        logger.info(f"Generating abstract for paper: {paper_id}")

        # Get LLM response
        abstract = self.generate_text(prompt, max_tokens=500, temperature=0.6)

        # Clean up the abstract
        abstract = abstract.strip()

        # Update paper
        with get_db_context() as db:
            paper = db.query(Paper).filter(Paper.id == paper_id).first()
            if paper:
                paper.abstract = abstract
                db.commit()

        word_count = self.text_processor.count_words(abstract)

        self.log_action("write_abstract", {"paper_id": paper_id, "word_count": word_count})

        return {
            "abstract": abstract,
            "word_count": word_count,
            "within_limit": word_count <= word_limit
        }

    def polish_text(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Polish and improve academic text

        Args:
            task_input: Contains 'text', 'context'

        Returns:
            Polished text with changes noted
        """
        self.validate_input(task_input, ['text'])

        text = task_input['text']
        context = task_input.get('context', '')

        logger.info(f"Polishing text ({len(text)} characters)")

        # Generate polish prompt
        prompt = self.prompts.ACADEMIC_POLISH.format(
            text=text,
            context=context
        )

        # Get LLM response
        response = self.generate_text(prompt, max_tokens=2000, temperature=0.3)

        try:
            result = self.parse_json_response(response)

            polished_text = result.get('polished_text', text)
            changes = result.get('changes', [])
            suggestions = result.get('suggestions', [])

            # Calculate improvement metrics
            original_readability = self.text_processor.calculate_readability(text)
            polished_readability = self.text_processor.calculate_readability(polished_text)

            self.log_action("polish_text", {
                "original_length": len(text),
                "polished_length": len(polished_text),
                "num_changes": len(changes)
            })

            return {
                "original_text": text,
                "polished_text": polished_text,
                "changes": changes,
                "suggestions": suggestions,
                "readability_improvement": {
                    "original": original_readability,
                    "polished": polished_readability
                }
            }

        except Exception as e:
            logger.error(f"Failed to parse polishing result: {e}")
            return {"polished_text": text, "error": str(e)}

    def generate_complete_draft(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a complete paper draft

        Args:
            task_input: Contains 'paper_id' or full context

        Returns:
            Complete draft with all sections
        """
        paper_id = task_input.get('paper_id')

        if not paper_id:
            raise ValueError("paper_id required")

        logger.info(f"Generating complete draft for paper: {paper_id}")

        # Get paper structure
        with get_db_context() as db:
            paper = db.query(Paper).filter(Paper.id == paper_id).first()
            if not paper:
                raise ValueError(f"Paper {paper_id} not found")

            project_id = paper.project_id

        # Define sections to write
        sections = [
            'introduction',
            'literature review',
            'methodology',
            'results',
            'discussion',
            'conclusion'
        ]

        # Write each section
        for section in sections:
            logger.info(f"Writing section: {section}")

            section_input = {
                'section': section,
                'context': f"Paper: {paper.title}\nTarget Journal: {paper.target_journal}",
                'paper_id': paper_id,
                'key_points': []  # Could be populated from database
            }

            self.write_section(section_input)

        # Generate abstract
        self.write_abstract({'paper_id': paper_id})

        # Get complete paper
        with get_db_context() as db:
            paper = db.query(Paper).filter(Paper.id == paper_id).first()

            # Calculate total word count
            total_words = sum([
                self.text_processor.count_words(paper.abstract or ''),
                self.text_processor.count_words(paper.introduction or ''),
                self.text_processor.count_words(paper.literature_review or ''),
                self.text_processor.count_words(paper.methodology or ''),
                self.text_processor.count_words(paper.results or ''),
                self.text_processor.count_words(paper.discussion or ''),
                self.text_processor.count_words(paper.conclusion or ''),
            ])

            paper.word_count = total_words
            paper.status = TaskStatus.COMPLETED
            db.commit()

        self.log_action("generate_draft", {"paper_id": paper_id, "word_count": total_words})

        return {
            "paper_id": paper_id,
            "status": "completed",
            "total_word_count": total_words,
            "sections_written": len(sections)
        }

    def create_revision(self, paper_id: int, changes_summary: str, reviewer_comments: Optional[List[str]] = None) -> int:
        """
        Create a revision of the paper

        Args:
            paper_id: Paper ID
            changes_summary: Summary of changes
            reviewer_comments: Optional reviewer comments

        Returns:
            Revision ID
        """
        with get_db_context() as db:
            paper = db.query(Paper).filter(Paper.id == paper_id).first()
            if not paper:
                raise ValueError(f"Paper {paper_id} not found")

            # Get current version
            current_version = paper.version

            # Create new revision
            revision = PaperRevision(
                paper_id=paper_id,
                version=current_version,
                content=json.dumps({
                    "title": paper.title,
                    "abstract": paper.abstract,
                    "introduction": paper.introduction,
                    "literature_review": paper.literature_review,
                    "methodology": paper.methodology,
                    "results": paper.results,
                    "discussion": paper.discussion,
                    "conclusion": paper.conclusion
                }),
                changes_summary=changes_summary,
                reviewer_comments=reviewer_comments or []
            )

            db.add(revision)

            # Increment paper version
            paper.version += 1

            db.commit()
            db.refresh(revision)

            logger.info(f"Created revision {revision.id} for paper {paper_id}")

            return revision.id
