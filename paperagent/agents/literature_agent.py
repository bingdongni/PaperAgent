"""
Literature Agent - Handles literature search, analysis, and management
"""

from typing import Dict, Any, List, Optional
import json
from loguru import logger

from paperagent.agents.base_agent import BaseAgent
from paperagent.database.models import AgentType, Literature
from paperagent.database.database import get_db_context
from paperagent.tools.literature_collector import LiteratureCollector
from paperagent.core.prompts import LiteraturePrompts
from paperagent.core.config import settings


class LiteratureAgent(BaseAgent):
    """
    Literature Agent for research paper collection and analysis

    Capabilities:
    - Topic recommendation
    - Literature search across multiple databases
    - Paper analysis and summarization
    - Literature clustering and gap analysis
    """

    def __init__(self):
        super().__init__(AgentType.LITERATURE, "Literature Agent")
        self.collector = LiteratureCollector()
        self.prompts = LiteraturePrompts()

    def execute(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute literature-related tasks

        Args:
            task_input: Must contain 'action' key with value:
                - 'recommend_topics': Recommend research topics
                - 'search_literature': Search for papers
                - 'analyze_paper': Analyze a single paper
                - 'cluster_literature': Cluster and analyze multiple papers
                - 'identify_gaps': Identify research gaps

        Returns:
            Task results
        """
        action = task_input.get('action')

        if action == 'recommend_topics':
            return self.recommend_topics(task_input)
        elif action == 'search_literature':
            return self.search_literature(task_input)
        elif action == 'analyze_paper':
            return self.analyze_paper(task_input)
        elif action == 'cluster_literature':
            return self.cluster_literature(task_input)
        elif action == 'identify_gaps':
            return self.identify_research_gaps(task_input)
        else:
            raise ValueError(f"Unknown action: {action}")

    def recommend_topics(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recommend research topics based on field and keywords

        Args:
            task_input: Contains 'field' and 'keywords'

        Returns:
            Dictionary with recommended topics
        """
        self.validate_input(task_input, ['field', 'keywords'])

        field = task_input['field']
        keywords = task_input['keywords']

        logger.info(f"Recommending topics for field: {field}, keywords: {keywords}")

        # Generate prompt
        prompt = self.prompts.TOPIC_RECOMMENDATION.format(
            field=field,
            keywords=keywords if isinstance(keywords, str) else ', '.join(keywords)
        )

        # Get LLM response
        response = self.generate_text(prompt, temperature=0.7)

        # Parse response
        try:
            result = self.parse_json_response(response)
            self.log_action("recommend_topics", {"field": field, "num_topics": len(result.get('topics', []))})
            return result
        except Exception as e:
            logger.error(f"Failed to parse topic recommendations: {e}")
            return {"topics": [], "error": str(e)}

    def search_literature(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Search for literature across multiple sources

        Args:
            task_input: Contains 'query', 'max_results', 'sources', 'project_id'

        Returns:
            Dictionary with search results
        """
        self.validate_input(task_input, ['query', 'project_id'])

        query = task_input['query']
        max_results = task_input.get('max_results', 25)
        sources = task_input.get('sources', ['arxiv', 'google_scholar'])
        project_id = task_input['project_id']

        logger.info(f"Searching literature: {query}")

        # Search papers
        papers = self.collector.search_all(
            query=query,
            max_results_per_source=max_results,
            sources=sources
        )

        # Save papers to database
        saved_papers = []
        with get_db_context() as db:
            for paper_data in papers:
                # Check if paper already exists
                existing = db.query(Literature).filter(
                    Literature.title == paper_data.get('title'),
                    Literature.project_id == project_id
                ).first()

                if not existing:
                    literature = Literature(
                        project_id=project_id,
                        title=paper_data.get('title', ''),
                        authors=paper_data.get('authors', []),
                        abstract=paper_data.get('abstract', ''),
                        year=paper_data.get('year'),
                        doi=paper_data.get('doi'),
                        arxiv_id=paper_data.get('arxiv_id'),
                        url=paper_data.get('url') or paper_data.get('pdf_url'),
                        citation_count=paper_data.get('citation_count', 0),
                        source=paper_data.get('source'),
                    )
                    db.add(literature)
                    saved_papers.append(paper_data)

            db.commit()

        self.log_action("search_literature", {
            "query": query,
            "num_results": len(papers),
            "num_saved": len(saved_papers)
        })

        return {
            "query": query,
            "total_papers": len(papers),
            "saved_papers": len(saved_papers),
            "papers": papers[:10]  # Return first 10 for preview
        }

    def analyze_paper(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a single research paper

        Args:
            task_input: Contains 'paper_id' or 'paper_data', 'research_field'

        Returns:
            Paper analysis results
        """
        paper_data = task_input.get('paper_data', {})
        research_field = task_input.get('research_field', 'Computer Science')

        if not paper_data:
            # Load from database
            paper_id = task_input.get('paper_id')
            if not paper_id:
                raise ValueError("Either paper_id or paper_data required")

            with get_db_context() as db:
                literature = db.query(Literature).filter(Literature.id == paper_id).first()
                if not literature:
                    raise ValueError(f"Paper {paper_id} not found")

                paper_data = {
                    'title': literature.title,
                    'authors': literature.authors,
                    'abstract': literature.abstract
                }

        logger.info(f"Analyzing paper: {paper_data.get('title', 'Unknown')}")

        # Generate analysis prompt
        prompt = self.prompts.LITERATURE_SUMMARY.format(
            title=paper_data.get('title', ''),
            authors=', '.join(paper_data.get('authors', [])),
            abstract=paper_data.get('abstract', ''),
            research_field=research_field
        )

        # Get LLM analysis
        response = self.generate_text(prompt, temperature=0.3)

        try:
            analysis = self.parse_json_response(response)

            # Update database if paper_id provided
            if task_input.get('paper_id'):
                with get_db_context() as db:
                    literature = db.query(Literature).filter(
                        Literature.id == task_input['paper_id']
                    ).first()
                    if literature:
                        literature.summary = analysis.get('objective', '')
                        literature.key_findings = analysis.get('main_findings', [])
                        literature.relevance_score = analysis.get('relevance_score', 0)
                        db.commit()

            self.log_action("analyze_paper", {"paper_title": paper_data.get('title')})
            return analysis

        except Exception as e:
            logger.error(f"Failed to parse paper analysis: {e}")
            return {"error": str(e)}

    def cluster_literature(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Cluster and analyze multiple papers

        Args:
            task_input: Contains 'project_id' or 'paper_ids'

        Returns:
            Clustering results with themes and relationships
        """
        project_id = task_input.get('project_id')
        paper_ids = task_input.get('paper_ids', [])

        # Load papers from database
        with get_db_context() as db:
            if project_id:
                papers = db.query(Literature).filter(
                    Literature.project_id == project_id
                ).all()
            elif paper_ids:
                papers = db.query(Literature).filter(
                    Literature.id.in_(paper_ids)
                ).all()
            else:
                raise ValueError("Either project_id or paper_ids required")

            # Prepare papers list for analysis
            papers_list = []
            for i, paper in enumerate(papers):
                papers_list.append(
                    f"{i+1}. {paper.title} ({', '.join(paper.authors[:3]) if paper.authors else 'Unknown'}, {paper.year})\n"
                    f"   Abstract: {paper.abstract[:200]}..."
                )

        if not papers_list:
            return {"themes": [], "relationships": [], "research_gaps": []}

        papers_text = "\n\n".join(papers_list)

        logger.info(f"Clustering {len(papers)} papers")

        # Generate clustering prompt
        prompt = self.prompts.LITERATURE_CLUSTERING.format(
            papers_list=papers_text
        )

        # Get LLM analysis
        response = self.generate_text(prompt, max_tokens=2000, temperature=0.5)

        try:
            clustering = self.parse_json_response(response)
            self.log_action("cluster_literature", {"num_papers": len(papers)})
            return clustering

        except Exception as e:
            logger.error(f"Failed to parse clustering results: {e}")
            return {"themes": [], "relationships": [], "research_gaps": [], "error": str(e)}

    def identify_research_gaps(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Identify research gaps from literature review

        Args:
            task_input: Contains 'project_id', 'research_field'

        Returns:
            Identified research gaps and opportunities
        """
        self.validate_input(task_input, ['project_id', 'research_field'])

        project_id = task_input['project_id']
        research_field = task_input['research_field']

        # Get literature summary
        with get_db_context() as db:
            papers = db.query(Literature).filter(
                Literature.project_id == project_id
            ).all()

            if not papers:
                return {"research_gaps": [], "prioritized_opportunities": []}

            # Create literature summary
            summary_parts = []
            for paper in papers[:20]:  # Limit to avoid token overflow
                summary_parts.append(
                    f"- {paper.title} ({paper.year}): {paper.summary or 'No summary'}"
                )

            literature_summary = "\n".join(summary_parts)

        logger.info(f"Identifying research gaps for {len(papers)} papers")

        # Generate gap analysis prompt
        prompt = self.prompts.RESEARCH_GAP_ANALYSIS.format(
            literature_summary=literature_summary,
            research_field=research_field
        )

        # Get LLM analysis
        response = self.generate_text(prompt, max_tokens=2000, temperature=0.6)

        try:
            gaps = self.parse_json_response(response)
            self.log_action("identify_gaps", {"num_papers": len(papers)})
            return gaps

        except Exception as e:
            logger.error(f"Failed to parse research gaps: {e}")
            return {"research_gaps": [], "prioritized_opportunities": [], "error": str(e)}

    def download_papers(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Download PDF files for papers

        Args:
            task_input: Contains 'paper_ids' or 'project_id'

        Returns:
            Download results
        """
        project_id = task_input.get('project_id')
        paper_ids = task_input.get('paper_ids', [])

        # Load papers
        with get_db_context() as db:
            if project_id:
                papers = db.query(Literature).filter(
                    Literature.project_id == project_id
                ).all()
            elif paper_ids:
                papers = db.query(Literature).filter(
                    Literature.id.in_(paper_ids)
                ).all()
            else:
                raise ValueError("Either project_id or paper_ids required")

            downloaded = []
            failed = []

            for paper in papers:
                paper_data = {
                    'title': paper.title,
                    'arxiv_id': paper.arxiv_id,
                    'pdf_url': paper.url,
                    'source': paper.source
                }

                # Try to download
                save_dir = settings.literature_dir
                pdf_path = self.collector.download_paper(paper_data, save_dir)

                if pdf_path:
                    paper.pdf_path = pdf_path
                    downloaded.append(paper.title)
                else:
                    failed.append(paper.title)

            db.commit()

        self.log_action("download_papers", {
            "downloaded": len(downloaded),
            "failed": len(failed)
        })

        return {
            "downloaded": downloaded,
            "failed": failed,
            "total": len(papers)
        }
