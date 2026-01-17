"""
Literature collection tools for arXiv and Google Scholar
"""

import arxiv
from typing import List, Dict, Any, Optional
import requests
from bs4 import BeautifulSoup
import time
import random
from loguru import logger
from datetime import datetime

from paperagent.core.config import settings


class ArxivCollector:
    """
    arXiv paper collector using official API

    Compliant with arXiv Terms of Use
    """

    def __init__(self, max_results: int = 50):
        self.max_results = max_results

    def search(
        self,
        query: str,
        max_results: Optional[int] = None,
        sort_by: arxiv.SortCriterion = arxiv.SortCriterion.Relevance,
        sort_order: arxiv.SortOrder = arxiv.SortOrder.Descending
    ) -> List[Dict[str, Any]]:
        """
        Search arXiv papers

        Args:
            query: Search query string
            max_results: Maximum number of results
            sort_by: Sort criterion
            sort_order: Sort order

        Returns:
            List of paper metadata dictionaries
        """
        try:
            max_results = max_results or self.max_results

            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=sort_by,
                sort_order=sort_order
            )

            papers = []
            for result in search.results():
                paper = {
                    "title": result.title,
                    "authors": [author.name for author in result.authors],
                    "abstract": result.summary,
                    "arxiv_id": result.entry_id.split("/")[-1],
                    "published": result.published.isoformat() if result.published else None,
                    "updated": result.updated.isoformat() if result.updated else None,
                    "doi": result.doi,
                    "primary_category": result.primary_category,
                    "categories": result.categories,
                    "pdf_url": result.pdf_url,
                    "journal_ref": result.journal_ref,
                    "comment": result.comment,
                    "source": "arxiv"
                }
                papers.append(paper)

            logger.info(f"Found {len(papers)} papers from arXiv for query: {query}")
            return papers

        except Exception as e:
            logger.error(f"arXiv search error: {e}")
            return []

    def download_pdf(self, arxiv_id: str, save_path: str) -> bool:
        """
        Download PDF from arXiv

        Args:
            arxiv_id: arXiv ID (e.g., "2301.12345")
            save_path: Path to save PDF

        Returns:
            Success boolean
        """
        try:
            search = arxiv.Search(id_list=[arxiv_id])
            paper = next(search.results())
            paper.download_pdf(filename=save_path)
            logger.info(f"Downloaded PDF: {arxiv_id} to {save_path}")
            return True
        except Exception as e:
            logger.error(f"PDF download error for {arxiv_id}: {e}")
            return False

    def get_paper_by_id(self, arxiv_id: str) -> Optional[Dict[str, Any]]:
        """
        Get paper metadata by arXiv ID

        Args:
            arxiv_id: arXiv ID

        Returns:
            Paper metadata dictionary or None
        """
        try:
            search = arxiv.Search(id_list=[arxiv_id])
            result = next(search.results())

            paper = {
                "title": result.title,
                "authors": [author.name for author in result.authors],
                "abstract": result.summary,
                "arxiv_id": arxiv_id,
                "published": result.published.isoformat() if result.published else None,
                "updated": result.updated.isoformat() if result.updated else None,
                "doi": result.doi,
                "primary_category": result.primary_category,
                "categories": result.categories,
                "pdf_url": result.pdf_url,
                "journal_ref": result.journal_ref,
                "comment": result.comment,
                "source": "arxiv"
            }

            return paper
        except Exception as e:
            logger.error(f"Error fetching arXiv paper {arxiv_id}: {e}")
            return None


class GoogleScholarCollector:
    """
    Google Scholar collector with proxy support

    Note: Use responsibly and consider rate limiting
    """

    def __init__(self, use_proxy: bool = False, proxy_url: Optional[str] = None):
        self.use_proxy = use_proxy or settings.use_proxy
        self.proxy_url = proxy_url or settings.proxy_url
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

    def search(
        self,
        query: str,
        max_results: int = 20,
        year_start: Optional[int] = None,
        year_end: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Search Google Scholar papers

        Args:
            query: Search query
            max_results: Maximum results
            year_start: Start year filter
            year_end: End year filter

        Returns:
            List of paper metadata
        """
        try:
            # Use scholarly library for Google Scholar
            from scholarly import scholarly, ProxyGenerator

            # Setup proxy if enabled
            if self.use_proxy and self.proxy_url:
                pg = ProxyGenerator()
                pg.SingleProxy(http=self.proxy_url, https=self.proxy_url)
                scholarly.use_proxy(pg)

            # Search query
            search_query = scholarly.search_pubs(query)

            papers = []
            count = 0

            for result in search_query:
                if count >= max_results:
                    break

                try:
                    # Extract paper information
                    paper = {
                        "title": result.get('bib', {}).get('title', ''),
                        "authors": result.get('bib', {}).get('author', []),
                        "abstract": result.get('bib', {}).get('abstract', ''),
                        "year": result.get('bib', {}).get('pub_year'),
                        "venue": result.get('bib', {}).get('venue', ''),
                        "publisher": result.get('bib', {}).get('publisher', ''),
                        "citation_count": result.get('num_citations', 0),
                        "url": result.get('pub_url', ''),
                        "eprint_url": result.get('eprint_url', ''),
                        "source": "google_scholar"
                    }

                    # Filter by year if specified
                    if year_start and paper['year']:
                        try:
                            if int(paper['year']) < year_start:
                                continue
                        except (ValueError, TypeError):
                            pass

                    if year_end and paper['year']:
                        try:
                            if int(paper['year']) > year_end:
                                continue
                        except (ValueError, TypeError):
                            pass

                    papers.append(paper)
                    count += 1

                    # Rate limiting
                    time.sleep(random.uniform(1, 3))

                except Exception as e:
                    logger.warning(f"Error parsing Google Scholar result: {e}")
                    continue

            logger.info(f"Found {len(papers)} papers from Google Scholar for query: {query}")
            return papers

        except Exception as e:
            logger.error(f"Google Scholar search error: {e}")
            return []

    def get_paper_details(self, title: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific paper

        Args:
            title: Paper title

        Returns:
            Detailed paper metadata
        """
        try:
            from scholarly import scholarly

            search_query = scholarly.search_pubs(title)
            result = next(search_query, None)

            if not result:
                return None

            # Fill additional details
            filled = scholarly.fill(result)

            paper = {
                "title": filled.get('bib', {}).get('title', ''),
                "authors": filled.get('bib', {}).get('author', []),
                "abstract": filled.get('bib', {}).get('abstract', ''),
                "year": filled.get('bib', {}).get('pub_year'),
                "venue": filled.get('bib', {}).get('venue', ''),
                "publisher": filled.get('bib', {}).get('publisher', ''),
                "citation_count": filled.get('num_citations', 0),
                "url": filled.get('pub_url', ''),
                "eprint_url": filled.get('eprint_url', ''),
                "cited_by_url": filled.get('citedby_url', ''),
                "source": "google_scholar"
            }

            return paper

        except Exception as e:
            logger.error(f"Error getting paper details: {e}")
            return None


class LiteratureCollector:
    """
    Unified literature collector combining multiple sources
    """

    def __init__(self):
        self.arxiv = ArxivCollector()
        self.scholar = GoogleScholarCollector()

    def search_all(
        self,
        query: str,
        max_results_per_source: int = 25,
        sources: List[str] = ["arxiv", "google_scholar"]
    ) -> List[Dict[str, Any]]:
        """
        Search across all enabled sources

        Args:
            query: Search query
            max_results_per_source: Max results from each source
            sources: List of sources to search

        Returns:
            Combined list of papers from all sources
        """
        all_papers = []

        if "arxiv" in sources:
            arxiv_papers = self.arxiv.search(query, max_results=max_results_per_source)
            all_papers.extend(arxiv_papers)

        if "google_scholar" in sources:
            scholar_papers = self.scholar.search(query, max_results=max_results_per_source)
            all_papers.extend(scholar_papers)

        # Remove duplicates based on title similarity
        unique_papers = self._deduplicate_papers(all_papers)

        logger.info(f"Total unique papers found: {len(unique_papers)}")
        return unique_papers

    def _deduplicate_papers(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate papers based on title similarity

        Args:
            papers: List of papers

        Returns:
            Deduplicated list
        """
        unique_papers = []
        seen_titles = set()

        for paper in papers:
            title = paper.get('title', '').lower().strip()
            if title and title not in seen_titles:
                seen_titles.add(title)
                unique_papers.append(paper)

        return unique_papers

    def download_paper(self, paper: Dict[str, Any], save_dir: str) -> Optional[str]:
        """
        Download paper PDF if available

        Args:
            paper: Paper metadata
            save_dir: Directory to save PDF

        Returns:
            Path to downloaded PDF or None
        """
        import os

        # Generate filename
        title_slug = paper.get('title', 'paper')[:50].replace(' ', '_').replace('/', '_')
        filename = f"{title_slug}.pdf"
        save_path = os.path.join(save_dir, filename)

        # Try arXiv download
        if paper.get('source') == 'arxiv' and paper.get('arxiv_id'):
            if self.arxiv.download_pdf(paper['arxiv_id'], save_path):
                return save_path

        # Try direct URL download
        if paper.get('pdf_url'):
            try:
                response = requests.get(paper['pdf_url'], stream=True)
                if response.status_code == 200:
                    with open(save_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    logger.info(f"Downloaded PDF to {save_path}")
                    return save_path
            except Exception as e:
                logger.error(f"PDF download error: {e}")

        return None
