"""
Document Structure Analysis Module

Provides deep analysis of document structure including:
- Section and heading extraction
- Citation analysis and extraction
- Reference linking
- Document hierarchy detection
- Cross-reference resolution
"""

import re
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import pdfplumber
from pdfminer.high_level import extract_text
from loguru import logger


class DocumentStructureAnalyzer:
    """Analyze document structure and organization"""

    def __init__(self):
        self.section_patterns = [
            r'^(\d+\.?\d*\.?\d*)\s+([A-Z].*?)$',  # Numbered sections: 1. Introduction
            r'^([IVX]+\.)\s+([A-Z].*?)$',  # Roman numerals: I. Introduction
            r'^(Abstract|Introduction|Related Work|Methodology|Methods|Experiments|Results|Discussion|Conclusion|References|Acknowledgments)',
            r'^Chapter\s+(\d+):?\s+(.+)$',
        ]

        self.citation_patterns = [
            r'\[(\d+)\]',  # [1]
            r'\(([A-Za-z]+\s+et\s+al\.,?\s+\d{4})\)',  # (Smith et al., 2020)
            r'\(([A-Za-z]+,?\s+\d{4})\)',  # (Smith, 2020)
            r'\[([A-Za-z]+\d{2,4})\]',  # [Smith20]
        ]

    def analyze_document(self, file_path: str) -> Dict[str, Any]:
        """
        Comprehensive document structure analysis

        Args:
            file_path: Path to document (PDF or text)

        Returns:
            Structure analysis results
        """
        # Extract text
        text = self._extract_text(file_path)

        results = {
            'sections': self._extract_sections(text),
            'citations': self._extract_citations(text),
            'references': self._extract_references(text),
            'hierarchy': self._build_hierarchy(text),
            'metadata': self._extract_metadata(text),
            'statistics': self._calculate_statistics(text)
        }

        return results

    def _extract_text(self, file_path: str) -> str:
        """Extract text from document"""
        path = Path(file_path)

        if path.suffix.lower() == '.pdf':
            try:
                return extract_text(str(path))
            except Exception as e:
                logger.error(f"PDF extraction error: {e}")
                return ""
        elif path.suffix.lower() == '.txt':
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                logger.error(f"Text file error: {e}")
                return ""
        else:
            return ""

    def _extract_sections(self, text: str) -> List[Dict[str, Any]]:
        """Extract document sections"""
        lines = text.split('\n')
        sections = []

        for i, line in enumerate(lines):
            line_stripped = line.strip()

            if not line_stripped:
                continue

            # Try each pattern
            for pattern in self.section_patterns:
                match = re.match(pattern, line_stripped, re.IGNORECASE)
                if match:
                    section_info = {
                        'line_number': i,
                        'title': line_stripped,
                        'level': self._determine_section_level(line_stripped),
                        'number': match.group(1) if match.lastindex >= 1 else None
                    }
                    sections.append(section_info)
                    break

        return sections

    def _determine_section_level(self, section_title: str) -> int:
        """Determine section hierarchy level"""
        # Count leading numbers (1.2.3 -> level 3)
        number_match = re.match(r'^(\d+(?:\.\d+)*)', section_title)
        if number_match:
            return len(number_match.group(1).split('.'))

        # Check for specific keywords
        if any(kw in section_title for kw in ['Abstract', 'Introduction', 'Conclusion', 'References']):
            return 1

        # Check for subsection indicators
        if section_title.startswith('  ') or section_title.startswith('\t'):
            return 2

        return 1

    def _extract_citations(self, text: str) -> List[Dict[str, Any]]:
        """Extract in-text citations"""
        citations = []

        for pattern in self.citation_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                citations.append({
                    'text': match.group(0),
                    'key': match.group(1),
                    'position': match.start(),
                    'type': self._classify_citation_style(match.group(0))
                })

        return citations

    def _classify_citation_style(self, citation: str) -> str:
        """Classify citation style"""
        if re.match(r'\[\d+\]', citation):
            return 'numeric'
        elif re.match(r'\(.*et al\..*\)', citation):
            return 'author-year'
        elif re.match(r'\[.*\d{2,4}\]', citation):
            return 'alphanumeric'
        else:
            return 'unknown'

    def _extract_references(self, text: str) -> List[Dict[str, Any]]:
        """Extract reference list"""
        references = []

        # Find references section
        ref_match = re.search(
            r'(References|Bibliography|Works Cited)\s*\n(.*?)(?=\n\s*\n|\Z)',
            text,
            re.IGNORECASE | re.DOTALL
        )

        if not ref_match:
            return references

        ref_text = ref_match.group(2)
        ref_lines = ref_text.split('\n')

        current_ref = []
        for line in ref_lines:
            line_stripped = line.strip()

            if not line_stripped:
                if current_ref:
                    references.append(self._parse_reference(' '.join(current_ref)))
                    current_ref = []
            else:
                # Check if new reference (starts with number or author)
                if re.match(r'^\[\d+\]|^\d+\.|^[A-Z]', line_stripped):
                    if current_ref:
                        references.append(self._parse_reference(' '.join(current_ref)))
                    current_ref = [line_stripped]
                else:
                    current_ref.append(line_stripped)

        # Add last reference
        if current_ref:
            references.append(self._parse_reference(' '.join(current_ref)))

        return references

    def _parse_reference(self, ref_text: str) -> Dict[str, Any]:
        """Parse individual reference"""
        reference = {
            'raw_text': ref_text,
            'authors': self._extract_authors(ref_text),
            'title': self._extract_title(ref_text),
            'year': self._extract_year(ref_text),
            'venue': self._extract_venue(ref_text),
            'doi': self._extract_doi(ref_text),
            'url': self._extract_url(ref_text)
        }

        return reference

    def _extract_authors(self, ref_text: str) -> List[str]:
        """Extract author names from reference"""
        # Pattern: LastName, FirstInitial. and variations
        author_pattern = r'([A-Z][a-z]+(?:,\s+[A-Z]\.?)?)'
        matches = re.findall(author_pattern, ref_text[:200])  # Check first 200 chars

        return matches[:10]  # Limit to 10 authors

    def _extract_title(self, ref_text: str) -> Optional[str]:
        """Extract title from reference"""
        # Title usually in quotes or between author and year
        title_match = re.search(r'["\'](.+?)["\']', ref_text)
        if title_match:
            return title_match.group(1)

        # Try finding text between first period and year
        year_match = re.search(r'\d{4}', ref_text)
        if year_match:
            before_year = ref_text[:year_match.start()]
            parts = before_year.split('.')
            if len(parts) > 1:
                return parts[-1].strip()

        return None

    def _extract_year(self, ref_text: str) -> Optional[str]:
        """Extract publication year"""
        year_match = re.search(r'\b(19|20)\d{2}\b', ref_text)
        return year_match.group(0) if year_match else None

    def _extract_venue(self, ref_text: str) -> Optional[str]:
        """Extract publication venue"""
        # Look for journal/conference names (italicized or after title)
        venue_patterns = [
            r'In\s+(.+?),\s+\d{4}',
            r'Journal of\s+(.+?)[,.]',
            r'Proceedings of\s+(.+?)[,.]',
        ]

        for pattern in venue_patterns:
            match = re.search(pattern, ref_text, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return None

    def _extract_doi(self, ref_text: str) -> Optional[str]:
        """Extract DOI"""
        doi_match = re.search(r'doi:?\s*(10\.\d+/[^\s]+)', ref_text, re.IGNORECASE)
        return doi_match.group(1) if doi_match else None

    def _extract_url(self, ref_text: str) -> Optional[str]:
        """Extract URL"""
        url_match = re.search(r'https?://[^\s]+', ref_text)
        return url_match.group(0) if url_match else None

    def _build_hierarchy(self, text: str) -> Dict[str, Any]:
        """Build document hierarchy tree"""
        sections = self._extract_sections(text)

        hierarchy = {
            'root': [],
            'levels': {}
        }

        stack = []

        for section in sections:
            level = section['level']

            # Pop stack until we find parent level
            while stack and stack[-1]['level'] >= level:
                stack.pop()

            # Add to hierarchy
            section_node = {
                'title': section['title'],
                'level': level,
                'children': []
            }

            if stack:
                stack[-1]['children'].append(section_node)
            else:
                hierarchy['root'].append(section_node)

            stack.append(section_node)

            # Track by level
            if level not in hierarchy['levels']:
                hierarchy['levels'][level] = []
            hierarchy['levels'][level].append(section['title'])

        return hierarchy

    def _extract_metadata(self, text: str) -> Dict[str, Any]:
        """Extract document metadata"""
        metadata = {
            'title': None,
            'authors': [],
            'abstract': None,
            'keywords': []
        }

        lines = text.split('\n')[:50]  # Check first 50 lines

        # Extract title (usually first non-empty line or after specific marker)
        for line in lines:
            line_stripped = line.strip()
            if line_stripped and len(line_stripped) > 10:
                if not metadata['title']:
                    metadata['title'] = line_stripped
                    break

        # Extract abstract
        abstract_match = re.search(
            r'Abstract[:\s]+(.*?)(?=\n\s*\n|Introduction|\d+\.)',
            text,
            re.IGNORECASE | re.DOTALL
        )
        if abstract_match:
            metadata['abstract'] = abstract_match.group(1).strip()

        # Extract keywords
        keywords_match = re.search(
            r'Keywords?[:\s]+(.+?)(?=\n\s*\n)',
            text,
            re.IGNORECASE
        )
        if keywords_match:
            keywords_text = keywords_match.group(1)
            metadata['keywords'] = [kw.strip() for kw in re.split(r'[,;]', keywords_text)]

        return metadata

    def _calculate_statistics(self, text: str) -> Dict[str, Any]:
        """Calculate document statistics"""
        lines = text.split('\n')
        words = text.split()
        sentences = re.split(r'[.!?]+', text)

        return {
            'total_characters': len(text),
            'total_lines': len(lines),
            'total_words': len(words),
            'total_sentences': len([s for s in sentences if s.strip()]),
            'avg_words_per_sentence': len(words) / max(len(sentences), 1),
            'avg_chars_per_word': len(text.replace(' ', '')) / max(len(words), 1)
        }


class CitationAnalyzer:
    """Analyze citation patterns and relationships"""

    def __init__(self):
        pass

    def analyze_citations(
        self,
        in_text_citations: List[Dict[str, Any]],
        references: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze citation patterns

        Args:
            in_text_citations: List of in-text citations
            references: List of references

        Returns:
            Citation analysis
        """
        analysis = {
            'total_citations': len(in_text_citations),
            'total_references': len(references),
            'citation_style': self._determine_citation_style(in_text_citations),
            'citation_distribution': self._analyze_distribution(in_text_citations),
            'missing_references': self._find_missing_references(in_text_citations, references),
            'unused_references': self._find_unused_references(in_text_citations, references),
            'citation_frequency': self._calculate_frequency(in_text_citations)
        }

        return analysis

    def _determine_citation_style(self, citations: List[Dict[str, Any]]) -> str:
        """Determine predominant citation style"""
        if not citations:
            return 'unknown'

        styles = {}
        for citation in citations:
            style = citation.get('type', 'unknown')
            styles[style] = styles.get(style, 0) + 1

        return max(styles.items(), key=lambda x: x[1])[0]

    def _analyze_distribution(self, citations: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze citation distribution across document"""
        # Group by position (beginning, middle, end)
        if not citations:
            return {'beginning': 0, 'middle': 0, 'end': 0}

        total_length = max([c['position'] for c in citations]) if citations else 1

        distribution = {'beginning': 0, 'middle': 0, 'end': 0}

        for citation in citations:
            pos_ratio = citation['position'] / total_length

            if pos_ratio < 0.33:
                distribution['beginning'] += 1
            elif pos_ratio < 0.67:
                distribution['middle'] += 1
            else:
                distribution['end'] += 1

        return distribution

    def _find_missing_references(
        self,
        citations: List[Dict[str, Any]],
        references: List[Dict[str, Any]]
    ) -> List[str]:
        """Find citations without corresponding references"""
        citation_keys = set(c['key'] for c in citations)
        reference_keys = set(r.get('raw_text', '')[:20] for r in references)

        # Simplified matching
        missing = []
        for key in citation_keys:
            if not any(key in ref_key for ref_key in reference_keys):
                missing.append(key)

        return missing

    def _find_unused_references(
        self,
        citations: List[Dict[str, Any]],
        references: List[Dict[str, Any]]
    ) -> List[str]:
        """Find references that are not cited"""
        citation_keys = set(c['key'] for c in citations)

        unused = []
        for i, ref in enumerate(references):
            ref_key = str(i + 1)  # Simple numeric matching
            if ref_key not in citation_keys:
                unused.append(ref.get('raw_text', '')[:100])

        return unused[:10]  # Limit output

    def _calculate_frequency(self, citations: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate citation frequency"""
        frequency = {}

        for citation in citations:
            key = citation['key']
            frequency[key] = frequency.get(key, 0) + 1

        # Return top 10 most cited
        sorted_freq = sorted(frequency.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_freq[:10])


__all__ = ['DocumentStructureAnalyzer', 'CitationAnalyzer']
