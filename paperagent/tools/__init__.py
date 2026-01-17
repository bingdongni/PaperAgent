"""
Tools module for PaperAgent
"""

from paperagent.tools.literature_collector import (
    ArxivCollector,
    GoogleScholarCollector,
    LiteratureCollector,
)
from paperagent.tools.latex_processor import LaTeXProcessor, BibTeXManager
from paperagent.tools.document_processor import (
    PDFProcessor,
    DOCXProcessor,
    TextProcessor,
)

__all__ = [
    "ArxivCollector",
    "GoogleScholarCollector",
    "LiteratureCollector",
    "LaTeXProcessor",
    "BibTeXManager",
    "PDFProcessor",
    "DOCXProcessor",
    "TextProcessor",
]
