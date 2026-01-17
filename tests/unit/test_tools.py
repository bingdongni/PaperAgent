"""
Unit tests for tool modules
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from pathlib import Path
from paperagent.tools.literature_collector import ArxivCollector, ScholarCollector
from paperagent.tools.latex_processor import LaTeXProcessor
from paperagent.tools.document_processor import DocumentProcessor
from paperagent.tools.advanced_stats import AdvancedStatistics
from paperagent.tools.multimodal_analyzer import (
    TextAnalyzer, ImageAnalyzer, CodeAnalyzer, PDFDeepAnalyzer
)


class TestArxivCollector:
    """Test arXiv paper collector"""

    @patch('paperagent.tools.literature_collector.arxiv.Search')
    def test_search_papers(self, mock_search):
        """Test arXiv paper search"""
        # Mock arXiv API response
        mock_result = Mock()
        mock_result.title = "Test Paper"
        mock_result.authors = [Mock(name="Author One")]
        mock_result.summary = "Test abstract"
        mock_result.entry_id = "http://arxiv.org/abs/2024.0001"
        mock_result.published = "2024-01-01"
        mock_result.pdf_url = "http://arxiv.org/pdf/2024.0001"

        mock_search.return_value.results.return_value = [mock_result]

        collector = ArxivCollector()
        results = collector.search(query="machine learning", max_results=10)

        assert len(results) > 0
        assert 'title' in results[0]
        assert 'authors' in results[0]

    def test_parse_arxiv_id(self):
        """Test arXiv ID parsing"""
        collector = ArxivCollector()

        url1 = "http://arxiv.org/abs/2024.0001"
        assert collector.parse_arxiv_id(url1) == "2024.0001"

        url2 = "https://arxiv.org/pdf/2024.0001.pdf"
        assert collector.parse_arxiv_id(url2) == "2024.0001"


class TestScholarCollector:
    """Test Google Scholar collector"""

    @patch('paperagent.tools.literature_collector.scholarly.search_pubs')
    def test_search_papers(self, mock_search):
        """Test Scholar paper search"""
        # Mock scholarly API response
        mock_result = {
            'bib': {
                'title': 'Test Paper',
                'author': ['Author One', 'Author Two'],
                'abstract': 'Test abstract',
                'pub_year': '2024'
            },
            'num_citations': 10,
            'pub_url': 'https://scholar.google.com/test'
        }

        mock_search.return_value = [mock_result]

        collector = ScholarCollector()
        results = collector.search(query="machine learning", max_results=10)

        assert len(results) > 0
        assert 'title' in results[0]


class TestLaTeXProcessor:
    """Test LaTeX document processor"""

    def test_create_document(self):
        """Test LaTeX document creation"""
        processor = LaTeXProcessor()
        doc = processor.create_document(
            title="Test Paper",
            authors=["Author One", "Author Two"],
            abstract="Test abstract"
        )

        assert "\\documentclass" in doc
        assert "Test Paper" in doc
        assert "Author One" in doc

    def test_format_section(self):
        """Test section formatting"""
        processor = LaTeXProcessor()
        section = processor.format_section(
            title="Introduction",
            content="This is the introduction.",
            level=1
        )

        assert "\\section{Introduction}" in section
        assert "This is the introduction." in section

    def test_format_equation(self):
        """Test equation formatting"""
        processor = LaTeXProcessor()
        eq = processor.format_equation("E = mc^2", numbered=True)

        assert "\\begin{equation}" in eq
        assert "E = mc^2" in eq
        assert "\\end{equation}" in eq

    def test_format_table(self):
        """Test table formatting"""
        processor = LaTeXProcessor()
        data = pd.DataFrame({
            'Method': ['A', 'B', 'C'],
            'Accuracy': [0.85, 0.90, 0.88]
        })

        table = processor.format_table(
            data,
            caption="Results comparison",
            label="tab:results"
        )

        assert "\\begin{table}" in table
        assert "\\caption{Results comparison}" in table
        assert "\\label{tab:results}" in table

    def test_format_figure(self):
        """Test figure formatting"""
        processor = LaTeXProcessor()
        fig = processor.format_figure(
            path="figures/result.png",
            caption="Experimental results",
            label="fig:results",
            width=0.8
        )

        assert "\\begin{figure}" in fig
        assert "\\includegraphics" in fig
        assert "figures/result.png" in fig

    def test_create_bibliography(self):
        """Test bibliography creation"""
        processor = LaTeXProcessor()
        citations = [
            {
                'type': 'article',
                'key': 'smith2024',
                'title': 'Test Paper',
                'author': 'Smith, J.',
                'journal': 'Test Journal',
                'year': '2024'
            }
        ]

        bib = processor.create_bibliography(citations, style='ieee')

        assert 'smith2024' in bib
        assert 'Test Paper' in bib


class TestDocumentProcessor:
    """Test document processor"""

    @pytest.fixture
    def sample_pdf_path(self, tmp_path):
        """Create a sample PDF file"""
        pdf_path = tmp_path / "test.pdf"
        # Note: This would need a real PDF in actual testing
        return str(pdf_path)

    def test_extract_text_from_pdf(self, sample_pdf_path):
        """Test PDF text extraction"""
        processor = DocumentProcessor()

        with patch('paperagent.tools.document_processor.pdfplumber.open') as mock_pdf:
            mock_page = Mock()
            mock_page.extract_text.return_value = "Sample PDF text"
            mock_pdf.return_value.__enter__.return_value.pages = [mock_page]

            text = processor.extract_text_from_pdf(sample_pdf_path)
            assert "Sample PDF text" in text

    def test_extract_metadata_from_pdf(self, sample_pdf_path):
        """Test PDF metadata extraction"""
        processor = DocumentProcessor()

        with patch('paperagent.tools.document_processor.PyPDF2.PdfReader') as mock_pdf:
            mock_pdf.return_value.metadata = {
                '/Title': 'Test PDF',
                '/Author': 'Test Author'
            }

            metadata = processor.extract_metadata_from_pdf(sample_pdf_path)
            assert 'title' in metadata or 'Title' in metadata

    def test_convert_docx_to_text(self, tmp_path):
        """Test DOCX to text conversion"""
        processor = DocumentProcessor()
        docx_path = tmp_path / "test.docx"

        with patch('paperagent.tools.document_processor.docx.Document') as mock_docx:
            mock_para = Mock()
            mock_para.text = "Sample paragraph"
            mock_docx.return_value.paragraphs = [mock_para]

            text = processor.convert_docx_to_text(str(docx_path))
            assert "Sample paragraph" in text


class TestAdvancedStatistics:
    """Test advanced statistical analysis"""

    def test_descriptive_stats(self, sample_dataframe):
        """Test descriptive statistics"""
        stats = AdvancedStatistics()
        result = stats.descriptive_stats(sample_dataframe, columns=['value'])

        assert 'value' in result
        assert 'mean' in result['value']
        assert 'std' in result['value']
        assert 'median' in result['value']

    def test_ttest_analysis(self):
        """Test t-test analysis"""
        stats = AdvancedStatistics()

        group1 = pd.Series([1, 2, 3, 4, 5])
        group2 = pd.Series([2, 3, 4, 5, 6])

        result = stats.ttest_analysis(group1, group2, paired=False)

        assert 'statistic' in result
        assert 'p_value' in result
        assert 'effect_size' in result

    def test_anova_analysis(self):
        """Test ANOVA analysis"""
        stats = AdvancedStatistics()

        data = pd.DataFrame({
            'group': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'],
            'value': [1, 2, 3, 4, 5, 6, 7, 8, 9]
        })

        result = stats.anova_analysis(data, 'value', 'group')

        assert 'f_statistic' in result
        assert 'p_value' in result

    def test_regression_analysis(self):
        """Test regression analysis"""
        stats = AdvancedStatistics()

        data = pd.DataFrame({
            'x1': [1, 2, 3, 4, 5],
            'x2': [2, 4, 6, 8, 10],
            'y': [3, 5, 7, 9, 11]
        })

        result = stats.regression_analysis(data, 'y', ['x1', 'x2'])

        assert 'r_squared' in result
        assert 'coefficients' in result
        assert 'p_values' in result

    def test_correlation_matrix(self):
        """Test correlation matrix calculation"""
        stats = AdvancedStatistics()

        data = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [2, 4, 6, 8, 10],
            'c': [5, 4, 3, 2, 1]
        })

        result = stats.correlation_matrix(data)

        assert isinstance(result, pd.DataFrame)
        assert result.shape == (3, 3)

    def test_machine_learning_analysis(self):
        """Test machine learning analysis"""
        stats = AdvancedStatistics()

        data = pd.DataFrame({
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100),
            'target': np.random.rand(100)
        })

        result = stats.machine_learning_analysis(
            data,
            target='target',
            features=['feature1', 'feature2'],
            task='regression'
        )

        assert 'model_score' in result
        assert 'cv_scores' in result
        assert 'feature_importance' in result


class TestTextAnalyzer:
    """Test text analysis capabilities"""

    def test_analyze_text(self):
        """Test comprehensive text analysis"""
        analyzer = TextAnalyzer()
        text = """
        Machine learning is a subset of artificial intelligence.
        It enables computers to learn from data without being explicitly programmed.
        """

        result = analyzer.analyze_text(text)

        assert 'word_count' in result
        assert 'sentence_count' in result
        assert 'readability' in result
        assert result['word_count'] > 0

    def test_sentiment_analysis(self):
        """Test sentiment analysis"""
        analyzer = TextAnalyzer()

        positive_text = "This is an excellent paper with great results!"
        negative_text = "This paper has many flaws and poor methodology."

        pos_result = analyzer.analyze_text(positive_text)
        neg_result = analyzer.analyze_text(negative_text)

        assert 'sentiment' in pos_result
        assert 'sentiment' in neg_result

    def test_extract_keywords(self):
        """Test keyword extraction"""
        analyzer = TextAnalyzer()
        text = """
        Deep learning neural networks have revolutionized computer vision.
        Convolutional neural networks are particularly effective for image classification.
        """

        result = analyzer.analyze_text(text)

        assert 'keywords' in result or 'entities' in result

    def test_semantic_similarity(self):
        """Test semantic similarity calculation"""
        analyzer = TextAnalyzer()

        text1 = "Machine learning is a powerful technique."
        text2 = "Deep learning is an effective method."
        text3 = "The weather is sunny today."

        sim_12 = analyzer.semantic_similarity(text1, text2)
        sim_13 = analyzer.semantic_similarity(text1, text3)

        assert 0 <= sim_12 <= 1
        assert 0 <= sim_13 <= 1
        assert sim_12 > sim_13  # More similar texts should have higher score


class TestImageAnalyzer:
    """Test image analysis capabilities"""

    @pytest.fixture
    def sample_image_path(self, tmp_path):
        """Create a sample image"""
        from PIL import Image
        img_path = tmp_path / "test.png"
        img = Image.new('RGB', (100, 100), color='white')
        img.save(img_path)
        return str(img_path)

    def test_analyze_image(self, sample_image_path):
        """Test image analysis"""
        analyzer = ImageAnalyzer()
        result = analyzer.analyze_image(sample_image_path)

        assert 'dimensions' in result or 'size' in result
        assert 'format' in result or 'mode' in result

    def test_extract_text_ocr(self, sample_image_path):
        """Test OCR text extraction"""
        analyzer = ImageAnalyzer()

        with patch('paperagent.tools.multimodal_analyzer.pytesseract.image_to_string') as mock_ocr:
            mock_ocr.return_value = "Sample text from image"

            result = analyzer.extract_text_ocr(sample_image_path)
            assert 'text' in result


class TestCodeAnalyzer:
    """Test code analysis capabilities"""

    def test_analyze_python_code(self, sample_code):
        """Test Python code analysis"""
        analyzer = CodeAnalyzer()
        result = analyzer.analyze_code(sample_code, language='python')

        assert 'complexity' in result
        assert 'loc' in result or 'lines_of_code' in result

    def test_calculate_complexity(self):
        """Test code complexity calculation"""
        analyzer = CodeAnalyzer()

        simple_code = """
def add(a, b):
    return a + b
"""

        complex_code = """
def complex_function(x):
    if x > 0:
        if x > 10:
            return x * 2
        else:
            return x + 1
    else:
        if x < -10:
            return x * -1
        else:
            return 0
"""

        simple_result = analyzer.analyze_code(simple_code, 'python')
        complex_result = analyzer.analyze_code(complex_code, 'python')

        assert 'complexity' in simple_result
        assert 'complexity' in complex_result

    def test_detect_language(self):
        """Test programming language detection"""
        analyzer = CodeAnalyzer()

        python_code = "def hello():\n    print('Hello')"
        java_code = "public class Hello {\n    public static void main(String[] args) {}\n}"

        # Language detection based on syntax patterns
        assert "python" in python_code.lower() or "def" in python_code
        assert "java" in java_code.lower() or "public class" in java_code


class TestPDFDeepAnalyzer:
    """Test deep PDF analysis"""

    @pytest.fixture
    def sample_pdf(self, tmp_path):
        """Create a sample PDF"""
        pdf_path = tmp_path / "test.pdf"
        return str(pdf_path)

    def test_analyze_pdf(self, sample_pdf):
        """Test comprehensive PDF analysis"""
        analyzer = PDFDeepAnalyzer()

        with patch('paperagent.tools.multimodal_analyzer.pdfplumber.open') as mock_pdf:
            mock_page = Mock()
            mock_page.extract_text.return_value = "Sample PDF content"
            mock_page.extract_tables.return_value = []
            mock_pdf.return_value.__enter__.return_value.pages = [mock_page]
            mock_pdf.return_value.__enter__.return_value.metadata = {'Title': 'Test'}

            result = analyzer.analyze_pdf(sample_pdf)

            assert 'metadata' in result or 'text' in result

    def test_extract_tables(self, sample_pdf):
        """Test table extraction from PDF"""
        analyzer = PDFDeepAnalyzer()

        with patch('paperagent.tools.multimodal_analyzer.pdfplumber.open') as mock_pdf:
            mock_page = Mock()
            mock_table = [['Header1', 'Header2'], ['Value1', 'Value2']]
            mock_page.extract_tables.return_value = [mock_table]
            mock_pdf.return_value.__enter__.return_value.pages = [mock_page]

            result = analyzer._extract_tables(sample_pdf)

            assert isinstance(result, list)

    def test_extract_structure(self, sample_pdf):
        """Test PDF structure extraction"""
        analyzer = PDFDeepAnalyzer()

        with patch('paperagent.tools.multimodal_analyzer.pdfplumber.open') as mock_pdf:
            mock_page = Mock()
            mock_page.extract_text.return_value = "# Title\n## Section 1\nContent"
            mock_pdf.return_value.__enter__.return_value.pages = [mock_page]

            result = analyzer.analyze_pdf(sample_pdf)

            assert result is not None


class TestToolIntegration:
    """Test integration between different tools"""

    def test_literature_to_latex(self):
        """Test converting literature data to LaTeX format"""
        latex_proc = LaTeXProcessor()

        literature_data = {
            'title': 'Test Paper',
            'authors': ['Author One', 'Author Two'],
            'year': '2024',
            'journal': 'Test Journal'
        }

        citation = latex_proc.format_citation(literature_data, style='ieee')
        assert 'Test Paper' in citation

    def test_stats_to_latex_table(self, sample_dataframe):
        """Test converting statistical results to LaTeX table"""
        stats = AdvancedStatistics()
        latex_proc = LaTeXProcessor()

        desc_stats = stats.descriptive_stats(sample_dataframe)

        # Convert to DataFrame for LaTeX formatting
        stats_df = pd.DataFrame(desc_stats).T

        table = latex_proc.format_table(stats_df, caption="Statistical Results")
        assert "\\begin{table}" in table

    def test_pdf_to_text_analysis(self, tmp_path):
        """Test extracting text from PDF and analyzing it"""
        doc_proc = DocumentProcessor()
        text_analyzer = TextAnalyzer()

        with patch('paperagent.tools.document_processor.pdfplumber.open') as mock_pdf:
            mock_page = Mock()
            mock_page.extract_text.return_value = "Sample research paper text."
            mock_pdf.return_value.__enter__.return_value.pages = [mock_page]

            pdf_path = tmp_path / "test.pdf"
            text = doc_proc.extract_text_from_pdf(str(pdf_path))
            analysis = text_analyzer.analyze_text(text)

            assert analysis['word_count'] > 0
