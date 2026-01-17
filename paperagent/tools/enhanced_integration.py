"""
Enhanced Multimodal Analysis Integration Module

This module integrates all enhanced analysis capabilities:
- Advanced LaTeX formatting
- Chart and graph understanding
- Mathematical formula recognition
- Table structure analysis
- Document structure analysis
- Academic visualizations
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
from loguru import logger

# Import all enhanced analyzers
from paperagent.tools.latex_advanced import (
    AdvancedLaTeXFormatter,
    ComplexTableFormatter,
    MathematicalFormatter
)
from paperagent.tools.chart_analyzer import ChartAnalyzer
from paperagent.tools.formula_recognizer import FormulaRecognizer, TableStructureAnalyzer
from paperagent.tools.document_structure import DocumentStructureAnalyzer, CitationAnalyzer
from paperagent.tools.visualization import AcademicVisualizer, InteractiveVisualizer
from paperagent.tools.multimodal_analyzer import (
    TextAnalyzer,
    ImageAnalyzer,
    CodeAnalyzer,
    PDFDeepAnalyzer
)


class EnhancedMultimodalProcessor:
    """
    Unified interface for all enhanced multimodal analysis capabilities

    This class provides a single entry point for:
    - Document analysis (structure, citations, references)
    - Image analysis (charts, formulas, tables, general images)
    - Code analysis (complexity, quality, structure)
    - LaTeX generation (documents, equations, tables, algorithms)
    - Visualization creation (static and interactive)
    """

    def __init__(self):
        """Initialize all analyzers"""
        # Text and document analysis
        self.text_analyzer = TextAnalyzer()
        self.document_analyzer = DocumentStructureAnalyzer()
        self.citation_analyzer = CitationAnalyzer()

        # Image and visual analysis
        self.image_analyzer = ImageAnalyzer()
        self.chart_analyzer = ChartAnalyzer()
        self.formula_recognizer = FormulaRecognizer()
        self.table_analyzer = TableStructureAnalyzer()

        # Code analysis
        self.code_analyzer = CodeAnalyzer()

        # PDF analysis
        self.pdf_analyzer = PDFDeepAnalyzer()

        # LaTeX formatting
        self.latex_formatter = AdvancedLaTeXFormatter()
        self.table_formatter = ComplexTableFormatter()
        self.math_formatter = MathematicalFormatter()

        # Visualization
        self.visualizer = AcademicVisualizer()
        self.interactive_viz = InteractiveVisualizer()

        logger.info("Enhanced multimodal processor initialized successfully")

    def analyze_document(self, file_path: str) -> Dict[str, Any]:
        """
        Comprehensive document analysis

        Args:
            file_path: Path to document file

        Returns:
            Complete document analysis including structure, citations, and content
        """
        try:
            result = self.document_analyzer.analyze_document(file_path)

            # Add citation analysis if citations found
            if result.get('citations') and result.get('references'):
                result['citation_analysis'] = self.citation_analyzer.analyze_citations(
                    result['citations'],
                    result['references']
                )

            return {
                'status': 'success',
                'analysis': result
            }
        except Exception as e:
            logger.error(f"Document analysis error: {e}")
            return {'status': 'error', 'message': str(e)}

    def analyze_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Deep PDF analysis with text, images, and tables

        Args:
            pdf_path: Path to PDF file

        Returns:
            Comprehensive PDF analysis results
        """
        try:
            result = self.pdf_analyzer.analyze_pdf(pdf_path)
            return {
                'status': 'success',
                'analysis': result
            }
        except Exception as e:
            logger.error(f"PDF analysis error: {e}")
            return {'status': 'error', 'message': str(e)}

    def analyze_chart(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze chart or graph

        Args:
            image_path: Path to chart image

        Returns:
            Chart analysis including type, data, and insights
        """
        try:
            result = self.chart_analyzer.analyze_chart(image_path)
            return {
                'status': 'success',
                'analysis': result
            }
        except Exception as e:
            logger.error(f"Chart analysis error: {e}")
            return {'status': 'error', 'message': str(e)}

    def recognize_formula(self, image_path: str) -> Dict[str, Any]:
        """
        Recognize mathematical formula from image

        Args:
            image_path: Path to formula image

        Returns:
            Formula recognition results with LaTeX conversion
        """
        try:
            result = self.formula_recognizer.recognize_formula(image_path)
            return {
                'status': 'success',
                'formula': result
            }
        except Exception as e:
            logger.error(f"Formula recognition error: {e}")
            return {'status': 'error', 'message': str(e)}

    def analyze_table_image(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze table structure from image

        Args:
            image_path: Path to table image

        Returns:
            Table structure and extracted data
        """
        try:
            result = self.table_analyzer.analyze_table(image_path)
            return {
                'status': 'success',
                'table': result
            }
        except Exception as e:
            logger.error(f"Table analysis error: {e}")
            return {'status': 'error', 'message': str(e)}

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Comprehensive text analysis

        Args:
            text: Input text

        Returns:
            Text analysis including statistics, sentiment, readability, entities
        """
        try:
            result = self.text_analyzer.analyze_text(text)
            return {
                'status': 'success',
                'analysis': result
            }
        except Exception as e:
            logger.error(f"Text analysis error: {e}")
            return {'status': 'error', 'message': str(e)}

    def analyze_code(self, code: str, language: str = 'python') -> Dict[str, Any]:
        """
        Analyze source code

        Args:
            code: Source code
            language: Programming language

        Returns:
            Code analysis including complexity, quality, structure
        """
        try:
            result = self.code_analyzer.analyze_code(code, language)
            return {
                'status': 'success',
                'analysis': result
            }
        except Exception as e:
            logger.error(f"Code analysis error: {e}")
            return {'status': 'error', 'message': str(e)}

    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """
        Comprehensive image analysis

        Args:
            image_path: Path to image

        Returns:
            Image analysis including properties, OCR text, content analysis
        """
        try:
            result = self.image_analyzer.analyze_image(image_path)
            return {
                'status': 'success',
                'analysis': result
            }
        except Exception as e:
            logger.error(f"Image analysis error: {e}")
            return {'status': 'error', 'message': str(e)}

    def create_algorithm_latex(
        self,
        title: str,
        inputs: List[str],
        outputs: List[str],
        steps: List[str],
        label: Optional[str] = None
    ) -> str:
        """
        Create LaTeX algorithm environment

        Args:
            title: Algorithm title
            inputs: Input parameters
            outputs: Output parameters
            steps: Algorithm steps
            label: Optional label

        Returns:
            LaTeX code for algorithm
        """
        return self.latex_formatter.create_algorithm(title, inputs, outputs, steps, label)

    def create_theorem_latex(
        self,
        env_type: str,
        content: str,
        label: Optional[str] = None,
        title: Optional[str] = None
    ) -> str:
        """
        Create LaTeX theorem environment

        Args:
            env_type: theorem, lemma, proposition, corollary, definition
            content: Theorem content
            label: Optional label
            title: Optional title

        Returns:
            LaTeX code for theorem
        """
        return self.latex_formatter.create_theorem_environment(env_type, content, label, title)

    def create_complex_table_latex(
        self,
        df,
        caption: str,
        label: str,
        use_booktabs: bool = True
    ) -> str:
        """
        Create professional LaTeX table

        Args:
            df: Pandas DataFrame
            caption: Table caption
            label: Table label
            use_booktabs: Use booktabs package

        Returns:
            LaTeX table code
        """
        if use_booktabs:
            return self.table_formatter.create_booktabs_table(df, caption, label)
        else:
            # Use standard table
            return self.table_formatter.create_longtable(df, caption, label)

    def create_visualization(
        self,
        plot_type: str,
        data: Any,
        **kwargs
    ) -> str:
        """
        Create academic visualization

        Args:
            plot_type: Type of plot (scatter, bar, line, heatmap, box, histogram)
            data: Plot data
            **kwargs: Additional plot parameters

        Returns:
            Path to saved figure
        """
        try:
            if plot_type == 'scatter':
                return self.visualizer.create_scatter_plot(**data, **kwargs)
            elif plot_type == 'bar':
                return self.visualizer.create_bar_chart(**data, **kwargs)
            elif plot_type == 'line':
                return self.visualizer.create_line_plot(**data, **kwargs)
            elif plot_type == 'heatmap':
                return self.visualizer.create_heatmap(**data, **kwargs)
            elif plot_type == 'box':
                return self.visualizer.create_box_plot(**data, **kwargs)
            elif plot_type == 'histogram':
                return self.visualizer.create_histogram(**data, **kwargs)
            elif plot_type == 'correlation':
                return self.visualizer.create_correlation_matrix(**data, **kwargs)
            else:
                raise ValueError(f"Unknown plot type: {plot_type}")
        except Exception as e:
            logger.error(f"Visualization error: {e}")
            return ""

    def create_interactive_visualization(
        self,
        plot_type: str,
        df,
        **kwargs
    ) -> str:
        """
        Create interactive visualization

        Args:
            plot_type: Type of plot (scatter, 3d_scatter, line)
            df: DataFrame with data
            **kwargs: Plot parameters

        Returns:
            Path to saved HTML file
        """
        try:
            if plot_type == 'scatter':
                return self.interactive_viz.create_interactive_scatter(df, **kwargs)
            elif plot_type == '3d_scatter':
                return self.interactive_viz.create_3d_scatter(df, **kwargs)
            elif plot_type == 'line':
                return self.interactive_viz.create_interactive_line(df, **kwargs)
            else:
                raise ValueError(f"Unknown interactive plot type: {plot_type}")
        except Exception as e:
            logger.error(f"Interactive visualization error: {e}")
            return ""

    def process_multimodal_content(
        self,
        content_path: str,
        content_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Automatically detect and process multimodal content

        Args:
            content_path: Path to content file
            content_type: Optional type hint (auto-detect if None)

        Returns:
            Processing results based on content type
        """
        path = Path(content_path)

        if content_type is None:
            # Auto-detect based on extension
            ext = path.suffix.lower()
            if ext == '.pdf':
                content_type = 'pdf'
            elif ext in ['.png', '.jpg', '.jpeg', '.bmp']:
                content_type = 'image'
            elif ext in ['.py', '.java', '.cpp', '.js']:
                content_type = 'code'
            elif ext in ['.txt', '.md']:
                content_type = 'text'
            else:
                content_type = 'unknown'

        # Process based on type
        if content_type == 'pdf':
            return self.analyze_pdf(str(path))
        elif content_type == 'image':
            # Try to determine if it's a chart, formula, or table
            result = {
                'image_analysis': self.analyze_image(str(path)),
                'chart_analysis': self.analyze_chart(str(path)),
                'formula_analysis': self.recognize_formula(str(path)),
                'table_analysis': self.analyze_table_image(str(path))
            }
            return {'status': 'success', 'results': result}
        elif content_type == 'code':
            with open(path, 'r', encoding='utf-8') as f:
                code = f.read()
            language = path.suffix[1:]  # Remove dot
            return self.analyze_code(code, language)
        elif content_type == 'text':
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
            return self.analyze_text(text)
        elif content_type == 'document':
            return self.analyze_document(str(path))
        else:
            return {'status': 'error', 'message': f'Unknown content type: {content_type}'}


# Create singleton instance for easy access
enhanced_processor = EnhancedMultimodalProcessor()


__all__ = ['EnhancedMultimodalProcessor', 'enhanced_processor']
