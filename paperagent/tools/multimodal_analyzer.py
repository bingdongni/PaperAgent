"""
Multimodal Analysis Tools

Supports deep analysis of:
- Text (NLP, sentiment, topic modeling)
- Images (OCR, object detection, analysis)
- Code (complexity, quality, documentation)
- Documents (PDFs, presentations)
"""

import os
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import pytesseract
from pdf2image import convert_from_path
import pdfplumber
from pdfminer.high_level import extract_text as pdfminer_extract_text
from pdfminer.layout import LAParams
import pandas as pd

# NLP
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
import spacy
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import textstat

# Code analysis
import ast
from pygments import highlight
from pygments.lexers import get_lexer_by_name, guess_lexer
from pygments.formatters import HtmlFormatter
from radon.complexity import cc_visit
from radon.metrics import mi_visit, h_visit

from loguru import logger


class TextAnalyzer:
    """Advanced text analysis with NLP"""

    def __init__(self):
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('vader_lexicon', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
        except:
            pass

        self.sentiment_analyzer = SentimentIntensityAnalyzer()

        # Try to load spaCy model
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except:
            logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Comprehensive text analysis

        Args:
            text: Input text

        Returns:
            Analysis results
        """
        results = {}

        # Basic statistics
        results['statistics'] = self._basic_stats(text)

        # Sentiment analysis
        results['sentiment'] = self._sentiment_analysis(text)

        # Readability metrics
        results['readability'] = self._readability_metrics(text)

        # Named entities (if spaCy available)
        if self.nlp:
            results['entities'] = self._extract_entities(text)

        # Key phrases
        results['key_phrases'] = self._extract_key_phrases(text)

        return results

    def _basic_stats(self, text: str) -> Dict[str, Any]:
        """Calculate basic text statistics"""
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        words_lower = [w.lower() for w in words if w.isalnum()]

        # Remove stopwords
        try:
            stop_words = set(stopwords.words('english'))
            content_words = [w for w in words_lower if w not in stop_words]
        except:
            content_words = words_lower

        # Character count
        char_count = len(text)
        char_no_spaces = len(text.replace(' ', ''))

        return {
            'char_count': char_count,
            'char_count_no_spaces': char_no_spaces,
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
            'avg_sentence_length': len(words) / len(sentences) if sentences else 0,
            'unique_words': len(set(words_lower)),
            'lexical_diversity': len(set(words_lower)) / len(words_lower) if words_lower else 0
        }

    def _sentiment_analysis(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment"""
        scores = self.sentiment_analyzer.polarity_scores(text)

        # Determine overall sentiment
        if scores['compound'] >= 0.05:
            sentiment = 'positive'
        elif scores['compound'] <= -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'

        return {
            'positive': scores['pos'],
            'negative': scores['neg'],
            'neutral': scores['neu'],
            'compound': scores['compound'],
            'overall': sentiment
        }

    def _readability_metrics(self, text: str) -> Dict[str, Any]:
        """Calculate readability metrics"""
        return {
            'flesch_reading_ease': textstat.flesch_reading_ease(text),
            'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
            'gunning_fog': textstat.gunning_fog(text),
            'smog_index': textstat.smog_index(text),
            'automated_readability_index': textstat.automated_readability_index(text),
            'coleman_liau_index': textstat.coleman_liau_index(text),
            'text_standard': textstat.text_standard(text, float_output=False)
        }

    def _extract_entities(self, text: str) -> List[Dict[str, str]]:
        """Extract named entities using spaCy"""
        if not self.nlp:
            return []

        doc = self.nlp(text[:1000000])  # Limit to 1M chars

        entities = []
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            })

        return entities

    def _extract_key_phrases(self, text: str, top_n: int = 10) -> List[str]:
        """Extract key phrases using simple frequency"""
        words = word_tokenize(text.lower())

        try:
            stop_words = set(stopwords.words('english'))
            words = [w for w in words if w.isalnum() and w not in stop_words and len(w) > 3]
        except:
            words = [w for w in words if w.isalnum() and len(w) > 3]

        # Count frequency
        from collections import Counter
        word_freq = Counter(words)

        return [word for word, _ in word_freq.most_common(top_n)]

    def semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score (0-1)
        """
        try:
            model = SentenceTransformer('all-MiniLM-L6-v2')
            embeddings = model.encode([text1, text2])

            # Cosine similarity
            from sklearn.metrics.pairwise import cosine_similarity
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

            return float(similarity)
        except Exception as e:
            logger.error(f"Semantic similarity error: {e}")
            return 0.0


class ImageAnalyzer:
    """Image analysis and OCR"""

    def __init__(self):
        pass

    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """
        Comprehensive image analysis

        Args:
            image_path: Path to image file

        Returns:
            Analysis results
        """
        results = {}

        # Load image
        img = Image.open(image_path)
        cv_img = cv2.imread(image_path)

        # Basic properties
        results['properties'] = {
            'format': img.format,
            'mode': img.mode,
            'size': img.size,
            'width': img.width,
            'height': img.height
        }

        # OCR text extraction
        results['text'] = self.extract_text_ocr(image_path)

        # Basic image analysis
        results['analysis'] = self._analyze_image_content(cv_img)

        return results

    def extract_text_ocr(self, image_path: str) -> Dict[str, Any]:
        """
        Extract text from image using OCR

        Args:
            image_path: Path to image

        Returns:
            Extracted text and confidence
        """
        try:
            img = Image.open(image_path)

            # Extract text
            text = pytesseract.image_to_string(img)

            # Get detailed data
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

            # Calculate average confidence
            confidences = [int(conf) for conf in data['conf'] if conf != '-1']
            avg_confidence = np.mean(confidences) if confidences else 0

            return {
                'text': text,
                'word_count': len(text.split()),
                'avg_confidence': float(avg_confidence),
                'has_text': len(text.strip()) > 0
            }
        except Exception as e:
            logger.error(f"OCR error: {e}")
            return {'text': '', 'word_count': 0, 'avg_confidence': 0, 'has_text': False}

    def _analyze_image_content(self, img: np.ndarray) -> Dict[str, Any]:
        """Analyze image content"""
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Calculate statistics
        brightness = np.mean(gray)
        contrast = np.std(gray)

        # Detect edges
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size

        return {
            'brightness': float(brightness),
            'contrast': float(contrast),
            'edge_density': float(edge_density),
            'is_likely_diagram': edge_density > 0.1,
            'is_likely_photo': edge_density < 0.05 and contrast > 50
        }


class CodeAnalyzer:
    """Code analysis and quality metrics"""

    def __init__(self):
        pass

    def analyze_code(self, code: str, language: str = 'python') -> Dict[str, Any]:
        """
        Comprehensive code analysis

        Args:
            code: Source code
            language: Programming language

        Returns:
            Analysis results
        """
        results = {}

        # Basic statistics
        results['statistics'] = self._code_statistics(code)

        if language.lower() == 'python':
            # Python-specific analysis
            results['complexity'] = self._calculate_complexity(code)
            results['quality'] = self._analyze_quality(code)
            results['structure'] = self._analyze_structure(code)

        # Syntax highlighting
        results['highlighted'] = self._highlight_code(code, language)

        return results

    def _code_statistics(self, code: str) -> Dict[str, Any]:
        """Calculate basic code statistics"""
        lines = code.split('\n')

        # Count different line types
        code_lines = [l for l in lines if l.strip() and not l.strip().startswith('#')]
        comment_lines = [l for l in lines if l.strip().startswith('#')]
        blank_lines = [l for l in lines if not l.strip()]

        return {
            'total_lines': len(lines),
            'code_lines': len(code_lines),
            'comment_lines': len(comment_lines),
            'blank_lines': len(blank_lines),
            'comment_ratio': len(comment_lines) / len(code_lines) if code_lines else 0
        }

    def _calculate_complexity(self, code: str) -> Dict[str, Any]:
        """Calculate cyclomatic complexity"""
        try:
            complexity_results = cc_visit(code)

            complexities = []
            for item in complexity_results:
                complexities.append({
                    'name': item.name,
                    'complexity': item.complexity,
                    'type': item.classname or 'function'
                })

            avg_complexity = np.mean([c['complexity'] for c in complexities]) if complexities else 0

            return {
                'functions': complexities,
                'average_complexity': float(avg_complexity),
                'max_complexity': max([c['complexity'] for c in complexities]) if complexities else 0,
                'total_functions': len(complexities)
            }
        except Exception as e:
            logger.error(f"Complexity analysis error: {e}")
            return {}

    def _analyze_quality(self, code: str) -> Dict[str, Any]:
        """Analyze code quality metrics"""
        try:
            # Maintainability Index
            mi = mi_visit(code, multi=True)

            # Halstead metrics
            h = h_visit(code)

            return {
                'maintainability_index': float(mi) if isinstance(mi, (int, float)) else 0,
                'halstead_metrics': {
                    'volume': h.total.volume if h else 0,
                    'difficulty': h.total.difficulty if h else 0,
                    'effort': h.total.effort if h else 0
                } if h else {}
            }
        except Exception as e:
            logger.error(f"Quality analysis error: {e}")
            return {}

    def _analyze_structure(self, code: str) -> Dict[str, Any]:
        """Analyze code structure"""
        try:
            tree = ast.parse(code)

            # Count different node types
            functions = []
            classes = []
            imports = []

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append(node.name)
                elif isinstance(node, ast.ClassDef):
                    classes.append(node.name)
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    imports.append(ast.unparse(node))

            return {
                'num_functions': len(functions),
                'num_classes': len(classes),
                'num_imports': len(imports),
                'function_names': functions,
                'class_names': classes
            }
        except Exception as e:
            logger.error(f"Structure analysis error: {e}")
            return {}

    def _highlight_code(self, code: str, language: str) -> str:
        """Syntax highlight code"""
        try:
            lexer = get_lexer_by_name(language, stripall=True)
            formatter = HtmlFormatter(style='colorful', full=False)
            return highlight(code, lexer, formatter)
        except:
            return code


class PDFDeepAnalyzer:
    """Deep PDF analysis with text, images, and tables"""

    def __init__(self):
        self.text_analyzer = TextAnalyzer()
        self.image_analyzer = ImageAnalyzer()

    def analyze_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Comprehensive PDF analysis

        Args:
            pdf_path: Path to PDF file

        Returns:
            Complete analysis results
        """
        results = {
            'metadata': self._extract_metadata(pdf_path),
            'text': self._extract_and_analyze_text(pdf_path),
            'tables': self._extract_tables(pdf_path),
            'images': self._extract_and_analyze_images(pdf_path),
            'structure': self._analyze_structure(pdf_path)
        }

        return results

    def _extract_metadata(self, pdf_path: str) -> Dict[str, Any]:
        """Extract PDF metadata"""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                metadata = pdf.metadata or {}

                return {
                    'num_pages': len(pdf.pages),
                    'title': metadata.get('Title', ''),
                    'author': metadata.get('Author', ''),
                    'subject': metadata.get('Subject', ''),
                    'creator': metadata.get('Creator', ''),
                    'producer': metadata.get('Producer', ''),
                    'creation_date': str(metadata.get('CreationDate', ''))
                }
        except Exception as e:
            logger.error(f"Metadata extraction error: {e}")
            return {}

    def _extract_and_analyze_text(self, pdf_path: str) -> Dict[str, Any]:
        """Extract and analyze text from PDF"""
        try:
            # Extract text
            text = pdfminer_extract_text(pdf_path, laparams=LAParams())

            # Analyze text
            if text and len(text.strip()) > 0:
                analysis = self.text_analyzer.analyze_text(text)

                return {
                    'full_text': text,
                    'text_length': len(text),
                    'analysis': analysis
                }
            else:
                return {'full_text': '', 'text_length': 0, 'analysis': {}}
        except Exception as e:
            logger.error(f"Text extraction error: {e}")
            return {}

    def _extract_tables(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract tables from PDF"""
        tables_data = []

        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    tables = page.extract_tables()

                    for table_num, table in enumerate(tables, 1):
                        if table:
                            # Convert to DataFrame
                            df = pd.DataFrame(table[1:], columns=table[0])

                            tables_data.append({
                                'page': page_num,
                                'table_number': table_num,
                                'rows': len(df),
                                'columns': len(df.columns),
                                'data': df.to_dict(),
                                'preview': df.head().to_string()
                            })
        except Exception as e:
            logger.error(f"Table extraction error: {e}")

        return tables_data

    def _extract_and_analyze_images(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract and analyze images from PDF"""
        images_data = []

        try:
            # Convert PDF pages to images
            images = convert_from_path(pdf_path, dpi=200)

            for page_num, img in enumerate(images, 1):
                # Save temporarily
                temp_path = f"temp_page_{page_num}.png"
                img.save(temp_path, 'PNG')

                # Analyze
                analysis = self.image_analyzer.analyze_image(temp_path)

                images_data.append({
                    'page': page_num,
                    'analysis': analysis
                })

                # Clean up
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        except Exception as e:
            logger.error(f"Image extraction error: {e}")

        return images_data

    def _analyze_structure(self, pdf_path: str) -> Dict[str, Any]:
        """Analyze PDF document structure"""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                # Analyze page sizes and orientations
                page_sizes = [{'width': p.width, 'height': p.height} for p in pdf.pages]

                return {
                    'num_pages': len(pdf.pages),
                    'page_sizes': page_sizes,
                    'uniform_size': len(set(str(p) for p in page_sizes)) == 1
                }
        except Exception as e:
            logger.error(f"Structure analysis error: {e}")
            return {}


# Export
__all__ = ['TextAnalyzer', 'ImageAnalyzer', 'CodeAnalyzer', 'PDFDeepAnalyzer']
