"""
Mathematical Formula Recognition and Analysis

Provides capabilities for:
- LaTeX formula extraction from images
- Formula structure analysis
- Mathematical symbol recognition
- Equation type classification
"""

import re
import cv2
import numpy as np
from PIL import Image
import pytesseract
from typing import Dict, Any, List, Optional
from loguru import logger


class FormulaRecognizer:
    """Recognize and analyze mathematical formulas"""

    def __init__(self):
        self.math_symbols = {
            'operators': ['+', '-', '×', '÷', '=', '≠', '≈', '≤', '≥'],
            'greek': ['α', 'β', 'γ', 'δ', 'ε', 'θ', 'λ', 'μ', 'π', 'σ', 'φ', 'ω'],
            'special': ['∫', '∑', '∏', '√', '∂', '∇', '∞', '∈', '∉', '⊂', '⊃']
        }

    def recognize_formula(self, image_path: str) -> Dict[str, Any]:
        """
        Recognize mathematical formula from image

        Args:
            image_path: Path to formula image

        Returns:
            Recognition results
        """
        img = cv2.imread(image_path)
        if img is None:
            return {'error': 'Failed to load image'}

        # Preprocess image
        processed = self._preprocess_formula_image(img)

        # Extract text/symbols
        text = pytesseract.image_to_string(
            Image.fromarray(processed),
            config='--psm 6'  # Assume uniform block of text
        )

        results = {
            'raw_text': text,
            'latex': self._convert_to_latex(text),
            'symbols': self._extract_symbols(text),
            'structure': self._analyze_formula_structure(text),
            'type': self._classify_formula_type(text)
        }

        return results

    def _preprocess_formula_image(self, img: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR"""
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Increase contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Denoise
        denoised = cv2.fastNlMeansDenoising(enhanced)

        # Binarize
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return binary

    def _convert_to_latex(self, text: str) -> str:
        """
        Convert recognized text to LaTeX format

        Args:
            text: Recognized text

        Returns:
            LaTeX representation
        """
        latex = text

        # Replace common symbols
        replacements = {
            '×': '\\times',
            '÷': '\\div',
            '≠': '\\neq',
            '≈': '\\approx',
            '≤': '\\leq',
            '≥': '\\geq',
            '∫': '\\int',
            '∑': '\\sum',
            '∏': '\\prod',
            '√': '\\sqrt',
            '∂': '\\partial',
            '∇': '\\nabla',
            '∞': '\\infty',
            '∈': '\\in',
            '∉': '\\notin',
            '⊂': '\\subset',
            '⊃': '\\supset',
            'α': '\\alpha',
            'β': '\\beta',
            'γ': '\\gamma',
            'δ': '\\delta',
            'ε': '\\epsilon',
            'θ': '\\theta',
            'λ': '\\lambda',
            'μ': '\\mu',
            'π': '\\pi',
            'σ': '\\sigma',
            'φ': '\\phi',
            'ω': '\\omega'
        }

        for symbol, latex_cmd in replacements.items():
            latex = latex.replace(symbol, latex_cmd)

        # Detect and format fractions
        latex = self._format_fractions(latex)

        # Detect and format exponents
        latex = self._format_exponents(latex)

        return latex

    def _format_fractions(self, text: str) -> str:
        """Detect and format fractions"""
        # Pattern: number/number or expression/expression
        pattern = r'(\w+)/(\w+)'
        return re.sub(pattern, r'\\frac{\1}{\2}', text)

    def _format_exponents(self, text: str) -> str:
        """Detect and format exponents"""
        # Pattern: base^exponent
        pattern = r'(\w+)\^(\w+)'
        return re.sub(pattern, r'\1^{\2}', text)

    def _extract_symbols(self, text: str) -> Dict[str, List[str]]:
        """Extract mathematical symbols from text"""
        found_symbols = {
            'operators': [],
            'greek': [],
            'special': [],
            'variables': [],
            'numbers': []
        }

        # Check for each symbol type
        for symbol in self.math_symbols['operators']:
            if symbol in text:
                found_symbols['operators'].append(symbol)

        for symbol in self.math_symbols['greek']:
            if symbol in text:
                found_symbols['greek'].append(symbol)

        for symbol in self.math_symbols['special']:
            if symbol in text:
                found_symbols['special'].append(symbol)

        # Extract variables (single letters)
        variables = re.findall(r'\b[a-zA-Z]\b', text)
        found_symbols['variables'] = list(set(variables))

        # Extract numbers
        numbers = re.findall(r'\b\d+\.?\d*\b', text)
        found_symbols['numbers'] = list(set(numbers))

        return found_symbols

    def _analyze_formula_structure(self, text: str) -> Dict[str, Any]:
        """Analyze formula structure"""
        structure = {
            'has_fraction': '/' in text or '\\frac' in text,
            'has_exponent': '^' in text,
            'has_subscript': '_' in text,
            'has_integral': '∫' in text or '\\int' in text,
            'has_sum': '∑' in text or '\\sum' in text,
            'has_product': '∏' in text or '\\prod' in text,
            'has_sqrt': '√' in text or '\\sqrt' in text,
            'has_limit': 'lim' in text.lower(),
            'has_derivative': '∂' in text or 'd/d' in text,
            'complexity': self._estimate_complexity(text)
        }

        return structure

    def _estimate_complexity(self, text: str) -> str:
        """Estimate formula complexity"""
        complexity_score = 0

        # Count complexity indicators
        if '/' in text or '\\frac' in text:
            complexity_score += 2
        if '^' in text:
            complexity_score += 1
        if '∫' in text or '\\int' in text:
            complexity_score += 3
        if '∑' in text or '\\sum' in text:
            complexity_score += 2
        if '∂' in text:
            complexity_score += 2

        # Count nested structures
        nested_count = text.count('{') + text.count('(')
        complexity_score += nested_count

        if complexity_score <= 2:
            return 'simple'
        elif complexity_score <= 5:
            return 'moderate'
        else:
            return 'complex'

    def _classify_formula_type(self, text: str) -> str:
        """Classify formula type"""
        text_lower = text.lower()

        if '∫' in text or '\\int' in text:
            return 'integral'
        elif '∑' in text or '\\sum' in text:
            return 'summation'
        elif 'lim' in text_lower:
            return 'limit'
        elif '∂' in text or 'd/d' in text:
            return 'derivative'
        elif '=' in text:
            return 'equation'
        elif any(op in text for op in ['<', '>', '≤', '≥', '≠']):
            return 'inequality'
        else:
            return 'expression'


class TableStructureAnalyzer:
    """Analyze table structure and extract data"""

    def __init__(self):
        pass

    def analyze_table(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze table structure from image

        Args:
            image_path: Path to table image

        Returns:
            Table analysis results
        """
        img = cv2.imread(image_path)
        if img is None:
            return {'error': 'Failed to load image'}

        results = {
            'structure': self._detect_table_structure(img),
            'cells': self._extract_cells(img),
            'data': self._extract_table_data(image_path)
        }

        return results

    def _detect_table_structure(self, img: np.ndarray) -> Dict[str, Any]:
        """Detect table grid structure"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Detect horizontal and vertical lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))

        horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
        vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)

        # Count lines
        h_lines = cv2.HoughLinesP(horizontal_lines, 1, np.pi/180, threshold=100,
                                   minLineLength=100, maxLineGap=10)
        v_lines = cv2.HoughLinesP(vertical_lines, 1, np.pi/180, threshold=100,
                                   minLineLength=100, maxLineGap=10)

        num_rows = len(h_lines) - 1 if h_lines is not None else 0
        num_cols = len(v_lines) - 1 if v_lines is not None else 0

        return {
            'num_rows': num_rows,
            'num_cols': num_cols,
            'has_borders': h_lines is not None and v_lines is not None,
            'grid_type': 'bordered' if h_lines is not None else 'borderless'
        }

    def _extract_cells(self, img: np.ndarray) -> List[Dict[str, Any]]:
        """Extract individual table cells"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cells = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Filter for cell-like rectangles
            if w > 20 and h > 20 and w < img.shape[1] * 0.9 and h < img.shape[0] * 0.9:
                cells.append({
                    'position': (x, y),
                    'width': w,
                    'height': h,
                    'area': w * h
                })

        return sorted(cells, key=lambda c: (c['position'][1], c['position'][0]))

    def _extract_table_data(self, image_path: str) -> List[List[str]]:
        """Extract text data from table"""
        img = Image.open(image_path)

        # Use pytesseract to extract data
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

        # Group text by rows
        rows = {}
        n_boxes = len(data['text'])

        for i in range(n_boxes):
            if int(data['conf'][i]) > 30:
                text = data['text'][i].strip()
                if text:
                    top = data['top'][i]
                    row_key = top // 20  # Group by approximate row

                    if row_key not in rows:
                        rows[row_key] = []

                    rows[row_key].append({
                        'text': text,
                        'left': data['left'][i]
                    })

        # Sort and format
        table_data = []
        for row_key in sorted(rows.keys()):
            row_cells = sorted(rows[row_key], key=lambda x: x['left'])
            table_data.append([cell['text'] for cell in row_cells])

        return table_data


__all__ = ['FormulaRecognizer', 'TableStructureAnalyzer']
