"""
Chart and Graph Analysis Module

Provides deep understanding of charts, graphs, and diagrams including:
- Chart type detection
- Data extraction from charts
- Trend analysis
- Statistical insights
"""

import cv2
import numpy as np
from PIL import Image
import pytesseract
from typing import Dict, Any, List, Optional, Tuple
from loguru import logger


class ChartAnalyzer:
    """Analyze and understand charts and graphs"""

    def __init__(self):
        self.chart_types = [
            'bar_chart', 'line_chart', 'pie_chart', 'scatter_plot',
            'histogram', 'box_plot', 'heatmap', 'area_chart'
        ]

    def analyze_chart(self, image_path: str) -> Dict[str, Any]:
        """
        Comprehensive chart analysis

        Args:
            image_path: Path to chart image

        Returns:
            Analysis results including type, data, insights
        """
        img = cv2.imread(image_path)
        if img is None:
            return {'error': 'Failed to load image'}

        results = {
            'chart_type': self._detect_chart_type(img),
            'colors': self._extract_colors(img),
            'text_elements': self._extract_text_elements(image_path),
            'structure': self._analyze_structure(img),
            'insights': {}
        }

        # Type-specific analysis
        chart_type = results['chart_type']
        if chart_type == 'bar_chart':
            results['insights'] = self._analyze_bar_chart(img)
        elif chart_type == 'line_chart':
            results['insights'] = self._analyze_line_chart(img)
        elif chart_type == 'pie_chart':
            results['insights'] = self._analyze_pie_chart(img)

        return results

    def _detect_chart_type(self, img: np.ndarray) -> str:
        """
        Detect chart type using visual features

        Args:
            img: Image array

        Returns:
            Detected chart type
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Detect lines (for line charts, bar charts)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100,
                                minLineLength=50, maxLineGap=10)

        # Detect circles (for pie charts)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                                   param1=50, param2=30, minRadius=20, maxRadius=200)

        # Heuristic detection
        if circles is not None and len(circles[0]) > 0:
            return 'pie_chart'
        elif lines is not None:
            # Analyze line orientations
            vertical_lines = 0
            horizontal_lines = 0

            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

                if angle < 10 or angle > 170:
                    horizontal_lines += 1
                elif 80 < angle < 100:
                    vertical_lines += 1

            if vertical_lines > horizontal_lines * 2:
                return 'bar_chart'
            elif horizontal_lines > 10 and vertical_lines > 10:
                return 'line_chart'

        return 'unknown'

    def _extract_colors(self, img: np.ndarray) -> List[Dict[str, Any]]:
        """Extract dominant colors from chart"""
        # Reshape image to list of pixels
        pixels = img.reshape(-1, 3)

        # Use k-means to find dominant colors
        from sklearn.cluster import KMeans

        n_colors = 5
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)

        colors = []
        for i, color in enumerate(kmeans.cluster_centers_):
            colors.append({
                'rgb': [int(c) for c in color],
                'hex': '#{:02x}{:02x}{:02x}'.format(int(color[2]), int(color[1]), int(color[0])),
                'percentage': float(np.sum(kmeans.labels_ == i) / len(kmeans.labels_) * 100)
            })

        return sorted(colors, key=lambda x: x['percentage'], reverse=True)

    def _extract_text_elements(self, image_path: str) -> Dict[str, Any]:
        """Extract text from chart (labels, title, legend)"""
        img = Image.open(image_path)

        # OCR
        text = pytesseract.image_to_string(img)
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

        # Group text by position
        text_elements = {
            'all_text': text,
            'title': '',
            'labels': [],
            'legend': []
        }

        # Extract positioned text
        n_boxes = len(data['text'])
        for i in range(n_boxes):
            if int(data['conf'][i]) > 30:  # Confidence threshold
                text_elements['labels'].append({
                    'text': data['text'][i],
                    'position': (data['left'][i], data['top'][i]),
                    'confidence': data['conf'][i]
                })

        return text_elements

    def _analyze_structure(self, img: np.ndarray) -> Dict[str, Any]:
        """Analyze chart structure (axes, grid, legend area)"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Detect axes
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100,
                                minLineLength=100, maxLineGap=10)

        structure = {
            'has_axes': lines is not None and len(lines) > 0,
            'has_grid': self._detect_grid(edges),
            'dimensions': {'width': img.shape[1], 'height': img.shape[0]}
        }

        return structure

    def _detect_grid(self, edges: np.ndarray) -> bool:
        """Detect if chart has grid lines"""
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50,
                                minLineLength=30, maxLineGap=5)

        if lines is None:
            return False

        # Count parallel lines
        horizontal = 0
        vertical = 0

        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

            if angle < 10 or angle > 170:
                horizontal += 1
            elif 80 < angle < 100:
                vertical += 1

        return horizontal > 3 and vertical > 3

    def _analyze_bar_chart(self, img: np.ndarray) -> Dict[str, Any]:
        """Analyze bar chart specific features"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect rectangular bars
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bars = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h > 0 else 0

            # Filter for bar-like shapes
            if 0.1 < aspect_ratio < 10 and w * h > 100:
                bars.append({
                    'position': (x, y),
                    'width': w,
                    'height': h,
                    'area': w * h
                })

        return {
            'num_bars': len(bars),
            'bars': sorted(bars, key=lambda x: x['position'][0]),
            'orientation': 'vertical' if bars and bars[0]['height'] > bars[0]['width'] else 'horizontal'
        }

    def _analyze_line_chart(self, img: np.ndarray) -> Dict[str, Any]:
        """Analyze line chart specific features"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Detect lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50,
                                minLineLength=50, maxLineGap=20)

        if lines is None:
            return {'num_lines': 0}

        # Group connected lines
        line_segments = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            line_segments.append({
                'start': (x1, y1),
                'end': (x2, y2),
                'length': np.sqrt((x2-x1)**2 + (y2-y1)**2)
            })

        return {
            'num_line_segments': len(line_segments),
            'total_length': sum(seg['length'] for seg in line_segments),
            'trend': self._detect_trend(line_segments)
        }

    def _detect_trend(self, line_segments: List[Dict]) -> str:
        """Detect overall trend in line chart"""
        if not line_segments:
            return 'unknown'

        # Calculate average slope
        slopes = []
        for seg in line_segments:
            x1, y1 = seg['start']
            x2, y2 = seg['end']
            if x2 != x1:
                slope = (y2 - y1) / (x2 - x1)
                slopes.append(slope)

        if not slopes:
            return 'flat'

        avg_slope = np.mean(slopes)

        if avg_slope > 0.1:
            return 'increasing'
        elif avg_slope < -0.1:
            return 'decreasing'
        else:
            return 'stable'

    def _analyze_pie_chart(self, img: np.ndarray) -> Dict[str, Any]:
        """Analyze pie chart specific features"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect circles
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                                   param1=50, param2=30, minRadius=20, maxRadius=200)

        if circles is None:
            return {'num_slices': 0}

        # Detect pie slices using color segmentation
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Simple slice detection based on color regions
        num_slices = self._count_color_regions(hsv)

        return {
            'num_slices': num_slices,
            'center': (int(circles[0][0][0]), int(circles[0][0][1])) if circles is not None else None,
            'radius': int(circles[0][0][2]) if circles is not None else None
        }

    def _count_color_regions(self, hsv_img: np.ndarray) -> int:
        """Count distinct color regions in pie chart"""
        # Simplified region counting
        h, s, v = cv2.split(hsv_img)

        # Threshold and find contours
        _, thresh = cv2.threshold(s, 50, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter significant regions
        significant_regions = [c for c in contours if cv2.contourArea(c) > 100]

        return len(significant_regions)


__all__ = ['ChartAnalyzer']
