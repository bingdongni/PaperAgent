"""
Advanced LaTeX Processing Tools

Provides enhanced LaTeX formatting capabilities including:
- Complex multi-column layouts
- Advanced mathematical environments
- Algorithm and pseudocode formatting
- Theorem and proof environments
- Custom environments and commands
- TikZ diagrams support
"""

import re
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from loguru import logger


class AdvancedLaTeXFormatter:
    """Advanced LaTeX formatting and layout tools"""

    def __init__(self):
        self.custom_commands = []
        self.custom_environments = []

    def create_multi_column_layout(
        self,
        content: List[Dict[str, Any]],
        num_columns: int = 2,
        column_sep: str = "1cm"
    ) -> str:
        """
        Create multi-column layout

        Args:
            content: List of content blocks with 'text' and optional 'span_columns'
            num_columns: Number of columns
            column_sep: Separation between columns

        Returns:
            LaTeX multi-column code
        """
        latex = []
        latex.append(f"\\begin{{multicols}}{{{num_columns}}}[{column_sep}]\n")

        for block in content:
            text = block.get('text', '')
            span = block.get('span_columns', False)

            if span:
                latex.append("\\end{multicols}\n")
                latex.append(text + "\n")
                latex.append(f"\\begin{{multicols}}{{{num_columns}}}\n")
            else:
                latex.append(text + "\n")

        latex.append("\\end{multicols}\n")
        return "".join(latex)

    def create_algorithm(
        self,
        title: str,
        inputs: List[str],
        outputs: List[str],
        steps: List[str],
        label: Optional[str] = None
    ) -> str:
        """
        Create algorithm environment

        Args:
            title: Algorithm title
            inputs: List of input parameters
            outputs: List of output parameters
            steps: Algorithm steps
            label: Optional label for referencing

        Returns:
            LaTeX algorithm code
        """
        latex = []
        latex.append("\\begin{algorithm}[H]\n")
        latex.append(f"\\caption{{{title}}}\n")

        if label:
            latex.append(f"\\label{{alg:{label}}}\n")

        latex.append("\\begin{algorithmic}[1]\n")

        # Inputs
        if inputs:
            latex.append("\\REQUIRE\n")
            for inp in inputs:
                latex.append(f"  {inp}\n")

        # Outputs
        if outputs:
            latex.append("\\ENSURE\n")
            for out in outputs:
                latex.append(f"  {out}\n")

        # Steps
        for step in steps:
            latex.append(f"\\STATE {step}\n")

        latex.append("\\end{algorithmic}\n")
        latex.append("\\end{algorithm}\n")

        return "".join(latex)

    def create_theorem_environment(
        self,
        env_type: str,
        content: str,
        label: Optional[str] = None,
        title: Optional[str] = None
    ) -> str:
        """
        Create theorem-like environment

        Args:
            env_type: Type (theorem, lemma, proposition, corollary, definition)
            content: Theorem content
            label: Optional label
            title: Optional title

        Returns:
            LaTeX theorem environment
        """
        latex = []

        if title:
            latex.append(f"\\begin{{{env_type}}}[{title}]\n")
        else:
            latex.append(f"\\begin{{{env_type}}}\n")

        if label:
            latex.append(f"\\label{{thm:{label}}}\n")

        latex.append(content + "\n")
        latex.append(f"\\end{{{env_type}}}\n")

        return "".join(latex)

    def create_proof(self, content: str, qed_symbol: str = "\\qed") -> str:
        """
        Create proof environment

        Args:
            content: Proof content
            qed_symbol: QED symbol (default: \\qed)

        Returns:
            LaTeX proof environment
        """
        return f"""
\\begin{{proof}}
{content}
{qed_symbol}
\\end{{proof}}
"""

    def create_matrix(
        self,
        data: np.ndarray,
        matrix_type: str = "bmatrix",
        precision: int = 2
    ) -> str:
        """
        Create LaTeX matrix from numpy array

        Args:
            data: Numpy array
            matrix_type: Matrix type (bmatrix, pmatrix, vmatrix, Bmatrix, Vmatrix)
            precision: Decimal precision

        Returns:
            LaTeX matrix code
        """
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)

        rows, cols = data.shape
        latex = [f"\\begin{{{matrix_type}}}\n"]

        for i in range(rows):
            row_str = " & ".join([f"{data[i, j]:.{precision}f}" for j in range(cols)])
            latex.append(f"  {row_str}")
            if i < rows - 1:
                latex.append(" \\\\\n")
            else:
                latex.append("\n")

        latex.append(f"\\end{{{matrix_type}}}")
        return "".join(latex)

    def create_aligned_equations(
        self,
        equations: List[Tuple[str, str]],
        label: Optional[str] = None
    ) -> str:
        """
        Create aligned equations

        Args:
            equations: List of (left_side, right_side) tuples
            label: Optional label

        Returns:
            LaTeX aligned equations
        """
        latex = []
        latex.append("\\begin{align}\n")

        for i, (left, right) in enumerate(equations):
            latex.append(f"  {left} &= {right}")

            if i == 0 and label:
                latex.append(f" \\label{{eq:{label}}}")

            if i < len(equations) - 1:
                latex.append(" \\\\\n")
            else:
                latex.append("\n")

        latex.append("\\end{align}\n")
        return "".join(latex)

    def create_cases_equation(
        self,
        cases: List[Tuple[str, str]],
        function_name: str = "f(x)"
    ) -> str:
        """
        Create piecewise function with cases

        Args:
            cases: List of (expression, condition) tuples
            function_name: Function name

        Returns:
            LaTeX cases equation
        """
        latex = []
        latex.append(f"{function_name} = \\begin{{cases}}\n")

        for expr, condition in cases:
            latex.append(f"  {expr} & \\text{{if }} {condition} \\\\\n")

        latex.append("\\end{cases}\n")
        return "".join(latex)


class ComplexTableFormatter:
    """Advanced table formatting with complex layouts"""

    def __init__(self):
        pass

    def create_booktabs_table(
        self,
        df: pd.DataFrame,
        caption: str,
        label: str,
        column_format: Optional[str] = None,
        bold_header: bool = True,
        highlight_rows: Optional[List[int]] = None
    ) -> str:
        """
        Create professional table using booktabs package

        Args:
            df: Pandas DataFrame
            caption: Table caption
            label: Table label
            column_format: Column format string (e.g., 'lccr')
            bold_header: Bold header row
            highlight_rows: List of row indices to highlight

        Returns:
            LaTeX booktabs table
        """
        if column_format is None:
            column_format = 'l' + 'c' * (len(df.columns) - 1)

        latex = []
        latex.append("\\begin{table}[htbp]\n")
        latex.append("  \\centering\n")
        latex.append(f"  \\caption{{{caption}}}\n")
        latex.append(f"  \\label{{tab:{label}}}\n")
        latex.append(f"  \\begin{{tabular}}{{{column_format}}}\n")
        latex.append("    \\toprule\n")

        # Header
        if bold_header:
            headers = [f"\\textbf{{{col}}}" for col in df.columns]
        else:
            headers = list(df.columns)

        latex.append("    " + " & ".join(headers) + " \\\\\n")
        latex.append("    \\midrule\n")

        # Data rows
        for idx, row in df.iterrows():
            row_data = [str(val) for val in row]

            if highlight_rows and idx in highlight_rows:
                latex.append("    \\rowcolor{lightgray}\n")

            latex.append("    " + " & ".join(row_data) + " \\\\\n")

        latex.append("    \\bottomrule\n")
        latex.append("  \\end{tabular}\n")
        latex.append("\\end{table}\n")

        return "".join(latex)

    def create_multirow_table(
        self,
        data: List[List[Any]],
        headers: List[str],
        merge_cells: List[Tuple[int, int, int, int]],  # (row, col, rowspan, colspan)
        caption: str,
        label: str
    ) -> str:
        """
        Create table with merged cells

        Args:
            data: Table data
            headers: Column headers
            merge_cells: List of cells to merge (row, col, rowspan, colspan)
            caption: Table caption
            label: Table label

        Returns:
            LaTeX table with multirow/multicolumn
        """
        num_cols = len(headers)
        latex = []

        latex.append("\\begin{table}[htbp]\n")
        latex.append("  \\centering\n")
        latex.append(f"  \\caption{{{caption}}}\n")
        latex.append(f"  \\label{{tab:{label}}}\n")
        latex.append(f"  \\begin{{tabular}}{{{'|'.join(['c'] * num_cols)}}}\n")
        latex.append("    \\hline\n")

        # Headers
        latex.append("    " + " & ".join(headers) + " \\\\\n")
        latex.append("    \\hline\n")

        # Process merge information
        merged_positions = set()
        for row, col, rowspan, colspan in merge_cells:
            for r in range(row, row + rowspan):
                for c in range(col, col + colspan):
                    if (r, c) != (row, col):
                        merged_positions.add((r, c))

        # Data rows
        for row_idx, row in enumerate(data):
            row_latex = []
            for col_idx, cell in enumerate(row):
                if (row_idx, col_idx) in merged_positions:
                    continue

                # Check if this cell should be merged
                merge_info = None
                for r, c, rs, cs in merge_cells:
                    if r == row_idx and c == col_idx:
                        merge_info = (rs, cs)
                        break

                if merge_info:
                    rowspan, colspan = merge_info
                    if rowspan > 1 and colspan > 1:
                        cell_str = f"\\multirow{{{rowspan}}}{{*}}{{\\multicolumn{{{colspan}}}{{c}}{{{cell}}}}}"
                    elif rowspan > 1:
                        cell_str = f"\\multirow{{{rowspan}}}{{*}}{{{cell}}}"
                    elif colspan > 1:
                        cell_str = f"\\multicolumn{{{colspan}}}{{c}}{{{cell}}}"
                    else:
                        cell_str = str(cell)
                else:
                    cell_str = str(cell)

                row_latex.append(cell_str)

            latex.append("    " + " & ".join(row_latex) + " \\\\\n")
            latex.append("    \\hline\n")

        latex.append("  \\end{tabular}\n")
        latex.append("\\end{table}\n")

        return "".join(latex)

    def create_longtable(
        self,
        df: pd.DataFrame,
        caption: str,
        label: str,
        header_repeat: bool = True
    ) -> str:
        """
        Create long table that spans multiple pages

        Args:
            df: Pandas DataFrame
            caption: Table caption
            label: Table label
            header_repeat: Repeat header on each page

        Returns:
            LaTeX longtable
        """
        num_cols = len(df.columns)
        col_format = 'l' + 'c' * (num_cols - 1)

        latex = []
        latex.append(f"\\begin{{longtable}}{{{col_format}}}\n")
        latex.append(f"\\caption{{{caption}}} \\label{{tab:{label}}} \\\\\n")
        latex.append("\\toprule\n")

        # Header
        headers = " & ".join([f"\\textbf{{{col}}}" for col in df.columns])
        latex.append(headers + " \\\\\n")
        latex.append("\\midrule\n")
        latex.append("\\endfirsthead\n\n")

        if header_repeat:
            latex.append("\\multicolumn{" + str(num_cols) + "}{c}\n")
            latex.append("{\\tablename\\ \\thetable\\ -- continued from previous page} \\\\\n")
            latex.append("\\toprule\n")
            latex.append(headers + " \\\\\n")
            latex.append("\\midrule\n")
            latex.append("\\endhead\n\n")

            latex.append("\\midrule\n")
            latex.append("\\multicolumn{" + str(num_cols) + "}{r}{Continued on next page} \\\\\n")
            latex.append("\\endfoot\n\n")

            latex.append("\\bottomrule\n")
            latex.append("\\endlastfoot\n\n")

        # Data
        for _, row in df.iterrows():
            row_data = " & ".join([str(val) for val in row])
            latex.append(row_data + " \\\\\n")

        latex.append("\\end{longtable}\n")

        return "".join(latex)


class MathematicalFormatter:
    """Advanced mathematical formatting"""

    def __init__(self):
        pass

    def create_fraction(self, numerator: str, denominator: str, display: bool = False) -> str:
        """Create fraction"""
        if display:
            return f"\\dfrac{{{numerator}}}{{{denominator}}}"
        else:
            return f"\\frac{{{numerator}}}{{{denominator}}}"

    def create_sum(
        self,
        expression: str,
        lower_limit: str,
        upper_limit: str,
        display: bool = True
    ) -> str:
        """Create summation"""
        if display:
            return f"\\sum_{{{lower_limit}}}^{{{upper_limit}}} {expression}"
        else:
            return f"\\sum_{{{lower_limit}}}^{{{upper_limit}}} {expression}"

    def create_integral(
        self,
        expression: str,
        lower_limit: str,
        upper_limit: str,
        variable: str = "x"
    ) -> str:
        """Create integral"""
        return f"\\int_{{{lower_limit}}}^{{{upper_limit}}} {expression} \\, d{variable}"

    def create_limit(
        self,
        expression: str,
        variable: str,
        limit_value: str
    ) -> str:
        """Create limit"""
        return f"\\lim_{{{variable} \\to {limit_value}}} {expression}"

    def create_derivative(
        self,
        function: str,
        variable: str,
        order: int = 1
    ) -> str:
        """Create derivative notation"""
        if order == 1:
            return f"\\frac{{d{function}}}{{d{variable}}}"
        else:
            return f"\\frac{{d^{{{order}}}{function}}}{{d{variable}^{{{order}}}}}"

    def create_partial_derivative(
        self,
        function: str,
        variable: str,
        order: int = 1
    ) -> str:
        """Create partial derivative"""
        if order == 1:
            return f"\\frac{{\\partial {function}}}{{\\partial {variable}}}"
        else:
            return f"\\frac{{\\partial^{{{order}}} {function}}}{{\\partial {variable}^{{{order}}}}}"

    def create_vector(self, components: List[str], notation: str = "column") -> str:
        """
        Create vector

        Args:
            components: Vector components
            notation: 'column', 'row', or 'arrow'
        """
        if notation == "arrow":
            return "\\vec{" + ", ".join(components) + "}"
        elif notation == "row":
            return "\\begin{bmatrix} " + " & ".join(components) + " \\end{bmatrix}"
        else:  # column
            return "\\begin{bmatrix} " + " \\\\ ".join(components) + " \\end{bmatrix}"

    def create_set(self, elements: List[str], set_builder: bool = False) -> str:
        """Create set notation"""
        if set_builder:
            return "\\{" + " \\mid ".join(elements) + "\\}"
        else:
            return "\\{" + ", ".join(elements) + "\\}"


# Export classes
__all__ = [
    'AdvancedLaTeXFormatter',
    'ComplexTableFormatter',
    'MathematicalFormatter'
]
