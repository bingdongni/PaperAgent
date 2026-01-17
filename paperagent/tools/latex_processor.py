"""
LaTeX processing tools
"""

import os
import re
import subprocess
from typing import Optional, Dict, Any, List
from pathlib import Path
from loguru import logger


class LaTeXProcessor:
    """LaTeX document processor"""

    def __init__(self):
        self.temp_dir = Path("./temp/latex")
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def create_document(
        self,
        title: str,
        authors: List[str],
        abstract: str,
        sections: Dict[str, str],
        references: Optional[List[str]] = None,
        document_class: str = "article",
        packages: Optional[List[str]] = None
    ) -> str:
        """
        Create a complete LaTeX document

        Args:
            title: Document title
            authors: List of author names
            abstract: Abstract text
            sections: Dictionary of section_name: content
            references: List of BibTeX entries
            document_class: LaTeX document class
            packages: Additional LaTeX packages

        Returns:
            Complete LaTeX source code
        """
        # Default packages
        default_packages = [
            "amsmath",
            "amssymb",
            "graphicx",
            "hyperref",
            "cite",
            "algorithm",
            "algorithmic"
        ]

        if packages:
            default_packages.extend(packages)

        # Build LaTeX document
        latex_content = []

        # Document class
        latex_content.append(f"\\documentclass[11pt,a4paper]{{{document_class}}}\n")

        # Packages
        for package in default_packages:
            latex_content.append(f"\\usepackage{{{package}}}\n")

        # Title and authors
        latex_content.append("\n\\title{" + self.escape_latex(title) + "}\n")

        if authors:
            author_str = " \\and ".join([self.escape_latex(a) for a in authors])
            latex_content.append(f"\\author{{{author_str}}}\n")

        latex_content.append("\\date{\\today}\n")

        # Begin document
        latex_content.append("\n\\begin{document}\n")
        latex_content.append("\\maketitle\n")

        # Abstract
        if abstract:
            latex_content.append("\n\\begin{abstract}\n")
            latex_content.append(self.escape_latex(abstract))
            latex_content.append("\n\\end{abstract}\n")

        # Sections
        for section_name, content in sections.items():
            latex_content.append(f"\n\\section{{{self.escape_latex(section_name)}}}\n")
            latex_content.append(self.escape_latex(content) + "\n")

        # References
        if references:
            latex_content.append("\n\\begin{thebibliography}{99}\n")
            for i, ref in enumerate(references, 1):
                latex_content.append(f"\\bibitem{{ref{i}}} {self.escape_latex(ref)}\n")
            latex_content.append("\\end{thebibliography}\n")

        # End document
        latex_content.append("\n\\end{document}\n")

        return "".join(latex_content)

    def escape_latex(self, text: str) -> str:
        """
        Escape special LaTeX characters

        Args:
            text: Input text

        Returns:
            Escaped text safe for LaTeX
        """
        # Characters that need escaping
        escape_chars = {
            '&': r'\&',
            '%': r'\%',
            '$': r'\$',
            '#': r'\#',
            '_': r'\_',
            '{': r'\{',
            '}': r'\}',
            '~': r'\textasciitilde{}',
            '^': r'\^{}',
            '\\': r'\textbackslash{}',
        }

        for char, escaped in escape_chars.items():
            text = text.replace(char, escaped)

        return text

    def compile_to_pdf(
        self,
        latex_source: str,
        output_path: str,
        num_runs: int = 2
    ) -> bool:
        """
        Compile LaTeX source to PDF

        Args:
            latex_source: LaTeX source code
            output_path: Path for output PDF
            num_runs: Number of compilation runs (for references)

        Returns:
            Success boolean
        """
        try:
            # Create temporary .tex file
            tex_file = self.temp_dir / "document.tex"
            with open(tex_file, 'w', encoding='utf-8') as f:
                f.write(latex_source)

            # Compile with pdflatex
            for i in range(num_runs):
                result = subprocess.run(
                    ['pdflatex', '-interaction=nonstopmode', '-output-directory',
                     str(self.temp_dir), str(tex_file)],
                    capture_output=True,
                    text=True,
                    timeout=60
                )

                if result.returncode != 0:
                    logger.error(f"LaTeX compilation failed:\n{result.stdout}\n{result.stderr}")
                    return False

            # Move PDF to output path
            pdf_file = self.temp_dir / "document.pdf"
            if pdf_file.exists():
                import shutil
                shutil.move(str(pdf_file), output_path)
                logger.info(f"PDF created successfully: {output_path}")
                return True
            else:
                logger.error("PDF file not generated")
                return False

        except Exception as e:
            logger.error(f"LaTeX compilation error: {e}")
            return False

    def create_figure(
        self,
        image_path: str,
        caption: str,
        label: str,
        width: str = "0.8\\textwidth"
    ) -> str:
        """
        Create LaTeX figure environment

        Args:
            image_path: Path to image file
            caption: Figure caption
            label: Figure label for referencing
            width: Figure width

        Returns:
            LaTeX figure code
        """
        return f"""
\\begin{{figure}}[htbp]
    \\centering
    \\includegraphics[width={width}]{{{image_path}}}
    \\caption{{{self.escape_latex(caption)}}}
    \\label{{fig:{label}}}
\\end{{figure}}
"""

    def create_table(
        self,
        headers: List[str],
        rows: List[List[str]],
        caption: str,
        label: str
    ) -> str:
        """
        Create LaTeX table environment

        Args:
            headers: Table header row
            rows: Table data rows
            caption: Table caption
            label: Table label for referencing

        Returns:
            LaTeX table code
        """
        num_cols = len(headers)
        col_spec = "c" * num_cols

        table_lines = []
        table_lines.append("\\begin{table}[htbp]")
        table_lines.append("    \\centering")
        table_lines.append(f"    \\caption{{{self.escape_latex(caption)}}}")
        table_lines.append(f"    \\label{{tab:{label}}}")
        table_lines.append(f"    \\begin{{tabular}}{{{col_spec}}}")
        table_lines.append("        \\hline")

        # Headers
        header_str = " & ".join([self.escape_latex(h) for h in headers])
        table_lines.append(f"        {header_str} \\\\")
        table_lines.append("        \\hline")

        # Rows
        for row in rows:
            row_str = " & ".join([self.escape_latex(str(cell)) for cell in row])
            table_lines.append(f"        {row_str} \\\\")

        table_lines.append("        \\hline")
        table_lines.append("    \\end{tabular}")
        table_lines.append("\\end{table}")

        return "\n".join(table_lines)

    def create_equation(self, equation: str, label: Optional[str] = None) -> str:
        """
        Create LaTeX equation environment

        Args:
            equation: Equation content (LaTeX math)
            label: Optional label for referencing

        Returns:
            LaTeX equation code
        """
        if label:
            return f"""
\\begin{{equation}}
    {equation}
    \\label{{eq:{label}}}
\\end{{equation}}
"""
        else:
            return f"""
\\begin{{equation*}}
    {equation}
\\end{{equation*}}
"""

    def apply_journal_template(
        self,
        latex_source: str,
        journal_template: Dict[str, Any]
    ) -> str:
        """
        Apply journal-specific formatting template

        Args:
            latex_source: Original LaTeX source
            journal_template: Journal template specifications

        Returns:
            Modified LaTeX source
        """
        # Extract document class
        doc_class = journal_template.get('document_class', 'article')

        # Replace document class
        latex_source = re.sub(
            r'\\documentclass\[.*?\]\{.*?\}',
            f"\\documentclass[{journal_template.get('options', '11pt,a4paper')}]{{{doc_class}}}",
            latex_source
        )

        # Add journal-specific packages
        if 'packages' in journal_template:
            packages_section = "\n".join([
                f"\\usepackage{{{pkg}}}" for pkg in journal_template['packages']
            ])
            latex_source = re.sub(
                r'(\\documentclass.*?\n)',
                r'\1' + packages_section + '\n',
                latex_source
            )

        return latex_source


class BibTeXManager:
    """BibTeX reference manager"""

    def __init__(self):
        pass

    def create_entry(
        self,
        entry_type: str,
        citation_key: str,
        fields: Dict[str, str]
    ) -> str:
        """
        Create a BibTeX entry

        Args:
            entry_type: Entry type (article, book, etc.)
            citation_key: Citation key
            fields: Dictionary of field_name: value

        Returns:
            BibTeX entry string
        """
        lines = [f"@{entry_type}{{{citation_key},"]

        for key, value in fields.items():
            lines.append(f"  {key} = {{{value}}},")

        lines.append("}")

        return "\n".join(lines)

    def parse_entry(self, bibtex_entry: str) -> Dict[str, Any]:
        """
        Parse a BibTeX entry

        Args:
            bibtex_entry: BibTeX entry string

        Returns:
            Dictionary of parsed fields
        """
        try:
            import bibtexparser

            bib_database = bibtexparser.loads(bibtex_entry)

            if bib_database.entries:
                return bib_database.entries[0]
            else:
                return {}

        except Exception as e:
            logger.error(f"BibTeX parsing error: {e}")
            return {}

    def format_citation(
        self,
        entry: Dict[str, str],
        style: str = "ieee"
    ) -> str:
        """
        Format citation according to style

        Args:
            entry: BibTeX entry dictionary
            style: Citation style (ieee, apa, mla)

        Returns:
            Formatted citation string
        """
        authors = entry.get('author', 'Unknown')
        title = entry.get('title', '')
        year = entry.get('year', '')
        journal = entry.get('journal', '')
        volume = entry.get('volume', '')
        pages = entry.get('pages', '')

        if style.lower() == 'ieee':
            # IEEE format
            citation = f"{authors}, \"{title},\" "
            if journal:
                citation += f"{journal}, "
            if volume:
                citation += f"vol. {volume}, "
            if pages:
                citation += f"pp. {pages}, "
            citation += f"{year}."

        elif style.lower() == 'apa':
            # APA format
            citation = f"{authors} ({year}). {title}. "
            if journal:
                citation += f"{journal}"
            if volume:
                citation += f", {volume}"
            if pages:
                citation += f", {pages}"
            citation += "."

        else:
            # Default format
            citation = f"{authors}. {title}. {journal}, {year}."

        return citation


# Export classes
__all__ = ["LaTeXProcessor", "BibTeXManager"]
