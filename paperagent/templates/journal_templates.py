"""
Comprehensive Journal Templates Library

Contains 20+ major academic journal templates with formatting specifications.
"""

from typing import Dict, Any, List
from dataclasses import dataclass


@dataclass
class JournalTemplate:
    """Journal template dataclass"""
    name: str
    publisher: str
    field: str
    citation_style: str
    document_class: str
    page_layout: Dict[str, Any]
    font_settings: Dict[str, Any]
    requirements: Dict[str, Any]
    latex_template: str
    guidelines_url: str
    submission_system: str


class JournalTemplates:
    """Comprehensive journal template library"""

    @staticmethod
    def get_template(journal_name: str) -> Dict[str, Any]:
        """Get template by journal name"""
        templates = JournalTemplates.get_all_templates()
        return templates.get(journal_name.lower().replace(' ', '_'), {})

    @staticmethod
    def get_all_templates() -> Dict[str, Dict[str, Any]]:
        """Get all available journal templates"""
        return {
            'ieee_access': JournalTemplates.ieee_access(),
            'ieee_transactions': JournalTemplates.ieee_transactions(),
            'acm_transactions': JournalTemplates.acm_transactions(),
            'nature': JournalTemplates.nature(),
            'science': JournalTemplates.science(),
            'springer': JournalTemplates.springer(),
            'elsevier': JournalTemplates.elsevier(),
            'plos_one': JournalTemplates.plos_one(),
            'frontiers': JournalTemplates.frontiers(),
            'mdpi': JournalTemplates.mdpi(),
            'arxiv': JournalTemplates.arxiv(),
            'jmlr': JournalTemplates.jmlr(),
            'aaai': JournalTemplates.aaai(),
            'neurips': JournalTemplates.neurips(),
            'icml': JournalTemplates.icml(),
            'cvpr': JournalTemplates.cvpr(),
            'acl': JournalTemplates.acl(),
            'emnlp': JournalTemplates.emnlp(),
            'cell': JournalTemplates.cell(),
            'lancet': JournalTemplates.lancet(),
            'jama': JournalTemplates.jama(),
            'pnas': JournalTemplates.pnas(),
            'aps_journals': JournalTemplates.aps_journals(),
        }

    @staticmethod
    def ieee_access() -> Dict[str, Any]:
        """IEEE Access template"""
        return {
            'name': 'IEEE Access',
            'publisher': 'IEEE',
            'field': 'Engineering/Computer Science',
            'citation_style': 'IEEE',
            'document_class': 'IEEEtran',
            'page_layout': {
                'columns': 2,
                'page_size': 'letter',
                'margins': {'top': '0.75in', 'bottom': '1in', 'left': '0.625in', 'right': '0.625in'}
            },
            'font_settings': {
                'main_font': 'Times New Roman',
                'font_size': '10pt',
                'title_font_size': '24pt'
            },
            'requirements': {
                'word_limit': None,
                'abstract_limit': 200,
                'max_figures': None,
                'max_tables': None,
                'keywords': 5
            },
            'latex_template': r'''
\documentclass[10pt,journal,compsoc]{IEEEtran}
\usepackage{cite}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{algorithmic}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{xcolor}

\begin{document}
\title{Your Paper Title}
\author{Author Names}

\markboth{IEEE Access}{}

\maketitle

\begin{abstract}
Your abstract here.
\end{abstract}

\begin{IEEEkeywords}
keyword1, keyword2, keyword3
\end{IEEEkeywords}

\section{Introduction}
Your introduction here.

\bibliographystyle{IEEEtran}
\bibliography{references}
\end{document}
''',
            'guidelines_url': 'https://ieeeaccess.ieee.org/for-authors/',
            'submission_system': 'ScholarOne'
        }

    @staticmethod
    def ieee_transactions() -> Dict[str, Any]:
        """IEEE Transactions template"""
        return {
            'name': 'IEEE Transactions',
            'publisher': 'IEEE',
            'field': 'Engineering/Computer Science',
            'citation_style': 'IEEE',
            'document_class': 'IEEEtran',
            'page_layout': {'columns': 2, 'page_size': 'letter'},
            'font_settings': {'main_font': 'Times', 'font_size': '10pt'},
            'requirements': {'abstract_limit': 200, 'keywords': 5},
            'latex_template': r'''
\documentclass[journal]{IEEEtran}
\usepackage{cite}
\usepackage{amsmath}
\usepackage{graphicx}

\begin{document}
\title{Paper Title}
\author{Authors}
\maketitle

\begin{abstract}
Abstract text.
\end{abstract}

\begin{IEEEkeywords}
Keywords here.
\end{IEEEkeywords}

\section{Introduction}
Introduction text.

\bibliographystyle{IEEEtran}
\bibliography{references}
\end{document}
''',
            'guidelines_url': 'https://www.ieee.org/publications/authors/',
            'submission_system': 'ScholarOne'
        }

    @staticmethod
    def acm_transactions() -> Dict[str, Any]:
        """ACM Transactions template"""
        return {
            'name': 'ACM Transactions',
            'publisher': 'ACM',
            'field': 'Computer Science',
            'citation_style': 'ACM',
            'document_class': 'acmart',
            'page_layout': {'columns': 2, 'page_size': 'letter'},
            'font_settings': {'main_font': 'Libertine', 'font_size': '10pt'},
            'requirements': {'abstract_limit': 150, 'keywords': 5},
            'latex_template': r'''
\documentclass[acmtog]{acmart}
\usepackage{graphicx}

\begin{document}
\title{Paper Title}
\author{Author Name}
\affiliation{\institution{Institution}}

\begin{abstract}
Abstract text.
\end{abstract}

\keywords{keyword1, keyword2}

\maketitle

\section{Introduction}
Introduction.

\bibliographystyle{ACM-Reference-Format}
\bibliography{references}
\end{document}
''',
            'guidelines_url': 'https://www.acm.org/publications/authors',
            'submission_system': 'Various'
        }

    @staticmethod
    def nature() -> Dict[str, Any]:
        """Nature journal template"""
        return {
            'name': 'Nature',
            'publisher': 'Nature Publishing Group',
            'field': 'Multidisciplinary Science',
            'citation_style': 'Nature',
            'document_class': 'article',
            'page_layout': {'columns': 1, 'page_size': 'a4'},
            'font_settings': {'main_font': 'Times', 'font_size': '12pt'},
            'requirements': {
                'word_limit': 5000,
                'abstract_limit': 200,
                'max_figures': 6,
                'max_display_items': 8
            },
            'latex_template': r'''
\documentclass[12pt,a4paper]{article}
\usepackage{times}
\usepackage{graphicx}
\usepackage{natbib}

\begin{document}
\title{Title}
\author{Authors}
\date{}
\maketitle

\begin{abstract}
Abstract (max 200 words).
\end{abstract}

\section{Introduction}
Introduction text.

\bibliographystyle{naturemag}
\bibliography{references}
\end{document}
''',
            'guidelines_url': 'https://www.nature.com/nature/for-authors',
            'submission_system': 'Editorial Manager'
        }

    @staticmethod
    def science() -> Dict[str, Any]:
        """Science journal template"""
        return {
            'name': 'Science',
            'publisher': 'AAAS',
            'field': 'Multidisciplinary Science',
            'citation_style': 'Science',
            'document_class': 'article',
            'page_layout': {'columns': 1, 'page_size': 'letter'},
            'font_settings': {'main_font': 'Times', 'font_size': '12pt'},
            'requirements': {
                'word_limit': 4500,
                'abstract_limit': 125,
                'max_references': 40
            },
            'latex_template': r'''
\documentclass[12pt]{article}
\usepackage{times}
\usepackage{graphicx}

\begin{document}
\title{Title}
\author{Authors}
\maketitle

\begin{abstract}
Abstract (max 125 words).
\end{abstract}

\section*{Introduction}
Text here.

\bibliographystyle{Science}
\bibliography{references}
\end{document}
''',
            'guidelines_url': 'https://www.science.org/content/page/instructions-authors',
            'submission_system': 'GEMS'
        }

    @staticmethod
    def springer() -> Dict[str, Any]:
        """Springer journal template"""
        return {
            'name': 'Springer',
            'publisher': 'Springer Nature',
            'field': 'Various',
            'citation_style': 'Springer',
            'document_class': 'svjour3',
            'page_layout': {'columns': 1, 'page_size': 'a4'},
            'font_settings': {'main_font': 'Times', 'font_size': '10pt'},
            'requirements': {'abstract_limit': 250, 'keywords': 6},
            'latex_template': r'''
\documentclass{svjour3}
\usepackage{graphicx}

\begin{document}
\title{Title}
\author{Author Name}
\institute{Institution}
\date{}

\maketitle

\begin{abstract}
Abstract text.
\keywords{keyword1, keyword2}
\end{abstract}

\section{Introduction}
Introduction.

\bibliographystyle{spbasic}
\bibliography{references}
\end{document}
''',
            'guidelines_url': 'https://www.springer.com/gp/authors-editors',
            'submission_system': 'Editorial Manager'
        }

    @staticmethod
    def elsevier() -> Dict[str, Any]:
        """Elsevier journal template"""
        return {
            'name': 'Elsevier',
            'publisher': 'Elsevier',
            'field': 'Various',
            'citation_style': 'Numbered',
            'document_class': 'elsarticle',
            'page_layout': {'columns': 1, 'page_size': 'a4'},
            'font_settings': {'main_font': 'Times', 'font_size': '12pt'},
            'requirements': {'abstract_limit': 300},
            'latex_template': r'''
\documentclass[review]{elsarticle}
\usepackage{graphicx}

\begin{document}
\begin{frontmatter}
\title{Title}
\author{Author}
\address{Institution}

\begin{abstract}
Abstract text.
\end{abstract}

\begin{keyword}
keyword1 \sep keyword2
\end{keyword}
\end{frontmatter}

\section{Introduction}
Introduction.

\bibliographystyle{elsarticle-num}
\bibliography{references}
\end{document}
''',
            'guidelines_url': 'https://www.elsevier.com/authors',
            'submission_system': 'Editorial System'
        }

    @staticmethod
    def plos_one() -> Dict[str, Any]:
        """PLOS ONE template"""
        return {
            'name': 'PLOS ONE',
            'publisher': 'PLOS',
            'field': 'Multidisciplinary Science',
            'citation_style': 'Vancouver',
            'document_class': 'plos2015',
            'page_layout': {'columns': 1, 'page_size': 'letter'},
            'font_settings': {'main_font': 'Arial', 'font_size': '11pt'},
            'requirements': {'word_limit': None, 'abstract_limit': 300},
            'latex_template': r'''
\documentclass{plos2015}
\usepackage{graphicx}

\begin{document}
\title{Title}
\author{Author Names}

\begin{abstract}
Abstract (max 300 words).
\end{abstract}

\section*{Introduction}
Introduction.

\bibliographystyle{plos2015}
\bibliography{references}
\end{document}
''',
            'guidelines_url': 'https://journals.plos.org/plosone/s/submission-guidelines',
            'submission_system': 'Editorial Manager'
        }

    @staticmethod
    def arxiv() -> Dict[str, Any]:
        """arXiv preprint template"""
        return {
            'name': 'arXiv',
            'publisher': 'Cornell University',
            'field': 'All fields',
            'citation_style': 'Flexible',
            'document_class': 'article',
            'page_layout': {'columns': 1, 'page_size': 'letter'},
            'font_settings': {'main_font': 'Computer Modern', 'font_size': '11pt'},
            'requirements': {'word_limit': None, 'abstract_limit': 1920},
            'latex_template': r'''
\documentclass[11pt]{article}
\usepackage{arxiv}
\usepackage{graphicx}
\usepackage{amsmath}

\title{Paper Title}
\author{Author Names}

\begin{document}
\maketitle

\begin{abstract}
Abstract text.
\end{abstract}

\keywords{keyword1 \and keyword2}

\section{Introduction}
Introduction.

\bibliographystyle{plain}
\bibliography{references}
\end{document}
''',
            'guidelines_url': 'https://arxiv.org/help/submit',
            'submission_system': 'arXiv.org'
        }

    @staticmethod
    def neurips() -> Dict[str, Any]:
        """NeurIPS conference template"""
        return {
            'name': 'NeurIPS',
            'publisher': 'Neural Information Processing Systems',
            'field': 'Machine Learning/AI',
            'citation_style': 'NeurIPS',
            'document_class': 'article',
            'page_layout': {'columns': 2, 'page_size': 'letter'},
            'font_settings': {'main_font': 'Times', 'font_size': '10pt'},
            'requirements': {
                'word_limit': None,
                'page_limit': 9,
                'abstract_limit': None
            },
            'latex_template': r'''
\documentclass{article}
\usepackage{neurips_2023}
\usepackage{graphicx}

\title{Paper Title}
\author{Author Names}

\begin{document}
\maketitle

\begin{abstract}
Abstract text.
\end{abstract}

\section{Introduction}
Introduction.

\bibliographystyle{plain}
\bibliography{references}
\end{document}
''',
            'guidelines_url': 'https://neurips.cc/Conferences/2024/CallForPapers',
            'submission_system': 'OpenReview'
        }

    @staticmethod
    def icml() -> Dict[str, Any]:
        """ICML conference template"""
        return {
            'name': 'ICML',
            'publisher': 'International Conference on Machine Learning',
            'field': 'Machine Learning',
            'citation_style': 'ICML',
            'document_class': 'article',
            'page_layout': {'columns': 2, 'page_size': 'letter'},
            'font_settings': {'main_font': 'Times', 'font_size': '10pt'},
            'requirements': {'page_limit': 8, 'abstract_limit': None},
            'latex_template': r'''
\documentclass{article}
\usepackage{icml2024}
\usepackage{graphicx}

\title{Paper Title}
\author{Authors}

\begin{document}
\maketitle

\begin{abstract}
Abstract.
\end{abstract}

\section{Introduction}
Text.

\bibliography{references}
\bibliographystyle{icml2024}
\end{document}
''',
            'guidelines_url': 'https://icml.cc/Conferences/2024/StyleAuthorInstructions',
            'submission_system': 'CMT'
        }

    @staticmethod
    def cvpr() -> Dict[str, Any]:
        """CVPR conference template"""
        return {
            'name': 'CVPR',
            'publisher': 'IEEE Computer Vision',
            'field': 'Computer Vision',
            'citation_style': 'IEEE',
            'document_class': 'article',
            'page_layout': {'columns': 2, 'page_size': 'letter'},
            'font_settings': {'main_font': 'Times', 'font_size': '10pt'},
            'requirements': {'page_limit': 8},
            'latex_template': r'''
\documentclass[10pt,twocolumn,letterpaper]{article}
\usepackage{cvpr}
\usepackage{graphicx}

\title{Paper Title}
\author{Authors}

\begin{document}
\maketitle

\begin{abstract}
Abstract.
\end{abstract}

\section{Introduction}
Text.

\bibliographystyle{ieee_fullname}
\bibliography{references}
\end{document}
''',
            'guidelines_url': 'https://cvpr.thecvf.com/Conferences/2024/AuthorGuidelines',
            'submission_system': 'CMT'
        }

    @staticmethod
    def frontiers() -> Dict[str, Any]:
        """Frontiers journal template"""
        return {
            'name': 'Frontiers',
            'publisher': 'Frontiers Media',
            'field': 'Various',
            'citation_style': 'Vancouver',
            'document_class': 'article',
            'page_layout': {'columns': 1, 'page_size': 'a4'},
            'font_settings': {'main_font': 'Arial', 'font_size': '12pt'},
            'requirements': {'word_limit': None, 'abstract_limit': 350},
            'latex_template': r'''
\documentclass[12pt]{article}
\usepackage{graphicx}

\title{Title}
\author{Authors}

\begin{document}
\maketitle

\begin{abstract}
Abstract (max 350 words).
\end{abstract}

\section{Introduction}
Text.

\bibliographystyle{frontiersinSCNS_ENG_HUMS}
\bibliography{references}
\end{document}
''',
            'guidelines_url': 'https://www.frontiersin.org/guidelines/author-guidelines',
            'submission_system': 'Frontiers Editorial System'
        }

    @staticmethod
    def mdpi() -> Dict[str, Any]:
        """MDPI journal template"""
        return {
            'name': 'MDPI',
            'publisher': 'MDPI',
            'field': 'Various',
            'citation_style': 'Vancouver',
            'document_class': 'article',
            'page_layout': {'columns': 1, 'page_size': 'a4'},
            'font_settings': {'main_font': 'Palatino', 'font_size': '10pt'},
            'requirements': {'word_limit': None, 'abstract_limit': 200},
            'latex_template': r'''
\documentclass[journal,article,submit,pdftex]{Definitions/mdpi}
\usepackage{graphicx}

\Title{Paper Title}
\Author{Author Names}

\begin{document}
\maketitle

\begin{abstract}
Abstract (200 words max).
\end{abstract}

\keywords{keyword1; keyword2}

\section{Introduction}
Text.

\bibliographystyle{mdpi}
\bibliography{references}
\end{document}
''',
            'guidelines_url': 'https://www.mdpi.com/authors',
            'submission_system': 'SUSY'
        }

    @staticmethod
    def jmlr() -> Dict[str, Any]:
        """JMLR template"""
        return {
            'name': 'Journal of Machine Learning Research',
            'publisher': 'JMLR',
            'field': 'Machine Learning',
            'citation_style': 'Harvard',
            'document_class': 'article',
            'page_layout': {'columns': 1, 'page_size': 'letter'},
            'font_settings': {'main_font': 'Times', 'font_size': '11pt'},
            'requirements': {'word_limit': None},
            'latex_template': r'''
\documentclass[twoside,11pt]{article}
\usepackage{jmlr2e}
\usepackage{graphicx}

\title{Paper Title}
\author{Author Names}

\begin{document}
\maketitle

\begin{abstract}
Abstract.
\end{abstract}

\section{Introduction}
Text.

\bibliography{references}
\end{document}
''',
            'guidelines_url': 'https://jmlr.org/author-info.html',
            'submission_system': 'JMLR Editorial System'
        }

    @staticmethod
    def aaai() -> Dict[str, Any]:
        """AAAI conference template"""
        return {
            'name': 'AAAI',
            'publisher': 'Association for the Advancement of Artificial Intelligence',
            'field': 'Artificial Intelligence',
            'citation_style': 'AAAI',
            'document_class': 'article',
            'page_layout': {'columns': 2, 'page_size': 'letter'},
            'font_settings': {'main_font': 'Times', 'font_size': '10pt'},
            'requirements': {'page_limit': 7},
            'latex_template': r'''
\documentclass[letterpaper]{article}
\usepackage{aaai24}
\usepackage{graphicx}

\title{Paper Title}
\author{Authors}

\begin{document}
\maketitle

\begin{abstract}
Abstract.
\end{abstract}

\section{Introduction}
Text.

\bibliography{references}
\bibliographystyle{aaai24}
\end{document}
''',
            'guidelines_url': 'https://aaai.org/authorkit24/',
            'submission_system': 'CMT'
        }

    @staticmethod
    def acl() -> Dict[str, Any]:
        """ACL conference template"""
        return {
            'name': 'ACL',
            'publisher': 'Association for Computational Linguistics',
            'field': 'Natural Language Processing',
            'citation_style': 'ACL',
            'document_class': 'article',
            'page_layout': {'columns': 2, 'page_size': 'letter'},
            'font_settings': {'main_font': 'Times', 'font_size': '11pt'},
            'requirements': {'page_limit': 8},
            'latex_template': r'''
\documentclass[11pt]{article}
\usepackage{acl}
\usepackage{graphicx}

\title{Paper Title}
\author{Authors}

\begin{document}
\maketitle

\begin{abstract}
Abstract.
\end{abstract}

\section{Introduction}
Text.

\bibliography{references}
\bibliographystyle{acl_natbib}
\end{document}
''',
            'guidelines_url': 'https://acl-org.github.io/ACLPUB/formatting.html',
            'submission_system': 'START'
        }

    @staticmethod
    def emnlp() -> Dict[str, Any]:
        """EMNLP conference template"""
        return JournalTemplates.acl()  # Same as ACL

    @staticmethod
    def cell() -> Dict[str, Any]:
        """Cell journal template"""
        return {
            'name': 'Cell',
            'publisher': 'Cell Press',
            'field': 'Life Sciences',
            'citation_style': 'Cell',
            'document_class': 'article',
            'page_layout': {'columns': 1, 'page_size': 'letter'},
            'font_settings': {'main_font': 'Arial', 'font_size': '12pt'},
            'requirements': {
                'word_limit': 5000,
                'abstract_limit': 150,
                'max_figures': 7
            },
            'latex_template': r'''
\documentclass[12pt]{article}
\usepackage{times}
\usepackage{graphicx}

\title{Title}
\author{Authors}

\begin{document}
\maketitle

\begin{abstract}
Abstract (150 words max).
\end{abstract}

\section{Introduction}
Text.

\bibliographystyle{cell}
\bibliography{references}
\end{document}
''',
            'guidelines_url': 'https://www.cell.com/cell/authors',
            'submission_system': 'Editorial Manager'
        }

    @staticmethod
    def lancet() -> Dict[str, Any]:
        """The Lancet template"""
        return {
            'name': 'The Lancet',
            'publisher': 'Elsevier',
            'field': 'Medicine',
            'citation_style': 'Vancouver',
            'document_class': 'article',
            'page_layout': {'columns': 1, 'page_size': 'a4'},
            'font_settings': {'main_font': 'Times', 'font_size': '12pt'},
            'requirements': {
                'word_limit': 4500,
                'abstract_limit': 250,
                'max_references': 40
            },
            'latex_template': r'''
\documentclass[12pt]{article}
\usepackage{times}
\usepackage{graphicx}

\title{Title}
\author{Authors}

\begin{document}
\maketitle

\begin{abstract}
Abstract (max 250 words).
\end{abstract}

\section{Introduction}
Text.

\bibliographystyle{vancouver}
\bibliography{references}
\end{document}
''',
            'guidelines_url': 'https://www.thelancet.com/lancet/information-for-authors',
            'submission_system': 'Editorial Manager'
        }

    @staticmethod
    def jama() -> Dict[str, Any]:
        """JAMA template"""
        return {
            'name': 'JAMA',
            'publisher': 'American Medical Association',
            'field': 'Medicine',
            'citation_style': 'AMA',
            'document_class': 'article',
            'page_layout': {'columns': 1, 'page_size': 'letter'},
            'font_settings': {'main_font': 'Times', 'font_size': '12pt'},
            'requirements': {
                'word_limit': 3000,
                'abstract_limit': 350,
                'max_references': 50
            },
            'latex_template': r'''
\documentclass[12pt]{article}
\usepackage{times}
\usepackage{graphicx}

\title{Title}
\author{Authors}

\begin{document}
\maketitle

\begin{abstract}
Structured abstract (max 350 words).
\end{abstract}

\section{Introduction}
Text.

\bibliographystyle{jama}
\bibliography{references}
\end{document}
''',
            'guidelines_url': 'https://jamanetwork.com/journals/jama/pages/instructions-for-authors',
            'submission_system': 'Editorial Manager'
        }

    @staticmethod
    def pnas() -> Dict[str, Any]:
        """PNAS template"""
        return {
            'name': 'PNAS',
            'publisher': 'National Academy of Sciences',
            'field': 'Multidisciplinary Science',
            'citation_style': 'PNAS',
            'document_class': 'pnas',
            'page_layout': {'columns': 2, 'page_size': 'letter'},
            'font_settings': {'main_font': 'Times', 'font_size': '9pt'},
            'requirements': {
                'word_limit': 3000,
                'abstract_limit': 250,
                'max_figures': 6
            },
            'latex_template': r'''
\documentclass[9pt,twoside,lineno]{pnas-new}
\usepackage{graphicx}

\title{Title}
\author{Authors}

\begin{document}
\maketitle

\begin{abstract}
Abstract (max 250 words).
\end{abstract}

\keywords{keyword1, keyword2}

\section*{Introduction}
Text.

\bibliographystyle{pnas-new}
\bibliography{references}
\end{document}
''',
            'guidelines_url': 'https://www.pnas.org/author-center',
            'submission_system': 'Editorial Express'
        }

    @staticmethod
    def aps_journals() -> Dict[str, Any]:
        """APS (American Physical Society) journals template"""
        return {
            'name': 'APS Journals (Physical Review)',
            'publisher': 'American Physical Society',
            'field': 'Physics',
            'citation_style': 'APS',
            'document_class': 'revtex4-2',
            'page_layout': {'columns': 2, 'page_size': 'letter'},
            'font_settings': {'main_font': 'Computer Modern', 'font_size': '10pt'},
            'requirements': {'word_limit': None},
            'latex_template': r'''
\documentclass[aps,prl,reprint]{revtex4-2}
\usepackage{graphicx}

\begin{document}
\title{Paper Title}
\author{Author Names}
\affiliation{Institution}

\begin{abstract}
Abstract text.
\end{abstract}

\maketitle

\section{Introduction}
Text.

\bibliography{references}
\end{document}
''',
            'guidelines_url': 'https://journals.aps.org/authors',
            'submission_system': 'Physical Review Editorial System'
        }


# Export
__all__ = ['JournalTemplates']
