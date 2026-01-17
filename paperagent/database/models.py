"""
Database models for PaperAgent
"""

from datetime import datetime
from typing import Optional, List
from sqlalchemy import (
    Column, Integer, String, Text, DateTime, Boolean,
    ForeignKey, JSON, Float, Enum as SQLEnum
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import enum

Base = declarative_base()


class TaskStatus(enum.Enum):
    """Task status enumeration"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentType(enum.Enum):
    """Agent type enumeration"""
    BOSS = "boss"
    LITERATURE = "literature"
    EXPERIMENT = "experiment"
    WRITING = "writing"
    FORMATTING = "formatting"
    MANAGEMENT = "management"


class Project(Base):
    """Research project model"""
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    research_field = Column(String(255))
    keywords = Column(JSON)  # List of keywords
    status = Column(SQLEnum(TaskStatus), default=TaskStatus.PENDING)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    tasks = relationship("Task", back_populates="project", cascade="all, delete-orphan")
    papers = relationship("Paper", back_populates="project", cascade="all, delete-orphan")
    literature = relationship("Literature", back_populates="project", cascade="all, delete-orphan")
    experiments = relationship("Experiment", back_populates="project", cascade="all, delete-orphan")


class Task(Base):
    """Task model for tracking agent activities"""
    __tablename__ = "tasks"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    agent_type = Column(SQLEnum(AgentType), nullable=False)
    status = Column(SQLEnum(TaskStatus), default=TaskStatus.PENDING)
    priority = Column(Integer, default=0)
    parent_task_id = Column(Integer, ForeignKey("tasks.id"), nullable=True)

    # Task metadata
    input_data = Column(JSON)
    output_data = Column(JSON)
    error_message = Column(Text)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    project = relationship("Project", back_populates="tasks")
    subtasks = relationship("Task", remote_side=[id], backref="parent_task")


class Literature(Base):
    """Literature/Paper reference model"""
    __tablename__ = "literature"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)

    # Basic information
    title = Column(Text, nullable=False)
    authors = Column(JSON)  # List of authors
    abstract = Column(Text)
    keywords = Column(JSON)

    # Publication details
    journal = Column(String(255))
    conference = Column(String(255))
    year = Column(Integer)
    volume = Column(String(50))
    issue = Column(String(50))
    pages = Column(String(50))

    # Identifiers
    doi = Column(String(255), unique=True, nullable=True)
    arxiv_id = Column(String(100), unique=True, nullable=True)
    pmid = Column(String(50), unique=True, nullable=True)
    url = Column(Text)

    # Metadata
    citation_count = Column(Integer, default=0)
    source = Column(String(50))  # arxiv, google_scholar, etc.
    bibtex = Column(Text)
    pdf_path = Column(Text)

    # Analysis
    relevance_score = Column(Float)
    summary = Column(Text)
    key_findings = Column(JSON)
    research_gaps = Column(JSON)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    project = relationship("Project", back_populates="literature")


class Experiment(Base):
    """Experiment model for tracking research experiments"""
    __tablename__ = "experiments"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)

    # Basic information
    name = Column(String(255), nullable=False)
    description = Column(Text)
    methodology = Column(Text)
    hypothesis = Column(Text)

    # Experiment setup
    parameters = Column(JSON)
    dataset_info = Column(JSON)
    equipment = Column(JSON)

    # Results
    results = Column(JSON)
    data_files = Column(JSON)  # List of file paths
    figures = Column(JSON)  # List of figure paths
    tables = Column(JSON)

    # Analysis
    statistical_tests = Column(JSON)
    conclusions = Column(Text)
    limitations = Column(Text)

    # Status
    status = Column(SQLEnum(TaskStatus), default=TaskStatus.PENDING)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    project = relationship("Project", back_populates="experiments")


class Paper(Base):
    """Research paper model"""
    __tablename__ = "papers"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)

    # Basic information
    title = Column(Text, nullable=False)
    abstract = Column(Text)
    keywords = Column(JSON)

    # Content sections
    introduction = Column(Text)
    literature_review = Column(Text)
    methodology = Column(Text)
    results = Column(Text)
    discussion = Column(Text)
    conclusion = Column(Text)

    # References
    references = Column(JSON)

    # Formatting
    target_journal = Column(String(255))
    journal_format = Column(String(100))  # IEEE, APA, MLA, etc.
    word_count = Column(Integer)

    # Files
    latex_content = Column(Text)
    latex_path = Column(Text)
    pdf_path = Column(Text)
    docx_path = Column(Text)

    # Metadata
    version = Column(Integer, default=1)
    status = Column(SQLEnum(TaskStatus), default=TaskStatus.PENDING)

    # Quality metrics
    readability_score = Column(Float)
    grammar_score = Column(Float)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    submitted_at = Column(DateTime, nullable=True)

    # Relationships
    project = relationship("Project", back_populates="papers")
    revisions = relationship("PaperRevision", back_populates="paper", cascade="all, delete-orphan")


class PaperRevision(Base):
    """Paper revision history"""
    __tablename__ = "paper_revisions"

    id = Column(Integer, primary_key=True, index=True)
    paper_id = Column(Integer, ForeignKey("papers.id"), nullable=False)

    version = Column(Integer, nullable=False)
    content = Column(Text)
    changes_summary = Column(Text)
    reviewer_comments = Column(JSON)

    created_at = Column(DateTime, default=datetime.utcnow)
    created_by = Column(String(255))

    # Relationships
    paper = relationship("Paper", back_populates="revisions")


class AgentLog(Base):
    """Agent execution logs"""
    __tablename__ = "agent_logs"

    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(Integer, ForeignKey("tasks.id"), nullable=True)
    agent_type = Column(SQLEnum(AgentType), nullable=False)

    action = Column(String(255))
    details = Column(JSON)
    input_data = Column(JSON)
    output_data = Column(JSON)

    success = Column(Boolean, default=True)
    error_message = Column(Text)
    execution_time = Column(Float)  # seconds

    timestamp = Column(DateTime, default=datetime.utcnow)


class Citation(Base):
    """Citation model for managing references"""
    __tablename__ = "citations"

    id = Column(Integer, primary_key=True, index=True)

    # Citation details
    citation_key = Column(String(255), unique=True, nullable=False)
    entry_type = Column(String(50))  # article, book, inproceedings, etc.

    # Basic fields
    title = Column(Text, nullable=False)
    authors = Column(JSON)
    year = Column(Integer)

    # Publication details
    journal = Column(String(255))
    booktitle = Column(String(255))
    publisher = Column(String(255))
    volume = Column(String(50))
    number = Column(String(50))
    pages = Column(String(50))

    # Identifiers
    doi = Column(String(255))
    isbn = Column(String(50))
    issn = Column(String(50))
    url = Column(Text)

    # BibTeX
    bibtex = Column(Text)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class JournalTemplate(Base):
    """Journal template model"""
    __tablename__ = "journal_templates"

    id = Column(Integer, primary_key=True, index=True)

    name = Column(String(255), nullable=False, unique=True)
    publisher = Column(String(255))
    field = Column(String(255))  # Computer Science, Biology, etc.

    # Format specifications
    citation_style = Column(String(50))  # IEEE, APA, MLA, etc.
    document_class = Column(String(100))
    page_layout = Column(JSON)
    font_settings = Column(JSON)

    # Requirements
    word_limit = Column(Integer)
    abstract_limit = Column(Integer)
    max_figures = Column(Integer)
    max_tables = Column(Integer)

    # Template files
    latex_template = Column(Text)
    template_path = Column(Text)

    # Metadata
    guidelines_url = Column(Text)
    submission_system = Column(String(255))

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
