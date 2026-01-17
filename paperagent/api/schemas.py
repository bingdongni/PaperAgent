"""
Pydantic schemas for API requests and responses
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


# Enums
class TaskStatusEnum(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentTypeEnum(str, Enum):
    BOSS = "boss"
    LITERATURE = "literature"
    EXPERIMENT = "experiment"
    WRITING = "writing"
    FORMATTING = "formatting"
    MANAGEMENT = "management"


# Project Schemas
class ProjectCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    research_field: str
    keywords: Optional[List[str]] = []


class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    research_field: Optional[str] = None
    keywords: Optional[List[str]] = None
    status: Optional[TaskStatusEnum] = None


class ProjectResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    research_field: str
    keywords: Optional[List[str]]
    status: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# Literature Schemas
class LiteratureSearch(BaseModel):
    query: str
    max_results: int = 25
    sources: List[str] = ["arxiv", "google_scholar"]


class LiteratureResponse(BaseModel):
    id: int
    title: str
    authors: Optional[List[str]]
    abstract: Optional[str]
    year: Optional[int]
    doi: Optional[str]
    url: Optional[str]
    citation_count: int
    source: Optional[str]
    relevance_score: Optional[float]

    class Config:
        from_attributes = True


# Experiment Schemas
class ExperimentCreate(BaseModel):
    name: str
    description: Optional[str]
    objective: str
    field: str
    resources: Optional[str] = "Standard"


class ExperimentResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    methodology: Optional[str]
    hypothesis: Optional[str]
    status: str
    created_at: datetime

    class Config:
        from_attributes = True


# Paper Schemas
class PaperCreate(BaseModel):
    title: str
    objective: str
    target_journal: Optional[str] = None


class PaperUpdate(BaseModel):
    title: Optional[str] = None
    abstract: Optional[str] = None
    introduction: Optional[str] = None
    literature_review: Optional[str] = None
    methodology: Optional[str] = None
    results: Optional[str] = None
    discussion: Optional[str] = None
    conclusion: Optional[str] = None


class PaperResponse(BaseModel):
    id: int
    title: str
    abstract: Optional[str]
    target_journal: Optional[str]
    word_count: Optional[int]
    status: str
    version: int
    created_at: datetime

    class Config:
        from_attributes = True


# Task Schemas
class TaskCreate(BaseModel):
    project_id: int
    name: str
    description: Optional[str]
    agent_type: AgentTypeEnum
    input_data: Optional[Dict[str, Any]] = {}
    priority: int = 0


class TaskResponse(BaseModel):
    id: int
    project_id: int
    name: str
    description: Optional[str]
    agent_type: str
    status: str
    priority: int
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]

    class Config:
        from_attributes = True


# Agent Action Schemas
class LiteratureAction(BaseModel):
    action: str  # recommend_topics, search_literature, analyze_paper, etc.
    project_id: Optional[int] = None
    query: Optional[str] = None
    max_results: Optional[int] = 25
    field: Optional[str] = None
    keywords: Optional[str] = None
    paper_id: Optional[int] = None
    research_field: Optional[str] = None


class ExperimentAction(BaseModel):
    action: str  # design_experiment, analyze_data, etc.
    project_id: int
    name: Optional[str] = None
    objective: Optional[str] = None
    field: Optional[str] = None
    resources: Optional[str] = None
    experiment_id: Optional[int] = None
    data: Optional[Any] = None


class WritingAction(BaseModel):
    action: str  # create_structure, write_section, write_abstract, etc.
    project_id: int
    paper_id: Optional[int] = None
    title: Optional[str] = None
    objective: Optional[str] = None
    section: Optional[str] = None
    context: Optional[str] = None
    text: Optional[str] = None


class BossAction(BaseModel):
    action: str  # create_project, decompose_task, execute_workflow, etc.
    name: Optional[str] = None
    description: Optional[str] = None
    research_field: Optional[str] = None
    keywords: Optional[List[str]] = None
    project_id: Optional[int] = None
    goal: Optional[str] = None
    output_type: Optional[str] = None
    output_id: Optional[int] = None


# Response Models
class ActionResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None


class ErrorResponse(BaseModel):
    detail: str
