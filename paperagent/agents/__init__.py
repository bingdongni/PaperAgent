"""
Agents module for PaperAgent
"""

from paperagent.agents.base_agent import BaseAgent
from paperagent.agents.literature_agent import LiteratureAgent
from paperagent.agents.experiment_agent import ExperimentAgent
from paperagent.agents.writing_agent import WritingAgent
from paperagent.agents.boss_agent import BossAgent
from paperagent.agents.formatting_agent import FormattingAgent

__all__ = [
    "BaseAgent",
    "LiteratureAgent",
    "ExperimentAgent",
    "WritingAgent",
    "BossAgent",
    "FormattingAgent",
]
