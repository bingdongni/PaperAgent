"""
Base agent class for all PaperAgent agents
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime
from loguru import logger

from paperagent.core.llm_manager import llm_manager
from paperagent.database.models import Task, TaskStatus, AgentType
from paperagent.database.database import get_db_context


class BaseAgent(ABC):
    """
    Base class for all agents in PaperAgent

    Implements common functionality:
    - Task logging
    - LLM interaction
    - Error handling
    - Progress tracking
    """

    def __init__(self, agent_type: AgentType, name: str):
        self.agent_type = agent_type
        self.name = name
        self.llm = llm_manager

    @abstractmethod
    def execute(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent's main task

        Args:
            task_input: Input data for the task

        Returns:
            Task output dictionary
        """
        pass

    def run_task(self, task_id: int) -> Dict[str, Any]:
        """
        Run a task with full lifecycle management

        Args:
            task_id: Database task ID

        Returns:
            Task result
        """
        try:
            # Update task status to in_progress
            with get_db_context() as db:
                task = db.query(Task).filter(Task.id == task_id).first()
                if not task:
                    raise ValueError(f"Task {task_id} not found")

                task.status = TaskStatus.IN_PROGRESS
                task.started_at = datetime.utcnow()
                db.commit()

                logger.info(f"[{self.name}] Starting task {task_id}: {task.name}")

                # Execute task
                result = self.execute(task.input_data or {})

                # Update task with result
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.utcnow()
                task.output_data = result
                db.commit()

                logger.info(f"[{self.name}] Completed task {task_id}")

                return result

        except Exception as e:
            logger.error(f"[{self.name}] Task {task_id} failed: {e}")

            # Update task with error
            with get_db_context() as db:
                task = db.query(Task).filter(Task.id == task_id).first()
                if task:
                    task.status = TaskStatus.FAILED
                    task.error_message = str(e)
                    task.completed_at = datetime.utcnow()
                    db.commit()

            raise

    def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        Generate text using LLM

        Args:
            prompt: User prompt
            system_prompt: System prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional LLM parameters

        Returns:
            Generated text
        """
        try:
            return self.llm.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
        except Exception as e:
            logger.error(f"[{self.name}] LLM generation error: {e}")
            raise

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        Chat with LLM

        Args:
            messages: List of message dictionaries
            max_tokens: Maximum tokens
            temperature: Sampling temperature
            **kwargs: Additional parameters

        Returns:
            LLM response
        """
        try:
            return self.llm.chat(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
        except Exception as e:
            logger.error(f"[{self.name}] LLM chat error: {e}")
            raise

    def log_action(
        self,
        action: str,
        details: Optional[Dict[str, Any]] = None,
        success: bool = True
    ):
        """
        Log agent action to database

        Args:
            action: Action description
            details: Action details
            success: Whether action was successful
        """
        from paperagent.database.models import AgentLog

        try:
            with get_db_context() as db:
                log = AgentLog(
                    agent_type=self.agent_type,
                    action=action,
                    details=details or {},
                    success=success,
                    timestamp=datetime.utcnow()
                )
                db.add(log)
                db.commit()
        except Exception as e:
            logger.error(f"Failed to log action: {e}")

    def validate_input(self, task_input: Dict[str, Any], required_fields: List[str]):
        """
        Validate task input has required fields

        Args:
            task_input: Input dictionary
            required_fields: List of required field names

        Raises:
            ValueError: If required field is missing
        """
        for field in required_fields:
            if field not in task_input:
                raise ValueError(f"Missing required field: {field}")

    def parse_json_response(self, response: str) -> Dict[str, Any]:
        """
        Parse JSON response from LLM

        Args:
            response: LLM response text

        Returns:
            Parsed JSON dictionary
        """
        import json
        import re

        try:
            # Try to extract JSON from markdown code blocks
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                response = json_match.group(1)

            # Parse JSON
            return json.loads(response)

        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}\nResponse: {response}")
            raise ValueError(f"Failed to parse JSON response: {e}")

    def create_subtask(
        self,
        project_id: int,
        parent_task_id: int,
        name: str,
        description: str,
        agent_type: AgentType,
        input_data: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Create a subtask

        Args:
            project_id: Project ID
            parent_task_id: Parent task ID
            name: Task name
            description: Task description
            agent_type: Agent type for this task
            input_data: Task input data

        Returns:
            Created task ID
        """
        with get_db_context() as db:
            task = Task(
                project_id=project_id,
                parent_task_id=parent_task_id,
                name=name,
                description=description,
                agent_type=agent_type,
                status=TaskStatus.PENDING,
                input_data=input_data or {}
            )
            db.add(task)
            db.commit()
            db.refresh(task)

            logger.info(f"[{self.name}] Created subtask {task.id}: {name}")
            return task.id
