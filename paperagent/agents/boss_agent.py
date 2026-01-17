"""
Boss Agent - Central orchestration and quality control
"""

from typing import Dict, Any, List, Optional
import json
from datetime import datetime
from loguru import logger

from paperagent.agents.base_agent import BaseAgent
from paperagent.agents.literature_agent import LiteratureAgent
from paperagent.agents.experiment_agent import ExperimentAgent
from paperagent.agents.writing_agent import WritingAgent
from paperagent.database.models import AgentType, Task, Project, TaskStatus
from paperagent.database.database import get_db_context
from paperagent.core.prompts import BossPrompts


class BossAgent(BaseAgent):
    """
    Boss Agent - Central orchestrator for PaperAgent

    Responsibilities:
    - Task decomposition and planning
    - Agent coordination
    - Progress monitoring
    - Quality control
    - Workflow management
    """

    def __init__(self):
        super().__init__(AgentType.BOSS, "Boss Agent")
        self.prompts = BossPrompts()

        # Initialize sub-agents
        self.literature_agent = LiteratureAgent()
        self.experiment_agent = ExperimentAgent()
        self.writing_agent = WritingAgent()

    def execute(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute orchestration tasks

        Args:
            task_input: Must contain 'action' key with value:
                - 'create_project': Create new research project
                - 'decompose_task': Decompose project into tasks
                - 'execute_workflow': Execute full research workflow
                - 'monitor_progress': Check project progress
                - 'quality_check': Perform quality control

        Returns:
            Task results
        """
        action = task_input.get('action')

        if action == 'create_project':
            return self.create_project(task_input)
        elif action == 'decompose_task':
            return self.decompose_project_tasks(task_input)
        elif action == 'execute_workflow':
            return self.execute_research_workflow(task_input)
        elif action == 'monitor_progress':
            return self.monitor_project_progress(task_input)
        elif action == 'quality_check':
            return self.perform_quality_check(task_input)
        else:
            raise ValueError(f"Unknown action: {action}")

    def create_project(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new research project

        Args:
            task_input: Contains 'name', 'description', 'research_field', 'keywords'

        Returns:
            Project details
        """
        self.validate_input(task_input, ['name', 'research_field'])

        name = task_input['name']
        description = task_input.get('description', '')
        research_field = task_input['research_field']
        keywords = task_input.get('keywords', [])

        logger.info(f"Creating project: {name}")

        # Create project in database
        with get_db_context() as db:
            project = Project(
                name=name,
                description=description,
                research_field=research_field,
                keywords=keywords,
                status=TaskStatus.PENDING
            )
            db.add(project)
            db.commit()
            db.refresh(project)

            project_id = project.id

        self.log_action("create_project", {"project_id": project_id, "name": name})

        return {
            "project_id": project_id,
            "name": name,
            "research_field": research_field,
            "status": "created"
        }

    def decompose_project_tasks(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decompose project into specific tasks

        Args:
            task_input: Contains 'project_id', 'goal', 'timeline', 'resources'

        Returns:
            Task decomposition plan
        """
        self.validate_input(task_input, ['project_id', 'goal'])

        project_id = task_input['project_id']
        goal = task_input['goal']
        timeline = task_input.get('timeline', 'Not specified')
        resources = task_input.get('resources', 'Standard')

        # Get project details
        with get_db_context() as db:
            project = db.query(Project).filter(Project.id == project_id).first()
            if not project:
                raise ValueError(f"Project {project_id} not found")

            field = project.research_field

        logger.info(f"Decomposing tasks for project: {project_id}")

        # Generate task decomposition prompt
        prompt = self.prompts.TASK_DECOMPOSITION.format(
            goal=goal,
            field=field,
            timeline=timeline,
            resources=resources
        )

        # Get LLM response
        response = self.generate_text(prompt, max_tokens=2000, temperature=0.6)

        try:
            decomposition = self.parse_json_response(response)

            # Create tasks in database
            task_ids = []
            with get_db_context() as db:
                for task_spec in decomposition.get('tasks', []):
                    # Map agent name to AgentType
                    agent_map = {
                        'literature': AgentType.LITERATURE,
                        'experiment': AgentType.EXPERIMENT,
                        'writing': AgentType.WRITING,
                        'formatting': AgentType.FORMATTING,
                        'management': AgentType.MANAGEMENT
                    }

                    agent_type = agent_map.get(task_spec.get('agent', 'literature'), AgentType.LITERATURE)

                    task = Task(
                        project_id=project_id,
                        name=task_spec.get('name', ''),
                        description=task_spec.get('description', ''),
                        agent_type=agent_type,
                        status=TaskStatus.PENDING,
                        priority=1 if task_spec.get('priority') == 'high' else 0,
                        input_data=task_spec
                    )
                    db.add(task)
                    db.commit()
                    db.refresh(task)
                    task_ids.append(task.id)

            self.log_action("decompose_tasks", {
                "project_id": project_id,
                "num_tasks": len(task_ids)
            })

            return {
                "project_id": project_id,
                "tasks": decomposition.get('tasks', []),
                "task_ids": task_ids,
                "workflow": decomposition.get('workflow', {})
            }

        except Exception as e:
            logger.error(f"Failed to decompose tasks: {e}")
            return {"error": str(e)}

    def execute_research_workflow(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute complete research workflow

        Args:
            task_input: Contains 'project_id'

        Returns:
            Workflow execution results
        """
        self.validate_input(task_input, ['project_id'])

        project_id = task_input['project_id']

        logger.info(f"Executing workflow for project: {project_id}")

        # Get project and tasks
        with get_db_context() as db:
            project = db.query(Project).filter(Project.id == project_id).first()
            if not project:
                raise ValueError(f"Project {project_id} not found")

            project.status = TaskStatus.IN_PROGRESS
            db.commit()

        results = {}

        # Phase 1: Literature Review
        logger.info("Phase 1: Literature Review")
        lit_result = self.literature_agent.execute({
            'action': 'search_literature',
            'query': f"{project.research_field} {' '.join(project.keywords or [])}",
            'max_results': 25,
            'project_id': project_id
        })
        results['literature_search'] = lit_result

        # Analyze top papers
        if lit_result.get('saved_papers', 0) > 0:
            gap_result = self.literature_agent.execute({
                'action': 'identify_gaps',
                'project_id': project_id,
                'research_field': project.research_field
            })
            results['research_gaps'] = gap_result

        # Phase 2: Experiment Design (if applicable)
        logger.info("Phase 2: Experiment Design")
        if 'experimental' in project.description.lower() or 'experiment' in project.description.lower():
            exp_result = self.experiment_agent.execute({
                'action': 'design_experiment',
                'objective': project.description,
                'field': project.research_field,
                'project_id': project_id
            })
            results['experiment_design'] = exp_result

        # Phase 3: Paper Writing
        logger.info("Phase 3: Paper Writing")
        paper_structure = self.writing_agent.execute({
            'action': 'create_structure',
            'title': project.name,
            'objective': project.description,
            'findings': results.get('research_gaps', {}).get('research_gaps', []),
            'project_id': project_id
        })
        results['paper_structure'] = paper_structure

        # Generate draft
        if paper_structure.get('paper_id'):
            draft_result = self.writing_agent.execute({
                'action': 'generate_draft',
                'paper_id': paper_structure['paper_id']
            })
            results['paper_draft'] = draft_result

        # Update project status
        with get_db_context() as db:
            project = db.query(Project).filter(Project.id == project_id).first()
            if project:
                project.status = TaskStatus.COMPLETED
                project.updated_at = datetime.utcnow()
                db.commit()

        self.log_action("execute_workflow", {"project_id": project_id})

        return {
            "project_id": project_id,
            "status": "completed",
            "results": results,
            "phases_completed": ["literature", "experiment", "writing"]
        }

    def monitor_project_progress(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Monitor project progress

        Args:
            task_input: Contains 'project_id'

        Returns:
            Progress report
        """
        self.validate_input(task_input, ['project_id'])

        project_id = task_input['project_id']

        with get_db_context() as db:
            project = db.query(Project).filter(Project.id == project_id).first()
            if not project:
                raise ValueError(f"Project {project_id} not found")

            # Get all tasks
            tasks = db.query(Task).filter(Task.project_id == project_id).all()

            # Calculate statistics
            total_tasks = len(tasks)
            completed_tasks = sum(1 for t in tasks if t.status == TaskStatus.COMPLETED)
            in_progress_tasks = sum(1 for t in tasks if t.status == TaskStatus.IN_PROGRESS)
            failed_tasks = sum(1 for t in tasks if t.status == TaskStatus.FAILED)

            # Get literature count
            from paperagent.database.models import Literature
            literature_count = db.query(Literature).filter(
                Literature.project_id == project_id
            ).count()

            # Get papers count
            from paperagent.database.models import Paper
            papers_count = db.query(Paper).filter(
                Paper.project_id == project_id
            ).count()

        progress_percentage = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0

        return {
            "project_id": project_id,
            "project_name": project.name,
            "status": project.status.value,
            "progress_percentage": round(progress_percentage, 2),
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "in_progress_tasks": in_progress_tasks,
            "failed_tasks": failed_tasks,
            "literature_collected": literature_count,
            "papers_count": papers_count
        }

    def perform_quality_check(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform quality check on output

        Args:
            task_input: Contains 'output_type', 'output_id', 'requirements'

        Returns:
            Quality assessment
        """
        self.validate_input(task_input, ['output_type', 'output_id'])

        output_type = task_input['output_type']
        output_id = task_input['output_id']
        requirements = task_input.get('requirements', {})

        logger.info(f"Performing quality check on {output_type}: {output_id}")

        # Load content based on type
        content = ""
        if output_type == 'paper':
            with get_db_context() as db:
                from paperagent.database.models import Paper
                paper = db.query(Paper).filter(Paper.id == output_id).first()
                if paper:
                    content = f"""
                    Title: {paper.title}
                    Abstract: {paper.abstract}
                    Introduction: {paper.introduction[:500] if paper.introduction else ''}
                    """

        # Generate quality check prompt
        prompt = self.prompts.QUALITY_CHECK.format(
            output_type=output_type,
            content=content,
            requirements=json.dumps(requirements) if isinstance(requirements, dict) else requirements
        )

        # Get LLM assessment
        response = self.generate_text(prompt, max_tokens=1500, temperature=0.3)

        try:
            assessment = self.parse_json_response(response)

            self.log_action("quality_check", {
                "output_type": output_type,
                "output_id": output_id,
                "quality": assessment.get('overall_quality'),
                "decision": assessment.get('decision')
            })

            return assessment

        except Exception as e:
            logger.error(f"Failed to parse quality assessment: {e}")
            return {"error": str(e), "decision": "needs_review"}

    def get_project_summary(self, project_id: int) -> Dict[str, Any]:
        """
        Get comprehensive project summary

        Args:
            project_id: Project ID

        Returns:
            Complete project summary
        """
        with get_db_context() as db:
            project = db.query(Project).filter(Project.id == project_id).first()
            if not project:
                raise ValueError(f"Project {project_id} not found")

            # Gather all project data
            from paperagent.database.models import Literature, Paper, Experiment

            literature = db.query(Literature).filter(Literature.project_id == project_id).all()
            papers = db.query(Paper).filter(Paper.project_id == project_id).all()
            experiments = db.query(Experiment).filter(Experiment.project_id == project_id).all()

            summary = {
                "project": {
                    "id": project.id,
                    "name": project.name,
                    "description": project.description,
                    "research_field": project.research_field,
                    "keywords": project.keywords,
                    "status": project.status.value,
                    "created_at": project.created_at.isoformat()
                },
                "literature": {
                    "total": len(literature),
                    "papers": [{"title": lit.title, "year": lit.year} for lit in literature[:10]]
                },
                "experiments": {
                    "total": len(experiments),
                    "list": [{"name": exp.name, "status": exp.status.value} for exp in experiments]
                },
                "papers": {
                    "total": len(papers),
                    "list": [{"title": p.title, "status": p.status.value, "word_count": p.word_count} for p in papers]
                }
            }

            return summary
