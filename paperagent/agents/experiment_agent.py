"""
Experiment Agent - Handles experiment design, data analysis, and result management
"""

from typing import Dict, Any, List, Optional
import json
import pandas as pd
from loguru import logger

from paperagent.agents.base_agent import BaseAgent
from paperagent.database.models import AgentType, Experiment, TaskStatus
from paperagent.database.database import get_db_context
from paperagent.core.prompts import ExperimentPrompts
from paperagent.core.config import settings


class ExperimentAgent(BaseAgent):
    """
    Experiment Agent for research experiment management

    Capabilities:
    - Experiment design
    - Data analysis
    - Statistical testing
    - Result visualization recommendations
    - Experiment-paper synchronization
    """

    def __init__(self):
        super().__init__(AgentType.EXPERIMENT, "Experiment Agent")
        self.prompts = ExperimentPrompts()

    def execute(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute experiment-related tasks

        Args:
            task_input: Must contain 'action' key with value:
                - 'design_experiment': Design an experiment
                - 'analyze_data': Analyze experimental data
                - 'generate_figures': Generate figure recommendations
                - 'summarize_results': Summarize experiment results

        Returns:
            Task results
        """
        action = task_input.get('action')

        if action == 'design_experiment':
            return self.design_experiment(task_input)
        elif action == 'analyze_data':
            return self.analyze_data(task_input)
        elif action == 'generate_figures':
            return self.generate_figure_recommendations(task_input)
        elif action == 'summarize_results':
            return self.summarize_results(task_input)
        else:
            raise ValueError(f"Unknown action: {action}")

    def design_experiment(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Design a research experiment

        Args:
            task_input: Contains 'objective', 'field', 'resources', 'project_id'

        Returns:
            Experiment design details
        """
        self.validate_input(task_input, ['objective', 'field', 'project_id'])

        objective = task_input['objective']
        field = task_input['field']
        resources = task_input.get('resources', 'Standard lab equipment')
        project_id = task_input['project_id']

        logger.info(f"Designing experiment for: {objective}")

        # Generate experiment design prompt
        prompt = self.prompts.EXPERIMENT_DESIGN.format(
            objective=objective,
            field=field,
            resources=resources
        )

        # Get LLM response
        response = self.generate_text(prompt, max_tokens=2000, temperature=0.6)

        try:
            design = self.parse_json_response(response)

            # Save experiment to database
            with get_db_context() as db:
                experiment = Experiment(
                    project_id=project_id,
                    name=task_input.get('name', f"Experiment: {objective[:50]}"),
                    description=objective,
                    methodology=design.get('methodology', ''),
                    hypothesis=design.get('hypothesis', ''),
                    parameters=design,
                    status=TaskStatus.PENDING
                )
                db.add(experiment)
                db.commit()
                db.refresh(experiment)

                design['experiment_id'] = experiment.id

            self.log_action("design_experiment", {
                "objective": objective,
                "experiment_id": design.get('experiment_id')
            })

            return design

        except Exception as e:
            logger.error(f"Failed to parse experiment design: {e}")
            return {"error": str(e)}

    def analyze_data(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze experimental data

        Args:
            task_input: Contains 'experiment_id' or 'data', 'analysis_type'

        Returns:
            Data analysis results
        """
        experiment_id = task_input.get('experiment_id')
        data = task_input.get('data')
        analysis_type = task_input.get('analysis_type', 'descriptive')

        # Load experiment details if experiment_id provided
        experiment_details = ""
        if experiment_id:
            with get_db_context() as db:
                experiment = db.query(Experiment).filter(
                    Experiment.id == experiment_id
                ).first()

                if experiment:
                    experiment_details = f"""
                    Hypothesis: {experiment.hypothesis}
                    Methodology: {experiment.methodology}
                    """

        # Prepare data summary
        data_summary = ""
        if isinstance(data, dict):
            data_summary = json.dumps(data, indent=2)
        elif isinstance(data, str):
            # Assume it's a file path
            try:
                df = pd.read_csv(data)
                data_summary = f"""
                Dataset shape: {df.shape}
                Columns: {list(df.columns)}
                Summary statistics:
                {df.describe().to_string()}
                First few rows:
                {df.head().to_string()}
                """
            except Exception as e:
                logger.error(f"Failed to load data file: {e}")
                data_summary = str(data)
        else:
            data_summary = str(data)

        logger.info(f"Analyzing data for experiment: {experiment_id}")

        # Generate analysis prompt
        prompt = self.prompts.DATA_ANALYSIS.format(
            experiment_details=experiment_details,
            data_summary=data_summary
        )

        # Get LLM analysis
        response = self.generate_text(prompt, max_tokens=2000, temperature=0.4)

        try:
            analysis = self.parse_json_response(response)

            # Update experiment with results
            if experiment_id:
                with get_db_context() as db:
                    experiment = db.query(Experiment).filter(
                        Experiment.id == experiment_id
                    ).first()

                    if experiment:
                        experiment.results = analysis
                        experiment.statistical_tests = analysis.get('statistical_tests', [])
                        experiment.conclusions = json.dumps(analysis.get('key_findings', []))
                        experiment.status = TaskStatus.COMPLETED
                        db.commit()

            self.log_action("analyze_data", {"experiment_id": experiment_id})

            return analysis

        except Exception as e:
            logger.error(f"Failed to parse data analysis: {e}")
            return {"error": str(e)}

    def generate_figure_recommendations(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate recommendations for data visualization

        Args:
            task_input: Contains 'experiment_id' or 'analysis_results'

        Returns:
            Figure recommendations
        """
        experiment_id = task_input.get('experiment_id')
        analysis_results = task_input.get('analysis_results', {})

        # Load experiment if ID provided
        if experiment_id and not analysis_results:
            with get_db_context() as db:
                experiment = db.query(Experiment).filter(
                    Experiment.id == experiment_id
                ).first()

                if experiment and experiment.results:
                    analysis_results = experiment.results

        if not analysis_results:
            return {"visualizations": []}

        logger.info(f"Generating figure recommendations for experiment: {experiment_id}")

        # Extract visualization recommendations if already present
        visualizations = analysis_results.get('visualizations', [])

        # Add additional context-specific recommendations
        enhanced_viz = []
        for viz in visualizations:
            enhanced_viz.append({
                **viz,
                "recommended_tools": ["matplotlib", "seaborn", "plotly"],
                "code_example": self._generate_plot_code(viz)
            })

        return {
            "visualizations": enhanced_viz,
            "total_figures": len(enhanced_viz)
        }

    def _generate_plot_code(self, viz_spec: Dict[str, Any]) -> str:
        """Generate sample Python code for a visualization"""
        viz_type = viz_spec.get('type', 'scatter_plot')
        variables = viz_spec.get('variables', [])

        if viz_type == 'scatter_plot':
            return f"""
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='{variables[0] if len(variables) > 0 else "x"}',
                y='{variables[1] if len(variables) > 1 else "y"}')
plt.title('{viz_spec.get("purpose", "Scatter Plot")}')
plt.xlabel('{variables[0] if len(variables) > 0 else "X"}')
plt.ylabel('{variables[1] if len(variables) > 1 else "Y"}')
plt.show()
"""
        elif viz_type == 'bar_chart':
            return f"""
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
df['{variables[0] if len(variables) > 0 else "category"}'].value_counts().plot(kind='bar')
plt.title('{viz_spec.get("purpose", "Bar Chart")}')
plt.xlabel('{variables[0] if len(variables) > 0 else "Category"}')
plt.ylabel('Count')
plt.show()
"""
        else:
            return f"""
# Custom plot for {viz_type}
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
# Add your plotting code here
plt.title('{viz_spec.get("purpose", "Plot")}')
plt.show()
"""

    def summarize_results(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Summarize experiment results for paper writing

        Args:
            task_input: Contains 'experiment_id' or 'results_data'

        Returns:
            Formatted summary for paper
        """
        experiment_id = task_input.get('experiment_id')

        with get_db_context() as db:
            experiment = db.query(Experiment).filter(
                Experiment.id == experiment_id
            ).first()

            if not experiment:
                raise ValueError(f"Experiment {experiment_id} not found")

            # Create structured summary
            summary = {
                "experiment_name": experiment.name,
                "hypothesis": experiment.hypothesis,
                "methodology": experiment.methodology,
                "key_findings": experiment.results.get('key_findings', []) if experiment.results else [],
                "statistical_significance": self._extract_significance(experiment.results),
                "conclusions": experiment.conclusions,
                "limitations": experiment.limitations,
                "figures": experiment.figures or [],
                "tables": experiment.tables or []
            }

        logger.info(f"Summarized results for experiment: {experiment_id}")
        self.log_action("summarize_results", {"experiment_id": experiment_id})

        return summary

    def _extract_significance(self, results: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract statistical significance from results"""
        if not results or 'statistical_tests' not in results:
            return []

        significance = []
        for test in results['statistical_tests']:
            if test.get('p_value', 1.0) < 0.05:
                significance.append({
                    "test": test.get('test_name'),
                    "p_value": test.get('p_value'),
                    "interpretation": test.get('interpretation'),
                    "significant": True
                })

        return significance

    def update_experiment_status(self, experiment_id: int, status: TaskStatus) -> bool:
        """
        Update experiment status

        Args:
            experiment_id: Experiment ID
            status: New status

        Returns:
            Success boolean
        """
        try:
            with get_db_context() as db:
                experiment = db.query(Experiment).filter(
                    Experiment.id == experiment_id
                ).first()

                if experiment:
                    experiment.status = status
                    db.commit()
                    return True

            return False

        except Exception as e:
            logger.error(f"Failed to update experiment status: {e}")
            return False
