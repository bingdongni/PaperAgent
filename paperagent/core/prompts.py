"""
Prompt templates for PaperAgent agents

Based on KtR (Knowledge-to-Role) framework for systematic task decomposition
"""

from typing import Dict, Any, List, Optional


class PromptTemplate:
    """Base class for prompt templates"""

    @staticmethod
    def format(template: str, **kwargs) -> str:
        """Format template with variables"""
        return template.format(**kwargs)


class LiteraturePrompts:
    """Prompts for Literature Agent"""

    TOPIC_RECOMMENDATION = """You are an expert research advisor with deep knowledge across multiple academic fields.

Task: Analyze the research field "{field}" and keywords "{keywords}", then recommend 3-5 potential research topics.

For each topic, provide:
1. Topic title
2. Research motivation and significance
3. Current research gaps
4. Feasibility analysis
5. 3-5 key references that support this topic

Output format:
```json
{{
    "topics": [
        {{
            "title": "Topic title",
            "motivation": "Why this topic matters",
            "research_gaps": ["Gap 1", "Gap 2"],
            "feasibility": "Assessment of feasibility",
            "key_references": [
                {{
                    "title": "Paper title",
                    "authors": ["Author names"],
                    "year": 2024,
                    "key_contribution": "Main contribution"
                }}
            ]
        }}
    ]
}}
```

Be specific, evidence-based, and focus on topics with clear research value and innovation potential."""

    LITERATURE_SUMMARY = """You are an expert academic literature reviewer.

Task: Analyze and summarize the following research paper:

Title: {title}
Authors: {authors}
Abstract: {abstract}

Provide a comprehensive analysis including:
1. Research objective and problem statement
2. Methodology and approach
3. Main findings and contributions
4. Limitations and future work
5. Relevance score (0-10) for research field: {research_field}

Output format:
```json
{{
    "objective": "Clear statement of research goal",
    "methodology": "Brief description of methods used",
    "main_findings": ["Finding 1", "Finding 2", "Finding 3"],
    "contributions": ["Contribution 1", "Contribution 2"],
    "limitations": ["Limitation 1", "Limitation 2"],
    "future_work": ["Direction 1", "Direction 2"],
    "relevance_score": 8,
    "relevance_justification": "Why this score",
    "key_insights": ["Insight 1", "Insight 2"]
}}
```

Be thorough, objective, and focus on extracting actionable insights."""

    LITERATURE_CLUSTERING = """You are an expert at analyzing and organizing research literature.

Task: Given the following set of papers, cluster them by research themes and identify relationships.

Papers:
{papers_list}

Analysis required:
1. Identify 3-5 main research themes
2. Group papers by theme
3. Identify citation relationships and influences
4. Highlight gaps in current research
5. Suggest potential research directions

Output format:
```json
{{
    "themes": [
        {{
            "theme_name": "Theme title",
            "description": "Theme description",
            "papers": [paper_ids],
            "key_contributions": ["Contribution 1", "Contribution 2"],
            "research_direction": "Potential future work"
        }}
    ],
    "relationships": [
        {{
            "paper1_id": "id1",
            "paper2_id": "id2",
            "relationship": "builds_on|contradicts|extends",
            "description": "How they relate"
        }}
    ],
    "research_gaps": ["Gap 1", "Gap 2", "Gap 3"]
}}
```"""

    RESEARCH_GAP_ANALYSIS = """You are an expert at identifying research opportunities and gaps in academic literature.

Task: Based on the following literature review, identify specific research gaps and opportunities.

Literature Summary:
{literature_summary}

Research Field: {research_field}

Provide:
1. Specific research gaps with evidence
2. Why each gap is important
3. Feasibility of addressing each gap
4. Potential research questions
5. Required resources and expertise

Output format:
```json
{{
    "research_gaps": [
        {{
            "gap": "Description of gap",
            "evidence": "Citations and reasoning",
            "importance": "Why this matters",
            "feasibility": "High|Medium|Low",
            "research_questions": ["Question 1", "Question 2"],
            "required_resources": ["Resource 1", "Resource 2"]
        }}
    ],
    "prioritized_opportunities": [
        {{
            "opportunity": "Research opportunity",
            "priority": "High|Medium|Low",
            "justification": "Why prioritize this"
        }}
    ]
}}
```"""


class ExperimentPrompts:
    """Prompts for Experiment Agent"""

    EXPERIMENT_DESIGN = """You are an expert experimental researcher skilled in designing rigorous scientific studies.

Task: Design a comprehensive experiment plan for the following research objective:

Research Objective: {objective}
Research Field: {field}
Available Resources: {resources}

Provide a detailed experimental design including:
1. Research hypothesis
2. Experimental methodology
3. Variables (independent, dependent, control)
4. Sample size and selection criteria
5. Data collection procedures
6. Statistical analysis methods
7. Expected outcomes
8. Potential limitations

Output format:
```json
{{
    "hypothesis": "Clear testable hypothesis",
    "methodology": "Detailed methodology description",
    "variables": {{
        "independent": ["Var 1", "Var 2"],
        "dependent": ["Var 1", "Var 2"],
        "control": ["Var 1", "Var 2"]
    }},
    "sample_design": {{
        "sample_size": 100,
        "selection_criteria": ["Criterion 1", "Criterion 2"],
        "sampling_method": "Random|Stratified|Convenience"
    }},
    "data_collection": {{
        "methods": ["Method 1", "Method 2"],
        "instruments": ["Instrument 1", "Instrument 2"],
        "procedure": "Step-by-step procedure"
    }},
    "statistical_analysis": {{
        "tests": ["Test 1", "Test 2"],
        "software": "R|Python|SPSS",
        "significance_level": 0.05
    }},
    "expected_outcomes": ["Outcome 1", "Outcome 2"],
    "limitations": ["Limitation 1", "Limitation 2"]
}}
```"""

    DATA_ANALYSIS = """You are an expert data analyst with strong statistical knowledge.

Task: Analyze the experimental data and provide insights.

Experiment Details:
{experiment_details}

Data Summary:
{data_summary}

Provide:
1. Descriptive statistics
2. Statistical test results
3. Data visualization recommendations
4. Key findings and interpretations
5. Limitations and considerations

Output format:
```json
{{
    "descriptive_stats": {{
        "mean": 0.0,
        "median": 0.0,
        "std_dev": 0.0,
        "range": [0.0, 10.0]
    }},
    "statistical_tests": [
        {{
            "test_name": "t-test|ANOVA|regression",
            "p_value": 0.05,
            "effect_size": 0.5,
            "interpretation": "Significant|Not significant"
        }}
    ],
    "visualizations": [
        {{
            "type": "bar_chart|scatter_plot|histogram",
            "variables": ["var1", "var2"],
            "purpose": "Show relationship between X and Y"
        }}
    ],
    "key_findings": ["Finding 1", "Finding 2"],
    "interpretations": ["Interpretation 1", "Interpretation 2"],
    "limitations": ["Limitation 1", "Limitation 2"]
}}
```"""


class WritingPrompts:
    """Prompts for Writing Agent"""

    PAPER_STRUCTURE = """You are an expert academic writer specialized in {field}.

Task: Generate a structured outline for a research paper on:

Title: {title}
Research Objective: {objective}
Key Findings: {findings}
Target Journal: {journal}

Create a detailed outline following standard academic structure:
1. Title and Abstract
2. Introduction
3. Literature Review
4. Methodology
5. Results
6. Discussion
7. Conclusion
8. References

For each section, provide:
- Main points to cover
- Key arguments
- Approximate word count
- Critical elements

Output format:
```json
{{
    "title": "Refined title",
    "sections": [
        {{
            "section": "Introduction",
            "subsections": ["Background", "Problem Statement", "Research Questions"],
            "main_points": ["Point 1", "Point 2"],
            "word_count": 800,
            "critical_elements": ["Element 1", "Element 2"]
        }}
    ],
    "estimated_total_words": 6000
}}
```"""

    SECTION_WRITING = """You are an expert academic writer with publication experience in top-tier journals.

Task: Write the {section} section of a research paper with the following details:

Paper Context:
{context}

Section Requirements:
{requirements}

Key Points to Cover:
{key_points}

Guidelines:
1. Use formal academic language
2. Support claims with evidence
3. Maintain logical flow
4. Follow {citation_style} citation style
5. Write for audience: {target_audience}

Write a comprehensive, well-structured section that meets academic standards."""

    ACADEMIC_POLISH = """You are an expert academic editor specializing in English language polishing for scientific publications.

Task: Polish and improve the following academic text while maintaining its technical accuracy:

Original Text:
{text}

Context: {context}

Polish the text focusing on:
1. Grammar and syntax
2. Academic tone and style
3. Clarity and conciseness
4. Logical flow
5. Technical precision

Provide:
- Polished text
- List of major changes made
- Suggestions for further improvement

Output format:
```json
{{
    "polished_text": "Improved version",
    "changes": [
        {{
            "type": "grammar|style|clarity|flow",
            "original": "Original phrase",
            "revised": "Revised phrase",
            "reason": "Why this change improves the text"
        }}
    ],
    "suggestions": ["Suggestion 1", "Suggestion 2"]
}}
```"""

    ABSTRACT_GENERATION = """You are an expert at writing compelling research paper abstracts.

Task: Write a concise abstract (max {word_limit} words) for the following research paper:

Title: {title}
Introduction: {introduction}
Methodology: {methodology}
Results: {results}
Conclusion: {conclusion}

The abstract should include:
1. Background/Context (1-2 sentences)
2. Research objective (1 sentence)
3. Methodology (1-2 sentences)
4. Main findings (2-3 sentences)
5. Significance/Implications (1 sentence)

Write a clear, compelling abstract that makes readers want to read the full paper."""


class FormattingPrompts:
    """Prompts for Formatting Agent"""

    CITATION_FORMATTING = """You are an expert at academic citation formatting.

Task: Format the following references according to {style} style:

References:
{references}

Provide:
1. Formatted reference list
2. In-text citation format examples
3. Special cases handling

Output format:
```json
{{
    "formatted_references": [
        {{
            "id": "ref1",
            "formatted": "Full formatted reference",
            "in_text": "(Author, Year)"
        }}
    ],
    "style_notes": ["Note 1", "Note 2"]
}}
```"""

    LATEX_GENERATION = """You are a LaTeX expert specialized in academic paper formatting.

Task: Generate LaTeX code for a research paper with the following specifications:

Journal: {journal}
Document Class: {document_class}
Content Sections: {sections}

Generate complete LaTeX code including:
1. Document class and packages
2. Title, authors, affiliations
3. Abstract
4. Main sections
5. References
6. Figures and tables

Ensure the code compiles without errors and follows journal guidelines."""


class BossPrompts:
    """Prompts for Boss Agent (Orchestration)"""

    TASK_DECOMPOSITION = """You are an expert project manager specialized in research workflows.

Task: Decompose the following research project into specific, actionable tasks:

Project Goal: {goal}
Research Field: {field}
Timeline: {timeline}
Resources: {resources}

Decompose into:
1. Literature review tasks
2. Experiment design tasks
3. Data collection tasks
4. Analysis tasks
5. Writing tasks
6. Formatting/submission tasks

For each task provide:
- Task name and description
- Agent responsible
- Dependencies
- Estimated time
- Priority

Output format:
```json
{{
    "tasks": [
        {{
            "task_id": "T1",
            "name": "Task name",
            "description": "Detailed description",
            "agent": "literature|experiment|writing|formatting",
            "dependencies": ["T0"],
            "estimated_hours": 4,
            "priority": "high|medium|low"
        }}
    ],
    "workflow": {{
        "phases": ["Phase 1: Literature", "Phase 2: Experiments"],
        "milestones": ["Milestone 1", "Milestone 2"]
    }}
}}
```"""

    QUALITY_CHECK = """You are an expert quality assurance reviewer for academic research.

Task: Review the following research output and provide quality assessment:

Output Type: {output_type}
Content: {content}
Requirements: {requirements}

Check for:
1. Completeness
2. Accuracy
3. Academic rigor
4. Formatting compliance
5. Citation correctness

Provide detailed feedback and approve/reject decision.

Output format:
```json
{{
    "overall_quality": "excellent|good|needs_improvement|poor",
    "completeness_score": 9.0,
    "accuracy_score": 8.5,
    "rigor_score": 9.0,
    "formatting_score": 8.0,
    "issues": [
        {{
            "severity": "critical|major|minor",
            "category": "completeness|accuracy|formatting",
            "description": "Issue description",
            "suggestion": "How to fix"
        }}
    ],
    "decision": "approve|revise|reject",
    "overall_feedback": "Summary feedback"
}}
```"""


# Export all prompt classes
__all__ = [
    "PromptTemplate",
    "LiteraturePrompts",
    "ExperimentPrompts",
    "WritingPrompts",
    "FormattingPrompts",
    "BossPrompts",
]
