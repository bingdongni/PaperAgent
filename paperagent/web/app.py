"""
PaperAgent Streamlit Web Interface
"""

import streamlit as st
import requests
import json
from typing import Dict, Any, Optional

# Configuration
API_BASE_URL = "http://localhost:8000/api"

# Page config
st.set_page_config(
    page_title="PaperAgent - Academic Research Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        padding: 1rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.25rem;
        color: #155724;
    }
    .error-box {
        padding: 1rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 0.25rem;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)


def api_request(endpoint: str, method: str = "GET", data: Optional[Dict] = None) -> Dict[str, Any]:
    """Make API request"""
    url = f"{API_BASE_URL}/{endpoint}"

    try:
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=data)
        elif method == "PUT":
            response = requests.put(url, json=data)
        elif method == "DELETE":
            response = requests.delete(url)

        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return {"error": str(e)}


def main():
    """Main application"""

    # Header
    st.markdown('<div class="main-header">📚 PaperAgent</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Academic Multi-Agent Collaboration Framework</div>',
        unsafe_allow_html=True
    )

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        [
            "🏠 Home",
            "📁 Projects",
            "📖 Literature Research",
            "🔬 Experiments",
            "✍️ Paper Writing",
            "📊 Dashboard"
        ]
    )

    # Page routing
    if page == "🏠 Home":
        show_home_page()
    elif page == "📁 Projects":
        show_projects_page()
    elif page == "📖 Literature Research":
        show_literature_page()
    elif page == "🔬 Experiments":
        show_experiments_page()
    elif page == "✍️ Paper Writing":
        show_writing_page()
    elif page == "📊 Dashboard":
        show_dashboard_page()


def show_home_page():
    """Home page"""
    st.header("Welcome to PaperAgent")

    st.write("""
    PaperAgent is an AI-powered academic research assistant that helps you through the entire research lifecycle:

    - **Literature Research**: Automated paper search, analysis, and gap identification
    - **Experiment Design**: AI-guided experimental design and data analysis
    - **Paper Writing**: Intelligent academic writing assistance
    - **Quality Control**: Multi-layer quality checks and formatting
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("📚 Research Areas", "50+")
    with col2:
        st.metric("🤖 AI Agents", "6")
    with col3:
        st.metric("📝 Papers Generated", "1000+")

    st.subheader("Quick Start")
    st.write("1. Create a new project in the **Projects** tab")
    st.write("2. Search for literature in **Literature Research**")
    st.write("3. Design experiments in **Experiments**")
    st.write("4. Generate your paper in **Paper Writing**")


def show_projects_page():
    """Projects management page"""
    st.header("📁 Project Management")

    tab1, tab2 = st.tabs(["Create Project", "View Projects"])

    with tab1:
        st.subheader("Create New Project")

        with st.form("create_project_form"):
            name = st.text_input("Project Name", placeholder="e.g., Deep Learning for Medical Imaging")
            description = st.text_area("Description", placeholder="Describe your research project...")
            research_field = st.selectbox(
                "Research Field",
                ["Computer Science", "Biology", "Physics", "Chemistry", "Medicine", "Engineering", "Mathematics"]
            )
            keywords = st.text_input("Keywords (comma-separated)", placeholder="e.g., deep learning, medical imaging, CNN")

            submitted = st.form_submit_button("Create Project")

            if submitted:
                if name and research_field:
                    data = {
                        "name": name,
                        "description": description,
                        "research_field": research_field,
                        "keywords": [k.strip() for k in keywords.split(",") if k.strip()]
                    }

                    result = api_request("projects/", "POST", data)

                    if "error" not in result:
                        st.success(f"✅ Project '{name}' created successfully!")
                        st.json(result)
                    else:
                        st.error(f"Failed to create project: {result['error']}")
                else:
                    st.warning("Please fill in all required fields")

    with tab2:
        st.subheader("Existing Projects")

        projects = api_request("projects/")

        if isinstance(projects, list) and projects:
            for project in projects:
                with st.expander(f"📁 {project['name']} - {project['research_field']}"):
                    st.write(f"**Description:** {project.get('description', 'N/A')}")
                    st.write(f"**Status:** {project['status']}")
                    st.write(f"**Keywords:** {', '.join(project.get('keywords', []))}")
                    st.write(f"**Created:** {project['created_at']}")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button(f"View Details", key=f"view_{project['id']}"):
                            show_project_details(project['id'])
                    with col2:
                        if st.button(f"Execute Workflow", key=f"exec_{project['id']}"):
                            execute_workflow(project['id'])
                    with col3:
                        if st.button(f"Progress", key=f"prog_{project['id']}"):
                            show_project_progress(project['id'])
        else:
            st.info("No projects found. Create your first project!")


def show_project_details(project_id: int):
    """Show project details"""
    summary = api_request(f"projects/{project_id}/summary")

    if "error" not in summary:
        st.json(summary)
    else:
        st.error("Failed to load project summary")


def show_project_progress(project_id: int):
    """Show project progress"""
    progress = api_request(f"projects/{project_id}/progress")

    if "error" not in progress:
        st.progress(progress['progress_percentage'] / 100)
        st.write(f"**Progress:** {progress['progress_percentage']}%")
        st.write(f"**Completed Tasks:** {progress['completed_tasks']}/{progress['total_tasks']}")
        st.write(f"**Literature Collected:** {progress['literature_collected']}")
    else:
        st.error("Failed to load progress")


def execute_workflow(project_id: int):
    """Execute workflow for project"""
    with st.spinner("Executing workflow..."):
        result = api_request(f"projects/{project_id}/execute-workflow", "POST")

        if result.get("success"):
            st.success("Workflow executed successfully!")
            st.json(result['data'])
        else:
            st.error(f"Workflow execution failed: {result.get('error', 'Unknown error')}")


def show_literature_page():
    """Literature research page"""
    st.header("📖 Literature Research")

    # Project selection
    projects = api_request("projects/")
    if isinstance(projects, list) and projects:
        project_options = {p['name']: p['id'] for p in projects}
        selected_project = st.selectbox("Select Project", list(project_options.keys()))
        project_id = project_options[selected_project]

        tab1, tab2, tab3 = st.tabs(["Search Literature", "Analyze Papers", "Research Gaps"])

        with tab1:
            st.subheader("Search for Literature")

            query = st.text_input("Search Query", placeholder="e.g., deep learning medical imaging")
            max_results = st.slider("Max Results", 10, 100, 25)
            sources = st.multiselect("Sources", ["arxiv", "google_scholar"], default=["arxiv"])

            if st.button("Search"):
                if query:
                    with st.spinner("Searching..."):
                        data = {
                            "action": "search_literature",
                            "query": query,
                            "max_results": max_results,
                            "sources": sources,
                            "project_id": project_id
                        }

                        result = api_request("literature/search", "POST", data)

                        if result.get("success"):
                            st.success(f"Found {result['data']['total_papers']} papers!")
                            st.json(result['data'])
                        else:
                            st.error("Search failed")
                else:
                    st.warning("Please enter a search query")

        with tab2:
            st.subheader("View Literature")

            literature = api_request(f"literature/?project_id={project_id}")

            if isinstance(literature, list) and literature:
                for lit in literature:
                    with st.expander(f"📄 {lit['title']}"):
                        st.write(f"**Authors:** {', '.join(lit.get('authors', []))}")
                        st.write(f"**Year:** {lit.get('year', 'N/A')}")
                        st.write(f"**Abstract:** {lit.get('abstract', 'N/A')[:300]}...")
                        st.write(f"**Citations:** {lit.get('citation_count', 0)}")
            else:
                st.info("No literature found. Try searching first!")

        with tab3:
            st.subheader("Identify Research Gaps")

            if st.button("Analyze Research Gaps"):
                with st.spinner("Analyzing..."):
                    data = {
                        "action": "identify_gaps",
                        "project_id": project_id,
                        "research_field": "Computer Science"  # Get from project
                    }

                    result = api_request("literature/search", "POST", data)

                    if result.get("success"):
                        gaps = result['data'].get('research_gaps', [])
                        if gaps:
                            for i, gap in enumerate(gaps, 1):
                                st.write(f"**Gap {i}:** {gap.get('gap', '')}")
                                st.write(f"*Importance:* {gap.get('importance', '')}")
                                st.write("---")
                        else:
                            st.info("No research gaps identified yet")
    else:
        st.info("Please create a project first!")


def show_experiments_page():
    """Experiments page"""
    st.header("🔬 Experiment Management")

    projects = api_request("projects/")
    if isinstance(projects, list) and projects:
        project_options = {p['name']: p['id'] for p in projects}
        selected_project = st.selectbox("Select Project", list(project_options.keys()))
        project_id = project_options[selected_project]

        tab1, tab2 = st.tabs(["Design Experiment", "View Experiments"])

        with tab1:
            st.subheader("Design New Experiment")

            with st.form("design_experiment"):
                name = st.text_input("Experiment Name")
                objective = st.text_area("Research Objective")
                field = st.selectbox("Field", ["Computer Science", "Biology", "Physics"])
                resources = st.text_input("Available Resources")

                submitted = st.form_submit_button("Design Experiment")

                if submitted and name and objective:
                    data = {
                        "action": "design_experiment",
                        "project_id": project_id,
                        "name": name,
                        "objective": objective,
                        "field": field,
                        "resources": resources
                    }

                    result = api_request("experiments/design", "POST", data)

                    if result.get("success"):
                        st.success("Experiment designed successfully!")
                        st.json(result['data'])

        with tab2:
            st.subheader("Existing Experiments")

            experiments = api_request(f"experiments/?project_id={project_id}")

            if isinstance(experiments, list) and experiments:
                for exp in experiments:
                    with st.expander(f"🔬 {exp['name']}"):
                        st.write(f"**Hypothesis:** {exp.get('hypothesis', 'N/A')}")
                        st.write(f"**Methodology:** {exp.get('methodology', 'N/A')[:200]}...")
                        st.write(f"**Status:** {exp['status']}")
            else:
                st.info("No experiments yet. Design your first experiment!")
    else:
        st.info("Please create a project first!")


def show_writing_page():
    """Paper writing page"""
    st.header("✍️ Paper Writing")

    projects = api_request("projects/")
    if isinstance(projects, list) and projects:
        project_options = {p['name']: p['id'] for p in projects}
        selected_project = st.selectbox("Select Project", list(project_options.keys()))
        project_id = project_options[selected_project]

        tab1, tab2 = st.tabs(["Create Paper", "View Papers"])

        with tab1:
            st.subheader("Generate Paper")

            with st.form("create_paper"):
                title = st.text_input("Paper Title")
                objective = st.text_area("Research Objective")
                journal = st.selectbox("Target Journal", ["IEEE", "ACM", "Springer", "Elsevier"])

                submitted = st.form_submit_button("Generate Paper Structure")

                if submitted and title:
                    data = {
                        "action": "create_structure",
                        "project_id": project_id,
                        "title": title,
                        "objective": objective,
                        "journal": journal
                    }

                    result = api_request("papers/create", "POST", data)

                    if result.get("success"):
                        st.success("Paper structure created!")
                        st.json(result['data'])

        with tab2:
            st.subheader("Your Papers")

            papers = api_request(f"papers/?project_id={project_id}")

            if isinstance(papers, list) and papers:
                for paper in papers:
                    with st.expander(f"📝 {paper['title']}"):
                        st.write(f"**Abstract:** {paper.get('abstract', 'Not generated')[:200]}...")
                        st.write(f"**Word Count:** {paper.get('word_count', 0)}")
                        st.write(f"**Status:** {paper['status']}")
                        st.write(f"**Version:** {paper['version']}")
            else:
                st.info("No papers yet. Create your first paper!")
    else:
        st.info("Please create a project first!")


def show_dashboard_page():
    """Dashboard page"""
    st.header("📊 Dashboard")

    projects = api_request("projects/")

    if isinstance(projects, list):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Projects", len(projects))
        with col2:
            completed = sum(1 for p in projects if p['status'] == 'completed')
            st.metric("Completed", completed)
        with col3:
            in_progress = sum(1 for p in projects if p['status'] == 'in_progress')
            st.metric("In Progress", in_progress)
        with col4:
            st.metric("Success Rate", f"{completed/len(projects)*100 if projects else 0:.0f}%")

        st.subheader("Recent Projects")
        for project in projects[:5]:
            st.write(f"• {project['name']} - {project['status']}")


if __name__ == "__main__":
    main()
