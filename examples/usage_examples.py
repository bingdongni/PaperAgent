"""
Example usage script for PaperAgent

This script demonstrates how to use PaperAgent for a complete research workflow.
"""

from paperagent.agents import BossAgent, LiteratureAgent, WritingAgent
from paperagent.database import get_db_context


def example_1_complete_workflow():
    """
    Example 1: Complete research workflow from start to finish
    """
    print("=" * 60)
    print("Example 1: Complete Research Workflow")
    print("=" * 60)

    boss = BossAgent()

    # Step 1: Create a project
    print("\n📁 Creating project...")
    project = boss.execute({
        'action': 'create_project',
        'name': 'Transformer Models in NLP',
        'description': 'Investigating recent advances in transformer architectures',
        'research_field': 'Computer Science',
        'keywords': ['transformers', 'NLP', 'attention mechanism']
    })
    print(f"✅ Project created with ID: {project['project_id']}")

    # Step 2: Execute complete workflow
    print("\n🚀 Executing research workflow...")
    print("This will:")
    print("  - Search for relevant literature")
    print("  - Analyze papers and identify research gaps")
    print("  - Create paper structure")
    print("  - Generate paper draft")

    result = boss.execute({
        'action': 'execute_workflow',
        'project_id': project['project_id']
    })

    print(f"\n✅ Workflow completed!")
    print(f"   Literature collected: {result['results'].get('literature_search', {}).get('total_papers', 0)}")
    print(f"   Paper created: {result['results'].get('paper_structure', {}).get('paper_id', 'N/A')}")

    # Step 3: Check progress
    print("\n📊 Checking progress...")
    progress = boss.execute({
        'action': 'monitor_progress',
        'project_id': project['project_id']
    })

    print(f"   Progress: {progress['progress_percentage']}%")
    print(f"   Completed tasks: {progress['completed_tasks']}/{progress['total_tasks']}")

    return project['project_id']


def example_2_literature_research():
    """
    Example 2: Literature research and analysis
    """
    print("\n" + "=" * 60)
    print("Example 2: Literature Research")
    print("=" * 60)

    # Create project first
    boss = BossAgent()
    project = boss.execute({
        'action': 'create_project',
        'name': 'Deep Learning for Computer Vision',
        'research_field': 'Computer Science',
        'keywords': ['deep learning', 'computer vision', 'CNN']
    })
    project_id = project['project_id']

    lit_agent = LiteratureAgent()

    # Search for papers
    print("\n🔍 Searching for papers...")
    search_result = lit_agent.execute({
        'action': 'search_literature',
        'query': 'convolutional neural networks image classification',
        'max_results': 20,
        'sources': ['arxiv'],
        'project_id': project_id
    })

    print(f"✅ Found {search_result['total_papers']} papers")
    print(f"   Saved {search_result['saved_papers']} new papers")

    # Analyze research gaps
    print("\n🔬 Identifying research gaps...")
    gaps = lit_agent.execute({
        'action': 'identify_gaps',
        'project_id': project_id,
        'research_field': 'Computer Science'
    })

    print(f"✅ Identified {len(gaps.get('research_gaps', []))} research gaps:")
    for i, gap in enumerate(gaps.get('research_gaps', [])[:3], 1):
        print(f"\n   Gap {i}:")
        print(f"   - Description: {gap.get('gap', 'N/A')[:100]}...")
        print(f"   - Importance: {gap.get('importance', 'N/A')[:80]}...")
        print(f"   - Feasibility: {gap.get('feasibility', 'N/A')}")

    return project_id


def example_3_paper_writing():
    """
    Example 3: Academic paper writing
    """
    print("\n" + "=" * 60)
    print("Example 3: Paper Writing")
    print("=" * 60)

    # Create project
    boss = BossAgent()
    project = boss.execute({
        'action': 'create_project',
        'name': 'Survey of Attention Mechanisms',
        'research_field': 'Computer Science',
        'keywords': ['attention', 'neural networks']
    })
    project_id = project['project_id']

    writer = WritingAgent()

    # Create paper structure
    print("\n📝 Creating paper structure...")
    structure = writer.execute({
        'action': 'create_structure',
        'title': 'A Comprehensive Survey of Attention Mechanisms in Deep Learning',
        'objective': 'Review and categorize attention mechanisms used in modern deep learning',
        'findings': [
            'Self-attention enables modeling long-range dependencies',
            'Multi-head attention improves representation learning',
            'Efficient attention variants reduce computational complexity'
        ],
        'project_id': project_id,
        'journal': 'IEEE Transactions on Pattern Analysis'
    })

    print(f"✅ Paper structure created (ID: {structure.get('paper_id')})")
    print(f"   Estimated word count: {structure.get('estimated_total_words', 'N/A')}")

    # Write introduction
    print("\n✍️  Writing introduction section...")
    intro = writer.execute({
        'action': 'write_section',
        'section': 'introduction',
        'context': 'Survey paper on attention mechanisms in deep learning',
        'key_points': [
            'Background on attention in neural networks',
            'Importance of attention for modern AI',
            'Overview of paper structure'
        ],
        'paper_id': structure['paper_id']
    })

    print(f"✅ Introduction written:")
    print(f"   Word count: {intro.get('word_count', 0)}")
    print(f"   Readability score: {intro.get('readability', {}).get('flesch_reading_ease', 'N/A'):.1f}")

    # Generate abstract
    print("\n📄 Generating abstract...")
    abstract = writer.execute({
        'action': 'write_abstract',
        'paper_id': structure['paper_id'],
        'word_limit': 250
    })

    print(f"✅ Abstract generated:")
    print(f"   Word count: {abstract.get('word_count', 0)}")
    print(f"   Within limit: {'Yes' if abstract.get('within_limit') else 'No'}")
    print(f"\n   Abstract preview:")
    print(f"   {abstract.get('abstract', '')[:200]}...")

    return structure['paper_id']


def example_4_project_summary():
    """
    Example 4: Get project summary
    """
    print("\n" + "=" * 60)
    print("Example 4: Project Summary")
    print("=" * 60)

    # Use project from example 1
    project_id = 1  # Assuming project 1 exists

    boss = BossAgent()

    print(f"\n📊 Getting summary for project {project_id}...")
    try:
        summary = boss.get_project_summary(project_id)

        print(f"\n✅ Project: {summary['project']['name']}")
        print(f"   Field: {summary['project']['research_field']}")
        print(f"   Status: {summary['project']['status']}")

        print(f"\n📚 Literature:")
        print(f"   Total papers: {summary['literature']['total']}")

        print(f"\n🔬 Experiments:")
        print(f"   Total experiments: {summary['experiments']['total']}")

        print(f"\n📝 Papers:")
        print(f"   Total papers: {summary['papers']['total']}")
        for paper in summary['papers']['list']:
            print(f"   - {paper['title']} ({paper['status']})")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("   (Project might not exist yet)")


def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("    PaperAgent Usage Examples")
    print("=" * 60)

    try:
        # Example 1: Complete workflow
        example_1_complete_workflow()

        # Example 2: Literature research
        example_2_literature_research()

        # Example 3: Paper writing
        example_3_paper_writing()

        # Example 4: Project summary
        example_4_project_summary()

        print("\n" + "=" * 60)
        print("✅ All examples completed successfully!")
        print("=" * 60)

        print("\n💡 Next steps:")
        print("   - Explore the Web UI at http://localhost:8501")
        print("   - Check the API docs at http://localhost:8000/docs")
        print("   - Read the full documentation in README.md")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("   Make sure the database is initialized and services are running")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
