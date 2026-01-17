"""
Command-line interface for PaperAgent

This module provides a CLI for managing and interacting with PaperAgent,
including database initialization, server management, and utility commands.
"""

import click
import sys
from pathlib import Path
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@click.group()
@click.version_option(version='1.0.0', prog_name='PaperAgent')
def cli():
    """
    PaperAgent - Academic Multi-Agent Collaboration Framework

    A comprehensive tool for academic research workflow automation,
    from literature review to paper writing and submission.
    """
    pass


@cli.command()
@click.option('--host', default='0.0.0.0', help='Host to bind to')
@click.option('--port', default=8000, type=int, help='Port to bind to')
@click.option('--reload', is_flag=True, help='Enable auto-reload on code changes')
def serve(host: str, port: int, reload: bool):
    """
    Start the FastAPI server

    Example:
        paperagent serve --host 0.0.0.0 --port 8000 --reload
    """
    import uvicorn

    click.echo(f"🚀 Starting PaperAgent API server on {host}:{port}")

    if reload:
        click.echo("⚡ Auto-reload enabled")

    uvicorn.run(
        "paperagent.api.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


@cli.command()
@click.option('--host', default='0.0.0.0', help='Host to bind to')
@click.option('--port', default=8501, type=int, help='Port to bind to')
def web(host: str, port: int):
    """
    Start the Streamlit web interface

    Example:
        paperagent web --host 0.0.0.0 --port 8501
    """
    import subprocess

    click.echo(f"🌐 Starting PaperAgent web interface on {host}:{port}")

    subprocess.run([
        'streamlit', 'run',
        'paperagent/web/app.py',
        '--server.address', host,
        '--server.port', str(port)
    ])


@cli.command()
@click.option('--concurrency', default=4, type=int, help='Number of worker processes')
@click.option('--loglevel', default='info', help='Log level')
def worker(concurrency: int, loglevel: str):
    """
    Start a Celery worker for background tasks

    Example:
        paperagent worker --concurrency 4 --loglevel info
    """
    from paperagent.core.tasks import celery_app

    click.echo(f"👷 Starting Celery worker with {concurrency} concurrent workers")

    celery_app.worker_main([
        'worker',
        '--loglevel', loglevel,
        '--concurrency', str(concurrency)
    ])


@cli.command()
def init():
    """
    Initialize the database schema

    Creates all necessary tables and initial data.
    """
    from paperagent.database.database import init_db

    click.echo("🔧 Initializing PaperAgent database...")

    try:
        init_db()
        click.echo("✅ Database initialized successfully!")
    except Exception as e:
        click.echo(f"❌ Database initialization failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--yes', is_flag=True, help='Skip confirmation')
def reset(yes: bool):
    """
    Reset the database (WARNING: This will delete all data!)

    Example:
        paperagent reset --yes
    """
    if not yes:
        click.confirm(
            '⚠️  This will delete ALL data. Are you sure?',
            abort=True
        )

    from paperagent.database.database import reset_db

    click.echo("🗑️  Resetting database...")

    try:
        reset_db()
        click.echo("✅ Database reset successfully!")
    except Exception as e:
        click.echo(f"❌ Database reset failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('project_name')
@click.option('--field', default='Computer Science', help='Research field')
@click.option('--keywords', multiple=True, help='Project keywords')
def create(project_name: str, field: str, keywords: tuple):
    """
    Create a new research project

    Example:
        paperagent create "My Research Project" --field "AI" --keywords ml --keywords nlp
    """
    from paperagent.database.database import get_db
    from paperagent.database.models import Project

    click.echo(f"📝 Creating project: {project_name}")

    db = next(get_db())

    try:
        project = Project(
            name=project_name,
            research_field=field,
            keywords=list(keywords) if keywords else [],
            status='active'
        )

        db.add(project)
        db.commit()
        db.refresh(project)

        click.echo(f"✅ Project created successfully!")
        click.echo(f"   ID: {project.id}")
        click.echo(f"   Name: {project.name}")
        click.echo(f"   Field: {project.research_field}")

    except Exception as e:
        click.echo(f"❌ Project creation failed: {e}", err=True)
        sys.exit(1)
    finally:
        db.close()


@cli.command()
def list():
    """
    List all research projects
    """
    from paperagent.database.database import get_db
    from paperagent.database.models import Project

    db = next(get_db())

    try:
        projects = db.query(Project).all()

        if not projects:
            click.echo("No projects found.")
            return

        click.echo("\n📚 Research Projects:\n")

        for project in projects:
            click.echo(f"  [{project.id}] {project.name}")
            click.echo(f"      Field: {project.research_field}")
            click.echo(f"      Status: {project.status}")
            click.echo(f"      Created: {project.created_at}\n")

    except Exception as e:
        click.echo(f"❌ Failed to list projects: {e}", err=True)
        sys.exit(1)
    finally:
        db.close()


@cli.command()
@click.option('--check-db', is_flag=True, help='Check database connection')
@click.option('--check-redis', is_flag=True, help='Check Redis connection')
@click.option('--check-llm', is_flag=True, help='Check LLM connectivity')
def health(check_db: bool, check_redis: bool, check_llm: bool):
    """
    Check system health

    Example:
        paperagent health --check-db --check-redis --check-llm
    """
    import os

    click.echo("🏥 PaperAgent Health Check\n")

    all_healthy = True

    # Check environment variables
    click.echo("📋 Environment Variables:")
    required_vars = ['DATABASE_URL', 'REDIS_URL', 'DEFAULT_LLM_PROVIDER']

    for var in required_vars:
        value = os.getenv(var)
        if value:
            click.echo(f"  ✅ {var}: {value[:50]}...")
        else:
            click.echo(f"  ❌ {var}: Not set")
            all_healthy = False

    # Check database
    if check_db:
        click.echo("\n🗄️  Database:")
        try:
            from paperagent.database.database import engine
            with engine.connect() as conn:
                conn.execute("SELECT 1")
            click.echo("  ✅ Database connection successful")
        except Exception as e:
            click.echo(f"  ❌ Database connection failed: {e}")
            all_healthy = False

    # Check Redis
    if check_redis:
        click.echo("\n🔴 Redis:")
        try:
            import redis
            redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
            r = redis.from_url(redis_url)
            r.ping()
            click.echo("  ✅ Redis connection successful")
        except Exception as e:
            click.echo(f"  ❌ Redis connection failed: {e}")
            all_healthy = False

    # Check LLM
    if check_llm:
        click.echo("\n🤖 LLM Provider:")
        try:
            from paperagent.core.llm_manager import LLMManager
            llm = LLMManager()
            click.echo(f"  ✅ LLM provider: {llm.provider}")
        except Exception as e:
            click.echo(f"  ❌ LLM initialization failed: {e}")
            all_healthy = False

    click.echo()

    if all_healthy:
        click.echo("✅ All systems operational!")
    else:
        click.echo("⚠️  Some systems have issues. Please check the errors above.")
        sys.exit(1)


@cli.command()
@click.argument('config_key', required=False)
def config(config_key: str):
    """
    View configuration settings

    Example:
        paperagent config                    # View all settings
        paperagent config DEFAULT_LLM_PROVIDER  # View specific setting
    """
    import os
    from paperagent.core.config import settings

    if config_key:
        value = getattr(settings, config_key.lower(), None)
        if value:
            click.echo(f"{config_key}: {value}")
        else:
            click.echo(f"❌ Configuration key '{config_key}' not found")
    else:
        click.echo("⚙️  PaperAgent Configuration:\n")

        config_dict = settings.model_dump()
        for key, value in config_dict.items():
            # Mask sensitive values
            if 'key' in key.lower() or 'password' in key.lower():
                display_value = '***' if value else 'Not set'
            else:
                display_value = value

            click.echo(f"  {key}: {display_value}")


@cli.command()
def version():
    """
    Show version information
    """
    from paperagent import __version__

    click.echo(f"""
╔═══════════════════════════════════════════╗
║                                           ║
║            PaperAgent v{__version__}              ║
║                                           ║
║  Academic Multi-Agent Collaboration       ║
║           Framework                       ║
║                                           ║
╚═══════════════════════════════════════════╝

Python version: {sys.version.split()[0]}
Project: https://github.com/yourusername/paperagent
Documentation: See README.md
    """)


if __name__ == '__main__':
    cli()
