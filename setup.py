"""
Quick start script for PaperAgent
"""

import sys
import subprocess
import os
from pathlib import Path


def check_python_version():
    """Check if Python version is 3.10+"""
    if sys.version_info < (3, 10):
        print("❌ Python 3.10 or higher is required")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")


def check_env_file():
    """Check if .env file exists"""
    if not Path(".env").exists():
        print("⚠️  .env file not found. Copying from .env.example...")
        if Path(".env.example").exists():
            import shutil
            shutil.copy(".env.example", ".env")
            print("✅ .env file created. Please edit it with your configuration.")
        else:
            print("❌ .env.example not found")
            sys.exit(1)
    else:
        print("✅ .env file found")


def install_dependencies():
    """Install required packages"""
    print("\n📦 Installing dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        sys.exit(1)


def initialize_database():
    """Initialize database"""
    print("\n🗄️  Initializing database...")
    try:
        from paperagent.database import init_db
        init_db()
        print("✅ Database initialized")
    except Exception as e:
        print(f"❌ Failed to initialize database: {e}")
        print("Make sure PostgreSQL is running or use SQLite by setting:")
        print("DATABASE_URL=sqlite:///./paperagent.db")


def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    directories = [
        "data/papers",
        "data/experiments",
        "data/literature",
        "data/outputs",
        "logs"
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

    print("✅ Directories created")


def main():
    """Main setup function"""
    print("=" * 50)
    print("   PaperAgent Quick Start Setup")
    print("=" * 50)

    # Check Python version
    check_python_version()

    # Check environment file
    check_env_file()

    # Install dependencies
    install_dependencies()

    # Create directories
    create_directories()

    # Initialize database
    initialize_database()

    # Success message
    print("\n" + "=" * 50)
    print("✅ Setup completed successfully!")
    print("=" * 50)

    print("\n🚀 To start PaperAgent:")
    print("\n1. Start API server:")
    print("   uvicorn paperagent.api.main:app --host 0.0.0.0 --port 8000")

    print("\n2. Start Web UI (in another terminal):")
    print("   streamlit run paperagent/web/app.py")

    print("\n3. Access the application:")
    print("   Web UI: http://localhost:8501")
    print("   API Docs: http://localhost:8000/docs")

    print("\n📚 Documentation: README.md")
    print("🐛 Issues: https://github.com/yourusername/paperagent/issues")


if __name__ == "__main__":
    main()
