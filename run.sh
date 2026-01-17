#!/bin/bash

echo "========================================"
echo "    PaperAgent Quick Start"
echo "========================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo ""
echo "Installing/Updating dependencies..."
pip install -r requirements.txt

echo ""
echo "Initializing database..."
python setup.py

echo ""
echo "========================================"
echo "  Setup Complete!"
echo "========================================"
echo ""
echo "To start PaperAgent:"
echo ""
echo "1. Start API (in this terminal):"
echo "   uvicorn paperagent.api.main:app --host 0.0.0.0 --port 8000 --reload"
echo ""
echo "2. Start Web UI (in another terminal):"
echo "   streamlit run paperagent/web/app.py"
echo ""
echo "Access:"
echo "- Web UI: http://localhost:8501"
echo "- API Docs: http://localhost:8000/docs"
echo ""
