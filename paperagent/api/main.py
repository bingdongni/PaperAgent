"""
FastAPI main application
"""

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Optional
import uvicorn

from paperagent import __version__
from paperagent.database import get_db, init_db
from paperagent.api import schemas
from paperagent.api.routers import projects, literature, experiments, papers, tasks

# Initialize FastAPI app
app = FastAPI(
    title="PaperAgent API",
    description="Academic Multi-Agent Collaboration Framework API",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(projects.router, prefix="/api/projects", tags=["projects"])
app.include_router(literature.router, prefix="/api/literature", tags=["literature"])
app.include_router(experiments.router, prefix="/api/experiments", tags=["experiments"])
app.include_router(papers.router, prefix="/api/papers", tags=["papers"])
app.include_router(tasks.router, prefix="/api/tasks", tags=["tasks"])


@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    init_db()


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "PaperAgent API",
        "version": __version__,
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(
        "paperagent.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
