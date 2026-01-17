"""
Projects API router
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from paperagent.database import get_db
from paperagent.database.models import Project, TaskStatus
from paperagent.api import schemas
from paperagent.agents.boss_agent import BossAgent

router = APIRouter()
boss_agent = BossAgent()


@router.post("/", response_model=schemas.ProjectResponse, status_code=status.HTTP_201_CREATED)
def create_project(project: schemas.ProjectCreate, db: Session = Depends(get_db)):
    """Create a new research project"""
    result = boss_agent.execute({
        'action': 'create_project',
        'name': project.name,
        'description': project.description,
        'research_field': project.research_field,
        'keywords': project.keywords
    })

    db_project = db.query(Project).filter(Project.id == result['project_id']).first()
    return db_project


@router.get("/", response_model=List[schemas.ProjectResponse])
def list_projects(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """List all projects"""
    projects = db.query(Project).offset(skip).limit(limit).all()
    return projects


@router.get("/{project_id}", response_model=schemas.ProjectResponse)
def get_project(project_id: int, db: Session = Depends(get_db)):
    """Get project by ID"""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


@router.get("/{project_id}/summary")
def get_project_summary(project_id: int):
    """Get comprehensive project summary"""
    try:
        summary = boss_agent.get_project_summary(project_id)
        return summary
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/{project_id}/progress")
def get_project_progress(project_id: int):
    """Get project progress"""
    result = boss_agent.execute({
        'action': 'monitor_progress',
        'project_id': project_id
    })
    return result


@router.post("/{project_id}/execute-workflow")
def execute_workflow(project_id: int):
    """Execute complete research workflow for project"""
    try:
        result = boss_agent.execute({
            'action': 'execute_workflow',
            'project_id': project_id
        })
        return {"success": True, "message": "Workflow executed", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{project_id}", response_model=schemas.ProjectResponse)
def update_project(
    project_id: int,
    project_update: schemas.ProjectUpdate,
    db: Session = Depends(get_db)
):
    """Update project"""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    update_data = project_update.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(project, key, value)

    db.commit()
    db.refresh(project)
    return project


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_project(project_id: int, db: Session = Depends(get_db)):
    """Delete project"""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    db.delete(project)
    db.commit()
    return None
