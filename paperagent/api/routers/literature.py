"""Literature API router"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from paperagent.database import get_db
from paperagent.database.models import Literature
from paperagent.api import schemas
from paperagent.agents.literature_agent import LiteratureAgent

router = APIRouter()
literature_agent = LiteratureAgent()

@router.post("/search")
def search_literature(action: schemas.LiteratureAction):
    """Search for literature"""
    try:
        result = literature_agent.execute(action.model_dump())
        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/", response_model=List[schemas.LiteratureResponse])
def list_literature(project_id: int, skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """List literature for a project"""
    literature = db.query(Literature).filter(Literature.project_id == project_id).offset(skip).limit(limit).all()
    return literature

@router.get("/{lit_id}", response_model=schemas.LiteratureResponse)
def get_literature(lit_id: int, db: Session = Depends(get_db)):
    """Get literature by ID"""
    lit = db.query(Literature).filter(Literature.id == lit_id).first()
    if not lit:
        raise HTTPException(status_code=404, detail="Literature not found")
    return lit

@router.post("/analyze")
def analyze_paper(action: schemas.LiteratureAction):
    """Analyze a research paper"""
    try:
        result = literature_agent.execute(action.model_dump())
        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
