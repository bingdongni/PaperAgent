"""Papers API router"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from paperagent.database import get_db
from paperagent.database.models import Paper
from paperagent.api import schemas
from paperagent.agents.writing_agent import WritingAgent

router = APIRouter()
writing_agent = WritingAgent()

@router.post("/create")
def create_paper(action: schemas.WritingAction):
    """Create paper structure"""
    try:
        result = writing_agent.execute(action.model_dump())
        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/", response_model=List[schemas.PaperResponse])
def list_papers(project_id: int, db: Session = Depends(get_db)):
    """List papers for a project"""
    papers = db.query(Paper).filter(Paper.project_id == project_id).all()
    return papers

@router.get("/{paper_id}", response_model=schemas.PaperResponse)
def get_paper(paper_id: int, db: Session = Depends(get_db)):
    """Get paper by ID"""
    paper = db.query(Paper).filter(Paper.id == paper_id).first()
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    return paper

@router.post("/write")
def write_section(action: schemas.WritingAction):
    """Write a paper section"""
    try:
        result = writing_agent.execute(action.model_dump())
        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/{paper_id}", response_model=schemas.PaperResponse)
def update_paper(paper_id: int, paper_update: schemas.PaperUpdate, db: Session = Depends(get_db)):
    """Update paper"""
    paper = db.query(Paper).filter(Paper.id == paper_id).first()
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")

    update_data = paper_update.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(paper, key, value)

    db.commit()
    db.refresh(paper)
    return paper
