"""Experiments API router"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from paperagent.database import get_db
from paperagent.database.models import Experiment
from paperagent.api import schemas
from paperagent.agents.experiment_agent import ExperimentAgent

router = APIRouter()
experiment_agent = ExperimentAgent()

@router.post("/design")
def design_experiment(action: schemas.ExperimentAction):
    """Design a new experiment"""
    try:
        result = experiment_agent.execute(action.model_dump())
        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/", response_model=List[schemas.ExperimentResponse])
def list_experiments(project_id: int, db: Session = Depends(get_db)):
    """List experiments for a project"""
    experiments = db.query(Experiment).filter(Experiment.project_id == project_id).all()
    return experiments

@router.get("/{exp_id}", response_model=schemas.ExperimentResponse)
def get_experiment(exp_id: int, db: Session = Depends(get_db)):
    """Get experiment by ID"""
    exp = db.query(Experiment).filter(Experiment.id == exp_id).first()
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return exp

@router.post("/analyze")
def analyze_data(action: schemas.ExperimentAction):
    """Analyze experimental data"""
    try:
        result = experiment_agent.execute(action.model_dump())
        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
