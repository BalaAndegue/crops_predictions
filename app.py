# Sauvegarde ce code dans app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from batch_processor import predict_batch_majority

app = FastAPI(
    title="Crop Recommendation API - Batch Mode (Max 10)",
    description="Envoie jusqu'à 10 échantillons → retourne la culture majoritaire",
    version="1.0"
)

class SoilSample(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

class BatchRequest(BaseModel):
    samples: List[SoilSample]

@app.post("/predict/batch")
async def predict_batch(request: BatchRequest):
    if len(request.samples) == 0:
        raise HTTPException(status_code=400, detail="Batch vide")
    if len(request.samples) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 échantillons")
    
    data = [s.dict() for s in request.samples]
    result = predict_batch_majority(data)
    return result

@app.get("/")
def home():
    return {"message": "API Crop Recommendation - POST /predict/batch"}