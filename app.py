from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, field_validator
from typing import List
import logging
from batch_processor import predict_batch_majority
from fastapi.middleware.cors import CORSMiddleware

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Crop Recommendation API - Cameroun (6 features)",
    description="API cultures agricoles: N,P,K,temp,humidity,ph → 26 cultures recommandées",
    version="1.0",
    contact={
        "name": "Bala Andegue",
        "email": "balaandeguefrancoislionnel@gmail.com",
    }
)

# Autoriser le domaine frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://smart-agro-three.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ✅ 6 FEATURES (PAS rainfall)
class SoilSample(BaseModel):
    N: float = Field(..., gt=0, description="Azote (kg/ha)")
    P: float = Field(..., gt=0, description="Phosphore (kg/ha)") 
    K: float = Field(..., gt=0, description="Potassium (kg/ha)")
    temperature: float = Field(..., description="Température (°C)")
    humidity: float = Field(ge=0, le=100, description="Humidité (%)")
    ph: float = Field(ge=0, le=14, description="pH sol")
    
    @field_validator('temperature')
    @classmethod
    def validate_temperature(cls, v):
        if not -50 <= v <= 60:
            raise ValueError('Température agriculture: -50°C à 60°C')
        return v

class BatchRequest(BaseModel):
    samples: List[SoilSample] = Field(
        ...,
        min_items=1, max_items=10,
        description="1-10 échantillons sol Cameroun"
    )

# Response flexible (batch_processor retourne dict)
@app.post("/predict/batch")
async def predict_batch(request: BatchRequest):
    """
    🔹 Recommande culture majoritaire (26 cultures)
    🔹 Features: N,P,K,temp,humidity,ph (6)
    """
    try:
        logger.info(f"Batch {len(request.samples)} échantillons")
        data = [s.dict() for s in request.samples]  # 6 features
        result = predict_batch_majority(data)
        logger.info(f"✅ {result.get('recommended_crop', 'OK')}")
        return result
        
    except Exception as e:
        logger.error(f"❌ {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", tags=["Accueil"])
async def home():
    return {
        "🌱": "Agriculture Cameroun API",
        "version": "1.0",
        "endpoints": ["/predict/batch POST", "/health"],
        "docs": "/docs",  # Swagger automatique
        "features": ["N","P","K","temperature","humidity","ph"]
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "crops-26classes"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
