# Sauvegarde ce code dans app.py
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, validator
from typing import List, Optional
import logging
from batch_processor import predict_batch_majority

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Crop Recommendation API - Batch Mode",
    description="API pour recommander des cultures agricoles. Envoie jusqu'à 10 échantillons → retourne la culture majoritaire",
    version="1.0",
    contact={
        "name": "Votre Nom/Équipe",
        "email": "agriculture@example.com",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    }
)

# Modèles Pydantic avec validation
class SoilSample(BaseModel):
    N: float = Field(..., gt=0, description="Teneur en azote (N)")
    P: float = Field(..., gt=0, description="Teneur en phosphore (P)")
    K: float = Field(..., gt=0, description="Teneur en potassium (K)")
    temperature: float = Field(..., description="Température en °C")
    humidity: float = Field(..., ge=0, le=100, description="Humidité relative en %")
    ph: float = Field(..., ge=0, le=14, description="pH du sol")
    rainfall: float = Field(..., gt=0, description="Précipitations en mm")
    
    @validator('temperature')
    def validate_temperature(cls, v):
        if not -50 <= v <= 60:  # Plage raisonnable pour l'agriculture
            raise ValueError('Température hors plage valide (-50°C à 60°C)')
        return v

class BatchRequest(BaseModel):
    samples: List[SoilSample] = Field(
        ...,
        min_items=1,
        max_items=10,
        description="Liste de 1 à 10 échantillons de sol"
    )

class PredictionResponse(BaseModel):
    majority_crop: str = Field(..., description="Culture majoritaire recommandée")
    predictions: List[str] = Field(..., description="Prédictions pour chaque échantillon")
    confidence: float = Field(..., description="Confiance de la prédiction (pourcentage)")
    sample_count: int = Field(..., description="Nombre d'échantillons traités")

@app.post(
    "/predict/batch",
    response_model=PredictionResponse,
    summary="Prédire la culture majoritaire",
    description="Soumet un batch de 1 à 10 échantillons de sol et retourne la culture majoritaire recommandée",
    response_description="La culture recommandée avec les détails de prédiction",
    tags=["Prédictions"]
)
async def predict_batch(request: BatchRequest):
    """
    Endpoint pour la prédiction par batch.
    
    - **samples**: Liste d'échantillons de sol (1-10)
    - Retourne la culture la plus fréquemment recommandée
    """
    try:
        # Log de la requête
        logger.info(f"Requête batch reçue avec {len(request.samples)} échantillons")
        
        # Conversion des données
        data = [s.dict() for s in request.samples]
        
        # Appel au batch processor
        result = predict_batch_majority(data)
        
        # Log du résultat
        logger.info(f"Prédiction terminée: {result.get('majority_crop', 'N/A')}")
        
        return result
        
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur interne du serveur: {str(e)}"
        )

@app.get("/", tags=["Accueil"])
def home():
    """Page d'accueil de l'API"""
    return {
        "message": "Bienvenue sur l'API de Recommandation de Cultures",
        "version": "1.0",
        "documentation": "/docs",
        "endpoints": {
            "batch_prediction": "/predict/batch [POST]"
        }
    }

@app.get("/health", tags=["Santé"])
def health_check():
    """Vérification de l'état de l'API"""
    return {
        "status": "healthy",
        "service": "crop-recommendation-api"
    }

# Gestionnaire d'erreurs global
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return {
        "error": exc.detail,
        "status_code": exc.status_code
    }