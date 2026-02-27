"""
app.py — FastAPI : Recommandation Top-3 cultures par lot (1–10 échantillons)
==============================================================================
POST /predict/batch
   Entrée  : 1–10 échantillons de sol (N, P, K, temperature, humidity, ph, rainfall)
   Sortie  :
     - resultats_par_echantillon : Top-3 cultures + confiance pour chaque échantillon
     - top3_global               : Top-3 agrégé sur l'ensemble du lot
     - nb_echantillons           : nombre d'échantillons traités
"""

import logging
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from batch_processor import predict_batch_top3

# ─── Configuration ────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Crop Recommendation API — Top-3 (Afrique sub-saharienne / Cameroun)",
    description=(
        "Pour 1 à 10 échantillons de sol, retourne :\n"
        "- le **Top-3 de cultures** pour chaque échantillon individuel ;\n"
        "- le **Top-3 agrégé** sur l'ensemble du lot.\n\n"
        "Unités : N, P, K en **mg/kg** | Pluviométrie en **mm/an**.\n\n"
        "Sources : FAO [1] · IITA [2] · IRAD [3] · CIRAD [4]."
    ),
    version="2.0",
    contact={
        "name": "Bala Andegue",
        "email": "balaandeguefrancoislionnel@gmail.com",
    },
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://smart-agro-three.vercel.app", "*",'http://10.179.122.11:8000'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Schémas Pydantic ─────────────────────────────────────────────────────────
class SoilSample(BaseModel):
    """Un seul échantillon de sol."""

    N: float = Field(..., gt=0, description="Azote (mg/kg)")
    P: float = Field(..., gt=0, description="Phosphore (mg/kg)")
    K: float = Field(..., gt=0, description="Potassium (mg/kg)")
    temperature: float = Field(..., description="Température (°C)")
    humidity: float = Field(..., ge=0, le=100, description="Humidité relative (%)")
    ph: float = Field(..., ge=0, le=14, description="pH du sol")
    rainfall: float = Field(..., gt=0, description="Pluviométrie annuelle (mm)")

    @field_validator("temperature")
    @classmethod
    def check_temperature(cls, v: float) -> float:
        if not -10 <= v <= 50:
            raise ValueError("Température hors plage agricole (-10 °C à 50 °C).")
        return v


class BatchRequest(BaseModel):
    """Lot de 1 à 10 échantillons."""

    samples: List[SoilSample] = Field(
        ...,
        min_length=1,
        max_length=10,
        description="Liste de 1 à 10 échantillons de sol.",
    )


# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.post(
    "/predict/batch",
    summary="Top-3 cultures par échantillon + Top-3 global",
    response_description=(
        "resultats_par_echantillon : Top-3 par échantillon ; "
        "top3_global : Top-3 agrégé sur le lot."
    ),
)
async def predict_batch(request: BatchRequest):
    """
    ### Logique
    1. Pour **chaque** échantillon de sol → Top-3 cultures distinctes
       ordonnées par niveau de confiance décroissant (probabilité RF).
    2. Pour l'**ensemble du lot** → Top-3 agrégées (moyenne des probabilités
       sur tous les échantillons).

    ### Exemple de réponse
    ```json
    {
      "nb_echantillons": 2,
      "resultats_par_echantillon": [
        {
          "echantillon": 1,
          "top3": [
            {"rang": 1, "culture": "palmier_a_huile", "confiance": 45.3},
            {"rang": 2, "culture": "banane_plantain",  "confiance": 30.1},
            {"rang": 3, "culture": "cafe_robusta",     "confiance": 12.7}
          ]
        },
        ...
      ],
      "top3_global": [
        {"rang": 1, "culture": "palmier_a_huile", "confiance_agregee": 44.1},
        {"rang": 2, "culture": "banane_plantain",  "confiance_agregee": 28.9},
        {"rang": 3, "culture": "cafe_robusta",     "confiance_agregee": 13.5}
      ]
    }
    ```
    """
    logger.info("POST /predict/batch — %d échantillon(s)", len(request.samples))
    data   = [s.model_dump() for s in request.samples]
    result = predict_batch_top3(data)

    if "error" in result:
        logger.error("❌ %s", result["error"])
        raise HTTPException(status_code=500, detail=result["error"])

    top1 = result["top3_global"][0]["culture"] if result.get("top3_global") else "?"
    logger.info("✅ Top-1 global : %s", top1)
    return result


@app.get("/", tags=["Info"])
async def home():
    return {
        "message":   "Crop Recommendation API — Top-3 (Cameroun / Afrique sub-saharienne)",
        "version":   "2.0",
        "endpoints": {"POST /predict/batch": "1–10 échantillons → Top-3 par sol + Top-3 global"},
        "features":  ["N (mg/kg)", "P (mg/kg)", "K (mg/kg)", "temperature (°C)",
                      "humidity (%)", "ph", "rainfall (mm)"],
        "cultures":  15,
        "docs":      "/docs",
    }


@app.get("/health", tags=["Info"])
async def health():
    return {"status": "healthy", "service": "crops-top3-v2"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
