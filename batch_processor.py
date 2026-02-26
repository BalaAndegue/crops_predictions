"""
batch_processor.py — Pipeline de recommandation Top-3 par lot (1–10 échantillons)
===================================================================================
Pour chaque échantillon du lot :
   → Top-3 cultures distinctes + niveau de confiance (probabilité RF)

Pour l'ensemble du lot :
   → Top-3 agrégées (moyenne des probabilités sur tous les échantillons)
"""

import pickle
import numpy as np
import pandas as pd

# ─── Chargement du modèle ────────────────────────────────────────────────────
_MODEL_PATH = "model/top3_crop_model.pkl"
_bundle     = None   # chargement paresseux

FEATURE_NAMES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


def _load_bundle():
    global _bundle
    if _bundle is None:
        with open(_MODEL_PATH, "rb") as f:
            _bundle = pickle.load(f)
    return _bundle


# ─── Helpers ─────────────────────────────────────────────────────────────────
def _top3_for_sample(proba: np.ndarray, classes: list, top_k: int = 3) -> list:
    """Retourne le Top-3 (culture, confiance) pour un vecteur de probabilités."""
    idx = np.argsort(proba)[::-1][:top_k]
    return [
        {
            "rang":       i + 1,
            "culture":    classes[j],
            "confiance":  round(float(proba[j]) * 100, 1),  # %
        }
        for i, j in enumerate(idx)
    ]


def _aggregate_top3(all_probas: np.ndarray, classes: list, top_k: int = 3) -> list:
    """
    Agrège les probabilités de tous les échantillons (moyenne).
    Retourne le Top-3 global avec confiance agrégée.
    """
    mean_proba = all_probas.mean(axis=0)
    idx        = np.argsort(mean_proba)[::-1][:top_k]
    return [
        {
            "rang":               i + 1,
            "culture":            classes[j],
            "confiance_agregee":  round(float(mean_proba[j]) * 100, 1),  # %
        }
        for i, j in enumerate(idx)
    ]


# ─── Point d'entrée principal ─────────────────────────────────────────────────
def predict_batch_top3(samples: list) -> dict:
    """
    Prédit le Top-3 de cultures pour chaque échantillon, puis fournit
    le Top-3 agrégé sur l'ensemble du lot.

    Args:
        samples: liste de 1 à 10 dicts avec les clés
                 [N, P, K, temperature, humidity, ph, rainfall].

    Returns:
        dict avec :
            - "resultats_par_echantillon": list[dict] (Top-3 par échantillon)
            - "top3_global":               list[dict] (Top-3 agrégé du lot)
            - "nb_echantillons":           int
    """
    try:
        bundle  = _load_bundle()
        model   = bundle["model"]
        classes = bundle["classes"]

        if not (1 <= len(samples) <= 10):
            return {"error": f"Le lot doit contenir entre 1 et 10 échantillons (reçu : {len(samples)})."}

        # ── Construire la matrice de features
        df      = pd.DataFrame(samples)
        X_batch = df[FEATURE_NAMES].values

        # ── Probabilités RF pour tous les échantillons d'un coup
        all_probas = model.predict_proba(X_batch)   # shape (n_samples, n_classes)

        # ── Top-3 par échantillon
        resultats = []
        for i, proba in enumerate(all_probas):
            top3 = _top3_for_sample(proba, classes)
            resultats.append({
                "echantillon": i + 1,
                "top3":        top3,
            })

        # ── Top-3 global (agrégé)
        top3_global = _aggregate_top3(all_probas, classes)

        return {
            "resultats_par_echantillon": resultats,
            "top3_global":               top3_global,
            "nb_echantillons":           len(samples),
        }

    except FileNotFoundError:
        return {
            "error": (
                f"Modèle introuvable : '{_MODEL_PATH}'. "
                "Lancez d'abord train_top3_model.ipynb."
            )
        }
    except Exception as exc:
        return {"error": str(exc)}
