from collections import Counter
import numpy as np
import pickle
import pandas as pd
def predict_batch_majority(samples: list, model_path="/model/crop_model.pkl"):
    """
    samples : liste de max 10 dictionnaires ou listes avec les 7 features
    Retourne la culture majoritaire + détails des votes
    """
    # Chargement du modèle
    with open(model_path, "rb") as f:
        saved = pickle.load(f)
    model = saved['model']
    le = saved['label_encoder']
    
    # Conversion en DataFrame si besoin
    if isinstance(samples[0], dict):
        df_batch = pd.DataFrame(samples)
    else:
        df_batch = pd.DataFrame(samples, columns=saved['features'])
    
    if len(df_batch) == 0:
        return {"error": "Batch vide"}
    if len(df_batch) > 10:
        raise ValueError("Maximum 10 échantillons autorisés par batch")
    
    # Prédictions individuelles
    preds_encoded = model.predict(df_batch)
    preds_labels = le.inverse_transform(preds_encoded)
    
    # Vote majoritaire
    vote = Counter(preds_labels)
    winner = vote.most_common(1)[0][0]
    confidence = vote[winner] / len(preds_labels)
    
    return {
        "recommended_crop": winner,
        "confidence": round(confidence, 3),
        "total_samples": len(df_batch),
        "all_predictions": preds_labels.tolist(),
        "vote_details": dict(vote)
    }