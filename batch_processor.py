from collections import Counter
import joblib
import pandas as pd
import numpy as np

def predict_batch_majority(samples: list):
    try:
        # VOS fichiers
        model = joblib.load('model/crop_recommendation_model_26crops.pkl')
        le = joblib.load('model/label_encoder_26crops.pkl')

        # ✅ ORDRE EXACT DU MODÈLE
        MODEL_FEATURES = ['N', 'P', 'K', 'ph', 'humidity', 'temperature']
        
        # Réorganiser colonnes dans BON ORDRE
        df_batch = pd.DataFrame(samples)
        X_batch = df_batch[MODEL_FEATURES].values  # ← ORDRE CORRECT !
        
        # Prédictions
        preds_encoded = model.predict(X_batch)
        preds_labels = le.inverse_transform(preds_encoded)
        
        # Vote majoritaire
        vote = Counter(preds_labels)
        winner = vote.most_common(1)[0][0]
        confidence = vote[winner] / len(preds_labels)
        
        return {
            "recommended_crop": winner,
            "confidence": round(confidence, 3),
            "total_samples": len(samples),
            "features_order": MODEL_FEATURES,  # ← Documentation
            "all_predictions": preds_labels.tolist(),
            "vote_details": dict(vote)
        }
        
    except Exception as e:
        return {"error": f"{str(e)}"}
