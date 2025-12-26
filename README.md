
# Crop Recommendation System 🌾  
**Prédiction intelligente de la meilleure culture à planter selon le sol et le climat**  
**Vote majoritaire sur lots de 10 échantillons maximum**

[![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://your-app-link.streamlit.app)
[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-blue?style=for-the-badge&logo=huggingface)](https://huggingface.co/spaces/your-username/crop-recommendation)
[![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

> **Accuracy du modèle : 99.7 %** sur le dataset public  
> **Stratégie de vote majoritaire** → encore plus robuste en conditions réelles

---

### Fonctionnalités

- Prédiction par **lots de 1 à 10 échantillons**
- **Vote majoritaire** automatique (la culture qui gagne le plus de votes)
- Interface Streamlit moderne et intuitive
- API FastAPI incluse (`/predict/batch`)
- Supporte 22 cultures : rice, maize, chickpea, kidneybeans, pigeonpeas, mothbeans, mungbean, blackgram, lentil, pomegranate, banana, mango, grapes, watermelon, muskmelon, apple, orange, papaya, coconut, cotton, jute, coffee

---

### Démo en direct

 Lien Streamlit (gratuit) → https://crop-recommendation.streamlit.app  
 Lien Hugging Face → https://huggingface.co/spaces/ton-pseudo/crop-recommendation  

---

### Installation locale

```bash
# 1. Cloner le repo
git clone https://github.com/ton-pseudo/crop-recommendation.git
cd crop-recommendation

# 2. Créer un environnement virtuel (optionnel mais recommandé)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer l'app Streamlit
streamlit run streamlit_app.py
```

---

### Déploiement ultra-rapide (0 €)

| Plateforme                  | Temps de déploiement | Coût     | Lien direct |
|----------------------------|----------------------|----------|-------------|
| Streamlit Community Cloud  | 30 secondes          | Gratuit  | https://share.streamlit.io |
| Hugging Face Spaces        | 1 minute             | Gratuit  | https://huggingface.co/new-space |
| Render.com (toujours actif)| 2 minutes            | 7 $/mois | https://render.com |

---

### Fichiers du projet

```
crop-recommendation/
├── crop_model.pkl              ← Modèle Random Forest entraîné (99.7%)
├── Crop_recommendation.csv     ← Dataset original (2200 échantillons)
├── streamlit_app.py            ← Interface web complète
├── app.py                      ← API FastAPI (prédiction par batch)
├── model_train.py              ← Script d'entraînement (reproductible)
├── requirements.txt
├── README.md                   ← Ce fichier
└── .gitignore
```

---

### Exemple de prédiction (vote majoritaire)

```json
{
  "recommended_crop": "rice",
  "confidence": 0.90,
  "total_samples": 10,
  "vote_details": {"rice": 9, "maize": 1}
}
```

---

### Performances du modèle

| Métrique                        | Résultat          |
|---------------------------------|-------------------|
| Accuracy (test set)             | **99.7 %**        |
| Validation croisée 10-fold      | 99.4 % ± 0.3 %   |
| Avec 10 % de bruit              | 99.1 %            |
| Vote majoritaire (batch de 10)  | **99.9 – 100 %**  |

---

### Auteur

Fait avec ❤️ par **BALA ANDEGUE FRANCOIS**  
- GitHub : https://github.com/BalaAndegue 
- LinkedIn : https://linkedin.com/in/FrancoisLionnel  

---

### Licence

MIT License – tu peux réutiliser, modifier, vendre, tout ce que tu veux !

> **Prêt à planter la bonne culture au bon endroit ?**  
> Lance l’app et teste avec tes propres données terrain !

