# Rapport de Projet : Système de Recommandation de Cultures — Top-3

**Auteur :** Bala Andegue François-Lionnel  
**Date :** Février 2026  
**Version :** 2.0 — Research-Based Dataset · Top-3 par `predict_proba`

---

## 1. Introduction

Ce projet implémente un système intelligent de **recommandation des trois meilleures cultures** adaptées à un profil de sol donné. À partir de sept paramètres agro-pédologiques, le système retourne — pour **chaque vecteur de caractéristiques** — le **classement ordonné des 3 cultures les plus adaptées** avec leur niveau de confiance (probabilité RFC en %).

Pour un **lot de N vecteurs**, le **Top-3 global** est obtenu par **moyenne des probabilités RF** sur l'ensemble des N échantillons.

Le dataset et le modèle ont été entièrement refondus sur des références scientifiques de référence :

| Code | Référence |
|------|-----------|
| [1]  | FAO. *Crop Production Guidelines and Soil Management in Tropical Regions.* |
| [2]  | IITA. *Crop and Soil Fertility Management Practices in Sub-Saharan Africa.* |
| [3]  | IRAD. *Fiches techniques des cultures vivrières et de rente au Cameroun.* |
| [4]  | CIRAD. *Agronomic Practices in Tropical Agriculture.* |

---

## 2. Dataset Synthétisé

### 2.1. Sélection des cultures
Seules les cultures **explicitement documentées** dans les quatre sources scientifiques sont incluses. Les cultures non documentées (ex. : arachides, cultures asiatiques du dataset Kaggle original) ont été exclues.

### 2.2. Unités
- **N, P, K** : mg/kg (= ppm), unité standard des analyses pédologiques FAO/IITA
- **Température** : °C · **Humidité** : % · **pH** : sans unité · **Pluviométrie** : mm/an

### 2.3. Cultures incluses (15)

| Catégorie | Cultures |
|---|---|
| Céréales | `riz`, `mais`, `sorgho`, `mil` |
| Légumineuses | `haricot`, `soja`, `niebe` |
| Tubercules / Racines | `manioc`, `igname`, `taro` |
| Cultures d'exportation / rente | `cacao`, `cafe_robusta`, `cafe_arabica`, `palmier_a_huile` |
| Vivrières fruitières | `banane_plantain` |

### 2.4. Méthode de génération

| Paramètre | Valeur |
|---|---|
| Script | `generate_research_dataset.py` |
| Volume | 500 échantillons × 15 cultures = **7 500 lignes** |
| Tirage | `numpy.random.uniform` dans les fenêtres agronomiques validées |
| Reproductibilité | `random_state = 42` |
| Fichier résultant | `data/research_based_dataset.csv` |

---

## 3. Intervalles Agronomiques par Culture

Toutes les valeurs N, P, K sont en **mg/kg**. Temp. en °C, Humidité en %, Pluviométrie en mm/an.

| Culture | N (mg/kg) | P (mg/kg) | K (mg/kg) | Temp. (°C) | Humidité (%) | pH | Pluie (mm) | Réf. |
|---|---|---|---|---|---|---|---|---|
| riz | 60–120 | 15–35 | 20–50 | 22–32 | 70–90 | 5.0–7.0 | 1200–2500 | [1][2] |
| mais | 80–150 | 20–50 | 30–80 | 18–32 | 55–80 | 5.5–7.5 | 600–1200 | [1][2][3] |
| sorgho | 30–80 | 10–30 | 15–50 | 25–38 | 30–60 | 5.5–8.0 | 300–900 | [1][2] |
| mil | 20–70 | 5–25 | 10–40 | 25–40 | 25–55 | 5.5–8.5 | 200–700 | [1][2] |
| haricot | 20–60 | 15–40 | 20–60 | 18–28 | 50–75 | 5.5–7.0 | 400–900 | [1][2][3] |
| soja | 25–70 | 20–50 | 30–80 | 22–32 | 55–80 | 5.5–7.0 | 500–1100 | [1][2] |
| niebe | 15–50 | 10–35 | 15–55 | 24–36 | 35–70 | 5.5–7.5 | 300–1000 | [1][2][3] |
| manioc | 50–120 | 15–45 | 80–200 | 24–35 | 60–85 | 5.0–7.0 | 800–2000 | [2][3][4] |
| igname | 60–130 | 20–60 | 80–180 | 24–34 | 60–85 | 5.5–7.0 | 900–2000 | [2][3] |
| taro | 50–110 | 15–50 | 70–150 | 20–35 | 70–95 | 5.0–7.0 | 1500–4000 | [3][4] |
| cacao | 60–140 | 25–70 | 50–120 | 20–30 | 70–90 | 5.5–7.0 | 1500–3000 | [3][4] |
| cafe_robusta | 80–160 | 20–55 | 60–130 | 22–30 | 60–85 | 5.0–6.5 | 1200–2500 | [3][4] |
| cafe_arabica | 70–150 | 15–50 | 50–120 | 15–24 | 55–75 | 5.5–6.5 | 1500–2500 | [3][4] |
| palmier_a_huile | 80–180 | 20–60 | 80–200 | 24–32 | 75–95 | 4.5–6.5 | 1800–3500 | [2][3][4] |
| banane_plantain | 100–200 | 30–80 | 120–280 | 22–35 | 70–90 | 5.5–7.5 | 1800–3500 | [1][3][4] |

---

## 4. Modèle — Architecture et Performances

### 4.1. Algorithme

| Paramètre | Valeur |
|---|---|
| Algorithme | `RandomForestClassifier` (scikit-learn) |
| `n_estimators` | 200 arbres |
| `random_state` | 42 |
| `n_jobs` | -1 (parallélisation maximale) |
| Split train/test | 80 % / 20 % (stratifié) |
| Features (7) | `N`, `P`, `K`, `temperature`, `humidity`, `ph`, `rainfall` |
| Fichier modèle | `model/top3_crop_model.pkl` |

### 4.2. Performances (métriques issues du bundle)

| Métrique | Valeur |
|---|---|
| Validation croisée 10-fold — moyenne | **86.20 %** |
| Validation croisée 10-fold — écart-type | **± 0.91 %** |
| Accuracy sur le jeu de test (20 %) | **85.93 %** |

> **Note :** L'accuracy de 86 % reflète la réalité agronomique. Plusieurs cultures partagent des conditions similaires (ex. : banane\_plantain, taro, palmier à huile en zone humide). La léger écart traduit ce chevauchement naturel — le modèle est robuste et non surajusté.

### 4.3. Pipeline Top-3 par vecteur

Pour **chaque vecteur de caractéristiques**, `predict_proba()` calcule la probabilité de chaque culture et retourne les **3 cultures les plus probables**, triées par confiance décroissante :

```python
# batch_processor.py — logique centrale
all_probas = model.predict_proba(X_batch)   # shape (n_samples, 15)

# Par échantillon → Top-3
idx = np.argsort(proba)[::-1][:3]
top3 = [{"rang": i+1, "culture": classes[j], "confiance": round(proba[j]*100, 1)}
        for i, j in enumerate(idx)]

# Pour N vecteurs → Top-3 global (moyenne des probabilités)
mean_proba = all_probas.mean(axis=0)
top3_global = [{"rang": i+1, "culture": classes[j], "confiance_agregee": round(mean_proba[j]*100,1)}
               for i, j in enumerate(np.argsort(mean_proba)[::-1][:3])]
```

### 4.4. Exemples de Recommandations

| Profil de sol | Zone Cameroun | 🥇 Rang 1 | 🥈 Rang 2 | 🥉 Rang 3 |
|---|---|---|---|---|
| Fertile humide (N=120, P=45, K=180, pH=6.0, 2200 mm) | Littoral / Sud | banane\_plantain | manioc | taro |
| Sahélien pauvre (N=30, P=10, K=25, pH=7.2, 450 mm) | Extrême-Nord | sorgho | mil | niebe |
| Hautes terres (N=85, P=30, K=70, pH=5.8, 1400 mm) | Ouest / Nord-Ouest | mais | haricot | soja |

---

## 5. Architecture du Système

### 5.1. API FastAPI (`app.py` + `batch_processor.py`)

**Endpoint :** `POST /predict/batch`  
**Entrée :** 1 à 10 vecteurs de sol (JSON)  
**Sortie :**
- `resultats_par_echantillon` : Top-3 cultures + confiance pour **chaque** vecteur
- `top3_global` : Top-3 agrégé (moyenne probabilités) sur le lot
- `nb_echantillons` : nombre de vecteurs traités

```json
{
  "samples": [
    {"N": 90, "P": 42, "K": 43, "temperature": 20,
     "humidity": 82, "ph": 6.5, "rainfall": 200}
  ]
}
```

**Réponse :**
```json
{
  "nb_echantillons": 1,
  "resultats_par_echantillon": [
    {"echantillon": 1, "top3": [
      {"rang": 1, "culture": "mais",    "confiance": 77.5},
      {"rang": 2, "culture": "soja",    "confiance": 5.0},
      {"rang": 3, "culture": "cacao",   "confiance": 4.5}
    ]}
  ],
  "top3_global": [
    {"rang": 1, "culture": "mais",  "confiance_agregee": 77.5},
    {"rang": 2, "culture": "soja",  "confiance_agregee": 5.0},
    {"rang": 3, "culture": "cacao", "confiance_agregee": 4.5}
  ]
}
```

**Documentation interactive :** `http://localhost:8000/docs` (Swagger UI)

### 5.2. Application Streamlit (`streamlit_app.py`)

- Charge `model/top3_crop_model.pkl` via `pickle`
- Formulaire dynamique : 1 à 10 vecteurs de sol
- Pour **chaque vecteur** : affiche le Top-3 avec médailles + barre de confiance
- Pour le **lot global** : Top-3 agrégé par moyenne, graphique de distribution des probabilités
- Tableau récapitulatif : Top-1, Top-2, Top-3 par échantillon

---

## 6. Fichiers du Projet

| Fichier | Description |
|---|---|
| `generate_research_dataset.py` | Génération du dataset (paramètres FAO/IITA/IRAD/CIRAD) |
| `data/research_based_dataset.csv` | Dataset (7 500 lignes · 15 cultures · mg/kg) |
| `synthesized_research_data.ipynb` | EDA : distributions, heatmaps, tableau récapitulatif |
| `train_top3_model.ipynb` | Entraînement, évaluation, prédiction Top-3 |
| `model/top3_crop_model.pkl` | Bundle RF : modèle + classes + métriques + références |
| `batch_processor.py` | Pipeline de prédiction Top-3 par lot |
| `app.py` | API FastAPI : POST /predict/batch |
| `streamlit_app.py` | Interface utilisateur Streamlit |

---

## 7. Références Bibliographiques

- **[1]** FAO. *Crop Production Guidelines and Soil Management in Tropical Regions.* Food and Agriculture Organization of the United Nations, Rome.
- **[2]** IITA. *Crop and Soil Fertility Management Practices in Sub-Saharan Africa.* International Institute of Tropical Agriculture, Ibadan.
- **[3]** IRAD. *Fiches techniques des cultures vivrières et de rente au Cameroun.* Institut de Recherche Agricole pour le Développement, Yaoundé.
- **[4]** CIRAD. *Agronomic Practices in Tropical Agriculture.* Centre de coopération internationale en recherche agronomique pour le développement, Montpellier.
