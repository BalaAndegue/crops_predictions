# Rapport de Projet : Système de Recommandation de Cultures — Top-3 (Afrique sub-saharienne / Cameroun)

**Date :** Février 2026 (mis à jour)  
**Projet :** Crop Recommendation System — Research-Based Dataset

---

## 1. Introduction

Ce rapport détaille les travaux techniques réalisés pour la conception d'un système intelligent de **recommandation des trois meilleures cultures** adaptées à un sol donné. L'objectif est, à partir des paramètres nutritifs du sol (Azote `N`, Phosphore `P`, Potassium `K`, `pH`) et des conditions climatiques (`Température`, `Humidité`, `Pluviométrie`), de retourner le **classement des 3 cultures les plus adaptées**, ordonnées par probabilité.

Ce dataset et ce modèle ont été **entièrement refondus** pour s'appuyer exclusivement sur des références scientifiques de référence :

| Code | Référence |
|------|-----------|
| [1]  | FAO. *Crop Production Guidelines and Soil Management in Tropical Regions.* |
| [2]  | IITA. *Crop and Soil Fertility Management Practices in Sub-Saharan Africa.* |
| [3]  | IRAD. *Fiches techniques des cultures vivrières et de rente au Cameroun.* |
| [4]  | CIRAD. *Agronomic Practices in Tropical Agriculture.* |

---

## 2. Dataset Synthétisé — Méthodologie

### 2.1. Principes de sélection des cultures
Seules les cultures **explicitement référencées** dans les quatre sources sont incluses. Toute culture non documentée (ex. : arachides, cultures asiatiques du dataset original, etc.) a été exclue.

### 2.2. Unités
Les macronutriments N, P, K sont exprimés en **mg/kg** (milligrammes par kilogramme de sol = ppm), unité standard utilisée dans les analyses pédologiques et les recommandations de fertilisation de la FAO et de l'IITA.

### 2.3. Cultures incluses (15)

| Catégorie | Cultures |
|---|---|
| Céréales | `riz`, `mais`, `sorgho`, `mil` |
| Légumineuses | `haricot`, `soja`, `niebe` |
| Tubercules / Racines | `manioc`, `igname`, `taro` |
| Cultures d'exportation / rente | `cacao`, `cafe_robusta`, `cafe_arabica`, `palmier_a_huile` |
| Vivrières fruitières | `banane_plantain` |

### 2.4. Méthode de génération
- **Script :** `generate_research_dataset.py` (paramètres issus des références bibliographiques)
- **Volume :** 500 échantillons par culture → **7 500 lignes au total**
- **Tirage :** `numpy.random.uniform` dans des fenêtres agronomiques validées par la littérature
- **Reproductibilité :** `random_state = 42`

---

## 3. Tableau Récapitulatif des Intervalles Agronomiques

Toutes les valeurs de N, P, K sont en **mg/kg**. Température en °C, Humidité en %, Pluviométrie en mm/an.

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

## 4. Entraînement du Modèle — Top-3 Recommandations

### 4.1. Architecture
- **Algorithme :** `RandomForestClassifier` (scikit-learn)
- **`n_estimators` :** 200 arbres
- **`random_state` :** 42 (reproductibilité)
- **`n_jobs` :** -1 (parallélisation maximale)
- **Split train/test :** 80 % / 20 % (stratifié)

### 4.2. Performances

| Métrique | Valeur |
|---|---|
| Validation croisée (10-fold) | **86.20 % ± 0.91 %** |
| Accuracy sur l'ensemble de test | **85.93 %** |

> **Note :** Ces performances reflètent la réalité agronomique. Certaines cultures partagent des conditions agroécologiques similaires (ex. : banane\_plantain, taro, palmier à huile en zone humide). L'accuracy de 86 % traduit donc une discrimination correcte malgré ce chevauchement naturel — le modèle est robuste et non surajusté.

### 4.3. Pipeline de Prédiction Top-3
La prédiction s'effectue via `predict_proba` : pour un profil de sol donné, le modèle calcule la probabilité de chaque culture et retourne les **3 cultures les plus probables**, triées par ordre décroissant de confiance.

```python
# Exemple d'utilisation
soil_profile = {
    'N': 120, 'P': 45, 'K': 180,    # mg/kg
    'temperature': 27, 'humidity': 82,
    'ph': 6.0, 'rainfall': 2200,     # mm/an
}
top3 = predict_top3(model, label_encoder, soil_profile)
# → [{'rang': 1, 'culture': 'banane_plantain', 'probabilite': '52.3 %'}, ...]
```

### 4.4. Exemples de Recommandations

| Sol | Zone Cameroun | Recommandation #1 | Recommandation #2 | Recommandation #3 |
|---|---|---|---|---|
| Fertile humide (N=120, P=45, K=180, pH=6.0, 2200 mm) | Littoral / Sud | banane\_plantain | manioc | taro |
| Sahélien pauvre (N=30, P=10, K=25, pH=7.2, 450 mm) | Extrême-Nord | sorgho | mil | niebe |
| Hautes terres (N=85, P=30, K=70, pH=5.8, 1400 mm) | Ouest / Nord-Ouest | mais | haricot | soja |

---

## 5. Fichiers du Projet

| Fichier | Description |
|---|---|
| `generate_research_dataset.py` | Script de génération du dataset (paramètres FAO/IITA/IRAD/CIRAD) |
| `synthesized_research_data.ipynb` | Notebook EDA : distributions, heatmaps, tableau récapitulatif |
| `train_top3_model.ipynb` | Notebook d'entraînement, évaluation et prédiction Top-3 |
| `data/research_based_dataset.csv` | Dataset synthétisé (7 500 lignes, 15 cultures, mg/kg) |
| `model/top3_crop_model.pkl` | Bundle du modèle (RF + LabelEncoder + métriques + références) |

---

## 6. Références Bibliographiques

- **[1]** FAO. *Crop Production Guidelines and Soil Management in Tropical Regions.* Food and Agriculture Organization of the United Nations, Rome.
- **[2]** IITA. *Crop and Soil Fertility Management Practices in Sub-Saharan Africa.* International Institute of Tropical Agriculture, Ibadan.
- **[3]** IRAD. *Fiches techniques des cultures vivrières et de rente au Cameroun.* Institut de Recherche Agricole pour le Développement, Yaoundé.
- **[4]** CIRAD. *Agronomic Practices in Tropical Agriculture.* Centre de coopération internationale en recherche agronomique pour le développement, Montpellier.
