# Rapport — Dataset Synthétisé pour la Recommandation de Cultures
## Afrique sub-saharienne / Cameroun — Version 2.0 · Février 2026

**Auteur :** Bala Andegue François-Lionnel  
**Dépôt :** [BalaAndegue/crops\_predictions](https://github.com/BalaAndegue/crops_predictions)

---

## 1. Contexte et Objectif

Le dataset Kaggle original (*Crop Recommendation Dataset*, 2200 lignes, 22 cultures) présente deux limitations majeures pour une application en contexte camerounais / africain :

1. **Unités ambiguës** : N, P, K en ratios sans dimension au lieu de mg/kg
2. **Cultures non pertinentes** : pomme, raisin, papaye, mangue, noix de coco — absentes du contexte sub-saharien

Ce dataset synthétisé répond à ces limitations en s'appuyant exclusivement sur **4 références scientifiques primaires** (FAO, IITA, IRAD, CIRAD).

---

## 2. Sources Bibliographiques

| Code | Organisation | Référence complète |
|------|---|---|
| [1] | FAO | *Crop Production Guidelines and Soil Management in Tropical Regions.* Food and Agriculture Organization of the United Nations, Rome. |
| [2] | IITA | *Crop and Soil Fertility Management Practices in Sub-Saharan Africa.* International Institute of Tropical Agriculture, Ibadan. |
| [3] | IRAD | *Fiches techniques des cultures vivrières et de rente au Cameroun.* Institut de Recherche Agricole pour le Développement, Yaoundé. |
| [4] | CIRAD | *Agronomic Practices in Tropical Agriculture.* Centre de coopération internationale en recherche agronomique pour le développement, Montpellier. |

---

## 3. Caractéristiques du Dataset

| Paramètre | Valeur |
|---|---|
| Fichier | `data/research_based_dataset.csv` |
| Nombre de lignes | **7 500** |
| Nombre de cultures | **15** |
| Échantillons par culture | 500 |
| Script de génération | `generate_research_dataset.py` |
| Méthode de tirage | `numpy.random.uniform` dans les fenêtres agronomiques |
| Reproductibilité | `random_state = 42` |

### Variables (colonnes)

| Variable | Unité | Description |
|---|---|---|
| `N` | mg/kg | Azote assimilable |
| `P` | mg/kg | Phosphore assimilable |
| `K` | mg/kg | Potassium assimilable |
| `temperature` | °C | Température moyenne annuelle |
| `humidity` | % | Humidité relative moyenne |
| `ph` | — | pH du sol |
| `rainfall` | mm/an | Pluviométrie annuelle |
| `label` | — | Culture cible (15 classes) |

### Statistiques globales

| Variable | Min | Moy | Max |
|---|---|---|---|
| N (mg/kg) | 15.0 | 86.0 | 199.8 |
| P (mg/kg) | 5.0 | ~35.0 | ~80.0 |
| K (mg/kg) | 10.0 | ~80.0 | 280.0 |
| Température (°C) | 15.0 | ~27.0 | 40.0 |
| Humidité (%) | 25.0 | ~72.0 | 95.0 |
| pH | 4.5 | ~6.3 | 8.5 |
| Pluviométrie (mm) | 200.9 | 1 527.1 | 3 998.1 |

---

## 4. Cultures Incluses et Intervalles Agronomiques

### 4.1. Liste des 15 cultures

| Catégorie | Cultures |
|---|---|
| Céréales | `riz`, `mais`, `sorgho`, `mil` |
| Légumineuses | `haricot`, `soja`, `niebe` |
| Tubercules / Racines | `manioc`, `igname`, `taro` |
| Cultures de rente / export | `cacao`, `cafe_robusta`, `cafe_arabica`, `palmier_a_huile` |
| Vivrières fruitières | `banane_plantain` |

### 4.2. Fenêtres agronomiques validées

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

## 5. Utilisation du Dataset

### Chargement rapide

```python
import pandas as pd
df = pd.read_csv("data/research_based_dataset.csv")
print(df.shape)        # (7500, 8)
print(df['label'].value_counts())  # 500 par culture
```

### Features et target

```python
FEATURES = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
X = df[FEATURES]
y = df['label']
```

### Entraînement du modèle Top-3

Voir `train_top3_model.ipynb` et le bundle sauvegardé :  
`model/top3_crop_model.pkl` — contient : modèle RF, classes, feature\_names, métriques CV, références.

---

## 6. Métriques du Modèle Entraîné

| Métrique | Valeur |
|---|---|
| Validation croisée 10-fold — moyenne | **86.20 %** |
| Validation croisée 10-fold — écart-type | **± 0.91 %** |
| Accuracy sur le jeu de test (20 %) | **85.93 %** |
| Algorithme | Random Forest — 200 arbres |
| Stratégie Top-3 | `predict_proba` → 3 meilleures probabilités |

---

## 7. Notes sur la Qualité

- **Chevauchement naturel** : Banane plantain, taro et palmier à huile partagent des zones humides — une légère confusion entre eux est attendue et normale.
- **Généralisation** : Le modèle est entraîné sur des fenêtres agronomiques théoriques. Pour une application terrain, il est recommandé de compléter avec des données de mesures réelles.
- **Équilibre des classes** : Parfait (500 échantillons par culture × 15 = 7 500).
