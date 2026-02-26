"""
Script de Génération du Dataset Synthétisé basé sur des Recherches Scientifiques
====================================================================================
Sources :
    [1] FAO. Crop Production Guidelines and Soil Management in Tropical Regions.
    [2] IITA. Crop and Soil Fertility Management Practices in Sub-Saharan Africa.
    [3] IRAD. Fiches techniques des cultures vivrières et de rente au Cameroun.
    [4] CIRAD. Agronomic Practices in Tropical Agriculture.

Unités : N, P, K en mg/kg (ppm)
Objectif : Dataset orienté Afrique sub-saharienne / Cameroun.
           Seules les cultures référencées dans les 4 sources sont incluses.
"""

import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
#  Graine aléatoire pour la reproductibilité
# ─────────────────────────────────────────────────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ─────────────────────────────────────────────────────────────────────────────
#  Paramètres agronomiques par culture
#  - N, P, K : milligrammes par kilogramme de sol (mg/kg = ppm)
#  - temperature  : °C
#  - humidity     : % humidité relative
#  - ph           : sans dimension
#  - rainfall     : mm/an
#
#  Références :
#  [1] FAO  | [2] IITA | [3] IRAD | [4] CIRAD
# ─────────────────────────────────────────────────────────────────────────────
CROP_PARAMS = {

    # ── Céréales ──────────────────────────────────────────────────────────────
    "riz": {
        # [1][2] Riz pluvial / irrigué tropical.  N min 60–90, P 15–30, K 20–50 mg/kg.
        "N":           (60, 120),
        "P":           (15, 35),
        "K":           (20, 50),
        "temperature": (22, 32),
        "humidity":    (70, 90),
        "ph":          (5.0, 7.0),
        "rainfall":    (1200, 2500),
        "refs":        "[1][2]",
    },
    "mais": {
        # [1][2][3] Maïs tropical/subtropical.  N 80–150, P 20–50, K 30–80 mg/kg.
        "N":           (80, 150),
        "P":           (20, 50),
        "K":           (30, 80),
        "temperature": (18, 32),
        "humidity":    (55, 80),
        "ph":          (5.5, 7.5),
        "rainfall":    (600, 1200),
        "refs":        "[1][2][3]",
    },
    "sorgho": {
        # [1][2] Céréale sèche sahélienne.  N 30–80, P 10–30, K 15–50 mg/kg.
        "N":           (30, 80),
        "P":           (10, 30),
        "K":           (15, 50),
        "temperature": (25, 38),
        "humidity":    (30, 60),
        "ph":          (5.5, 8.0),
        "rainfall":    (300, 900),
        "refs":        "[1][2]",
    },
    "mil": {
        # [1][2] Mil (Pennisetum glaucum).  N 20–70, P 5–25, K 10–40 mg/kg.
        "N":           (20, 70),
        "P":           (5, 25),
        "K":           (10, 40),
        "temperature": (25, 40),
        "humidity":    (25, 55),
        "ph":          (5.5, 8.5),
        "rainfall":    (200, 700),
        "refs":        "[1][2]",
    },

    # ── Légumineuses ──────────────────────────────────────────────────────────
    "haricot": {
        # [1][2][3] Haricot commun (Phaseolus vulgaris).  N 20–60, P 15–40, K 20–60 mg/kg.
        "N":           (20, 60),
        "P":           (15, 40),
        "K":           (20, 60),
        "temperature": (18, 28),
        "humidity":    (50, 75),
        "ph":          (5.5, 7.0),
        "rainfall":    (400, 900),
        "refs":        "[1][2][3]",
    },
    "soja": {
        # [1][2] Soja tropical.  N 25–70, P 20–50, K 30–80 mg/kg.
        "N":           (25, 70),
        "P":           (20, 50),
        "K":           (30, 80),
        "temperature": (22, 32),
        "humidity":    (55, 80),
        "ph":          (5.5, 7.0),
        "rainfall":    (500, 1100),
        "refs":        "[1][2]",
    },
    "niebe": {
        # [1][2][3] Niébé (Vigna unguiculata).  N 15–50, P 10–35, K 15–55 mg/kg.
        "N":           (15, 50),
        "P":           (10, 35),
        "K":           (15, 55),
        "temperature": (24, 36),
        "humidity":    (35, 70),
        "ph":          (5.5, 7.5),
        "rainfall":    (300, 1000),
        "refs":        "[1][2][3]",
    },

    # ── Tubercules / Racines ───────────────────────────────────────────────────
    "manioc": {
        # [2][3][4] Manioc (Manihot esculenta).  N 50–120, P 15–45, K 80–200 mg/kg.
        "N":           (50, 120),
        "P":           (15, 45),
        "K":           (80, 200),
        "temperature": (24, 35),
        "humidity":    (60, 85),
        "ph":          (5.0, 7.0),
        "rainfall":    (800, 2000),
        "refs":        "[2][3][4]",
    },
    "igname": {
        # [2][3] Igname (Dioscorea spp.).  N 60–130, P 20–60, K 80–180 mg/kg.
        "N":           (60, 130),
        "P":           (20, 60),
        "K":           (80, 180),
        "temperature": (24, 34),
        "humidity":    (60, 85),
        "ph":          (5.5, 7.0),
        "rainfall":    (900, 2000),
        "refs":        "[2][3]",
    },
    "taro": {
        # [3][4] Taro (Colocasia esculenta).  N 50–110, P 15–50, K 70–150 mg/kg.
        "N":           (50, 110),
        "P":           (15, 50),
        "K":           (70, 150),
        "temperature": (20, 35),
        "humidity":    (70, 95),
        "ph":          (5.0, 7.0),
        "rainfall":    (1500, 4000),
        "refs":        "[3][4]",
    },

    # ── Cultures d'exportation / rente ────────────────────────────────────────
    "cacao": {
        # [3][4] Cacao (Theobroma cacao).  N 60–140, P 25–70, K 50–120 mg/kg.
        "N":           (60, 140),
        "P":           (25, 70),
        "K":           (50, 120),
        "temperature": (20, 30),
        "humidity":    (70, 90),
        "ph":          (5.5, 7.0),
        "rainfall":    (1500, 3000),
        "refs":        "[3][4]",
    },
    "cafe_robusta": {
        # [3][4] Café robusta (Coffea canephora).  N 80–160, P 20–55, K 60–130 mg/kg.
        "N":           (80, 160),
        "P":           (20, 55),
        "K":           (60, 130),
        "temperature": (22, 30),
        "humidity":    (60, 85),
        "ph":          (5.0, 6.5),
        "rainfall":    (1200, 2500),
        "refs":        "[3][4]",
    },
    "cafe_arabica": {
        # [3][4] Café arabica (Coffea arabica).  N 70–150, P 15–50, K 50–120 mg/kg.
        "N":           (70, 150),
        "P":           (15, 50),
        "K":           (50, 120),
        "temperature": (15, 24),
        "humidity":    (55, 75),
        "ph":          (5.5, 6.5),
        "rainfall":    (1500, 2500),
        "refs":        "[3][4]",
    },
    "palmier_a_huile": {
        # [2][3][4] Palmier à huile (Elaeis guineensis).  N 80–180, P 20–60, K 80–200 mg/kg.
        "N":           (80, 180),
        "P":           (20, 60),
        "K":           (80, 200),
        "temperature": (24, 32),
        "humidity":    (75, 95),
        "ph":          (4.5, 6.5),
        "rainfall":    (1800, 3500),
        "refs":        "[2][3][4]",
    },

    # ── Autres vivrières ──────────────────────────────────────────────────────
    "banane_plantain": {
        # [1][3][4] Banane plantain (Musa paradisiaca).  N 100–200, P 30–80, K 120–280 mg/kg.
        "N":           (100, 200),
        "P":           (30, 80),
        "K":           (120, 280),
        "temperature": (22, 35),
        "humidity":    (70, 90),
        "ph":          (5.5, 7.5),
        "rainfall":    (1800, 3500),
        "refs":        "[1][3][4]",
    },
}

SAMPLES_PER_CROP = 500   # nombre d'échantillons générés par culture


def generate_sample(params: dict, crop_name: str) -> dict:
    """Génère un échantillon pour une culture donnée."""
    return {
        "N":           round(np.random.uniform(params["N"][0],           params["N"][1]),           2),
        "P":           round(np.random.uniform(params["P"][0],           params["P"][1]),           2),
        "K":           round(np.random.uniform(params["K"][0],           params["K"][1]),           2),
        "temperature": round(np.random.uniform(params["temperature"][0], params["temperature"][1]), 2),
        "humidity":    round(np.random.uniform(params["humidity"][0],     params["humidity"][1]),    2),
        "ph":          round(np.random.uniform(params["ph"][0],           params["ph"][1]),          2),
        "rainfall":    round(np.random.uniform(params["rainfall"][0],     params["rainfall"][1]),    2),
        "label":       crop_name,
    }


def generate_dataset(samples_per_crop: int = SAMPLES_PER_CROP) -> pd.DataFrame:
    """Génère le dataset complet.

    Args:
        samples_per_crop: Nombre d'échantillons à générer par culture.

    Returns:
        DataFrame avec colonnes [N, P, K, temperature, humidity, ph, rainfall, label].
    """
    rows = []
    for crop, params in CROP_PARAMS.items():
        for _ in range(samples_per_crop):
            rows.append(generate_sample(params, crop))
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
#  Point d'entrée principal
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("  Génération du Dataset — Sources : FAO / IITA / IRAD / CIRAD")
    print("=" * 70)

    df = generate_dataset()

    print(f"\n✔ Nombre total d'échantillons : {len(df)}")
    print(f"✔ Nombre de cultures         : {df['label'].nunique()}")
    print(f"✔ Cultures incluses          :\n  {list(df['label'].unique())}")

    # Vérifie qu'aucune arachide n'est incluse
    arachides = [c for c in df["label"].unique() if "arachide" in c.lower()]
    assert len(arachides) == 0, f"Arachides inattendues : {arachides}"
    print("\n✔ Vérification : Aucune arachide n'est présente.")

    # Statistiques sommaires pour N, P, K
    print("\n── Statistiques N, P, K (mg/kg) ──")
    print(df[["N", "P", "K"]].describe().round(2).to_string())

    # Sauvegarde
    output_path = "data/research_based_dataset.csv"
    df.to_csv(output_path, index=False)
    print(f"\n✔ Dataset sauvegardé dans '{output_path}'")
