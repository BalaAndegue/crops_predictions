import pandas as pd
from collections import Counter

def analyze_target_variable(csv_file, target_column):
    """
    Analyse complète de la variable cible
    """
    df = pd.read_csv(csv_file)
    
    if target_column not in df.columns:
        print(f"Colonne '{target_column}' non trouvée. Colonnes disponibles: {list(df.columns)}")
        return
    
    target_data = df[target_column]
    
    # Informations de base
    classes = set(target_data.unique())
    count_per_class = Counter(target_data)
    total_samples = len(target_data)
    
    print("=" * 50)
    print(f"ANALYSE DE LA VARIABLE: '{target_column}'")
    print("=" * 50)
    
    print(f"\nClasses uniques ({len(classes)}):")
    print("-" * 30)
    
    for i, classe in enumerate(sorted(classes), 1):
        count = count_per_class[classe]
        percentage = (count / total_samples) * 100
        print(f"{i:2d}. {classe:15} | {count:5d} échantillons | {percentage:5.1f}%")
    
    print(f"\nRésumé:")
    print(f"- Total d'échantillons: {total_samples}")
    print(f"- Nombre de classes: {len(classes)}")
    print(f"- Type de données: {target_data.dtype}")
    
    # Vérifier les valeurs manquantes
    missing_values = target_data.isnull().sum()
    if missing_values > 0:
        print(f"- Valeurs manquantes: {missing_values} ({missing_values/total_samples*100:.1f}%)")
    
    return classes

# Utilisation
if __name__ == "__main__":
    fichier_csv = "data/Crop_recommendation.csv"
    colonne_cible = "label"
    
    classes = analyze_target_variable(fichier_csv, colonne_cible)