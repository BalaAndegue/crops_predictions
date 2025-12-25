import pandas as pd
import numpy as np

def generate_cameroon_crops_data():
    """Génère des données pour les cultures spécifiques au Cameroun incluant les arachides"""
    
    # Paramètres climatiques et de sol pour le Cameroun
    cameroon_params = {
        # ARACHIDES - Variétés principales au Cameroun
        'arachide_florida': {'N': (20, 50), 'P': (15, 40), 'K': (25, 60),
                           'temperature': (25, 35), 'humidity': (50, 80),
                           'ph': (5.5, 7.5), 'rainfall': (500, 1200)},
        
        'arachide_47_16': {'N': (25, 55), 'P': (20, 45), 'K': (30, 65),
                          'temperature': (24, 34), 'humidity': (55, 75),
                          'ph': (5.8, 7.2), 'rainfall': (600, 1300)},
        
        'arachide_sodepa': {'N': (22, 52), 'P': (18, 42), 'K': (28, 62),
                           'temperature': (26, 36), 'humidity': (45, 70),
                           'ph': (6.0, 7.8), 'rainfall': (400, 1000)},
        
        'arachide_chalimbana': {'N': (18, 48), 'P': (12, 35), 'K': (22, 58),
                               'temperature': (23, 33), 'humidity': (60, 85),
                               'ph': (5.5, 7.0), 'rainfall': (700, 1500)},
        
        # Cultures de rente
        'cacao': {'N': (60, 120), 'P': (40, 80), 'K': (50, 100), 
                 'temperature': (20, 30), 'humidity': (70, 90), 
                 'ph': (5.5, 7.0), 'rainfall': (1500, 3000)},
        
        'cafe_robusta': {'N': (80, 140), 'P': (30, 60), 'K': (60, 110),
                        'temperature': (22, 28), 'humidity': (60, 80),
                        'ph': (5.0, 6.5), 'rainfall': (1200, 2500)},
        
        'cafe_arabica': {'N': (70, 130), 'P': (25, 50), 'K': (50, 100),
                        'temperature': (15, 24), 'humidity': (50, 70),
                        'ph': (5.5, 6.5), 'rainfall': (1500, 2500)},
        
        # Cultures vivrières
        'banane_plantain': {'N': (100, 180), 'P': (40, 80), 'K': (120, 200),
                           'temperature': (20, 35), 'humidity': (70, 85),
                           'ph': (5.5, 7.5), 'rainfall': (2000, 3500)},
        
        'igname': {'N': (60, 110), 'P': (30, 60), 'K': (80, 150),
                  'temperature': (25, 35), 'humidity': (60, 80),
                  'ph': (5.5, 7.0), 'rainfall': (1000, 2000)},
        
        'taro': {'N': (50, 100), 'P': (25, 50), 'K': (60, 120),
                'temperature': (20, 35), 'humidity': (70, 90),
                'ph': (5.0, 7.0), 'rainfall': (2000, 4000)},
        
        # Légumes et épices
        'piment': {'N': (40, 80), 'P': (30, 60), 'K': (50, 100),
                  'temperature': (20, 30), 'humidity': (50, 70),
                  'ph': (5.5, 7.0), 'rainfall': (600, 1200)},
        
        'aubergine_africaine': {'N': (50, 90), 'P': (25, 50), 'K': (60, 110),
                               'temperature': (22, 32), 'humidity': (60, 80),
                               'ph': (5.5, 7.0), 'rainfall': (800, 1500)},
        
        'haricot_niebe': {'N': (20, 50), 'P': (15, 40), 'K': (20, 60),
                         'temperature': (25, 35), 'humidity': (40, 70),
                         'ph': (5.5, 7.5), 'rainfall': (400, 1200)},
        
        # Céréales
        'sorgho': {'N': (30, 70), 'P': (15, 40), 'K': (25, 60),
                  'temperature': (20, 35), 'humidity': (30, 60),
                  'ph': (5.5, 8.0), 'rainfall': (400, 1000)},
        
        'mil': {'N': (25, 60), 'P': (10, 35), 'K': (20, 50),
               'temperature': (25, 35), 'humidity': (30, 50),
               'ph': (5.5, 8.0), 'rainfall': (300, 800)}
    }
    
    new_data = []
    samples_per_crop = 150  # Nombre d'échantillons par culture
    
    for crop, params in cameroon_params.items():
        for _ in range(samples_per_crop):
            # Ajouter un peu de variabilité pour rendre les données plus réalistes
            sample = {
                'N': np.random.randint(params['N'][0], params['N'][1]),
                'P': np.random.randint(params['P'][0], params['P'][1]),
                'K': np.random.randint(params['K'][0], params['K'][1]),
                'temperature': np.random.uniform(params['temperature'][0], params['temperature'][1]),
                'humidity': np.random.uniform(params['humidity'][0], params['humidity'][1]),
                'ph': np.random.uniform(params['ph'][0], params['ph'][1]),
                'rainfall': np.random.uniform(params['rainfall'][0], params['rainfall'][1]),
                'label': crop
            }
            new_data.append(sample)
    
    return pd.DataFrame(new_data)

# Générer les nouvelles données
print("Génération des données pour les cultures camerounaises...")
cameroon_crops_df = generate_cameroon_crops_data()

print(f"Nouvelles données générées : {len(cameroon_crops_df)} échantillons")
print(f"Cultures ajoutées : {cameroon_crops_df['label'].unique()}")
print(f"Nombre de variétés d'arachides : {len([c for c in cameroon_crops_df['label'].unique() if 'arachide' in c])}")

# Afficher les statistiques pour les arachides
print("\n=== STATISTIQUES DES ARACHIDES ===")
arachides_data = cameroon_crops_df[cameroon_crops_df['label'].str.contains('arachide')]
print(f"Total échantillons arachides : {len(arachides_data)}")
print(f"Variétés d'arachides : {arachides_data['label'].unique()}")

for variete in arachides_data['label'].unique():
    variete_data = arachides_data[arachides_data['label'] == variete]
    print(f"\n{variete}:")
    print(f"  N: {variete_data['N'].mean():.1f} ± {variete_data['N'].std():.1f}")
    print(f"  P: {variete_data['P'].mean():.1f} ± {variete_data['P'].std():.1f}")
    print(f"  K: {variete_data['K'].mean():.1f} ± {variete_data['K'].std():.1f}")
    print(f"  Temp: {variete_data['temperature'].mean():.1f}°C")
    print(f"  Humidité: {variete_data['humidity'].mean():.1f}%")
    print(f"  pH: {variete_data['ph'].mean():.2f}")
    print(f"  Pluie: {variete_data['rainfall'].mean():.0f}mm")

print("\nAperçu des données arachides :")
print(arachides_data.head())

# Sauvegarder les nouvelles données
cameroon_crops_df.to_csv("data/cameroon_crops_data.csv", index=False)
print("\nDonnées sauvegardées dans 'data/cameroon_crops_data.csv'")





# Fusionner avec le dataset existant
print("Chargement du dataset original...")
original_df = pd.read_csv("data/Crop_recommendation.csv")

print(f"Dataset original : {original_df.shape}")
print(f"Cultures originales : {original_df['label'].nunique()}")

# Fusionner les datasets
combined_df = pd.concat([original_df, cameroon_crops_df], ignore_index=True)

print(f"\nDataset combiné : {combined_df.shape}")
print(f"Total des cultures : {combined_df['label'].nunique()}")
print(f"Total des échantillons : {len(combined_df)}")

# Compter les arachides spécifiquement
arachides_count = combined_df[combined_df['label'].str.contains('arachide', na=False)].shape[0]
print(f"Échantillons d'arachides : {arachides_count}")

# Vérifier la distribution
print("\nDistribution des cultures (top 15) :")
print(combined_df['label'].value_counts().head(15))

# Sauvegarder le dataset combiné
combined_df.to_csv("data/combined_crop_recommendation.csv", index=False)
print("\nDataset combiné sauvegardé")