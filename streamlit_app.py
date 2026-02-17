# streamlit_app.py
import streamlit as st
import pandas as pd
import joblib
from collections import Counter
import numpy as np

# ========================= CONFIG =========================
st.set_page_config(
    page_title="Crop Recommendation Cameroun 🌱",
    page_icon="🌾",
    layout="centered"
)

# ========================= CHARGEMENT MODÈLE =========================
@st.cache_resource
def load_model():
    model = joblib.load("model/crop_recommendation_model_26crops.pkl")
    le = joblib.load("model/label_encoder_26crops.pkl")
    return model, le

model, le = load_model()

# ========================= TITRE =========================
st.title("🌾 Crop Recommendation System")
st.markdown("### Prédiction par **lots de 1 à 10 échantillons** → Vote majoritaire")

st.info("Entrez les paramètres du sol et du climat. L'application retourne la culture la plus probable selon le vote majoritaire.")

# ========================= FORMULAIRE DYNAMIQUE =========================
with st.form(key="batch_form"):
    st.subheader("Ajouter des échantillons (max 10)")

    # Nombre d'échantillons
    num_samples = st.slider("Nombre d'échantillons à tester", 1, 10, 5, help="Vous pouvez en ajouter/supprimer dynamiquement")

    samples = []
    for i in range(num_samples):
        st.markdown(f"#### Échantillon {i+1}")
        cols = st.columns(7)
        with cols[0]:
            N = st.number_input(f"N (Azote)", min_value=0.0, max_value=200.0, value=90.0, step=1.0, key=f"N_{i}")
        with cols[1]:
            P = st.number_input(f"P (Phosphore)", min_value=0.0, max_value=200.0, value=42.0, step=1.0, key=f"P_{i}")
        with cols[2]:
            K = st.number_input(f"K (Potassium)", min_value=0.0, max_value=200.0, value=43.0, step=1.0, key=f"K_{i}")
        with cols[3]:
            temp = st.number_input(f"Température (°C)", min_value=0.0, max_value=50.0, value=20.0, step=0.5, key=f"temp_{i}")
        with cols[4]:
            hum = st.number_input(f"Humidité (%)", min_value=0.0, max_value=100.0, value=82.0, step=1.0, key=f"hum_{i}")
        with cols[5]:
            ph = st.number_input(f"pH", min_value=0.0, max_value=14.0, value=6.5, step=0.1, key=f"ph_{i}")
        with cols[6]:
            rain = st.number_input(f"Pluviométrie (mm)", min_value=0.0, max_value=500.0, value=200.0, step=5.0, key=f"rain_{i}")

        samples.append({
            "N": N, "P": P, "K": K,
            "temperature": temp,
            "humidity": hum,
            "ph": ph,
            "rainfall": rain
        })

    submit_button = st.form_submit_button("🚀 Lancer la prédiction par vote majoritaire")

# ========================= PRÉDICTION =========================
if submit_button:
    with st.spinner("Prédiction en cours..."):
        # Conversion en DataFrame
        df_batch = pd.DataFrame(samples)
        
        # ✅ ORDRE EXACT DU MODÈLE (comme dans batch_processor)
        MODEL_FEATURES = ['N', 'P', 'K', 'ph', 'humidity', 'temperature']
        X_batch = df_batch[MODEL_FEATURES]

        # Prédictions individuelles
        preds_encoded = model.predict(X_batch)
        preds_labels = le.inverse_transform(preds_encoded)

        # Vote majoritaire
        vote = Counter(preds_labels)
        winner = vote.most_common(1)[0][0]
        confidence = vote[winner] / len(preds_labels)

        # ========================= AFFICHAGE RÉSULTAT =========================
        st.success("✅ Prédiction terminée !")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Culture recommandée", winner.upper())
        with col2:
            st.metric("Confiance (vote majoritaire)", f"{confidence:.1%}")
        with col3:
            st.metric("Nombre d'échantillons", len(samples))

        st.markdown("---")
        st.subheader("Détail des votes")
        
        vote_df = pd.DataFrame([
            {"Culture": crop, "Votes": count, "Pourcentage": f"{count/len(samples):.1%}"}
            for crop, count in vote.most_common()
        ])
        vote_df = vote_df.sort_values("Votes", ascending=False)
        
        st.dataframe(vote_df, use_container_width=True)

        # Bar chart des votes
        st.bar_chart(vote_df.set_index("Culture")["Votes"])

        # Tableau complet des prédictions individuelles
        st.subheader("Prédictions individuelles")
        result_df = df_batch.copy()
        result_df["Prédiction"] = preds_labels
        result_df.index = [f"Échantillon {i+1}" for i in range(len(result_df))]
        st.dataframe(result_df.style.apply(lambda x: ['background: lightgreen' if x["Prédiction"] == winner else '' for i in x], axis=1))

    st.balloons()

# ========================= SIDEBAR =========================
with st.sidebar:
    st.header("À propos")
    st.write("""
    - Modèle : Random Forest optimisé
    - 8 cultures spécifiques au Cameroun
    - Stratégie : **Vote majoritaire** sur max 10 échantillons
    - Features : N, P, K, pH, humidité, température
    """)
    
    st.markdown("### Cultures possibles")
    crops = sorted(le.classes_)
    for crop in crops:
        st.write(f"• {crop.capitalize()}")

    st.markdown("---")
    st.markdown("Made with ❤️ by Grok")

# ========================= FIN =========================
st.markdown("---")
st.caption("Déploie cette app en 1 clic sur Streamlit Community Cloud, Hugging Face Spaces, ou Railway !")