# streamlit_app.py — Crop Recommendation Top-3 (Cameroun / Afrique sub-saharienne)
# ==============================================================================
# Pour chaque échantillon de sol : Top-3 cultures + niveau de confiance (probas RF)
# Pour le lot complet            : Top-3 agrégé (moyenne des probabilités sur N)
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from collections import Counter

# ─── Config page ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🌿 Recommandation de Cultures — Cameroun",
    page_icon="🌾",
    layout="wide",
)

# ─── CSS personnalisé ─────────────────────────────────────────────────────────
st.markdown("""
<style>
.top3-card {
    background: #1e2a1e;
    border-left: 5px solid #4caf50;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 10px;
}
.rank1 { border-left-color: #ffd700; }
.rank2 { border-left-color: #c0c0c0; }
.rank3 { border-left-color: #cd7f32; }
.culture-name { font-size: 1.1em; font-weight: 700; color: #f0f0f0; }
.confidence    { font-size: 0.95em; color: #a5d6a7; }
.global-box {
    background: #102010;
    border: 2px solid #4caf50;
    border-radius: 10px;
    padding: 20px;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)


# ─── Chargement du modèle ────────────────────────────────────────────────────
@st.cache_resource
def load_bundle():
    with open("model/top3_crop_model.pkl", "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], list(bundle["classes"])

try:
    model, classes = load_bundle()
except FileNotFoundError:
    st.error("❌ Modèle introuvable : `model/top3_crop_model.pkl`.\n\n"
             "Veuillez d'abord entraîner le modèle via `train_top3_model.ipynb`.")
    st.stop()

FEATURE_NAMES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

# ─── Helpers ─────────────────────────────────────────────────────────────────

def top3_for_sample(proba: np.ndarray) -> list:
    """Retourne le Top-3 (rang, culture, confiance %) trié par confiance décroissante."""
    idx = np.argsort(proba)[::-1][:3]
    return [
        {
            "rang":      i + 1,
            "culture":   classes[j],
            "confiance": round(float(proba[j]) * 100, 1),
        }
        for i, j in enumerate(idx)
    ]


def aggregate_top3(all_probas: np.ndarray) -> list:
    """Agrège les probabilités (moyenne) → Top-3 global du lot."""
    mean_proba = all_probas.mean(axis=0)
    idx        = np.argsort(mean_proba)[::-1][:3]
    return [
        {
            "rang":              i + 1,
            "culture":           classes[j],
            "confiance_agregee": round(float(mean_proba[j]) * 100, 1),
        }
        for i, j in enumerate(idx)
    ]


MEDALS = {1: "🥇", 2: "🥈", 3: "🥉"}
RANK_CSS = {1: "rank1", 2: "rank2", 3: "rank3"}


def render_top3_cards(top3: list):
    for item in top3:
        rang = item["rang"]
        css  = RANK_CSS.get(rang, "")
        st.markdown(
            f'<div class="top3-card {css}">'
            f'<span class="culture-name">{MEDALS.get(rang, rang)} {item["culture"].replace("_", " ").title()}</span>&nbsp;&nbsp;'
            f'<span class="confidence">Confiance : <strong>{item["confiance"]}%</strong></span>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ─── Titre ────────────────────────────────────────────────────────────────────
st.title("🌾 Recommandation de Cultures — Top-3")
st.markdown(
    "### Pour **chaque échantillon de sol** → Top-3 cultures adaptées + confiance  \n"
    "### Pour le **lot complet (N vecteurs)** → Top-3 agrégé par moyenne des probabilités"
)
st.info(
    "📐 Unités : **N, P, K en mg/kg** | **Température en °C** | "
    "**Humidité en %** | **pH** | **Pluviométrie en mm/an**"
)

# ─── Formulaire ───────────────────────────────────────────────────────────────
with st.form(key="batch_form"):
    st.subheader("Paramètres des échantillons de sol")
    num_samples = st.slider(
        "Nombre d'échantillons (vecteurs de caractéristiques)",
        min_value=1, max_value=10, value=3,
        help="Chaque ligne correspond à un vecteur de caractéristiques de sol indépendant."
    )

    # En-têtes du tableau manuel
    header_cols = st.columns([1, 1, 1, 1, 1, 1, 1])
    labels = ["N (mg/kg)", "P (mg/kg)", "K (mg/kg)", "Temp (°C)", "Humidité (%)", "pH", "Pluie (mm)"]
    for col, lbl in zip(header_cols, labels):
        col.markdown(f"**{lbl}**")

    samples = []
    for i in range(num_samples):
        cols = st.columns([1, 1, 1, 1, 1, 1, 1])
        N    = cols[0].number_input(f"N_{i}",    min_value=0.0,   max_value=500.0,  value=90.0,  step=1.0,  key=f"N_{i}",    label_visibility="collapsed")
        P    = cols[1].number_input(f"P_{i}",    min_value=0.0,   max_value=500.0,  value=42.0,  step=1.0,  key=f"P_{i}",    label_visibility="collapsed")
        K    = cols[2].number_input(f"K_{i}",    min_value=0.0,   max_value=500.0,  value=43.0,  step=1.0,  key=f"K_{i}",    label_visibility="collapsed")
        temp = cols[3].number_input(f"temp_{i}", min_value=-10.0, max_value=50.0,   value=20.0,  step=0.5,  key=f"temp_{i}", label_visibility="collapsed")
        hum  = cols[4].number_input(f"hum_{i}",  min_value=0.0,   max_value=100.0,  value=82.0,  step=1.0,  key=f"hum_{i}",  label_visibility="collapsed")
        ph   = cols[5].number_input(f"ph_{i}",   min_value=0.0,   max_value=14.0,   value=6.5,   step=0.1,  key=f"ph_{i}",   label_visibility="collapsed")
        rain = cols[6].number_input(f"rain_{i}", min_value=0.0,   max_value=5000.0, value=200.0, step=5.0,  key=f"rain_{i}", label_visibility="collapsed")
        samples.append({
            "N": N, "P": P, "K": K,
            "temperature": temp, "humidity": hum,
            "ph": ph, "rainfall": rain,
        })

    submitted = st.form_submit_button("🚀 Lancer la prédiction Top-3", use_container_width=True)

# ─── Prédiction ───────────────────────────────────────────────────────────────
if submitted:
    with st.spinner("Calcul des probabilités RF en cours..."):
        df         = pd.DataFrame(samples)
        X_batch    = df[FEATURE_NAMES].values
        all_probas = model.predict_proba(X_batch)   # (n_samples, n_classes)

        # Top-3 par échantillon
        per_sample = [top3_for_sample(proba) for proba in all_probas]

        # Top-3 agrégé (moyenne)
        top3_global = aggregate_top3(all_probas)

    st.success(f"✅ {len(samples)} échantillon(s) traité(s).")

    # ── Résultat global ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="global-box">', unsafe_allow_html=True)
    st.subheader("🌍 Top-3 Global (agrégé sur l'ensemble du lot)")
    st.caption(
        f"Agrégation par **moyenne des probabilités RF** sur les **{len(samples)} vecteur(s)**."
    )

    gcols = st.columns(3)
    for item in top3_global:
        rang = item["rang"]
        with gcols[rang - 1]:
            st.metric(
                label=f"{MEDALS.get(rang, rang)} Rang {rang}",
                value=item["culture"].replace("_", " ").title(),
                delta=f"Confiance agrégée : {item['confiance_agregee']}%",
            )
    st.markdown('</div>', unsafe_allow_html=True)

    # Graphique des probabilités agrégées (tous les crops)
    st.markdown("#### Distribution des probabilités moyennes (toutes cultures)")
    mean_proba = all_probas.mean(axis=0)
    proba_df   = pd.DataFrame({
        "Culture":   [c.replace("_", " ").title() for c in classes],
        "Confiance (%)": np.round(mean_proba * 100, 2),
    }).sort_values("Confiance (%)", ascending=False)
    st.bar_chart(proba_df.set_index("Culture")["Confiance (%)"], height=300)

    # ── Résultats par échantillon ────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📋 Résultats par échantillon (Top-3 individuel)")

    for i, (sample, top3) in enumerate(zip(samples, per_sample)):
        with st.expander(f"Échantillon {i+1}  —  Top-1 : **{top3[0]['culture'].replace('_',' ').title()}** ({top3[0]['confiance']}%)", expanded=(len(samples) <= 3)):
            ecol1, ecol2 = st.columns([1, 1])
            with ecol1:
                render_top3_cards(top3)
            with ecol2:
                mini_df = pd.DataFrame(top3).rename(columns={
                    "rang": "Rang", "culture": "Culture", "confiance": "Confiance (%)"
                })
                mini_df["Culture"] = mini_df["Culture"].str.replace("_", " ").str.title()
                st.dataframe(mini_df.set_index("Rang"), use_container_width=True)
                # Mini bar chart confiance
                st.bar_chart(mini_df.set_index("Culture")["Confiance (%)"], height=150)

    # ── Tableau récapitulatif ────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📊 Tableau récapitulatif")
    rows = []
    for i, (sample, top3) in enumerate(zip(samples, per_sample)):
        row = {
            "Échantillon": f"#{i+1}",
            "N": sample["N"], "P": sample["P"], "K": sample["K"],
            "Temp": sample["temperature"], "Hum": sample["humidity"],
            "pH": sample["ph"], "Pluie": sample["rainfall"],
            "Top-1": f"{top3[0]['culture']} ({top3[0]['confiance']}%)",
            "Top-2": f"{top3[1]['culture']} ({top3[1]['confiance']}%)" if len(top3) > 1 else "—",
            "Top-3": f"{top3[2]['culture']} ({top3[2]['confiance']}%)" if len(top3) > 2 else "—",
        }
        rows.append(row)
    recap_df = pd.DataFrame(rows).set_index("Échantillon")
    st.dataframe(recap_df, use_container_width=True)

    st.balloons()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🌿 À propos")
    st.markdown("""
- **Modèle** : Random Forest (top3_crop_model.pkl)
- **Cultures couvertes** : Cameroun / Afrique sub-saharienne
- **Features** : N, P, K, Température, Humidité, pH, Pluviométrie
- **Stratégie par échantillon** : Top-3 via `predict_proba`
- **Stratégie globale** : Moyenne des probabilités RF sur le lot
    """)

    st.markdown("### 🌱 Cultures disponibles")
    for c in sorted(classes):
        st.write(f"• {c.replace('_', ' ').title()}")

    st.markdown("---")
    st.caption("Crop Recommendation System v2.0 — Bala Andegue")