import streamlit as st

# MUST be the very first Streamlit command in the script
st.set_page_config(page_title="Retail AI Assistant", layout="centered")

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity


gradient_bg = """
<style>
    .stApp {
        background: linear-gradient(180deg, #E6E6FA, #FFE4E1);
        /* Lavender to very light pink */
        height: 100vh;
    }
</style>
"""

st.markdown(gradient_bg, unsafe_allow_html=True)

# --- Load Models and Data ---
@st.cache_data
def load_similarity_matrix():
    return pd.read_csv("item_similarity.csv", index_col=0)

@st.cache_resource
def load_models():
    kmeans = joblib.load("kmeans_model.pkl")
    scaler = joblib.load("rfm_scaler.pkl")
    return kmeans, scaler

item_similarity = load_similarity_matrix()
kmeans, scaler = load_models()

# --- App Layout ---
st.title("🛍️ Retail Intelligence App")

tab1, tab2 = st.tabs(["🎯 Product Recommendations", "🧍 Customer Segmentation"])

# --- Product Recommendation Module ---
with tab1:
    st.header("🎯 Product Recommendation Engine")

    product_input = st.text_input("Enter a Product Name (case-sensitive):")

    if st.button("Get Recommendations"):
        if product_input.strip() == "":
            st.warning("Please enter a product name.")
        elif product_input not in item_similarity.index:
            st.error("Product not found. Try a different name.")
        else:
            st.subheader(f"📦 Products similar to: `{product_input}`")
            recommendations = item_similarity.loc[product_input].sort_values(ascending=False).iloc[1:6]
            for i, rec in enumerate(recommendations.index, 1):
                st.markdown(f"{i}. {rec}")

# --- Customer Segmentation Module ---
with tab2:
    st.header("🧍 Customer Segmentation Predictor")

    recency = st.number_input("📅 Recency (days since last purchase)", min_value=0, step=1)
    frequency = st.number_input("🔁 Frequency (number of purchases)", min_value=0, step=1)
    monetary = st.number_input("💰 Monetary (total spend)", min_value=0.0, step=1.0)

    if st.button("Predict Cluster"):
        input_rfm = np.array([[recency, frequency, monetary]])
        input_scaled = scaler.transform(input_rfm)
        cluster = kmeans.predict(input_scaled)[0]

        # Simple cluster labeling logic (adjust per your analysis)
        def label_cluster(r, f, m):
            if r < 90 and f > 10 and m > 1000:
                return "High-Value"
            elif f > 5 and m > 500:
                return "Regular"
            elif r > 180 and f < 2:
                return "At-Risk"
            else:
                return "Occasional"

        label = label_cluster(recency, frequency, monetary)

        st.success(f"📊 Predicted Segment: **{label}** (Cluster {cluster})")