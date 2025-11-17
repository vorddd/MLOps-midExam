from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).resolve().parent / "best_model_pipeline.joblib"

FEATURE_ORDER = [
    "Customer_care_calls",
    "Cost_of_the_Product",
    "Prior_purchases",
    "Discount_offered",
    "Weight_in_gms",
    "Product_importance",
]

@st.cache_resource(show_spinner=False)
def load_model():
    """Load model pipeline yang berisi preprocessing + model."""
    return joblib.load(MODEL_PATH)


def _get_feature_ranges(data: pd.DataFrame) -> dict:
    ranges = {}
    for column in FEATURE_ORDER[:-1]:  # exclude categorical
        series = data[column]
        ranges[column] = (
            int(series.min()),
            int(series.max()),
            int(series.median()),
        )
    return ranges


def model_page(reference_data: Optional[pd.DataFrame] = None) -> None:
    st.header("Model Prediction")

    if reference_data is None:
        raise ValueError("reference_data is required")

    feature_ranges = _get_feature_ranges(reference_data)
    product_options = sorted(reference_data["Product_importance"].unique())

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        customer_care_calls = col1.slider(
            "Jumlah Panggilan Customer Care",
            min_value=feature_ranges["Customer_care_calls"][0],
            max_value=feature_ranges["Customer_care_calls"][1],
            value=feature_ranges["Customer_care_calls"][2],
        )
        cost_of_product = col2.slider(
            "Biaya Produk",
            min_value=feature_ranges["Cost_of_the_Product"][0],
            max_value=feature_ranges["Cost_of_the_Product"][1],
            value=feature_ranges["Cost_of_the_Product"][2],
        )

        prior_purchases = col1.slider(
            "Jumlah Pembelian Sebelumnya",
            min_value=feature_ranges["Prior_purchases"][0],
            max_value=feature_ranges["Prior_purchases"][1],
            value=feature_ranges["Prior_purchases"][2],
        )
        discount_offered = col2.slider(
            "Diskon yang Ditawarkan",
            min_value=feature_ranges["Discount_offered"][0],
            max_value=feature_ranges["Discount_offered"][1],
            value=feature_ranges["Discount_offered"][2],
        )

        weight_in_gms = st.slider(
            "Berat Produk (gram)",
            min_value=feature_ranges["Weight_in_gms"][0],
            max_value=feature_ranges["Weight_in_gms"][1],
            value=feature_ranges["Weight_in_gms"][2],
        )

        product_importance = st.selectbox(
            "Pentingnya Produk",
            options=product_options,
        )

        submitted = st.form_submit_button("Prediksi Pengiriman")

    if not submitted:
        st.info("Masukkan parameter dan klik tombol prediksi.")
        return

    # THIS IS KEY: build DataFrame with EXACT SAME COLUMN ORDER
    features = pd.DataFrame([[ 
        customer_care_calls,
        cost_of_product,
        prior_purchases,
        discount_offered,
        weight_in_gms,
        product_importance,
    ]], columns=FEATURE_ORDER)

    model = load_model()
    prediction_raw = model.predict(features)[0]

    prediction_label = "Tepat Waktu" if prediction_raw == 1 else "Tidak Tepat Waktu"

    st.subheader("Hasil Prediksi")
    if prediction_label == "Tepat Waktu":
        st.success("Pengiriman diprediksi **tepat waktu**.")
    else:
        st.error("Pengiriman diprediksi **terlambat**.")

    st.dataframe(features, use_container_width=True)
