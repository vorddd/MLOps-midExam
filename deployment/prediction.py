from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_MODEL_PATH = PROJECT_ROOT / "models" / "best_model_pipeline.joblib"
DATA_PATH = Path(__file__).resolve().parent / "shipping.csv"
HF_REPO_ID = "vorddd/shipping-delay-knn"
HF_MODEL_FILENAME = "best_model_pipeline.joblib"


@st.cache_resource(show_spinner=False)
def load_model():
    if LOCAL_MODEL_PATH.exists():
        return joblib.load(LOCAL_MODEL_PATH)

    model_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_MODEL_FILENAME,
    )
    return joblib.load(model_path)


def _get_feature_ranges(data: pd.DataFrame) -> dict:
    ranges = {}
    for column in [
        "Customer_care_calls",
        "Cost_of_the_Product",
        "Prior_purchases",
        "Discount_offered",
        "Weight_in_gms",
    ]:
        series = data[column]
        ranges[column] = (
            int(series.min()),
            int(series.max()),
            int(series.median()),
        )
    return ranges


def model_page(reference_data: Optional[pd.DataFrame] = None) -> None:
    st.header("Model Prediction")
    st.write(
        "Isi form di bawah untuk melihat probabilitas paket sampai tepat waktu. "
        "Antarmuka sudah disederhanakan agar nyaman digunakan pada Hugging Face Spaces."
    )

    if reference_data is None:
        reference_data = pd.read_csv(DATA_PATH)

    feature_ranges = _get_feature_ranges(reference_data)
    product_options = sorted(reference_data["Product_importance"].unique())

    with st.form("prediction_form"):
        st.subheader("Masukkan Detail Pengiriman")
        col1, col2 = st.columns(2)

        customer_care_calls = col1.slider(
            "Jumlah Panggilan Customer Care",
            min_value=feature_ranges["Customer_care_calls"][0],
            max_value=feature_ranges["Customer_care_calls"][1],
            value=feature_ranges["Customer_care_calls"][2],
            help="Jumlah interaksi pelanggan sebelum paket dikirim.",
        )
        cost_of_product = col2.slider(
            "Biaya Produk",
            min_value=feature_ranges["Cost_of_the_Product"][0],
            max_value=feature_ranges["Cost_of_the_Product"][1],
            value=feature_ranges["Cost_of_the_Product"][2],
            help="Harga barang dalam USD.",
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
            help="Gunakan slider ini untuk mensimulasikan promo.",
        )

        weight_in_gms = st.slider(
            "Berat Produk (gram)",
            min_value=feature_ranges["Weight_in_gms"][0],
            max_value=feature_ranges["Weight_in_gms"][1],
            value=feature_ranges["Weight_in_gms"][2],
            step=10,
        )
        product_importance = st.selectbox(
            "Pentingnya Produk",
            options=product_options,
        )

        submitted = st.form_submit_button("Prediksi Pengiriman")

    if not submitted:
        st.info("Masukkan parameter dan klik tombol prediksi.")
        return

    features = pd.DataFrame(
        {
            "Customer_care_calls": [customer_care_calls],
            "Cost_of_the_Product": [cost_of_product],
            "Prior_purchases": [prior_purchases],
            "Discount_offered": [discount_offered],
            "Weight_in_gms": [weight_in_gms],
            "Product_importance": [product_importance],
        }
    )

    model = load_model()
    prediction_raw = model.predict(features)[0]
    prediction_label = "Tepat Waktu" if prediction_raw == 1 else "Tidak Tepat Waktu"

    probability = None
    if hasattr(model, "predict_proba"):
        try:
            probability = float(model.predict_proba(features)[0][1])
        except Exception:
            probability = None

    st.subheader("Hasil Prediksi")
    if prediction_label == "Tepat Waktu":
        st.success("Pengiriman diprediksi **tepat waktu**.")
    else:
        st.error("Pengiriman diprediksi **terlambat**.")

    if probability is not None:
        st.metric("Probabilitas Tepat Waktu", f"{probability:.1%}")

    st.write("Detail input yang digunakan:")
    st.dataframe(features, use_container_width=True)

    st.caption(
        "Model menggunakan pipeline yang sama dengan artefak `.joblib` sehingga hasil "
        "di aplikasi konsisten dengan eksperimen offline."
    )
