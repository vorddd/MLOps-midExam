from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

# ==== Model configuration ====
# Local path (used for CI tests & when you commit the artifact)
LOCAL_MODEL_PATH = Path(__file__).resolve().parent / "best_model_pipeline.joblib"

# Model repo on Hugging Face Hub (fallback when local file is not available)
MODEL_REPO_ID = "vorddd/shipping-delay-knn-v1"
MODEL_FILENAME = "best_model_pipeline.joblib"

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
    """
    Load the trained model pipeline.

    Priority:
    1. If LOCAL_MODEL_PATH exists -> use that (for unit tests & local dev).
    2. Otherwise -> download from Hugging Face Hub (for Spaces).
    """
    if LOCAL_MODEL_PATH.exists():
        model_path = LOCAL_MODEL_PATH
    else:
        model_path = hf_hub_download(
            repo_id=MODEL_REPO_ID,
            filename=MODEL_FILENAME,
            repo_type="model",
        )

    model = joblib.load(model_path)
    return model


def _get_feature_ranges(data: pd.DataFrame) -> dict:
    """Get min, max, median for numeric features to build friendly sliders."""
    ranges = {}
    for column in FEATURE_ORDER[:-1]:  # numeric only, last one is categorical
        series = data[column]
        ranges[column] = (
            int(series.min()),
            int(series.max()),
            int(series.median()),
        )
    return ranges


def model_page(reference_data: Optional[pd.DataFrame] = None) -> None:
    st.header("Shipment Delay Prediction")

    st.write(
        "Use this tool to estimate **whether a shipment is likely to arrive on time or late** "
        "based on key business inputs such as product cost, discount, and customer history."
    )
    st.caption(
        "Fill in the form below with realistic values. "
        "The model will return a prediction and the estimated probabilities."
    )

    if reference_data is None:
        raise ValueError("reference_data is required to build sensible input ranges")

    # Build slider ranges from real data so the UI feels realistic
    feature_ranges = _get_feature_ranges(reference_data)
    product_options = sorted(reference_data["Product_importance"].unique())

    with st.form("prediction_form"):
        st.subheader("Shipment details")

        col1, col2 = st.columns(2)

        customer_care_calls = col1.slider(
            "Customer care calls",
            min_value=feature_ranges["Customer_care_calls"][0],
            max_value=feature_ranges["Customer_care_calls"][1],
            value=feature_ranges["Customer_care_calls"][2],
            help="How many times this customer contacted customer service about this order.",
        )
        cost_of_product = col2.slider(
            "Cost of the product",
            min_value=feature_ranges["Cost_of_the_Product"][0],
            max_value=feature_ranges["Cost_of_the_Product"][1],
            value=feature_ranges["Cost_of_the_Product"][2],
            help="Total product cost. Higher-value items may be treated differently in operations.",
        )

        prior_purchases = col1.slider(
            "Prior purchases",
            min_value=feature_ranges["Prior_purchases"][0],
            max_value=feature_ranges["Prior_purchases"][1],
            value=feature_ranges["Prior_purchases"][2],
            help="How many times this customer has purchased before.",
        )
        discount_offered = col2.slider(
            "Discount offered (%)",
            min_value=feature_ranges["Discount_offered"][0],
            max_value=feature_ranges["Discount_offered"][1],
            value=feature_ranges["Discount_offered"][2],
            help="Discount given for this order, in percent.",
        )

        weight_in_gms = st.slider(
            "Product weight (grams)",
            min_value=feature_ranges["Weight_in_gms"][0],
            max_value=feature_ranges["Weight_in_gms"][1],
            value=feature_ranges["Weight_in_gms"][2],
            help="Heavier products may take more time to handle and ship.",
        )

        product_importance = st.selectbox(
            "Product importance",
            options=product_options,
            help="Business importance of the product (for example: low, medium, high).",
        )

        submitted = st.form_submit_button("Predict shipment status")

    if not submitted:
        st.info("Fill in the shipment details and click **Predict shipment status**.")
        return

    # Build feature vector in the same order used during training
    features = pd.DataFrame(
        [[
            customer_care_calls,
            cost_of_product,
            prior_purchases,
            discount_offered,
            weight_in_gms,
            product_importance,
        ]],
        columns=FEATURE_ORDER,
    )

    model = load_model()
    prediction_raw = model.predict(features)[0]
    is_on_time = prediction_raw == 1

    # Try to get probability if the model supports it
    on_time_prob = None
    late_prob = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)[0]
        # assuming class 0 = Late, class 1 = On Time
        late_prob = float(proba[0])
        on_time_prob = float(proba[1])

    st.subheader("Prediction result")

    if is_on_time:
        st.success("This shipment is **predicted to arrive ON TIME**.")
    else:
        st.error("This shipment is **predicted to be LATE**.")

    if on_time_prob is not None and late_prob is not None:
        c1, c2 = st.columns(2)
        c1.metric("On-time probability", f"{on_time_prob * 100:.1f}%")
        c2.metric("Late probability", f"{late_prob * 100:.1f}%")

        st.caption(
            "Probabilities are model estimates based on historical data. "
            "They are not guarantees, but they can help you prioritize risky shipments."
        )
    else:
        st.caption(
            "This model does not expose class probabilities. "
            "Only the predicted class (On Time / Late) is shown."
        )

    st.markdown("---")
    st.markdown("### Input summary")
    st.write(
        "These are the values you entered. You can tweak them and run the prediction again "
        "to see how the risk changes for different shipment profiles."
    )
    st.dataframe(features, use_container_width=True)
