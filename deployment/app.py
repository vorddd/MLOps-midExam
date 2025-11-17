from pathlib import Path

import pandas as pd
import streamlit as st

from deployment.eda import eda_page
from deployment.prediction import model_page

st.set_page_config(
    page_title="Shipping Service Monitor",
    page_icon=":package:",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    """Read the dataset packaged with the deployment bundle."""
    return pd.read_csv(BASE_DIR / "shipping.csv")


def render_overview(data: pd.DataFrame) -> None:
    st.title("Shipping Service Monitor")
    st.caption("MLOps Mid Exam - Shipping delay prediction")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Shipments", f"{len(data):,}")
    col2.metric("Average Cost", f"${data['Cost_of_the_Product'].mean():.0f}")
    on_time_rate = data["Reached.on.Time_Y.N"].mean() * 100
    col3.metric("On-time Rate", f"{on_time_rate:.1f}%")

    st.divider()
    st.subheader("Sample of the Dataset")
    st.dataframe(
        data.head(5),
        use_container_width=True,
        hide_index=True,
    )
    st.info(
        "This app mirrors the Hugging Face Space layout and reads the same CSV + model "
        "artifacts, so local development and production behave identically."
    )


def main() -> None:
    data = load_data()
    render_overview(data)

    st.sidebar.header("Navigation")
    selected_option = st.sidebar.radio(
        "Choose a page",
        options=("Data Analysis", "Model Prediction"),
        index=0,
    )

    if selected_option == "Data Analysis":
        eda_page(data)
    else:
        model_page(data)


if __name__ == "__main__":
    main()
