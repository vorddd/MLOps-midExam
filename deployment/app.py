from pathlib import Path

import pandas as pd
import streamlit as st

from eda import eda_page
from prediction import model_page

st.set_page_config(
    page_title="Shipping Service Monitor",
    page_icon="📦",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    return pd.read_csv(BASE_DIR / "shipping.csv")


def render_overview(data: pd.DataFrame) -> None:
    st.title("Shipping Service Monitor")
    st.caption("Milestone 2 • Iqbal Saputra • RMT-032")
    st.write(
        "Aplikasi ini dirancang agar nyaman dipakai di **Hugging Face Spaces**, "
        "dengan layout yang ringkas dan fokus pada insight logistik."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Pengiriman", f"{len(data):,}")
    col2.metric("Rata-rata Biaya", f"${data['Cost_of_the_Product'].mean():.0f}")
    on_time_rate = data["Reached.on.Time_Y.N"].mean() * 100
    col3.metric("Ketepatan Waktu", f"{on_time_rate:.1f}%")

    st.divider()
    st.subheader("Cuplikan Dataset")
    st.dataframe(
        data.head(5),
        use_container_width=True,
        hide_index=True,
    )
    st.info(
        "Seluruh file penting (`shipping.csv`, pipeline preprocessing, dan model) berada "
        "dalam folder `deployment/` sehingga otomatis dideteksi oleh Hugging Face saat "
        "membangun aplikasi."
    )


def main():
    data = load_data()
    render_overview(data)

    menu_options = ["Data Analysis", "Model Prediction"]
    selected_option = st.sidebar.radio("Pilih Halaman", menu_options, index=0)

    if selected_option == "Data Analysis":
        eda_page(data)
    elif selected_option == "Model Prediction":
        model_page(data)


if __name__ == "__main__":
    main()
