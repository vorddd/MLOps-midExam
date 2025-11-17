import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

TARGET_COLUMN = "Reached.on.Time_Y.N"
TARGET_LABELS = {1: "Tepat Waktu", 0: "Tidak Tepat Waktu"}


def _label_target(series: pd.Series) -> pd.Series:
    return series.map(TARGET_LABELS).fillna("Tidak Diketahui")


def _get_categorical_columns(data: pd.DataFrame) -> list[str]:
    return data.select_dtypes(include=["object"]).columns.tolist()


def _get_numeric_columns(data: pd.DataFrame) -> list[str]:
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    return [col for col in numeric_cols if col != TARGET_COLUMN]


def eda_page(data: pd.DataFrame) -> None:
    st.header("Exploratory Data Analysis")

    c1, c2, c3 = st.columns(3)
    c1.metric("Median Diskon", f"{data['Discount_offered'].median():.1f}%")
    c2.metric("Median Berat", f"{data['Weight_in_gms'].median():.0f} gram")
    c3.metric("Pelanggan Loyal", f"{(data['Prior_purchases'] > 3).mean():.0%}")

    tab_target, tab_category, tab_numeric, tab_segments = st.tabs(
        ["Target Distribusi", "Perbandingan Kategori", "Numerikal", "Segmentasi Bisnis"]
    )

    # -------------------- FIX TARGET DISTRIBUSI --------------------
    with tab_target:
        target_series = _label_target(data[TARGET_COLUMN])

        target_counts = pd.DataFrame({
            "Status": target_series.unique(),
        })
        target_counts["Jumlah"] = target_counts["Status"].apply(
            lambda s: (target_series == s).sum()
        )

        fig = px.pie(
            target_counts,
            values="Jumlah",
            names="Status",
            color="Status",
            color_discrete_map={
                "Tepat Waktu": "#2eb88a",
                "Tidak Tepat Waktu": "#f7685b",
            },
            hole=0.35,
        )
        st.plotly_chart(fig, use_container_width=True)

    # -------------------- KATEGORI --------------------
    with tab_category:
        categorical_columns = _get_categorical_columns(data)
        if categorical_columns:
            selected_cat = st.selectbox(
                "Pilih fitur kategori", options=categorical_columns
            )
            fig = px.histogram(
                data,
                x=selected_cat,
                color=_label_target(data[TARGET_COLUMN]),
                barmode="group",
                text_auto=True,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Tidak ada fitur kategori pada dataset.")

    # -------------------- NUMERIC --------------------
    with tab_numeric:
        numeric_columns = _get_numeric_columns(data)
        if numeric_columns:
            selected_num = st.selectbox("Pilih fitur numerik", numeric_columns)

            col1, col2 = st.columns(2)
            with col1:
                fig_hist = px.histogram(data, x=selected_num,
                                        color=_label_target(data[TARGET_COLUMN]))
                st.plotly_chart(fig_hist, use_container_width=True)

            with col2:
                fig_box = px.box(data, y=selected_num,
                                 color=_label_target(data[TARGET_COLUMN]))
                st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.info("Tidak ada fitur numerik.")

    # -------------------- SEGMENTS --------------------
    with tab_segments:
        grouping_column = st.selectbox(
            "Kelompokkan berdasarkan",
            ["Mode_of_Shipment", "Warehouse_block", "Product_importance", "Gender"],
        )

        summary = (
            data.groupby(grouping_column)
            .agg(
                on_time_rate=(TARGET_COLUMN, "mean"),
                avg_cost=("Cost_of_the_Product", "mean"),
                avg_discount=("Discount_offered", "mean"),
            )
            .reset_index()
        )
        summary["on_time_rate"] *= 100

        fig_segment = px.bar(
            summary,
            x=grouping_column,
            y="on_time_rate",
            text_auto=".1f",
            color="avg_discount",
            color_continuous_scale="Blues",
        )
        st.plotly_chart(fig_segment, use_container_width=True)
