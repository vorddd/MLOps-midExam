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
    st.write(
        "Gunakan visualisasi interaktif di bawah untuk memahami distribusi pelanggan, "
        "karakteristik produk, dan faktor yang memengaruhi ketepatan waktu pengiriman."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Median Diskon", f"{data['Discount_offered'].median():.1f}%")
    c2.metric("Median Berat", f"{data['Weight_in_gms'].median():.0f} gram")
    c3.metric("Pelanggan Loyal", f"{(data['Prior_purchases'] > 3).mean():.0%}")

    tab_target, tab_category, tab_numeric, tab_segments = st.tabs(
        [
            "Target Distribusi",
            "Perbandingan Kategori",
            "Numerikal",
            "Segmentasi Bisnis",
        ]
    )

    with tab_target:
        target_counts = (
            _label_target(data[TARGET_COLUMN])
            .value_counts()
            .reset_index()
            .rename(columns={"index": "Status", "count": "Jumlah"})
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
        fig.update_layout(
            showlegend=True,
            title="Proporsi Pengiriman Tepat Waktu",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Proporsi ini menjadi indikator kesehatan proses logistik. "
            "Ketidakseimbangan ekstrem berarti ada peluang perbaikan."
        )

    with tab_category:
        categorical_columns = _get_categorical_columns(data)
        if categorical_columns:
            selected_cat = st.selectbox(
                "Pilih fitur kategori",
                options=categorical_columns,
                help="Visualisasi ini membantu melihat pola keterlambatan per kategori.",
            )
            fig = px.histogram(
                data,
                x=selected_cat,
                color=_label_target(data[TARGET_COLUMN]),
                barmode="group",
                text_auto=True,
                category_orders={selected_cat: sorted(data[selected_cat].unique())},
            )
            fig.update_layout(
                xaxis_title=selected_cat,
                yaxis_title="Jumlah pengiriman",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Tidak ada fitur kategori pada dataset.")

    with tab_numeric:
        numeric_columns = _get_numeric_columns(data)
        if numeric_columns:
            selected_num = st.selectbox(
                "Pilih fitur numerik",
                options=numeric_columns,
                help="Amati persebaran data dan indikasi potensi outlier.",
            )
            col_hist, col_box = st.columns(2)
            with col_hist:
                fig_hist = px.histogram(
                    data,
                    x=selected_num,
                    nbins=25,
                    color=_label_target(data[TARGET_COLUMN]),
                )
                fig_hist.update_layout(showlegend=False)
                st.plotly_chart(fig_hist, use_container_width=True)

            with col_box:
                fig_box = px.box(
                    data,
                    y=selected_num,
                    color=_label_target(data[TARGET_COLUMN]),
                )
                fig_box.update_layout(showlegend=False)
                st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.info("Tidak ada fitur numerik tambahan.")

    with tab_segments:
        grouping_column = st.selectbox(
            "Kelompokkan berdasarkan",
            options=[
                "Mode_of_Shipment",
                "Warehouse_block",
                "Product_importance",
                "Gender",
            ],
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
        summary["on_time_rate"] = summary["on_time_rate"] * 100

        fig_segment = px.bar(
            summary,
            x=grouping_column,
            y="on_time_rate",
            text_auto=".1f",
            labels={"on_time_rate": "Ketepatan Waktu (%)"},
            color="avg_discount",
            color_continuous_scale="Bluyl",
        )
        st.plotly_chart(fig_segment, use_container_width=True)

        with st.expander("Detail angka per segmen"):
            summary_display = summary.copy()
            summary_display["on_time_rate"] = summary_display["on_time_rate"].map(
                lambda val: f"{val:.1f}%"
            )
            summary_display["avg_cost"] = summary_display["avg_cost"].map(
                lambda val: f"${val:,.0f}"
            )
            summary_display["avg_discount"] = summary_display["avg_discount"].map(
                lambda val: f"{val:.1f}%"
            )
            st.dataframe(summary_display, use_container_width=True)

        st.caption(
            "Gunakan segmentasi ini untuk berdiskusi dengan tim operasional terkait "
            "prioritas optimisasi."
        )
