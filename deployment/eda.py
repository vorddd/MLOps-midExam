import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

TARGET_COLUMN = "Reached.on.Time_Y.N"
TARGET_LABELS = {1: "On Time", 0: "Late"}


def _label_target(series: pd.Series) -> pd.Series:
    # Map 0/1 into human-friendly labels
    return series.map(TARGET_LABELS).fillna("Unknown")


def _get_categorical_columns(data: pd.DataFrame) -> list[str]:
    return data.select_dtypes(include=["object"]).columns.tolist()


def _get_numeric_columns(data: pd.DataFrame) -> list[str]:
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude target from numeric EDA
    return [col for col in numeric_cols if col != TARGET_COLUMN]


def eda_page(data: pd.DataFrame) -> None:
    st.header("Exploratory Data Analysis")

    # ==================== TOP SUMMARY CARDS ====================
    total_shipments = len(data)
    on_time_rate = data[TARGET_COLUMN].mean() * 100  # 1 = On Time
    median_discount = data["Discount_offered"].median()
    median_weight = data["Weight_in_gms"].median()
    loyal_share = (data["Prior_purchases"] > 3).mean() * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total shipments", f"{total_shipments:,}")
    c2.metric("On-time delivery rate", f"{on_time_rate:.1f}%")
    c3.metric("Median discount", f"{median_discount:.1f}%")
    c4.metric(
        "Loyal customers\n(> 3 prior purchases)",
        f"{loyal_share:.1f}%"
    )

    st.caption(
        "These cards give a quick overview of reliability (on-time deliveries), "
        "commercial strategy (discount), and how many customers keep coming back."
    )

    # ==================== TABS ====================
    tab_target, tab_category, tab_numeric, tab_segments = st.tabs(
        ["Delivery status", "By category", "By numeric feature", "Business segments"]
    )

    # -------------------- DELIVERY STATUS --------------------
    with tab_target:
        st.subheader("Overall delivery status")

        target_series = _label_target(data[TARGET_COLUMN])

        target_counts = (
            target_series.value_counts()
            .rename_axis("Status")
            .reset_index(name="Count")
        )
        target_counts["Percentage"] = (
            target_counts["Count"] / target_counts["Count"].sum() * 100
        )

        fig = px.pie(
            target_counts,
            values="Count",
            names="Status",
            hole=0.35,
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

        late_row = target_counts.loc[target_counts["Status"] == "Late"]
        if not late_row.empty:
            late_pct = late_row["Percentage"].iloc[0]
            st.write(
                f"**Insight:** about **{late_pct:.1f}%** of shipments arrive late. "
                "Improving this number has a direct impact on customer satisfaction."
            )
        else:
            st.write(
                "All recorded shipments are marked as **On Time** in this dataset."
            )

    # -------------------- BY CATEGORY --------------------
    with tab_category:
        st.subheader("How delivery status changes by category")

        categorical_columns = _get_categorical_columns(data)
        if categorical_columns:
            selected_cat = st.selectbox(
                "Choose a categorical feature",
                options=categorical_columns,
            )

            fig = px.histogram(
                data,
                x=selected_cat,
                color=_label_target(data[TARGET_COLUMN]),
                barmode="group",
                barnorm="percent",      # show percentage per category
                text_auto=".1f",
            )
            fig.update_yaxes(title="Percentage of shipments")
            st.plotly_chart(fig, use_container_width=True)

            st.caption(
                "Each group shows the share of **On Time vs Late** deliveries "
                "inside that category. Look for categories where the late share is high."
            )
        else:
            st.info("This dataset does not contain categorical features.")

    # -------------------- BY NUMERIC FEATURE --------------------
    with tab_numeric:
        st.subheader("Distribution of numeric features")

        numeric_columns = _get_numeric_columns(data)
        if numeric_columns:
            selected_num = st.selectbox("Choose a numeric feature", numeric_columns)

            col1, col2 = st.columns(2)
            with col1:
                fig_hist = px.histogram(
                    data,
                    x=selected_num,
                    color=_label_target(data[TARGET_COLUMN]),
                    nbins=30,
                )
                st.plotly_chart(fig_hist, use_container_width=True)

            with col2:
                fig_box = px.box(
                    data,
                    y=selected_num,
                    color=_label_target(data[TARGET_COLUMN]),
                )
                st.plotly_chart(fig_box, use_container_width=True)

            st.caption(
                "Use the **histogram** to see the overall distribution, and the **box plot** "
                "to compare typical values and outliers for On Time vs Late shipments."
            )
        else:
            st.info("This dataset does not contain numeric features (other than the target).")

    # -------------------- BUSINESS SEGMENTS --------------------
    with tab_segments:
        st.subheader("Business segments: where do we perform well or poorly?")

        grouping_column = st.selectbox(
            "Group by",
            ["Mode_of_Shipment", "Warehouse_block", "Product_importance", "Gender"],
        )

        summary = (
            data.groupby(grouping_column)
            .agg(
                on_time_percent=(TARGET_COLUMN, lambda x: x.mean() * 100),
                avg_cost=("Cost_of_the_Product", "mean"),
                avg_discount=("Discount_offered", "mean"),
            )
            .reset_index()
        )

        fig_segment = px.bar(
            summary,
            x=grouping_column,
            y="on_time_percent",
            text_auto=".1f",
            color="avg_discount",
            color_continuous_scale="Blues",
            labels={
                "on_time_percent": "On-time delivery rate (%)",
                "avg_discount": "Average discount",
            },
        )
        st.plotly_chart(fig_segment, use_container_width=True)

        st.dataframe(
            summary.rename(
                columns={
                    "on_time_percent": "On-time rate (%)",
                    "avg_cost": "Avg product cost",
                    "avg_discount": "Avg discount",
                }
            ),
            use_container_width=True,
        )

        st.caption(
            "Bars show which segments have **higher or lower on-time delivery rates**. "
            "The color scale indicates the **average discount** in each segment. "
            "You can use this to spot segments that are both heavily discounted **and** still late."
        )
