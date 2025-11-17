import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

TARGET_COLUMN = "Reached.on.Time_Y.N"
TARGET_LABELS = {1: "On Time", 0: "Late"}


def _label_target(series: pd.Series) -> pd.Series:
    return series.map(TARGET_LABELS).fillna("Unknown")


def _get_categorical_columns(data: pd.DataFrame) -> list[str]:
    return data.select_dtypes(include=["object"]).columns.tolist()


def _get_numeric_columns(data: pd.DataFrame) -> list[str]:
    """Return only business-relevant numeric features (exclude ID & target)."""
    candidate_cols = [
        "Customer_care_calls",
        "Customer_rating",
        "Cost_of_the_Product",
        "Prior_purchases",
        "Discount_offered",
        "Weight_in_gms",
    ]
    return [c for c in candidate_cols if c in data.columns]


def _make_numeric_bins(series: pd.Series, n_bins: int = 5) -> pd.Categorical:
    """Create human-friendly ranges like '0–1000' instead of raw numbers."""
    min_v = float(series.min())
    max_v = float(series.max())

    if min_v == max_v:
        return pd.cut(series, bins=[min_v - 1, max_v + 1], labels=[f"{min_v:.0f}"])

    bins = np.linspace(min_v, max_v, n_bins + 1)
    labels = [f"{bins[i]:.0f}–{bins[i+1]:.0f}" for i in range(len(bins) - 1)]
    return pd.cut(series, bins=bins, labels=labels, include_lowest=True)


def eda_page(data: pd.DataFrame) -> None:
    st.header("Exploratory Data Analysis")

    total_shipments = len(data)
    on_time_rate = data[TARGET_COLUMN].mean() * 100

    median_discount = data["Discount_offered"].median()
    median_weight = data["Weight_in_gms"].median()
    loyal_share = (data["Prior_purchases"] > 3).mean() * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total shipments", f"{total_shipments:,}")
    c2.metric("On-time delivery rate", f"{on_time_rate:.1f}%")
    c3.metric("Median discount", f"{median_discount:.1f}%")
    c4.metric("Loyal customers (> 3 prior purchases)", f"{loyal_share:.1f}%")

    st.caption(
        "Quick overview: how many shipments you have, how reliable deliveries are, "
        "and how generous you are with discounts and loyal customers."
    )

    tab_target, tab_category, tab_numeric, tab_segments = st.tabs(
        ["Delivery status", "By category", "By numeric feature", "Business segments"]
    )

    with tab_target:
        st.subheader("Overall delivery status")

        target_series = _label_target(data[TARGET_COLUMN])
        target_counts = (
            target_series.value_counts().rename_axis("Status").reset_index(name="Count")
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
                f"**Insight:** about **{late_pct:.1f}%** of all shipments arrive late. "
                "This is the high-level reliability of your operation."
            )

    with tab_category:
        st.subheader("How delivery status changes by category")

        categorical_columns = _get_categorical_columns(data)
        if not categorical_columns:
            st.info("This dataset does not contain categorical features.")
        else:
            selected_cat = st.selectbox(
                "Choose a categorical feature",
                options=categorical_columns,
            )

            # Instead of 100% bars, show on-time rate per category
            summary = (
                data.groupby(selected_cat)
                .agg(
                    on_time_percent=(TARGET_COLUMN, lambda x: x.mean() * 100),
                    total_shipments=(TARGET_COLUMN, "size"),
                )
                .reset_index()
            )

            fig_cat = px.bar(
                summary,
                x=selected_cat,
                y="on_time_percent",
                text_auto=".1f",
                labels={
                    "on_time_percent": "On-time delivery rate (%)",
                    "total_shipments": "Number of shipments",
                },
            )
            st.plotly_chart(fig_cat, use_container_width=True)

            st.caption(
                "Each bar shows **what percentage of shipments are delivered on time** "
                f"for each value of **{selected_cat}**. Lower bars = higher risk segments."
            )

            st.write("Detailed numbers:")
            st.dataframe(
                summary.rename(
                    columns={
                        "on_time_percent": "On-time rate (%)",
                        "total_shipments": "Total shipments",
                    }
                ),
                use_container_width=True,
            )

    with tab_numeric:
        st.subheader("Distribution of numeric features")

        numeric_columns = _get_numeric_columns(data)
        if not numeric_columns:
            st.info("This dataset does not contain numeric features (other than ID/target).")
        else:
            selected_num = st.selectbox(
                "Choose a numeric feature",
                options=numeric_columns,
            )

            # Convert numeric values into simple ranges (bins)
            binned = _make_numeric_bins(data[selected_num])
            temp = data.copy()
            temp["range"] = binned

            summary_num = (
                temp.groupby("range")
                .agg(
                    on_time_percent=(TARGET_COLUMN, lambda x: x.mean() * 100),
                    total_shipments=(TARGET_COLUMN, "size"),
                )
                .reset_index()
                .dropna()
            )

            fig_num = px.bar(
                summary_num,
                x="range",
                y="on_time_percent",
                text_auto=".1f",
                labels={
                    "range": f"{selected_num} range",
                    "on_time_percent": "On-time delivery rate (%)",
                },
            )
            st.plotly_chart(fig_num, use_container_width=True)

            st.caption(
                f"Bars show how **on-time delivery rate** changes across different ranges of "
                f"**{selected_num}**. For example, you can see whether very high values are "
                "associated with more late shipments."
            )

            st.write("Detailed numbers:")
            st.dataframe(
                summary_num.rename(
                    columns={
                        "range": f"{selected_num} range",
                        "on_time_percent": "On-time rate (%)",
                        "total_shipments": "Total shipments",
                    }
                ),
                use_container_width=True,
            )

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
            "Use this view to spot **high-risk segments**: groups with low on-time rate, "
            "especially if they also receive high discounts or involve high product cost."
        )
