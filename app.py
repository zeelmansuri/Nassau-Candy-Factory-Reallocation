from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.optimization import (
    FACTORIES,
    PRODUCT_FACTORY_MAP,
    load_orders,
    recommendation_table,
    route_clusters,
    simulate_product,
    train_models,
)


st.set_page_config(
    page_title="Nassau Candy Factory Optimization",
    page_icon="factory",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def get_data() -> pd.DataFrame:
    return load_orders()


@st.cache_resource(show_spinner=False)
def get_model(data: pd.DataFrame):
    return train_models(data)


def metric_card(label: str, value: str, delta: str | None = None):
    st.metric(label, value, delta=delta)


df = get_data()
bundle = get_model(df)

st.title("Factory Reallocation & Shipping Optimization")
st.caption("Decision intelligence dashboard for Nassau Candy Distributor")

with st.sidebar:
    st.header("Optimization Controls")
    product = st.selectbox("Product", sorted(df["Product Name"].dropna().unique()))
    region = st.selectbox("Destination region", ["All"] + sorted(df["Region"].dropna().unique()))
    ship_mode = st.selectbox("Ship mode", ["All"] + sorted(df["Ship Mode"].dropna().unique()))
    priority = st.slider("Optimization priority", 0, 100, 65, help="0 favors profit stability; 100 favors speed.")
    speed_weight = priority / 100
    top_n = st.slider("Recommendations to show", 5, 30, 15)

filtered = df.copy()
if region != "All":
    filtered = filtered[filtered["Region"] == region]
if ship_mode != "All":
    filtered = filtered[filtered["Ship Mode"] == ship_mode]

scenarios = simulate_product(df, bundle, product, region, ship_mode, speed_weight)
current = scenarios[scenarios["Is Current"]].iloc[0]
best = scenarios[~scenarios["Is Current"]].iloc[0]

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    metric_card("Orders analyzed", f"{len(filtered):,}")
with col2:
    metric_card("Avg lead time", f"{filtered['Lead Time'].mean():.1f} days")
with col3:
    metric_card("Best model", bundle.metrics.iloc[0]["Model"])
with col4:
    metric_card("Model RMSE", f"{bundle.metrics.iloc[0]['RMSE']:.2f} days")
with col5:
    metric_card("Potential gain", f"{best['Lead Time Reduction %']:.1f}%", delta=f"{best['Factory']}")

tab_sim, tab_recs, tab_routes, tab_model, tab_data = st.tabs(
    ["Factory Simulator", "Recommendations", "Route Clusters", "Model Quality", "Data Explorer"]
)

with tab_sim:
    left, right = st.columns([1.2, 0.8], gap="large")
    with left:
        st.subheader("Predicted Performance Across Factories")
        fig = px.bar(
            scenarios,
            x="Factory",
            y="Predicted Lead Time",
            color="Risk",
            text="Predicted Lead Time",
            color_discrete_map={"Low": "#2E7D32", "Medium": "#F9A825", "High": "#C62828"},
            hover_data=["Distance Miles", "Lead Time Reduction %", "Profit Stability", "Scenario Confidence"],
        )
        fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig.update_layout(xaxis_title="", yaxis_title="Predicted lead time (days)", height=420)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Factory Map")
        map_df = FACTORIES.rename(columns={"Latitude": "lat", "Longitude": "lon"})
        st.map(map_df, latitude="lat", longitude="lon", size=80)

    with right:
        st.subheader("Current vs Recommended")
        compare = pd.DataFrame(
            [
                {
                    "Scenario": "Current",
                    "Factory": current["Factory"],
                    "Lead Time": current["Predicted Lead Time"],
                    "Profit Stability": current["Profit Stability"],
                    "Confidence": current["Scenario Confidence"],
                    "Risk": current["Risk"],
                },
                {
                    "Scenario": "Recommended",
                    "Factory": best["Factory"],
                    "Lead Time": best["Predicted Lead Time"],
                    "Profit Stability": best["Profit Stability"],
                    "Confidence": best["Scenario Confidence"],
                    "Risk": best["Risk"],
                },
            ]
        )
        st.dataframe(compare, use_container_width=True, hide_index=True)
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=max(best["Lead Time Reduction %"], 0),
                title={"text": "Lead Time Reduction %"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#176B87"},
                    "steps": [
                        {"range": [0, 10], "color": "#F2F2F2"},
                        {"range": [10, 35], "color": "#D9EAF7"},
                        {"range": [35, 100], "color": "#B7E4C7"},
                    ],
                },
            )
        )
        gauge.update_layout(height=260, margin=dict(l=20, r=20, t=55, b=10))
        st.plotly_chart(gauge, use_container_width=True)
        st.info(
            f"Recommended move: assign **{product}** from **{current['Factory']}** to "
            f"**{best['Factory']}** for the selected conditions."
        )

with tab_recs:
    st.subheader("Ranked Factory Reassignment Suggestions")
    recs = recommendation_table(df, bundle, speed_weight, top_n=top_n)
    st.dataframe(
        recs,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Lead Time Reduction %": st.column_config.ProgressColumn(
                "Lead Time Reduction %", min_value=-20, max_value=100, format="%.1f%%"
            ),
            "Profit Stability": st.column_config.ProgressColumn(
                "Profit Stability", min_value=0, max_value=115, format="%.1f%%"
            ),
            "Scenario Confidence": st.column_config.ProgressColumn(
                "Scenario Confidence", min_value=0, max_value=100, format="%.1f%%"
            ),
        },
    )
    col_a, col_b = st.columns(2)
    with col_a:
        fig = px.scatter(
            recs,
            x="Lead Time Reduction %",
            y="Profit Stability",
            color="Risk",
            size="Scenario Confidence",
            hover_name="Product Name",
            color_discrete_map={"Low": "#2E7D32", "Medium": "#F9A825", "High": "#C62828"},
        )
        fig.update_layout(height=390)
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        risk_counts = recs["Risk"].value_counts().reset_index()
        risk_counts.columns = ["Risk", "Recommendations"]
        fig = px.pie(
            risk_counts,
            names="Risk",
            values="Recommendations",
            hole=0.48,
            color="Risk",
            color_discrete_map={"Low": "#2E7D32", "Medium": "#F9A825", "High": "#C62828"},
        )
        fig.update_layout(height=390)
        st.plotly_chart(fig, use_container_width=True)

with tab_routes:
    st.subheader("Route & Product Performance Similarity")
    clusters = route_clusters(df)
    col_a, col_b = st.columns([1.1, 0.9], gap="large")
    with col_a:
        fig = px.scatter(
            clusters,
            x="Avg_Distance",
            y="Avg_Lead_Time",
            size="Orders",
            color="Cluster Label",
            hover_data=["Region", "Ship Mode", "Current Factory", "Avg_Margin"],
        )
        fig.update_layout(xaxis_title="Average distance (miles)", yaxis_title="Average lead time (days)", height=430)
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        slow = clusters.sort_values("Avg_Lead_Time", ascending=False).head(12)
        st.dataframe(slow, use_container_width=True, hide_index=True)

with tab_model:
    st.subheader("Model Evaluation")
    st.dataframe(bundle.metrics, use_container_width=True, hide_index=True)
    fig = px.bar(bundle.metrics, x="Model", y=["RMSE", "MAE"], barmode="group")
    fig.update_layout(height=360, yaxis_title="Days", xaxis_title="")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        f"""
        **Training rows after outlier handling:** {bundle.training_rows:,}  
        **Features:** product, division, region, ship mode, factory, distance, units, sales, profit, margin, month, and weekday.
        """
    )

with tab_data:
    st.subheader("Operational Data Explorer")
    st.dataframe(filtered, use_container_width=True, hide_index=True)
    product_factory = pd.DataFrame(
        [{"Product Name": k, "Assigned Factory": v} for k, v in PRODUCT_FACTORY_MAP.items()]
    )
    st.download_button(
        "Download recommendations CSV",
        recommendation_table(df, bundle, speed_weight, top_n=top_n).to_csv(index=False),
        file_name="nassau_factory_recommendations.csv",
        mime="text/csv",
    )
    st.subheader("Product to Factory Rules")
    st.dataframe(product_factory, use_container_width=True, hide_index=True)
