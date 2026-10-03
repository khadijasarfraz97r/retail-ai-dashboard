import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Retail AI | Customer Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# SIMPLE PROFESSIONAL STYLE
# =========================================================

st.markdown("""
<style>
.stApp {
    background-color: #f5f7fb;
}

section[data-testid="stSidebar"] {
    background-color: #111827;
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

h1, h2, h3 {
    color: #172033 !important;
}

p, label {
    color: #334155 !important;
}

div[data-testid="stMetric"] {
    background-color: white;
    border: 1px solid #e5e7eb;
    border-radius: 16px;
    padding: 18px;
}

div[data-testid="stMetricLabel"] {
    color: #64748b !important;
}

div[data-testid="stMetricValue"] {
    color: #172033 !important;
}

div[data-baseweb="select"] > div {
    background-color: white !important;
    border-radius: 10px !important;
}

.stButton > button {
    background-color: #2563eb;
    color: white !important;
    border-radius: 10px;
    border: none;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD DATA
# =========================================================

try:
    rfm = pd.read_csv("customer_rfm.csv")
except FileNotFoundError:
    st.error("customer_rfm.csv was not found. Please run analysis.py first.")
    st.stop()

# =========================================================
# CHECK DATA
# =========================================================

required_columns = ["CustomerID", "Recency", "Frequency", "Monetary"]

if not all(column in rfm.columns for column in required_columns):
    st.error("The required RFM columns are missing from customer_rfm.csv.")
    st.stop()

rfm["CustomerID"] = rfm["CustomerID"].astype(str)
rfm = rfm.dropna(subset=["Recency", "Frequency", "Monetary"]).copy()

# =========================================================
# K-MEANS
# =========================================================

features = ["Recency", "Frequency", "Monetary"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(rfm[features])

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

rfm["Cluster"] = kmeans.fit_predict(X_scaled)

# =========================================================
# BUSINESS SEGMENTS
# =========================================================

cluster_summary = rfm.groupby("Cluster")[features].mean()

cluster_summary["Score"] = (
    cluster_summary["Frequency"].rank(pct=True)
    + cluster_summary["Monetary"].rank(pct=True)
    - cluster_summary["Recency"].rank(pct=True)
)

sorted_clusters = cluster_summary["Score"].sort_values(
    ascending=False
).index.tolist()

segment_names = [
    "VIP Customers ⭐",
    "Regular Customers 💙",
    "New Customers 🆕",
    "At-Risk Customers ⚠️"
]

cluster_to_segment = {
    cluster: segment_names[i]
    for i, cluster in enumerate(sorted_clusters)
}

rfm["Segment"] = rfm["Cluster"].map(cluster_to_segment)

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.title("📊 Retail AI")
    st.caption("CUSTOMER INTELLIGENCE DASHBOARD")

    st.divider()

    page = st.radio(
        "Dashboard",
        [
            "Overview",
            "Customer Explorer",
            "Customer Segments"
        ]
    )

    st.divider()

    st.subheader("About")
    st.write(
        "AI-powered customer intelligence using "
        "RFM analysis and K-Means clustering."
    )

    st.divider()
    st.caption("Retail AI Dashboard • Machine Learning")

# =========================================================
# MAIN HEADER
# =========================================================

st.info("AI CUSTOMER INTELLIGENCE")

st.title("📊 Customer Intelligence Dashboard")

st.write(
    "Turn retail transaction data into useful customer insights "
    "using RFM Analysis and K-Means Machine Learning."
)

st.divider()

# =========================================================
# OVERVIEW
# =========================================================

if page == "Overview":

    st.header("Business Overview")

    total_customers = len(rfm)
    total_revenue = rfm["Monetary"].sum()
    avg_spending = rfm["Monetary"].mean()
    total_segments = rfm["Segment"].nunique()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("👥 Total Customers", f"{total_customers:,}")

    with col2:
        st.metric("💰 Total Revenue", f"£{total_revenue:,.0f}")

    with col3:
        st.metric(
            "📈 Avg. Customer Spending",
            f"£{avg_spending:,.0f}"
        )

    with col4:
        st.metric("🎯 Customer Segments", total_segments)

    st.header("Customer Analytics")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎯 Customer Segments")

        segment_counts = (
            rfm["Segment"]
            .value_counts()
            .rename_axis("Segment")
            .reset_index(name="Customers")
        )

        st.bar_chart(
            segment_counts.set_index("Segment")
        )

    with col2:
        st.subheader("💰 Top Customer Spending")

        spending_chart = (
            rfm[["CustomerID", "Monetary"]]
            .sort_values("Monetary", ascending=False)
            .head(15)
            .set_index("CustomerID")
        )

        st.bar_chart(spending_chart)

# =========================================================
# CUSTOMER EXPLORER
# =========================================================

elif page == "Customer Explorer":

    st.header("🔎 Customer Explorer")

    st.write(
        "Select a customer to see their purchasing behaviour."
    )

    customer_id = st.selectbox(
        "Select Customer",
        sorted(rfm["CustomerID"].unique())
    )

    customer = rfm[
        rfm["CustomerID"] == customer_id
    ].iloc[0]

    st.subheader(f"Customer #{customer_id}")

    st.success(
        f"Customer Segment: {customer['Segment']}"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "🕒 Recency",
            f"{int(customer['Recency'])} days"
        )

    with col2:
        st.metric(
            "🛒 Frequency",
            f"{int(customer['Frequency'])} orders"
        )

    with col3:
        st.metric(
            "💰 Total Spending",
            f"£{customer['Monetary']:,.2f}"
        )

    st.subheader("What does this mean?")

    st.write(
        "🕒 Recency: how many days have passed since the "
        "customer's last purchase. Lower is better."
    )

    st.write(
        "🛒 Frequency: how many different orders the customer "
        "has made."
    )

    st.write(
        "💰 Total Spending: the total value of the customer's "
        "purchases."
    )

# =========================================================
# CUSTOMER SEGMENTS
# =========================================================

elif page == "Customer Segments":

    st.header("🎯 Customer Segments")

    st.write(
        "Explore customers grouped according to their "
        "purchasing behaviour."
    )

    segment_filter = st.selectbox(
        "Select Segment",
        ["All Segments"] + sorted(rfm["Segment"].unique())
    )

    if segment_filter == "All Segments":
        display_data = rfm[
            [
                "CustomerID",
                "Recency",
                "Frequency",
                "Monetary",
                "Segment"
            ]
        ].copy()
    else:
        display_data = rfm[
            rfm["Segment"] == segment_filter
        ][
            [
                "CustomerID",
                "Recency",
                "Frequency",
                "Monetary",
                "Segment"
            ]
        ].copy()

    display_data = display_data.sort_values(
        "Monetary",
        ascending=False
    )

    display_data = display_data.rename(
        columns={
            "CustomerID": "Customer ID",
            "Recency": "Recency (days)",
            "Frequency": "Orders",
            "Monetary": "Total Spending",
            "Segment": "Customer Segment"
        }
    )

    st.dataframe(
        display_data,
        use_container_width=True,
        height=500
    )

# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption("Retail AI • Customer Intelligence Dashboard")
st.caption("Built with Python • Pandas • Scikit-Learn • Streamlit")
