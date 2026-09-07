"""
Real-Time Fraud Detection Dashboard - Simplified Version
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import time

# Page config
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🔴",
    layout="wide"
)

# API Configuration
API_URL = "http://localhost:8000"

# Title
st.title("🔴 Real-Time Fraud Detection System")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")

    # API Health Check
    try:
        health = requests.get(f"{API_URL}/health", timeout=2)
        if health.status_code == 200:
            st.success("✅ API Connected")
            model_info = requests.get(f"{API_URL}/model-info", timeout=2).json()
            st.info(f"Model: v{model_info['model_version']}")
            st.info(f"Threshold: {model_info['threshold']*100:.0f}%")
        else:
            st.error("❌ API Not Responding")
            st.stop()
    except Exception as e:
        st.error("❌ Cannot Connect to API")
        st.error(str(e))
        st.stop()

    st.markdown("---")

    # Auto-refresh
    auto_refresh = st.checkbox("Auto-refresh (5s)", value=False)

    # Manual refresh button
    if st.button("🔄 Refresh Now"):
        st.rerun()

# Fetch transactions
try:
    response = requests.get(f"{API_URL}/transactions?limit=100", timeout=5)

    if response.status_code != 200:
        st.error(f"❌ API returned status {response.status_code}")
        st.stop()

    data = response.json()

    st.sidebar.markdown("---")
    st.sidebar.write(f"📊 **Data Status**")
    st.sidebar.write(f"Transactions found: **{data['count']}**")

    if data['count'] == 0:
        st.warning("⚠️ No transactions in database yet!")
        st.info("Run the simulator to generate transactions:")
        st.code("uv run python scripts/simulate_transactions.py", language="bash")
        st.stop()

    # Convert to DataFrame
    df = pd.DataFrame(data['transactions'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    st.sidebar.success(f"✅ Loaded {len(df)} rows")

except Exception as e:
    st.error(f"❌ Error fetching data: {str(e)}")
    st.stop()

# Calculate metrics
total_transactions = len(df)
fraud_count = len(df[df['prediction'] == 'FRAUD'])
legitimate_count = len(df[df['prediction'] == 'LEGITIMATE'])
fraud_rate = (fraud_count / total_transactions * 100) if total_transactions > 0 else 0
avg_latency = df['latency_ms'].mean()

# === SECTION 1: KEY METRICS ===
st.header("📊 Key Metrics")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total Transactions", f"{total_transactions:,}")

with col2:
    st.metric("🔴 Fraud Detected", fraud_count)

with col3:
    st.metric("🟢 Legitimate", legitimate_count)

with col4:
    st.metric("Fraud Rate", f"{fraud_rate:.1f}%")

with col5:
    st.metric("Avg Latency", f"{avg_latency:.1f}ms")

st.markdown("---")

# === SECTION 2: RECENT TRANSACTIONS TABLE ===
st.header("📋 Recent Transactions (Last 20)")

recent_df = df.head(20).copy()
recent_df['time'] = recent_df['timestamp'].dt.strftime('%H:%M:%S')

# Display simplified table
display_cols = ['transaction_id', 'time', 'amount', 'prediction', 'fraud_probability', 'risk_level', 'latency_ms']
st.dataframe(
    recent_df[display_cols],
    use_container_width=True,
    hide_index=True
)

st.markdown("---")

# === SECTION 3: FRAUD PIE CHART ===
st.header("📈 Prediction Distribution")

fig_pie = go.Figure(data=[go.Pie(
    labels=['Legitimate', 'Fraud'],
    values=[legitimate_count, fraud_count],
    marker=dict(colors=['#00cc00', '#ff4b4b']),
    hole=0.4
)])
fig_pie.update_layout(
    title="Fraud vs Legitimate",
    height=400
)
st.plotly_chart(fig_pie, use_container_width=True)

st.markdown("---")

# === SECTION 4: PROBABILITY HISTOGRAM ===
st.header("🎯 Fraud Probability Distribution")

fig_hist = px.histogram(
    df,
    x='fraud_probability',
    nbins=30,
    color='prediction',
    color_discrete_map={'LEGITIMATE': '#00cc00', 'FRAUD': '#ff4b4b'},
    title="Distribution of Fraud Probabilities"
)
fig_hist.update_layout(height=400)
st.plotly_chart(fig_hist, use_container_width=True)

st.markdown("---")

# === SECTION 5: AMOUNT BOX PLOT ===
st.header("💰 Transaction Amounts by Prediction")

fig_box = px.box(
    df,
    x='prediction',
    y='amount',
    color='prediction',
    color_discrete_map={'LEGITIMATE': '#00cc00', 'FRAUD': '#ff4b4b'},
    title="Amount Distribution"
)
fig_box.update_layout(height=400)
st.plotly_chart(fig_box, use_container_width=True)

st.markdown("---")

# === SECTION 6: PERFORMANCE METRICS ===
st.header("⚡ System Performance")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Min Latency", f"{df['latency_ms'].min():.2f}ms")

with col2:
    st.metric("Max Latency", f"{df['latency_ms'].max():.2f}ms")

with col3:
    st.metric("Median Latency", f"{df['latency_ms'].median():.2f}ms")

with col4:
    st.metric("P95 Latency", f"{df['latency_ms'].quantile(0.95):.2f}ms")

# Footer
st.markdown("---")
from datetime import datetime
st.caption(f"Dashboard last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Auto-refresh logic
if auto_refresh:
    time.sleep(5)
    st.rerun()
