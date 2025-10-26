import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime, timezone
import plotly.express as px
import os

# -------------------------
# Page Config (Modern Theme)
st.set_page_config(
    page_title="LayerDetective Live Dashboard",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS for a prettier, more modern look ---
st.markdown("""
<style>
    /* Main app background */
    .stApp {
        background-color: #0f1116;
    }
    /* Styling for metric containers to look like cards */
    div[data-testid="stMetric"] {
        background-color: #1a1c24;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
        transition: 0.3s;
    }
    div[data-testid="stMetric"]:hover {
        box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
    }
    /* Center the metric labels and values */
    div[data-testid="stMetric"] > div {
        text-align: center;
    }
    /* Dataframe styling */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }
    /* Subheader font */
    h2 {
        color: #d1d1d1;
        font-family: 'sans-serif';
    }
    /* General text color */
    body {
        color: #fafafa;
    }
</style>
""", unsafe_allow_html=True)


st.title("📡 LayerDetective Live Fraud Dashboard")
st.markdown("##### Displaying real-time predictions from the live Transformer model.")


# -------------------------
# Configuration
LOG_FILE_PATH = "prediction_log.csv"
REFRESH_INTERVAL_SECONDS = 2  # How often to check the log file for updates

# -------------------------
# Session state initialization
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame()
if "last_mtime" not in st.session_state:
    st.session_state.last_mtime = 0


# -------------------------
# Data Loading Logic
def load_data():
    """Checks the log file's modification time and reloads it if it has changed."""
    try:
        if not os.path.exists(LOG_FILE_PATH):
            return

        current_mtime = os.path.getmtime(LOG_FILE_PATH)
        if current_mtime > st.session_state.last_mtime:
            st.session_state.last_mtime = current_mtime
            df_new = pd.read_csv(LOG_FILE_PATH)

            df_new = df_new.rename(columns={
                'timestamp': 'unix_time',
                'amount': 'amt',
                'predicted_label': 'is_fraud',
                'fraud_probability': 'score'
            })
            df_new = df_new.sort_values(by='unix_time', ascending=False).reset_index(drop=True)
            st.session_state.df = df_new

    except Exception as e:
        st.error(f"Error loading data from {LOG_FILE_PATH}: {e}")


load_data()

# -------------------------
# UI Layout
left_col, right_col = st.columns([2, 1])

# -------------------------
# LEFT COLUMN - LIVE FEED
with left_col:
    st.subheader("📈 Live Predictions Feed")

    df_display = st.session_state.df.head(50)

    # Use a less intense red for highlighting fraud on a dark background
    def highlight_fraud(row):
        return ['background-color: #8B0000' if row["is_fraud"] == 1 else '' for _ in row]

    if df_display.empty:
        st.info(f"Waiting for predictions... Ensure 'live_predictor.py' is running and writing to '{LOG_FILE_PATH}'.")
    else:
        st.dataframe(
            df_display.style.apply(highlight_fraud, axis=1),
            use_container_width=True,
            height=600,
            column_config={
                "is_fraud": st.column_config.NumberColumn(label="Fraud?", width="small"),
                "score": st.column_config.NumberColumn(format="%.3f", width="small"),
                "amt": st.column_config.NumberColumn(label="Amount", format="$%.2f", width="small"),
                "unix_time": st.column_config.DatetimeColumn(label="Time", format="h:mm:ss a"),
                "category": st.column_config.TextColumn(width="medium"),
                "cc_num": st.column_config.TextColumn(label="CC Number", width="medium"),
                "dob": None,
                "trans_num": None,
            },
            column_order=("is_fraud", "score", "amt", "unix_time", "category", "cc_num")
        )

# -------------------------
# RIGHT COLUMN - TOP ANALYTICS
with right_col:
    st.subheader("📊 Real-Time Analytics")

    df = st.session_state.df.copy()

    if df.empty:
        st.info("Waiting for data...")
    else:
        total_alerts = df[df["is_fraud"] == 1].shape[0]
        st.metric(
            label=f"Total Fraud Alerts (All Time)",
            value=f"{total_alerts}"
        )
        st.markdown("---", unsafe_allow_html=True)

        st.subheader("🚨 High-Risk Transactions (Last 5 Mins)")
        five_mins_ago = int(datetime.now(timezone.utc).timestamp()) - 5 * 60
        df_last_5_mins = df[df["unix_time"] >= five_mins_ago]
        high_risk = df_last_5_mins[df_last_5_mins['score'] > 0.9].sort_values(by='score', ascending=False).head(5)

        if not high_risk.empty:
            st.dataframe(
                high_risk,
                column_config={"score": "Score", "amt": "Amount", "category": "Category", "trans_num": None,
                               "cc_num": None, "unix_time": None, "is_fraud": None, "dob": None},
                hide_index=True
            )
        else:
            st.info("No high-risk transactions detected in the last 5 minutes.")

# ------------------------------------------------------------------
# NEW BOTTOM SECTION - AGE ANALYSIS (Full Width)
# ------------------------------------------------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.subheader("🎂 All-Time Fraud Analysis by Age Segment")

df_age_analysis = st.session_state.df.copy()

if df_age_analysis.empty or 'dob' not in df_age_analysis.columns:
    st.info("No data with 'dob' field yet to analyze age segments.")
else:
    frauds_df = df_age_analysis[df_age_analysis["is_fraud"] == 1].copy()

    if not frauds_df.empty:
        def compute_age(dob_str):
            try:
                return datetime.now().year - pd.to_datetime(dob_str).year
            except:
                return None

        frauds_df["age"] = frauds_df["dob"].apply(compute_age)
        frauds_df = frauds_df.dropna(subset=['age'])

        bins = [0, 25, 35, 50, 65, 200]
        labels = ["18-25", "26-35", "36-50", "51-65", "65+"]
        frauds_df["age_segment"] = pd.cut(frauds_df["age"], bins=bins, labels=labels, right=True)

        age_distribution = frauds_df["age_segment"].value_counts().reindex(labels).fillna(0)

        # Using a vibrant, qualitative color palette for distinct colors
        fig = px.pie(
            age_distribution,
            values=age_distribution.values,
            names=age_distribution.index,
            title="Distribution of Fraud Alerts by Age",
            color_discrete_sequence=px.colors.qualitative.Vivid,
            hole=0.4 # Creates a donut chart for a more modern look
        )
        fig.update_layout(
            margin=dict(t=30, b=0, l=0, r=0),
            paper_bgcolor='rgba(0,0,0,0)', # Transparent background
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#fafafa') # Light font for dark theme
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No fraud data yet to analyze age segments.")

# -------------------------
# Auto-refresh logic
time.sleep(REFRESH_INTERVAL_SECONDS)
st.rerun()