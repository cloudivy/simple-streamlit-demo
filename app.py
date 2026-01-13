# app.py - Streamlit app for Pipeline Pilferage Classification (Single Value Mode)
import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

st.set_page_config(page_title="🛢️ Pipeline Pilferage Detection", layout="wide")

st.title("🛢️ Pipeline Pilferage Detection")
st.markdown("**Single Value Classification Mode**")

# Fixed parameters (as per original code)
CHAINAGE_TOL = 0.5
TIME_WINDOW_HOURS = 48

st.sidebar.info(f"**Fixed Parameters:**\nChainage Tol: {CHAINAGE_TOL}km\nTime Window: {TIME_WINDOW_HOURS}h")

# File upload - Single files only
uploaded_pidws = st.file_uploader("📁 PIDWS Events (df_pidws_III.xlsx)", type=['xlsx'], key="pidws")
uploaded_lds = st.file_uploader("📁 LDS Leaks (df_lds_III.xlsx)", type=['xlsx'], key="lds")

@st.cache_data
def parse_duration(dur_str):
    if pd.isna(dur_str):
        return pd.Timedelta(0)
    dur_str = str(dur_str).strip().lower().replace(' ', '')
    mins, secs = 0, 0
    if 'm' in dur_str:
        m_part = dur_str.split('m')[0]
        if m_part.isdigit(): mins = int(m_part)
        dur_str = dur_str.split('m')[1]
    if 's' in dur_str:
        s_part = dur_str.replace('s', '')
        if s_part.isdigit(): secs = int(s_part)
    return pd.Timedelta(minutes=mins, seconds=secs)

@st.cache_data
def classify_pilferage(pidws_df, lds_df):
    classified = []
    for _, event in pidws_df.iterrows():
        window_end = event['end_time'] + pd.Timedelta(hours=TIME_WINDOW_HOURS)
        mask = (lds_df['DateTime'] > window_end) & \
               (np.abs(lds_df['chainage'] - event['chainage']) <= CHAINAGE_TOL)
        matches = lds_df[mask].copy()
        if not matches.empty:
            matches['linked_event_time'] = event['DateTime']
            matches['linked_chainage'] = event['chainage']
            matches['pilferage_score'] = 1 / (1 + (matches['DateTime'] - window_end).dt.total_seconds() / 3600)
            classified.append(matches)
    return pd.concat(classified, ignore_index=True) if classified else pd.DataFrame()

if uploaded_pidws and uploaded_lds:
    with st.spinner("🔄 Processing..."):
        # Load PIDWS
        df_pidws = pd.read_excel(uploaded_pidws)
        df_pidws['DateTime'] = pd.to_datetime(df_pidws['Date'] + ' ' + df_pidws['Time'], format='%d-%m-%Y %H:%M:%S')
        df_pidws['duration_td'] = df_pidws['Event Duration'].apply(parse_duration)
        df_pidws['end_time'] = df_pidws['DateTime'] + df_pidws['duration_td']
        
        # Load LDS
        df_lds = pd.read_excel(uploaded_lds)
        df_lds['DateTime'] = pd.to_datetime(df_lds['Date'].astype(str) + ' ' + df_lds['Time'])
        
        # Classify
        pilferage_leaks = classify_pilferage(df_pidws, df_lds)
        df_lds_classified = df_lds.copy()
        df_lds_classified['is_pilferage'] = False
        
        if not pilferage_leaks.empty:
            pilferage_ids = pilferage_leaks[['DateTime', 'chainage']].drop_duplicates()
            mask_pilferage = df_lds_classified.set_index(['DateTime', 'chainage']).index.isin(
                pilferage_ids.set_index(['DateTime', 'chainage']).index
            )
            df_lds_classified.loc[mask_pilferage, 'is_pilferage'] = True
        
        st.session_state.results = {
            'pilferage_count': len(pilferage_leaks),
            'total_leaks': len(df_lds),
            'df_pidws': df_pidws,
            'df_lds_classified': df_lds_classified,
            'pilferage_leaks': pilferage_leaks
        }
        st.success("✅ Analysis Complete!")

    # Single Value Results
    results = st.session_state.results
    col1, col2, col3 = st.columns(3)
    col1.metric("🎯 Pilferage Events", results['pilferage_count'])
    col2.metric("📊 Total Leaks", results['total_leaks'])
    col3.metric("⚡ Pilferage Rate", f"{results['pilferage_count']/results['total_leaks']*100:.1f}%")

    # Results Table
    st.subheader("📋 Classified Leaks")
    st.dataframe(results['df_lds_classified'][['DateTime', 'chainage', 'leak size', 'is_pilferage']], 
                use_container_width=True)

    # Single Chart
    fig = px.scatter(results['df_lds_classified'], x='DateTime', y='chainage', 
                    color='is_pilferage', color_discrete_map={True: 'red', False: 'blue'},
                    title="🔴 Red = Pilferage | 🔵 Other Leaks",
                    hover_data=['leak size'])
    st.plotly_chart(fig, use_container_width=True)

    # Download
    csv = results['df_lds_classified'].to_csv(index=False)
    st.download_button("💾 Download Results", csv, "lds_classified.csv", "text/csv")

else:
    st.info("👆 Upload **df_pidws_III.xlsx** and **df_lds_III.xlsx** to analyze")
