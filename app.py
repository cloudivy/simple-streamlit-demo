import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

# FAST config - minimal plotting
plt.ioff()  # Turn off interactive mode
sns.set_style("whitegrid")

st.set_page_config(page_title="Pilferage Detector", layout="wide")

st.title("🛢️ Pipeline Pilferage Detector")
st.markdown("**Fast analysis • No waiting**")

# SIDEBAR - Compact
pidws_file = st.sidebar.file_uploader("📊 PIDWS Excel", type="xlsx")
lds_file = st.sidebar.file_uploader("🔍 LDS Excel", type="xlsx")

col1, col2 = st.sidebar.columns(2)
tol = col1.slider("Tol", 0.1, 2.0, 0.5, 0.1)
win = col2.slider("Win", 12, 72, 48, 6)

if st.sidebar.button("🚀 RUN ANALYSIS", type="primary") and pidws_file and lds_file:
    with st.spinner("⚡ Analyzing..."):
        # LOAD FAST
        pidws = pd.read_excel(pidws_file)
        lds = pd.read_excel(lds_file)
        
        # SIMPLE PREPROCESS
        pidws['dt'] = pd.to_datetime(pidws['Date'].astype(str) + ' ' + pidws['Time'].astype(str))
        lds['dt'] = pd.to_datetime(lds['Date'].astype(str) + ' ' + lds['Time'].astype(str))
        
        # QUICK CLASSIFICATION
        pilf = []
        for _, r in pidws.iterrows():
            end = r['dt'] + np.timedelta64(win, 'h')
            mask = (lds['dt'] > end) & (np.abs(lds['chainage'] - r['chainage']) <= tol)
            matches = lds[mask]
            if len(matches):
                pilf.append(matches)
        
        pilferage = pd.concat(pilf) if pilf else pd.DataFrame()
        lds['pilferage'] = lds.index.isin(pilferage.index)
        
        # STORE
        st.session_state.results = {
            'lds': lds, 'pilf': pilferage, 'pidws': pidws,
            'count': len(pilferage), 'pct': len(pilferage)/len(lds)*100,
            'tol': tol, 'win': win
        }
    st.rerun()

# RESULTS
if 'results' in st.session_state:
    res = st.session_state.results
    lds = res['lds']
    pilf = res['pilf']
    
    # METRICS ROW 1
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Leaks", len(lds))
    c2.metric("PIDWS Events", len(res['pidws']))
    c3.metric("🟡 Pilferage", res['count'])
    c4.metric("Hit Rate", f"{res['pct']:.1f}%")
    
    # CHARTS ROW
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(7,4))
        ax.hist(res['pidws']['chainage'], bins=20, alpha=0.6, label='PIDWS', color='orange')
        ax.hist(lds['chainage'], bins=20, alpha=0.6, label='Leaks', color='lightblue')
        if len(pilf):
            ax.axvline(pilf['chainage'].mean(), color='red', ls='--', lw=2)
        ax.legend()
        ax.set_title('Chainage Distribution')
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(6,4))
        lds.boxplot('leak size', by='pilferage', ax=ax, patch_artist=True)
        ax.set_title('Leak Sizes')
        plt.suptitle('')
        st.pyplot(fig)
    
    # TIMELINE
    st.subheader("Timeline View")
    fig, ax = plt.subplots(figsize=(12,5))
    colors = ['red' if x else 'blue' for x in lds.pilferage]
    sizes = np.minimum(lds['leak size']*5, 100)
    ax.scatter(lds['dt'], lds['chainage'], c=colors, s=sizes, alpha=0.6)
    ax.set_xlabel('Time'); ax.set_ylabel('Chainage (km)')
    ax.grid(alpha=0.2)
    ax.legend(['Pilferage','Normal'])
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    # TOP HITS
    if len(pilf):
        st.subheader("Top Pilferage Spots")
        top = pilf.groupby('chainage')['leak size'].agg(['count','mean']).round(1).sort_values('count', ascending=False).head(8)
        st.dataframe(top, use_container_width=True)
    
    # DOWNLOAD
    csv = lds.to_csv(index=False).encode()
    st.download_button("💾 Download Results", csv, "pilferage_results.csv", use_container_width=True)

else:
    st.markdown("""
    ### 🚀 **Quick Start**
    1. Upload your **PIDWS** and **LDS** Excel files
    2. Adjust tolerance/window (defaults work great)
    3. Click **RUN ANALYSIS**
    
    **Expected columns:**
    ```
    PIDWS: Date, Time, chainage, Event Duration
    LDS: Date, Time, chainage, leak size
    ```
    """)
