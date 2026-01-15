import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io

st.set_page_config(page_title="Pipeline Digging vs Leak Analyzer", layout="wide", page_icon="🛢️")

st.title("🛢️ Digging vs Leak Events Visualizer")
st.markdown("Upload manual digging and LDS datasets to analyze correlations at specific chainage.")

col1, col2 = st.columns(2)
with col1:
    digging_file = st.file_uploader("Upload df_manual_digging.xlsx", type=["xlsx"])
with col2:
    lds_file = st.file_uploader("Upload df_lds_IV.xlsx", type=["xlsx"])

if digging_file and lds_file:
    @st.cache_data
    def load_data(dig_file, lds_file):
        df_manual_digging = pd.read_excel(dig_file)
        df_manual_digging['DateTime'] = pd.to_datetime(df_manual_digging['DateTime'])
        
        df_lds_IV = pd.read_excel(lds_file)
        df_lds_IV['Date'] = pd.to_datetime(df_lds_IV['Date'])
        df_lds_IV['Time'] = pd.to_timedelta(df_lds_IV['Time'].astype(str))
        df_lds_IV['DateTime'] = df_lds_IV['Date'] + df_lds_IV['Time']
        return df_manual_digging, df_lds_IV

    df_manual_digging, df_lds_IV = load_data(digging_file, lds_file)

    st.success("✅ Data loaded successfully!")
    
    col3, col4 = st.columns(2)
    with col3:
        target_chainage = st.number_input("Target Chainage", value=89.0, step=0.1)
    with col4:
        tolerance = st.number_input("Tolerance (km)", value=1.0, step=0.1)

    df_digging_filtered = df_manual_digging[np.abs(df_manual_digging['Original_chainage'] - target_chainage) <= tolerance]
    df_leaks_filtered = df_lds_IV[np.abs(df_lds_IV['chainage'] - target_chainage) <= tolerance]

    st.info(f"📊 Digging events: {len(df_digging_filtered)}, Leak events: {len(df_leaks_filtered)}")

    if not df_digging_filtered.empty or not df_leaks_filtered.empty:
        fig, ax = plt.subplots(figsize=(18, 10))
        if not df_digging_filtered.empty:
            sns.scatterplot(data=df_digging_filtered, x='DateTime', y='Original_chainage', 
                           color='blue', label='Digging Events', marker='o', s=50, ax=ax)
        if not df_leaks_filtered.empty:
            sns.scatterplot(data=df_leaks_filtered, x='DateTime', y='chainage', 
                           color='red', label='Leak Events', marker='X', s=80, ax=ax)
        
        ax.set_title(f'Digging vs. Leak Events at Chainage {target_chainage:.1f} (Tolerance: {tolerance:.1f})')
        ax.set_xlabel('Date and Time')
        ax.set_ylabel('Chainage')
        ax.grid(True)
        ax.legend(title='Event Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.subplots_adjust(right=0.75)
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.warning(f"No events found for chainage {target_chainage} with tolerance {tolerance}.")
else:
    st.info("👆 Please upload both Excel files to start analysis.")

# Download filtered data option
if 'df_manual_digging' in locals() and 'df_lds_IV' in locals():
    csv_buffer = io.StringIO()
    combined = pd.concat([
        df_digging_filtered.assign(Type='Digging'),
        df_leaks_filtered.assign(Original_chainage='chainage').rename(columns={'chainage': 'Original_chainage'})
    ])
    combined.to_csv(csv_buffer, index=False)
    st.download_button("📥 Download Filtered Events (CSV)", csv_buffer.getvalue(), "filtered_events.csv")
