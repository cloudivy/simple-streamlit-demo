import streamlit as st
import pandas as pd
import numpy as np
import tsfel
from sklearn.preprocessing import StandardScaler
from ts2vec import TS2Vec
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch

st.set_page_config(page_title="Pipeline Anomaly Detection", layout="wide", page_icon="🔍")

st.title("🛢️ Pipeline Sensor Anomaly Detection")
st.markdown("Upload your sensor data CSV and detect anomalies using TSFEL + TS2Vec.")

# Sidebar for file upload and parameters
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Choose CSV file", type="csv", help="Expected columns: TIME, KAN, BV5, BV4, BV3, BV2, BV1, VIR")

sensor_cols = ['KAN', 'BV5', 'BV4', 'BV3', 'BV2', 'BV1', 'VIR']

# Parameters
st.sidebar.header("Parameters")
contamination = st.sidebar.slider("Isolation Forest Contamination", 0.01, 0.2, 0.05)
n_clusters = st.sidebar.slider("KMeans Clusters", 2, 5, 3)
window_length = st.sidebar.slider("TS2Vec Window Length", 32, 128, 64)
step_size = st.sidebar.slider("Step Size", 8, 32, 16)
percentile = st.sidebar.slider("TS2Vec Anomaly Percentile", 90, 99, 95)

if uploaded_file is not None:
    df3 = pd.read_csv(uploaded_file)
    st.success(f"Data loaded: {df3.shape[0]} rows, {df3.shape[1]} columns")
    
    if st.button("🚀 Run Anomaly Detection", type="primary"):
        with st.spinner("Processing data... This may take a few minutes."):
            # Preprocess
            df3['TIME'] = pd.to_datetime(df3['TIME'], format='%d:%b:%H:%M:%S', errors='coerce')
            for col in sensor_cols:
                if col in df3.columns:
                    df3[col] = pd.to_numeric(df3[col], errors='coerce')
                    df3[col] = df3[col].fillna(df3[col].mean())
                else:
                    st.warning(f"Column {col} not found in data.")
            
            X = df3[sensor_cols].values
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # TSFEL Features
            st.info("Extracting TSFEL features...")
            cfg = tsfel.get_features_by_domain()
            feature_matrix = None
            for idx, col in enumerate(sensor_cols):
                if idx < X_scaled.shape[1]:
                    feats = tsfel.time_series_features_extractor(cfg, X_scaled[:, idx], verbose=0)
                    if feature_matrix is None:
                        feature_matrix = feats
                    else:
                        feature_matrix = pd.concat([feature_matrix, feats], axis=1)
            
            # TS2Vec Windows
            def to_windows(arr, window=64, step=16):
                return np.array([arr[i:i+window] for i in range(0, len(arr)-window+1, step)])
            
            X_windows = to_windows(X, window=window_length, step=step_size)
            window_start_indices = list(range(0, len(X) - window_length + 1, step_size))
            window_timestamps = df3['TIME'].iloc[window_start_indices]
            
            # TS2Vec Model
            st.info("Training TS2Vec...")
            model = TS2Vec(input_dims=len(sensor_cols), device='cpu')
            model.fit(X_windows)
            seq_embeds = model.encode(X_windows)
            seq_embeds_averaged = np.mean(seq_embeds, axis=1)
            
            # Anomaly Detection TSFEL
            anomaly_detector_tsfel = IsolationForest(contamination=contamination, random_state=42)
            anomaly_labels_tsfel = anomaly_detector_tsfel.fit_predict(feature_matrix)
            anomalous_indices_tsfel = np.where(anomaly_labels_tsfel == -1)[0]
            
            # TS2Vec KMeans
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels_ts2vec = kmeans.fit_predict(seq_embeds_averaged)
            distances_ts2vec = np.linalg.norm(seq_embeds_averaged - kmeans.cluster_centers_[labels_ts2vec], axis=1)
            threshold_ts2vec = np.percentile(distances_ts2vec, percentile)
            anomalous_windows_ts2vec = np.where(distances_ts2vec > threshold_ts2vec)[0]
        
        # Results
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("TSFEL Anomalies", len(anomalous_indices_tsfel))
            st.metric("TS2Vec Anomalies", len(anomalous_windows_ts2vec))
        
        with col2:
            tsfel_pct = len(anomalous_indices_tsfel) / len(feature_matrix) * 100
            ts2vec_pct = len(anomalous_windows_ts2vec) / len(distances_ts2vec) * 100
            st.metric("TSFEL % Anomalies", f"{tsfel_pct:.2f}%")
            st.metric("TS2Vec % Anomalies", f"{ts2vec_pct:.2f}%")
        
        # Plot TS2Vec
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=window_timestamps, y=distances_ts2vec, 
                                mode='lines', name='Anomaly Score', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=window_timestamps.iloc[anomalous_windows_ts2vec], 
                                y=distances_ts2vec[anomalous_windows_ts2vec],
                                mode='markers', name='Anomalies', marker=dict(color='red', size=8)))
        fig.add_hline(y=threshold_ts2vec, line_dash="dash", line_color="red", annotation_text="Threshold")
        fig.update_layout(title="TS2Vec Anomaly Scores Over Time", xaxis_title="Timestamp", 
                         yaxis_title="Distance to Cluster Center", height=500, xaxis_tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Data tables
        st.subheader("Anomaly Indices")
        anomalies_df = pd.DataFrame({
            'Method': ['TSFEL']*len(anomalous_indices_tsfel) + ['TS2Vec']*len(anomalous_windows_ts2vec),
            'Index': np.concatenate([anomalous_indices_tsfel, anomalous_windows_ts2vec]),
            'Score': np.concatenate([anomaly_labels_tsfel[anomalous_indices_tsfel], distances_ts2vec[anomalous_windows_ts2vec]])
        })
        st.dataframe(anomalies_df.sort_values('Index'))
        
        # Download results
        csv = anomalies_df.to_csv(index=False)
        st.download_button("Download Results CSV", csv, "anomalies.csv", "text/csv")
        
        # Raw data preview
        st.subheader("Raw Data Preview")
        st.dataframe(df3.head())
        
else:
    st.info("👈 Please upload a CSV file with TIME and sensor columns in the sidebar.")

# Footer
st.markdown("---")
st.markdown("Built for pipeline monitoring using TSFEL & TS2Vec [web:1][web:21]")
