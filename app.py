import streamlit as st

st.title("🚀 Simple Pipeline Simulator")
st.write("Upload your CSV dataset to see leak detection results!")

uploaded_file = st.file_uploader("Choose CSV file", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())
    st.success("✅ File loaded successfully!")
