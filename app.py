import streamlit as st
import pandas as pd
import joblib
import requests

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Sales Dashboard", layout="wide")

st.title("📊 Sales Prediction Dashboard")
st.markdown("ML Model + Power BI Integration")

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("predicted_sales.csv")

df = load_data()

# -------------------------------
# LOAD MODEL
# -------------------------------
@st.cache_resource
def load_model():
    return joblib.load("sales_model.pkl")

model = load_model()

# -------------------------------
# SIDEBAR INPUT
# -------------------------------
st.sidebar.header("🔧 Input Features")

year = st.sidebar.slider("Year", 2015, 2025, 2023)
month = st.sidebar.slider("Month", 1, 12, 6)
day = st.sidebar.slider("Day", 1, 31, 15)
quantity = st.sidebar.slider("Quantity", 1, 20, 5)

category = st.sidebar.selectbox("Category", [0, 1, 2])
region = st.sidebar.selectbox("Region", [0, 1, 2, 3])

# -------------------------------
# PREDICTION
# -------------------------------
input_data = pd.DataFrame([[year, month, day, quantity, category, region]],
                          columns=['Year','Month','Day','Quantity','Category','Region'])

prediction = model.predict(input_data)

st.sidebar.success(f"💰 Predicted Sales: {prediction[0]:,.2f}")

# -------------------------------
# KPI CARDS
# -------------------------------
col1, col2, col3 = st.columns(3)

col1.metric("Total Sales", f"{df['Sales'].sum():,.0f}")
col2.metric("Average Sales", f"{df['Sales'].mean():,.2f}")
col3.metric("Max Sales", f"{df['Sales'].max():,.0f}")

# -------------------------------
# CHART
# -------------------------------
st.subheader("📈 Sales vs Predicted Sales")
st.line_chart(df[['Sales', 'Predicted_Sales']])

# -------------------------------
# TABLE
# -------------------------------
st.subheader("📋 Dataset Preview")
st.dataframe(df.head(50))

# -------------------------------
# POWER BI (FROM GITHUB)
# -------------------------------
st.subheader("📊 Power BI Dashboard")

pbix_url = "https://github.com/shrutiiijadhav/Sales-Prediction-Dashboard/upload/superstore.pbix"


st.write("Download and open in Power BI Desktop 👇")

response = requests.get(pbix_url)

st.download_button(
    label="📥 Download Power BI File",
    data=response.content,
    file_name="Sales_Dashboard.pbix"
)

st.image("dashboard1.png", caption="Desktop View")
st.image("dashboard2.png", caption="Mobile View")

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.markdown("Created by Rushikesh Patil 🚀")