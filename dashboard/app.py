import requests
import streamlit as st
import pandas as pd
import plotly.express as px
import snowflake.connector
import os

# Snowflake connection details
SNOWFLAKE_USER = os.getenv("SNOWFLAKE_USER", "FMDEV")
SNOWFLAKE_PASSWORD = os.getenv("SNOWFLAKE_PASSWORD", "1Conceptzzz@#$")
SNOWFLAKE_ACCOUNT = os.getenv("SNOWFLAKE_ACCOUNT", "UGJTNYL-SN24834")
SNOWFLAKE_DATABASE = "TELECOMDB"


# FastAPI endpoint
API_URL = "http://localhost:8000/predict/"

# Connect to Snowflake
def get_snowflake_data():
    conn = snowflake.connector.connect(
        user=SNOWFLAKE_USER,
        password=SNOWFLAKE_PASSWORD,
        account=SNOWFLAKE_ACCOUNT,
        database=SNOWFLAKE_DATABASE
    )
    query = "SELECT * FROM TELECOMDB.PUBLIC.telecom_kpis ORDER BY DATE DESC LIMIT 100"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# Streamlit UI
st.title("📡 Telecom Network & Customer Performance Dashboard")

# Load data
df = get_snowflake_data()

# KPI Cards
col1, col2, col3 = st.columns(3)
col1.metric("Avg Latency Rate", f"{df['LATENCY'].mean():.2f}%")
col2.metric("Avg Download Speed (GB)", f"{df['DOWNLOAD_SPEED'].mean():.2f}")
col3.metric("Avg Upload Speed (ms)", f"{df['UPLOAD_SPEED'].mean():.2f}")

# Time Series Chart
fig = px.line(df, x="DATE", y="LATENCY", title="Call Drop Rate Over Time")
st.plotly_chart(fig)

# Tower-wise Analysis
fig2 = px.bar(df, x="REGION", y="DOWNLOAD_SPEED", title="Data Usage Per Tower")
st.plotly_chart(fig2)

# Show Raw Data
if st.checkbox("Show Raw Data"):
    st.dataframe(df)

# Churn Prediction Section
st.subheader("🔮 Churn Prediction")

# Select a customer for prediction
selected_row = st.selectbox("Select a Customer ID:", df["ID"])

# Extract customer data
customer_data = df[df["ID"] == selected_row].iloc[0]
input_data = {
    "UPLOAD_SPEED": customer_data["UPLOAD_SPEED"],
    "DOWNLOAD_SPEED": customer_data["DOWNLOAD_SPEED"],
    "LATENCY": customer_data["LATENCY"]
}

# Get Prediction
if st.button("Predict Churn"):
    try:
        response = requests.post(API_URL, json=input_data)

        # Debugging: Print raw response
        print("Raw API Response:", response.text)

        # Check if response is valid JSON
        response_data = response.json()

        if "churn_prediction" in response_data:
            churn_result = response_data["churn_prediction"]
            churn_text = "✅ Likely to Stay" if churn_result == "No" else "⚠️ Likely to Churn"
            st.write(f"**Prediction:** {churn_text}")
        else:
            st.error("API Response Error: " + str(response_data))

    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to API: {e}")

    except requests.exceptions.JSONDecodeError:
        st.error("Error: Received an invalid response from the API.")
        st.write(f"Raw Response: {response.text}")

# KPI Visualization
st.subheader("📊 Network Performance Trends")
fig = px.line(df, x="DATE", y="LATENCY", title="Latency Rate Over Time")
st.plotly_chart(fig)
