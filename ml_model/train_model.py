import pandas as pd
import snowflake.connector
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# Snowflake connection details
SNOWFLAKE_USER = os.getenv("SNOWFLAKE_USER", "FMDEV")
SNOWFLAKE_PASSWORD = os.getenv("SNOWFLAKE_PASSWORD", "1Conceptzzz@#$")
SNOWFLAKE_ACCOUNT = os.getenv("SNOWFLAKE_ACCOUNT", "UGJTNYL-SN24834")
SNOWFLAKE_DATABASE = "TELECOMDB"

# Fetch data from Snowflake
def get_data():
    conn = snowflake.connector.connect(
        user=SNOWFLAKE_USER,
        password=SNOWFLAKE_PASSWORD,
        account=SNOWFLAKE_ACCOUNT,
        database=SNOWFLAKE_DATABASE
    )
    query = "SELECT * FROM TELECOMDB.PUBLIC.telecom_kpis"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# Load data
df = get_data()

# Feature Engineering
df["high_latency"] = df["LATENCY"] > 100  # Label high latency
#df["high_call_drops"] = df["CALL_DROP_RATE"] > 0.05  # Label high call drop rate

# Define features and target
X = df[["UPLOAD_SPEED", "DOWNLOAD_SPEED", "LATENCY"]]
y = df["CHURN"]  # Binary churn label

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save Model
joblib.dump(model, "churn_model.pkl")
