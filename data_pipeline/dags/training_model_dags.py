from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
import mlflow
import mlflow.sklearn
import snowflake.connector  
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime, timedelta
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os
import logging
from mlflow.tracking import MlflowClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Snowflake credentials
SNOWFLAKE_CONFIG = {
    "user": "FMDEV",
    "password": "1Conceptzzz@#$",
    "account": "UGJTNYL-SN24834",
    "warehouse": "FMDEV_WAREHOUSE",
    "database": "TELECOMDB",
    "schema": "PUBLIC"
}

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://mlflow:5000")  # Ensure MLflow server is running

# DAG default arguments
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2025, 3, 20),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# Ensure data and MLflow artifact directories exist
os.makedirs("/opt/airflow/data", exist_ok=True)
os.makedirs("/opt/airflow/mlflow", exist_ok=True)


# Function to fetch data from Snowflake
def fetch_data():
    logger.info("Connecting to Snowflake...")

    try:
        conn = snowflake.connector.connect(**SNOWFLAKE_CONFIG)
        query = "SELECT * FROM TELECOMDB.PUBLIC.telecom_kpis"
        
        df = pd.read_sql(query, conn)
        conn.close()

        logger.info(f"Fetched {df.shape[0]} rows from Snowflake.")

        # ✅ Save data to the correct path
        df.to_csv("/opt/airflow/data/churn_data.csv", index=False)
        logger.info("Data saved successfully.")

    except Exception as e:
        logger.error(f"Error fetching data from Snowflake: {e}")
        raise


# Function to train ML model
# Function to train ML model
def train_model():
    logger.info("Loading dataset...")
    
    try:
        df = pd.read_csv("/opt/airflow/data/churn_data.csv")
        
        if "CHURN" not in df.columns:
            raise ValueError("CHURN column is missing in dataset!")
        
        # Convert 'Yes'/'No' in CHURN to 1/0
        df["CHURN"] = df["CHURN"].str.strip().map({"Yes": 1, "No": 0})

        logger.info("Performing feature engineering...")
        df["high_latency"] = df["LATENCY"] > 100  # Example feature

        # Define features and target
        X = df[["DOWNLOAD_SPEED", "UPLOAD_SPEED", "LATENCY"]]
        y = df["CHURN"]

        logger.info("Splitting dataset...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)

        logger.info("Training RandomForest model...")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Predictions & Metrics
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label=1, zero_division=1)
        recall = recall_score(y_test, y_pred, pos_label=1, zero_division=1)
        f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=1)

        logger.info(f"Model Performance: Accuracy={accuracy}, Precision={precision}, Recall={recall}, F1 Score={f1}")

        # ✅ Ensure MLflow experiment exists
        client = MlflowClient()
        experiment_name = "Churn_Prediction"

        if not client.get_experiment_by_name(experiment_name):
            experiment_id = client.create_experiment(experiment_name)
            logger.info(f"Created new MLflow experiment: {experiment_name}")
        else:
            experiment_id = client.get_experiment_by_name(experiment_name).experiment_id
            logger.info(f"Using existing MLflow experiment: {experiment_name}")

        mlflow.set_experiment(experiment_name)

        # ✅ Check if MLflow server is running before logging
        try:
            experiments = client.search_experiments()  # ✅ FIXED: Replaced `list_experiments()`
            logger.info(f"MLflow Experiments: {[exp.name for exp in experiments]}")
        except Exception as e:
            logger.error(f"MLflow server not reachable: {e}")
            raise

        # ✅ Log model & metrics in MLflow
        with mlflow.start_run():
            mlflow.log_param("model_type", "RandomForest")
            mlflow.log_param("n_estimators", 100)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)

            # ✅ Log full classification report
            from sklearn.metrics import classification_report
            report = classification_report(y_test, y_pred, output_dict=True)
            mlflow.log_dict(report, "classification_report.json")

            # ✅ Fix artifact path issue
            mlflow.sklearn.log_model(model, artifact_path="model")

        # ✅ Save trained model locally
        joblib.dump(model, "/opt/airflow/data/churn_model.pkl")
        logger.info("Model training complete and saved.")

    except Exception as e:
        logger.error(f"Error in training pipeline: {e}")
        raise


# Define DAG
with DAG(
    "train_churn_model",
    default_args=default_args,
    schedule_interval="0 0 * * *",  # Run daily at midnight
    catchup=False,
) as dag:

    fetch_data_task = PythonOperator(
        task_id="fetch_data",
        python_callable=fetch_data
    )

    train_model_task = PythonOperator(
        task_id="train_model",
        python_callable=train_model
    )

    deploy_model_task = BashOperator(
        task_id="deploy_model",
        bash_command="[ -f /opt/airflow/data/churn_model.pkl ] && cp /opt/airflow/data/churn_model.pkl /app/ml_model/churn_model.pkl || echo 'Model not found!'"
    )

    fetch_data_task >> train_model_task >> deploy_model_task
