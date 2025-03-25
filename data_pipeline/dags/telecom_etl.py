from airflow import DAG
from airflow.providers.snowflake.operators.snowflake import SnowflakeOperator
from airflow.utils.dates import days_ago

default_args = {
    'owner': 'airflow',
    'start_date': days_ago(1),
    'retries': 1,
}

with DAG(
    'telecom_etl',
    default_args=default_args,
    schedule_interval='@daily',
    catchup=False,
) as dag:

    load_data = SnowflakeOperator(
        task_id='load_data_into_snowflake',
        sql="""
            INSERT INTO TELECOMDB.PUBLIC.telecom_kpis(date, region, network_type, latency, download_speed, upload_speed, downtime, users_affected)
            VALUES (TO_DATE('2025-01-31', 'YYYY-MM-DD'), 'Lagos', '4G', 50.4, 90.3, 88.5, 9, 600);

        """,
        snowflake_conn_id='snowflake_conn',
    )

    load_data
