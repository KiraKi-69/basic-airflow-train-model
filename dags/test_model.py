import airflow
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python_operator import PythonOperator
from datetime import timedelta
from airflow.utils.dates import days_ago

from kubernetes.client import models as k8s

default_args = {
    'owner': 'akotkova',    
    'retry_delay': timedelta(minutes=5),
}

IMAGE='harbor.neoflex.ru/dognauts/dognauts-airflow:2.5.3-py3.9-ml'

pod_override = k8s.V1Pod(
            spec=k8s.V1PodSpec(
                containers=[k8s.V1Container(name="base", image=IMAGE)],
            )
)

with DAG(
    dag_id = "run_csgb",
    default_args=default_args,
    schedule_interval='@once',
    dagrun_timeout=timedelta(minutes=60),
    description='run csgb model on batch data',
    start_date = airflow.utils.dates.days_ago(1),
    catchup=False
) as dag:

    def finetune_model():
        
        import pandas as pd
        import psycopg2 as pg
        import mlflow
        from sqlalchemy import create_engine
        
        dataset_name = "test_cs"

        engine = pg.connect("host=cassandra-postgresql.feast-db port=5432 dbname=FEAST_OFFLINE_STORE user=postgres password=postgres")
        df_test = pd.read_sql(f'select * from {dataset_name}', con=engine)

        model_name = "csgb"
        stage = "Production"

        model = mlflow.sklearn.load_model(f"models:/{model_name}/{stage}") 
        
        results = model.predict(df_test)
        
        df_test['results'] = results
        
        engine = create_engine('postgresql://postgres:postgres@cassandra-postgresql.feast-db:5432/FEAST_OFFLINE_STORE')
        df_test.to_sql('results_cs', engine, index=False, if_exists='replace')
        
                                
    run_model = PythonOperator(
        task_id="run_model",
        python_callable=finetune_model,
        provide_context=True,
        executor_config = {
        "pod_override": pod_override
    },
    )


run_model
