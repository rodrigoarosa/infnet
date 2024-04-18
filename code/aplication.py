import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import mlflow

from sklearn.metrics import log_loss, f1_score

mlflow.set_tracking_uri("sqlite:///mlruns.db")

experiment_name = 'Aplicação - Projeto Kobe'
data_cols = ['lat', 'lon', 'minutes_remaining', 'period', 'playoffs', 'shot_distance']
target_col = 'shot_made_flag'


experiment = mlflow.get_experiment_by_name(experiment_name)

if experiment is None:
    experiment_id = mlflow.create_experiment(experiment_name)
    experiment = mlflow.get_experiment(experiment_id)

experiment_id = experiment.experiment_id

with mlflow.start_run(experiment_id=experiment_id, run_name='PipelineAplicacao'):
    
    model_uri = f'models:/model_kobe@staging'
    loaded_model = mlflow.sklearn.load_model(model_uri)
    data_prod = pd.read_parquet('../data/raw/dataset_kobe_prod.parquet')
    data_prod.dropna(inplace=True)

    data_prod.info()
    data_prod_features = data_prod[data_cols]

    Y = loaded_model.predict_proba(data_prod_features)[:,1]
    
    data_prod['predict_score'] = Y

    data_prod.to_parquet('../data/processed/prediction_prod.parquet')
    mlflow.log_artifact('../data/processed/prediction_prod.parquet')

    print(data_prod)

    mlflow.log_metric('log_loss', log_loss(data_prod[target_col], Y))
    mlflow.log_metric('f1_score', f1_score(data_prod[target_col], Y.round()))