import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR

SEED = 2024
DATA_DIR = "data"
RESULT_JSON = "results/qnn_benchmark_results.json"

DATASET_FILES = {
    "yacht": "yacht_hydrodynamics.data",
    "energy": "ENB2012_data.xlsx",
    "concrete": "Concrete_Data.xls",
}


def load_dataset(dataset_name):
    path = os.path.join(DATA_DIR, DATASET_FILES[dataset_name])
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing dataset file: {path}")

    if dataset_name == "yacht":
        data = np.loadtxt(path)
        x = data[:, :-1]
        y = data[:, -1]
    elif dataset_name == "energy":
        data = pd.read_excel(path).values
        x = data[:, :-2]
        y = data[:, -2]
    elif dataset_name == "concrete":
        data = pd.read_excel(path).values
        x = data[:, :-1]
        y = data[:, -1]
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    sx = MinMaxScaler()
    sy = MinMaxScaler()
    x = sx.fit_transform(x)
    y = sy.fit_transform(y.reshape(-1, 1)).ravel()

    return train_test_split(x, y, test_size=0.2, random_state=SEED), sy


def maybe_load_qnn_results():
    if not os.path.exists(RESULT_JSON):
        return {}
    with open(RESULT_JSON, "r", encoding="utf-8") as file:
        payload = json.load(file)
    return payload if isinstance(payload, dict) else {}


def run_comparison():
    datasets = ["yacht", "energy", "concrete"]
    qnn_results = maybe_load_qnn_results()

    models = {
        "MLP": MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=1000, random_state=42),
        "SVR": SVR(C=10),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    }

    print(f"{'Dataset':<10} | {'Model':<12} | {'RMSE':<10} | {'Delta_vs_MLP':<12}")
    print("-" * 56)

    for dataset_name in datasets:
        (x_train, x_test, y_train, y_test), scaler_y = load_dataset(dataset_name)
        mlp_rmse = None

        for model_name, model in models.items():
            model.fit(x_train, y_train)
            pred_scaled = model.predict(x_test).reshape(-1, 1)
            pred = scaler_y.inverse_transform(pred_scaled)
            true = scaler_y.inverse_transform(y_test.reshape(-1, 1))
            rmse = float(np.sqrt(mean_squared_error(true, pred)))

            if model_name == "MLP":
                mlp_rmse = rmse

            print(f"{dataset_name:<10} | {model_name:<12} | {rmse:<10.4f} | {'-':<12}")

        qnn_rmse = qnn_results.get(dataset_name)
        if isinstance(qnn_rmse, (float, int)) and mlp_rmse is not None:
            delta = float(mlp_rmse - qnn_rmse)
            print(f"{dataset_name:<10} | {'HybridQNN':<12} | {qnn_rmse:<10.4f} | {delta:<+12.4f}")
        else:
            print(f"{dataset_name:<10} | {'HybridQNN':<12} | {'N/A':<10} | {'N/A':<12}")

        print("-" * 56)

    if not qnn_results:
        print("Tip: run `python benchmark_final_strategy.py` first to generate QNN RMSE.")


if __name__ == "__main__":
    run_comparison()
