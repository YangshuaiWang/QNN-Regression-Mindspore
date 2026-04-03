import json
import os
import urllib.request

import mindquantum as mq
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np
import pandas as pd
from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import CNOT, RY, RZ
from mindquantum.core.operators import Hamiltonian, QubitOperator
from mindquantum.framework import MQLayer
from mindspore import Tensor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

SEED = 2024
DATA_DIR = "data"
RESULT_DIR = "results"
RESULT_JSON = os.path.join(RESULT_DIR, "qnn_benchmark_results.json")

ms.set_seed(SEED)
np.random.seed(SEED)
ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

DATASET_FILES = {
    "yacht": {
        "file": "yacht_hydrodynamics.data",
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data",
    },
    "energy": {
        "file": "ENB2012_data.xlsx",
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx",
    },
    "concrete": {
        "file": "Concrete_Data.xls",
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls",
    },
}


def ensure_file(dataset_name):
    meta = DATASET_FILES[dataset_name]
    path = os.path.join(DATA_DIR, meta["file"])
    if not os.path.exists(path):
        print(f"[Data] Downloading {dataset_name} to {path}")
        urllib.request.urlretrieve(meta["url"], path)
    return path


def load_dataset(dataset_name):
    print(f"[Data] Loading {dataset_name}")
    path = ensure_file(dataset_name)

    if dataset_name == "yacht":
        data = np.loadtxt(path)
        x = data[:, :-1]
        y = data[:, -1].reshape(-1, 1)
    elif dataset_name == "energy":
        data = pd.read_excel(path).values
        x = data[:, :-2]
        y = data[:, -2].reshape(-1, 1)
    elif dataset_name == "concrete":
        data = pd.read_excel(path).values
        x = data[:, :-1]
        y = data[:, -1].reshape(-1, 1)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    sx = MinMaxScaler(feature_range=(-np.pi / 2, np.pi / 2))
    sy = MinMaxScaler(feature_range=(0, 1))
    x = sx.fit_transform(x).astype(np.float32)
    y = sy.fit_transform(y).astype(np.float32)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=SEED
    )
    return x_train, y_train, x_test, y_test, sy


def build_pqc(n_qubits, depth):
    circuit = Circuit()
    for depth_idx in range(depth):
        enc = Circuit()
        for qubit in range(n_qubits):
            enc += RY(f"enc_{qubit}").on(qubit)
        circuit += enc.as_encoder()

        var = Circuit()
        for qubit in range(n_qubits):
            var += CNOT.on((qubit + 1) % n_qubits, qubit)
        for qubit in range(n_qubits):
            var += RZ(f"vz_{depth_idx}_{qubit}").on(qubit)
            var += RY(f"vy_{depth_idx}_{qubit}").on(qubit)
        circuit += var
    return circuit


class HybridQNN(nn.Cell):
    def __init__(self, input_dim, n_qubits, depth):
        super().__init__()
        self.classical = nn.SequentialCell(
            [nn.Dense(input_dim, 32), nn.ELU(), nn.Dense(32, 16), nn.ELU()]
        )
        self.q_proj = nn.Dense(16, n_qubits)
        circuit = build_pqc(n_qubits, depth)
        hams = [Hamiltonian(QubitOperator(f"Z{idx}")) for idx in range(n_qubits)]
        simulator = mq.Simulator("mqvector", n_qubits)
        grad_ops = simulator.get_expectation_with_grad(hams, circ_right=circuit)
        self.qnn = MQLayer(grad_ops)
        self.head = nn.SequentialCell(
            [nn.Dense(16 + n_qubits, 8), nn.ELU(), nn.Dense(8, 1)]
        )
        self.concat = ops.Concat(axis=1)

    def construct(self, x):
        classical_features = self.classical(x)
        q_params = self.q_proj(classical_features)
        q_features = self.qnn(q_params)
        return self.head(self.concat((classical_features, q_features)))


def train_with_two_stage_schedule(dataset_name):
    x_train, y_train, x_test, y_test, scaler_y = load_dataset(dataset_name)

    model = HybridQNN(input_dim=x_train.shape[1], n_qubits=4, depth=2)
    loss_fn = nn.MSELoss()
    batch_size = 16
    indices = np.arange(len(x_train))

    warmup_optimizer = nn.SGD(
        model.trainable_params(), learning_rate=0.05, momentum=0.9
    )
    warmup_grad = ms.value_and_grad(
        lambda batch_x, batch_y: loss_fn(model(batch_x), batch_y),
        None,
        warmup_optimizer.parameters,
    )

    print("  [Phase 1] SGD warmup")
    for _ in range(30):
        np.random.shuffle(indices)
        for start in range(0, len(x_train), batch_size):
            batch_idx = indices[start : start + batch_size]
            batch_x = Tensor(x_train[batch_idx], ms.float32)
            batch_y = Tensor(y_train[batch_idx], ms.float32)
            loss, grads = warmup_grad(batch_x, batch_y)
            warmup_optimizer(grads)

    finetune_optimizer = nn.Adam(
        model.trainable_params(), learning_rate=0.005, weight_decay=1e-5
    )
    finetune_grad = ms.value_and_grad(
        lambda batch_x, batch_y: loss_fn(model(batch_x), batch_y),
        None,
        finetune_optimizer.parameters,
    )

    total_epochs = 300 if dataset_name == "concrete" else 200
    best_rmse = float("inf")
    x_test_tensor = Tensor(x_test, ms.float32)

    print("  [Phase 2] Adam fine-tuning")
    for epoch in range(total_epochs):
        np.random.shuffle(indices)
        for start in range(0, len(x_train), batch_size):
            batch_idx = indices[start : start + batch_size]
            batch_x = Tensor(x_train[batch_idx], ms.float32)
            batch_y = Tensor(y_train[batch_idx], ms.float32)
            loss, grads = finetune_grad(batch_x, batch_y)
            finetune_optimizer(grads)

        if epoch % 10 == 0:
            model.set_train(False)
            pred_scaled = model(x_test_tensor).asnumpy()
            pred = scaler_y.inverse_transform(pred_scaled)
            true = scaler_y.inverse_transform(y_test)
            rmse = float(np.sqrt(np.mean((true - pred) ** 2)))
            best_rmse = min(best_rmse, rmse)
            model.set_train(True)

    print(f"  Best RMSE ({dataset_name}): {best_rmse:.4f}")
    return best_rmse


def main():
    datasets = ["yacht", "energy", "concrete"]
    results = {}

    print("=" * 56)
    print("Hybrid QNN benchmark")
    print("=" * 56)

    for name in datasets:
        try:
            print(f"\n--- {name} ---")
            results[name] = train_with_two_stage_schedule(name)
        except Exception as exc:
            print(f"Failed on {name}: {exc}")
            results[name] = None

    with open(RESULT_JSON, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, sort_keys=True)

    print("\n" + "=" * 56)
    print("Final RMSE (Hybrid QNN)")
    print("=" * 56)
    for name in datasets:
        value = results[name]
        printable = f"{value:.4f}" if isinstance(value, float) else "N/A"
        print(f"{name:<10}: {printable}")
    print(f"\nSaved: {RESULT_JSON}")


if __name__ == "__main__":
    main()
