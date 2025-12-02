# iPaDGNN: Joint Learning with Imputation, Pruning, and Distillation for Robust Lightweight Graph Neural Networks [[Paper]](https://thesis.lib.ncku.edu.tw/thesis/detail/df9bd469e5148d2d9ff385f47cd10220/)

[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.0-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-2.6.1-3C2179?style=for-the-badge&logo=python&logoColor=white)](https://pytorch-geometric.readthedocs.io/)
[![Optuna](https://img.shields.io/badge/Optuna-4.3.0-4A73B2?style=for-the-badge)](https://optuna.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](./LICENSE)

> **"Turning Dirty Data into Efficient Models."**
> A unified framework integrating **Imputation**, **Pruning**, and **Distillation** to accelerate GNN inference by **5x** while handling **99.5%** missing features. Designed for resource-constrained environments (e.g., single consumer GPU).

---

## 🏗️ Architecture

![img.png](assets%2Fimg.png)

The framework consists of three core modules designed to work in synergy:
1.  **APCFI (Imputation):** Recovers missing node features using parallel diffusion. (build on [PCFI](https://github.com/daehoum1/pcfi))
2.  **MPP (Pruning):** Compresses the model using a Mirror Projection technique.
3.  **MP-KRD (Distillation):** Transfers knowledge to a lightweight student model. (base on [KRD](https://github.com/LirongWu/KRD))

---

## 🚀 Key Features & Engineering

### 1. Ultra-Fast Imputation (APCFI)
* **Problem:** Traditional methods (like PCFI) use slow BFS for shortest-path calculation ($O(N^3)$ complexity).
* **Solution:** Implemented **Vectorized Matrix Operations** and **Sliding-Window Batching** to handle large graphs on limited VRAM.
* **Result:** **200x Speedup** (85s $\to$ 0.39s on CiteSeer).

### 2. Resource-Constrained Optimization
* **Mixed Precision Training (AMP):** Optimized for consumer-grade GPUs (e.g., RTX 4090), reducing memory usage while maintaining accuracy.
* **Robust Error Handling:** Implemented defensive programming patterns (`safe_rely`) with automatic garbage collection to ensure stability during long-running experiments.

### 3. Automated MLOps Pipeline
* **Closed-Loop Tuning:** Integrated **Optuna** for automated hyperparameter search.
* **Auto-Configuration:** Scripts automatically inject best parameters into YAML configs, eliminating manual copy-paste errors and ensuring reproducibility.

---

## 📂 Project Structure

A modularized architecture designed for extensibility and maintainability:

```text
iPaD-GNN/
├── limelight/          # [Core Library] The heart of the framework
│   ├── imputer/        # APCFI & Feature Propagation implementation
│   ├── model/          # TunedGNN, Mirror Projection logic
│   └── dataset/        # Data loading & preprocessing wrappers
├── engine/             # [Training Logic] OOP-based Trainers
│   ├── trainer.py      # Base Trainer class
│   ├── trainer_prune.py# Logic for Model Pruning (MPP)
│   └── trainer_kd.py   # Logic for Knowledge Distillation (MP-KRD)
├── tuning/             # [AutoML] Optuna objectives & optimization engines
├── configs/            # [Configuration] YAML-based experiment configs
├── scripts/            # [Batch Processing] Shell scripts for mass experiments
├── legacy/             # Archived experiments and prototypes
├── main.py             # Application entry point
└── requirements.txt    # Python dependencies
```


## 📦 Installation
```Bash
# Clone the repository
git clone [https://github.com/your-username/iPaDGNN.git](https://github.com/your-username/iPaDGNN.git)
cd iPaDGNN

# Install dependencies
pip install -r requirements.txt
```

## 🚴‍♂️ Running Experiments
```Bash
python enginn/main.py 
python enginn/main_sota.py 
```

🛠️ Tech Stack
- Deep Learning: PyTorch, PyTorch Geometric (PyG)
- Optimization: Torch-Pruning, Optuna

📜 Citation
If you find this code useful, please cite [our work](https://thesis.lib.ncku.edu.tw/thesis/detail/df9bd469e5148d2d9ff385f47cd10220/):
```BibTeX
@article{li2025ipadgnn,
  title={iPaDGNN: Joint Learning with Imputation, Pruning, and Distillation},
  author={Man-Ho Li},
  year={2025}
}
```