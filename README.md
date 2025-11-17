---
title: MLOps Mid Exam - Shipping Delay Prediction
emoji: 🧬
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# MLOps MidExam – Shipping Delay Prediction

End-to-end machine learning project to predict whether a shipment will arrive on time or late based on operational logistics data. The repository covers model training, deployment-ready artifacts, automated testing, CI/CD, containerization, and a public app on Hugging Face Spaces.

Live app: **https://huggingface.co/spaces/vorddd/MLOps-MidExam**

---

## Project Structure

```
MLOps-midExam/
├─ .github/workflows/
│  ├─ ci.yml                        # Test suite (CI)
│  └─ cd.yml                        # Build + deploy (CD)
├─ deployment/
│  ├─ app.py                        # Streamlit entry point
│  ├─ eda.py                        # Interactive visual analytics
│  ├─ prediction.py                 # Inference UI + model loader
│  ├─ best_model_pipeline.joblib    # Production pipeline
│  ├─ preprocessing_pipeline.joblib # (optional) preprocessing artifact
│  ├─ requirements.txt              # Runtime dependencies
│  └─ shipping.csv                  # Data copy for deployment
├─ tests/test_data_integrity.py     # Dataset & artifact smoke tests
├─ Dockerfile                       # Container definition for the app
├─ models/
│  ├─ best_model_pipeline.joblib    # Training output archive
│  └─ preprocessing_pipeline.joblib # Training preprocessing archive
├─ shipping.csv                     # Main dataset
├─ requirements-dev.txt             # Dev/test dependencies
├─ iqbal_saputra.ipynb              # Exploration + training notebook
├─ iqbal_saputa_inf.ipynb           # Inference walkthrough notebook
└─ url.txt                          # Deployment + dataset references
```

---

## Overview

### Objective
Predict the target `Reached.on.Time_Y.N` (1 = on time, 0 = delayed) using historical shipment information to support operational decisions.

### Dataset
- Source: Kaggle – Shipping data (`shipping.csv`)
- Key features: `Warehouse_block`, `Mode_of_Shipment`, `Customer_care_calls`, `Customer_rating`, `Cost_of_the_Product`, `Prior_purchases`, `Product_importance`, `Discount_offered`, `Weight_in_gms`.

---

## Model & Metrics

- Preprocessing: `ColumnTransformer` with scaling for numeric columns and encoding for categorical columns.
- Best estimator: **K-Nearest Neighbors** (selected after experimentation and hyperparameter tuning).
- Serialized artifacts: `best_model_pipeline.joblib` (inference-ready) plus `preprocessing_pipeline.joblib` for reference.

| Metric     | Score |
| ---------- | ----- |
| Accuracy   | 0.927 |
| Precision  | 0.940 |
| Recall     | 0.937 |
| F1-score   | 0.938 |

Scores are computed on the full dataset to give a reference point for downstream monitoring.

---

## Testing

`tests/test_data_integrity.py` ensures:

- `shipping.csv` exists with the expected schema.
- The serialized pipeline loads successfully and exposes `.predict`.

Run locally:

```bash
python -m venv .venv
.venv\Scripts\activate        # use `source .venv/bin/activate` on macOS/Linux
pip install -r deployment/requirements.txt -r requirements-dev.txt
pytest
```

---

## CI/CD Pipeline

- **CI (GitHub Actions)** – `.github/workflows/ci.yml`
  1. Set up Python 3.11 with pip caching.
  2. Install deployment + dev dependencies.
  3. Run the pytest suite.

- **CD (Hugging Face Spaces)** – `.github/workflows/cd.yml`
  1. Build and package the Streamlit Docker image.
  2. Prepare a lightweight copy of the repo using `rsync` (skipping notebooks, tests, dev tooling, etc.) while keeping the `models/` folder intact.
  3. Force-push the deployment bundle to the Hugging Face Space `vorddd/MLOps-MidExam`.

### Managing Binary Artifacts with git-xet

Large `.joblib` files inside `models/` are tracked via [git-xet](https://github.com/xetdata/git-xet) to avoid pushing huge blobs directly to Hugging Face or GitHub. To work with the repository locally:

1. Install git-xet (one-time):
   ```bash
   curl -L https://github.com/xetdata/xet-tools/releases/latest/download/git-xet -o git-xet
   sudo install -m 0755 git-xet /usr/local/bin/git-xet
   rm git-xet
   ```
2. Inside the repo, run:
   ```bash
   git xet install
   ```
   This configures the Git filters so pulling/pushing automatically uploads the binaries to XetHub.
3. Commit normally; git-xet deduplicates the `.joblib` artifacts behind the scenes.

The CI/CD workflows perform these steps automatically, ensuring both GitHub Actions and Hugging Face deploys always see the model files.

---

## Deployment

- Public URL: **https://huggingface.co/spaces/vorddd/MLOps-MidExam**
- Stack: Streamlit single-page app with two sections:
  - **Exploratory Data Analysis** – interactive Plotly visuals using tabs.
  - **Model Prediction** – form-based inference with probabilities.
- The Streamlit app loads `models/best_model_pipeline.joblib` directly, so the CD job copies that directory during deployment (no artifacts sit inside `deployment/`).

---

## Running Locally

```bash
cd deployment
pip install -r requirements.txt
streamlit run app.py
```

The Streamlit config in `app.py` matches the Hugging Face runtime, so the local experience mirrors production. Ensure `../models/best_model_pipeline.joblib` is present before launching.

---

## Docker

Build and run the containerized app:

```bash
docker build -t shipping-app .
docker run --rm -p 8501:8501 shipping-app
```

---

## Retraining

Re-run `iqbal_saputra.ipynb` to explore data, retrain models, and export updated `.joblib` artifacts. Place the refreshed files inside `models/`; the deployment workflow will include that directory when publishing to Hugging Face.

---

## License

Academic use only for the **MLOps MidExam** submission. Contact the author before reusing the assets commercially.
