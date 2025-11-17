---
title: MLOps Mid Exam - Shipping Delay Prediction
emoji: 📦
colorFrom: blue
colorTo: purple
sdk: streamlit
app_file: deployment/app.py
pinned: false
---

# MLOps Mid Exam – Shipping Delay Prediction

A lightweight MLOps project that predicts whether a shipment will arrive on time. The model is a scikit-learn pipeline (KNN + preprocessing) and the Streamlit app is deployed to Hugging Face Spaces while CI keeps the training artifacts healthy.

- **Demo**: https://huggingface.co/spaces/vorddd/MLOps-MidExam  
- **Model repo**: `vorddd/shipping-delay-knn`

## How It Works

- The training notebook exports `models/best_model_pipeline.joblib`.
- `deployment/prediction.py` loads that file from `models/` during development and from the Hugging Face Hub in production (via `hf_hub_download`).
- `deployment/app.py` stitches a simple overview page, an EDA tab (`deployment/eda.py`), and the prediction form.
- Runtime dependencies live in `deployment/requirements.txt`; dev/test tooling stays in `requirements-dev.txt`.

## Run Locally

```bash
python -m venv .venv
.venv\Scripts\activate          # or source .venv/bin/activate
pip install -r deployment/requirements.txt -r requirements-dev.txt
streamlit run deployment/app.py
```

Place the exported pipelines inside `models/` (already ignored in CD) and Streamlit will use them automatically. `pytest` runs the quick smoke tests.

## CI/CD

- **CI** (`.github/workflows/ci.yml`): runs on pushes/PRs to `main`, installs runtime + dev requirements, then executes `pytest`.
- **CD** (`.github/workflows/cd.yml`): mirrors the minimal app bundle (README + `deployment/` folder + requirements) into a temp directory and force-pushes it to the Hugging Face Space `vorddd/MLOps-MidExam` with `HF_TOKEN`. If the token is missing, the deploy step exits gracefully.

This setup keeps the repository easy to iterate on locally while ensuring the public app always downloads the latest pipeline from the Hub.
