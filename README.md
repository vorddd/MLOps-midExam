# 📦 MLOPS-MIDEXAM — Shipping Delay Prediction  
**Machine Learning • Deployment • CI/CD • Testing • Docker**

Repository ini berisi project Machine Learning untuk memprediksi apakah sebuah pengiriman **tiba tepat waktu** atau **terlambat** berdasarkan data operasional logistik.  
Proyek ini mencakup:

- Training model ML (scikit-learn)  
- Deployment aplikasi (FastAPI/Streamlit — sesuaikan)  
- CI/CD menggunakan GitHub Actions  
- Unit testing + data integrity testing  
- Containerization (Dockerfile)  
- Deployment ke HuggingFace Spaces  

---

## 📁 Project Structure

```
MLOPS-MIDEXAM/
│
├── .github/
│   └── workflows/
│       └── ci-cd.yml              # GitHub Actions (CI/CD Pipeline)
│
├── deployment/
│   ├── app.py                     # Main application (API/UI)
│   ├── eda.py                     # EDA script (optional)
│   ├── prediction.py              # Prediction logic (load model + inference)
│   ├── best_model_pipeline.joblib # Trained model pipeline
│   ├── preprocessing_pipeline.joblib
│   ├── requirements.txt           # Dependencies for deployment
│   └── shipping.csv               # Deployment resource (optional)
│
├── tests/
│   ├── __pycache__/
│   └── test_data_integrity.py     # Test for inference pipeline & dataset validation
│
├── Dockerfile                      # Container image for deployment
├── best_model_pipeline.joblib      # Model output from training (root copy)
├── preprocessing_pipeline.joblib    # Preprocessing pipeline (root copy)
├── shipping.csv                    # Main dataset used for training/testing
├── requirements-dev.txt           # Dev dependencies (pytest, linters, etc.)
├── iqbal_saputra.ipynb            # Main training notebook
├── iqbal_saputra_inf.ipynb        # Inference notebook (optional)
├── url.txt                        # Deployment URL (HuggingFace)
└── README.md                      # Project documentation
```

---

## 🚀 Overview

### 🎯 Objective  
Membangun sistem prediksi apakah paket *Reached on Time* (1) atau *Delayed* (0) berdasarkan data logistik.

### 🔍 Dataset  
- File: `shipping.csv`  
- Target: `Reached.on.Time_Y.N`  

Feature examples:
- Warehouse Block  
- Mode of Transport  
- Distance & Duration  
- Customer Rating  
- Cost of Delivery  
- Product Importance  
- dll.

---

## 🧠 Model

- Preprocessing menggunakan `ColumnTransformer`
- Scaling numerik + encoding kategorikal
- Model terbaik hasil tuning: **KNN**
- Pipeline disimpan sebagai:
  - `best_model_pipeline.joblib`
  - `preprocessing_pipeline.joblib`

Metrics (sesuaikan dengan hasil akhir):

- Accuracy: ...
- Precision: ...
- Recall: ...
- F1-score: ...

---

## 🧪 Unit Testing

Folder: `tests/test_data_integrity.py`

Test mencakup:

- Cek dataset tersedia & kolom wajib ada  
- Cek model bisa diload (`joblib`)  
- Inference `predict()` berjalan tanpa error  
- Jumlah output sesuai input  

Jalankan test:

```bash
pytest
```

---

## ⚙️ CI/CD Pipeline

### CI — Continuous Integration (GitHub Actions)

Workflow: `.github/workflows/ci-cd.yml`

Pipeline otomatis:

1. Install dependencies  
2. Install dev dependencies (`requirements-dev.txt`)  
3. Run unit tests (`pytest`)  
4. (Opsional) Linting / formatting  

Pipeline berjalan otomatis setiap push atau PR.

---

### CD — Continuous Deployment (HuggingFace Spaces)

- Repo ini terkoneksi ke **HuggingFace Space**
- Jika CI *lulus*:
  - Commit terbaru otomatis ditarik  
  - Dibuild ulang  
  - Dideploy ulang  

Tidak perlu upload manual.

---

## 🌐 Deployment

URL aplikasi tercantum pada:

```
url.txt
```

Contoh:

👉 Live App: **https://huggingface.co/spaces/USERNAME/NAMA-SPACE**

---

## 🖥️ Menjalankan Aplikasi Secara Lokal

Masuk folder:

```bash
cd deployment
pip install -r requirements.txt
```

Jika menggunakan **Streamlit**:

```bash
streamlit run app.py
```

Jika menggunakan **FastAPI**:

```bash
uvicorn app:app --reload
```

---

## 🐳 Docker Support

Build image:

```bash
docker build -t shipping-app .
```

Run container:

```bash
docker run -p 8080:8080 shipping-app
```

---

## 🔁 Re-Training Model

Training dilakukan melalui notebook:

```
iqbal_saputra.ipynb
```

Isi notebook:

- EDA  
- Preprocessing  
- Training model  
- Evaluasi  
- Export model pipeline `.joblib`  

---

## 📄 License

Project ini dibuat untuk keperluan akademik MLOps Mid Exam.

---
