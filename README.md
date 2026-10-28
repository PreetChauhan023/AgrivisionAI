# 🌾 AgriVision AI — Intelligent Crop Health Monitoring System

A full-featured AI-powered agricultural intelligence platform built with **Streamlit**, **Scikit-learn**, and **OpenCV**.

---

## 📦 Project Structure

```
agrivision_ai/
├── app.py                          # Main Streamlit entry point
├── requirements.txt                # Python dependencies
├── README.md
├── .streamlit/
│   └── config.toml                 # Theme and server settings
├── data/                           # Auto-generated at first run
│   ├── agri_dataset.csv            # 10,000-row synthetic dataset
│   └── users.csv                   # User credentials (hashed)
└── modules/
    ├── __init__.py
    ├── auth.py                     # Login / Signup / Session management
    ├── generate_dataset.py         # Synthetic dataset generator
    ├── models.py                   # ML model training & inference
    ├── image_detection.py          # OpenCV disease detection pipeline
    ├── recommendations.py          # Rule-based recommendation engine
    ├── charts.py                   # Plotly chart factory functions
    ├── page_overview.py            # Overview tab
    ├── page_dataset.py             # Dataset Explorer tab
    ├── page_dashboard.py           # Dashboard tab
    ├── page_health_prediction.py   # Crop Health Prediction tab
    ├── page_disease_classification.py  # Disease Classification tab
    ├── page_image_detection.py     # Image Disease Detection tab
    ├── page_yield_prediction.py    # Yield Prediction tab
    ├── page_soil_analysis.py       # Soil Analysis tab
    ├── page_recommendations.py     # Recommendation System tab
    ├── page_model_comparison.py    # Model Comparison tab
    └── page_profile_settings.py    # Profile & Settings tabs
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the app

```bash
streamlit run app.py
```

### 3. Open in browser

The app opens automatically at **http://localhost:8501**

---

## 🔐 Authentication

- **Sign up** with a username, email and password (min 6 characters)
- Credentials stored securely in `data/users.csv` (SHA-256 hashed passwords)
- No duplicate usernames or emails allowed
- Session managed via `st.session_state`

---

## 📊 Dataset

The synthetic dataset is **auto-generated on first run** (10,000 rows × 30 columns) and saved to `data/agri_dataset.csv`. It includes:

| Category | Features |
|---|---|
| Crop Info | Crop_Type, Growth_Stage, Fertilizer_Used |
| Soil | Soil_Type, pH_Level, Nitrogen, Phosphorus, Potassium, Soil_Moisture |
| Environment | Temperature, Humidity, Rainfall, Sunlight_Hours, Season, Region |
| Plant Health | Leaf_Color, Leaf_Spots, NDVI_Index, Chlorophyll_Level, Leaf_Area |
| Conditions | Pest_Presence, Disease_Severity, Stress_Level, Water_Quality, Air_Quality |
| Targets | Disease_Label (multi-class), Healthy (binary), Yield (continuous) |

---

## 🧠 Machine Learning Models

| Model | Task | Target |
|---|---|---|
| Logistic Regression | Binary classification | `Healthy` (0/1) |
| Random Forest | Multi-class classification | `Disease_Label` (7 classes) |
| Linear Regression | Regression | `Yield` (t/ha) |

All models are trained at app startup and cached for the session.

---

## 📷 Image Disease Detection (OpenCV)

The image pipeline:
1. Convert to grayscale
2. Apply Canny edge detection
3. Extract colour fractions (green, yellow, brown, orange, white) in HSV space
4. Count disease spots via contour detection
5. Apply heuristic rules to map features → disease prediction
6. Return treatment & fertilizer recommendations

---

## 🌿 Tabs Overview

| # | Tab | Description |
|---|---|---|
| 1 | Overview | Project intro, key metrics, quick charts |
| 2 | Dataset Explorer | Browse, filter, download dataset |
| 3 | Dashboard | 8 interactive Plotly visualisations |
| 4 | Crop Health Prediction | LR model: Healthy / Unhealthy |
| 5 | Disease Classification | RF model: 7-class disease type |
| 6 | Image Disease Detection | OpenCV leaf image analysis |
| 7 | Yield Prediction | Linear Regression yield estimator |
| 8 | Soil Analysis | Soil health scoring with gauge chart |
| 9 | Recommendation System | Fertilizer / irrigation / crop engine |
| 10 | Model Comparison | Side-by-side model metrics & plots |
| 11 | Profile | Logged-in user information |
| 12 | Settings | Theme preferences, logout |

---

## 🛠️ Technologies

- **Python 3.10+**
- **Streamlit** — UI framework
- **streamlit-option-menu** — Sidebar navigation
- **Pandas / NumPy** — Data manipulation
- **Scikit-learn** — Machine learning models
- **OpenCV** (headless) — Image processing
- **Plotly** — Interactive charts
- **Pillow** — Image loading

---

## 📝 Notes

- All model training is cached with `@st.cache_resource` — runs only once per session
- Dataset generation is cached with `@st.cache_data`
- CSV download available on prediction, soil analysis, and dataset pages
- No external database required — everything runs locally with CSV files
## 🌐 Live Demo

**Try it here: [agrivisionai-023.streamlit.app](https://agrivisionai-023.streamlit.app/)**
