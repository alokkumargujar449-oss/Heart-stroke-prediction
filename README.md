# ❤️ Heart Disease Risk Predictor

A Streamlit web app that predicts heart disease risk using a KNN classifier
trained on the Cleveland Heart Disease dataset.

---

## Project Structure

```
heart_disease_app/
├── heart.csv          ← dataset (you must supply this)
├── train_model.py     ← trains & saves model artifacts
├── app.py             ← Streamlit web app
├── requirements.txt
└── README.md
```

---

## Quick Start (Local)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add dataset
Place `heart.csv` in this folder.  
Download from: https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction

### 3. Train the model (run once)
```bash
python train_model.py
```
This generates: `knn.pkl`, `scaler.pkl`, `columns.pkl`

### 4. Launch the app
```bash
streamlit run app.py
```
Open http://localhost:8501 in your browser.

---

## Deploy to Streamlit Community Cloud (Free)

1. Push this folder to a **public GitHub repo**
2. Go to https://share.streamlit.io → **New app**
3. Select your repo, branch `main`, and set **Main file** to `app.py`
4. Click **Deploy** — done! You get a public URL like  
   `https://your-app.streamlit.app`

> **Note:** The `.pkl` model files must be committed to the repo
> (they are already generated locally by `train_model.py`).
> `heart.csv` does **not** need to be committed if the pkl files are present.

---

## Deploy to Hugging Face Spaces (Free, alternative)

1. Create a Space at https://huggingface.co/new-space
2. Choose **Streamlit** as the SDK
3. Upload all files (including the 3 `.pkl` files)
4. Your app is live instantly

---

## Model Details

| Item | Value |
|------|-------|
| Algorithm | K-Nearest Neighbours |
| Scaler | StandardScaler |
| Train/Test split | 80 / 20 |
| Typical accuracy | ~85–88% |
| Target | HeartDisease (0 = No, 1 = Yes) |

Features: Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS,
RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope

---

⚕️ *Educational purposes only — not a substitute for medical advice.*
