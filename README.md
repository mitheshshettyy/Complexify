# Complexify - Code Complexity Analyzer

Complexify is a lightweight full-stack application that analyzes Python code snippets and returns a compact quality report through a simple web interface.

It uses a **FastAPI backend** and a **static frontend** to predict:
- ⏱️ **Time complexity**
- 💾 **Space complexity**
- 📝 **Readability score**
- 💡 **Optimization suggestions**

The goal of the project is to help users quickly understand how efficient, maintainable, and improvable a Python snippet may be.

---

## 🛠️ Tech Stack

### Backend
- FastAPI
- Python
- Scikit-learn
- AST
- Pickle

### Frontend
- HTML
- CSS
- JavaScript

### Data / ML
- TF-IDF vectorization
- Trained ML models
- JSONL dataset

---

## 1. 📁 Project Structure

```text
Complexify/
|- backend/
|  |- config.py               # Reads .env settings
|  |- main.py                 # FastAPI API (/ and /analyze)
|  |- ml/
|  |  |- features.py
|  |  |- preprocess.py
|  |  |- models.py
|  |  |- train.py
|  |- vectorizer.pkl
|  |- time_model.pkl
|  |- read_model.pkl
|  |- time_encoder.pkl
|- frontend/
|  |- index.html
|  |- style.css
|  |- script.js
|- dataset/
|  |- python_data.jsonl
|- .env.example
|- requirements.txt
```

## 2. Environment Configuration

1. Copy template:
```powershell
copy .env.example .env
```

2. Edit `.env` values as needed.

Recommended keys in `.env`:
```env
DATASET_PATH=dataset/python_data.jsonl
```

Important:
- `.env` is ignored in git.
- `.env.example` is safe to commit.

## 3. Setup

```powershell
cd d:\P\Complexify
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 4. Run Backend

```powershell
python -m uvicorn backend.main:app --reload --port 8000
```

Health check:
- Open `http://127.0.0.1:8000`
- Expected: `{"status":"Complexify API is running"}`

## 5. Run Frontend

```powershell
cd frontend
start index.html
```

Frontend calls: `http://127.0.0.1:8000/analyze`

## 6. API

### `GET /`
Returns health status.

### `POST /analyze`
Request body:
```json
{ "code": "def f(n):\n    return n" }
```

Response fields:
- `time_complexity`
- `space_complexity`
- `readability_score`
- `optimization_suggestions`

## 7. Retrain Models

The training script reads dataset path from `.env` key `DATASET_PATH` (defaults to `dataset/python_data.jsonl`).

```powershell
python -m backend.ml.train
```
