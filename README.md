# Complexify - Code Complexity Analyzer

A web-based tool for analyzing code complexity metrics including time complexity, cyclomatic complexity, and readability scores.

## Project Structure

```text
complexify/
|- backend/
|  |- main.py              # FastAPI server
|  |- ml/
|  |  |- train.py          # Training pipeline
|  |  |- preprocess.py     # NLP preprocessing
|  |  |- models.py         # Model loading
|  |  |- features.py       # AST feature extraction
|  |- vectorizer.pkl
|  |- time_model.pkl
|  |- cyclo_model.pkl
|  |- read_model.pkl
|  |- time_encoder.pkl
|- dataset/
|  |- python_data.jsonl
|- frontend/
|  |- index.html
|  |- style.css
|  |- script.js
|- requirements.txt
```

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

1. Start the backend server:
```bash
python -m uvicorn backend.main:app --reload
```

2. Open the frontend:
- Open `frontend/index.html` in your browser.

## Features

- Code Analysis: Paste code snippets for instant complexity analysis.
- Multiple Metrics: Analyze time complexity, cyclomatic complexity, and readability.
- Space Complexity: Currently returns `Unknown`.
- ML-Based Predictions: Uses trained machine learning models.

## Development

To train models:
```bash
python -m backend.ml.train
```

## License

MIT License
