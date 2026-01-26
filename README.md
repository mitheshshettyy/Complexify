# Complexify - Code Complexity Analyzer

A web-based tool for analyzing code complexity metrics including time complexity, space complexity, cyclomatic complexity, and readability scores.

## Project Structure

```
complexify/
├── backend/
│   ├── main.py              # FastAPI server
│   └── ml/
│       ├── train.py         # Training pipeline
│       ├── preprocess.py    # NLP preprocessing
│       ├── models.py        # Model loading
│       ├── vectorizer.pkl
│       ├── time_model.pkl
│       ├── space_model.pkl
│       ├── cyclo_model.pkl
│       └── read_model.pkl
├── dataset/
│   └── code_complexity.csv
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
└── requirements.txt
```

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate


2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

1. Start the backend server:
```bash
python backend/main.py
```

2. Open your browser and navigate to:
```
http://localhost:8000
```

## Features

- **Code Analysis**: Paste code snippets for instant complexity analysis
- **Multiple Metrics**: Analyze time complexity, space complexity, cyclomatic complexity, and readability
- **ML-Based Predictions**: Uses trained machine learning models for accurate predictions
- **User-Friendly Interface**: Clean, modern web interface

## Development

To train new models, run:
```bash
python backend/ml/train.py
```
