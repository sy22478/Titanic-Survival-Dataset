# Titanic Survival Prediction

A machine learning project that predicts passenger survival on the Titanic using Python. The model analyzes various features like passenger class, gender, age, and fare to make predictions.

## Features

- Data preprocessing and exploratory data analysis
- Visualization of survival patterns using seaborn and matplotlib
- Implementation of Logistic Regression and Random Forest models
- Model evaluation with accuracy metrics and confusion matrices
- Feature importance analysis

## Requirements

- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## Installation

```bash
pip install numpy pandas matplotlib seaborn scikit-learn flask
```

## Usage

1. Place your Titanic dataset in the `data` folder as `train.csv`
2. Run the Jupyter notebook to train the model:
```bash
jupyter notebook Titanic-project.py
```

## Key Findings

- Gender was the most influential feature in survival prediction
- First-class passengers had higher survival rates
- Passengers who paid higher fares had better survival chances
- Family size impacted survival probability
