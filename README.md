# 🏠 House Price Predictor

An ML-powered web app that predicts house prices using Linear Regression.

## What It Does

- **Manual Input** — Type in house details and get an instant price prediction with a range
- **CSV Upload** — Upload a CSV file with multiple houses and get predictions for all of them

## Tech Stack

- **Python** — Core language
- **Scikit-learn** — Linear Regression model, StandardScaler, train/test split
- **Pandas & NumPy** — Data handling
- **Flask** — Web server
- **HTML/CSS/JS** — Frontend (single file, no frameworks)

## Project Structure

```
house_price_predictor/
├── app.py                ← main file (run this)
├── requirements.txt
├── sample_houses.csv     ← test this with the CSV upload feature
└── README.md
```

## Model Details

- **Algorithm:** Linear Regression
- **Training Data:** 1,000 synthetic samples
- **Features:** Size, bedrooms, bathrooms, age, garage, location
- **Accuracy:** ~95% R²
- **Validation:** 80/20 train/test split
