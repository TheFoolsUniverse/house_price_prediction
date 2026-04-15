from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import io
import os

app = Flask(__name__)

# ─────────────────────────────────────────────
# TRAIN MODEL ON STARTUP (using synthetic data)
# ─────────────────────────────────────────────

def generate_training_data(n=1000):
    """Generate realistic synthetic housing data for training."""
    np.random.seed(42)
    size       = np.random.randint(500, 5000, n)
    bedrooms   = np.random.randint(1, 6, n)
    bathrooms  = np.random.randint(1, 4, n)
    age        = np.random.randint(0, 50, n)
    garage     = np.random.randint(0, 3, n)
    location   = np.random.randint(1, 5, n)  # 1=rural, 2=suburb, 3=city, 4=premium

    price = (
        size * 120
        + bedrooms * 15000
        + bathrooms * 10000
        - age * 1500
        + garage * 12000
        + location * 25000
        + np.random.normal(0, 20000, n)
    )
    price = np.maximum(price, 50000)

    df = pd.DataFrame({
        "size_sqft": size,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "age_years": age,
        "garage_spaces": garage,
        "location_score": location,
        "price": price
    })
    return df


# Train on startup
df_train = generate_training_data()
FEATURES = ["size_sqft", "bedrooms", "bathrooms", "age_years", "garage_spaces", "location_score"]

X = df_train[FEATURES]
y = df_train["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

y_pred    = model.predict(X_test_scaled)
MODEL_R2  = round(r2_score(y_test, y_pred) * 100, 1)
MODEL_MAE = round(mean_absolute_error(y_test, y_pred))

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route("/")
def index():
    return render_template(
        "index.html", 
        r2=MODEL_R2, 
        mae=f"{MODEL_MAE:,}"
        )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        d = request.get_json()
        features = [[
            float(d["size_sqft"]),
            float(d["bedrooms"]),
            float(d["bathrooms"]),
            float(d["age_years"]),
            float(d["garage_spaces"]),
            float(d["location_score"]),
        ]]
        scaled = scaler.transform(features)
        price  = model.predict(scaled)[0]
        return jsonify({
            "price": round(price),
            "low":   round(price * 0.92),
            "high":  round(price * 1.08)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/predict-csv", methods=["POST"])
def predict_csv():
    try:
        raw = request.get_json().get("csv", "")
        df  = pd.read_csv(io.StringIO(raw))

        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            return jsonify({"error": f"Missing columns: {', '.join(missing)}"}), 400

        X_new    = df[FEATURES].astype(float)
        scaled   = scaler.transform(X_new)
        prices   = model.predict(scaled)

        results = []
        for i, row in df[FEATURES].iterrows():
            r = row.to_dict()
            r["predicted_price"] = round(prices[i])
            results.append(r)

        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🏠 House Price Predictor")
    print(f"   Model Accuracy (R²): {MODEL_R2}%")
    print(f"   Mean Absolute Error: ${MODEL_MAE:,}")
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
