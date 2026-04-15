# 🏠 House Price Predictor
🚀 **[Live Demo](https://house-price-prediction-vleh.onrender.com)**

> **Note on Performance:** This application is hosted on Render's free tier. If the site has been inactive, it may take **30-50 seconds** to "wake up" (cold start) while the server provisions resources. Once active, it will respond instantly.

An end-to-end Machine Learning web application that predicts residential property values using a trained Linear Regression model. This project bridges the gap between data science and software engineering by providing a functional web interface for model inference.

## ✨ Key Features
* **Real-time Prediction:** Enter house specifications (size, bedrooms, etc.) and receive an instant price estimation.
* **CSV Batch Processing:** Upload a `.csv` file with multiple listings to receive bulk predictions in a tabular format.
* **Synthetic Data Engine:** Includes a custom data generation script to train the model on 1,000 realistic housing samples.
* **Professional UI:** A modern, dark-themed dashboard built with responsive CSS for optimal viewing on all devices.

## 🛠️ Tech Stack
* **Backend:** Python (Flask)
* **Machine Learning:** Scikit-learn (Linear Regression, StandardScaler)
* **Data Handling:** Pandas, NumPy
* **Deployment:** Render + Gunicorn (WSGI server)
* **Frontend:** HTML5, CSS3, JavaScript (Vanilla)

## 📊 Model Performance
The model is trained using a standard 80/20 train-test split:
* **Accuracy (R² Score):** ~95%
* **Features Used:** Size (sqft), Bedrooms, Bathrooms, Age, Garage Spaces, and Location Score.

## 📂 Project Structure
```text
house_price_prediction/
├── app.py               # Main Flask application & ML logic
├── requirements.txt     # Python dependencies for deployment
├── sample_houses.csv    # Example file for batch prediction testing
├── templates/           # Frontend directory
│   └── index.html       # Single-page dashboard UI
└── .gitignore           # Standard Python ignore rules