import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn

# --- KONFIGURASI MLFLOW (PENTING!) ---
# Ganti dengan URI DagsHub Anda sendiri (Ambil dari tombol "Remote" di DagsHub)
mlflow.set_tracking_uri("https://dagshub.com/andiarifabc/Eksperimen_SML_AndiArif.mlflow")

# Setup Eksperimen
mlflow.set_experiment("California_Housing_Andi")

def main():
    with mlflow.start_run():
        # 1. Load Data (Contoh pakai data dummy atau load dataset sklearn)
        from sklearn.datasets import fetch_california_housing
        data = fetch_california_housing()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target

        # 2. Split Data
        X = df.drop('target', axis=1)
        y = df['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 3. Train Model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # 4. Evaluasi
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        print(f"MSE: {mse}")

        # 5. Log Metrics ke MLflow
        mlflow.log_metric("mse", mse)
        mlflow.sklearn.log_model(model, "model")

        # 6. Buat Plot & Log Artifact
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, predictions, alpha=0.5)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Actual vs Prediction")
        plt.savefig("pred_plot.png")
        
        mlflow.log_artifact("pred_plot.png")
        print("Training selesai & data terkirim ke MLflow!")

if __name__ == "__main__":
    main()