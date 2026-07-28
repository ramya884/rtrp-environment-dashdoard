import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def generate_data(num_records=500):
    np.random.seed(42)
    
    timestamps = pd.date_range(start="2024-01-01", periods=num_records, freq='h')
    temperature = np.random.uniform(15, 35, num_records)
    humidity = np.random.uniform(30, 90, num_records)
    air_quality = np.random.uniform(50, 150, num_records)
    energy_usage = 200 + (temperature * 15) + (humidity * 2) + np.random.normal(0, 50, num_records)
    water_consumption = 100 + (temperature * 8) + (energy_usage * 0.2) + np.random.normal(0, 30, num_records)
    carbon_emissions = (energy_usage * 0.3) + np.random.normal(0, 20, num_records)
    waste_generation = 10 + (water_consumption * 0.1) + np.random.normal(0, 10, num_records)
    
    df = pd.DataFrame({
        "Timestamp": timestamps,
        "Temperature (°C)": temperature,
        "Humidity (%)": humidity,
        "Air Quality Index": air_quality,
        "Water Consumption (Liters)": water_consumption,
        "Energy Usage (kWh)": energy_usage,
        "Carbon Emissions (kg)": carbon_emissions,
        "Waste Generation (kg)": waste_generation
    })
    
    return df

def convert_df_to_csv(df):
    output = BytesIO()
    df.to_csv(output, index=False)
    processed_data = output.getvalue()
    return processed_data

def train_models(df):
    X = df[["Temperature (°C)", "Humidity (%)", "Air Quality Index", "Water Consumption (Liters)", "Energy Usage (kWh)"]]
    y = df["Carbon Emissions (kg)"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Support Vector Machine": SVR()
    }
    
    results = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        results[name] = {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2 Score": r2}
    
    return results

# Streamlit UI
st.title("Environment & Resource Management Dashboard")

st.write("This application generates a dataset of environmental parameters for resource monitoring and analysis.")

df = generate_data()

st.subheader("Preview of Generated Dataset")
st.dataframe(df.head())

csv_data = convert_df_to_csv(df)

st.download_button(
    label="Download Dataset as CSV",
    data=csv_data,
    file_name="environment_resource_data.csv",
    mime="text/csv"
)

st.subheader("Train Multiple Machine Learning Models")
if st.button("Train Models"):
    results = train_models(df)
    st.success("Models Trained Successfully!")
    for model_name, metrics in results.items():
        st.subheader(model_name)
        st.write(f"Mean Absolute Error: {metrics['MAE']:.2f}")
        st.write(f"Mean Squared Error: {metrics['MSE']:.2f}")
        st.write(f"Root Mean Squared Error: {metrics['RMSE']:.2f}")
        st.write(f"R2 Score: {metrics['R2 Score']:.2f}")