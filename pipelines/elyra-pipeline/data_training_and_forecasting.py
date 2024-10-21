import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from model_saver import save_model_to_s3  # Import the model saving function

def ticket_forecast(file_name="clean-data.csv", s3_key="models/sarimax_model.joblib"):
    # Load the dataset
    data = pd.read_csv(file_name)
    data["Date"] = pd.to_datetime(data["Date"])
    data.set_index("Date", inplace=True)

    # Fit the SARIMAX model
    model = SARIMAX(data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
    model_fit = model.fit()

    # Save the model to S3 using the function from model_saver.py
    save_model_to_s3(model_fit, s3_key)

    # Forecast the next 28 days
    forecast = model_fit.get_forecast(steps=28)
    forecast_index = pd.date_range(
        data.index[-1] + pd.DateOffset(days=1), periods=28, freq="D"
    )
    forecast_df = pd.DataFrame(
        {"n_tickets": forecast.predicted_mean.values}, index=forecast_index
    )
    forecast_df["n_tickets"] = forecast_df["n_tickets"].round().astype(int)

    # Save forecast to CSV
    forecast_df.to_csv("forecast-data.csv")


if __name__ == "__main__":
    ticket_forecast(file_name="clean-data.csv", s3_key="models/sarimax_model.joblib")
