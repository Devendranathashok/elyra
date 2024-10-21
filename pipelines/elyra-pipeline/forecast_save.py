import os
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import boto3
import joblib
from io import BytesIO

def save_model_to_s3(model, s3_key):
    """
    Save a serialized model to an S3 bucket using joblib.
    
    :param model: The fitted SARIMAX model to save.
    :param s3_key: The key (path) where the model will be stored in S3.
    """
    # Fetch S3 credentials and configuration from environment variables
    s3_endpoint_url = os.environ.get("AWS_S3_ENDPOINT")
    s3_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    s3_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    s3_bucket_name = os.environ.get("AWS_S3_BUCKET")

    # Initialize S3 client with custom endpoint and credentials
    s3_client = boto3.client(
        's3',
        endpoint_url=s3_endpoint_url,  # Custom S3 endpoint (if using a service like MinIO)
        aws_access_key_id=s3_access_key,
        aws_secret_access_key=s3_secret_key
    )

    # Save the model to an in-memory bytes object using BytesIO and joblib
    model_byte_obj = BytesIO()
    joblib.dump(model, model_byte_obj)
    model_byte_obj.seek(0)  # Move pointer to the start of the file

    # Upload the model to S3
    s3_client.upload_fileobj(model_byte_obj, s3_bucket_name, s3_key)
    print(f"Model uploaded to s3://{s3_bucket_name}/{s3_key}")


def ticket_forecast_and_save(file_name="clean-data.csv", s3_key="models/sarimax_model.joblib"):
    """
    Train the SARIMAX model, generate a forecast, and save the model to S3.
    
    :param file_name: CSV file containing the dataset.
    :param s3_key: The S3 key where the model will be saved.
    """
    # Load the dataset
    data = pd.read_csv(file_name)
    data["Date"] = pd.to_datetime(data["Date"])
    data.set_index("Date", inplace=True)

    # Fit the SARIMAX model
    model = SARIMAX(data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
    model_fit = model.fit()

    # Save the model to S3
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
    print("Forecast saved to forecast-data.csv")


if __name__ == "__main__":
    # Specify file name and S3 key for saving the model
    ticket_forecast_and_save(file_name="clean-data.csv", s3_key="models/sarimax_model.joblib")
