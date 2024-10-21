import os
import boto3
import joblib
from io import BytesIO

def save_model_to_s3(model, s3_key):
    """
    Save a serialized model to an S3 bucket using joblib, using environment variables for S3 configuration.

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
