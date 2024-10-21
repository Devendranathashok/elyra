import os
import boto3
import joblib
from io import BytesIO

def load_model_from_s3(s3_key):
    """
    Load a serialized model from an S3 bucket using joblib.

    :param s3_key: The key (path) where the model is stored in S3.
    :return: The loaded SARIMAX model.
    """

    # Fetch S3 credentials and configuration from environment variables
    s3_endpoint_url = os.environ.get("AWS_S3_ENDPOINT")
    s3_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    s3_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    s3_bucket_name = os.environ.get("AWS_S3_BUCKET")

    # Initialize S3 client with custom endpoint and credentials
    s3_client = boto3.client(
        's3',
        endpoint_url=s3_endpoint_url,
        aws_access_key_id=s3_access_key,
        aws_secret_access_key=s3_secret_key
    )

    # Download the model from S3 into an in-memory byte stream
    model_byte_obj = BytesIO()
    s3_client.download_fileobj(s3_bucket_name, s3_key, model_byte_obj)
    model_byte_obj.seek(0)

    # Load the model using joblib
    model = joblib.load(model_byte_obj)
    return model
