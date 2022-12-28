import os
import boto3

SUCCESS_STATUS_CODE = 200

def fetch_s3_files(
    bucket_name: str,
    target_folder: str
) -> None:
    ACCESS_KEY = os.getenv(key="AWS_ACCESS_KEY")
    SECRET_KEY = os.getenv(key="AWS_SECRET_KEY")
    if not ACCESS_KEY or not SECRET_KEY:
        raise ValueError(
          "AWS Credentials not set in environment variables `ACCESS_KEY` and `SECRET_KEY`"
        )
        client = boto3.client(
            "s3", aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY
        )
    content_response = client.list_objects(Bucket=bucket_name)
    if content_response["ResponseMetadata"]["HTTPStatusCode"] != SUCCESS_STATUS_CODE:
        raise ValueError(
          "Please check either bucket name or permissions to access S3 bucket!"
        )

    for content in content_response["Contents"]:
        client.download_file(
            Bucket=bucket_name,
            Key=content["Key"],
            Filename=os.path.join(target_folder, content["Key"])
        )
