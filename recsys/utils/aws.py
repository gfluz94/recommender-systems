import os
import boto3

from recsys.utils.errors import S3ClientError, InvalidBucketName, MissingCredentials

SUCCESS_STATUS_CODE = "200"


def fetch_s3_files(
    bucket_name: str,
    target_folder: str,
    aws_access_key: str,
    aws_secret_key: str,
) -> None:
    """Function that fetches data from specified S3 bucket and saves it to the target folder.

    Args:
        bucket_name (str): Name of the bucket in S3 where files are nested
        target_folder (str): Patht to folder where data is going to be saved.
        aws_access_key (str): AWS Access Key in order to fetch data from S3
        aws_secret_key (str): AWS Secret Key in order to fetch data from S3

    Raises:
        S3ClientError: Download from S3 fails - Status Code is different from 200.
    """
    if bucket_name == "" or not isinstance(bucket_name, str):
        raise InvalidBucketName("Please provide a valid string for bucket name!")

    if not aws_access_key or not aws_secret_key:
        raise MissingCredentials("Please provide valid AWS Credentials!")

    client = boto3.client(
        "s3", aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key
    )
    content_response = client.list_objects(Bucket=bucket_name)
    if content_response["ResponseMetadata"]["HTTPStatusCode"] != SUCCESS_STATUS_CODE:
        raise S3ClientError(
            "Please check either bucket name or permissions to access S3 bucket!"
        )

    for content in content_response["Contents"]:
        client.download_file(
            Bucket=bucket_name,
            Key=content["Key"],
            Filename=os.path.join(target_folder, content["Key"]),
        )
