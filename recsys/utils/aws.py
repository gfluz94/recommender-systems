import os
import boto3

from recsys.utils.errors import EnvironmentVariablesMissing, S3ClientError

SUCCESS_STATUS_CODE = 200


def fetch_s3_files(
    bucket_name: str,
    target_folder: str
) -> None:
    """Function that fetches data from specified S3 bucket and saves it to the target folder.
    
        Args:
            bucket_name (str): Name of the bucket in S3 where files are nested
            target_folder (str): Patht to folder where data is going to be saved.
    
        Raises:
            SolverUnsuccessful: In case optimization encounters a problem along the way.
    """
    ACCESS_KEY = os.getenv(key="AWS_ACCESS_KEY")
    SECRET_KEY = os.getenv(key="AWS_SECRET_KEY")
    if not ACCESS_KEY or not SECRET_KEY:
        raise EnvironmentVariablesMissing(
          "AWS Credentials not set in environment variables `ACCESS_KEY` and `SECRET_KEY`"
        )
        client = boto3.client(
            "s3", aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY
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
            Filename=os.path.join(target_folder, content["Key"])
        )
