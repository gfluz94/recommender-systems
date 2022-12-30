import pytest

from recsys.utils.aws import fetch_s3_files
from recsys.utils.errors import InvalidBucketName, MissingCredentials


class TestAWS(object):
    def test_fetch_s3_files_RaisesExceptionInvalidBucketName(self) -> None:
        with pytest.raises(InvalidBucketName):
            fetch_s3_files(
                bucket_name="",
                target_folder="data",
                aws_access_key="1234",
                aws_secret_key="1234",
            )

    def test_fetch_s3_files_RaisesExceptionInvalidCredentials(self) -> None:
        with pytest.raises(MissingCredentials):
            fetch_s3_files(
                bucket_name="data-ml-gfluz94",
                target_folder="data",
                aws_access_key=None,
                aws_secret_key=None,
            )
