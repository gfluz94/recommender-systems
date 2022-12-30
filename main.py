import os
from enum import Enum
from argparse import ArgumentParser
import pandas as pd

from recsys.utils.aws import fetch_s3_files
from recsys.utils.logging import logger
from recsys.utils.errors import EnvironmentVariablesMissing


class AlgorithmType(Enum):
    CollaborativeFiltering = 1
    MatrixFactorization = 2
    ContentBased = 3


MOVIES_FILE = "movies.csv"
RATINGS_FILE = "ratings.csv"


if __name__ == "__main__":
    logger.info("Defining parameters...")
    parser = ArgumentParser(
        description="Input parameters for designing and building a recommender system."
    )
    parser.add_argument(
        "algorithm",
        type=str,
        choices=list(AlgorithmType.__members__.keys()),
        default=AlgorithmType.ContentBased.name,
        help="Type of algorithm to be used when running the code.",
    )
    parser.add_argument(
        "--data-folder",
        metavar="N",
        type=str,
        help="Folder where data will be uploaded to.",
        default="data",
    )
    parser.add_argument(
        "--s3-data-bucket",
        metavar="N",
        type=str,
        help="Name of the bucket in AWS S3 where data is currelty stored.",
        default="data-ml-gfluz94",
    )
    parser.add_argument(
        "--aws-access-key-env",
        metavar="N",
        type=str,
        help="Name of the environment variable for AWS Access Key.",
        default="AWS_ACCESS_KEY",
    )
    parser.add_argument(
        "--aws-secret-key-env",
        metavar="N",
        type=str,
        help="Name of the environment variable for AWS Secret Key.",
        default="AWS_SECRET_KEY",
    )
    args = parser.parse_args()
    logger.info("Parameters defined!")

    logger.info("Finding and reading raw data...")
    DATA_FOLDER = os.path.join(os.curdir, args.data_folder)
    FILES_IN_FOLDER = os.listdir(DATA_FOLDER)
    if MOVIES_FILE not in FILES_IN_FOLDER or RATINGS_FILE not in FILES_IN_FOLDER:
        logger.info(
            f"Data not found in {DATA_FOLDER}. Downloading from `{args.s3_data_bucket}` S3 bucket..."
        )
        aws_access_key = os.getenv(key=args.aws_access_key_env)
        aws_secret_key = os.getenv(key=args.aws_secret_key_env)
        if not aws_access_key or not aws_secret_key:
            raise EnvironmentVariablesMissing("AWS Credentials not set accordingly.`")
        fetch_s3_files(
            bucket_name=args.s3_data_bucket,
            target_folder=DATA_FOLDER,
            aws_access_key=aws_access_key,
            aws_secret_key=aws_secret_key,
        )
        logger.info(f"Datasets successfully uploaded to local folder {DATA_FOLDER}...")
    df_ratings = pd.read_csv(os.path.join(DATA_FOLDER, RATINGS_FILE))
    df_movies = pd.read_csv(os.path.join(DATA_FOLDER, MOVIES_FILE))
    logger.info("Dataframes imported!")
