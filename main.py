import os
from enum import Enum, auto
from argparse import ArgumentParser
import pandas as pd

from recsys.utils.aws import fetch_s3_files
from recsys.preprocessing import remove_extreme_movies_and_users
from recsys.utils.logging import logger
from recsys.utils.errors import EnvironmentVariablesMissing


class AlgorithmType(Enum):
    CollaborativeFiltering = auto()
    MatrixFactorization = auto()
    ContentBased = auto()


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
    parser.add_argument(
        "--movie-id-field",
        metavar="N",
        type=str,
        help="Name of the column containing movie ids.",
        default="movieId",
    )
    parser.add_argument(
        "--user-id-field",
        metavar="N",
        type=str,
        help="Name of the column containing user ids.",
        default="userId",
    )
    parser.add_argument(
        "--rating-field",
        metavar="N",
        type=str,
        help="Name of the column containing movie ids.",
        default="rating",
    )
    parser.add_argument(
        "--movie-n-ratings-lower",
        metavar="N",
        type=int,
        help="Lower threshhold for # of ratings a movie has got.",
        default=3,
    )
    parser.add_argument(
        "--movie-n-ratings-upper",
        metavar="N",
        type=int,
        help="Lower threshhold for # of ratings a movie has got.",
        default=None,
    )
    parser.add_argument(
        "--user-n-ratings-lower",
        metavar="N",
        type=int,
        help="Lower threshhold for # of ratings a user has given.",
        default=None,
    )
    parser.add_argument(
        "--user-n-ratings-upper",
        metavar="N",
        type=int,
        help="Lower threshhold for # of ratings a user has given.",
        default=10_000,
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

    logger.info("General preprocessing started...")
    n_users = df_ratings[args.user_id_field].nunique()
    n_movies = df_ratings[args.movie_id_field].nunique()
    logger.info("Dataset has %s unique movies and %s unique users.", n_movies, n_users)
    logger.info(
        "Getting rid of movies and users with too many and/or too few ratings..."
    )
    df_ratings = remove_extreme_movies_and_users(
        ratings_df=df_ratings,
        movie_id_column=args.movie_id_field,
        user_id_column=args.user_id_field,
        movie_ratings_boundaries=(
            args.movie_n_ratings_lower,
            args.movie_n_ratings_upper,
        ),
        user_ratings_boundaries=(args.user_n_ratings_lower, args.user_n_ratings_upper),
    )
    n_users = df_ratings[args.user_id_field].nunique()
    n_movies = df_ratings[args.movie_id_field].nunique()
    logger.info(
        "After preprocessing, dataset has %s unique movies and %s unique users.",
        n_movies,
        n_users,
    )
    logger.info("General preprocessing finished!")

    if args.algorithm == AlgorithmType.CollaborativeFiltering.name:
        pass
    elif args.algorithm == AlgorithmType.MatrixFactorization.name:
        pass
    elif args.algorithm == AlgorithmType.ContentBased.name:
        pass
