class EnvironmentVariablesMissing(Exception):
    pass


class S3ClientError(Exception):
    pass


class InvalidBucketName(Exception):
    pass


class MissingCredentials(Exception):
    pass


class SimilarityMethodNotAvailable(Exception):
    pass


class SimilarityMethodRequiresRating(Exception):
    pass


class ModelNotFittedYet(Exception):
    pass


class UserNotPresent(Exception):
    pass


class ColdStartProblem(Exception):
    pass


class NotA2DArray(Exception):
    pass


class FeaturesNotAllowedForMatrixFactorization(Exception):
    pass
