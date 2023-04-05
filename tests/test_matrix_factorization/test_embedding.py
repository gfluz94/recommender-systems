from typing import Dict, Tuple
import pytest

from recsys.matrix_factorization.embedding import EmbeddingMatrixFactorization
from recsys.utils.errors import ColdStartProblem


class TestEmbeddingMatrixFactorization(object):

    _LATENT_DIMENSION = 32
    _DIM_THRESHOLD = 8

    def test___init___WithDenseLayers(
        self, movielens_mappings: Tuple[Dict[int, int], Dict[int, int]]
    ):
        # OUTPUT
        user_mapping, item_mapping = movielens_mappings
        model = EmbeddingMatrixFactorization(
            latent_dimension=self._LATENT_DIMENSION,
            user_mapping=user_mapping,
            item_mapping=item_mapping,
            use_dense_layers=True,
        )
        model_layers = model.layers

        # EXPECTED
        dim = self._LATENT_DIMENSION // 2
        layers = 0
        while dim >= self._DIM_THRESHOLD:
            layers += 1
            dim //= 2

        expected_number_of_layers = 2 * (layers + 1) + 1

        # ASSERT
        assert len(model_layers) == expected_number_of_layers

    def test___init___WithoutDenseLayers(
        self, movielens_mappings: Tuple[Dict[int, int], Dict[int, int]]
    ):
        # OUTPUT
        user_mapping, item_mapping = movielens_mappings
        model = EmbeddingMatrixFactorization(
            latent_dimension=self._LATENT_DIMENSION,
            user_mapping=user_mapping,
            item_mapping=item_mapping,
            use_dense_layers=False,
        )
        model_layers = model.layers

        # EXPECTED
        expected_number_of_layers = 3

        # ASSERT
        assert len(model_layers) == expected_number_of_layers

    def test_get_user_embedding_RaisesColdStartProblem(
        self, movielens_mappings: Tuple[Dict[int, int], Dict[int, int]]
    ):
        user_mapping, item_mapping = movielens_mappings
        model = EmbeddingMatrixFactorization(
            latent_dimension=self._LATENT_DIMENSION,
            user_mapping=user_mapping,
            item_mapping=item_mapping,
            use_dense_layers=True,
        )
        with pytest.raises(ColdStartProblem):
            _ = model.get_user_embedding(user_id=20202021)

    def test_get_item_embedding_RaisesColdStartProblem(
        self, movielens_mappings: Tuple[Dict[int, int], Dict[int, int]]
    ):
        user_mapping, item_mapping = movielens_mappings
        model = EmbeddingMatrixFactorization(
            latent_dimension=self._LATENT_DIMENSION,
            user_mapping=user_mapping,
            item_mapping=item_mapping,
            use_dense_layers=True,
        )
        with pytest.raises(ColdStartProblem):
            _ = model.get_item_embedding(item_id=20202021)

    def test_predict_rating_for_user_item_RaisesColdStartProblem(
        self, movielens_mappings: Tuple[Dict[int, int], Dict[int, int]]
    ):
        user_mapping, item_mapping = movielens_mappings
        model = EmbeddingMatrixFactorization(
            latent_dimension=self._LATENT_DIMENSION,
            user_mapping=user_mapping,
            item_mapping=item_mapping,
            use_dense_layers=True,
        )
        with pytest.raises(ColdStartProblem):
            _ = model.predict_rating_for_user_item(user_id=1, item_id=20202021)

    def test_predict_rating_for_user_RaisesColdStartProblem(
        self, movielens_mappings: Tuple[Dict[int, int], Dict[int, int]]
    ):
        user_mapping, item_mapping = movielens_mappings
        model = EmbeddingMatrixFactorization(
            latent_dimension=self._LATENT_DIMENSION,
            user_mapping=user_mapping,
            item_mapping=item_mapping,
            use_dense_layers=True,
        )
        with pytest.raises(ColdStartProblem):
            _ = model.predict_rating_for_user(user_id=20202021)
