from contextlib import closing
import warnings

import pytest
from qdrant_client import QdrantClient


@pytest.fixture
def qdrant():
    """Exercise the Qdrant API locally; real ANN checks remain server tests."""
    with warnings.catch_warnings(), closing(QdrantClient(':memory:')) as client:
        warnings.filterwarnings('ignore', message='Payload indexes have no effect in the local Qdrant.*')
        warnings.filterwarnings('ignore', message='Local mode performs exact.*')
        yield client


@pytest.fixture
def qdrant_config():
    from arkb.knowledge.models import QdrantConfig
    return QdrantConfig()
