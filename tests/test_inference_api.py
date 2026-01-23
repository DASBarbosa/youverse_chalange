from fastapi import FastAPI
from fastapi.testclient import TestClient
from apis.inference_api import InferenceAPI,InferenceApiSettings
import pytest


@pytest.fixture
def mock_app() -> FastAPI:
    app = FastAPI()
    api_Settings = InferenceApiSettings(
        IMAGE_LOADER_TYPE = None,
        MODEL_LOADER_TYPE = None,
        MODEL_PATH = "",
        LABELS_PATH = "",
        TOP_K = 1,
        NUM_THREADS = 1
    )
    InferenceAPI(app=app, api_settings=api_Settings)
    return app


def test_health_endpoint(mock_app):
    # Since using the mock does not trigger the startup method,
    # these are the expected values
    expected_response = {'status': 'ok', 'model_loaded': False, 'loading_error': None}
    client = TestClient(mock_app)
    response = client.get("/health")
    response_content = response.json()
    assert response.status_code == 200
    assert response_content == expected_response