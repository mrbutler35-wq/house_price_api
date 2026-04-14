import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


class TestHealthEndpoint:
    """Test cases for the health check endpoint"""

    def test_health_endpoint_success(self):
        """Test health endpoint returns healthy status when model is loaded"""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "message" in data
        # Note: Actual status depends on whether model loads successfully

    def test_health_endpoint_structure(self):
        """Test health endpoint returns properly structured response"""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert isinstance(data["status"], str)
        assert isinstance(data["model_loaded"], bool)
        assert isinstance(data["message"], str)


class TestModelInfoEndpoint:
    """Test cases for the model info endpoint"""

    def test_model_info_success(self):
        """Test model info endpoint returns model metadata"""
        response = client.get("/model/info")
        # Status code depends on model loading - could be 200 or 503
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert "model_type" in data
            assert "version" in data
            assert "features" in data
            assert "training_date" in data
            assert "rmse" in data
            assert "description" in data

    def test_model_info_structure(self):
        """Test model info endpoint returns properly structured response when successful"""
        response = client.get("/model/info")

        if response.status_code == 200:
            data = response.json()
            assert isinstance(data["model_type"], str)
            assert isinstance(data["version"], str)
            assert isinstance(data["features"], list)
            assert isinstance(data["training_date"], str)
            assert isinstance(data["rmse"], (int, float))
            assert isinstance(data["description"], str)


class TestPredictionEndpoint:
    """Test cases for the prediction endpoint"""

    def test_prediction_success(self):
        """Test prediction endpoint with valid input"""
        # Sample valid request data
        request_data = {
            "total_images": 10,
            "beds": 3,
            "baths": 2.5,
            "area": 1800.0,
            "latitude": 40.7128,
            "longitude": -74.0060,
            "garden": 1,
            "garage": 1,
            "new_construction": 0,
            "pool": 0,
            "terrace": 1,
            "air_conditioning": 1,
            "parking": 1
        }

        response = client.post("/predict", json=request_data)
        # Status code depends on model loading - could be 200 or 503
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert "predicted_price" in data
            assert "currency" in data
            assert "model_version" in data
            assert isinstance(data["predicted_price"], (int, float))
            assert data["currency"] == "USD"

    def test_prediction_invalid_input(self):
        """Test prediction endpoint with invalid input"""
        # Invalid request - missing required fields
        request_data = {
            "total_images": 10,
            "beds": 3
            # Missing other required fields
        }

        response = client.post("/predict", json=request_data)
        assert response.status_code == 422  # Validation error

    def test_prediction_out_of_range_values(self):
        """Test prediction endpoint with out-of-range values"""
        request_data = {
            "total_images": 100,  # Over max limit of 50
            "beds": 3,
            "baths": 2.5,
            "area": 1800.0,
            "latitude": 40.7128,
            "longitude": -74.0060,
            "garden": 1,
            "garage": 1,
            "new_construction": 0,
            "pool": 0,
            "terrace": 1,
            "air_conditioning": 1,
            "parking": 1
        }

        response = client.post("/predict", json=request_data)
        assert response.status_code == 422  # Validation error


class TestRootEndpoint:
    """Test cases for the root endpoint"""

    def test_root_endpoint(self):
        """Test root endpoint returns API information"""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "endpoints" in data
        assert isinstance(data["endpoints"], list)