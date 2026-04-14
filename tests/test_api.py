import pytest
import numpy as np
from unittest.mock import patch, mock_open
import json
from api import load_model_and_metadata, make_prediction, get_model_info, check_health


class TestLoadModelAndMetadata:
    """Test cases for model and metadata loading"""

    @patch('api.joblib.load')
    @patch('api.open', new_callable=mock_open, read_data='{"test": "data"}')
    def test_load_success(self, mock_file, mock_joblib):
        """Test successful model and metadata loading"""
        mock_joblib.return_value = "mock_model"

        result = load_model_and_metadata()

        assert result is True
        mock_joblib.assert_called_once_with("model.pkl")
        mock_file.assert_called_once_with('model_metadata.json', 'r')

    @patch('api.joblib.load')
    def test_load_model_failure(self, mock_joblib):
        """Test model loading failure"""
        mock_joblib.side_effect = Exception("Model load failed")

        result = load_model_and_metadata()

        assert result is False

    @patch('api.joblib.load')
    @patch('api.open', new_callable=mock_open)
    @patch('api.json.load')
    def test_load_metadata_failure(self, mock_json, mock_file, mock_joblib):
        """Test metadata loading failure"""
        mock_joblib.return_value = "mock_model"
        mock_json.side_effect = Exception("JSON load failed")

        result = load_model_and_metadata()

        assert result is False


class TestMakePrediction:
    """Test cases for prediction logic"""

    def test_make_prediction_success(self):
        """Test successful prediction with valid input"""
        # First, set up global model and metadata
        import api
        api.model = MockModel()
        api.metadata = {
            "features": ["feature1", "feature2", "feature3"]
        }

        house_features = {
            "feature1": 1.0,
            "feature2": 2.0,
            "feature3": 3.0
        }

        result = make_prediction(house_features)

        assert isinstance(result, float)
        assert result == 42.0  # Mock prediction value

    def test_make_prediction_no_model(self):
        """Test prediction failure when model is not loaded"""
        import api
        api.model = None

        house_features = {"feature1": 1.0}

        with pytest.raises(ValueError, match="Model not loaded"):
            make_prediction(house_features)

    def test_make_prediction_missing_features(self):
        """Test prediction with missing features"""
        import api
        api.model = MockModel()
        api.metadata = {
            "features": ["feature1", "feature2", "feature3"]
        }

        # Missing feature3
        house_features = {
            "feature1": 1.0,
            "feature2": 2.0
        }

        with pytest.raises(KeyError):
            make_prediction(house_features)


class TestGetModelInfo:
    """Test cases for model info retrieval"""

    def test_get_model_info_success(self):
        """Test successful model info retrieval"""
        import api
        api.metadata = {
            "model_type": "RandomForest",
            "version": "1.0.0",
            "features": ["f1", "f2"],
            "training_date": "2023-01-01",
            "rmse": 123.45,
            "description": "Test model"
        }

        result = get_model_info()

        assert result["model_type"] == "RandomForest"
        assert result["version"] == "1.0.0"
        assert result["features"] == ["f1", "f2"]

    def test_get_model_info_no_metadata(self):
        """Test model info retrieval when metadata is not loaded"""
        import api
        api.metadata = None

        with pytest.raises(ValueError, match="Model metadata not loaded"):
            get_model_info()


class TestCheckHealth:
    """Test cases for health check logic"""

    def test_check_health_both_loaded(self):
        """Test health check when both model and metadata are loaded"""
        import api
        api.model = "mock_model"
        api.metadata = {"test": "data"}

        result = check_health()

        assert result["status"] == "healthy"
        assert result["model_loaded"] is True
        assert "successfully" in result["message"]

    def test_check_health_model_missing(self):
        """Test health check when model is not loaded"""
        import api
        api.model = None
        api.metadata = {"test": "data"}

        result = check_health()

        assert result["status"] == "unhealthy"
        assert result["model_loaded"] is False
        assert "not loaded" in result["message"]

    def test_check_health_metadata_missing(self):
        """Test health check when metadata is not loaded"""
        import api
        api.model = "mock_model"
        api.metadata = None

        result = check_health()

        assert result["status"] == "unhealthy"
        assert result["model_loaded"] is True  # Model is loaded, but metadata is not
        assert "not loaded" in result["message"]

    def test_check_health_both_missing(self):
        """Test health check when both model and metadata are missing"""
        import api
        api.model = None
        api.metadata = None

        result = check_health()

        assert result["status"] == "unhealthy"
        assert result["model_loaded"] is False
        assert "not loaded" in result["message"]


# Mock classes for testing
class MockModel:
    """Mock model for testing predictions"""

    def predict(self, X):
        """Return a mock prediction"""
        return np.array([42.0])