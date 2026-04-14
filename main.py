from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import logging
from schemas import (
    HousePredictionRequest,
    PredictionResponse,
    ModelInfoResponse,
    HealthCheckResponse
)
from api import (

    load_model_and_metadata,
    make_prediction,
    get_model_info,
    check_health
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown events"""
    # Startup: Load model and metadata
    logger.info("Starting House Price Prediction API...")
    success = load_model_and_metadata()
    if not success:
        logger.error("Failed to load model at startup")
    else:
        logger.info("Model and metadata loaded successfully")
    yield
    # Shutdown: Add cleanup logic here if needed
    logger.info("Shutting down House Price Prediction API...")

# Create FastAPI application
app = FastAPI(
    title="House Price Prediction API",
    description="Machine learning service for predicting US apartment prices based on 13 features",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """
    Check if the service is healthy and model is loaded.

    Returns:
        HealthCheckResponse with current service status
    """
    logger.info("Health check requested")
    health = check_health()

    return HealthCheckResponse(
        status=health["status"],
        model_loaded=health["model_loaded"],
        message=health["message"]
    )


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    """
    Get information about the loaded model.

    Returns:
        ModelInfoResponse with model metadata

    Raises:
        HTTPException: If model metadata is not loaded
    """
    logger.info("Model info requested")
    try:
        info = get_model_info()
        return ModelInfoResponse(
            model_type=info.get("model_type", "unknown"),
            version=info.get("version", "unknown"),
            features=info.get("features", []),
            training_date=info.get("training_date", "unknown"),
            rmse=info.get("rmse", 0.0),
            description=info.get("description", "No description available")
        )
    except ValueError as e:
        logger.error(f"Model info request failed: {e}")
        raise HTTPException(status_code=503, detail=str(e))

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: HousePredictionRequest):
    """
    Predict house price based on property features.

    Args:
        request: HousePredictionRequest with all 13 property features

    Returns:
        PredictionResponse with predicted price

    Raises:
        HTTPException: If model is not loaded or prediction fails
    """
    logger.info("Prediction request received")
    try:
        # Convert Pydantic model to dictionary
        features_dict = request.model_dump()

        # Make prediction using ML model
        predicted_price = make_prediction(features_dict)

        # Get model version from metadata
        model_info_data = get_model_info()
        model_version = model_info_data.get("version", "1.0.0")

        logger.info(f"Prediction completed: ${predicted_price:.2f}")

        return PredictionResponse(
            predicted_price=predicted_price,
            currency="USD",
            model_version=model_version
        )

    except ValueError as e:
        logger.error(f"Prediction failed - model not loaded: {e}")
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Root endpoint for basic information
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "House Price Prediction API",
        "version": "1.0.0",
        "endpoints": [
            "/health - Check service health",
            "/model/info - Get model information",
            "/predict - Make price prediction",
            "/docs - Interactive API documentation"
        ]
    }
