# House Price Prediction API

A production-ready FastAPI service that predicts US apartment prices using machine learning. This API leverages a trained Random Forest model to provide accurate price estimates based on 13 key property features.

## 🚀 Key Features

- **Real-time Predictions**: Instant house price predictions using a trained ML model
- **Comprehensive Validation**: Input validation with Pydantic schemas for all 13 features
- **Health Monitoring**: Built-in health checks and model status monitoring
- **Model Metadata**: Detailed information about the trained model and its performance
- **RESTful API**: Clean, documented endpoints following REST principles
- **Error Handling**: Robust error handling with appropriate HTTP status codes

## 🛠 Tech Stack

- **Framework**: FastAPI (high-performance async web framework)
- **ML Library**: scikit-learn (Random Forest Regressor)
- **Data Processing**: NumPy, Joblib
- **Validation**: Pydantic (automatic request/response validation)
- **Language**: Python 3.8+

## 📋 API Endpoints

### GET `/health`
Check service health and model loading status.
```json
{
  "status": "healthy",
  "model_loaded": true,
  "message": "Model and metadata loaded successfully"
}
```

### GET `/model/info`
Retrieve detailed information about the trained model.
```json
{
  "model_type": "RandomForestRegressor",
  "version": "1.0.0",
  "features": ["total_images", "beds", "baths", ...],
  "training_date": "2025-11-17",
  "rmse": 15928.86,
  "description": "Random forest regressor"
}
```

### POST `/predict`
Make a house price prediction based on property features.
```json
// Request
{
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

// Response
{
  "predicted_price": 425000.00,
  "currency": "USD",
  "model_version": "1.0.0"
}
```

## 🏃‍♂️ Running Locally

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/house-price-api.git
   cd house-price-api
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   uvicorn main:app --reload
   ```

5. **Access the API**
   - API: http://localhost:8000
   - Interactive docs: http://localhost:8000/docs
   - Health check: http://localhost:8000/health

### Docker Setup

**Using Docker Compose (Recommended):**
```bash
docker-compose up --build
```

**Using Docker directly:**
```bash
docker build -t house-price-api .
docker run -p 8000:8000 house-price-api
```

### Running Tests

**Run all tests:**
```bash
pytest tests/
```

**Run with coverage:**
```bash
pytest tests/ --cov=api --cov=main --cov-report=html
```

**Run specific test file:**
```bash
pytest tests/test_api.py -v
```

## 🔮 Future Improvements

- **✅ Containerization**: Docker support added for easy deployment
- **✅ Testing**: Comprehensive unit and integration tests implemented
- **✅ Logging**: Structured logging with proper levels and formatting
- **✅ Modern FastAPI**: Updated to use lifespan events instead of deprecated startup events
- **Model Updates**: Implement model versioning and A/B testing capabilities
- **Batch Predictions**: Add support for predicting multiple properties simultaneously
- **Advanced Features**: Include image analysis for property photos
- **Database Integration**: Store prediction history and user feedback
- **Authentication**: Add API key authentication for production use
- **Monitoring**: Integrate logging and metrics collection (partially implemented)
- **Model Retraining**: Automated pipeline for model updates with new data

## 👨‍💻 Author

**Larry Butler**  
AI/ML Engineer  
Passionate about building scalable machine learning solutions and APIs.

---

*Built with ❤️ using FastAPI and scikit-learn*</content>
<parameter name="filePath">c:\Users\Thatr\Documents\Triple Ten\house_price_api (2)\house_price_api\README.md