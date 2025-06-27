from flask import Flask, request, jsonify, Response
import requests
import pandas as pd
import numpy as np
import json
import os
import logging
import time
import traceback
import sys
from datetime import datetime
from pathlib import Path

from prometheus_client import Histogram, generate_latest, REGISTRY, Counter, Gauge, CONTENT_TYPE_LATEST, Summary
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Prometheus metrics - sama seperti sebelumnya
api_response_duration = Histogram('api_response_duration_seconds', 'API response duration')
api_request_count = Counter('api_request_count', 'Total number of requests to the API')
api_error_rate = Counter('api_error_rate', 'Total number of errors in API', ['error_type'])
api_status_codes = Counter('api_status_codes', 'HTTP status codes from the API', ['status_code'])
api_concurrent_requests = Gauge('api_concurrent_requests', 'Current number of concurrent API requests')

# MLflow-specific metrics
mlflow_request_duration = Histogram('mlflow_request_duration_seconds', 'MLflow model server response time')
mlflow_request_count = Counter('mlflow_request_count', 'Total requests to MLflow server')
mlflow_error_count = Counter('mlflow_error_count', 'MLflow server errors', ['error_type'])
mlflow_server_health = Gauge('mlflow_server_health', 'MLflow server health status (1=up, 0=down)')

# Model metrics
model_latency = Histogram('model_prediction_latency_seconds', 'Model prediction latency')
model_predictions_total = Counter('model_predictions_total', 'Total predictions made')
model_confidence_score = Histogram('model_confidence_score', 'Model confidence scores', 
                                 buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
prediction_class_distribution = Counter('prediction_class_distribution', 'Distribution of prediction classes', ['prediction_class'])
low_confidence_predictions = Counter('low_confidence_predictions', 'Predictions with low confidence')
high_risk_predictions = Counter('high_risk_predictions', 'High risk predictions')

# Threshold metrics
threshold_adjustments = Counter('threshold_adjustments_total', 'Total threshold adjustments')
threshold_override_requests = Counter('threshold_override_requests', 'Requests with custom threshold')
prediction_threshold_diff = Histogram('prediction_threshold_difference', 'Difference between default and custom threshold predictions')

def track_metrics(func):
    """Decorator to track metrics for API endpoints"""
    from functools import wraps
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        api_request_count.inc()
        api_concurrent_requests.inc()
        
        try:
            with api_response_duration.time():
                response = func(*args, **kwargs)
            
            if isinstance(response, tuple):
                status_code = response[1]
            else:
                status_code = 200
            
            api_status_codes.labels(status_code=str(status_code)).inc()
            return response
            
        except ValueError as e:
            api_error_rate.labels(error_type='validation_error').inc()
            api_status_codes.labels(status_code='400').inc()
            raise
        except requests.exceptions.RequestException as e:
            api_error_rate.labels(error_type='mlflow_connection_error').inc()
            api_status_codes.labels(status_code='503').inc()
            raise
        except Exception as e:
            api_error_rate.labels(error_type='internal_error').inc()
            api_status_codes.labels(status_code='500').inc()
            raise
        finally:
            api_concurrent_requests.dec()
    
    return wrapper


class MLflowModelProxy:
    """Enhanced proxy for MLflow model server with threshold support"""
    
    def __init__(self, mlflow_url="http://127.0.0.1:5001/invocations", threshold=0.5):
        self.mlflow_url = mlflow_url
        self.threshold = threshold
        self.server_info = {}
        self.last_health_check = None
        self.is_healthy = False
        self.feature_names = []
        self.model_metadata = {}
        
        # Expected feature names untuk loan model (sesuai dengan inference.py)
        self.expected_features = [
            "umur", "pendapatan", "skor_kredit", "jumlah_pinjaman", "rasio_pinjaman_pendapatan",
            "pekerjaan_Freelance", "pekerjaan_Kontrak", "pekerjaan_Tetap",
            "kategori_umur_Dewasa", "kategori_umur_Muda", "kategori_umur_Senior",
            "kategori_skor_kredit_Fair", "kategori_skor_kredit_Good", "kategori_skor_kredit_Poor",
            "kategori_pendapatan_Rendah", "kategori_pendapatan_Sedang", "kategori_pendapatan_Tinggi"
        ]
        
    def set_threshold(self, threshold):
        """Set prediction threshold for binary classification"""
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        
        old_threshold = self.threshold
        self.threshold = threshold
        threshold_adjustments.inc()
        
        logger.info(f"Threshold updated from {old_threshold} to {threshold}")
        
    def check_mlflow_server_health(self):
        """Check if MLflow server is responding"""
        try:
            # Try basic health check
            response = requests.get(
                self.mlflow_url.replace('/invocations', '/ping'),
                timeout=5
            )
            
            if response.status_code == 200:
                self.is_healthy = True
                mlflow_server_health.set(1)
                self.last_health_check = datetime.now()
                return True
                
        except:
            pass
        
        try:
            # Alternative: try simple prediction test
            test_data = {f"feature_{i}": 0.0 for i in range(len(self.expected_features))}
            response = requests.post(
                self.mlflow_url,
                headers={"Content-Type": "application/json"},
                json={"inputs": [test_data]},
                timeout=10
            )
            
            if response.status_code == 200:
                self.is_healthy = True
                mlflow_server_health.set(1)
                self.last_health_check = datetime.now()
                return True
                
        except Exception as e:
            logger.warning(f"MLflow health check failed: {e}")
        
        self.is_healthy = False
        mlflow_server_health.set(0)
        return False
    
    def get_model_info(self):
        """Get model information from MLflow server"""
        try:
            # Try to get model metadata
            info_url = self.mlflow_url.replace('/invocations', '/version')
            try:
                response = requests.get(info_url, timeout=5)
                if response.status_code == 200:
                    self.model_metadata = response.json()
            except:
                pass
            
            return {
                "model_loaded": self.is_healthy,
                "model_type": "MLflow Model Server",
                "mlflow_url": self.mlflow_url,
                "threshold": self.threshold,
                "last_health_check": self.last_health_check.isoformat() if self.last_health_check else None,
                "server_healthy": self.is_healthy,
                "expected_features": self.expected_features,
                "n_features": len(self.expected_features),
                "model_metadata": self.model_metadata,
                "supports_threshold": True,
                "supports_probabilities": True  # Assume MLflow model supports probabilities
            }
            
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {
                "model_loaded": False,
                "error": str(e),
                "threshold": self.threshold
            }
    
    def validate_input(self, data):
        """Validate and preprocess input data"""
        try:
            # Handle different input formats
            if isinstance(data, dict):
                if 'instances' in data:
                    # MLflow style
                    instances = data['instances']
                    if not isinstance(instances, list):
                        raise ValueError("'instances' must be a list")
                    df = pd.DataFrame(instances)
                else:
                    # Single instance
                    df = pd.DataFrame([data])
            elif isinstance(data, list):
                # List of instances
                df = pd.DataFrame(data)
            else:
                raise ValueError("Input must be dict or list")
            
            # Check feature alignment with expected features
            missing_features = set(self.expected_features) - set(df.columns)
            extra_features = set(df.columns) - set(self.expected_features)
            
            if missing_features:
                logger.warning(f"Missing features (will fill with 0): {missing_features}")
                for feature in missing_features:
                    df[feature] = 0
            
            if extra_features:
                logger.info(f"Extra features (will be ignored): {extra_features}")
            
            # Select and order features correctly
            df = df[self.expected_features]
            
            return df.to_dict('records')
            
        except Exception as e:
            raise ValueError(f"Input validation failed: {str(e)}")
    
    def call_mlflow_server(self, data):
        """Make request to MLflow server"""
        try:
            mlflow_request_count.inc()
            
            with mlflow_request_duration.time():
                response = requests.post(
                    self.mlflow_url,
                    headers={"Content-Type": "application/json"},
                    json={"inputs": data},
                    timeout=30
                )
            
            if response.status_code != 200:
                mlflow_error_count.labels(error_type=f'http_{response.status_code}').inc()
                raise requests.exceptions.RequestException(f"MLflow server returned {response.status_code}: {response.text}")
            
            return response.json()
            
        except requests.exceptions.Timeout:
            mlflow_error_count.labels(error_type='timeout').inc()
            raise
        except requests.exceptions.ConnectionError:
            mlflow_error_count.labels(error_type='connection_error').inc()
            raise
        except Exception as e:
            mlflow_error_count.labels(error_type='unknown').inc()
            raise
    
    def predict(self, data, custom_threshold=None):
        """Make predictions with configurable threshold"""
        if not self.is_healthy:
            self.check_mlflow_server_health()
            if not self.is_healthy:
                raise RuntimeError("MLflow server is not healthy")
        
        # Validate input
        validated_data = self.validate_input(data)
        
        # Call MLflow server
        mlflow_response = self.call_mlflow_server(validated_data)
        
        # Extract predictions
        predictions = mlflow_response.get("predictions", [])
        if not predictions:
            raise ValueError("No predictions returned from MLflow server")
        
        # Determine threshold to use
        threshold_to_use = custom_threshold if custom_threshold is not None else self.threshold
        
        # Check if we need to apply custom threshold
        apply_threshold = threshold_to_use != 0.5 or custom_threshold is not None
        
        if apply_threshold:
            if custom_threshold is not None:
                threshold_override_requests.inc()
            
            # Assume MLflow returns probabilities or we need to interpret predictions
            if isinstance(predictions[0], list) and len(predictions[0]) == 2:
                # MLflow returned probabilities [prob_class_0, prob_class_1]
                probabilities = predictions
                custom_predictions = []
                
                for prob_pair in probabilities:
                    # Apply custom threshold to probability of class 1
                    custom_pred = 1 if prob_pair[1] >= threshold_to_use else 0
                    custom_predictions.append(custom_pred)
                
                # Calculate difference from default threshold
                default_predictions = [1 if prob_pair[1] >= 0.5 else 0 for prob_pair in probabilities]
                diff_count = sum(1 for i in range(len(custom_predictions)) if custom_predictions[i] != default_predictions[i])
                if diff_count > 0:
                    prediction_threshold_diff.observe(diff_count / len(custom_predictions))
                
                result = {
                    "predictions": custom_predictions,
                    "probabilities": probabilities,
                    "threshold_used": threshold_to_use,
                    "threshold_applied": True,
                    "n_instances": len(custom_predictions),
                    "probability_class_0": [prob[0] for prob in probabilities],
                    "probability_class_1": [prob[1] for prob in probabilities],
                    "predictions_default_threshold": default_predictions
                }
                
            elif isinstance(predictions[0], (int, float)):
                # MLflow returned single values - assume they are probabilities
                probabilities = [[1-p, p] for p in predictions]
                custom_predictions = [1 if p >= threshold_to_use else 0 for p in predictions]
                
                result = {
                    "predictions": custom_predictions,
                    "probabilities": probabilities,
                    "threshold_used": threshold_to_use,
                    "threshold_applied": True,
                    "n_instances": len(custom_predictions),
                    "note": "Interpreted single values as probabilities for class 1"
                }
            else:
                # MLflow returned binary predictions - can't apply threshold
                result = {
                    "predictions": predictions,
                    "threshold_used": "not_applicable",
                    "threshold_applied": False,
                    "n_instances": len(predictions),
                    "note": "MLflow returned binary predictions, threshold not applicable"
                }
        else:
            # Use MLflow predictions as-is
            result = {
                "predictions": predictions,
                "threshold_used": "mlflow_default",
                "threshold_applied": False,
                "n_instances": len(predictions),
                "mlflow_response": mlflow_response
            }
        
        return result


# Global model proxy
model_proxy = MLflowModelProxy()

@app.route('/health', methods=['GET'])
@track_metrics
def health():
    """Health check endpoint"""
    health_status = model_proxy.check_mlflow_server_health()
    
    status = {
        "status": "healthy" if health_status else "unhealthy",
        "mlflow_server": "up" if health_status else "down",
        "mlflow_url": model_proxy.mlflow_url,
        "threshold": model_proxy.threshold,
        "timestamp": datetime.now().isoformat(),
        "last_health_check": model_proxy.last_health_check.isoformat() if model_proxy.last_health_check else None
    }
    
    status_code = 200 if health_status else 503
    return jsonify(status), status_code

@app.route('/info', methods=['GET'])
@track_metrics
def info():
    """Get model information"""
    return jsonify(model_proxy.get_model_info())

@app.route('/debug', methods=['GET'])
@track_metrics
def debug():
    """Debug endpoint for troubleshooting"""
    debug_info = {
        "mlflow_url": model_proxy.mlflow_url,
        "threshold": model_proxy.threshold,
        "server_healthy": model_proxy.is_healthy,
        "last_health_check": model_proxy.last_health_check.isoformat() if model_proxy.last_health_check else None,
        "expected_features": model_proxy.expected_features,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    }
    
    # Test MLflow connectivity
    try:
        response = requests.get(model_proxy.mlflow_url.replace('/invocations', '/ping'), timeout=5)
        debug_info["mlflow_ping_status"] = response.status_code
        debug_info["mlflow_ping_response"] = response.text[:200]
    except Exception as e:
        debug_info["mlflow_ping_error"] = str(e)
    
    # Test prediction
    try:
        test_data = {feature: 0.0 for feature in model_proxy.expected_features}
        result = model_proxy.call_mlflow_server([test_data])
        debug_info["test_prediction"] = "success"
        debug_info["test_prediction_result"] = result
    except Exception as e:
        debug_info["test_prediction_error"] = str(e)
    
    return jsonify(debug_info)

@app.route('/threshold', methods=['GET'])
@track_metrics
def get_threshold():
    """Get current threshold setting"""
    return jsonify({
        "threshold": model_proxy.threshold,
        "mlflow_url": model_proxy.mlflow_url,
        "server_healthy": model_proxy.is_healthy,
        "supports_threshold": True
    })

@app.route('/threshold', methods=['POST'])
@track_metrics
def set_threshold():
    """Set prediction threshold"""
    try:
        data = request.get_json()
        if data is None:
            return jsonify({"error": "No JSON data provided"}), 400
        
        if 'threshold' not in data:
            return jsonify({"error": "Missing 'threshold' field"}), 400
        
        threshold = data['threshold']
        
        if not isinstance(threshold, (int, float)):
            return jsonify({"error": "Threshold must be a number"}), 400
        
        if not 0 <= threshold <= 1:
            return jsonify({"error": "Threshold must be between 0 and 1"}), 400
        
        model_proxy.set_threshold(threshold)
        
        return jsonify({
            "status": "success",
            "message": f"Threshold updated to {threshold}",
            "threshold": model_proxy.threshold,
            "updated_at": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Set threshold error: {e}")
        return jsonify({
            "status": "error",
            "message": f"Failed to set threshold: {str(e)}"
        }), 500

@app.route('/mlflow_url', methods=['GET'])
@track_metrics
def get_mlflow_url():
    """Get current MLflow URL"""
    return jsonify({
        "mlflow_url": model_proxy.mlflow_url,
        "server_healthy": model_proxy.is_healthy,
        "last_health_check": model_proxy.last_health_check.isoformat() if model_proxy.last_health_check else None
    })

@app.route('/mlflow_url', methods=['POST'])
@track_metrics
def set_mlflow_url():
    """Set MLflow server URL"""
    try:
        data = request.get_json()
        if data is None:
            return jsonify({"error": "No JSON data provided"}), 400
        
        if 'mlflow_url' not in data:
            return jsonify({"error": "Missing 'mlflow_url' field"}), 400
        
        new_url = data['mlflow_url']
        old_url = model_proxy.mlflow_url
        
        # Update URL
        model_proxy.mlflow_url = new_url
        
        # Test new URL
        health_check = model_proxy.check_mlflow_server_health()
        
        return jsonify({
            "status": "success" if health_check else "warning",
            "message": f"MLflow URL updated from {old_url} to {new_url}",
            "mlflow_url": model_proxy.mlflow_url,
            "server_healthy": health_check,
            "updated_at": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Set MLflow URL error: {e}")
        return jsonify({
            "status": "error",
            "message": f"Failed to set MLflow URL: {str(e)}"
        }), 500

@app.route('/predict', methods=['POST'])
@track_metrics
def predict():
    """Make predictions with current threshold"""
    try:
        start_time = time.time()
        
        # Get input data
        data = request.get_json()
        if data is None:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Make prediction
        result = model_proxy.predict(data)
        
        # Monitoring
        try:
            latency = time.time() - start_time
            model_latency.observe(latency)
            model_predictions_total.inc()
            
            # Prediction distribution
            if 'predictions' in result and len(result['predictions']) > 0:
                prediction_value = result['predictions'][0]
                prediction_class_distribution.labels(prediction_class=str(prediction_value)).inc()
            
            # Confidence monitoring
            if 'probabilities' in result and len(result['probabilities']) > 0:
                prob_class_1 = result['probabilities'][0][1] if isinstance(result['probabilities'][0], list) else result['probabilities'][0]
                confidence = max(prob_class_1, 1 - prob_class_1)
                model_confidence_score.observe(confidence)
                
                if confidence < 0.6:
                    low_confidence_predictions.inc()
            
            # High risk monitoring
            if 'predictions' in result and len(result['predictions']) > 0:
                if result['predictions'][0] == 1:
                    high_risk_predictions.inc()
        
        except Exception as monitoring_error:
            app.logger.warning(f"Monitoring error: {monitoring_error}")
        
        return jsonify(result)
        
    except requests.exceptions.RequestException as e:
        logger.error(f"MLflow server error: {e}")
        return jsonify({"error": f"MLflow server error: {str(e)}"}), 503
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route('/predict_with_threshold', methods=['POST'])
@track_metrics
def predict_with_threshold():
    """Make predictions with optional custom threshold for this request only"""
    try:
        data = request.get_json()
        if data is None:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Extract threshold if provided
        custom_threshold = data.pop('threshold', None)
        
        if custom_threshold is not None:
            if not isinstance(custom_threshold, (int, float)):
                return jsonify({"error": "Threshold must be a number"}), 400
            if not 0 <= custom_threshold <= 1:
                return jsonify({"error": "Threshold must be between 0 and 1"}), 400
        
        # Make prediction with custom threshold
        result = model_proxy.predict(data, custom_threshold=custom_threshold)
        
        return jsonify(result)
        
    except ValueError as e:
        return jsonify({"error": f"Input validation error: {str(e)}"}), 400
    except requests.exceptions.RequestException as e:
        logger.error(f"MLflow server error: {e}")
        return jsonify({"error": f"MLflow server error: {str(e)}"}), 503
    except Exception as e:
        logger.error(f"Prediction with threshold error: {e}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route('/threshold_analysis', methods=['POST'])
@track_metrics
def threshold_analysis():
    """Analyze predictions across multiple threshold values"""
    try:
        data = request.get_json()
        if data is None:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Extract threshold range
        thresholds = data.pop('thresholds', [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        
        # Get base predictions with probabilities
        base_result = model_proxy.predict(data)
        
        if 'probabilities' not in base_result:
            return jsonify({"error": "Model doesn't provide probabilities for threshold analysis"}), 400
        
        probabilities = base_result['probabilities']
        
        # Analyze across thresholds
        results = []
        for threshold in thresholds:
            if isinstance(probabilities[0], list) and len(probabilities[0]) == 2:
                # Binary classification probabilities
                predictions = [1 if prob[1] >= threshold else 0 for prob in probabilities]
            else:
                # Single probability values
                predictions = [1 if prob >= threshold else 0 for prob in probabilities]
            
            positive_count = sum(predictions)
            negative_count = len(predictions) - positive_count
            
            results.append({
                "threshold": threshold,
                "predictions": predictions,
                "positive_predictions": positive_count,
                "negative_predictions": negative_count,
                "positive_rate": positive_count / len(predictions) if len(predictions) > 0 else 0
            })
        
        return jsonify({
            "probabilities": probabilities,
            "threshold_analysis": results,
            "n_instances": len(probabilities),
            "base_prediction": base_result
        })
        
    except Exception as e:
        logger.error(f"Threshold analysis error: {e}")
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

@app.route('/features', methods=['GET'])
@track_metrics
def get_features():
    """Get feature information and examples"""
    feature_info = {
        "expected_features": model_proxy.expected_features,
        "n_features": len(model_proxy.expected_features),
        "model_type": "MLflow Model Server",
        "mlflow_url": model_proxy.mlflow_url,
        "threshold": model_proxy.threshold,
        "server_healthy": model_proxy.is_healthy
    }
    
    # Create realistic examples for loan prediction
    example_approved = {
        "umur": 0.5, "pendapatan": 1.8, "skor_kredit": 1.2, "jumlah_pinjaman": 0.3, "rasio_pinjaman_pendapatan": -0.8,
        "pekerjaan_Freelance": False, "pekerjaan_Kontrak": False, "pekerjaan_Tetap": True,
        "kategori_umur_Dewasa": True, "kategori_umur_Muda": False, "kategori_umur_Senior": False,
        "kategori_skor_kredit_Fair": False, "kategori_skor_kredit_Good": True, "kategori_skor_kredit_Poor": False,
        "kategori_pendapatan_Rendah": False, "kategori_pendapatan_Sedang": False, "kategori_pendapatan_Tinggi": True
    }
    
    example_rejected = {
        "umur": -1.5, "pendapatan": -1.2, "skor_kredit": -1.3, "jumlah_pinjaman": 0.8, "rasio_pinjaman_pendapatan": 1.5,
        "pekerjaan_Freelance": True, "pekerjaan_Kontrak": False, "pekerjaan_Tetap": False,
        "kategori_umur_Dewasa": False, "kategori_umur_Muda": True, "kategori_umur_Senior": False,
        "kategori_skor_kredit_Fair": False, "kategori_skor_kredit_Good": False, "kategori_skor_kredit_Poor": True,
        "kategori_pendapatan_Rendah": True, "kategori_pendapatan_Sedang": False, "kategori_pendapatan_Tinggi": False
    }
    
    feature_info["examples"] = {
        "likely_approved": example_approved,
        "likely_rejected": example_rejected,
        "batch_prediction": [example_approved, example_rejected],
        "with_custom_threshold": {
            **example_approved,
            "threshold": 0.7
        },
        "mlflow_style": {
            "instances": [example_approved, example_rejected]
        }
    }
    
    return jsonify(feature_info)

@app.route('/test', methods=['GET'])
@track_metrics
def test_prediction():
    """Test prediction with sample data"""
    try:
        # Create test data
        test_data = {
            "umur": 0.5, "pendapatan": 1.2, "skor_kredit": 0.8, "jumlah_pinjaman": 0.3, "rasio_pinjaman_pendapatan": -0.2,
            "pekerjaan_Freelance": False, "pekerjaan_Kontrak": False, "pekerjaan_Tetap": True,
            "kategori_umur_Dewasa": True, "kategori_umur_Muda": False, "kategori_umur_Senior": False,
            "kategori_skor_kredit_Fair": False, "kategori_skor_kredit_Good": True, "kategori_skor_kredit_Poor": False,
            "kategori_pendapatan_Rendah": False, "kategori_pendapatan_Sedang": True, "kategori_pendapatan_Tinggi": False
        }
        
        # Make prediction
        result = model_proxy.predict(test_data)
        
        return jsonify({
            "test_input": test_data,
            "prediction_result": result,
            "current_threshold": model_proxy.threshold,
            "mlflow_url": model_proxy.mlflow_url,
            "status": "success"
        })
        
    except Exception as e:
        logger.error(f"Test prediction error: {e}")
        return jsonify({
            "error": f"Test prediction failed: {str(e)}",
            "status": "error",
            "mlflow_url": model_proxy.mlflow_url
        }), 500

@app.route('/metrics', methods=['GET'])
def metrics():
    """Prometheus metrics endpoint"""
    try:
        api_request_count.inc()
        metrics_data = generate_latest(REGISTRY)
        return Response(metrics_data, mimetype=CONTENT_TYPE_LATEST, status=200)
    except Exception as e:
        logger.error(f"Metrics endpoint error: {e}")
        return Response(f"# Error generating metrics: {str(e)}", mimetype="text/plain", status=500)

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    api_error_rate.labels(error_type='not_found').inc()
    api_status_codes.labels(status_code='404').inc()
    return jsonify({
        "error": "Endpoint not found",
        "available_endpoints": [
            "GET  /health - Health check",
            "GET  /info - Model information", 
            "GET  /debug - Debug information",
            "GET  /features - Feature information and examples",
            "GET  /test - Test prediction with sample data",
            "GET  /threshold - Get current threshold",
            "POST /threshold - Set threshold",
            "GET  /mlflow_url - Get MLflow server URL",
            "POST /mlflow_url - Set MLflow server URL",
            "POST /predict - Make predictions with current threshold",
            "POST /predict_with_threshold - Predict with custom threshold",
            "POST /threshold_analysis - Analyze multiple thresholds",
            "GET /metrics - Metrics for Prometheus"
        ]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    api_error_rate.labels(error_type='internal_server_error').inc()
    api_status_codes.labels(status_code='500').inc()
    return jsonify({
        "error": "Internal server error",
        "message": "Please check server logs for details"
    }), 500

@app.errorhandler(503)
def service_unavailable_error(error):
    """Handle 503 errors"""
    api_error_rate.labels(error_type='service_unavailable').inc()
    api_status_codes.labels(status_code='503').inc()
    return jsonify({
        "error": "Service unavailable",
        "message": "MLflow model server is not responding",
        "mlflow_url": model_proxy.mlflow_url
    }), 503

def initialize_app(mlflow_url="http://127.0.0.1:5001/invocations", threshold=0.5):
    """Initialize the application with MLflow configuration"""
    global model_proxy
    
    logger.info("Initializing MLflow Proxy Flask app...")
    
    # Set MLflow URL and threshold
    model_proxy.mlflow_url = mlflow_url
    model_proxy.threshold = threshold
    
    try:
        logger.info(f"Connecting to MLflow server: {mlflow_url}")
        logger.info(f"Default threshold: {threshold}")
        
        # Check MLflow server health
        health_check = model_proxy.check_mlflow_server_health()
        
        if health_check:
            logger.info("Application initialized successfully!")
            logger.info(f"   MLflow URL: {model_proxy.mlflow_url}")
            logger.info(f"   Server Status: Healthy")
            logger.info(f"   Threshold: {model_proxy.threshold}")
            logger.info(f"   Expected Features: {len(model_proxy.expected_features)}")
            return True
        else:
            logger.warning("MLflow server is not responding")
            logger.warning(f"   MLflow URL: {model_proxy.mlflow_url}")
            logger.warning("   Application will start but predictions may fail")
            return False
            
    except Exception as e:
        logger.error(f"Initialization error: {e}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        return False

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='MLflow Model Proxy Server with Monitoring and Threshold Support')
    parser.add_argument('--host', default='0.0.0.0', help='Host address')
    parser.add_argument('--port', type=int, default=5002, help='Port number')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--mlflow-url', default='http://127.0.0.1:5001/invocations', help='MLflow model server URL')
    parser.add_argument('--threshold', type=float, default=0.5, help='Default prediction threshold (0-1)')
    
    args = parser.parse_args()
    
    # Validate threshold
    if not 0 <= args.threshold <= 1:
        logger.error("Threshold must be between 0 and 1")
        sys.exit(1)
    
    # Initialize app
    init_success = initialize_app(args.mlflow_url, args.threshold)
    
    if init_success:
        logger.info(f"Starting MLflow proxy server at http://{args.host}:{args.port}")
        logger.info(f"   MLflow Server: {model_proxy.mlflow_url}")
        logger.info(f"   Default Threshold: {model_proxy.threshold}")
        logger.info(f"   Server Healthy: {model_proxy.is_healthy}")
    else:
        logger.warning("Starting server with MLflow connection issues...")
        logger.warning("   Check /debug endpoint for more information")
    
    # Run Flask app
    app.run(host=args.host, port=args.port, debug=args.debug)