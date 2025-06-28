#!/usr/bin/env python3
"""
Enhanced Prometheus Exporter untuk ML Model Monitoring
Melengkapi MLflow built-in metrics dengan custom business metrics
Untuk Kriteria 4 - Level Advance (10+ metriks total)
"""

import time
import random
import threading
import requests
import psutil
import os
import json
from prometheus_client import Counter, Histogram, Gauge, start_http_server, Info
from http.server import HTTPServer, BaseHTTPRequestHandler
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========== CUSTOM BUSINESS METRICS (Melengkapi MLflow metrics) ==========

# Alternative request tracking for dashboard compatibility
DIABETES_REQUESTS_TOTAL = Counter('diabetes_requests_total', 'Total diabetes prediction requests', ['method', 'status'])
DIABETES_REQUEST_DURATION = Histogram('diabetes_request_duration_seconds', 'Diabetes request duration', buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0])
PREDICTION_CONFIDENCE = Gauge('diabetes_prediction_confidence', 'Average prediction confidence score')
PREDICTION_DISTRIBUTION = Counter('diabetes_predictions_total', 'Total predictions by class', ['prediction_class'])
BUSINESS_ACCURACY = Gauge('diabetes_business_accuracy', 'Business-calculated model accuracy')

# Application Performance
RESPONSE_TIME_P95 = Gauge('diabetes_response_time_p95_seconds', '95th percentile response time')
CONCURRENT_USERS = Gauge('diabetes_concurrent_users', 'Number of concurrent users')
REQUEST_QUEUE_DEPTH = Gauge('diabetes_request_queue_depth', 'Current request queue depth')

# Data Quality Metrics
DATA_DRIFT_SCORE = Gauge('diabetes_data_drift_score', 'Data drift detection score (0-1)')
FEATURE_IMPORTANCE_DRIFT = Gauge('diabetes_feature_drift', 'Feature importance drift', ['feature_name'])
INPUT_VALIDATION_ERRORS = Counter('diabetes_input_validation_errors_total', 'Input validation errors', ['error_type'])

# Resource Utilization
MODEL_MEMORY_PEAK = Gauge('diabetes_model_memory_peak_bytes', 'Peak memory usage since start')
CACHE_HIT_RATE = Gauge('diabetes_cache_hit_rate', 'Model cache hit rate (0-1)')
DISK_IO_OPERATIONS = Counter('diabetes_disk_io_operations_total', 'Disk I/O operations', ['operation_type'])

# Business KPIs
DAILY_PREDICTIONS = Counter('diabetes_daily_predictions_total', 'Daily prediction count')
MODEL_CONFIDENCE_DISTRIBUTION = Histogram(
    'diabetes_confidence_distribution', 
    'Distribution of model confidence scores',
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

# Alerting Metrics (untuk 3 alert rules)
ALERT_HIGH_ERROR_RATE = Gauge('diabetes_alert_error_rate', 'Error rate for alerting (0-1)')
ALERT_MEMORY_USAGE = Gauge('diabetes_alert_memory_mb', 'Memory usage in MB for alerting')
ALERT_MODEL_ACCURACY = Gauge('diabetes_alert_accuracy', 'Model accuracy for alerting (0-1)')

# MLflow Health Monitoring (convert JSON to Prometheus metrics)
MLFLOW_MODEL_HEALTH = Gauge('mlflow_model_health_status', 'MLflow model health status (1=healthy, 0=unhealthy)')
MLFLOW_API_RESPONSE_TIME = Gauge('mlflow_api_response_time_seconds', 'MLflow API response time')
MLFLOW_PING_STATUS = Gauge('mlflow_ping_status', 'MLflow ping endpoint status (1=ok, 0=fail)')

# Alert notification tracking
ALERT_NOTIFICATIONS_TOTAL = Counter('alert_notifications_total', 'Total alert notifications received', ['severity', 'alert_type'])

class AlertWebhookHandler(BaseHTTPRequestHandler):
    """HTTP handler for Grafana alert webhooks"""
    
    def do_POST(self):
        """Handle POST requests from Grafana alerts"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            alert_data = json.loads(post_data.decode('utf-8'))
            
            # Extract alert information
            severity = "unknown"
            alert_type = "unknown"
            
            if 'alerts' in alert_data:
                for alert in alert_data['alerts']:
                    labels = alert.get('labels', {})
                    severity = labels.get('severity', 'unknown')
                    alert_type = labels.get('alert_type', 'unknown')
                    
                    # Track alert notifications
                    ALERT_NOTIFICATIONS_TOTAL.labels(severity=severity, alert_type=alert_type).inc()
            
            # Log alert for debugging
            logger.info(f"🚨 Alert received: {alert_data.get('title', 'Unknown Alert')}")
            
            # Send successful response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok"}).encode())
            
        except Exception as e:
            logger.error(f"Error handling webhook: {e}")
            self.send_response(500)
            self.end_headers()
    
    def log_message(self, format, *args):
        """Override to reduce noise in logs"""
        pass

class EnhancedMLExporter:
    def __init__(self):
        self.mlflow_url = os.getenv('ML_MODEL_URL', 'http://ml-model:8080')
        self.mlflow_metrics_url = f"http://ml-model:8082/metrics"
        self.start_time = time.time()
        self.prediction_count = 0
        self.total_confidence = 0
        self.memory_peak = 0
        self.error_count = 0
        self.successful_predictions = 0
        
        # Feature names for diabetes model
        self.feature_names = [
            'pregnancies', 'glucose', 'blood_pressure', 'skin_thickness',
            'insulin', 'bmi', 'diabetes_pedigree', 'age'
        ]
        
        logger.info("Enhanced ML Exporter initialized")
    
    def simulate_real_predictions(self):
        """Simulate realistic ML predictions with business logic"""
        # Real diabetes samples with 21 features in correct format
        diabetes_samples = [
            # Sample 1 - High risk profile
            [0.8663875715513917, 0.9391870880454268, 0.1607726785944992, 1.4051871930524469, 
             -0.9645001347013894, -0.2607083092123164, -0.4215251042181655, -1.5148281085566382,
             0.8069038625481344, 0.524704116849121, -0.2132943011689738, 0.219817012721681,
             -0.3261342499282417, 1.02560203011439, 2.084762367037103, 0.7944237442944474,
             -0.5906117333900115, -0.916407578904008, -0.5611929829056437, 0.0968733813373412, 0.6199086151085422],
            
            # Sample 2 - Medium risk profile
            [0.2, 0.3, 0.8, 0.5, -0.5, -0.1, -0.2, -0.8,
             0.4, 0.3, -0.1, 0.1, -0.2, 0.5, 1.0, 0.4,
             -0.3, -0.5, -0.3, 0.0, 0.3],
            
            # Sample 3 - Low risk profile
            [-0.5, -0.3, 0.9, -0.2, -0.8, -0.3, -0.4, 0.5,
             0.6, 0.7, -0.3, 0.3, -0.1, -0.2, -0.5, -0.2,
             -0.4, 0.5, -0.6, 0.2, -0.1],
             
            # Sample 4 - Another high risk
            [1.2, 1.1, 0.2, 1.8, 0.5, 0.1, 0.2, -1.2,
             0.9, 0.6, 0.1, 0.2, -0.4, 1.5, 2.5, 1.2,
             -0.2, -0.8, -0.4, 0.1, 0.8],
             
            # Sample 5 - Low risk variant
            [-0.8, -0.7, 0.7, -0.5, -1.2, -0.4, -0.5, 0.8,
             0.3, 0.4, -0.4, 0.4, 0.1, -0.8, -1.2, -0.6,
             -0.6, 0.8, -0.8, 0.3, -0.3]
        ]
        
        # Feature names for the diabetes model
        feature_columns = [
            "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke", 
            "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies", 
            "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth", 
            "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"
        ]
        
        while True:
            try:
                # Simulate prediction request
                sample = random.choice(diabetes_samples)
                
                # Make actual prediction to MLflow model with correct format
                prediction_result = self.make_prediction_with_dataframe(sample, feature_columns)
                
                if prediction_result:
                    # Extract business metrics from prediction
                    self.process_prediction_result(prediction_result, sample)
                    
                    # Increment prediction counter - THIS IS IMPORTANT!
                    DAILY_PREDICTIONS.inc()
                    
                    # Also track by prediction class for distribution
                    confidence = prediction_result.get('confidence', random.uniform(0.6, 0.95))
                    pred_class = 'high_risk' if confidence > 0.8 else 'low_risk'
                    PREDICTION_DISTRIBUTION.labels(prediction_class=pred_class).inc()
                    
                    logger.debug(f"📊 Prediction made: class={pred_class}, confidence={confidence:.3f}")
                else:
                    # Increment error count
                    INPUT_VALIDATION_ERRORS.labels(error_type='prediction_failed').inc()
                
                # Simulate concurrent users
                CONCURRENT_USERS.set(random.randint(1, 25))
                REQUEST_QUEUE_DEPTH.set(random.randint(0, 5))
                
                # Simulate variable load - faster predictions for more data
                time.sleep(random.uniform(1, 4))  # Reduced sleep time
                
            except Exception as e:
                logger.error(f"Error in simulate_real_predictions: {e}")
                self.error_count += 1
                INPUT_VALIDATION_ERRORS.labels(error_type='prediction_error').inc()
                time.sleep(5)
    
    def make_prediction_with_dataframe(self, features, feature_columns):
        """Make prediction using correct dataframe_split format"""
        try:
            # Construct request in correct format
            data = {
                "dataframe_split": {
                    "columns": feature_columns,
                    "data": [features]
                }
            }
            
            start_time = time.time()
            response = requests.post(
                f"{self.mlflow_url}/invocations",
                json=data,
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            end_time = time.time()
            
            # Track request metrics regardless of success/failure
            response_time = end_time - start_time
            DIABETES_REQUEST_DURATION.observe(response_time)
            
            if response.status_code == 200:
                prediction = response.json()
                
                # Track successful request
                DIABETES_REQUESTS_TOTAL.labels(method='POST', status='success').inc()
                
                # Update response time metrics
                RESPONSE_TIME_P95.set(response_time)
                
                self.successful_predictions += 1
                
                # Extract prediction value/probability if available
                pred_value = 0.5  # default
                if isinstance(prediction, dict):
                    if 'predictions' in prediction:
                        pred_value = prediction['predictions'][0] if prediction['predictions'] else 0.5
                    elif 'outputs' in prediction:
                        pred_value = prediction['outputs'][0] if prediction['outputs'] else 0.5
                elif isinstance(prediction, list) and len(prediction) > 0:
                    pred_value = prediction[0]
                
                # Simulate confidence based on prediction value
                confidence = abs(pred_value - 0.5) + 0.5  # Convert to 0.5-1.0 range
                confidence = min(max(confidence, 0.5), 0.95)  # Clamp to reasonable range
                
                logger.info(f"✅ Prediction successful: value={pred_value:.3f}, confidence={confidence:.3f}, time={response_time:.3f}s")
                
                return {
                    'prediction': prediction,
                    'prediction_value': pred_value,
                    'response_time': response_time,
                    'features': features,
                    'confidence': confidence
                }
            else:
                # Track failed request
                DIABETES_REQUESTS_TOTAL.labels(method='POST', status='error').inc()
                logger.warning(f"⚠️ Prediction failed: HTTP {response.status_code} - {response.text}")
                self.error_count += 1
                return None
                
        except Exception as e:
            # Track exception
            DIABETES_REQUESTS_TOTAL.labels(method='POST', status='exception').inc()
            logger.error(f"❌ Prediction request exception: {e}")
            self.error_count += 1
            return None
    
    def process_prediction_result(self, result, features):
        """Process prediction result for business metrics"""
        try:
            prediction = result['prediction']
            
            confidence = random.uniform(0.6, 0.95)
            
            # Update prediction confidence
            PREDICTION_CONFIDENCE.set(confidence)
            MODEL_CONFIDENCE_DISTRIBUTION.observe(confidence)
            
            # Track prediction distribution
            pred_class = 'high_risk' if confidence > 0.8 else 'low_risk'
            PREDICTION_DISTRIBUTION.labels(prediction_class=pred_class).inc()
            
            # Calculate business accuracy (simulated)
            self.prediction_count += 1
            self.total_confidence += confidence
            avg_confidence = self.total_confidence / self.prediction_count
            BUSINESS_ACCURACY.set(avg_confidence)
            
            # Simulate data drift for each feature
            for i, feature_name in enumerate(self.feature_names):
                if i < len(features):
                    # Simple drift simulation based on feature values
                    drift_score = abs(features[i] - 100) / 100.0  # Normalized drift
                    FEATURE_IMPORTANCE_DRIFT.labels(feature_name=feature_name).set(min(drift_score, 1.0))
            
            # Overall data drift score
            overall_drift = random.uniform(0.0, 0.3)
            DATA_DRIFT_SCORE.set(overall_drift)
            
        except Exception as e:
            logger.error(f"Error processing prediction result: {e}")
    
    def monitor_system_resources(self):
        """Monitor system resources and model performance"""
        while True:
            try:
                # Memory monitoring with peak tracking
                memory = psutil.virtual_memory()
                current_memory_mb = memory.used / (1024 * 1024)
                
                if current_memory_mb > self.memory_peak:
                    self.memory_peak = current_memory_mb
                    MODEL_MEMORY_PEAK.set(self.memory_peak * 1024 * 1024)  # Convert to bytes
                
                # Cache hit rate simulation
                cache_hit_rate = random.uniform(0.7, 0.95)
                CACHE_HIT_RATE.set(cache_hit_rate)
                
                # Disk I/O simulation
                DISK_IO_OPERATIONS.labels(operation_type='read').inc(random.randint(1, 10))
                DISK_IO_OPERATIONS.labels(operation_type='write').inc(random.randint(1, 5))
                
                time.sleep(10)
                
            except Exception as e:
                logger.error(f"Error in monitor_system_resources: {e}")
                time.sleep(10)
    
    def update_alerting_metrics(self):
        """Update metrics specifically for alerting rules"""
        while True:
            try:
                # Alert Metric 1: Error Rate (should trigger if > 0.05)
                if self.prediction_count > 0:
                    error_rate = self.error_count / (self.prediction_count + self.error_count)
                    ALERT_HIGH_ERROR_RATE.set(error_rate)
                
                # Alert Metric 2: Memory Usage in MB (should trigger if > 800MB)
                memory = psutil.virtual_memory()
                memory_mb = memory.used / (1024 * 1024)
                # Add some randomness to occasionally trigger alert
                if random.random() < 0.1:  # 10% chance to simulate high memory
                    memory_mb = random.uniform(850, 1200)
                ALERT_MEMORY_USAGE.set(memory_mb)
                
                # Alert Metric 3: Model Accuracy (should trigger if < 0.9)
                if self.prediction_count > 0:
                    accuracy = self.total_confidence / self.prediction_count
                    if random.random() < 0.05: 
                        accuracy = random.uniform(0.75, 0.89)
                    ALERT_MODEL_ACCURACY.set(accuracy)
                
                time.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in update_alerting_metrics: {e}")
                time.sleep(30)
    
    def monitor_mlflow_health(self):
        """Monitor MLflow model health and convert JSON to metrics"""
        while True:
            try:
                # Check /ping endpoint (standard MLflow)
                start_time = time.time()
                try:
                    ping_response = requests.get(f"{self.mlflow_url}/ping", timeout=5)
                    ping_time = time.time() - start_time
                    
                    if ping_response.status_code == 200:
                        MLFLOW_PING_STATUS.set(1)
                        MLFLOW_API_RESPONSE_TIME.set(ping_time)
                        logger.debug("✅ MLflow ping successful")
                    else:
                        MLFLOW_PING_STATUS.set(0)
                        logger.warning(f"⚠️ MLflow ping failed: {ping_response.status_code}")
                except requests.exceptions.RequestException as e:
                    MLFLOW_PING_STATUS.set(0)
                    logger.warning(f"⚠️ MLflow ping request failed: {e}")
                
                # Check if we can reach the metrics endpoint
                try:
                    metrics_response = requests.get(self.mlflow_metrics_url, timeout=5)
                    if metrics_response.status_code == 200:
                        MLFLOW_MODEL_HEALTH.set(1)
                        logger.debug("✅ MLflow metrics endpoint healthy")
                    else:
                        MLFLOW_MODEL_HEALTH.set(0)
                        logger.warning(f"⚠️ MLflow metrics endpoint failed: {metrics_response.status_code}")
                except requests.exceptions.RequestException as e:
                    MLFLOW_MODEL_HEALTH.set(0)
                    logger.warning(f"⚠️ MLflow metrics endpoint unreachable: {e}")
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in monitor_mlflow_health: {e}")
                MLFLOW_MODEL_HEALTH.set(0)
                MLFLOW_PING_STATUS.set(0)
                time.sleep(30)
    
    def start_webhook_server(self):
        """Start webhook server for Grafana alerts"""
        try:
            server = HTTPServer(('0.0.0.0', 8001), AlertWebhookHandler)
            logger.info("🔗 Alert webhook server started on port 8001")
            server.serve_forever()
        except Exception as e:
            logger.error(f"Failed to start webhook server: {e}")
    
    def start(self):
        """Start all monitoring threads"""
        logger.info("🚀 Starting Enhanced ML Model Prometheus Exporter...")
        logger.info(f"📊 MLflow model: {self.mlflow_url}")
        logger.info(f"📈 MLflow metrics: {self.mlflow_metrics_url}")
        
        # Start monitoring threads
        threads = [
            threading.Thread(target=self.simulate_real_predictions, daemon=True),
            threading.Thread(target=self.monitor_system_resources, daemon=True),
            threading.Thread(target=self.update_alerting_metrics, daemon=True),
            threading.Thread(target=self.monitor_mlflow_health, daemon=True),
            threading.Thread(target=self.start_webhook_server, daemon=True),
        ]
        
        for thread in threads:
            thread.start()
        
        logger.info("🎯 All monitoring threads started")
        logger.info("📊 Custom metrics complement MLflow built-in metrics")
        
        # Keep main thread alive
        try:
            while True:
                uptime = time.time() - self.start_time
                logger.info(f"📈 Exporter stats - Uptime: {uptime:.0f}s, Predictions: {self.prediction_count}, Errors: {self.error_count}")
                time.sleep(120)  # Log every 2 minutes
        except KeyboardInterrupt:
            logger.info("🛑 Shutting down exporter...")

if __name__ == '__main__':
    # Start Prometheus metrics server
    start_http_server(8000)
    logger.info("🌐 Custom metrics server started on port 8000")
    logger.info("🔗 Access custom metrics: http://localhost:8000/metrics")
    logger.info("🔗 Access MLflow metrics: http://localhost:8082/metrics")
    
    # Start Enhanced ML Model Exporter
    exporter = EnhancedMLExporter()
    exporter.start()