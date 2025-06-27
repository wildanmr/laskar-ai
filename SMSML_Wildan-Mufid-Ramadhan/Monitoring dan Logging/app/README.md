# HOW TO

## Buat Venv dan install Libray
- `python -m venv venv`
- `source venv/bin/activate`
- `pip install -r requirement.txt`

## Set MLFlow Tracking URI
- Buka terminal untuk linux / macos dan setup mlflow tracking URI

	`export MLFLOW_TRACKING_URI=https://dagshub.com/agusprasetyo811/kredit_pinjaman_2.mlflow`

## Serving Model
- Buka terminal untuk linux / macos dan jalankan mlflow model serve

	`mlflow models serve -m "mlflow-artifacts:/18887086bb5943aaa2ec1b7e979ea37b/60d6c8524e614acf959dfa5bf957704a/artifacts/production_model" -p 5001 --env-manager local`

- Model API akan berjalan di `http://127.0.0.1:5001`	

## Jalankan testing by inference.py
- Jalankan `python inference.py` untuk testing serving model api. pastikan MLFLOW_URL mengarah
ke `http://127.0.0.1:5001`
- Ikuti step by step dari inference.py untuk melihat sistem berjalan

## Jalankan prometheus_exporter untuk persiapan Monitoring
- Jalankan `prometheus_exporter.py --port 5002`
- Setelah berjalan Model API yang terkoneksi dengan server mlflow model `http://127.0.0.1:5001` akan mengelurakan metrics yang siap tarik untuk monitoring. 
- Buka browser dan lihat enpoint yang bisa dicek
	- "GET  /health - Health check",
	- "GET  /info - Model information",
	- "GET  /debug - Debug information",
	- "GET  /features - Feature information and examples",
	- "GET  /test - Test prediction with sample data",
	- "GET  /threshold - Get current threshold",
	- "POST /threshold - Set threshold",
	- "POST /predict - Make predictions with current threshold",
	- "POST /predict_with_threshold - Predict with custom threshold",
	- "POST /threshold_analysis - Analyze multiple thresholds",
	- "POST /reload - Reload model from file",
	- "GET /metrics - Metrics for Prometheus"

## Jalankan Monitoring Prometheus dan Grafana
- Pastikan docker sudah terinstall di PC / Leptop
- Jalankan `docker-compose.yml` ini akan menjalankan aplikasi monitoring prometheus dan grafana
- Prometheus akan berjalan di port `9090` -> `http://localhost:9090`
- Grafana akan berjalan di port `4000` -> `http://localhost:4000`
- Login Grafana dengan user `admin`  & password `admin`. kemudan bisa pilih opsi skip untuk set password baru
- Dashboard akan langsung tergenerate dengan konfigurasi yang sudah diset di `grafana/provisioning` directory

## Jalankan Testing Apps
- Jalankan `python test.py` untuk melakukan testing
- Pastikan konfigurasi mengarah ke prometheus exporter:
	- API_URL = "http://localhost:5002/predict"
	- HEALTH_URL = "http://localhost:5002/health"
	- THRESHOLD_URL = "http://localhost:5002/predict_with_threshold"
	



