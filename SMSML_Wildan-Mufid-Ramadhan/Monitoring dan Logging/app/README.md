# 🚀 Monitoring Setup

Contains a complete stack of services including ML inference, monitoring, and dashboarding. All services can be started with a single command.

## 🧠 Included Services

| Service             | Description                         | Port       |
| ------------------- | ----------------------------------- | ---------- |
| ML Model            | MLflow-based model server           | 8080, 8082 |
| Streamlit Inference | Streamlit app for inference UI      | 8501       |
| Prometheus Exporter | Custom Python metrics exporter      | 8000       |
| Prometheus          | Monitoring and alerting toolkit     | 9090       |
| Grafana             | Analytics & visualization dashboard | 3000       |
| cAdvisor            | Container resource usage monitoring | 8081       |
| Node Exporter       | Host metrics exporter               | 9100       |

## 🏁 Quick Start

Make sure you have Docker and Docker Compose installed.

### 1. Start all services

```bash
docker compose up -d
```

### 2. Access Services

* **ML Model API:** [http://localhost:8080](http://localhost:8080)
* **ML Model Metrics:** [http://localhost:8082](http://localhost:8082)
* **Streamlit Serving App:** [http://localhost:8501](http://localhost:8501)
* **Prometheus Exporter:** [http://localhost:8000](http://localhost:8000)
* **Prometheus:** [http://localhost:9090](http://localhost:9090)
* **Grafana:** [http://localhost:3000](http://localhost:3000) (Login with admin / admin123)
* **cAdvisor:** [http://localhost:8081](http://localhost:8081)
* **Node Exporter:** [http://localhost:9100](http://localhost:9100)

## 🔐 Grafana Credentials

* **Username:** `admin`
* **Password:** `admin123`

## ✅ Health Status

All containers are configured with health checks. Use the following command to monitor them:

```bash
docker compose ps
```

## 🛑 Stop All Services

```bash
docker compose down
```

---