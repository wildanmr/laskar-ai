#!/bin/bash

echo "🚀 Starting ML Model Monitoring Stack"
echo "====================================="

# Create necessary directories
mkdir -p grafana/provisioning/datasources
mkdir -p grafana/dashboards
mkdir -p prometheus-data
mkdir -p grafana-data

# Set permissions
chmod 777 prometheus-data grafana-data

echo "📁 Directories created"

# Start the stack
echo "🐳 Starting Docker Compose stack..."
docker-compose up -d

echo "⏳ Waiting for services to start..."
sleep 30

echo "🔍 Checking service status..."
docker-compose ps

echo ""
echo "🎉 Monitoring Stack is ready!"
echo "====================================="
echo "📊 Grafana: http://localhost:3000"
echo "   Username: admin"
echo "   Password: admin123"
echo ""
echo "📈 Prometheus: http://localhost:9090"
echo "🤖 ML Model API: http://localhost:8080"
echo "📊 MLflow Built-in Metrics: http://localhost:8082/metrics"
echo "📊 Custom Metrics Exporter: http://localhost:8000/metrics"
echo "💻 Node Exporter: http://localhost:9100/metrics"
echo "🐳 cAdvisor: http://localhost:8081"
echo ""
echo "🔍 Available Metrics Sources:"
echo "   • MLflow built-in: Real model performance metrics"
echo "   • Custom exporter: Business logic & alerting metrics"
echo "   • Node exporter: System resource metrics"
echo "   • cAdvisor: Container resource metrics"
echo ""
echo "🧪 To test inference:"
echo "   docker-compose exec prometheus-exporter python inference.py"
echo ""
echo "📝 To view logs:"
echo "   docker-compose logs -f [service_name]"
echo ""
echo "⛔ To stop all services:"
echo "   docker-compose down"

# Check if user wants to run test
read -p "🤔 Do you want to run inference test now? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    test_inference
fi

echo "✅ Setup complete! Happy monitoring! 🎯"