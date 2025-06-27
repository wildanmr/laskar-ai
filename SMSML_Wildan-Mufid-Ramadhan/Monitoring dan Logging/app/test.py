import requests
import json
import time
import random
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# API endpoint (MLflow proxy)
API_URL = "http://localhost:5002/predict"
HEALTH_URL = "http://localhost:5002/health"
THRESHOLD_URL = "http://localhost:5002/predict_with_threshold"

class MLflowProxyLoadTester:
    def __init__(self):
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.response_times = []
        self.start_time = None
        
        # MLflow-specific tracking
        self.threshold_requests = 0
        self.mlflow_errors = 0
        self.proxy_errors = 0
        
    def generate_loan_case(self):
        """Generate realistic loan application case"""
        # Create realistic loan application data
        
        # Random continuous features (normalized/standardized)
        umur = random.uniform(-2, 2)  # Age (standardized)
        pendapatan = random.uniform(-2, 2)  # Income (standardized)
        skor_kredit = random.uniform(-2, 2)  # Credit score (standardized)
        jumlah_pinjaman = random.uniform(-1, 2)  # Loan amount
        rasio_pinjaman_pendapatan = random.uniform(-2, 3)  # Debt ratio
        
        # Random job type (one-hot encoded)
        job_types = ['Tetap', 'Kontrak', 'Freelance']
        job = random.choice(job_types)
        
        # Random age category
        age_categories = ['Muda', 'Dewasa', 'Senior']
        age_cat = random.choice(age_categories)
        
        # Random credit category
        credit_categories = ['Poor', 'Fair', 'Good']
        credit_cat = random.choice(credit_categories)
        
        # Random income category
        income_categories = ['Rendah', 'Sedang', 'Tinggi']
        income_cat = random.choice(income_categories)
        
        # Convert to expected format
        case_data = {
            "umur": umur,
            "pendapatan": pendapatan,
            "skor_kredit": skor_kredit,
            "jumlah_pinjaman": jumlah_pinjaman,
            "rasio_pinjaman_pendapatan": rasio_pinjaman_pendapatan,
            
            # Job type (one-hot)
            "pekerjaan_Freelance": job == 'Freelance',
            "pekerjaan_Kontrak": job == 'Kontrak',
            "pekerjaan_Tetap": job == 'Tetap',
            
            # Age category (one-hot)
            "kategori_umur_Dewasa": age_cat == 'Dewasa',
            "kategori_umur_Muda": age_cat == 'Muda',
            "kategori_umur_Senior": age_cat == 'Senior',
            
            # Credit category (one-hot)
            "kategori_skor_kredit_Fair": credit_cat == 'Fair',
            "kategori_skor_kredit_Good": credit_cat == 'Good',
            "kategori_skor_kredit_Poor": credit_cat == 'Poor',
            
            # Income category (one-hot)
            "kategori_pendapatan_Rendah": income_cat == 'Rendah',
            "kategori_pendapatan_Sedang": income_cat == 'Sedang',
            "kategori_pendapatan_Tinggi": income_cat == 'Tinggi'
        }
        
        return case_data
    
    def make_standard_prediction(self):
        """Make standard prediction request"""
        try:
            case_data = self.generate_loan_case()
            start_time = time.time()
            
            response = requests.post(
                API_URL,
                headers={"Content-Type": "application/json"},
                json=case_data,
                timeout=30
            )
            
            response_time = time.time() - start_time
            self.response_times.append(response_time)
            self.total_requests += 1
            
            if response.status_code == 200:
                self.successful_requests += 1
                result = response.json()
                return True, result, response_time
            else:
                self.failed_requests += 1
                error_msg = f"HTTP {response.status_code}"
                if response.status_code == 503:
                    self.mlflow_errors += 1
                    error_msg += " (MLflow server error)"
                else:
                    self.proxy_errors += 1
                    error_msg += " (Proxy error)"
                return False, error_msg, response_time
                
        except Exception as e:
            self.failed_requests += 1
            self.total_requests += 1
            self.proxy_errors += 1
            return False, str(e), 0
    
    def make_threshold_prediction(self, threshold=None):
        """Make prediction with custom threshold"""
        try:
            case_data = self.generate_loan_case()
            
            # Add threshold if specified
            if threshold is not None:
                case_data['threshold'] = threshold
                self.threshold_requests += 1
            
            start_time = time.time()
            
            response = requests.post(
                THRESHOLD_URL,
                headers={"Content-Type": "application/json"},
                json=case_data,
                timeout=30
            )
            
            response_time = time.time() - start_time
            self.response_times.append(response_time)
            self.total_requests += 1
            
            if response.status_code == 200:
                self.successful_requests += 1
                result = response.json()
                return True, result, response_time
            else:
                self.failed_requests += 1
                return False, f"HTTP {response.status_code}", response_time
                
        except Exception as e:
            self.failed_requests += 1
            self.total_requests += 1
            return False, str(e), 0
    
    def make_mixed_request(self):
        """Make either standard or threshold prediction (mixed load)"""
        # 70% standard, 30% with custom threshold
        if random.random() < 0.7:
            return self.make_standard_prediction()
        else:
            threshold = random.choice([0.3, 0.4, 0.6, 0.7, 0.8])
            return self.make_threshold_prediction(threshold)
    
    def check_health(self):
        """Check if proxy and MLflow are healthy"""
        try:
            response = requests.get(HEALTH_URL, timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                mlflow_status = health_data.get('mlflow_server', 'unknown')
                print(f"✓ Proxy Health: {health_data.get('status', 'unknown')}")
                print(f"✓ MLflow Server: {mlflow_status}")
                print(f"✓ Threshold: {health_data.get('threshold', 'unknown')}")
                return health_data.get('status') == 'healthy'
            else:
                print(f"✗ Proxy Health check failed: HTTP {response.status_code}")
                return False
        except Exception as e:
            print(f"✗ Health check failed: {e}")
            return False
    
    def worker_thread(self, requests_per_second, duration, request_type="mixed"):
        """Worker thread untuk generate load"""
        end_time = time.time() + duration
        delay = 1.0 / requests_per_second if requests_per_second > 0 else 0
        
        while time.time() < end_time:
            if request_type == "standard":
                success, result, response_time = self.make_standard_prediction()
            elif request_type == "threshold":
                threshold = random.choice([0.3, 0.4, 0.6, 0.7, 0.8])
                success, result, response_time = self.make_threshold_prediction(threshold)
            else:  # mixed
                success, result, response_time = self.make_mixed_request()
            
            if success:
                prediction = result.get('predictions', [None])[0]
                threshold_used = result.get('threshold_used', 'default')
                threshold_applied = result.get('threshold_applied', False)
                
                status_icon = "🎯" if threshold_applied else "📊"
                print(f"{status_icon} Pred: {prediction} | Threshold: {threshold_used} | Time: {response_time:.3f}s")
            else:
                error_icon = "🔥" if "MLflow" in str(result) else "⚠️"
                print(f"{error_icon} Error: {result} | Time: {response_time:.3f}s")
            
            if delay > 0:
                time.sleep(delay)
    
    def run_load_test(self, total_requests=100, concurrent_users=5, requests_per_second=10, duration=60, request_type="mixed"):
        """Run load test"""
        print("=" * 60)
        print("MLFLOW PROXY - LOAD TESTING")
        print("=" * 60)
        print(f"Target: {API_URL}")
        print(f"Request type: {request_type}")
        print(f"Concurrent users: {concurrent_users}")
        print(f"Requests per second: {requests_per_second}")
        print(f"Duration: {duration} seconds")
        print("=" * 60)
        
        # Health check
        if not self.check_health():
            print("System is not healthy. Continuing with limited functionality...")
        
        print(f"\nStarting {request_type} load test...")
        self.start_time = time.time()
        
        # Start worker threads
        with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = []
            rps_per_worker = requests_per_second / concurrent_users
            
            for i in range(concurrent_users):
                future = executor.submit(self.worker_thread, rps_per_worker, duration, request_type)
                futures.append(future)
            
            # Wait for all workers to complete
            for future in futures:
                future.result()
        
        self.show_results()
    
    def run_threshold_comparison_test(self, thresholds=[0.3, 0.5, 0.7], requests_per_threshold=20):
        """Test different thresholds with same data"""
        print("=" * 60)
        print("MLFLOW PROXY - THRESHOLD COMPARISON TEST")
        print("=" * 60)
        print(f"Testing thresholds: {thresholds}")
        print(f"Requests per threshold: {requests_per_threshold}")
        print("=" * 60)
        
        if not self.check_health():
            print("System is not healthy. Exiting...")
            return
        
        self.start_time = time.time()
        results_by_threshold = {}
        
        for threshold in thresholds:
            print(f"\n🎯 Testing threshold: {threshold}")
            print("-" * 30)
            
            threshold_results = []
            
            for i in range(requests_per_threshold):
                success, result, response_time = self.make_threshold_prediction(threshold)
                
                if success:
                    prediction = result.get('predictions', [None])[0]
                    prob_class_1 = None
                    
                    if 'probability_class_1' in result:
                        prob_class_1 = result['probability_class_1'][0]
                    elif 'probabilities' in result and len(result['probabilities']) > 0:
                        probs = result['probabilities'][0]
                        if isinstance(probs, list) and len(probs) == 2:
                            prob_class_1 = probs[1]
                    
                    threshold_results.append({
                        'prediction': prediction,
                        'probability_class_1': prob_class_1,
                        'response_time': response_time
                    })
                    
                    print(f"  {i+1:2d}/{requests_per_threshold} ✓ Pred: {prediction} | Prob: {prob_class_1:.3f if prob_class_1 else 'N/A'} | {response_time:.3f}s")
                else:
                    print(f"  {i+1:2d}/{requests_per_threshold} ✗ {result}")
            
            # Analyze results for this threshold
            if threshold_results:
                predictions = [r['prediction'] for r in threshold_results]
                approval_rate = sum(predictions) / len(predictions)
                avg_response_time = sum(r['response_time'] for r in threshold_results) / len(threshold_results)
                
                results_by_threshold[threshold] = {
                    'approval_rate': approval_rate,
                    'avg_response_time': avg_response_time,
                    'total_requests': len(threshold_results),
                    'approved': sum(predictions),
                    'rejected': len(predictions) - sum(predictions)
                }
                
                print(f"    📊 Approval rate: {approval_rate:.1%}")
                print(f"    ⏱️  Avg response time: {avg_response_time:.3f}s")
        
        # Show comparison
        print("\n" + "=" * 60)
        print("THRESHOLD COMPARISON RESULTS")
        print("=" * 60)
        
        print(f"{'Threshold':<10} {'Approval Rate':<15} {'Approved':<10} {'Rejected':<10} {'Avg Time':<10}")
        print("-" * 55)
        
        for threshold, stats in results_by_threshold.items():
            print(f"{threshold:<10} {stats['approval_rate']:<15.1%} {stats['approved']:<10} {stats['rejected']:<10} {stats['avg_response_time']:<10.3f}s")
        
        print("=" * 60)
    
    def run_burst_test(self, burst_size=50, burst_interval=10, num_bursts=5):
        """Run burst test to simulate traffic spikes"""
        print("=" * 60)
        print("MLFLOW PROXY - BURST TESTING")
        print("=" * 60)
        print(f"Burst size: {burst_size} requests")
        print(f"Burst interval: {burst_interval} seconds")
        print(f"Number of bursts: {num_bursts}")
        print("=" * 60)
        
        if not self.check_health():
            print("System is not healthy. Exiting...")
            return
        
        self.start_time = time.time()
        
        for burst_num in range(num_bursts):
            print(f"\n💥 Burst {burst_num + 1}/{num_bursts}")
            print("-" * 30)
            
            # Execute burst with mixed request types
            with ThreadPoolExecutor(max_workers=min(burst_size, 20)) as executor:
                futures = [executor.submit(self.make_mixed_request) for _ in range(burst_size)]
                
                for i, future in enumerate(futures):
                    success, result, response_time = future.result()
                    if success:
                        prediction = result.get('predictions', [None])[0]
                        threshold_used = result.get('threshold_used', 'default')
                        threshold_applied = result.get('threshold_applied', False)
                        icon = "🎯" if threshold_applied else "📊"
                        print(f"  {i+1:2d}/{burst_size} {icon} {prediction} | T:{threshold_used} | {response_time:.3f}s")
                    else:
                        print(f"  {i+1:2d}/{burst_size} ✗ {result}")
            
            if burst_num < num_bursts - 1:
                print(f"⏳ Waiting {burst_interval} seconds...")
                time.sleep(burst_interval)
        
        self.show_results()
    
    def show_results(self):
        """Show test results"""
        total_time = time.time() - self.start_time
        
        print("\n" + "=" * 60)
        print("LOAD TEST RESULTS")
        print("=" * 60)
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Total requests: {self.total_requests}")
        print(f"Successful: {self.successful_requests}")
        print(f"Failed: {self.failed_requests}")
        print(f"Success rate: {(self.successful_requests/self.total_requests)*100:.1f}%")
        print(f"Requests per second: {self.total_requests/total_time:.2f}")
        
        # MLflow-specific stats
        print(f"\nMLflow Proxy Stats:")
        print(f"Threshold requests: {self.threshold_requests}")
        print(f"MLflow errors: {self.mlflow_errors}")
        print(f"Proxy errors: {self.proxy_errors}")
        
        if self.response_times:
            response_times = np.array(self.response_times)
            print(f"\nResponse Times:")
            print(f"  Mean: {np.mean(response_times):.3f}s")
            print(f"  Median: {np.median(response_times):.3f}s")
            print(f"  95th percentile: {np.percentile(response_times, 95):.3f}s")
            print(f"  99th percentile: {np.percentile(response_times, 99):.3f}s")
            print(f"  Min: {np.min(response_times):.3f}s")
            print(f"  Max: {np.max(response_times):.3f}s")
        
        print(f"\nMonitoring URLs:")
        print(f"  Proxy metrics: http://localhost:5002/metrics")
        print(f"  Proxy health: {HEALTH_URL}")
        print(f"  Proxy debug: http://localhost:5002/debug")
        print("=" * 60)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Load test MLflow proxy server')
    parser.add_argument('--mode', choices=['load', 'burst', 'threshold', 'continuous'], default='load',
                       help='Test mode')
    parser.add_argument('--requests', type=int, default=100,
                       help='Total requests for load test')
    parser.add_argument('--users', type=int, default=5,
                       help='Concurrent users')
    parser.add_argument('--rps', type=int, default=10,
                       help='Requests per second')
    parser.add_argument('--duration', type=int, default=60,
                       help='Test duration in seconds')
    parser.add_argument('--request-type', choices=['standard', 'threshold', 'mixed'], default='mixed',
                       help='Type of requests to make')
    parser.add_argument('--burst-size', type=int, default=50,
                       help='Requests per burst')
    parser.add_argument('--burst-interval', type=int, default=10,
                       help='Interval between bursts')
    parser.add_argument('--num-bursts', type=int, default=5,
                       help='Number of bursts')
    parser.add_argument('--thresholds', nargs='+', type=float, default=[0.3, 0.5, 0.7],
                       help='Thresholds to test in threshold mode')
    parser.add_argument('--requests-per-threshold', type=int, default=20,
                       help='Requests per threshold in threshold mode')
    
    args = parser.parse_args()
    
    tester = MLflowProxyLoadTester()
    
    if args.mode == 'load':
        tester.run_load_test(
            total_requests=args.requests,
            concurrent_users=args.users,
            requests_per_second=args.rps,
            duration=args.duration,
            request_type=args.request_type
        )
    elif args.mode == 'burst':
        tester.run_burst_test(
            burst_size=args.burst_size,
            burst_interval=args.burst_interval,
            num_bursts=args.num_bursts
        )
    elif args.mode == 'threshold':
        tester.run_threshold_comparison_test(
            thresholds=args.thresholds,
            requests_per_threshold=args.requests_per_threshold
        )
    elif args.mode == 'continuous':
        print("Starting continuous testing (Ctrl+C to stop)...")
        try:
            while True:
                tester.run_load_test(
                    total_requests=args.requests,
                    concurrent_users=args.users,
                    requests_per_second=args.rps,
                    duration=args.duration,
                    request_type=args.request_type
                )
                print("Waiting 30 seconds before next cycle...")
                time.sleep(30)
        except KeyboardInterrupt:
            print("\nStopping continuous test...")
            tester.show_results()

if __name__ == "__main__":
    main()