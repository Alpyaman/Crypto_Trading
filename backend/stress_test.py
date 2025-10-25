#!/usr/bin/env python3
"""
Stress Test for Timeout Fixes
Tests the system under load to ensure robustness
"""
import requests
import time
from concurrent.futures import ThreadPoolExecutor
import statistics

BASE_URL = "http://localhost:8000/api/v1"

def test_single_request(endpoint):
    """Test a single request and return response time"""
    start_time = time.time()
    try:
        response = requests.get(f"{BASE_URL}{endpoint}", timeout=35)
        end_time = time.time()
        return {
            'success': response.status_code == 200,
            'status_code': response.status_code,
            'response_time': end_time - start_time,
            'error': None
        }
    except Exception as e:
        end_time = time.time()
        return {
            'success': False,
            'status_code': None,
            'response_time': end_time - start_time,
            'error': str(e)
        }

def stress_test_endpoint(endpoint, num_requests=20, max_workers=5):
    """Stress test an endpoint with concurrent requests"""
    print(f"\n🔥 Stress Testing: {endpoint}")
    print(f"Requests: {num_requests}, Concurrent Workers: {max_workers}")
    print("-" * 50)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(test_single_request, endpoint) for _ in range(num_requests)]
        results = [future.result() for future in futures]
    
    # Analyze results
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    response_times = [r['response_time'] for r in successful]
    
    print(f"✅ Successful: {len(successful)}/{num_requests} ({len(successful)/num_requests*100:.1f}%)")
    print(f"❌ Failed: {len(failed)}")
    
    if response_times:
        print("⏱️  Response Times:")
        print(f"   Average: {statistics.mean(response_times):.2f}s")
        print(f"   Median:  {statistics.median(response_times):.2f}s")
        print(f"   Min:     {min(response_times):.2f}s")
        print(f"   Max:     {max(response_times):.2f}s")
    
    if failed:
        print("\n⚠️  Error Details:")
        for i, failure in enumerate(failed[:3]):  # Show first 3 errors
            print(f"   {i+1}. Status: {failure['status_code']}, Error: {failure['error']}")
    
    return len(successful), len(failed), response_times

def main():
    """Run comprehensive stress tests"""
    print("🚀 Crypto Trading API - Stress Test Suite")
    print("=" * 60)
    
    # Test if server is running
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        if response.status_code != 200:
            print("❌ Server not responding correctly")
            return
        print("✅ Server is running and responsive")
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        return
    
    # Define test endpoints
    endpoints_to_test = [
        "/health",
        "/health/binance",
        "/market/price/BTCUSDT",
        "/account/balance",
        "/trading/status"
    ]
    
    total_success = 0
    total_failed = 0
    all_response_times = []
    
    # Run stress tests
    for endpoint in endpoints_to_test:
        success, failed, times = stress_test_endpoint(endpoint, num_requests=15, max_workers=3)
        total_success += success
        total_failed += failed
        all_response_times.extend(times)
        time.sleep(1)  # Brief pause between tests
    
    # Overall summary
    print("\n" + "=" * 60)
    print("📊 OVERALL STRESS TEST RESULTS")
    print("=" * 60)
    print(f"Total Requests: {total_success + total_failed}")
    print(f"✅ Total Successful: {total_success} ({total_success/(total_success+total_failed)*100:.1f}%)")
    print(f"❌ Total Failed: {total_failed}")
    
    if all_response_times:
        print("\n⏱️  Overall Performance:")
        print(f"   Average Response Time: {statistics.mean(all_response_times):.2f}s")
        print(f"   95th Percentile: {sorted(all_response_times)[int(len(all_response_times)*0.95)]:.2f}s")
    
    # Determine if system passed stress test
    success_rate = total_success / (total_success + total_failed) if (total_success + total_failed) > 0 else 0
    
    if success_rate >= 0.95:  # 95% success rate threshold
        print(f"\n🎉 STRESS TEST PASSED! Success rate: {success_rate*100:.1f}%")
        print("✅ System is robust and handles concurrent load well")
    else:
        print(f"\n⚠️  STRESS TEST NEEDS ATTENTION. Success rate: {success_rate*100:.1f}%")
        print("❌ System may need further optimization for production load")

if __name__ == "__main__":
    main()