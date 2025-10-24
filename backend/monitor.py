"""
Monitoring and testing script for the Crypto Trading AI
"""
import requests
import json
import time
from typing import Dict, Any


class CryptoTradingMonitor:
    """Monitor the crypto trading API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def test_health(self) -> Dict[str, Any]:
        """Test the health endpoint"""
        try:
            response = self.session.get(f"{self.base_url}/")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e), "status": "offline"}
    
    def test_config(self) -> Dict[str, Any]:
        """Test the configuration endpoint"""
        try:
            response = self.session.get(f"{self.base_url}/config")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def test_api_endpoints(self) -> Dict[str, Any]:
        """Test various API endpoints"""
        results = {}
        
        # Test market endpoints (if credentials available)
        endpoints = [
            "/api/v1/market/price/BTCUSDT",
            "/api/v1/account/balance", 
            "/api/v1/ml/status",
            "/api/v1/trading/status"
        ]
        
        for endpoint in endpoints:
            try:
                response = self.session.get(f"{self.base_url}{endpoint}")
                if response.status_code == 200:
                    results[endpoint] = {"status": "success", "data": response.json()}
                elif response.status_code == 503:
                    results[endpoint] = {"status": "service_unavailable", "message": "Service not initialized"}
                else:
                    results[endpoint] = {"status": "error", "code": response.status_code}
            except Exception as e:
                results[endpoint] = {"status": "error", "message": str(e)}
        
        return results
    
    def run_full_test(self) -> Dict[str, Any]:
        """Run complete test suite"""
        print("🔍 Running Crypto Trading AI Monitoring Test")
        print("=" * 50)
        
        # Test health
        print("📊 Testing health endpoint...")
        health = self.test_health()
        print(f"Status: {health.get('status', 'unknown')}")
        if health.get('services'):
            for service, status in health['services'].items():
                print(f"  {service}: {'✓' if status else '✗'}")
        
        # Test configuration
        print("\n⚙️ Testing configuration...")
        config = self.test_config()
        if 'error' not in config:
            print(f"Environment: {config.get('environment')}")
            print(f"Debug mode: {config.get('debug')}")
            print(f"Testnet: {config.get('testnet_mode')}")
            print(f"API credentials: {'✓' if config.get('api_credentials_configured') else '✗'}")
        
        # Test API endpoints
        print("\n🌐 Testing API endpoints...")
        api_results = self.test_api_endpoints()
        for endpoint, result in api_results.items():
            status_icon = "✓" if result['status'] == 'success' else "⚠️" if result['status'] == 'service_unavailable' else "✗"
            print(f"  {endpoint}: {status_icon} {result['status']}")
        
        return {
            "health": health,
            "config": config,
            "api_tests": api_results,
            "timestamp": time.time()
        }


def main():
    """Main monitoring function"""
    monitor = CryptoTradingMonitor()
    
    try:
        # Wait a moment for server to be ready
        print("Waiting for server to be ready...")
        time.sleep(2)
        
        # Run tests
        results = monitor.run_full_test()
        
        print("\n📋 Test Summary:")
        health_status = results['health'].get('status')
        if health_status == 'online':
            print("✅ Server is running successfully")
        else:
            print("❌ Server has issues")
        
        # Check if API credentials are configured
        config_ok = results.get('config', {}).get('api_credentials_configured', False)
        if config_ok:
            print("✅ API credentials are configured")
        else:
            print("⚠️ API credentials not configured - limited functionality")
        
        print(f"\n🔗 API Documentation: {monitor.base_url}/docs")
        print(f"🔗 Interactive API: {monitor.base_url}/redoc")
        
        # Save detailed results
        with open('monitoring_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print("\n💾 Detailed results saved to: monitoring_results.json")
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Monitoring stopped by user")
    except Exception as e:
        print(f"\n❌ Monitoring failed: {e}")


if __name__ == "__main__":
    main()