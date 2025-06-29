#!/usr/bin/env python3
"""
 Model Deployment Demo Script

This script demonstrates:
1. Docker containerization
2. API endpoints testing
3. MLflow model loading
4. CI/CD pipeline validation
"""

import requests
import json
import time
import subprocess
import os


def test_api_locally():
    """Test the API endpoints locally"""
    
    print("🧪 Testing API Endpoints Locally")
    print("=" * 50)
    
    base_url = "http://localhost:8000"
    
    try:
        # Test root endpoint
        print("1. Testing root endpoint...")
        response = requests.get(f"{base_url}/")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
        
        # Test health endpoint
        print("\n2. Testing health endpoint...")
        response = requests.get(f"{base_url}/health")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
        
        # Test predict endpoint
        print("\n3. Testing predict endpoint...")
        test_customer = {
            "customer_id": "DEMO_001",
            "recency": 15.0,
            "frequency": 25,
            "monetary": 7500.0,
            "total_amount": 7500.0,
            "avg_amount": 300.0,
            "transaction_count": 25
        }
        
        response = requests.post(f"{base_url}/predict", json=test_customer)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            print(f"   Prediction: {response.json()}")
        else:
            print(f"   Error: {response.text}")
        
        # Test model info endpoint
        print("\n4. Testing model info endpoint...")
        response = requests.get(f"{base_url}/model/info")
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            print(f"   Model Info: {response.json()}")
        else:
            print(f"   Error: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Make sure it's running on localhost:8000")
        print("   Start with: uvicorn src.api.main:app --host 0.0.0.0 --port 8000")


def test_docker_deployment():
    """Test Docker deployment"""
    
    print("\n🐳 Testing Docker Deployment")
    print("=" * 50)
    
    try:
        # Build Docker image
        print("1. Building Docker image...")
        result = subprocess.run([
            "docker", "build", "-t", "credit-risk-api:task6", "."
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("   ✅ Docker image built successfully")
        else:
            print("   ❌ Docker build failed:")
            print(f"   {result.stderr}")
            return
        
        # Run Docker container
        print("\n2. Starting Docker container...")
        result = subprocess.run([
            "docker", "run", "-d", "--name", "test-api-task6", 
            "-p", "8001:8000", "credit-risk-api:task6"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            container_id = result.stdout.strip()
            print(f"   ✅ Container started: {container_id[:12]}")
            
            # Wait for container to start
            print("   Waiting for API to start...")
            time.sleep(15)
            
            # Test container health
            try:
                response = requests.get("http://localhost:8001/health", timeout=10)
                if response.status_code == 200:
                    print("   ✅ Container health check passed")
                    print(f"   Response: {response.json()}")
                else:
                    print(f"   ❌ Health check failed: {response.status_code}")
            except Exception as e:
                print(f"   ❌ Could not reach containerized API: {e}")
        else:
            print("   ❌ Failed to start container:")
            print(f"   {result.stderr}")
        
    except FileNotFoundError:
        print("   ❌ Docker not found. Please install Docker.")
    except Exception as e:
        print(f"   ❌ Docker test failed: {e}")
    finally:
        # Cleanup
        print("\n3. Cleaning up...")
        subprocess.run(["docker", "stop", "test-api-task6"], 
                      capture_output=True, text=True)
        subprocess.run(["docker", "rm", "test-api-task6"], 
                      capture_output=True, text=True)
        print("   ✅ Cleanup completed")


def test_linting():
    """Test code linting"""
    
    print("\n🔍 Testing Code Linting")
    print("=" * 50)
    
    try:
        # Test flake8
        print("1. Running flake8...")
        result = subprocess.run([
            "flake8", "src/", "--max-line-length=88", "--count"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("   ✅ Flake8 passed")
        else:
            print("   ❌ Flake8 issues found:")
            print(f"   {result.stdout}")
        
        # Test black formatting
        print("\n2. Checking black formatting...")
        result = subprocess.run([
            "black", "--check", "--diff", "src/"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("   ✅ Black formatting check passed")
        else:
            print("   ❌ Black formatting issues found:")
            print(f"   {result.stdout}")
            
    except FileNotFoundError as e:
        print(f"   ❌ Linting tool not found: {e}")


def run_tests():
    """Run unit tests"""
    
    print("\n🧪 Running Unit Tests")
    print("=" * 50)
    
    try:
        result = subprocess.run([
            "python", "-m", "pytest", "tests/", "-v"
        ], capture_output=True, text=True)
        
        print(f"Return code: {result.returncode}")
        print(f"Output: {result.stdout}")
        
        if result.stderr:
            print(f"Errors: {result.stderr}")
        
        if result.returncode == 0:
            print("   ✅ All unit tests passed")
        else:
            print("   ❌ Some tests failed")
            
    except FileNotFoundError:
        print("   ❌ pytest not found")


def check_requirements():
    """Check if all requirements are satisfied"""
    
    print("📋 Task 6 Requirements Check")
    print("=" * 50)
    
    checks = [
        ("FastAPI in requirements.txt", check_requirement_file("fastapi")),
        ("Uvicorn in requirements.txt", check_requirement_file("uvicorn")),
        ("Flake8 in requirements.txt", check_requirement_file("flake8")),
        ("API main.py exists", os.path.exists("src/api/main.py")),
        ("Pydantic models exist", os.path.exists("src/api/pydantic_models.py")),
        ("Dockerfile exists", os.path.exists("Dockerfile")),
        ("docker-compose.yml exists", os.path.exists("docker-compose.yml")),
        ("CI workflow exists", os.path.exists(".github/workflows/ci.yml")),
    ]
    
    for description, status in checks:
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {description}")
    
    all_passed = all(status for _, status in checks)
    print(f"\nOverall: {'✅ All requirements met' if all_passed else '❌ Some requirements missing'}")


def check_requirement_file(package):
    """Check if package is in requirements.txt"""
    try:
        with open("requirements.txt", "r") as f:
            content = f.read().lower()
            return package.lower() in content
    except FileNotFoundError:
        return False


def main():
    """Main demonstration function"""
    
    print("🚀 TASK 6 - MODEL DEPLOYMENT DEMO")
    print("=" * 70)
    
    # Check requirements
    check_requirements()
    
    # Test linting
    test_linting()
    
    # Run tests
    run_tests()
    
    # Instructions for manual testing
    print("\n📖 Manual Testing Instructions")
    print("=" * 50)
    print("1. Start API locally:")
    print("   uvicorn src.api.main:app --host 0.0.0.0 --port 8000")
    print("\n2. Test endpoints:")
    print("   python task6_demo.py --test-api")
    print("\n3. Start with Docker:")
    print("   docker-compose up --build")
    print("\n4. Test CI/CD:")
    print("   git push origin main  # Triggers GitHub Actions")
    
    print("\n✅ Task 6 Demo Complete!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test-api":
        test_api_locally()
    elif len(sys.argv) > 1 and sys.argv[1] == "--test-docker":
        test_docker_deployment()
    else:
        main() 